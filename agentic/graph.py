"""The LangGraph runtime.

    intake -> supervisor --(conditional)--> agent_llm <-> agent_tools --> agent_done -> supervisor
                 |
                 +--> finalize -> [propose -> approval -> commit] -> END

* The supervisor (LLM) picks agents dynamically; Python guards veto invalid picks.
* Each agent is a tool-calling loop: the LLM chooses a tool + arguments, the
  GuardedExecutor runs it, the result is fed back, and the LLM decides whether to
  call another tool, revise, or submit. Python enforces limits and stops loops.
* Nodes return partial state updates; the trace is append-only.
"""

from __future__ import annotations

import contextvars
import copy
import json
import re

from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from agentic import evidence as ev
from agentic import fields as fields_mod
from agentic import gaps as gp
from agentic import parallel
from agentic import routing
from agentic.agents.common import system_prompt
from agentic.changeset import build_change_set
from agentic.context import compact_messages, compact_schema, fit_to_budget, ledger_hint
from agentic.guards import budget as bud
from agentic.guards.budget import BudgetStop, reset_agent_counters
from agentic.guards import select_budget as sb
from agentic.guards.executor import GuardedExecutor
from agentic.llm.base import LLMError, ToolCallRequest
from agentic.llm.pacing import estimate_request_tokens
from agentic.llmcall import call_llm
from agentic.prefetch import packet_followup_fields
from agentic.proposals import ProposalService
from agentic.report import build_report, build_tracker, run_status, status_reasons
from agentic.repo import resolve_company
from agentic.security import redact, wrap_untrusted
from agentic.state import Emitter, RunState
from agentic.supervisor import decide

MAX_CALLS_PER_TURN = 4
MAX_TOOL_MESSAGE_CHARS = 9000
WEB_TOOLS = {"web_search", "fetch_page"}


def clean_company_name(raw: str) -> str | None:
    name = re.sub(r"[\x00-\x1f\x7f]", " ", raw or "")
    name = re.sub(r"\s+", " ", name).strip()
    return name if 2 <= len(name) <= 200 else None


def _tpm_limit(llm) -> int | None:
    """Per-minute token limit of a paced client (the size above which PacedLLM refuses a request), else None."""
    while llm is not None:
        quota = getattr(llm, "quota", None)
        if quota is not None:
            return quota.tpm
        llm = getattr(llm, "inner", None)
    return None


def _tool_message(call: dict, payload: dict, untrusted: bool) -> dict:
    text = json.dumps(payload, default=str)
    if len(text) > MAX_TOOL_MESSAGE_CHARS:
        text = text[:MAX_TOOL_MESSAGE_CHARS] + '..."truncated"'
    if untrusted:
        text = wrap_untrusted(text, source=call["name"])
    return {"role": "tool", "tool_call_id": call["id"], "name": call["name"], "content": redact(text)}


def build_graph(deps, specs: dict, checkpointer=None, extra_nodes=None):
    limits = deps.limits
    registry = deps.registry
    executor = GuardedExecutor(registry, deps)

    # ------------------------------------------------------------------ helpers
    def persisted(fn):
        def wrapper(state):
            out = fn(state)
            merged = {**state, **out}
            merged["trace"] = list(state.get("trace", [])) + list(out.get("trace", []))
            if out.get("trace"):
                deps.store.append_events(state["run_id"], out["trace"])
            deps.store.upsert_run(state["run_id"], status=merged.get("status"),
                                  tracker=build_tracker(merged))
            return out
        wrapper.__name__ = fn.__name__
        return wrapper

    # --------------------------------------------------------------- CRM context
    CRM_GETTERS = {"products": deps.repo.get_products, "components": deps.repo.get_components,
                   "contact_summary": deps.repo.get_contact_summary, "suppliers": deps.repo.get_company_suppliers,
                   "lead_status": deps.repo.get_lead_status, "profile": deps.repo.get_profile}
    EAGER_SECTIONS = ["products", "components", "contact_summary", "suppliers", "lead_status"]   # legacy behaviour

    def load_crm_sections(existing: dict, sections: list, errors: list) -> dict:
        """Read the named CRM sections that are not loaded yet (read-only). Returns an updated copy."""
        existing = dict(existing)
        loaded = list(existing.get("loaded_sections", []))
        for key in sections:
            if key in loaded or key not in CRM_GETTERS:
                continue
            try:
                val = CRM_GETTERS[key](existing["company_id"])
                existing[key] = val[:30] if isinstance(val, list) else val
            except Exception as exc:  # noqa: BLE001 - a CRM outage must not stop research
                errors.append({"stage": f"db_{key}", "error": type(exc).__name__})
            loaded.append(key)
        existing["loaded_sections"] = loaded
        return existing

    def lazy_sections_needed(state: RunState) -> list:
        """CRM sections the agents that may run next actually use (see routing.CRM_NEEDS)."""
        eff = (state.get("options") or {}).get("efficiency") or {}
        plan = routing.classify_objective(state["objective"]) if eff.get("deterministic_routing") else None
        if (state.get("options") or {}).get("fields"):        # field request: the agents the plan needs + old values
            agents = [a for a in sorted(fields_mod.agents_needed(state)) if state["agent_runs"].get(a, 0) < 1]
            return routing.crm_sections_for(agents) + ["profile"]
        if plan is not None and plan.kind == "clear":
            agents = [a for a in routing.prerequisite_chain(plan.target, state) + [plan.target]
                      if state["agent_runs"].get(a, 0) < 1]
        else:                                                   # LLM routing: any agent may be chosen
            agents = [a for a in specs if state["agent_runs"].get(a, 0) < 1]
        return routing.crm_sections_for(agents)

    # ------------------------------------------------------------------ intake
    def intake(state: RunState):
        try:
            out = _intake(state)
        except sb.SelectBudgetExceeded as exc:        # never read as 'company not found'
            em = Emitter(state)
            em.emit("run_failed", None, reason="db_select_budget_exceeded", limit=exc.limit, query=exc.label)
            return {"status": "failed", "stop_reason": "db_select_budget_exceeded",
                    "errors": [{"stage": "db_select_budget", "error": f"limit_{exc.limit}"}],
                    "budgets": {**state["budgets"], "db_selects": sb.used()}, "trace": em.events}
        out.setdefault("budgets", {**state["budgets"], "db_selects": sb.used()})
        return out

    def _intake(state: RunState):
        em = Emitter(state)
        name = clean_company_name(state["subject"]["name"])
        if name is None:
            em.emit("run_failed", None, reason="invalid_company_name")
            return {"status": "failed", "stop_reason": "invalid_company_name",
                    "errors": [{"stage": "intake", "error": "invalid_company_name"}], "trace": em.events}
        em.emit("run_started", None, company=name, objective=state["objective"])
        errors = []
        try:
            res = resolve_company(deps.repo, name, deps.settings.min_company_match)
        except Exception as exc:  # noqa: BLE001 - DB outage must not stop research
            errors.append({"stage": "db_lookup", "error": f"{type(exc).__name__}"})
            res = {"status": "not_found", "matches": [], "primary": None,
                   "assumptions": ["Database lookup failed; proceeding as a new company (research only)."]}
        probe = getattr(deps.repo, "can_see_companies", None)
        if res["status"] == "not_found" and probe is not None:
            try:
                visible = bool(probe())
            except Exception:  # noqa: BLE001 - an outage is already handled by the lookup path above
                visible = True
            if not visible:                      # an unreadable CRM must not look like 'a brand-new company'
                em.emit("run_failed", None, reason="crm_not_readable")
                return {"status": "failed", "stop_reason": "crm_not_readable",
                        "errors": errors + [{"stage": "db_visibility", "error": "companies_table_returned_no_rows"}],
                        "trace": em.events}
        expected = (state.get("options") or {}).get("expected_company_id")
        if expected is not None:                 # enrichment of ONE known record: never research or create another
            primary_now = res["primary"]
            if primary_now is None or primary_now["company_id"] != expected:
                try:
                    row = deps.repo.get_company(expected)
                except Exception:  # noqa: BLE001
                    row = None
                from agentic import names as nm
                if row is None or nm.similarity(name, row.get("company_name") or "") < deps.settings.min_company_match:
                    em.emit("run_failed", None, reason="expected_company_not_matched", expected_company_id=expected)
                    return {"status": "failed", "stop_reason": "expected_company_not_matched",
                            "errors": errors + [{"stage": "intake", "error": "expected_company_not_matched"}],
                            "trace": em.events}
                res = {**res, "status": "found", "primary": row}
        existing = None
        primary = res["primary"]
        if primary:
            cid = primary["company_id"]
            existing = {"company_id": cid,
                        "fields": {k: primary.get(k) for k in ("company_name", "country", "sector_name", "website",
                                                              "linkedin_url", "revenue", "number_of_plants",
                                                              "company_type", "enrichment_status")},
                        "provenance": "database_unverified"}
            existing["loaded_sections"] = []
            lazy = bool(((state.get("options") or {}).get("efficiency") or {}).get("lazy_crm"))
            # lazy: only the company row is read now; each agent's tables are read when that agent is planned
            wants_profile = ["profile"] if (state.get("options") or {}).get("fields") else []   # old values for the change set
            existing = load_crm_sections(existing, (EAGER_SECTIONS if not lazy else []) + wants_profile, errors)
        em.emit("db_lookup", None, status=res["status"], matches=len(res["matches"]),
                company_id=(existing or {}).get("company_id"))
        subject = {"name": name, "match": {"status": res["status"], "matches": res["matches"],
                                           "assumptions": res["assumptions"]},
                   "existing": existing, "identity": {"verified": False}}
        return {"subject": subject, "status": "running", "errors": errors, "trace": em.events}

    # -------------------------------------------------------------- supervisor
    def supervisor(state: RunState):
        em = Emitter(state)
        b = copy.deepcopy(state["budgets"])
        steps = state["supervisor_steps"]
        reason = bud.run_limit_reason(limits, b)
        if reason is None and steps >= limits.max_supervisor_steps:
            reason = "max_supervisor_steps"
        if reason:
            em.emit("stop", "supervisor", reason=reason)
            return {"current_agent": None, "stop_reason": reason, "budgets": b, "trace": em.events}
        extra: dict = {}
        eff = (state.get("options") or {}).get("efficiency") or {}
        existing = state["subject"].get("existing")
        if (eff.get("lazy_crm") or (state.get("options") or {}).get("fields")) and existing:
            todo = [x for x in lazy_sections_needed(state) if x not in existing.get("loaded_sections", [])]
            if todo:
                errs: list = []
                try:
                    loaded = load_crm_sections(existing, todo, errs)
                except sb.SelectBudgetExceeded as exc:
                    em.emit("stop", "supervisor", reason="db_select_budget_exceeded", limit=exc.limit, query=exc.label)
                    b["db_selects"] = sb.used()
                    return {"current_agent": None, "stop_reason": "db_select_budget_exceeded", "budgets": b,
                            "trace": em.events}
                subject = {**state["subject"], "existing": loaded}
                state = {**state, "subject": subject}
                extra = {"subject": subject, "errors": list(state["errors"]) + errs}
                em.emit("crm_sections_loaded", "supervisor", sections=todo)
        if sb.used() is not None:
            b["db_selects"] = sb.used()
        decision, elig = decide({**state, "budgets": b}, deps, specs, em)
        history = list(state["route_history"]) + [{
            "step": steps + 1, "next": decision["next"], "reason": decision["reason"],
            "vetoes": decision["vetoes"], "eligible": [k for k, v in elig.items() if v["eligible"]],
            "deterministic": bool(decision.get("deterministic"))}]
        em.emit("route", "supervisor", next=decision["next"], reason=decision["reason"],
                deterministic=bool(decision.get("deterministic")), vetoes=decision["vetoes"], eligible=[k for k, v in elig.items() if v["eligible"]])
        update = {**extra, "budgets": b, "supervisor_steps": steps + 1, "route_history": history,
                  "trace": em.events}
        if decision["next"] == "finish":
            update.update({"current_agent": None,
                           "stop_reason": decision.get("stop") or decision["error"] or "supervisor_finished"})
            return update
        if decision["next"] == "parallel":
            update.update({"current_agent": None, "parallel_agents": list(decision["agents"])})
            return update
        spec = specs[decision["next"]]
        task = agent_task(state, spec, b)
        reset_agent_counters(b)
        runs = dict(state["agent_runs"])
        runs[spec.name] = runs.get(spec.name, 0) + 1
        update.update({"current_agent": spec.name, "agent_objective": spec.goal, "agent_runs": runs,
                       "agent_messages": [{"role": "user", "content": task}],
                       "agent_notes": {"no_tool_turns": 0, "accepted": 0, "rejected": 0,
                                       "ledger_size_start": len(state["ledger"])}})
        return update

    def agent_task(state, spec, b) -> str:
        """Task text for one agent: its own task, the budget, the requested-field focus and reusable evidence."""
        eff = (state.get("options") or {}).get("efficiency") or {}
        task = spec.task(state) + "\n\n" + _budget_line(spec, b)
        focus = fields_mod.focus_line(spec.name, state)
        if focus:
            task += "\n\n" + focus
        shared = eff.get("ledger_hint") or (state.get("options") or {}).get("fields")      # reuse evidence already retrieved
        if shared and spec.name != "lead" and (hint := ledger_hint(state["ledger"])):
            task += "\n\n" + hint
        return task

    def _budget_line(spec, b):
        return (f"Budget: at most {limits.max_tool_calls_agent} tool calls, "
                f"{bud.credits_remaining_run(limits, b)} search credits left "
                f"(basic search = 1 credit, advanced = 2). Submit with `{spec.finish_tool}` when done.")

    def followup_todo(name: str, claims: dict, state) -> list:
        """What the single allowed follow-up search/fetch would be for. Empty = the packet settled it."""
        opts = state.get("options") or {}
        if name == "company":
            if opts.get("fields"):
                focus = fields_mod.focus_company_fields({**state, "claims": claims})
                return fields_mod.unresolved_requested("company", claims, focus)
            return packet_followup_fields(claims)
        if opts.get("fields"):
            return fields_mod.unresolved_requested(name, claims, opts["fields"])
        items = claims.get(name)
        return [] if items else ["any verified finding"]

    # --------------------------------------------------------------- agent LLM
    def agent_llm(state: RunState):
        em = Emitter(state)
        b = copy.deepcopy(state["budgets"])
        notes = dict(state["agent_notes"])
        name = state["current_agent"]
        spec = specs[name]
        eff = (state.get("options") or {}).get("efficiency") or {}
        if spec.deterministic and eff.get("lead_deterministic") and not routing.wants_interpretation(state["objective"]):
            # Routine case: the score is already deterministic, so are the standard next actions. No model call.
            findings = spec.deterministic(state)
            claims = {k: (list(v) if isinstance(v, list) else v) for k, v in state["claims"].items()}
            for k, v in spec.process(state, findings, deps).extra.get("claims_override", {}).items():
                claims[k] = v
            notes.update(end_reason="submitted", accepted=0, rejected=0,
                         findings={k: findings.get(k) for k in ("status", "suggest_next", "unresolved", "notes")})
            em.emit("deterministic_agent", name, reason="routine result computed without a model call",
                    score=(claims.get(name) or {}).get("score"), band=(claims.get(name) or {}).get("band"))
            return {"claims": claims, "agent_notes": notes, "trace": em.events}
        reason = bud.run_limit_reason(limits, b)
        wrap_up = False
        if reason is None:
            reason = bud.agent_limit_reason(limits, b)
            if reason == "agent_tool_call_limit" and not notes.get("wrap_up_done"):
                wrap_up, notes["wrap_up_done"], reason = True, True, None
        if reason:
            notes["end_reason"] = reason
            em.emit("agent_limit", name, reason=reason)
            return {"agent_notes": notes, "budgets": b, "trace": em.events}
        msgs = list(state["agent_messages"])
        pre: dict = {}           # state updates made by the prefetch step (must survive every return below)
        if eff.get("prefetch") and spec.prefetch and not wrap_up and not notes.get("prefetch_done"):
            ledger = copy.deepcopy(state["ledger"])
            requested = (state.get("options") or {}).get("fields")
            focus = (fields_mod.focus_company_fields(state) if name == "company" else
                     [k for k in requested if fields_mod.BY_KEY[k].agent == name]) if requested else None
            outcome = spec.prefetch(executor=executor, agent=name, subject=state["subject"], ledger=ledger,
                                    budgets=b, claims=state["claims"], limits=limits, fields=focus,
                                    emit=lambda type_, **data: em.emit(type_, name, **data))
            notes["prefetch_done"] = True
            notes["prefetch"] = {"status": outcome.status, "reason": outcome.reason, "pages": outcome.pages}
            notes["packet_phase"] = outcome.status == "packet"      # first turn: extract from the packet, nothing else
            if outcome.text:           # appended to the task (first message); the stored history keeps it for audit
                msgs[0] = {**msgs[0], "content": msgs[0]["content"] + "\n\n" + outcome.text}
            pre = {"ledger": ledger, "agent_messages": msgs}
        if wrap_up:
            msgs.append({"role": "user", "content": "Tool budget reached. Submit your findings now with what "
                                                    "you have verified - do not call any other tool."})
        all_specs = [s for s in registry.specs_for(name) if s.name in spec.tool_names]
        packet_turn = bool(notes.get("packet_phase")) and not wrap_up
        tools = [s for s in all_specs if s.name == spec.finish_tool] if (wrap_up or packet_turn) else all_specs
        if eff.get("compact_schemas"):
            tools = [t.model_copy(update={"parameters": compact_schema(t.parameters)}) for t in tools]
        sent = compact_messages(msgs) if eff.get("compact_history") else msgs     # stored history stays complete
        system = system_prompt(spec, eff)
        if (budget := _tpm_limit(deps.llm)) is not None:
            before = estimate_request_tokens(system, sent, tools, limits.max_output_tokens)
            sent, steps = fit_to_budget(
                sent, lambda m: estimate_request_tokens(system, m, tools, limits.max_output_tokens), budget)
            if steps:
                em.emit("context_trimmed", name, budget=budget, estimate_before=before,
                        estimate_after=estimate_request_tokens(system, sent, tools, limits.max_output_tokens),
                        steps=steps)
        only_finish = len(tools) == 1 and tools[0].name == spec.finish_tool       # nothing else to choose: require it
        notes["offered_tools"] = [t.name for t in tools]        # the contract agent_tools enforces for this turn's calls
        try:
            resp = call_llm(deps, b, system=system, messages=sent, tools=tools,
                            emitter=em, agent=name, force_tool=wrap_up or only_finish, count_agent=not wrap_up)
        except BudgetStop as stop:
            notes["end_reason"] = stop.reason
            return {**pre, "agent_notes": notes, "budgets": b, "trace": em.events}
        except LLMError as exc:
            notes["end_reason"] = f"llm_{exc.kind}"
            return {**pre, "agent_notes": notes, "budgets": b, "trace": em.events,
                    "errors": list(state["errors"]) + [{"stage": f"{name}_llm", "error": f"llm_{exc.kind}"}]}
        msgs.append({"role": "assistant", "content": resp.text or "",
                     "tool_calls": [c.model_dump() for c in resp.tool_calls],
                     "provider_data": resp.provider_data})
        if resp.tool_calls:
            notes["no_tool_turns"] = 0
        else:
            notes["no_tool_turns"] = notes.get("no_tool_turns", 0) + 1
            if notes["no_tool_turns"] >= 2:
                notes["end_reason"] = "no_tool_calls"
            else:
                msgs.append({"role": "user", "content": "Call one of your tools. To finish, call "
                                                        f"`{spec.finish_tool}`."})
        return {**pre, "agent_messages": msgs, "budgets": b, "agent_notes": notes, "trace": em.events}

    # -------------------------------------------------------------- agent tools
    def agent_tools(state: RunState):
        em = Emitter(state)
        b = copy.deepcopy(state["budgets"])
        ledger = copy.deepcopy(state["ledger"])
        claims = {k: list(v) if isinstance(v, list) else v for k, v in state["claims"].items()}
        rejected = {k: list(v) for k, v in state["rejected"].items()}
        subject = copy.deepcopy(state["subject"])
        notes = dict(state["agent_notes"])
        msgs = list(state["agent_messages"])
        name = state["current_agent"]
        spec = specs[name]
        calls = msgs[-1]["tool_calls"]
        offered = frozenset(notes["offered_tools"]) if notes.get("offered_tools") else None
        stop = None
        finished = False
        for i, c in enumerate(calls):
            if stop or finished or i >= MAX_CALLS_PER_TURN:
                msgs.append(_tool_message(c, {"error": {"code": "skipped",
                                                        "message": "Call skipped (limit, stop or already finished)."}}, False))
                continue
            req = ToolCallRequest(**c)
            tool = registry.get(req.name)
            em.emit("tool_call", name, tool=req.name, args=req.args)
            res = executor.execute(agent=name, call=req, budgets=b, ledger=ledger, subject=subject, claims=claims,
                                   offered=offered)
            if not res.ok and (res.error or {}).get("code") == "tool_not_offered":
                em.emit("undeclared_tool_call", name, tool=req.name, offered=sorted(offered or []))
            is_finish = bool(tool and tool.kind == "finish")
            if is_finish and res.ok:
                b["agent_submit_attempts"] += 1
                view = {**state, "ledger": ledger, "claims": claims, "subject": subject}
                pr = spec.process(view, res.data, deps)
                for k, v in pr.extra.get("claims_override", {}).items():
                    claims[k] = v
                if pr.accepted:
                    claims.setdefault(name, []).extend(pr.accepted)
                if pr.rejected:
                    rejected.setdefault(name, []).extend(pr.rejected)
                subject.update(pr.subject_updates)
                notes["accepted"] = notes.get("accepted", 0) + len(pr.accepted)
                notes["rejected"] = notes.get("rejected", 0) + len(pr.rejected)
                em.emit("submit", name, accepted=len(pr.accepted), rejected=len(pr.rejected),
                        reasons=sorted({r for x in pr.rejected for r in x["reasons"]}),
                        status=res.data.get("status"))
                retry = (pr.rejected and (not pr.accepted or deps.settings.retry_partial_rejections) and
                         b["agent_submit_attempts"] < limits.max_submit_attempts and
                         bud.agent_limit_reason(limits, b) is None and
                         bud.run_limit_reason(limits, b) is None)
                msgs.append(_tool_message(c, {"result": pr.feedback()}, False))
                if retry:
                    continue
                if notes.get("packet_phase"):
                    notes["packet_phase"] = False                       # the packet turn is over
                    todo = followup_todo(name, claims, state)
                    if (todo and not notes.get("followup_used") and bud.agent_limit_reason(limits, b) is None
                            and bud.run_limit_reason(limits, b) is None and bud.credits_remaining_run(limits, b) > 0):
                        notes["followup_used"] = True                   # ONE targeted follow-up, then finish
                        em.emit("packet_followup", name, unresolved=todo)
                        msgs.append({"role": "user", "content": (
                            "The evidence packet did not settle: " + ", ".join(todo) + ". You may now use web_search "
                            "and fetch_page (a few targeted calls) for ONLY those fields, then submit just the new "
                            "claims. If they cannot be found, submit with status 'partial' or 'exhausted'.")})
                        continue
                finished = True
                notes["findings"] = {k: res.data.get(k) for k in ("status", "suggest_next", "unresolved", "notes")}
                notes["end_reason"] = "submitted"
                continue
            if res.ok:
                added = 0
                for d in res.docs:
                    if ev.add_doc(ledger, url=d["url"], text=d.get("text", ""), title=d.get("title", ""),
                                  tool=req.name, call_id=req.id, max_docs=limits.ledger_max_docs,
                                  max_chars=limits.ledger_doc_chars):
                        added += 1
                em.emit("tool_result", name, tool=req.name, ok=True, cached=res.cached, credits=res.credits,
                        duration_s=round(res.duration_s, 3), docs_added=added,
                        urls=[d["url"] for d in res.docs][:6])
                msgs.append(_tool_message(c, res.data, req.name in WEB_TOOLS))
            else:
                em.emit("tool_result", name, tool=req.name, ok=False, error=res.error,
                        duration_s=round(res.duration_s, 3))
                msgs.append(_tool_message(c, {"error": res.error}, False))
                if res.stop:
                    stop = res.stop
        if stop and not finished:
            notes["end_reason"] = stop
        return {"agent_messages": msgs, "ledger": ledger, "claims": claims, "rejected": rejected,
                "subject": subject, "budgets": b, "agent_notes": notes, "trace": em.events}

    # --------------------------------------------------------------- agent done
    def agent_done(state: RunState):
        em = Emitter(state)
        name = state["current_agent"]
        notes = state["agent_notes"]
        findings = notes.get("findings") or {}
        end = notes.get("end_reason", "unknown")
        status = findings.get("status") or ("partial" if notes.get("accepted") else "exhausted")
        if end != "submitted" and status == "satisfied":
            status = "partial"
        b = state["budgets"]
        result = {"status": status, "end_reason": end, "accepted": notes.get("accepted", 0),
                  "rejected": notes.get("rejected", 0), "suggest_next": findings.get("suggest_next") or [],
                  "unresolved": findings.get("unresolved") or [], "notes": findings.get("notes") or "",
                  "ledger_size": len(state["ledger"]), "tool_calls": b["agent_tool_calls"],
                  "llm_calls": b["agent_llm_calls"]}
        results = dict(state["agent_results"])
        results[name] = result
        completed = list(state["completed_agents"])
        if name not in completed:
            completed.append(name)
        em.emit("agent_finished", name, status=status, end_reason=end, accepted=result["accepted"],
                rejected=result["rejected"], tool_calls=result["tool_calls"])
        return {"agent_results": results, "completed_agents": completed, "current_agent": None,
                "agent_messages": [], "agent_notes": {}, "trace": em.events}

    # ------------------------------------------------------- parallel stage
    def parallel_run(state: RunState):
        """Run the independent agents of one stage concurrently on private state copies, then merge in plan order."""
        names = [n for n in state["parallel_agents"] if n in specs]
        base_b = copy.deepcopy(state["budgets"])
        shares = parallel.split_budgets(base_b, limits, len(names))
        em = Emitter(state)
        em.emit("parallel_start", "supervisor", agents=names,
                reserved_for_others=[dict(r) for _, r in shares])
        base_len = len(state["trace"])

        def run_branch(name: str, share: tuple):
            spec = specs[name]
            b = share[0]
            s = dict(state)
            s.update(ledger=copy.deepcopy(state["ledger"]), claims=copy.deepcopy(state["claims"]),
                     rejected=copy.deepcopy(state["rejected"]), subject=copy.deepcopy(state["subject"]),
                     errors=[], trace=[], budgets=b, parallel_agents=[], current_agent=name)
            reset_agent_counters(b)
            runs = dict(state["agent_runs"])
            runs[name] = runs.get(name, 0) + 1
            s.update(agent_runs=runs, agent_results=dict(state["agent_results"]),
                     completed_agents=list(state["completed_agents"]),
                     agent_messages=[{"role": "user", "content": agent_task(s, spec, b)}],
                     agent_notes={"no_tool_turns": 0, "accepted": 0, "rejected": 0,
                                  "ledger_size_start": len(state["ledger"])})

            def apply(upd):
                for k, v in upd.items():
                    s[k] = (s["trace"] + v) if k == "trace" else v
            guard = limits.max_llm_calls_agent * 3 + limits.max_tool_calls_agent * 2 + 10
            try:
                for _ in range(guard):                       # same transitions as the graph edges
                    apply(agent_llm(s))
                    if s["agent_notes"].get("end_reason"):
                        break
                    last = s["agent_messages"][-1]
                    if last["role"] == "assistant" and last.get("tool_calls"):
                        apply(agent_tools(s))
                        if s["agent_notes"].get("end_reason"):
                            break
                else:
                    s["agent_notes"]["end_reason"] = "branch_step_guard"
                apply(agent_done(s))
            except Exception as exc:  # noqa: BLE001 - one failing agent must not take the others down
                s["branch_error"] = type(exc).__name__
            return s

        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=max(1, min(len(names), limits.max_parallel_agents))) as pool:
            futures = [pool.submit(contextvars.copy_context().run, run_branch, n, shares[i])
                       for i, n in enumerate(names)]
            branches = [f.result() for f in futures]            # plan order, not completion order

        ledger = parallel.merge_ledgers(state["ledger"], [s["ledger"] for s in branches],
                                        max_docs=limits.ledger_max_docs, max_chars=limits.ledger_doc_chars)
        claims = copy.deepcopy(state["claims"])
        rejected = copy.deepcopy(state["rejected"])
        results, completed, runs = dict(state["agent_results"]), list(state["completed_agents"]), dict(state["agent_runs"])
        errors = list(state["errors"])
        for n, s in zip(names, branches):
            runs[n] = runs.get(n, 0) + 1
            if n not in completed:
                completed.append(n)
            if s.get("branch_error") or n not in s["agent_results"]:
                results[n] = {"status": "exhausted", "end_reason": "branch_exception", "accepted": 0, "rejected": 0,
                              "suggest_next": [], "unresolved": [], "notes": "", "ledger_size": len(ledger),
                              "tool_calls": s["budgets"]["agent_tool_calls"], "llm_calls": s["budgets"]["agent_llm_calls"]}
                errors.append({"stage": f"{n}_branch", "error": s.get("branch_error", "no_result")})
            else:
                results[n] = s["agent_results"][n]
                if n in s["claims"]:
                    claims[n] = s["claims"][n]
                if n in s["rejected"]:
                    rejected[n] = s["rejected"][n]
            errors.extend(s["errors"])
        budgets = parallel.merge_budgets(base_b, [(s["budgets"], shares[i][1]) for i, s in enumerate(branches)])
        events = parallel.renumber_events(base_len + len(em.events), [s["trace"] for s in branches])
        fin = Emitter({"trace": [None] * (base_len + len(em.events) + len(events))})
        fin.emit("parallel_finish", "supervisor", agents=names,
                 results={n: {"status": results[n].get("status"), "end_reason": results[n].get("end_reason"),
                              "accepted": results[n].get("accepted")} for n in names})
        return {"ledger": ledger, "claims": claims, "rejected": rejected, "agent_results": results,
                "completed_agents": completed, "agent_runs": runs, "errors": errors, "budgets": budgets,
                "parallel_agents": [], "current_agent": None, "agent_messages": [], "agent_notes": {},
                "trace": em.events + events + fin.events}

    # ----------------------------------------------------------------- finalize
    def finalize(state: RunState):
        em = Emitter(state)
        status = run_status(state)
        view = {**state, "status": status}
        report = build_report(view)
        report["requested_fields"] = (state.get("options") or {}).get("fields") or []
        report["change_set"] = build_change_set(view)
        em.emit("run_finished", None, status=status, stop_reason=state.get("stop_reason"))
        return {"status": status, "status_reasons": (report["run"]["status_reasons"] if status != "failed" else []),
                "report": report, "budgets": {**state["budgets"], "db_selects": sb.used()}, "trace": em.events}

    # ---------------------------------------------------- proposals / approval
    svc = (ProposalService(deps.store, deps.commit_repo, schema=getattr(deps, "schema", None))
           if deps.commit_repo is not None else None)

    def propose(state: RunState):
        em = Emitter(state)
        if svc is None:
            em.emit("propose_skipped", None, reason="no_commit_repository_configured")
            return {"trace": em.events}
        if (state.get("options") or {}).get("fields"):     # field request: ONLY the reviewed change set is proposed
            res = svc.propose_from_change_set(state, (state.get("report") or {}).get("change_set"))
            props = res["proposals"]
            if res["stale_items"]:
                em.emit("change_set_items_stale", None, items=res["stale_items"])
        else:
            props = svc.propose(state)
        summaries = [svc.summary(p) for p in props]
        em.emit("proposals_created", None, count=len(summaries),
                statuses=[p["status"] for p in summaries])
        out = {"proposals": summaries, "trace": em.events}
        if any(p["status"] == "pending" for p in summaries):
            out["status"] = "awaiting_approval"
        return out

    def approval(state: RunState):
        em = Emitter(state)
        pending = [p for p in state["proposals"] if p["status"] == "pending"]
        # Nothing with side effects may run before interrupt(): the node re-executes on resume.
        decisions = interrupt({"type": "approval_required", "run_id": state["run_id"], "proposals": pending})
        updated = []
        for p in state["proposals"]:
            d = (decisions or {}).get(p["proposal_id"])
            if p["status"] == "pending" and d:
                res = svc.decide(p["proposal_id"], bool(d.get("approve")), by=str(d.get("by", "")),
                                 reason=str(d.get("reason", "")), approved_overwrite=bool(d.get("approved_overwrite")),
                                 overrides=d.get("overrides") or {})
                row = deps.store.get_proposal(p["proposal_id"])
                p = {**p, "status": row["status"], "decision": {k: v for k, v in res.items() if k != "ok"}}
                em.emit("approval_decision", None, proposal_id=p["proposal_id"], approved=bool(d.get("approve")),
                        result=res.get("status") or res.get("error"))
            updated.append(p)
        return {"proposals": updated, "approvals": list(state.get("approvals", [])) + [
            {"proposal_id": k, "approve": bool(v.get("approve")), "by": v.get("by")}
            for k, v in (decisions or {}).items()], "trace": em.events}

    def commit(state: RunState):
        em = Emitter(state)
        updated = []
        for p in state["proposals"]:
            if p["status"] == "approved":
                res = svc.apply(p["proposal_id"], actor="commit_node")
                row = deps.store.get_proposal(p["proposal_id"])
                p = {**p, "status": row["status"]}
                em.emit("proposal_applied" if res.get("ok") else "proposal_apply_failed", None,
                        proposal_id=p["proposal_id"], result=res.get("status") or res.get("error"))
            updated.append(p)
        view = {**state, "status": "running", "proposals": updated}
        status = run_status(view)
        report = build_report({**view, "status": status})
        em.emit("run_finished", None, status=status, proposals=[p["status"] for p in updated])
        return {"proposals": updated, "status": status, "status_reasons": report["run"]["status_reasons"],
                "report": report, "trace": em.events}

    # ------------------------------------------------------------------- wiring
    g = StateGraph(RunState)
    for node_name, fn in (("intake", intake), ("supervisor", supervisor), ("agent_llm", agent_llm),
                          ("agent_tools", agent_tools), ("agent_done", agent_done), ("finalize", finalize)):
        g.add_node(node_name, persisted(fn))
    for node_name, fn in (("propose", propose), ("approval", approval), ("commit", commit)):
        g.add_node(node_name, persisted(fn))
    g.add_node("parallel_run", persisted(parallel_run))

    g.add_edge(START, "intake")
    g.add_conditional_edges("intake", lambda s: "finalize" if s.get("status") == "failed" else "supervisor",
                            {"finalize": "finalize", "supervisor": "supervisor"})
    g.add_conditional_edges("supervisor",
                            lambda s: "parallel_run" if s.get("parallel_agents") else
                            ("agent_llm" if s.get("current_agent") else "finalize"),
                            {"agent_llm": "agent_llm", "parallel_run": "parallel_run", "finalize": "finalize"})
    g.add_edge("parallel_run", "supervisor")

    def after_llm(s):
        notes = s["agent_notes"]
        if notes.get("end_reason"):
            return "agent_done"
        last = s["agent_messages"][-1]
        return "agent_tools" if last["role"] == "assistant" and last.get("tool_calls") else "agent_llm"
    g.add_conditional_edges("agent_llm", after_llm, {"agent_done": "agent_done", "agent_tools": "agent_tools",
                                                      "agent_llm": "agent_llm"})
    g.add_conditional_edges("agent_tools", lambda s: "agent_done" if s["agent_notes"].get("end_reason") else "agent_llm",
                            {"agent_done": "agent_done", "agent_llm": "agent_llm"})
    g.add_edge("agent_done", "supervisor")
    g.add_conditional_edges("finalize", lambda s: "propose" if (s.get("options") or {}).get("propose_changes") else END,
                            {"propose": "propose", END: END})
    g.add_conditional_edges("propose", lambda s: "approval" if any(p["status"] == "pending" for p in s.get("proposals", []))
                            else END, {"approval": "approval", END: END})
    g.add_edge("approval", "commit")
    g.add_edge("commit", END)
    return g.compile(checkpointer=checkpointer)

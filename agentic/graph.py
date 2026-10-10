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

import copy
import json
import re

from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from agentic import evidence as ev
from agentic import gaps as gp
from agentic.agents.common import system_prompt
from agentic.context import compact_messages, compact_schema, ledger_hint
from agentic.guards import budget as bud
from agentic.guards.budget import BudgetStop, reset_agent_counters
from agentic.guards.executor import GuardedExecutor
from agentic.llm.base import LLMError, ToolCallRequest
from agentic.llmcall import call_llm
from agentic.proposals import ProposalService
from agentic.report import build_report, build_tracker, run_status
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

    # ------------------------------------------------------------------ intake
    def intake(state: RunState):
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
        existing = None
        primary = res["primary"]
        if primary:
            cid = primary["company_id"]
            existing = {"company_id": cid,
                        "fields": {k: primary.get(k) for k in ("company_name", "country", "sector_name", "website",
                                                              "linkedin_url", "revenue", "number_of_plants",
                                                              "company_type", "enrichment_status")},
                        "provenance": "database_unverified"}
            for key, getter in (("products", deps.repo.get_products), ("components", deps.repo.get_components),
                                ("contact_summary", deps.repo.get_contact_summary),
                                ("suppliers", deps.repo.get_company_suppliers),
                                ("lead_status", deps.repo.get_lead_status)):
                try:
                    val = getter(cid)
                    existing[key] = val[:30] if isinstance(val, list) else val
                except Exception as exc:  # noqa: BLE001
                    errors.append({"stage": f"db_{key}", "error": type(exc).__name__})
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
        decision, elig = decide({**state, "budgets": b}, deps, specs, em)
        history = list(state["route_history"]) + [{
            "step": steps + 1, "next": decision["next"], "reason": decision["reason"],
            "vetoes": decision["vetoes"], "eligible": [k for k, v in elig.items() if v["eligible"]],
            "deterministic": bool(decision.get("deterministic"))}]
        em.emit("route", "supervisor", next=decision["next"], reason=decision["reason"],
                deterministic=bool(decision.get("deterministic")), vetoes=decision["vetoes"], eligible=[k for k, v in elig.items() if v["eligible"]])
        update = {"budgets": b, "supervisor_steps": steps + 1, "route_history": history,
                  "trace": em.events}
        if decision["next"] == "finish":
            update.update({"current_agent": None,
                           "stop_reason": decision["error"] or "supervisor_finished"})
            return update
        spec = specs[decision["next"]]
        eff = (state.get("options") or {}).get("efficiency") or {}
        task = spec.task(state) + "\n\n" + _budget_line(spec, b)
        if eff.get("ledger_hint") and spec.name != "lead" and (hint := ledger_hint(state["ledger"])):
            task += "\n\n" + hint
        reset_agent_counters(b)
        runs = dict(state["agent_runs"])
        runs[spec.name] = runs.get(spec.name, 0) + 1
        update.update({"current_agent": spec.name, "agent_objective": spec.goal, "agent_runs": runs,
                       "agent_messages": [{"role": "user", "content": task}],
                       "agent_notes": {"no_tool_turns": 0, "accepted": 0, "rejected": 0,
                                       "ledger_size_start": len(state["ledger"])}})
        return update

    def _budget_line(spec, b):
        return (f"Budget: at most {limits.max_tool_calls_agent} tool calls, "
                f"{bud.credits_remaining_run(limits, b)} search credits left "
                f"(basic search = 1 credit, advanced = 2). Submit with `{spec.finish_tool}` when done.")

    # --------------------------------------------------------------- agent LLM
    def agent_llm(state: RunState):
        em = Emitter(state)
        b = copy.deepcopy(state["budgets"])
        notes = dict(state["agent_notes"])
        name = state["current_agent"]
        spec = specs[name]
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
        if wrap_up:
            msgs.append({"role": "user", "content": "Tool budget reached. Submit your findings now with what "
                                                    "you have verified - do not call any other tool."})
        all_specs = [s for s in registry.specs_for(name) if s.name in spec.tool_names]
        tools = [s for s in all_specs if s.name == spec.finish_tool] if wrap_up else all_specs
        eff = (state.get("options") or {}).get("efficiency") or {}
        if eff.get("compact_schemas"):
            tools = [t.model_copy(update={"parameters": compact_schema(t.parameters)}) for t in tools]
        sent = compact_messages(msgs) if eff.get("compact_history") else msgs     # stored history stays complete
        try:
            resp = call_llm(deps, b, system=system_prompt(spec, eff), messages=sent, tools=tools,
                            emitter=em, agent=name, force_tool=wrap_up, count_agent=not wrap_up)
        except BudgetStop as stop:
            notes["end_reason"] = stop.reason
            return {"agent_notes": notes, "budgets": b, "trace": em.events}
        except LLMError as exc:
            notes["end_reason"] = f"llm_{exc.kind}"
            return {"agent_notes": notes, "budgets": b, "trace": em.events,
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
        return {"agent_messages": msgs, "budgets": b, "agent_notes": notes, "trace": em.events}

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
            res = executor.execute(agent=name, call=req, budgets=b, ledger=ledger, subject=subject, claims=claims)
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
                retry = (not pr.accepted and pr.rejected and
                         b["agent_submit_attempts"] < limits.max_submit_attempts and
                         bud.agent_limit_reason(limits, b) is None and
                         bud.run_limit_reason(limits, b) is None)
                msgs.append(_tool_message(c, {"result": pr.feedback()}, False))
                if retry:
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

    # ----------------------------------------------------------------- finalize
    def finalize(state: RunState):
        em = Emitter(state)
        status = run_status(state)
        view = {**state, "status": status}
        report = build_report(view)
        em.emit("run_finished", None, status=status, stop_reason=state.get("stop_reason"))
        return {"status": status, "report": report, "trace": em.events}

    # ---------------------------------------------------- proposals / approval
    svc = ProposalService(deps.store, deps.commit_repo) if deps.commit_repo is not None else None

    def propose(state: RunState):
        em = Emitter(state)
        if svc is None:
            em.emit("propose_skipped", None, reason="no_commit_repository_configured")
            return {"trace": em.events}
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
        return {"proposals": updated, "status": status, "report": report, "trace": em.events}

    # ------------------------------------------------------------------- wiring
    g = StateGraph(RunState)
    for node_name, fn in (("intake", intake), ("supervisor", supervisor), ("agent_llm", agent_llm),
                          ("agent_tools", agent_tools), ("agent_done", agent_done), ("finalize", finalize)):
        g.add_node(node_name, persisted(fn))
    for node_name, fn in (("propose", propose), ("approval", approval), ("commit", commit)):
        g.add_node(node_name, persisted(fn))

    g.add_edge(START, "intake")
    g.add_conditional_edges("intake", lambda s: "finalize" if s.get("status") == "failed" else "supervisor",
                            {"finalize": "finalize", "supervisor": "supervisor"})
    g.add_conditional_edges("supervisor", lambda s: "agent_llm" if s.get("current_agent") else "finalize",
                            {"agent_llm": "agent_llm", "finalize": "finalize"})

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

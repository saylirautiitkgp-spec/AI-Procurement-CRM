"""LLM-driven supervisor with deterministic routing guards.

The LLM chooses the next agent (or finish) by calling the `route` tool, based on
the objective, the information gaps and what the agents have already reported.
Python then checks the choice against prerequisites, run limits and
no-progress rules. A vetoed choice is explained to the model once; if it is
still invalid the run ends as partial with a recorded reason - Python never
silently substitutes its own plan."""

from __future__ import annotations

import json
from typing import Literal

from pydantic import BaseModel, Field, ValidationError

from agentic import gaps as gp
from agentic import routing
from agentic.agents.common import ALL_AGENTS
from agentic.guards import budget as bud
from agentic.guards.budget import BudgetStop
from agentic.llm.base import LLMError, ToolSpec
from agentic.state import Emitter

AGENT_DESCRIPTIONS = {
    "company": "Company Intelligence - confirms the organisation's identity/official website and profile facts "
               "(HQ, industry, products, size, locations). Needed before any other web research.",
    "contact": "Contact Intelligence - finds publicly listed procurement/sourcing decision-makers with source evidence.",
    "procurement": "Procurement Intelligence - finds supplier portals, procurement pages, purchasing categories and "
                   "what the company buys.",
    "supplier": "Supplier & Relationship Intelligence - finds suppliers/partners explicitly named for this company "
                "and for its components.",
    "lead": "Lead Intelligence & Prioritisation - computes an explainable lead-priority score from verified data "
            "(no web access).",
}

SUPERVISOR_SYSTEM = """\
You are the supervisor of a multi-agent procurement-intelligence system. You do not research anything yourself.
Each turn you must call the `route` tool exactly once to pick the NEXT agent, or `finish`.

How to decide:
- Read the user's objective and choose only the agents that help achieve it. Do not run an agent just because it exists.
- Look at the information gaps and at what earlier agents reported (including 'suggest_next').
- Prefer an agent whose prerequisites are met (see eligibility). If a prerequisite is missing, route to the agent
  that supplies it (e.g. identity must be confirmed by the company agent before web research on contacts).
- Do not re-run an agent that is satisfied or that found nothing new.
- Choose `finish` when the objective is satisfied or no eligible agent can add value within the remaining budget.
Give a short, concrete reason.
"""


SHORT_DESCRIPTIONS = {
    "company": "identity, official site, profile facts",
    "contact": "public procurement decision-makers",
    "procurement": "supplier portal, procurement pages, purchased categories",
    "supplier": "explicitly evidenced suppliers",
    "lead": "explainable lead score from verified data (no web)",
}


class RouteArgs(BaseModel):
    next: Literal["company", "contact", "procurement", "supplier", "lead", "finish"]
    reason: str = Field(max_length=300)


ROUTE_TOOL = ToolSpec(name="route", description="Choose the next agent or finish.",
                      parameters=RouteArgs.model_json_schema())


def eligibility(state: dict, specs: dict, limits) -> dict:
    b = state["budgets"]
    subj = state["subject"]
    identity_ok = bool((subj.get("identity") or {}).get("verified"))
    search_off = bool(b.get("search_disabled"))
    credits_left = bud.credits_remaining_run(limits, b)
    existing = subj.get("existing") or {}
    out: dict = {}
    for name in ALL_AGENTS:
        if name not in specs:
            out[name] = {"eligible": False, "reason": "not available in this build"}
            continue
        runs = state["agent_runs"].get(name, 0)
        last = state["agent_results"].get(name)
        if runs >= limits.max_agent_runs_per_agent:
            out[name] = {"eligible": False, "reason": "maximum runs for this agent reached"}
            continue
        if runs >= 1:
            grew = len(state["ledger"]) > (last or {}).get("ledger_size", 0)
            if (last or {}).get("status") == "satisfied" or not grew:
                out[name] = {"eligible": False, "reason": "already ran and nothing new has been learned since"}
                continue
        needs_web = name != "lead"
        if needs_web and search_off:
            out[name] = {"eligible": False, "reason": f"search unavailable ({b.get('search_disabled_reason')})"}
            continue
        if needs_web and credits_left <= 0:
            out[name] = {"eligible": False, "reason": "search credit budget for this run is used up"}
            continue
        if name in ("contact", "procurement", "supplier") and not identity_ok:
            out[name] = {"eligible": False, "reason": "organisation identity not confirmed yet - run company first"}
            continue
        if name == "supplier":
            known = (existing.get("components") or existing.get("products")
                     or gp.verified_fields(state["claims"], "procurement"))
            if not known:
                out[name] = {"eligible": False, "reason": "no known products/components to relate suppliers to - "
                                                          "run company/procurement first"}
                continue
        if name == "lead" and not (state["agent_runs"].get("company") or existing.get("fields")):
            out[name] = {"eligible": False, "reason": "no company data yet - run company first"}
            continue
        out[name] = {"eligible": True, "reason": "prerequisites met"}
    return out


def _summary(state: dict, elig: dict, limits) -> str:
    s = state["subject"]
    b = state["budgets"]
    eff = (state.get("options") or {}).get("efficiency") or {}
    comp = gp.company_gaps(s, state["claims"])
    lines = [
        f"OBJECTIVE: {state['objective']}",
        f"COMPANY: {s['name']} | database match: {s['match']['status']} | "
        f"identity verified: {bool((s.get('identity') or {}).get('verified'))}",
        "COMPANY FIELD STATUS: " + ", ".join(f"{g['field']}={g['status']}" for g in comp),
        "OTHER AREAS: " + json.dumps(gp.other_agent_gaps(s, state["claims"], state["agent_results"])),
        f"BUDGET LEFT: supervisor steps {limits.max_supervisor_steps - state['supervisor_steps']}, "
        f"search credits {bud.credits_remaining_run(limits, b)}, "
        f"llm calls {limits.max_llm_calls_run - b['llm_calls']}",
        "AGENT RESULTS SO FAR:",
    ]
    if not state["agent_results"]:
        lines.append("  (none yet)")
    for name, r in state["agent_results"].items():
        lines.append(f"  - {name}: status={r.get('status')} stop={r.get('stop_reason')} "
                     f"accepted={r.get('accepted')} rejected={r.get('rejected')} suggest_next={r.get('suggest_next')}")
    lines.append("ELIGIBILITY:")
    for name in ALL_AGENTS:
        e = elig[name]
        lines.append(f"  - {name}: {'ELIGIBLE' if e['eligible'] else 'not eligible'} ({e['reason']})")
        if not eff.get("short_supervisor_text"):
            lines.append(f"      {AGENT_DESCRIPTIONS[name]}")
        elif e["eligible"]:
            lines.append(f"      ({SHORT_DESCRIPTIONS[name]})")
    return "\n".join(lines)


def validate_route(choice: str, elig: dict, state: dict) -> str | None:
    """Return a veto reason, or None if the routing decision is acceptable."""
    if choice == "finish":
        any_eligible = any(e["eligible"] for e in elig.values())
        if state["completed_agents"] or not any_eligible:
            return None
        return "finish_before_any_work: at least one eligible agent must run first"
    if not elig.get(choice, {}).get("eligible"):
        return f"{choice} is not eligible: {elig.get(choice, {}).get('reason', 'unknown agent')}"
    return None


def decide(state: dict, deps, specs: dict, emitter: Emitter) -> tuple[dict, dict]:
    """Returns (decision, budget_updates). decision = {next, reason, vetoes, error}."""
    from agentic.llmcall import call_llm
    limits = deps.limits
    budgets = state["budgets"]
    elig = eligibility(state, specs, limits)
    eff = (state.get("options") or {}).get("efficiency") or {}
    if eff.get("skip_forced_finish") and not any(e["eligible"] for e in elig.values()):
        return {"next": "finish", "reason": "no agent is eligible - finish is forced (decided without an LLM call)",
                "vetoes": [], "error": None, "deterministic": True}, elig
    if (state.get("options") or {}).get("fields"):          # explicit field request: Python plans from the catalog
        return routing.route_for_fields(state, elig, parallel=bool(eff.get("parallel_agents")),
                                        max_parallel=limits.max_parallel_agents), elig
    if eff.get("deterministic_routing"):
        plan = routing.classify_objective(state["objective"])
        if plan.kind == "clear":                       # clear single-capability objective: Python plans, no LLM call
            return routing.route_deterministic(state, elig, plan), elig
    messages = [{"role": "user", "content": _summary(state, elig, limits)}]
    vetoes: list[str] = []
    for attempt in range(2):
        try:
            resp = call_llm(deps, budgets, system=SUPERVISOR_SYSTEM, messages=messages,
                            tools=[ROUTE_TOOL], emitter=emitter, agent="supervisor", force_tool=True)
        except LLMError as exc:
            return {"next": "finish", "reason": f"llm_{exc.kind}", "vetoes": vetoes,
                    "error": f"llm_{exc.kind}"}, elig
        except BudgetStop as stop:
            return {"next": "finish", "reason": stop.reason, "vetoes": vetoes, "error": stop.reason}, elig
        call = next((c for c in resp.tool_calls if c.name == "route"), None)
        err = None
        if call is None:
            err = "no route tool call was made"
        else:
            try:
                args = RouteArgs.model_validate(call.args)
                err = validate_route(args.next, elig, state)
            except ValidationError as exc:
                err = "invalid route arguments: " + "; ".join(e["msg"] for e in exc.errors()[:3])
        if err is None:
            return {"next": args.next, "reason": args.reason, "vetoes": vetoes, "error": None}, elig
        vetoes.append(err)
        emitter.emit("route_vetoed", "supervisor", reason=err, attempt=attempt + 1)
        messages.append({"role": "assistant", "content": resp.text or "",
                         "tool_calls": [c.model_dump() for c in resp.tool_calls],
                         "provider_data": resp.provider_data})
        if call is not None:
            messages.append({"role": "tool", "tool_call_id": call.id, "name": "route",
                             "content": json.dumps({"error": err})})
        messages.append({"role": "user", "content": f"Your routing choice was rejected: {err}. "
                                                      "Choose again using the eligibility table."})
    return {"next": "finish", "reason": "supervisor_invalid_routing", "vetoes": vetoes,
            "error": "supervisor_invalid_routing"}, elig

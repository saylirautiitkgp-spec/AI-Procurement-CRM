"""Deterministic objective -> agent routing (hybrid supervisor).

Python handles what does not need judgement: a CLEAR objective names exactly one target capability (company profile,
decision-makers, procurement structure, suppliers, lead ranking). The agents it needs first (prerequisites) follow from
state, so the next step can be chosen without an LLM call. Comprehensive, multi-agent and ambiguous objectives stay
with the LLM supervisor, which also keeps its veto checks.

Nothing here weakens a safeguard: the chosen agent must still be ELIGIBLE (identity confirmed, search available,
credits left, run limits). If a planned agent cannot run, the run finishes as `partial` with the reason recorded -
no silent substitute plan.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# --------------------------------------------------------------------------------------------------------- keywords
_COMPREHENSIVE = re.compile(r"\b(comprehensive|full (report|intelligence|research|profile)|everything|all agents|"
                            r"end[- ]to[- ]end|complete (report|intelligence|research))\b")
_GROUPS: dict[str, re.Pattern] = {
    "company": re.compile(r"\b(company profile|profile|overview|about the company|identity|official (web)?site|website|"
                          r"headquarters?|hq|industry|business overview|who is)\b"),
    "contact": re.compile(r"\b(decision[- ]?makers?|procurement (head|heads|manager|managers|lead|leads|contacts?|team)|"
                          r"purchas(e|ing) (heads?|managers?|leads?|teams?)|contacts?|who (buys|purchases|handles)|key people|stakeholders?)\b"),
    "procurement": re.compile(r"\b(procurement (structure|process|page|pages|portal)|supplier (portal|registration|onboarding)|"
                              r"sourcing (page|process)|purchasing categories|purchased (components?|products?)|"
                              r"what (does|do) \w+ (buy|purchase)|components? (purchased|bought)|how \w+ buys)\b"),
    "supplier": re.compile(r"\b(suppliers?|who supplies|vendors?|supply[- ]chain partners?)\b"),
    "lead": re.compile(r"\b(rank|prioriti[sz]e|prioriti[sz]ation|priority|lead (score|scoring|priority)|scoring|score|qualif\w+)\b"),
}
_SUPPLIER_PORTAL = re.compile(r"\bsupplier (portal|registration|onboarding)\b")      # belongs to procurement
_INTERPRET = re.compile(r"\b(explain|why|interpret\w*|recommend\w*|analy[sz]e|analysis|justify|reasoning|rationale)\b")

# CRM sections (tables) each agent reads. Everything else is not loaded for that agent. All values are UNVERIFIED.
CRM_NEEDS: dict[str, tuple[str, ...]] = {
    "company": ("products",),                                  # product names seed the products_services gap
    "contact": ("contact_summary",),                           # counts + designations only
    "procurement": ("products", "components"),
    "supplier": ("products", "components", "suppliers"),
    "lead": ("components",),                                   # scoring: CRM-listed components at half weight
}


@dataclass(frozen=True)
class ObjectivePlan:
    kind: str                       # "clear" (deterministic) | "llm" (comprehensive / multi-agent / ambiguous)
    target: str | None = None
    reason: str = ""
    matched: tuple[str, ...] = field(default_factory=tuple)


def classify_objective(objective: str | None) -> ObjectivePlan:
    text = " ".join((objective or "").lower().split())
    if not text:
        return ObjectivePlan("llm", reason="empty objective")
    if _COMPREHENSIVE.search(text):
        return ObjectivePlan("llm", reason="comprehensive objective - the supervisor decides which agents help")
    scrubbed = _SUPPLIER_PORTAL.sub(" ", text)
    matched = tuple(name for name, rx in _GROUPS.items()
                    if rx.search(text if name != "supplier" else scrubbed))
    if len(matched) == 1:
        return ObjectivePlan("clear", matched[0], f"objective names a single capability: {matched[0]}", matched)
    if not matched:
        return ObjectivePlan("llm", reason="objective matches no known capability - ambiguous", matched=matched)
    return ObjectivePlan("llm", reason=f"objective spans several agents {list(matched)} - the supervisor decides",
                         matched=matched)


def wants_interpretation(objective: str | None) -> bool:
    """True when the user asks for explanation/recommendations, where an LLM reading of the lead score adds value."""
    return bool(_INTERPRET.search((objective or "").lower()))


# ---------------------------------------------------------------------------------------------- prerequisite chain
def _known_products_or_components(state: dict) -> bool:
    from agentic import gaps as gp
    ex = (state["subject"].get("existing") or {})
    return bool(ex.get("components") or ex.get("products") or gp.verified_fields(state["claims"], "procurement"))


def prerequisite_chain(target: str, state: dict) -> list[str]:
    """Agents that must run, in order, for `target` (the target is last). Mirrors supervisor.eligibility()."""
    subject = state["subject"]
    identity_ok = bool((subject.get("identity") or {}).get("verified"))
    ran_company = bool(state["agent_runs"].get("company"))
    chain: list[str] = []
    if target == "company":
        return ["company"]
    if target in ("contact", "procurement", "supplier") and not identity_ok:
        chain.append("company")
    if target == "supplier" and not _known_products_or_components(state):
        chain.append("procurement")
    if target == "lead" and not (ran_company or (subject.get("existing") or {}).get("fields")):
        chain.append("company")
    chain.append(target)
    return chain


def route_deterministic(state: dict, elig: dict, plan: ObjectivePlan) -> dict:
    """Next step for a CLEAR objective. Same decision shape as the LLM supervisor, `deterministic=True`."""
    target = plan.target
    for agent in prerequisite_chain(target, state):
        if state["agent_runs"].get(agent, 0) >= 1:
            continue                                            # each planned agent runs once; no loops
        e = elig.get(agent, {"eligible": False, "reason": "unknown agent"})
        if e["eligible"]:
            why = "prerequisite of " + target if agent != target else plan.reason
            return {"next": agent, "reason": f"deterministic plan: {why}", "vetoes": [], "error": None,
                    "deterministic": True}
        return {"next": "finish", "vetoes": [], "deterministic": True, "error": "planned_agent_blocked",
                "reason": f"planned agent '{agent}' cannot run: {e['reason']}"}
    ran = [a for a, n in state["agent_runs"].items() if n >= 1]
    return {"next": "finish", "reason": f"deterministic plan complete ({', '.join(ran) or 'no agents'})",
            "vetoes": [], "error": None, "deterministic": True}


def crm_sections_for(agents) -> list[str]:
    seen: list[str] = []
    for a in agents:
        for s in CRM_NEEDS.get(a, ()):
            if s not in seen:
                seen.append(s)
    return seen


def route_for_fields(state: dict, elig: dict, *, parallel: bool, max_parallel: int) -> dict:
    """Next step for a FIELD REQUEST: the stage plan comes from agentic/fields.py. Independent agents of one stage are
    returned together (`next="parallel"`) when parallel execution is on. Blocked agents end the run as partial."""
    from agentic import fields as fl
    todo, need = fl.next_stage(state)
    if not todo:
        ran = sorted(a for a, n in state["agent_runs"].items() if n >= 1)
        if not ran:
            return {"next": "finish", "vetoes": [], "deterministic": True, "error": None, "stop": "nothing_to_research",
                    "reason": "requested fields are already stored in the CRM; nothing to research (use refresh to re-verify)"}
        return {"next": "finish", "vetoes": [], "deterministic": True, "error": None,
                "reason": f"field plan complete ({', '.join(ran)})"}
    runnable = [a for a in todo if elig.get(a, {}).get("eligible")]
    blocked = [a for a in todo if a not in runnable]
    if not runnable:
        why = elig.get(blocked[0], {}).get("reason", "unknown")
        return {"next": "finish", "vetoes": [], "deterministic": True, "error": "planned_agent_blocked",
                "reason": f"planned agent '{blocked[0]}' cannot run: {why}"}
    if parallel and len(runnable) > 1 and all(a in fl.PARALLEL_SAFE for a in runnable):
        agents = runnable[:max(1, max_parallel)]
        return {"next": "parallel", "agents": agents, "vetoes": [], "deterministic": True, "error": None,
                "reason": f"field plan: {', '.join(agents)} are independent and run in parallel"}
    return {"next": runnable[0], "vetoes": [], "deterministic": True, "error": None,
            "reason": f"field plan: {runnable[0]} (needed agents: {', '.join(need)})"}


def route_scoped(state: dict, elig: dict) -> dict:
    """Next step of an explicit SCOPED run (e.g. Company, then Procurement): the planned agents in stage order, each once.
    No supervisor LLM call, and no agent outside the scope can be chosen - not by a model, not by prerequisites, not by
    'suggest_next'. A planned agent that cannot run (identity not confirmed, no search) ends the run as partial."""
    from agentic import fields as fl
    opts = state.get("options") or {}
    scope = list(opts.get("scope_agents") or ["company"])
    base = {"vetoes": [], "deterministic": True, "error": None}
    todo, _ = fl.next_stage(state)
    if not todo:
        ran = sorted(a for a, n in state["agent_runs"].items() if n >= 1)
        if not ran:
            return {**base, "next": "finish", "stop": "nothing_to_research",
                    "reason": "requested fields are already stored in the CRM; nothing to research (use refresh)"}
        return {**base, "next": "finish", "reason": f"scoped run complete ({', '.join(ran)})"}
    nxt = todo[0]
    if nxt not in scope:
        return {**base, "next": "finish", "error": "scope_violation",
                "reason": f"planned agent '{nxt}' is outside the run's scope {scope}"}
    e = elig.get(nxt, {"eligible": False, "reason": "unknown agent"})
    if not e["eligible"]:
        return {**base, "next": "finish", "error": "planned_agent_blocked",
                "reason": f"planned agent '{nxt}' cannot run: {e['reason']}"}
    return {**base, "next": nxt, "reason": f"scoped run {scope}: {nxt} is next"}


route_company_only = route_scoped          # earlier name

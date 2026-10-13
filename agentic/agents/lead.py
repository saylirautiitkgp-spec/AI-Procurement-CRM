"""Lead Intelligence & Prioritization Agent.

The score is computed by deterministic code from VERIFIED data (agentic/scoring.py). The agent calls the
scoring tool, reads the factor breakdown and recommends next actions; it cannot alter the number."""

from __future__ import annotations

from pydantic import BaseModel, Field

from agentic.agents.common import AgentSpec, BaseFindings, ProcessResult
from agentic.scoring import score_lead
from agentic.tools.registry import Tool, ToolContext, ToolOutput


class ComputeLeadScoreArgs(BaseModel):
    pass


def compute_lead_score(ctx: ToolContext, args: ComputeLeadScoreArgs) -> ToolOutput:
    return ToolOutput(data=score_lead({"subject": ctx.subject, "claims": ctx.claims}))


class LeadFindings(BaseFindings):
    next_actions: list[str] = Field(default_factory=list, max_length=6,
                                    description="Concrete follow-ups that would raise confidence or priority.")


def process_lead(state: dict, findings: dict, deps) -> ProcessResult:
    res = ProcessResult()
    score = score_lead(state)                       # authoritative; ignores anything the model says
    score["agent_suggestions"] = [a.strip()[:200] for a in findings.get("next_actions", []) if a.strip()][:6]
    res.extra["claims_override"] = {"lead": score}
    return res


_STANDARD_ACTIONS = (          # (substring of a missing_information item, action) - fixed, reviewable wording
    ("confirmed official website", "Run Company Intelligence to confirm the official website and organisation identity."),
    ("unverified profile fields", "Verify the remaining company profile fields from first-party pages ({detail})."),
    ("procurement page / supplier portal", "Run Procurement Intelligence to find the supplier-registration portal or procurement page."),
    ("procurement decision-maker", "Run Contact Intelligence to find publicly documented procurement decision-makers."),
    ("verification of crm-listed components", "Verify the CRM-listed components against public procurement pages."),
    ("components purchased", "Identify the components or categories the company says it buys (Procurement Intelligence)."),
)


def standard_next_actions(score: dict) -> list[str]:
    """Routine follow-ups derived from the deterministic score's missing information. No LLM involved."""
    out: list[str] = []
    for item in score.get("missing_information", []):
        low = item.lower()
        for key, action in _STANDARD_ACTIONS:
            if key in low:
                detail = item.split(":", 1)[1].strip() if ":" in item else ""
                out.append(action.format(detail=detail[:120])[:200])
                break
    if not out and score.get("band") in ("high", "medium"):
        out.append("Review the evidence links behind each factor before contacting the company.")
    return out[:6]


def lead_deterministic_findings(state: dict) -> dict:
    """The routine Lead result: authoritative score (via process_lead) + standard actions, no model call."""
    score = score_lead(state)
    return {"status": "partial" if score["missing_information"] else "satisfied",
            "next_actions": standard_next_actions(score), "unresolved": score["missing_information"][:20],
            "notes": "Computed deterministically; no model call.", "suggest_next": []}


def lead_task(state: dict) -> str:
    s = state["subject"]
    head = f"Company: {s['name']}\nUser objective: {state['objective']}\n"
    if (state.get("options") or {}).get("efficiency", {}).get("lead_precomputed"):
        sc = score_lead(state)
        factors = "; ".join(f"{f['factor']} {f['earned']}/{f['weight']} ({f['basis']})" for f in sc["factors"])
        return (head + f"Deterministic lead-priority result (computed from verified data; you cannot change it): "
                f"score {sc['score']}/{sc['max_score']}, band {sc['band']}, data sufficiency {sc['data_sufficiency']}.\n"
                f"Factors: {factors}\nMissing information: {'; '.join(sc['missing_information']) or 'none'}\n"
                "Submit with concrete next actions that would raise confidence or priority.")
    return (head +
            "Call compute_lead_score to obtain the explainable priority score from verified data. Read the factor "
            "breakdown and missing information, then submit with concrete next actions. You cannot change the score.")


def lead_tools(agents: frozenset) -> list[Tool]:
    return [Tool(name="compute_lead_score",
                 description="Compute the deterministic, explainable lead-priority score from verified data.",
                 args_model=ComputeLeadScoreArgs, handler=compute_lead_score, agents=agents, kind="compute")]


LEAD_AGENT = AgentSpec(
    name="lead", title="Lead Intelligence & Prioritization Agent",
    goal="Produce an explainable lead-priority assessment from verified information only.",
    role_prompt=("Do not browse and do not call any tool that is not offered in this request. The deterministic score "
                 "is either in your task text or obtained with compute_lead_score when that tool is offered. Then "
                 "summarise what is missing and what to do next, and finish with submit_lead_findings. "
                 "Never state a priority that contradicts the computed band."),
    findings_model=LeadFindings, finish_tool="submit_lead_findings",
    tool_names=("compute_lead_score", "submit_lead_findings"),
    process=process_lead, task=lead_task, deterministic=lead_deterministic_findings)

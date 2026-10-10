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
    role_prompt=("Do not browse. Use compute_lead_score, then summarise what is missing and what to do next. "
                 "Never state a priority that contradicts the computed band."),
    findings_model=LeadFindings, finish_tool="submit_lead_findings",
    tool_names=("compute_lead_score", "submit_lead_findings"),
    process=process_lead, task=lead_task)

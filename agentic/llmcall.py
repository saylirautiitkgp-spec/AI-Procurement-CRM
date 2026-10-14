"""Single choke point for LLM requests: budget check, usage accounting, trace."""

from __future__ import annotations

import time

from agentic.guards import budget as bud
from agentic.guards.budget import BudgetStop
from agentic.llm.base import LLMError, LLMResponse
from agentic.state import Emitter


def call_llm(deps, budgets: dict, *, system: str, messages: list, tools: list, emitter: Emitter,
             agent: str, force_tool: bool = False, count_agent: bool = True,
             max_output_tokens: int | None = None) -> LLMResponse:
    limits = deps.limits
    out_tokens = max_output_tokens or limits.max_output_tokens
    reason = bud.run_limit_reason(limits, budgets)
    if reason:
        raise BudgetStop(reason)
    from agentic.llm.pacing import estimate_request_tokens
    est = estimate_request_tokens(system, messages, tools, out_tokens)
    if budgets["input_tokens"] + budgets["output_tokens"] + est > limits.max_total_tokens_run:
        raise BudgetStop("max_tokens_reached")           # this request could take the run past its token cap
    budgets["llm_calls"] += 1
    if agent != "supervisor" and count_agent:
        budgets["agent_llm_calls"] += 1
    llm = deps.llm
    retries_before = getattr(llm, "retries", 0)
    started = time.monotonic()
    try:
        resp = llm.generate(system=system, messages=messages, tools=tools,
                            max_output_tokens=out_tokens, force_tool=force_tool)
    except LLMError as exc:
        budgets["llm_retries"] += getattr(llm, "retries", 0) - retries_before
        emitter.emit("llm_error", agent, kind=exc.kind, message=str(exc)[:200],
                     retries=getattr(llm, "retries", 0) - retries_before)
        raise
    budgets["llm_retries"] += getattr(llm, "retries", 0) - retries_before
    budgets["input_tokens"] += resp.usage.input_tokens
    budgets["output_tokens"] += resp.usage.output_tokens
    emitter.emit("llm_call", agent, provider=getattr(llm, "name", "?"), model=getattr(llm, "model", "?"),
                 latency_s=round(time.monotonic() - started, 3),
                 input_tokens=resp.usage.input_tokens, output_tokens=resp.usage.output_tokens,
                 tool_calls=[c.name for c in resp.tool_calls], has_text=bool(resp.text))
    return resp

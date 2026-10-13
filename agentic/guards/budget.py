"""Run budgets. The counters live in graph state (a plain dict) so they are
checkpointed; limits come from config. Python enforces them - the LLM cannot
talk its way past a cap."""

from __future__ import annotations

import time

from agentic.config import Limits


class BudgetStop(Exception):
    """Raised when a hard run limit is reached before an LLM call."""

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


def new_budgets(now: float | None = None) -> dict:
    return {
        "started_epoch": time.time() if now is None else now,
        "llm_calls": 0, "tool_calls": 0, "search_credits": 0,
        "input_tokens": 0, "output_tokens": 0, "llm_retries": 0,
        "agent_llm_calls": 0, "agent_tool_calls": 0, "agent_submit_attempts": 0,
        "call_signatures": {},          # signature -> count (repeat detection)
        "search_disabled": False, "search_disabled_reason": None,
    }


def reset_agent_counters(b: dict) -> None:
    b["agent_llm_calls"] = 0
    b["agent_tool_calls"] = 0
    b["agent_submit_attempts"] = 0
    b["call_signatures"] = {}


def run_limit_reason(limits: Limits, b: dict, now: float | None = None) -> str | None:
    now = time.time() if now is None else now
    if now - b["started_epoch"] > limits.max_wall_clock_s:
        return "wall_clock_exceeded"
    if b["llm_calls"] >= limits.max_llm_calls_run:
        return "max_llm_calls_reached"
    if b["tool_calls"] >= limits.max_tool_calls_run:
        return "max_tool_calls_reached"
    if b["input_tokens"] + b["output_tokens"] >= limits.max_total_tokens_run:
        return "max_tokens_reached"
    return None


def agent_limit_reason(limits: Limits, b: dict) -> str | None:
    if b["agent_llm_calls"] >= limits.max_llm_calls_agent:
        return "agent_llm_call_limit"
    if b["agent_tool_calls"] >= limits.max_tool_calls_agent:
        return "agent_tool_call_limit"
    return None


def credits_remaining_run(limits: Limits, b: dict) -> int:
    return max(0, min(limits.max_search_credits_run, limits.max_search_credits_company)
               - b["search_credits"])


def usage_summary(b: dict) -> dict:
    out = {k: b[k] for k in ("llm_calls", "tool_calls", "search_credits",
                             "input_tokens", "output_tokens", "llm_retries")}
    if b.get("db_selects") is not None:
        out["db_selects"] = b["db_selects"]
    return out

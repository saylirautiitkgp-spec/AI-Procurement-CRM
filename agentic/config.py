"""Runtime configuration: hard limits and provider selection.

Nothing here reads secrets or prints anything. Live calls are OFF unless
AGENTIC_ALLOW_LIVE=1 is set explicitly.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Mapping

# Search depths and their Tavily credit cost (docs.tavily.com, search endpoint).
SEARCH_DEPTH_CREDITS = {"basic": 1, "fast": 1, "ultra-fast": 1, "advanced": 2}
ALLOWED_SEARCH_DEPTHS = ("basic", "fast", "advanced")


@dataclass(frozen=True)
class Limits:
    # supervisor / run
    max_supervisor_steps: int = 8
    max_agent_runs_per_agent: int = 2
    max_llm_calls_run: int = 40
    max_tool_calls_run: int = 40
    max_total_tokens_run: int = 60_000
    max_wall_clock_s: float = 180.0
    # per agent
    max_llm_calls_agent: int = 10
    max_tool_calls_agent: int = 8
    max_submit_attempts: int = 2
    max_repeated_calls: int = 2
    # search credits (Tavily free plan: 1,000 credits / month)
    max_search_credits_run: int = 10
    max_search_credits_company: int = 6
    monthly_credit_budget: int = 300
    default_search_depth: str = "basic"
    max_search_results: int = 5
    # llm
    max_output_tokens: int = 1024
    llm_max_retries: int = 2
    llm_backoff_base_s: float = 1.0
    # fetch
    fetch_max_bytes: int = 500_000
    fetch_timeout_s: float = 10.0
    fetch_max_redirects: int = 3
    # what the model sees / what we keep
    tool_result_chars_for_llm: int = 6_000
    ledger_doc_chars: int = 20_000
    ledger_max_docs: int = 40
    max_concurrent_runs: int = 1


@dataclass(frozen=True)
class Efficiency:
    """Token/request-saving switches. None of them relaxes a safety check: evidence is still validated against
    the full ledger, budgets are still enforced, and the full conversation is still stored for audit.

    Defaults ON = mechanical, behaviour-neutral (the model sees the same facts, fewer repeated tokens).
    Defaults OFF = changes what the model is told or can call; enable only after a live comparison."""
    skip_forced_finish: bool = True       # no LLM call when no agent is eligible (finish is forced)
    crm_digest_in_task: bool = True       # CRM context lives in the task text; get_crm_context is not offered
    lead_precomputed: bool = True         # deterministic score is in the task text; compute tool not offered
    compact_history: bool = True          # drop superseded search snippets / old link lists from the PROMPT
    short_supervisor_text: bool = True    # one-line agent descriptions in the supervisor prompt
    compact_schemas: bool = False         # strip validation keywords from tool schemas sent to the model
    ledger_hint: bool = False             # list already-retrieved documents in later agents' tasks
    batch_hint: bool = False              # ask the model to request independent calls in one turn


BASELINE_EFFICIENCY = Efficiency(False, False, False, False, False, False, False, False)


@dataclass(frozen=True)
class Settings:
    limits: Limits = field(default_factory=Limits)
    provider: str = "fake"            # fake | gemini  (groq not implemented yet)
    model: str = "gemini-2.5-flash"
    allow_live: bool = False
    data_dir: str = "data"
    min_company_match: float = 0.75
    efficiency: Efficiency = field(default_factory=Efficiency)

    @property
    def db_path(self) -> str:
        return os.path.join(self.data_dir, "agentic.sqlite")

    @property
    def checkpoint_path(self) -> str:
        return os.path.join(self.data_dir, "checkpoints.sqlite")


def load_settings(env: Mapping[str, str] | None = None) -> Settings:
    env = os.environ if env is None else env

    def _int(name: str, default: int) -> int:
        try:
            return int(env.get(name, default))
        except (TypeError, ValueError):
            return default

    limits = Limits(
        max_search_credits_run=_int("AGENTIC_MAX_CREDITS_RUN", Limits.max_search_credits_run),
        monthly_credit_budget=_int("AGENTIC_MONTHLY_CREDITS", Limits.monthly_credit_budget),
        max_llm_calls_run=_int("AGENTIC_MAX_LLM_CALLS", Limits.max_llm_calls_run),
        max_tool_calls_run=_int("AGENTIC_MAX_TOOL_CALLS", Limits.max_tool_calls_run),
        max_wall_clock_s=float(_int("AGENTIC_WALL_CLOCK_S", int(Limits.max_wall_clock_s))),
        max_total_tokens_run=_int("AGENTIC_MAX_TOKENS", Limits.max_total_tokens_run),
        tool_result_chars_for_llm=_int("AGENTIC_TOOL_RESULT_CHARS", Limits.tool_result_chars_for_llm),
    )
    return Settings(
        limits=limits,
        provider=env.get("AGENTIC_LLM_PROVIDER", "fake"),
        model=env.get("AGENTIC_MODEL", "gemini-2.5-flash"),
        allow_live=env.get("AGENTIC_ALLOW_LIVE", "0") == "1",
        data_dir=env.get("AGENTIC_DATA_DIR", "data"),
    )

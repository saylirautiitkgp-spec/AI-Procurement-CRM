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
    llm_request_timeout_s: float = 60.0      # per-request cap; see effective_llm_timeout_s() for the wall-clock fit
    # fetch
    fetch_max_bytes: int = 500_000
    fetch_timeout_s: float = 10.0
    fetch_max_redirects: int = 3
    # what the model sees / what we keep
    tool_result_chars_for_llm: int = 6_000
    ledger_doc_chars: int = 20_000
    ledger_max_docs: int = 40
    max_concurrent_runs: int = 1
    max_db_selects_run: int | None = None   # per-run cap on Supabase SELECTs (None = not capped); AGENTIC_MAX_DB_SELECTS
    max_parallel_agents: int = 3            # worker threads for one parallel stage
    # Company evidence packet (Efficiency.prefetch): the system pre-fetches the official pages so the model starts
    # with documents instead of spending turns on discovery. Bounded; every document is also in the ledger.
    prefetch_max_pages: int = 3             # fetch_page calls the prefetch may make (plus its one search)
    prefetch_doc_chars: int = 2_500         # page text per document shown to the model
    prefetch_packet_chars: int = 7_500      # hard cap for the whole packet
    prefetch_min_chars: int = 300           # less page text than this = packet "insufficient" (fallback)
    # Company-only single-call extraction (run option `company_only`): ONE structured request carries every requested
    # company field, so the packet may be larger (bounded) and the answer needs room for many claims.
    company_only_output_tokens: int = 4096
    company_only_packet_chars: int = 12_000
    company_only_doc_chars: int = 4_000
    company_correction_requests: int = 0    # extra request allowed after rejected claims (0 = none); AGENTIC_COMPANY_CORRECTION
    max_requests_total: int | None = None   # hard cap on requests SENT in one run, retries included; AGENTIC_MAX_REQUESTS_TOTAL
    retry_503_only: bool = False            # retry only a 503 capacity error (never 429, timeouts, 4xx); AGENTIC_RETRY_503_ONLY


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
    prefetch: bool = False                # Company agent starts from a system-prepared evidence packet (AGENTIC_PREFETCH=1)
    deterministic_routing: bool = False   # clear single-capability objectives are routed by Python, not the LLM
    lazy_crm: bool = False                # read only the CRM tables the needed agents use
    lead_deterministic: bool = False      # routine lead score + standard next actions without an LLM call
    parallel_agents: bool = False         # independent agents of one stage run concurrently (field-request runs)


BASELINE_EFFICIENCY = Efficiency(False, False, False, False, False, False, False, False)


@dataclass(frozen=True)
class Settings:
    limits: Limits = field(default_factory=Limits)
    provider: str = "fake"            # fake | gemini  (groq not implemented yet)
    model: str = "gemini-3-flash-preview"      # the model the live smoke test verified
    allow_live: bool = False
    data_dir: str = "data"
    min_company_match: float = 0.75
    # Send a submission back for correction when only SOME claims were rejected (not just when all were).
    # Bounded by Limits.max_submit_attempts (2) and the run budgets; costs one extra LLM request per corrected
    # submission. Validation rules are untouched. Disable with AGENTIC_RETRY_PARTIAL=0.
    retry_partial_rejections: bool = True
    efficiency: Efficiency = field(default_factory=Efficiency)

    @property
    def db_path(self) -> str:
        return os.path.join(self.data_dir, "agentic.sqlite")

    @property
    def checkpoint_path(self) -> str:
        return os.path.join(self.data_dir, "checkpoints.sqlite")


def effective_llm_timeout_s(limits: "Limits") -> float:
    """Per-request LLM timeout that fits the wall-clock budget: even if every allowed attempt (1 + retries) hits
    the timeout, the total stays within max_wall_clock_s. Never below 5 s. Retry counts are not changed here."""
    fit = limits.max_wall_clock_s / (max(0, limits.llm_max_retries) + 1)
    return max(5.0, min(limits.llm_request_timeout_s, fit))


def _hybrid_efficiency(env: Mapping[str, str]) -> Efficiency:
    """AGENTIC_HYBRID=1 turns on the hybrid pieces together; each can be set individually (1/0) and wins over it."""
    hybrid = env.get("AGENTIC_HYBRID", "0") == "1"

    def flag(name: str) -> bool:
        v = env.get(name)
        return (v == "1") if v in ("0", "1") else hybrid
    return Efficiency(prefetch=flag("AGENTIC_PREFETCH"), deterministic_routing=flag("AGENTIC_DETERMINISTIC_ROUTING"),
                      lazy_crm=flag("AGENTIC_LAZY_CRM"), lead_deterministic=flag("AGENTIC_LEAD_DETERMINISTIC"),
                      parallel_agents=flag("AGENTIC_PARALLEL"))


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
        max_db_selects_run=(_int("AGENTIC_MAX_DB_SELECTS", 0) or None),
        llm_max_retries=max(0, _int("AGENTIC_LLM_RETRIES", Limits.llm_max_retries)),
        llm_request_timeout_s=float(max(5, _int("AGENTIC_LLM_TIMEOUT_S", int(Limits.llm_request_timeout_s)))),
        company_correction_requests=min(1, max(0, _int("AGENTIC_COMPANY_CORRECTION", Limits.company_correction_requests))),
        max_requests_total=(_int("AGENTIC_MAX_REQUESTS_TOTAL", 0) or None),
        retry_503_only=env.get("AGENTIC_RETRY_503_ONLY", "0") == "1",
        llm_backoff_base_s=float(max(1, _int("AGENTIC_LLM_BACKOFF_S", int(Limits.llm_backoff_base_s)))),
    )
    live = env.get("AGENTIC_MODE") == "live"
    return Settings(
        limits=limits,
        provider=env.get("AGENTIC_LLM_PROVIDER") or ("gemini" if live else "fake"),     # Gemini is the default live provider
        model=env.get("AGENTIC_MODEL") or "gemini-3-flash-preview",
        retry_partial_rejections=env.get("AGENTIC_RETRY_PARTIAL", "1") != "0",
        allow_live=env.get("AGENTIC_ALLOW_LIVE", "0") == "1",
        data_dir=env.get("AGENTIC_DATA_DIR", "data"),
        efficiency=_hybrid_efficiency(env),
    )

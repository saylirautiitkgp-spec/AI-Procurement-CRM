"""Budget preflight for a Company-Agent-only run. Runs BEFORE any Gemini request, Tavily search or Supabase query.

It builds the complete planned sequence (every permitted request, retry and follow-up, not just the first one), estimates
tokens, reads the local usage guard (read-only) and compares the plan with the ACTUAL remaining allowance and with every
per-run budget. If any limit would be exceeded, or the remaining quota cannot be established, the run is blocked:

    PREFLIGHT BLOCKED — no external requests sent

Nothing here opens a network connection or imports a provider/database client. It never writes to the usage log, never
resets or bypasses the daily guard, and sends no "probe" request to find out whether quota is available.

Quota knowledge, stated honestly: Google exposes no API for remaining free-tier quota. "Remaining" is therefore
    operator-supplied console limits (AGENTIC_QUOTA)  -  usage recorded by this runtime's guard in the rolling last 24 h
(the guard reserves capacity before every request and also records requests that failed after being sent). Use
`--console-remaining N` (the number AI Studio shows) to tighten it; usage made outside this runtime is invisible.
"""

from __future__ import annotations

import json
import math
import os
import sqlite3
import sys
import time
from dataclasses import asdict, dataclass, field
from typing import Mapping

from agentic import fields as fl
from agentic.agents.common import system_prompt
from agentic.agents.company import COMPANY_AGENT, CompanyFindings
from agentic.config import Limits, Settings, load_settings
from agentic.llm.pacing import CHARS_PER_TOKEN, ProviderQuota

BLOCK_LINE = "PREFLIGHT BLOCKED — no external requests sent"
DAY_S = 86400

# The company-only task text is built from the CRM row, so its exact size is unknown before the first Supabase read.
# These are deliberately generous upper bounds (characters).
TASK_BASE_CHARS = 1_600
TASK_CHARS_PER_FIELD = 120
RULE_CHARS = 900                  # graph.COMPANY_ONLY_RULE
HEADER_CHARS = 900                # prefetch header + JSON framing
CORRECTION_FEEDBACK_CHARS = 1_800
MIN_PACKET_CHARS = 300            # below this the packet is "insufficient" and NO model request is made
# Planned Supabase SELECTs of the RESEARCH phase (read-only). Company: find_candidates, companies row, sectors, profile,
# products. Procurement adds the components read. The maximum adds the pinned-record lookup, the visibility probe and one
# repeated row read.
SELECTS = {"company": (5, 7), "procurement": (1, 1), "contact": (1, 1), "supplier": (2, 2), "lead": (1, 1)}
SELECTS_PROCUREMENT_ONLY = (6, 8)
PREFETCH_SEARCHES = 1             # per agent: one basic Tavily search = 1 credit (0 when served from the 30-day cache)
SEARCH_CREDIT_COST = 1
PREFETCH_FETCHES = {"company": 4, "procurement": 3}      # fetch_page calls cost no credits
# Planned writes (only when --apply is used). Upper bounds the runner enforces by filtering what is applied.
MAX_PROCUREMENT_INSIGHTS = 10
WRITE_SELECTS_CAP = 40            # verification reads of the write phase (re-read before write, duplicate checks, read-back)
WRITE_TABLES = ("companies", "company_profile", "ai_insights")


@dataclass
class Plan:
    fields: list
    retries: int
    model: str
    correction_requests: int
    agents: list = field(default_factory=lambda: ["company"])
    calls_min: int = 0                 # model requests the graph can make (0 = every packet insufficient)
    calls_max: int = 1
    requests_min: int = 0              # including provider retries
    requests_max: int = 1
    input_tokens_min: int = 0
    input_tokens_max: int = 0
    output_tokens_min: int = 0
    output_tokens_max: int = 0
    pacing_estimate_max: int = 0       # what PacedLLM compares with the TPM limit (largest single request)
    tavily_credits_max: int = 0
    selects_min: int = 5
    selects_max: int = 7
    tool_calls_max: int = 5
    wall_clock_s: float = 0.0
    write_ops_max: int = 0             # database row operations if --apply is used (0 = research only)
    write_selects_cap: int = 0
    notes: list = field(default_factory=list)

    @property
    def tokens_max(self) -> int:
        return self.input_tokens_max + self.output_tokens_max


def _tok(chars: int) -> int:
    return math.ceil(chars / CHARS_PER_TOKEN)


def build_plan(settings: Settings, fields: list[str] | None = None, agents=("company",), apply: bool = False) -> Plan:
    """The complete planned sequence for one scoped run, derived from the same limits the graph enforces."""
    from agentic.agents import AGENT_SPECS
    lim: Limits = settings.limits
    agents = list(agents)
    keys = list(fields or fl.agent_fields(agents))
    correction = int(lim.company_correction_requests)
    per_agent_calls = 1 + correction
    attempts = 1 + max(0, lim.llm_max_retries)             # a retried request is another request against the quota
    out_cap = lim.company_only_output_tokens
    in_max = in_min = pacing_max = 0
    for name in agents:
        spec = AGENT_SPECS[name]
        system_chars = len(system_prompt(spec, {}))
        schema_chars = len(json.dumps(spec.findings_model.model_json_schema())) + 120
        task_chars = TASK_BASE_CHARS + TASK_CHARS_PER_FIELD * len(keys) + RULE_CHARS
        first_in = _tok(system_chars + schema_chars + task_chars + lim.company_only_packet_chars + HEADER_CHARS)
        agent_in = first_in
        pacing_max = max(pacing_max, first_in + out_cap)
        if correction:                                      # 2nd request carries the 1st answer and the validator feedback
            second_in = first_in + out_cap + _tok(CORRECTION_FEEDBACK_CHARS)
            agent_in += second_in
            pacing_max = max(pacing_max, second_in + out_cap)
        in_max += agent_in
        in_min += _tok(system_chars + schema_chars + TASK_BASE_CHARS + MIN_PACKET_CHARS + HEADER_CHARS)
    calls_max = per_agent_calls * len(agents)
    total_cap = lim.max_requests_total
    selects_min = sum(SELECTS[a][0] for a in agents if a in SELECTS)
    selects_max = sum(SELECTS[a][1] for a in agents if a in SELECTS)
    if "company" not in agents:              # procurement-only: it carries the lookup + profile + products + components reads
        selects_min, selects_max = SELECTS_PROCUREMENT_ONLY
    plan = Plan(fields=keys, retries=max(0, lim.llm_max_retries), model=settings.model, correction_requests=correction,
                agents=agents, calls_min=0, calls_max=calls_max, requests_min=0,
                requests_max=min(calls_max * attempts, total_cap) if total_cap else calls_max * attempts,
                input_tokens_min=in_min, input_tokens_max=in_max * attempts,
                output_tokens_min=150 * len(agents), output_tokens_max=out_cap * calls_max * attempts,
                pacing_estimate_max=pacing_max,
                tavily_credits_max=PREFETCH_SEARCHES * SEARCH_CREDIT_COST * (len(agents) + (0 if "company" in agents else 1)),
                selects_min=selects_min, selects_max=selects_max,
                tool_calls_max=sum(PREFETCH_SEARCHES + PREFETCH_FETCHES.get(a, 3) for a in agents)
                + (0 if "company" in agents else 2),            # pinned-identity re-verification: one search + one fetch
                wall_clock_s=lim.max_wall_clock_s)
    if apply:
        company_fields = len([k for k in keys if fl.BY_KEY[k].agent == "company"])
        url_fields = len([k for k in keys if k in ("supplier_portal", "procurement_page")])
        proc = MAX_PROCUREMENT_INSIGHTS if "procurement" in agents else 0
        # 1 companies update + 1 company_profile insert/update + one provenance ai_insight per written field + insights
        plan.write_ops_max = 2 + company_fields + url_fields + proc
        plan.write_selects_cap = WRITE_SELECTS_CAP
    plan.notes.append("calls_min is 0: when an evidence packet is insufficient the model is never asked for that agent")
    if total_cap:
        plan.notes.append(f"hard cap: {total_cap} requests SENT in total, retries included (enforced by the quota guard); "
                          f"retries allowed: {plan.retries}" + (" - only for a 503 capacity error" if lim.retry_503_only else "")
                          + "; the second agent runs only if the cap still has room")
    if lim.company_correction_requests == 0:
        plan.notes.append("no correction request and no follow-up search are planned or permitted")
    else:
        plan.notes.append(f"one correction request per agent is permitted (counted: {calls_max} calls maximum)")
    return plan


def caps_for(plan: Plan) -> dict:
    """Environment values that make the runtime enforce exactly this plan (with a small token margin)."""
    return {"AGENTIC_MAX_LLM_CALLS": str(plan.requests_max),
            "AGENTIC_MAX_TOKENS": str(int(plan.tokens_max * 1.15) + 500),
            "AGENTIC_MAX_CREDITS_RUN": str(plan.tavily_credits_max),
            "AGENTIC_MAX_TOOL_CALLS": str(plan.tool_calls_max + 3),
            "AGENTIC_MAX_DB_SELECTS": str(min(8, plan.selects_max + 1))}


# ----------------------------------------------------------------------------------------------- usage (read-only)
@dataclass
class Usage:
    known: bool
    requests_24h: int = 0
    tokens_24h: int = 0
    credits_month: int = 0
    why_unknown: str = ""


def read_usage(db_path: str, model_key: str, now: float) -> Usage:
    """Rolling-24 h usage recorded by the guard. Opens the SQLite file read-only: it is never created or modified."""
    if not os.path.isfile(db_path):
        return Usage(False, why_unknown=f"usage log not found at {db_path}")
    try:
        conn = sqlite3.connect(f"file:{os.path.abspath(db_path)}?mode=ro", uri=True)
        try:
            tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            reqs = toks = credits = 0
            if "llm_usage" in tables:
                row = conn.execute("SELECT COUNT(*), COALESCE(SUM(tokens),0) FROM llm_usage WHERE key=? AND ts>?",
                                   (model_key, now - DAY_S)).fetchone()
                reqs, toks = int(row[0]), int(row[1])
            if "credit_ledger" in tables:
                month = time.strftime("%Y-%m", time.gmtime(now))
                row = conn.execute("SELECT used FROM credit_ledger WHERE month=?", (month,)).fetchone()
                credits = int(row[0]) if row else 0
        finally:
            conn.close()
    except sqlite3.Error as exc:
        return Usage(False, why_unknown=f"usage log unreadable ({type(exc).__name__})")
    return Usage(True, reqs, toks, credits)


# ------------------------------------------------------------------------------------------------------- verdict
@dataclass
class Verdict:
    blocked: bool
    reasons: list = field(default_factory=list)
    warnings: list = field(default_factory=list)       # not blocking: e.g. the run will be paced
    remaining_requests: int | None = None        # under the planning cap
    remaining_requests_raw: int | None = None    # under the console limit itself
    remaining_tokens: int | None = None
    usage: Usage | None = None
    quota: ProviderQuota | None = None


def evaluate(plan: Plan, settings: Settings, quota: ProviderQuota | None, db_path: str, *, now: float | None = None,
             console_remaining: int | None = None, require_model: str | None = None,
             require_selects: int = 8, max_credits: int | None = None, max_requests: int | None = None) -> Verdict:
    now = time.time() if now is None else now
    lim = settings.limits
    v = Verdict(blocked=False, quota=quota)

    def block(reason: str) -> None:
        v.blocked = True
        v.reasons.append(reason)

    require_model = require_model or settings.model
    if settings.provider != "gemini" or settings.model != require_model:
        block(f"provider/model is {settings.provider}/{settings.model}; this run requires gemini/{require_model} "
              "(no automatic fallback exists)")
    if max_requests is not None and plan.requests_max > max_requests:
        block(f"planned maximum {plan.requests_max} Gemini requests exceeds the authorised {max_requests}; "
              f"minimum budget needed for this scope: {len(plan.agents)} (one per agent, no correction, retries 0)")
    if max_credits is not None and plan.tavily_credits_max > max_credits:
        block(f"planned {plan.tavily_credits_max} Tavily credits exceeds the authorised {max_credits}")
    # --- per-run limits ---------------------------------------------------------------------------------------
    if plan.requests_max > lim.max_llm_calls_run:
        block(f"planned maximum {plan.requests_max} model requests exceeds the per-run limit {lim.max_llm_calls_run}")
    if plan.tokens_max > lim.max_total_tokens_run:
        block(f"planned maximum ~{plan.tokens_max} tokens exceeds the per-run token limit {lim.max_total_tokens_run}")
    if plan.tavily_credits_max > min(lim.max_search_credits_run, lim.max_search_credits_company):
        block(f"planned {plan.tavily_credits_max} Tavily credit(s) exceeds the per-run credit limit "
              f"{min(lim.max_search_credits_run, lim.max_search_credits_company)}")
    if lim.max_db_selects_run is None:
        block("no per-run Supabase SELECT cap is set (AGENTIC_MAX_DB_SELECTS); the shared cap is required")
    else:
        if lim.max_db_selects_run > require_selects:
            block(f"SELECT cap {lim.max_db_selects_run} is above the required shared maximum of {require_selects}")
        if plan.selects_max > lim.max_db_selects_run:
            block(f"planned maximum {plan.selects_max} Supabase SELECTs exceeds the cap {lim.max_db_selects_run}")
    if plan.tool_calls_max > lim.max_tool_calls_run:
        block(f"planned {plan.tool_calls_max} tool calls exceeds the per-run limit {lim.max_tool_calls_run}")
    # --- provider quota ---------------------------------------------------------------------------------------
    if quota is None:
        block("provider quota is UNKNOWN: AGENTIC_QUOTA (rpm/tpm/rpd from your console) is not set or unparsable")
    else:
        usage = read_usage(db_path, f"gemini:{settings.model}", now)
        v.usage = usage
        if not usage.known:
            block(f"remaining Gemini allowance is UNKNOWN: {usage.why_unknown}")
        else:
            cap = int(quota.rpd * quota.headroom)
            v.remaining_requests = cap - usage.requests_24h
            v.remaining_requests_raw = quota.rpd - usage.requests_24h
            if quota.tpd < 10 ** 11:
                v.remaining_tokens = int(quota.tpd * quota.headroom) - usage.tokens_24h
            if console_remaining is not None:
                v.remaining_requests = min(v.remaining_requests, int(console_remaining))
                v.remaining_requests_raw = min(v.remaining_requests_raw, int(console_remaining))
            if plan.requests_max > v.remaining_requests:
                block(f"planned maximum {plan.requests_max} Gemini request(s) exceeds the remaining daily allowance "
                      f"{max(v.remaining_requests, 0)} (used {usage.requests_24h} in the last 24 h; limit {quota.rpd}, "
                      f"planning cap {cap})")
            if v.remaining_tokens is not None and plan.tokens_max > v.remaining_tokens:
                block(f"planned ~{plan.tokens_max} tokens exceeds the remaining daily token allowance {v.remaining_tokens}")
            if plan.pacing_estimate_max > quota.tpm:
                block(f"largest planned request (~{plan.pacing_estimate_max} tokens) exceeds the per-minute limit {quota.tpm}")
            per_min = max(1, int(quota.rpm * quota.headroom))
            if plan.requests_max > per_min:
                # The paced client WAITS for the per-minute window instead of refusing, so a burst is slow, not forbidden:
                # it only blocks when the unavoidable waiting alone would exceed the run's wall-clock limit.
                wait_s = ((plan.requests_max - 1) // per_min) * 60
                if wait_s >= lim.max_wall_clock_s:
                    block(f"planned {plan.requests_max} requests need about {wait_s}s of pacing at {per_min} requests/minute, "
                          f"more than the {lim.max_wall_clock_s:.0f}s run limit")
                else:
                    v.warnings.append(f"{plan.requests_max} requests at {per_min}/minute: the run is paced and will wait "
                                      f"about {wait_s}s in total")
            left = lim.monthly_credit_budget - usage.credits_month
            if plan.tavily_credits_max > left:
                block(f"planned {plan.tavily_credits_max} Tavily credit(s) exceeds the remaining monthly budget {left}")
    return v


def render(plan: Plan, v: Verdict, settings: Settings) -> str:
    lim = settings.limits
    q = v.quota
    lines = [f"PREFLIGHT — agents: {', '.join(plan.agents)}", f"  model: {settings.provider}/{settings.model}   retries: {plan.retries}"
             f"   wall clock: {plan.wall_clock_s:.0f}s   per-request timeout cap: {lim.llm_request_timeout_s:.0f}s",
             f"  fields ({len(plan.fields)}): {', '.join(plan.fields)}",
             f"  Gemini requests planned: min {plan.requests_min}, max {plan.requests_max} "
             f"(max = {plan.calls_max} model call(s) x {1 + plan.retries} attempt(s); correction requests permitted: "
             f"{plan.correction_requests})",
             f"  estimated tokens: input {plan.input_tokens_min}..{plan.input_tokens_max}, "
             f"output {plan.output_tokens_min}..{plan.output_tokens_max}; largest single request ~{plan.pacing_estimate_max}",
             f"  per-run limits: {lim.max_llm_calls_run} requests, {lim.max_total_tokens_run} tokens, "
             f"{min(lim.max_search_credits_run, lim.max_search_credits_company)} Tavily credits, "
             f"{lim.max_tool_calls_run} tool calls, {lim.max_db_selects_run} SELECTs"]
    if q is not None and v.usage is not None and v.usage.known:
        lines.append(f"  Gemini daily allowance: console limit {q.rpd}/day, planning cap {int(q.rpd * q.headroom)}; used in the "
                     f"last 24 h (usage guard): {v.usage.requests_24h}; remaining under the cap: {max(v.remaining_requests, 0)} "
                     f"(raw {max(v.remaining_requests_raw, 0)})")
    else:
        lines.append("  Gemini daily allowance: UNKNOWN")
    lines.append(f"  Tavily credits: max planned {plan.tavily_credits_max} (0 if served from cache); month used so far: "
                 f"{v.usage.credits_month if v.usage and v.usage.known else 'unknown'} of {lim.monthly_credit_budget}")
    lines.append(f"  Supabase SELECTs (research): min {plan.selects_min}, max planned {plan.selects_max}; hard cap {lim.max_db_selects_run}")
    if plan.write_ops_max:
        lines.append(f"  proposed writes (--apply): at most {plan.write_ops_max} row operations on company 11 only, tables "
                     f"{', '.join(WRITE_TABLES)}; additive only (fill empty fields / insert sourced insights); "
                     f"write-phase verification SELECTs capped at {plan.write_selects_cap}")
    else:
        lines.append("  proposed writes: none (research only)")
    for n in plan.notes:
        lines.append(f"  note: {n}")
    lines.extend(f"  note: {w}" for w in v.warnings)
    if v.blocked:
        lines.append(BLOCK_LINE)
        lines.extend(f"  exceeded/unknown: {r}" for r in v.reasons)
    else:
        lines.append("PREFLIGHT OK — the complete planned sequence fits every budget")
    return "\n".join(lines)


def use_utf8_stdout() -> None:
    """The block line contains an em dash; never let a legacy console codepage turn printing into a crash."""
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass


def settings_from_env(env: Mapping[str, str]) -> Settings:
    return load_settings(env)


def quota_from_env(env: Mapping[str, str]) -> ProviderQuota | None:
    raw = env.get("AGENTIC_QUOTA")
    if not raw:
        return None
    try:
        return ProviderQuota.parse(raw, require_tpd=False)
    except ValueError:
        return None


def main(argv: list[str] | None = None, env: Mapping[str, str] | None = None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="Budget preflight (no external requests are made).")
    ap.add_argument("--fields", nargs="*", default=None)
    ap.add_argument("--console-remaining", type=int, default=None,
                    help="requests/day your AI Studio console shows as remaining (tightens the local figure)")
    args = ap.parse_args(argv)
    use_utf8_stdout()
    env = os.environ if env is None else env
    settings = settings_from_env(env)
    keys, unknown = fl.resolve_fields(args.fields) if args.fields else (fl.company_agent_fields(), [])
    if unknown:
        print(f"unknown fields: {unknown}")
        return 2
    plan = build_plan(settings, keys)
    v = evaluate(plan, settings, quota_from_env(env), settings.db_path, console_remaining=args.console_remaining)
    print(render(plan, v, settings))
    return 3 if v.blocked else 0


if __name__ == "__main__":
    sys.exit(main())

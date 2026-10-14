"""One controlled live enrichment of ONE existing company (research -> change set -> optional narrow write phase).

    # preflight only - no provider, search or database request is made:
    python -m agentic.company_live --company "AIRBUS India" --company-id 11 --agents company procurement \\
        --model gemini-2.5-flash --corrections 1 --max-requests 4 --max-credits 2 --apply --preflight-only
    # the live run (needs explicit approval for THIS run; the write phase needs --apply as well):
    AGENTIC_COMPANY_LIVE_APPROVED=1 python -m agentic.company_live ... --apply --load-dotenv

Order of events: settings -> complete-plan preflight (local files only) -> [blocked: print the block line, exit] ->
approval check -> only then .env is loaded (inside this process; never printed) and clients are built -> ONE research run
(scoped agents, writes impossible: no commit repository exists) -> [--apply] the narrow write phase of
agentic/apply_validated.py on a separate client with its own SELECT cap -> read-back -> report.
No retry, no rerun, no provider fallback, no delete/merge/migration.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

from agentic import fields as fl
from agentic import preflight as pf
from agentic.preflight import BLOCK_LINE

DEFAULT_MODEL = "gemini-3-flash-preview"
SCHEMA_FILE = "agentic/schema_introspection.json"
APPROVAL_ENV = "AGENTIC_COMPANY_LIVE_APPROVED"


def base_env(quota: str | None, data_dir: str, model: str = DEFAULT_MODEL, corrections: int = 0,
             thinking_budget: int | None = None, retries: int = 0, retry_503_only: bool = False,
             total_requests_cap: int | None = None, backoff_s: int = 1) -> dict:
    """Run configuration. Contains no secrets; provider keys are read later from the process environment."""
    env = {"AGENTIC_MODE": "live", "AGENTIC_ALLOW_LIVE": "1", "AGENTIC_ENABLE_WRITES": "0",
           "AGENTIC_LLM_PROVIDER": "gemini", "AGENTIC_MODEL": model, "AGENTIC_LLM_RETRIES": "0",
           "AGENTIC_WALL_CLOCK_S": "420", "AGENTIC_LLM_TIMEOUT_S": "60", "AGENTIC_HYBRID": "1",
           "AGENTIC_PARALLEL": "0", "AGENTIC_SCHEMA_FILE": SCHEMA_FILE, "AGENTIC_DATA_DIR": data_dir,
           "AGENTIC_COMPANY_CORRECTION": str(int(corrections))}
    env["AGENTIC_LLM_RETRIES"] = str(int(retries))
    if retry_503_only:
        env["AGENTIC_RETRY_503_ONLY"] = "1"
        env["AGENTIC_LLM_BACKOFF_S"] = str(int(backoff_s))
    if total_requests_cap:
        env["AGENTIC_MAX_REQUESTS_TOTAL"] = str(int(total_requests_cap))
    if thinking_budget is not None and not model.startswith("gemini-3"):
        env["AGENTIC_THINKING_BUDGET"] = str(int(thinking_budget))
    if quota:
        env["AGENTIC_QUOTA"] = quota
    return env


def item_view(it: dict) -> dict:
    return {"field": it["field"], "target": it.get("target"), "action": it.get("action"),
            "old_value": it.get("old_value"), "proposed_value": it.get("proposed_value"),
            "confidence": it.get("confidence"), "validation": it.get("validation_status"),
            "evidence": [{"url": e["url"], "quote": e["snippet"]} for e in it.get("evidence", [])]}


def summarise(result: dict, plan: pf.Plan, usage_before, usage_after, fingerprint: str, write: dict | None) -> dict:
    b = result.get("budgets") or {}
    report = result.get("report") or {}
    cs = report.get("change_set") or {}
    rejected = [r for rows in (result.get("rejected") or {}).values() for r in rows]
    return {
        "run_id": result.get("run_id"), "status": result.get("status"), "stop_reason": result.get("stop_reason"),
        "status_reasons": result.get("status_reasons"),
        "agents_run": sorted(a for a, n in (result.get("agent_runs") or {}).items() if n),
        "agent_results": {a: {k: r.get(k) for k in ("status", "end_reason", "accepted", "rejected")}
                          for a, r in (result.get("agent_results") or {}).items()},
        "supervisor_model_calls": sum(1 for e in result.get("trace", [])
                                      if e["type"] == "llm_call" and e.get("agent") == "supervisor"),
        "identity_verified": bool(((result.get("subject") or {}).get("identity") or {}).get("verified")),
        "planned": {"agents": plan.agents, "requests_min": plan.requests_min, "requests_max": plan.requests_max,
                    "tokens_in_max": plan.input_tokens_max, "tokens_out_max": plan.output_tokens_max,
                    "tavily_credits_max": plan.tavily_credits_max, "selects_max": plan.selects_max,
                    "write_ops_max": plan.write_ops_max},
        "actual": {"gemini_requests": b.get("llm_calls"), "llm_retries": b.get("llm_retries"),
                   "input_tokens": b.get("input_tokens"), "output_tokens": b.get("output_tokens"),
                   "tavily_credits": b.get("search_credits"), "tool_calls": b.get("tool_calls"),
                   "supabase_selects_research": b.get("db_selects"),
                   "guard_requests_24h_before": usage_before.requests_24h if usage_before else None,
                   "guard_requests_24h_after": usage_after.requests_24h if usage_after else None},
        "change_set_items": [item_view(i) for i in cs.get("items", [])],
        "unresolved": cs.get("unresolved", []), "skipped_in_change_set": cs.get("skipped", []),
        "rejected_claims": [{"field": r.get("field") or r.get("kind"), "value": r.get("value"),
                             "reasons": r.get("reasons")} for r in rejected],
        "change_set_hash": cs.get("change_set_hash"), "errors": result.get("errors", []),
        "schema_fingerprint": fingerprint, "write_phase": write,
        "research_phase_writes": 0,
    }


def write_phase(result: dict, deps, company_id: int, authoriser: str) -> dict:
    """Apply the validated additive items (agentic/apply_validated.py) through a counted, allow-listed client."""
    from supabase import create_client
    from agentic.apply_validated import guarded_write_phase
    from agentic.commit import SupabaseCommitRepo
    from agentic.guards.select_budget import CountedClient, SelectBudget
    budget = SelectBudget(pf.WRITE_SELECTS_CAP)
    client = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_KEY"])
    repo = SupabaseCommitRepo(CountedClient(client, budget))
    return guarded_write_phase(repo, deps.store, deps.schema, result, company_id, authoriser, budget)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Controlled live enrichment run (see module docstring).")
    ap.add_argument("--company", required=True)
    ap.add_argument("--company-id", type=int, required=True)
    ap.add_argument("--agents", nargs="+", default=["company"], choices=["company", "procurement"])
    ap.add_argument("--pinned-identity-website", default=None,
                    help="procurement-only: URL whose domain identity is RE-VERIFIED in the run (cache first); no Company agent")
    ap.add_argument("--fields", nargs="*", default=None, help="catalog field keys (default: every field of the agents)")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--corrections", type=int, default=0, choices=[0, 1])
    ap.add_argument("--thinking-budget", type=int, default=1024, help="Gemini 2.5 only; Gemini 3 uses thinking_level")
    ap.add_argument("--retries", type=int, default=0, choices=[0, 1], help="provider retries per call (counted as requests)")
    ap.add_argument("--retry-503-only", action="store_true", help="the only retry allowed is for a 503 capacity error")
    ap.add_argument("--backoff-s", type=int, default=20, help="wait before the single 503 retry")
    ap.add_argument("--total-requests-cap", type=int, default=None, help="hard cap on requests SENT in the run, retries included")
    ap.add_argument("--max-requests", type=int, default=None)
    ap.add_argument("--max-credits", type=int, default=None)
    ap.add_argument("--quota", default=os.environ.get("AGENTIC_QUOTA"), help='e.g. "rpm=5,tpm=250000,rpd=20"')
    ap.add_argument("--console-remaining", type=int, default=None)
    ap.add_argument("--apply", action="store_true", help="after the run, apply the validated additive items (narrow write phase)")
    ap.add_argument("--authoriser", default="user-authorised single AIRBUS India enrichment 2026-10-13")
    ap.add_argument("--preflight-only", action="store_true")
    ap.add_argument("--load-dotenv", action="store_true")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    pf.use_utf8_stdout()

    keys, unknown = fl.resolve_fields(args.fields) if args.fields else (fl.agent_fields(args.agents), [])
    foreign = [k for k in keys if fl.BY_KEY[k].agent not in args.agents]
    pinned_ok = args.agents == ["procurement"] and bool(args.pinned_identity_website)
    if ("company" not in args.agents and not pinned_ok) or unknown or foreign:
        print(f"refused: company agent required (or procurement-only with --pinned-identity-website) / "
              f"unknown fields {unknown} / fields outside the scope {foreign}")
        return 2

    # ---- complete-plan preflight (local files only; nothing external has been touched) ---------------------------
    env = base_env(args.quota, args.data_dir, args.model, args.corrections, args.thinking_budget, args.retries,
                   args.retry_503_only, args.total_requests_cap, args.backoff_s)
    settings = pf.settings_from_env(env)
    plan = pf.build_plan(settings, keys, args.agents, apply=args.apply)
    env.update(pf.caps_for(plan))                       # the runtime enforces exactly what was planned
    settings = pf.settings_from_env(env)
    plan = pf.build_plan(settings, keys, args.agents, apply=args.apply)
    verdict = pf.evaluate(plan, settings, pf.quota_from_env(env), settings.db_path, console_remaining=args.console_remaining,
                          require_model=args.model, max_credits=args.max_credits, max_requests=args.max_requests)
    print(pf.render(plan, verdict, settings))
    if verdict.blocked:
        return 3
    if args.preflight_only:
        return 0
    if os.environ.get(APPROVAL_ENV) != "1":
        print(f"not started: set {APPROVAL_ENV}=1 to approve this single live run")
        return 4

    # ---- the run ------------------------------------------------------------------------------------------------
    if args.load_dotenv:
        from dotenv import load_dotenv
        load_dotenv()                                   # inside this process only; values are never printed
    from agentic.service import build_live_service
    svc = build_live_service(env)
    deps = svc.runner.deps
    if deps.commit_repo is not None:                    # belt and braces: the research phase must not be able to write
        print("refused: a commit repository is configured for the research phase")
        return 5
    fingerprint = str(getattr(getattr(deps, "schema", None), "fingerprint", "") or "")
    key = f"gemini:{args.model}"
    before = pf.read_usage(settings.db_path, key, time.time())
    result = svc.runner.start(args.company, run_id=None, propose_changes=False, fields=keys, refresh=False,
                              expected_company_id=args.company_id, scope_agents=args.agents,
                              pinned_identity_website=args.pinned_identity_website)
    after = pf.read_usage(settings.db_path, key, time.time())
    write = None
    if args.apply:
        if result.get("status") == "failed" or not (result.get("report") or {}).get("change_set"):
            write = {"blocked": "run_failed_or_no_change_set", "writes": 0}
        else:
            write = write_phase(result, deps, args.company_id, args.authoriser)
    summary = summarise(result, plan, before, after, fingerprint, write)
    out = args.out or os.path.join(args.data_dir, f"company_live_{result.get('run_id')}.json")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, default=str)
    print(json.dumps(summary, indent=2, default=str))
    print(f"(saved to {out})")
    return 0


if __name__ == "__main__":
    sys.exit(main())

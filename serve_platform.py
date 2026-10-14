"""Start the AI CRM Procurement Platform on THIS machine.

    python serve_platform.py                       # CRM from Supabase if configured (else the bundled demo set); research = simulation
    python serve_platform.py --data demo           # force the bundled 20-company demo data (no database needed)
    python serve_platform.py --research live --model gemini-3.8-flash --quota "rpm=5,tpm=250000,rpd=20"
    python serve_platform.py --research live ... --enable-writes     # also allow the guarded 'apply to database' button
    python serve_platform.py --research off        # browse only

Then open http://127.0.0.1:8000/ui/ . The server listens on 127.0.0.1 only. API keys are read from .env inside this process
and are never printed, logged or sent to the browser.

Research modes
  simulation  scripted model + synthetic pages: every screen works, NO Gemini/Tavily quota is used, nothing can be written.
  live        real Gemini + Tavily + Supabase, every run is preflight-checked against the quota you give with --quota
              (copy it from YOUR AI Studio console; nothing is assumed) and the local usage guard. Writes stay OFF unless
              --enable-writes is given; even then each write needs the access token, a named approver and a confirmation.
  off         browsing only.
"""

from __future__ import annotations

import argparse
import os
import secrets
import sys
import webbrowser

PLATFORM_CAPS = {"AGENTIC_MAX_LLM_CALLS": "8", "AGENTIC_MAX_REQUESTS_TOTAL": "8", "AGENTIC_MAX_TOKENS": "160000",
                 "AGENTIC_MAX_CREDITS_RUN": "4", "AGENTIC_MAX_TOOL_CALLS": "40", "AGENTIC_MAX_DB_SELECTS": "14",
                 "AGENTIC_WALL_CLOCK_S": "600"}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="AI CRM Procurement Platform (local web app).")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--data", choices=["auto", "supabase", "demo"], default="auto")
    ap.add_argument("--research", choices=["off", "simulation", "live"], default="simulation")
    ap.add_argument("--model", default="gemini-3.8-flash")
    ap.add_argument("--quota", default=os.environ.get("AGENTIC_QUOTA"), help='your console limits, e.g. "rpm=5,tpm=250000,rpd=20"')
    ap.add_argument("--console-remaining", type=int, default=None, help="requests/day AI Studio shows as remaining")
    ap.add_argument("--thinking-budget", type=int, default=1024)
    ap.add_argument("--enable-writes", action="store_true", help="allow the guarded 'apply to database' action (live only)")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--open", action="store_true", help="open the browser")
    args = ap.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    if os.path.exists(".env"):
        from dotenv import load_dotenv
        load_dotenv(".env")                                  # inside this process only; values are never printed
    have_db = bool(os.environ.get("SUPABASE_URL") and os.environ.get("SUPABASE_KEY"))
    use_db = args.data == "supabase" or (args.data == "auto" and have_db)
    if use_db and not have_db:
        print("SUPABASE_URL / SUPABASE_KEY are not set; cannot use --data supabase")
        return 2

    from agentic.crm_read import DemoCrm, SupabaseCrm
    from api import platform_routes as pr
    client = None
    if use_db:
        from supabase import create_client
        client = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_KEY"])
        crm = SupabaseCrm(client)
    else:
        crm = DemoCrm()
    notes = []
    state: dict = {"crm": crm, "research_mode": args.research, "notes": notes}

    from agentic.service import configure_service
    if args.research == "simulation":
        from agentic.demo_service import build_demo_service, demo_repo_from
        if use_db:
            from agentic.guards.select_budget import GuardedReadClient
            from agentic.repo import SupabaseReadRepo
            repo = SupabaseReadRepo(GuardedReadClient(client))
            notes.append("Simulation over your real CRM records: the web pages and the model are synthetic.")
        else:
            repo = demo_repo_from(crm)
        configure_service(build_demo_service(crm, repo, args.data_dir))
    elif args.research == "live":
        if not use_db:
            print("--research live needs the Supabase CRM (set SUPABASE_URL / SUPABASE_KEY)")
            return 2
        missing = [k for k in ("GEMINI_API_KEY", "TAVILY_API_KEY") if not os.environ.get(k)]
        if missing:
            print("missing in .env: " + ", ".join(missing))
            return 2
        if not args.quota:
            print("--research live needs --quota \"rpm=..,tpm=..,rpd=..\" copied from YOUR AI Studio console (nothing is assumed)")
            return 2
        from agentic.company_live import base_env
        from agentic.llm.pacing import ProviderQuota
        from agentic.service import ServiceNotConfigured, build_live_service
        env = base_env(args.quota, args.data_dir, args.model, 0, args.thinking_budget, 1, True, None, 20)
        env.update(PLATFORM_CAPS)
        try:
            quota = ProviderQuota.parse(args.quota, require_tpd=False)
            svc = build_live_service(env)
        except (ValueError, ServiceNotConfigured) as exc:
            print(f"cannot start live research: {exc}")
            return 2
        configure_service(svc)
        state.update(settings=svc.settings, quota=quota, console_remaining=args.console_remaining)
        if args.enable_writes:
            from agentic.commit import SupabaseCommitRepo
            from agentic.guards.select_budget import CountedClient
            state.update(writes_enabled=True,
                         commit_factory=lambda budget: SupabaseCommitRepo(CountedClient(client, budget)))
            if not os.environ.get("AGENTIC_API_TOKEN"):
                os.environ["AGENTIC_API_TOKEN"] = secrets.token_urlsafe(24)
                print(f"\n  ACCESS TOKEN (needed in the web app's Settings for every request, because writes are enabled):\n"
                      f"  {os.environ['AGENTIC_API_TOKEN']}\n")
    pr.configure(**state)

    from api.platform_app import app
    import uvicorn
    url = f"http://{args.host}:{args.port}/ui/"
    print(f"AI CRM Procurement Platform\n  data source : {crm.kind}\n  research    : {args.research}"
          f"{' (' + args.model + ')' if args.research == 'live' else ''}\n  writes      : "
          f"{'ENABLED (guarded)' if state.get('writes_enabled') else 'disabled'}\n  open        : {url}\n")
    if args.open:
        webbrowser.open(url)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    return 0


if __name__ == "__main__":
    sys.exit(main())

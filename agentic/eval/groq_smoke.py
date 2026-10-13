"""Smallest useful Groq smoke test.   *** PROPOSED - NOT RUN - AWAITING YOUR EXPLICIT APPROVAL ***

Same four steps as the Gemini smoke test (provider-neutral `run_smoke`), synthetic inputs and mock tools only:
  1 forced supervisor routing, 2 initial tool selection, 3 continuation after a mock search result,
  4 forced structured submit validated against our schema + the evidence validator.
No Tavily, no Supabase, no real company/contact data, no retries, never more than 4 LLM requests.

It refuses to start unless ALL of these hold:
    AGENTIC_GROQ_SMOKE_APPROVED=1
    GROQ_API_KEY in the PROCESS ENVIRONMENT        (.env is read only if you pass --load-dotenv yourself)
    --model openai/gpt-oss-120b | openai/gpt-oss-20b   (the only approved candidates)
    --quota "rpm=..,tpm=..,rpd=..,tpd=.."          (YOUR console values for that model - nothing is assumed)
    --i-have-confirmed-free-plan                   (you confirmed: Free plan, no payment method, zero billing)

Safety behaviour: client-side pacing keeps requests under 80% of the per-minute limits and refuses locally when the
rolling 24 h budget would be exceeded; the first provider error of any kind (quota, rate limit, billing, 4xx/5xx)
stops the run - there is no retry and no alternative-model attempt. `service_tier` is never sent. Reasoning effort
is "low"; reasoning text is not requested.

    python -m agentic.eval.groq_smoke --dry-run --model openai/gpt-oss-120b
    AGENTIC_GROQ_SMOKE_APPROVED=1 python -m agentic.eval.groq_smoke --model openai/gpt-oss-120b \
        --quota "rpm=30,tpm=8000,rpd=1000,tpd=200000" --i-have-confirmed-free-plan data/groq_smoke.json
"""

from __future__ import annotations

import json
import os
import sys

from agentic.eval.gemini_smoke import CountingLLM, MAX_OUTPUT_TOKENS, MAX_REQUESTS, SmokeGuardError, _arg, run_smoke
from agentic.llm.base import LLMError
from agentic.llm.groq import APPROVED_MODELS, GroqClient
from agentic.llm.pacing import PacedLLM, ProviderQuota
from agentic.persistence.store import Store
from agentic.security import redact


def check_guards(env, argv) -> ProviderQuota:
    if env.get("AGENTIC_GROQ_SMOKE_APPROVED") != "1":
        raise SmokeGuardError("AGENTIC_GROQ_SMOKE_APPROVED=1 is required (live Groq calls are not approved by default)")
    model = _arg(argv, "--model")
    if model not in APPROVED_MODELS:
        raise SmokeGuardError(f"--model must be one of {', '.join(APPROVED_MODELS)}")
    if "--i-have-confirmed-free-plan" not in argv:
        raise SmokeGuardError("pass --i-have-confirmed-free-plan after confirming Free Plan, no payment method, no billing")
    if not _arg(argv, "--quota"):
        raise SmokeGuardError('--quota "rpm=..,tpm=..,rpd=..,tpd=.." is required (your console values; none are assumed)')
    try:
        quota = ProviderQuota.parse(_arg(argv, "--quota"))
    except ValueError as exc:
        raise SmokeGuardError(f"--quota invalid: {exc}")
    if not env.get("GROQ_API_KEY"):
        raise SmokeGuardError("GROQ_API_KEY is not configured in the process environment")
    return quota


def run_groq_smoke(client, quota: ProviderQuota, store: Store | None = None, *, clock=None, sleep=None) -> dict:
    kw = {k: v for k, v in (("clock", clock), ("sleep", sleep)) if v is not None}
    paced = PacedLLM(client, quota, store, **kw)
    llm = CountingLLM(paced, cap=MAX_REQUESTS)
    try:
        out = run_smoke(llm)
    except LLMError as exc:
        out = {"requests_made": llm.n, "aborted": f"{exc.kind}: {redact(str(exc))[:300]}",
               "stopped_on_first_error": True, "per_request": llm.log}
    out["pacing"] = {"seconds_waited": round(paced.waited_s, 1), "local_refusals": paced.local_refusals}
    out["model"] = client.model
    return out


def main(argv: list[str]) -> None:  # pragma: no cover - live, user-approved only
    if "--dry-run" in argv:
        print(__doc__)
        print("Dry run: no request made. Model:", _arg(argv, "--model") or "(none given)")
        return
    if "--load-dotenv" in argv:                      # opt-in only
        from dotenv import load_dotenv
        load_dotenv()
    quota = check_guards(os.environ, argv)
    client = GroqClient(model=_arg(argv, "--model"), reasoning_effort="low")
    out = run_groq_smoke(client, quota, Store(os.path.join("data", "groq_smoke_usage.sqlite")))
    path = next((a for a in argv if a.endswith(".json")), "data/groq_smoke.json")
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    json.dump(out, open(path, "w", encoding="utf-8"), indent=2)
    print(json.dumps({k: out.get(k) for k in ("model", "requests_made", "integration_passed", "evidence_quality_passed", "aborted")}, indent=2), "->", path)


if __name__ == "__main__":  # pragma: no cover
    main(sys.argv[1:])

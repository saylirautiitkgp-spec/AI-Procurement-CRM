"""Smallest possible Gemini schema / tool-calling smoke test.   *** NOT RUN - AWAITING USER APPROVAL ***

What it answers (4 requests, no Tavily, no database, no real company data):
  1. Does the configured model accept our flattened tool schemas (no HTTP 400)?
  2. Does it return a parseable `route` function call when forced?
  3. Given a task, does it choose a sensible first tool for the Company agent?
  4. After seeing (synthetic) search results, does it continue sensibly?
  5. Does the forced `submit_company_findings` call validate against our schema, and do its quoted snippets
     pass our evidence validator on a synthetic page?
Plus: latency, input/output (incl. thinking) tokens per request.

Hard limits: at most MAX_REQUESTS (4) `generate_content` calls, no retries (a failed request is NOT repeated and
the run stops), max_output_tokens=1024 per call,
inputs are synthetic text only (reserved .example domains), nothing is written anywhere except the JSON result
file you name. It refuses to start unless ALL of these hold:
    AGENTIC_SMOKE_APPROVED=1   GEMINI_API_KEY in the PROCESS ENVIRONMENT   --model <exact id>
    --i-have-checked-my-free-quota   (and not --dry-run)
The script never reads .env unless you pass --load-dotenv yourself; it does not list models and does not guess a
model id; it never prints the key.

Estimated usage: ~4 requests, roughly 5-8k input tokens and <2k output tokens in total - i.e. a tiny fraction
of any free-tier daily allowance. Check your own AI Studio quota page before running.

Usage:
    python -m agentic.eval.gemini_smoke --dry-run                      # prints the plan, makes no request
    AGENTIC_SMOKE_APPROVED=1 python -m agentic.eval.gemini_smoke --model gemini-3-flash-preview \
        --i-have-checked-my-free-quota out.json
"""

from __future__ import annotations

import json
import os
import sys
import time

from agentic import evidence as ev
from agentic.agents import AGENT_SPECS
from agentic.agents.common import system_prompt
from agentic.agents.company import CompanyFindings, process_company
from agentic.eval.quality import strict_assessment
from agentic.eval.synthetic import make_world
from agentic.llm.base import LLMError, LLMResponse
from agentic.runtime import build_registry
from agentic.security import redact, wrap_untrusted
from agentic.supervisor import ROUTE_TOOL, SUPERVISOR_SYSTEM

MAX_REQUESTS = 4
INTEGRATION_CHECKS = ("route_schema_accepted_and_call_parsed", "supervisor_chose_the_only_eligible_agent",
                      "agent_called_a_tool", "first_tool_is_a_search", "agent_continues_after_observing_results",
                      "submit_call_returned", "submit_args_validate_against_schema")
MAX_OUTPUT_TOKENS = 1024     # headroom: thinking tokens count against this cap on Gemini 3
COMPANY = {"name": "Acme Precision Works", "country": "India", "sector": "electronics",
           "products": ["Control systems", "Relays"], "components": ["PCBAs"]}


class SmokeGuardError(RuntimeError):
    pass


def _arg(argv, flag):
    return argv[argv.index(flag) + 1] if flag in argv and argv.index(flag) + 1 < len(argv) else None


def check_guards(env, argv) -> None:
    if env.get("AGENTIC_SMOKE_APPROVED") != "1":
        raise SmokeGuardError("AGENTIC_SMOKE_APPROVED=1 is required (the user has not approved live calls by default)")
    if not _arg(argv, "--model"):
        raise SmokeGuardError("--model <exact model id> is required (no default model is assumed)")
    if "--i-have-checked-my-free-quota" not in argv:
        raise SmokeGuardError("pass --i-have-checked-my-free-quota after confirming the free-tier limits in AI Studio")
    if not env.get("GEMINI_API_KEY"):
        raise SmokeGuardError("GEMINI_API_KEY is not configured")


class CountingLLM:
    """Hard cap on requests; never retries."""

    def __init__(self, inner, cap: int = MAX_REQUESTS):
        self.inner, self.cap, self.n = inner, cap, 0
        self.name, self.model = inner.name, inner.model
        self.log: list[dict] = []

    def generate(self, **kw) -> LLMResponse:
        if self.n >= self.cap:
            raise SmokeGuardError(f"request cap ({self.cap}) reached")
        self.n += 1
        t0 = time.monotonic()
        try:
            resp = self.inner.generate(**kw)
        except LLMError as exc:
            self.log.append({"request": self.n, "error_kind": exc.kind, "error": redact(str(exc))[:300]})
            raise
        self.log.append({"request": self.n, "latency_s": round(time.monotonic() - t0, 2),
                         "input_tokens": resp.usage.input_tokens, "output_tokens": resp.usage.output_tokens,
                         "reasoning_tokens": resp.usage.reasoning_tokens,
                         "provider_meta": (resp.provider_data or {}).get("groq"),
                         "tool_calls": [c.name for c in resp.tool_calls], "text_chars": len(resp.text),
                         # synthetic inputs only, so the raw model output is safe to keep (still redacted)
                         "raw": {"text": redact(resp.text)[:1500],
                                 "tool_calls": [{"name": c.name, "args": json.loads(redact(json.dumps(c.args))[:4000]) if len(json.dumps(c.args)) < 4000 else "[truncated]"} for c in resp.tool_calls]}})
        return resp


def run_smoke(llm, registry=None) -> dict:
    registry = registry or build_registry(AGENT_SPECS)
    spec = AGENT_SPECS["company"]
    tools = [t for t in registry.specs_for("company") if t.name in spec.tool_names]
    world = make_world(COMPANY, "standard")
    f = world["facts"]
    results: dict = {}
    strict = None

    def check(name, ok, detail=""):
        results[name] = {"ok": bool(ok), "detail": detail}

    # 1+2: forced route call (also proves the schema is accepted)
    summary = ("OBJECTIVE: company profile\nCOMPANY: Acme Precision Works | database match: not_found | identity verified: False\n"
               "ELIGIBILITY:\n  - company: ELIGIBLE (prerequisites met)\n  - contact: not eligible (identity not confirmed yet)\n"
               "  - procurement: not eligible (identity not confirmed yet)\n  - supplier: not eligible (no products)\n"
               "  - lead: not eligible (no company data yet)\nAGENT RESULTS SO FAR:\n  (none yet)")
    r1 = llm.generate(system=SUPERVISOR_SYSTEM, messages=[{"role": "user", "content": summary}], tools=[ROUTE_TOOL],
                      max_output_tokens=MAX_OUTPUT_TOKENS, force_tool=True)
    call = r1.tool_calls[0] if r1.tool_calls else None
    check("route_schema_accepted_and_call_parsed", call is not None and call.name == "route")
    check("supervisor_chose_the_only_eligible_agent", bool(call) and call.args.get("next") == "company",
          str(call.args if call else None))

    # 3: first agent step
    task = ("Company submitted by the user: 'Acme Precision Works'\nUser objective: company profile\nDatabase match: not_found\n"
            "Information gaps to fill: official_website, headquarters, industry, founded_year (all missing)\n"
            "Budget: at most 8 tool calls, 6 search credits left. Submit with `submit_company_findings` when done.")
    msgs = [{"role": "user", "content": task}]
    r2 = llm.generate(system=system_prompt(spec), messages=msgs, tools=tools, max_output_tokens=MAX_OUTPUT_TOKENS)
    c2 = r2.tool_calls[0] if r2.tool_calls else None
    check("agent_called_a_tool", c2 is not None, c2.name if c2 else r2.text[:80])
    check("first_tool_is_a_search", bool(c2) and c2.name == "web_search", c2.name if c2 else "")

    # 4: observe synthetic search results and continue
    sr = {"query": "Acme Precision Works official website", "results": [
        {"url": d["url"], "title": d["title"], "snippet": d["content"][:600]} for d in world["search_index"][:5]]}
    msgs += [{"role": "assistant", "content": "", "tool_calls": [c.model_dump() for c in r2.tool_calls],
              "provider_data": r2.provider_data},
             {"role": "tool", "tool_call_id": c2.id if c2 else "x", "name": c2.name if c2 else "web_search",
              "content": wrap_untrusted(json.dumps(sr), "web_search")}]
    r3 = llm.generate(system=system_prompt(spec), messages=msgs, tools=tools, max_output_tokens=MAX_OUTPUT_TOKENS)
    c3 = r3.tool_calls[0] if r3.tool_calls else None
    check("agent_continues_after_observing_results", c3 is not None and c3.name in ("fetch_page", "submit_company_findings"),
          c3.name if c3 else r3.text[:80])

    # 5: forced submit with a synthetic page -> schema + evidence validation
    home = world["pages"][f"https://www.{f['domain']}/"]
    msgs2 = [{"role": "user", "content": task},
             {"role": "user", "content": "You fetched the official page. Page text (untrusted data):\n"
              + wrap_untrusted(json.dumps({"url": f"https://www.{f['domain']}/", "text": home["text"]}), "fetch_page")
              + "\nNow submit your findings: quote exact snippets from this page."}]
    finish = [t for t in tools if t.name == spec.finish_tool]
    r4 = llm.generate(system=system_prompt(spec), messages=msgs2, tools=finish, max_output_tokens=MAX_OUTPUT_TOKENS,
                      force_tool=True)
    c4 = r4.tool_calls[0] if r4.tool_calls else None
    check("submit_call_returned", c4 is not None and c4.name == spec.finish_tool)
    if c4:
        try:
            findings = CompanyFindings.model_validate(c4.args).model_dump()
            check("submit_args_validate_against_schema", True)
            ledger: dict = {}
            ev.add_doc(ledger, url=f"https://www.{f['domain']}/", text=home["text"], title=home["title"])
            state = {"ledger": ledger, "claims": {}, "subject": {"name": COMPANY["name"], "existing": None,
                                                                  "identity": {"verified": False}}}
            pr = process_company(state, findings, None)
            total = len(pr.accepted) + len(pr.rejected)
            strict = strict_assessment(findings, pr)
            check("quoted_snippets_pass_evidence_validation", total > 0 and len(pr.accepted) >= 1,
                  f"accepted {len(pr.accepted)}/{total}; rejected reasons: {[r['reasons'] for r in pr.rejected][:4]}")
        except Exception as exc:  # noqa: BLE001
            check("submit_args_validate_against_schema", False, redact(str(exc))[:300])
    integration = all(c["ok"] for n, c in results.items() if n in INTEGRATION_CHECKS)
    evidence_ok = bool(strict and strict["strict_passed"])
    return {"requests_made": llm.n, "checks": results, "per_request": llm.log,
            # (1) PROTOCOL / INTEGRATION: schemas accepted, calls parsed, right tool order, args valid
            "integration_passed": integration,
            # (2) EVIDENCE QUALITY: every claim accepted by the evidence validator, no unquoted/fabricated evidence
            "evidence_quality_passed": evidence_ok,
            "strict": strict, "strict_passed": integration and evidence_ok,
            "all_passed": all(c["ok"] for c in results.values()),     # LEGACY name: integration + '>= 1 claim accepted'
            "total_input_tokens": sum(x.get("input_tokens", 0) for x in llm.log),
            "total_output_tokens": sum(x.get("output_tokens", 0) for x in llm.log),
            "total_reasoning_tokens": sum(x.get("reasoning_tokens", 0) for x in llm.log),
            "total_latency_s": round(sum(x.get("latency_s", 0) for x in llm.log), 2),
            "note": "synthetic inputs only; no Tavily, no database; results say nothing about real-page accuracy"}


def main(argv: list[str]) -> None:  # pragma: no cover - live
    if "--dry-run" in argv:
        print(__doc__)
        print("Dry run: no request made. Model:", _arg(argv, "--model") or "(none given)")
        return
    if "--load-dotenv" in argv:                      # opt-in only; by default .env is never read
        from dotenv import load_dotenv
        load_dotenv()
    check_guards(os.environ, argv)
    from agentic.llm.gemini import GeminiClient
    model = _arg(argv, "--model")
    level = _arg(argv, "--thinking-level") or ("low" if model.startswith("gemini-3") else None)
    llm = CountingLLM(GeminiClient(model=model, thinking_level=level))
    try:
        out = run_smoke(llm)
        out["model"] = model
    except LLMError as exc:
        out = {"model": model, "requests_made": llm.n, "aborted": f"{exc.kind}: {redact(str(exc))[:300]}",
               "per_request": llm.log}
    path = next((a for a in argv if a.endswith(".json")), "data/gemini_smoke_result.json")
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    json.dump(out, open(path, "w", encoding="utf-8"), indent=2)
    print(json.dumps({k: out.get(k) for k in ("model", "requests_made", "integration_passed", "evidence_quality_passed", "aborted")}, indent=2), "->", path)


if __name__ == "__main__":  # pragma: no cover
    main(sys.argv[1:])

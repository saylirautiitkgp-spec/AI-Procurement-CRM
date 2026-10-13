"""Provider comparison harness (Gemini vs Groq vs ...). OFFLINE by default.

* `PromptSizeProbe` wraps any LLMClient and records how large each request really is (system prompt + tool
  schemas + conversation), so free-tier token budgets can be estimated from real prompts, not guesses.
* `metrics_from_run` turns a finished run's trace into comparable reliability numbers.
* `run_stage1` runs the SAME mock-web cases through the full graph for a given LLM (no Tavily, no database).

A live Stage 1 needs AGENTIC_PROVIDER_EVAL_APPROVED=1 and a per-provider request cap; it is not run by tests.
"""

from __future__ import annotations

import json
import os
import statistics
import sys
import time

from agentic.eval.run_offline import evaluate_run
from agentic.eval.gemini_smoke import CountingLLM, SmokeGuardError
from agentic.llm.base import LLMResponse

CHARS_PER_TOKEN = 4          # rough English/JSON estimate; replace with provider-reported usage once live
from agentic.eval.quality import FABRICATION_REASONS   # noqa: E402 - shared definition


class PromptSizeProbe:
    def __init__(self, inner):
        self.inner, self.name, self.model = inner, getattr(inner, "name", "?"), getattr(inner, "model", "?")
        self.sizes: list[int] = []

    def generate(self, **kw) -> LLMResponse:
        from agentic.llm.gemini import simplify_schema     # the adapters send flattened schemas, not raw pydantic ones
        chars = len(kw["system"]) + len(json.dumps([simplify_schema(t.parameters) for t in kw["tools"]])) + \
            len(json.dumps([t.description for t in kw["tools"]])) + len(json.dumps(kw["messages"], default=str))
        self.sizes.append(chars)
        return self.inner.generate(**kw)

    def summary(self) -> dict:
        toks = [c // CHARS_PER_TOKEN for c in self.sizes]
        return {"requests": len(toks), "est_input_tokens_total": sum(toks),
                "est_input_tokens_max_request": max(toks, default=0),
                "est_input_tokens_mean_request": round(statistics.mean(toks), 0) if toks else 0}


def metrics_from_run(out: dict) -> dict:
    trace = out["trace"]
    calls = [e for e in trace if e["type"] == "tool_result"]
    bad = [e for e in calls if not e["data"]["ok"] and (e["data"].get("error") or {}).get("code") in
           ("unknown_tool", "invalid_arguments", "tool_not_permitted", "duplicate_call")]
    rejected = [r for items in out["rejected"].values() for r in items]
    fabricated = [r for r in rejected if set(r["reasons"]) & FABRICATION_REASONS]
    accepted = sum(len(v) for v in out["claims"].values() if isinstance(v, list))
    llm_calls = [e for e in trace if e["type"] == "llm_call"]
    submits = [e for e in trace if e["type"] == "submit"]
    return {
        "status": out["status"], "stop_reason": out.get("stop_reason"),
        "llm_requests": len(llm_calls), "tool_calls": len(calls),
        "invalid_tool_call_rate": round(len(bad) / len(calls), 3) if calls else 0.0,
        "route_vetoes": sum(1 for e in trace if e["type"] == "route_vetoed"),
        "agent_runs_ended_normally": sum(1 for r in out["agent_results"].values() if r["end_reason"] == "submitted"),
        "agent_runs_total": len(out["agent_results"]),
        "claims_accepted": accepted, "claims_rejected": len(rejected),
        "fabrication_attempt_rate": round(len(fabricated) / (accepted + len(rejected)), 3) if accepted + len(rejected) else 0.0,
        "submit_rejections": sum(1 for e in submits if e["data"]["rejected"]),
        "mean_llm_latency_s": round(statistics.mean([e["data"]["latency_s"] for e in llm_calls]), 3) if llm_calls else 0,
        "input_tokens": out["budgets"]["input_tokens"], "output_tokens": out["budgets"]["output_tokens"],
        "search_credits_simulated": out["budgets"]["search_credits"],
    }


def run_stage1(llm, cases: list[dict], max_requests: int, *, probe: bool = True) -> dict:
    """Same mock-web cases for every provider. `cases`: [{"company": {...}, "variant": "standard"}, ...]"""
    counted = CountingLLM(llm, cap=max_requests)
    sized = PromptSizeProbe(counted) if probe else counted
    per_case = []
    for c in cases:
        t0 = time.monotonic()
        try:
            m, out = evaluate_run(c["company"], c["variant"], llm=sized)
            per_case.append({"company": c["company"]["name"], "variant": c["variant"],
                             "wall_s": round(time.monotonic() - t0, 2), "expectation_ok": m.expectation_ok,
                             **metrics_from_run(out)})
        except SmokeGuardError as exc:
            per_case.append({"company": c["company"]["name"], "variant": c["variant"], "aborted": str(exc)})
            break
    return {"provider": getattr(llm, "name", "?"), "model": getattr(llm, "model", "?"),
            "requests_made": counted.n, "per_case": per_case, "prompt_sizes": sized.summary() if probe else None,
            "per_request": counted.log,
            "note": "mock web + synthetic data; says nothing about real-page accuracy"}


# The SAME three cases are used for every provider (company_ids from tests/fixtures/first20.json):
#   101 Pixxel (standard full loop), 187 Robonic (empty search -> must adapt), 251 Gannon Dunkerley (prompt injection)
STAGE2_CASE_IDS = (101, 187, 251)


def stage2_cases(path: str = "tests/fixtures/first20.json") -> list[dict]:
    by_id = {c["company_id"]: c for c in json.load(open(path, encoding="utf-8"))["companies"]}
    return [{"company": by_id[i], "variant": by_id[i]["web_variant"]} for i in STAGE2_CASE_IDS]


def _arg(argv, flag):
    return argv[argv.index(flag) + 1] if flag in argv and argv.index(flag) + 1 < len(argv) else None


def main(argv: list[str]) -> None:  # pragma: no cover - live, user-approved only
    """python -m agentic.eval.provider_eval <gemini|groq> --model <exact id> --i-have-checked-my-free-quota [out.json]"""
    provider = argv[0]
    if os.environ.get("AGENTIC_PROVIDER_EVAL_APPROVED") != "1" or "--i-have-checked-my-free-quota" not in argv:
        raise SmokeGuardError("live provider evaluation needs AGENTIC_PROVIDER_EVAL_APPROVED=1 and "
                              "--i-have-checked-my-free-quota")
    model = _arg(argv, "--model")
    if not model:
        raise SmokeGuardError("--model <exact model id> is required (no default model is assumed)")
    if "--load-dotenv" in argv:                    # opt-in only; .env is never read by default
        from dotenv import load_dotenv
        load_dotenv()
    if provider == "gemini":
        from agentic.llm.gemini import GeminiClient
        level = _arg(argv, "--thinking-level") or ("low" if model.startswith("gemini-3") else None)
        llm = GeminiClient(model=model, thinking_level=level)
    elif provider == "groq":
        from agentic.llm.groq import GroqClient
        llm = GroqClient(model=model)
    else:
        raise SystemExit("provider must be gemini or groq")
    cap = int(os.environ.get("AGENTIC_PROVIDER_EVAL_MAX_REQUESTS", "60"))
    out = run_stage1(llm, stage2_cases(), cap)
    path = next((a for a in argv if a.endswith(".json")), f"data/provider_eval_{provider}.json")
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    json.dump(out, open(path, "w", encoding="utf-8"), indent=2)
    print("wrote", path)


if __name__ == "__main__":  # pragma: no cover
    main(sys.argv[1:])

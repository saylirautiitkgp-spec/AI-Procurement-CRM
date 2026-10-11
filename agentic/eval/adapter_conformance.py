"""Fair OFFLINE comparison of the Gemini and Groq adapters on identical cases and identical mocked tool results.

The same deterministic decision-maker (PolicyLLM) answers behind both wire emulators, so any difference in the
research output would be an ADAPTER bug. What is measured: output identity, wire-protocol violations, request counts,
the real payload each adapter builds (tokens estimated as chars/4) and the largest single request.

Not measurable offline (needs live runs): tool-choice quality, adaptation, injection resistance of a real model,
real latency, real token accounting (e.g. gpt-oss reasoning tokens).

    python -m agentic.eval.adapter_conformance
"""

from __future__ import annotations

import json

from agentic.config import Efficiency, Limits
from agentic.eval.cost_report import signature
from agentic.eval.policy import PolicyLLM
from agentic.eval.provider_eval import stage2_cases
from agentic.eval.run_offline import evaluate_run
from agentic.eval.wire_emulators import CHARS_PER_TOKEN, GeminiWire, GroqWire
from agentic.llm.gemini import GeminiClient
from agentic.llm.groq import GroqClient

GROQ_MODELS = ("openai/gpt-oss-120b", "openai/gpt-oss-20b")


def make_client(provider: str, model: str):
    decider = PolicyLLM()
    if provider == "groq":
        wire = GroqWire(decider)
        return GroqClient(model=model, client=wire), wire
    wire = GeminiWire(decider)
    return GeminiClient(model=model, client=wire, thinking_level="low" if model.startswith("gemini-3") else None), wire


def run(provider: str, model: str, *, big_pages: bool = False, efficiency: Efficiency | None = None,
        limits: Limits | None = None, objective: str | None = None, cases: list[dict] | None = None) -> dict:
    # The per-run token BUDGET is lifted here on purpose: this compares adapters/payloads, not budget behaviour.
    limits = limits or Limits(max_total_tokens_run=1_000_000, max_wall_clock_s=3600)
    rows = []
    for case in (cases or stage2_cases()):
        _, direct = evaluate_run(case["company"], case["variant"], llm=PolicyLLM(), big_pages=big_pages,
                                 efficiency=efficiency, limits=limits, objective=objective)
        client, wire = make_client(provider, model)
        _, out = evaluate_run(case["company"], case["variant"], llm=client, big_pages=big_pages, efficiency=efficiency,
                              limits=limits, objective=objective)
        toks = [c // CHARS_PER_TOKEN for c in wire.payload_chars]
        rows.append({"company": case["company"]["name"], "variant": case["variant"],
                     "same_output_as_direct": signature(out) == signature(direct),
                     "wire_violations": list(wire.violations), "requests": wire.requests,
                     "est_prompt_tokens": sum(toks), "largest_request_tokens": max(toks, default=0),
                     "status": out["status"]})
    return {"provider": provider, "model": model, "cases": rows,
            "total_requests": sum(r["requests"] for r in rows), "total_prompt_tokens": sum(r["est_prompt_tokens"] for r in rows),
            "largest_request_tokens": max(r["largest_request_tokens"] for r in rows),
            "violations": sum(len(r["wire_violations"]) for r in rows),
            "all_outputs_identical": all(r["same_output_as_direct"] for r in rows)}


def main() -> None:  # pragma: no cover
    for big in (False, True):
        print("== realistic page sizes" if big else "== small synthetic pages")
        for prov, model in (("gemini", "gemini-3-flash-preview"), ("groq", GROQ_MODELS[0]), ("groq", GROQ_MODELS[1])):
            r = run(prov, model, big_pages=big)
            print(f"{prov:7}{model:26} requests={r['total_requests']:3} prompt_tokens={r['total_prompt_tokens']:7} "
                  f"largest={r['largest_request_tokens']:5} violations={r['violations']} identical={r['all_outputs_identical']}")


if __name__ == "__main__":  # pragma: no cover
    main()

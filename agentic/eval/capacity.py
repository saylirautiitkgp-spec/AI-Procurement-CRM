"""Free-tier capacity model: how many agent runs fit per day / how long one run takes, per provider.

Inputs are MEASURED OFFLINE (real prompts through the real adapters, mocked web, realistic page sizes). The only
unmeasured quantity is how many OUTPUT tokens a real model produces per request (reasoning tokens for gpt-oss, thinking
tokens for Gemini) - it is therefore a scenario parameter, never a result.

Limits are the operator's own console values (passed in, not assumed). Planning headroom: 80% of every limit.

    python -m agentic.eval.capacity
"""

from __future__ import annotations

from agentic.config import Efficiency, Limits
from agentic.eval.adapter_conformance import run
from agentic.eval.provider_eval import stage2_cases
from agentic.llm.pacing import ProviderQuota

GROQ = ProviderQuota(rpm=30, tpm=8000, rpd=1000, tpd=200_000)       # user-reported console values, UNVERIFIED for the plan
GEMINI3 = {"rpm": 5, "rpd": 20, "tpm": 250_000}                      # user-reported AI Studio values (free tier)
MAX_OUTPUT_RESERVATION = 1024                                         # what the runtime asks for per request
OUTPUT_SCENARIOS = (150, 400, 1000)                                   # output(+reasoning) tokens per request: ASSUMPTIONS

SCENARIOS = {
    "A. current defaults": (Efficiency(), 6000),
    "B. + tool results capped at 3,000 chars": (Efficiency(), 3000),
    "C. B + compact_schemas": (Efficiency(compact_schemas=True), 3000),
    "D. C + 2,000-char tool results": (Efficiency(compact_schemas=True), 2000),
}


def measure(scenario: str, objective: str | None, case_index: int = 0, provider: str = "groq",
            model: str = "openai/gpt-oss-120b") -> dict:
    eff, chars = SCENARIOS[scenario]
    lim = Limits(max_total_tokens_run=10_000_000, max_wall_clock_s=36000, tool_result_chars_for_llm=chars)
    case = stage2_cases()[case_index]
    r = run(provider, model, big_pages=True, efficiency=eff, limits=lim, objective=objective, cases=[case])
    c = r["cases"][0]
    return {"scenario": scenario, "requests": c["requests"], "prompt_tokens": c["est_prompt_tokens"],
            "largest_request": c["largest_request_tokens"], "identical": c["same_output_as_direct"],
            "violations": len(c["wire_violations"])}


def capacity(m: dict, out_per_request: int, q: ProviderQuota = GROQ) -> dict:
    tokens = m["prompt_tokens"] + m["requests"] * out_per_request
    return {"tokens_per_run": tokens,
            "runs_per_day_by_tokens": round(q.tpd * q.headroom / tokens, 2),
            "runs_per_day_by_requests": round(q.rpd * q.headroom / m["requests"], 1),
            "min_minutes_per_run_by_tpm": round(tokens / (q.tpm * q.headroom), 1),
            "largest_request_plus_reservation": m["largest_request"] + MAX_OUTPUT_RESERVATION,
            "fits_tpm_limit": m["largest_request"] + MAX_OUTPUT_RESERVATION <= q.tpm}


def gemini3_runs_per_day(requests: int, headroom: float = 0.8) -> float:
    return round(GEMINI3["rpd"] * headroom / requests, 2)


def report() -> str:
    lines = ["| Scenario (realistic pages, Pixxel) | Run type | LLM requests | prompt tokens | largest request | "
             "fits 8K TPM incl. 1,024 reserve | runs/day by TPD @150/400/1000 out tokens per request | min minutes/run (TPM) @400 |",
             "|---|---|---|---|---|---|---|---|"]
    for name in SCENARIOS:
        for label, obj in (("full five-agent", None), ("company only", "company profile")):
            m = measure(name, obj)
            caps = [capacity(m, o) for o in OUTPUT_SCENARIOS]
            lines.append(f"| {name} | {label} | {m['requests']} | {m['prompt_tokens']:,} | {m['largest_request']:,} | "
                         f"{'yes' if caps[0]['fits_tpm_limit'] else '**NO**'} | "
                         f"{caps[0]['runs_per_day_by_tokens']} / {caps[1]['runs_per_day_by_tokens']} / {caps[2]['runs_per_day_by_tokens']} | "
                         f"{caps[1]['min_minutes_per_run_by_tpm']} |")
    return "\n".join(lines)


if __name__ == "__main__":  # pragma: no cover
    print(report())
    full = measure("A. current defaults", None)
    print(f"\nGemini 3 Flash free tier (20 RPD, 80% headroom): {gemini3_runs_per_day(full['requests'])} full runs/day "
          f"({full['requests']} requests each)")

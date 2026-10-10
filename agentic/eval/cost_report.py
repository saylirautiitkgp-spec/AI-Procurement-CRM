"""Before/after token and request measurements on the SAME offline fixtures (mock web, PolicyLLM test double).

For every efficiency configuration it runs the same companies and checks that the research OUTPUT is identical
to the baseline (same accepted claims, statuses, lead score) - so a saving is never bought by dropping evidence.

Caveats (printed in the report): token counts are estimates (chars/4) of what the provider adapters actually send;
the policy double is not an LLM, so changes that alter what a real model is *told* (ledger hint, batching hint,
schema compaction) can only be shown to be mechanically correct here - their behavioural effect needs a live
comparison.

    python -m agentic.eval.cost_report [--large]
"""

from __future__ import annotations

import json
import sys
from dataclasses import fields, replace

from agentic.config import BASELINE_EFFICIENCY, Efficiency
from agentic.eval.policy import PolicyLLM
from agentic.eval.provider_eval import PromptSizeProbe
from agentic.eval.run_offline import evaluate_run

FULL_RUN_VARIANTS = ("standard", "ambiguous", "conflicting", "no_contacts", "injection")


def signature(out: dict) -> dict:
    claims = sorted((a, c.get("field") or c.get("kind"), json.dumps(c.get("value"), sort_keys=True, default=str))
                    for a, items in out["claims"].items() if isinstance(items, list) for c in items)
    lead = out["claims"].get("lead") or {}
    return {"status": out["status"], "agents": sorted(out["completed_agents"]), "claims": claims,
            "lead": (lead.get("score"), lead.get("band")), "identity": out["subject"]["identity"].get("verified")}


def cases(path="tests/fixtures/first20.json") -> list[dict]:
    return [c for c in json.load(open(path, encoding="utf-8"))["companies"] if c["web_variant"] in FULL_RUN_VARIANTS]


def measure(eff: Efficiency, big_pages: bool, data=None) -> dict:
    data = data or cases()
    tot = {"requests": 0, "input_tokens": 0, "search_credits": 0, "tool_calls": 0, "supervisor_requests": 0}
    sigs = []
    for c in data:
        probe = PromptSizeProbe(PolicyLLM())
        _, out = evaluate_run(c, c["web_variant"], llm=probe, efficiency=eff, big_pages=big_pages)
        s = probe.summary()
        tot["requests"] += s["requests"]
        tot["input_tokens"] += s["est_input_tokens_total"]
        tot["search_credits"] += out["budgets"]["search_credits"]
        tot["tool_calls"] += out["budgets"]["tool_calls"]
        tot["supervisor_requests"] += sum(1 for e in out["trace"] if e["type"] == "llm_call" and e["agent"] == "supervisor")
        sigs.append(signature(out))
    n = len(data)
    return {**{k: round(v / n, 1) for k, v in tot.items()}, "runs": n, "signatures": sigs}


def configurations() -> dict[str, Efficiency]:
    names = [f.name for f in fields(Efficiency)]
    cfg = {"baseline (all off)": BASELINE_EFFICIENCY}
    for n in names:
        cfg[f"+ {n}"] = replace(BASELINE_EFFICIENCY, **{n: True})
    cfg["defaults (safe set ON)"] = Efficiency()
    cfg["everything ON"] = Efficiency(*[True] * len(names))
    return cfg


def report(big_pages: bool) -> dict:
    data = cases()
    rows, base = {}, None
    for name, eff in configurations().items():
        m = measure(eff, big_pages, data)
        base = base or m
        m["output_identical_to_baseline"] = m["signatures"] == base["signatures"]
        m["tokens_saved_pct"] = round(100 * (1 - m["input_tokens"] / base["input_tokens"]), 1)
        m["requests_saved_pct"] = round(100 * (1 - m["requests"] / base["requests"]), 1)
        rows[name] = m
    return rows


def markdown(rows: dict, title: str) -> str:
    out = [f"### {title}", "", "| Configuration | LLM requests/run | of which supervisor | est. input tokens/run | search credits/run | tokens saved | output identical |",
           "|---|---|---|---|---|---|---|"]
    for name, m in rows.items():
        out.append(f"| {name} | {m['requests']} | {m['supervisor_requests']} | {m['input_tokens']:,.0f} | {m['search_credits']} | "
                   f"{m['tokens_saved_pct']}% | {'yes' if m['output_identical_to_baseline'] else '**NO**'} |")
    return "\n".join(out)


if __name__ == "__main__":  # pragma: no cover
    for big in ([False, True] if "--large" in sys.argv else [False]):
        print(markdown(report(big), "large realistic pages" if big else "small synthetic pages"))
        print()

"""Offline evaluation over the first-20 development set.

Everything here is SYNTHETIC and OFFLINE: web pages are generated fixtures on the reserved `.example`
TLD, and the "LLM" is the PolicyLLM test double. The numbers therefore measure the HARNESS (routing,
tool loop, evidence validation, limits, safety) - NOT real-world coverage, real model quality, or any
productivity gain. No baseline has been collected, so no improvement is claimed.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass

from agentic import evidence as ev
from agentic.agents import AGENT_SPECS
from agentic.backends.fixtures import FixtureFetcher, FixtureSearch
from agentic.config import Limits, Settings
from agentic.eval.policy import PolicyLLM
from agentic.eval.synthetic import make_world
from agentic.persistence.store import Store
from agentic.repo import InMemoryRepo
from agentic.runtime import AgenticRunner, make_deps

OBJECTIVE = ("Build a comprehensive, evidence-backed company intelligence report: identity, profile, "
             "procurement structure, public procurement decision-makers, purchased products/components, "
             "supplier relationships and lead priority - using only the agents that help.")


def repo_for(company: dict) -> InMemoryRepo:
    row = {"company_id": company["company_id"], "company_name": company["name"], "sector_id": 0,
           "sector_name": company.get("sector"), "country": company.get("country"), "website": None,
           "linkedin_url": None, "revenue": None, "number_of_plants": None, "company_type": None,
           "enrichment_status": "Pending"}
    cid = company["company_id"]
    return InMemoryRepo([row], products={cid: company.get("products", [])},
                        components={cid: company.get("components", [])},
                        contacts={cid: [{"designation": None, "source_url": None, "email": None}]})


def world_text(world: dict) -> str:
    """Everything an honest snippet could have come from."""
    parts = [p["text"] for p in world["pages"].values()] + [d["content"] for d in world["search_index"]]
    return ev.norm_text(" ".join(parts))


@dataclass
class RunMetrics:
    company: str
    variant: str
    status: str
    stop_reason: str | None
    wall_ms: float
    tool_calls: int
    llm_calls: int
    search_credits: int
    docs: int
    identity_verified: bool
    coverage: float
    claims_accepted: int
    claims_rejected: int
    evidence_checked: int
    evidence_valid: int
    fact_checked: int
    fact_correct: int
    secrets_in_trace: bool
    expectation_ok: bool
    expectation: str


def evaluate_run(company: dict, variant: str, *, specs=None, limits: Limits | None = None, llm=None,
                 secrets: tuple = (), in_db: bool = True, efficiency=None, big_pages: bool = False) -> tuple[RunMetrics, dict]:
    specs = specs or AGENT_SPECS
    world = make_world(company, variant, big_pages=big_pages)
    settings = Settings(limits=limits or Limits(), **({"efficiency": efficiency} if efficiency is not None else {}))
    deps = make_deps(llm=llm or PolicyLLM(), repo=repo_for(company) if in_db else InMemoryRepo([]), search=FixtureSearch(world),
                     fetcher=FixtureFetcher(world), store=Store(":memory:"), settings=settings,
                     specs=specs, sleep=lambda s: None)
    runner = AgenticRunner(deps, specs)
    t0 = time.perf_counter()
    out = runner.start(company["name"], OBJECTIVE)
    wall = (time.perf_counter() - t0) * 1000
    rep = out["report"]
    truth = ev.norm_text(" ".join([p["text"] for p in world["pages"].values()]
                                  + [d["content"] for d in world["search_index"]]))

    ev_checked = ev_valid = 0
    for items in out["claims"].values():
        if isinstance(items, dict):        # e.g. lead score object
            continue
        for c in items:
            for e in c.get("evidence", []):
                ev_checked += 1
                if ev.norm_text(e["snippet"]) in truth:
                    ev_valid += 1
    facts = world["facts"]
    fact_checked = fact_correct = 0
    prof = rep.get("company_profile") or {}
    checks = {"headquarters": facts["city"], "founded_year": str(facts["founded"]),
              "employee_count": f"{facts['employees']:,}"}
    for fld, want in checks.items():
        if fld in prof:
            fact_checked += 1
            fact_correct += int(str(prof[fld]["value"]) == want)
    gaps = rep["gaps"]["company_fields"]
    coverage = sum(1 for g in gaps if g["status"] == "verified") / len(gaps)
    blob = json.dumps(out, default=str)
    leaked = any(s and s in blob for s in secrets) or bool(re.search(r"sb_secret_|tvly-|AIza", blob))
    ok, why = expectation(variant, out, rep, runner)
    m = RunMetrics(
        company=company["name"], variant=variant, status=out["status"], stop_reason=out.get("stop_reason"),
        wall_ms=round(wall, 1), tool_calls=out["budgets"]["tool_calls"], llm_calls=out["budgets"]["llm_calls"],
        search_credits=out["budgets"]["search_credits"], docs=len(out["ledger"]),
        identity_verified=rep["identity"]["verified"], coverage=round(coverage, 3),
        claims_accepted=sum(len(v) for v in out["claims"].values() if isinstance(v, list)),
        claims_rejected=sum(len(v) for v in out["rejected"].values()),
        evidence_checked=ev_checked, evidence_valid=ev_valid, fact_checked=fact_checked,
        fact_correct=fact_correct, secrets_in_trace=leaked, expectation_ok=ok, expectation=why)
    return m, out


def expectation(variant: str, out: dict, rep: dict, runner) -> tuple[bool, str]:
    ident = rep["identity"]["verified"]
    if variant in ("standard", "no_contacts", "injection", "conflicting", "ambiguous"):
        ok = ident and out["status"] in ("completed", "partial")
        if variant == "conflicting":
            ok = ok and any(c["field"] == "employee_count" for c in rep["gaps"]["conflicts"])
        return ok, "identity verified from first-party evidence" + (" + conflict surfaced" if variant == "conflicting" else "")
    if variant == "no_website":
        return (not ident and not (rep["company_profile"] or {}).get("official_website")), \
            "no first-party site: identity stays unverified, no website invented"
    if variant == "search_quota":
        return (out["budgets"]["search_disabled"] and out["status"] == "partial"), \
            "quota exhausted -> search disabled, partial result, no retry storm"
    if variant == "search_rate_limit":
        return (out["status"] == "partial" and not ident), "persistent rate limit -> partial, nothing invented"
    if variant in ("search_error", "empty_search"):
        return (out["report"]["company_profile"] is None and not ident), \
            "no usable search results -> empty profile reported honestly"
    return False, "unknown variant"


def summarize(metrics: list[RunMetrics]) -> dict:
    n = len(metrics)
    ev_c = sum(m.evidence_checked for m in metrics) or 1
    fc = sum(m.fact_checked for m in metrics) or 1
    return {
        "runs": n, "expectations_met": sum(m.expectation_ok for m in metrics),
        "status_counts": {s: sum(1 for m in metrics if m.status == s) for s in {m.status for m in metrics}},
        "evidence_validity": round(sum(m.evidence_valid for m in metrics) / ev_c, 4),
        "synthetic_fact_accuracy": round(sum(m.fact_correct for m in metrics) / fc, 4),
        "mean_coverage_company_fields": round(sum(m.coverage for m in metrics) / n, 3),
        "mean_tool_calls": round(sum(m.tool_calls for m in metrics) / n, 2),
        "mean_llm_calls": round(sum(m.llm_calls for m in metrics) / n, 2),
        "total_search_credits_simulated": sum(m.search_credits for m in metrics),
        "mean_harness_wall_ms": round(sum(m.wall_ms for m in metrics) / n, 1),
        "secret_leaks": sum(m.secrets_in_trace for m in metrics),
        "note": "Synthetic fixtures + PolicyLLM test double: measures the harness only, not real-world "
                "accuracy, real LLM quality, or productivity. No baseline collected; no improvement claimed.",
    }


def run_all(path: str = "tests/fixtures/first20.json", specs=None, secrets=()) -> dict:
    data = json.load(open(path, encoding="utf-8"))
    metrics = []
    for c in data["companies"]:
        m, _ = evaluate_run(c, c["web_variant"], specs=specs, secrets=secrets)
        metrics.append(m)
    return {"summary": summarize(metrics), "runs": [m.__dict__ for m in metrics]}


if __name__ == "__main__":  # pragma: no cover
    import sys
    res = run_all(sys.argv[1] if len(sys.argv) > 1 else "tests/fixtures/first20.json")
    print(json.dumps(res["summary"], indent=2))
    for r in res["runs"]:
        print(f"{r['company'][:34]:34} {r['variant']:18} {r['status']:10} cov={r['coverage']:.2f} "
              f"tools={r['tool_calls']} credits={r['search_credits']} ok={r['expectation_ok']}")

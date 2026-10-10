"""First-20 development set + generalisation to arbitrary companies (offline, synthetic)."""

import json

from agentic.eval import select_first20 as sel
from agentic.eval.run_offline import evaluate_run, run_all, summarize
from agentic.eval.synthetic import VARIANTS

FIXTURE = "tests/fixtures/first20.json"


def rows(n=200):
    return [{"company_id": i, "company_name": f"Company {i}", "sector_id": 1 + i % 7, "country": "India"}
            for i in range(1, n + 1)]


def test_selection_is_deterministic_stratified_and_independent_of_input_order():
    a = sel.select(rows())
    b = sel.select(list(reversed(rows())))
    assert [r["company_id"] for r in a] == [r["company_id"] for r in b] and len(a) == 20
    per_sector = {}
    for r in a:
        per_sector[r["sector_id"]] = per_sector.get(r["sector_id"], 0) + 1
    assert max(per_sector.values()) - min(per_sector.values()) <= 1 and len(per_sector) == 7
    assert [r["company_id"] for r in sel.select(rows(), seed="another")] != [r["company_id"] for r in a]


def test_committed_fixture_matches_the_documented_method():
    data = json.load(open(FIXTURE, encoding="utf-8"))
    ids = [c["company_id"] for c in data["companies"]]
    assert data["seed"] == sel.SEED and data["n"] == 20 and len(ids) == len(set(ids)) == 20
    assert data["source_rows"] >= 900
    assert [c["web_variant"] for c in data["companies"]] == [VARIANTS[i % len(VARIANTS)] for i in range(20)]
    assert all("email" not in json.dumps(c).lower() for c in data["companies"])          # no contact data in fixtures
    # re-ranking the selected ids must reproduce the within-sector order the method specifies
    by_sector = {}
    for c in data["companies"]:
        by_sector.setdefault(c["sector"], []).append(c["company_id"])
    for ids_in_sector in by_sector.values():
        assert ids_in_sector == sorted(ids_in_sector, key=lambda i: sel.rank(i))


def test_offline_evaluation_over_the_first_20_meets_every_expectation():
    res = run_all(FIXTURE, secrets=("sb_secret_x",))
    s = res["summary"]
    assert s["runs"] == 20 and s["expectations_met"] == 20, [r for r in res["runs"] if not r["expectation_ok"]]
    assert s["evidence_validity"] == 1.0 and s["synthetic_fact_accuracy"] == 1.0 and s["secret_leaks"] == 0
    assert "No baseline collected" in s["note"]
    assert {"completed", "partial"} >= set(s["status_counts"])


def test_arbitrary_company_names_not_in_the_database_work_for_every_failure_mode():
    for variant in VARIANTS:
        company = {"company_id": -1, "name": "Orbit Vector Dynamics Pvt. Ltd.", "sector": "aerospace",
                   "country": "India", "products": ["Gyros"], "components": ["Bearings"]}
        m, out = evaluate_run(company, variant, in_db=False)
        assert out["report"]["database_match"]["status"] == "not_found", variant
        assert m.expectation_ok, (variant, m.expectation)
        assert m.evidence_valid == m.evidence_checked and not m.secrets_in_trace
        assert out["proposals"] == []                                            # research only, no DB involvement


def test_unusual_names_are_handled():
    for name in ["O'Reilly & Sons (Wind) Ltd.", "Müller Elektronik GmbH", "A", "x" * 150, "Société Générale d'Optique"]:
        company = {"company_id": -2, "name": name, "country": "India"}
        if len(name.strip()) < 2:
            continue
        m, out = evaluate_run(company, "standard", in_db=False)
        assert out["status"] in ("completed", "partial", "failed") and not m.secrets_in_trace


def test_summary_never_claims_productivity_gains():
    s = summarize([])if False else run_all(FIXTURE)["summary"]
    text = json.dumps(s).lower()
    assert "reduction" not in text and "faster than" not in text and "saved" not in text

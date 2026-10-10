"""Dry-run merge planner: the real 79 duplicate-name groups (read-only snapshot) + crafted edge cases.

Nothing here touches Supabase; the planner can only read and report."""

import copy
import json
import pathlib
import sys
import types

import pytest

from agentic import merge_plan as mp
from agentic.merge_sources import InMemoryMergeSource, SupabaseMergeSource

SNAPSHOT = "tests/fixtures/duplicate_groups.json"


@pytest.fixture(scope="module")
def legacy_env():
    """The legacy services create a DB client at import time; give them an inert stand-in."""
    mpatch = pytest.MonkeyPatch()
    mpatch.setitem(sys.modules, "database.supabase_client", types.SimpleNamespace(supabase=object()))
    for name in [m for m in sys.modules if m.startswith("services.")]:
        mpatch.delitem(sys.modules, name, raising=False)
    yield
    mpatch.undo()


@pytest.fixture(scope="module")
def snapshot():
    return json.load(open(SNAPSHOT, encoding="utf-8"))


@pytest.fixture(scope="module")
def plan(legacy_env, snapshot):
    return mp.MergePlanner(InMemoryMergeSource(snapshot)).build()


def pairs(plan):
    return [p for g in plan["groups"] for p in g["pairs"]]


# ============================================================= the real 79 groups
def test_finds_the_79_known_duplicate_name_groups_without_executing_anything(plan):
    assert plan["summary"]["groups"] == 79 and plan["summary"]["pairs"] == 101
    assert plan["dry_run"] is True and plan["executed_operations"] == 0
    assert plan["summary"]["pairs_ready_to_execute"] == 0
    assert all(p["execution"]["ready"] is False and len(p["execution"]["blockers"]) >= 3 for p in pairs(plan))
    sizes = sorted(len(g["members"]) for g in plan["groups"])
    assert sizes.count(2) == 68 and sizes.count(3) == 11            # same shape as the live data


def test_every_recommendation_carries_evidence_golden_record_fields_and_dependents(plan):
    for g in plan["groups"]:
        assert g["golden_record"]["master_id"] in [m["company_id"] for m in g["members"]]
        assert g["golden_record"]["rationale"] and len(g["golden_record"]["legacy_scores"]) == len(g["members"])
        for p in g["pairs"]:
            assert p["rationale"] and p["category"] and p["recommendation"] and p["confidence"] in ("high", "medium", "low")
            assert {p["proposed_master_id"], p["proposed_duplicate_id"]} == set(p["pair"])
            assert [t["table"] for t in p["dependents"]["tables"]] == list(mp.COMPANY_FK_TABLES)
            assert "companies" in p["field_decisions"] and "conflicts" in p["field_decisions"]


def test_cross_sector_same_name_companies_are_never_merge_candidates(plan, snapshot):
    sectors = {c["company_id"]: c["sector_id"] for c in snapshot["companies"]}
    cross = [p for p in pairs(plan) if sectors[p["pair"][0]] != sectors[p["pair"][1]] and p["name_match"] == "exact_name"]
    assert len(cross) >= 40
    for p in cross:
        assert p["category"] == "same_name_different_sector"
        assert p["recommendation"] in ("manual_review", "link_multi_sector")
        assert "sector_id" in p["field_decisions"]["conflicts"]
        assert any("sector_id differs" in b for b in p["execution"]["blockers"])


def test_merge_candidates_are_only_same_sector_same_country_exact_name_pairs(plan, snapshot):
    rows = {c["company_id"]: c for c in snapshot["companies"]}
    cands = [p for p in pairs(plan) if p["recommendation"] == "merge_candidate"]
    assert cands
    for p in cands:
        a, b = (rows[i] for i in p["pair"])
        assert a["sector_id"] == b["sector_id"] and a["country"] == b["country"]
        assert p["name_match"] == "exact_name" and p["category"] == "likely_duplicate"
    same = [p for p in pairs(plan) if rows[p["pair"][0]]["sector_id"] == rows[p["pair"][1]]["sector_id"]
            and p["name_match"] == "exact_name"]
    assert {p["recommendation"] for p in same} <= {"merge_candidate", "keep_separate"}


def test_uncertain_cases_are_queued_for_manual_review(plan):
    s = plan["summary"]
    assert s["manual_review_queue"] == sum(1 for p in pairs(plan) if p["recommendation"] in ("manual_review", "link_multi_sector")
                                           or p["review_level"] == "elevated")
    for p in pairs(plan):
        if p["recommendation"] == "manual_review":
            assert p["review_level"] == "elevated" and p["confidence"] in ("low", "medium")
        if p["confidence"] == "low":
            assert p["recommendation"] != "merge_candidate"


def test_legacy_foreign_key_mover_gaps_are_quantified_from_the_real_data(plan, snapshot):
    legacy_tables = {"company_profile", "ai_insights", "contacts", "projects", "meetings", "documents", "opportunities"}
    p0 = pairs(plan)[0]
    assert set(p0["dependents"]["legacy_tables_that_do_not_exist"]) == {"meetings", "documents", "opportunities"}
    total = 0
    for p in pairs(plan):
        dup = snapshot["views"][str(p["proposed_duplicate_id"])]["dependents"]
        expected = sum(n for t, n in dup.items() if t not in legacy_tables)
        assert p["dependents"]["rows_at_risk"] == expected
        total += expected
    assert total == plan["summary"]["total_rows_at_risk_with_legacy_mover"] > 0
    assert plan["summary"]["pairs_where_legacy_mover_would_lose_rows"] > 0


def test_one_row_per_company_tables_are_flagged_as_manual_merges(plan):
    flagged = 0
    for p in pairs(plan):
        for t in p["dependents"]["tables"]:
            if t["table"] == "lead_status" and t["master_rows"] and t["duplicate_rows"]:
                assert t["action"] == "merge_rows_manually" and "two" in " ".join(t["notes"])
                flagged += 1
    assert flagged > 90                      # every company has one lead_status row, so nearly every pair


def test_golden_record_uses_the_unchanged_legacy_scorer(plan):
    for g in plan["groups"]:
        scores = g["golden_record"]["legacy_scores"]
        best = max(scores.values())
        assert scores[g["golden_record"]["master_id"]] == best
        assert g["golden_record"]["master_id"] == min(i for i, sc in scores.items() if sc == best)   # tie -> lowest id
    for p in pairs(plan):
        assert p["legacy_similarity"] >= 0.9 or p["name_match"] == "near_name"


def test_plan_is_deterministic_and_renders_a_reviewable_report(legacy_env, snapshot, plan, tmp_path):
    again = mp.MergePlanner(InMemoryMergeSource(snapshot)).build()
    a, b = copy.deepcopy(plan), copy.deepcopy(again)
    a.pop("generated_at"), b.pop("generated_at")
    assert a == b
    jp, md = mp.write_report(plan, str(tmp_path))
    text = pathlib.Path(md).read_text(encoding="utf-8")
    assert "no merge, move or delete was executed" in text and "## Global blockers" in text
    assert text.count("### Pair") == 101 and "Golden Record: **#" in text
    assert json.loads(pathlib.Path(jp).read_text(encoding="utf-8"))["summary"]["groups"] == 79


def test_snapshot_contains_no_personal_contact_data(snapshot):
    blob = json.dumps(snapshot)
    assert "contact_name" not in blob and "@" not in blob


# ============================================================ crafted edge cases
BASE = {"company_id": 0, "company_name": "Acme Components", "sector_id": 1, "sector": None, "sector_name": "Electronics",
        "country": "India", "website": None, "linkedin_url": None, "revenue": None, "number_of_plants": None,
        "company_type": None, "priority": "Medium", "enrichment_status": "Pending", "profile": None, "insights_count": 0,
        "products": [], "components": [], "contact_fingerprints": [],
        "dependents": {t: 0 for t in mp.COMPANY_FK_TABLES}}


def co(cid, **kw):
    v = copy.deepcopy(BASE)
    v.update(company_id=cid, **{k: x for k, x in kw.items() if k != "deps"})
    v["dependents"].update(kw.get("deps", {}))
    return v


def plan_of(*views, legacy=None):
    data = {"companies": [{k: v.get(k) for k in ("company_id", "company_name", "sector_id", "country", "website")}
                          for v in views], "views": {str(v["company_id"]): v for v in views}}
    return mp.MergePlanner(InMemoryMergeSource(data), legacy=legacy).build()


def only_pair(plan):
    assert len(pairs(plan)) == 1
    return pairs(plan)[0]


def test_missing_fields_are_filled_from_the_duplicate_and_lists_are_unioned(legacy_env):
    a = co(1, website=None, revenue="~Rs 100 Cr", components=["PCBAs"],
           profile={"products": ["x"], "company_summary": "short"})
    b = co(2, website="https://acme.example", revenue=None, components=["PCBAs", "Relays"],
           profile={"products": ["y"], "company_summary": None, "headquarters": "Pune"})
    p = only_pair(plan_of(a, b))
    assert p["category"] == "likely_duplicate" and p["recommendation"] == "merge_candidate"
    by = {d["field"]: d for d in p["field_decisions"]["companies"] + p["field_decisions"]["company_profile"]}
    assert by["website"]["decision"] == "take_duplicate_value" and by["revenue"]["decision"] == "keep_master_value"
    assert by["products"]["decision"] == "union" and by["products"]["result"] == ["x", "y"]
    assert by["headquarters"]["decision"] == "take_duplicate_value"
    assert p["field_decisions"]["conflicts"] == []


def test_conflicting_values_need_a_human_and_expose_the_legacy_longer_text_rule(legacy_env):
    a = co(1, revenue="~Rs 100 Cr", number_of_plants=3)
    b = co(2, revenue="~Rs 100 Cr (estimated, FY24 annual report)", number_of_plants=5)
    p = only_pair(plan_of(a, b))
    by = {d["field"]: d for d in p["field_decisions"]["companies"]}
    assert set(p["field_decisions"]["conflicts"]) == {"revenue", "number_of_plants"}
    assert "LONGER text" in by["revenue"]["legacy_rule"] and by["revenue"]["legacy_default"].startswith("~Rs 100 Cr (est")
    assert by["number_of_plants"]["legacy_default"] == 3                      # legacy keeps master for numbers
    assert by["enrichment_status"]["decision"] == "identical"


def test_operational_fields_do_not_create_review_work(legacy_env):
    p = only_pair(plan_of(co(1, enrichment_status="Pending"), co(2, enrichment_status="Completed")))
    assert p["field_decisions"]["conflicts"] == []


def test_different_sector_with_matching_website_is_linked_not_merged(legacy_env):
    p = only_pair(plan_of(co(1, sector_id=1, website="https://www.acme.example"),
                          co(2, sector_id=5, sector_name="Solar", website="https://acme.example/en")))
    assert p["category"] == "same_name_different_sector" and p["recommendation"] == "link_multi_sector"
    assert p["confidence"] == "medium" and any("single-valued" in r for r in p["rationale"])
    assert p["review_level"] == "standard" and "sector_id" in p["field_decisions"]["conflicts"]


def test_different_sector_without_shared_identifiers_is_manual_review(legacy_env):
    p = only_pair(plan_of(co(1, sector_id=1), co(2, sector_id=5)))
    assert p["recommendation"] == "manual_review" and p["confidence"] == "low" and p["review_level"] == "elevated"


def test_shared_contact_fingerprint_is_evidence_for_the_same_organisation(legacy_env):
    p = only_pair(plan_of(co(1, sector_id=1, contact_fingerprints=["abc"]), co(2, sector_id=5, contact_fingerprints=["abc"])))
    assert p["recommendation"] == "link_multi_sector" and p["signals"]["shared_contacts"] == 1


def test_conflicting_websites_or_countries_mean_distinct_entities(legacy_env):
    a = only_pair(plan_of(co(1, website="https://acme.example"), co(2, website="https://other.example")))
    assert a["category"] == "distinct_entities_likely" and a["recommendation"] == "keep_separate"
    b = only_pair(plan_of(co(1, country="India"), co(2, country="France")))
    assert b["recommendation"] == "keep_separate" and "countries differ" in b["rationale"]


def test_same_sector_duplicate_with_disjoint_catalogues_is_elevated_for_review(legacy_env):
    p = only_pair(plan_of(co(1, components=["A"], products=["P"]), co(2, components=["B"], products=["Q"])))
    assert p["recommendation"] == "merge_candidate" and p["review_level"] == "elevated"
    assert any("different divisions" in r for r in p["rationale"])


def test_near_name_matches_are_always_manual_review(legacy_env):
    p = only_pair(plan_of(co(1, company_name="Accord Software and Systems"), co(2, company_name="Accord Softwares and Systems")))
    assert p["name_match"] == "near_name" and p["recommendation"] == "manual_review"


def test_words_are_not_mangled_into_false_duplicates(legacy_env):
    plan = plan_of(co(1, company_name="Lincoln Electric"), co(2, company_name="Ln Electric"), co(3, company_name="Unrelated Corp"))
    assert pairs(plan) == []                       # the legacy substring normaliser would equate Lincoln and Ln


def test_dependents_are_tallied_and_hazards_named(legacy_env):
    a = co(1, profile={"headquarters": "Pune"}, products=["p1"], contact_fingerprints=["x"],
           deps={"company_profile": 1, "lead_status": 1, "products": 1, "contacts": 1})
    b = co(2, profile={"headquarters": "Pune"}, products=["P1", "p2"], contact_fingerprints=["x"],
           deps={"company_profile": 1, "lead_status": 1, "products": 2, "contacts": 1, "company_suppliers": 3, "news": 2})
    p = only_pair(plan_of(a, b))
    t = {x["table"]: x for x in p["dependents"]["tables"]}
    assert t["company_profile"]["action"] == "merge_rows_manually" and t["lead_status"]["action"] == "merge_rows_manually"
    assert t["contacts"]["action"] == "move_then_deduplicate" and t["products"]["action"] == "move_then_deduplicate"
    assert t["tasks"]["action"] == "nothing_to_move"
    assert t["company_suppliers"]["handled_by_legacy_mover"] is False and t["contacts"]["handled_by_legacy_mover"] is True
    assert p["dependents"]["rows_at_risk"] == 2 + 1 + 3 + 2          # products + lead_status + company_suppliers + news
    assert any("company_suppliers: 3" in b for b in p["dependents"]["blockers"])


def test_three_way_group_picks_one_golden_record_by_legacy_score(legacy_env):
    rich = co(3, website="https://acme.example", linkedin_url="https://linkedin.com/company/acme", revenue="x",
              profile={"company_summary": "s", "employee_count": "10"}, insights_count=2)
    plan = plan_of(co(1), co(2), rich)
    assert len(plan["groups"]) == 1 and len(pairs(plan)) == 3
    assert plan["groups"][0]["golden_record"]["master_id"] == 3
    assert all(p["proposed_master_id"] == 3 for p in pairs(plan) if 3 in p["pair"])


def test_sparse_views_with_missing_keys_do_not_crash(legacy_env):
    bare = {"company_id": 1, "company_name": "Zed Works", "sector_id": 1, "country": None, "website": None}
    bare2 = {**bare, "company_id": 2}
    p = only_pair(plan_of(bare, bare2))
    assert p["signals"]["same_country"] is None and p["recommendation"] == "merge_candidate"


# ===================================================================== safety
def test_data_sources_expose_no_write_operations():
    for cls in (InMemoryMergeSource, SupabaseMergeSource):
        public = {n for n in dir(cls) if not n.startswith("_")}
        assert not any(w in n for n in public for w in ("insert", "update", "delete", "merge", "upsert", "write"))


def test_planner_and_sources_never_call_write_apis_or_legacy_mergers():
    text = "".join(pathlib.Path(f).read_text(encoding="utf-8") for f in ("agentic/merge_plan.py", "agentic/merge_sources.py"))
    for forbidden in (".insert(", ".update(", ".delete(", ".upsert(", "merge_all", "resolve(", "move_company", "merge_duplicates"):
        assert forbidden not in text, forbidden


def test_supabase_source_only_issues_select_requests():
    calls = []

    class Query:
        def __init__(self, table):
            self.table = table

        def __getattr__(self, name):
            if name in ("insert", "update", "delete", "upsert", "rpc"):
                raise AssertionError(f"write API used: {name}")
            calls.append(name)
            return lambda *a, **k: self

        def execute(self):
            rows = {"companies": [{"company_id": 1, "company_name": "A Co", "sector_id": 1, "country": "India"}],
                    "sectors": [{"sector_id": 1, "sector": "Electronics"}],
                    "contacts": [{"company_id": 1, "contact_name": "Some Person"}]}.get(self.table, [])
            return types.SimpleNamespace(data=rows)

    client = types.SimpleNamespace(table=lambda t: Query(t))
    src = SupabaseMergeSource(client)
    view = src.view(1)
    assert set(calls) <= {"select", "range"} and view["dependents"]["contacts"] == 1
    assert "Some Person" not in json.dumps(view) and len(view["contact_fingerprints"][0]) == 10

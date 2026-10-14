"""Persistence mapping for an existing company: which table holds which enriched field, additive-only writes to the
core `companies` row, one reusable `company_profile` row per company, and sourced provenance in `ai_insights`.
Offline only: in-memory commit repository, the real schema introspection, scripted models, no network."""

import pytest

from agentic import evidence as ev
from agentic import apply_validated as av
from agentic import fields as fl
from agentic import write_gate as wg
from agentic.agents.company import process_company
from agentic.commit import InMemoryCommitRepo
from agentic.persistence.store import Store
from agentic.proposals import ProposalService
from agentic.schema_contract import SchemaContract
from tests.agentic.fieldllm import FieldLLM
from tests.agentic.test_scoped_enrichment import CountingRepo, REAL_SCHEMA, apply, cs, item, run, tables

SCHEMA = SchemaContract.from_file(REAL_SCHEMA)


def ops_of(commit, table=None, kind=None):
    return [o for o in commit.ops if (table is None or o[1] == table) and (kind is None or o[0] == kind)]


# ============================================================================================ canonical destinations
def test_every_field_has_one_canonical_destination_that_exists_in_the_real_schema():
    dest = fl.canonical_destinations()
    assert set(dest) == set(fl.BY_KEY)
    assert {k for k, v in dest.items() if v["layer"] == "core"} == {"official_website", "linkedin_url"}
    assert dest["official_website"]["target"] == "companies.website" and dest["linkedin_url"]["target"] == "companies.linkedin_url"
    assert {k for k, v in dest.items() if v["layer"] == "insight"} == {"purchasing_category", "purchased_component", "process_note"}
    assert {k for k, v in dest.items() if v["layer"] == "other_entity"} == {"procurement_contacts", "suppliers"}
    for key, v in dest.items():
        if v["layer"] in ("core", "extended"):
            table, column = v["target"].split(".")
            assert (table, column) in SCHEMA.introspected, key                      # the column exists in the database
            assert column in wg.COLUMNS[table], key                                 # and the write gate allows it
            assert v["provenance"] == "ai_insights"
    assert dest["annual_revenue"]["target"] == "company_profile.annual_revenue"      # never the legacy companies.revenue
    assert dest["business_overview"]["target"] == "company_profile.company_summary"
    assert dest["industry"]["target"] == "company_profile.industry"                  # companies.sector_id stays curated
    assert dest["lead_score"]["layer"] == "report_only"
    assert "website_domain" in fl.DERIVED_NOT_WRITTEN
    assert SCHEMA.is_unique("company_profile", "company_id")                         # one profile per company, enforced by the DB


# ============================================================================================ core companies row
def test_the_core_companies_row_is_filled_only_where_empty_and_nothing_else_changes():
    r, out = run(FieldLLM())
    commit = InMemoryCommitRepo(tables(profile=None, website=None))
    before = dict(commit.tables["companies"][0])
    res = apply(out, commit, schema=SCHEMA)
    row = commit.tables["companies"][0]
    assert row["website"] and len(commit.tables["companies"]) == 1                    # filled, and no duplicate company
    changed = {k for k in row if row[k] != before.get(k)}
    assert changed == {"website"}                                                    # nothing else on the core row moved
    assert ops_of(commit, "companies") == [o for o in commit.ops if o[0] == "update" and o[1] == "companies"]
    assert not ops_of(commit, "companies", "insert")


def test_a_populated_core_value_is_never_overwritten_even_if_it_appeared_after_the_review():
    r, out = run(FieldLLM())                                                         # reviewed while the CRM website was empty
    commit = InMemoryCommitRepo(tables(profile=None, website="https://crm.example/"))
    res = apply(out, commit, schema=SCHEMA)
    assert commit.tables["companies"][0]["website"] == "https://crm.example/"
    assert any(s["field"] == "official_website" for s in res["stale_items"]) and not ops_of(commit, "companies")


# ============================================================================================ one reusable profile row
def test_an_existing_profile_row_is_reused_and_only_its_empty_fields_are_filled():
    r, out = run(FieldLLM())
    commit = InMemoryCommitRepo(tables(profile={"company_summary": "Legacy summary", "headquarters": None}))
    res = apply(out, commit, schema=SCHEMA)
    profiles = commit.tables["company_profile"]
    assert len(profiles) == 1 and profiles[0]["profile_id"] == 1                      # the same row, not a second one
    assert profiles[0]["company_summary"] == "Legacy summary"                         # populated: preserved
    assert profiles[0]["headquarters"]                                                # empty: filled
    assert not ops_of(commit, "company_profile", "insert") and ops_of(commit, "company_profile", "update")
    assert any(s["field"] == "business_overview" for s in res["stale_items"])


def test_a_missing_profile_is_inserted_once_with_the_real_not_null_columns_satisfied():
    r, out = run(FieldLLM())
    commit = InMemoryCommitRepo(tables(profile=None))
    res = apply(out, commit, schema=SCHEMA)
    assert all(p["applied"] for p in res["proposals"]), res["proposals"]
    assert len(commit.tables["company_profile"]) == 1 and commit.tables["company_profile"][0]["company_id"] == 7


def test_running_the_same_enrichment_again_creates_no_second_profile_and_no_second_provenance():
    r, out = run(FieldLLM())
    commit = InMemoryCommitRepo(tables(profile=None))
    apply(out, commit, schema=SCHEMA)
    n_profile, n_insights, n_ops = len(commit.tables["company_profile"]), len(commit.tables["ai_insights"]), len(commit.ops)
    again = apply(out, commit, schema=SCHEMA, store=Store(":memory:"))               # a fresh store: nothing remembered locally
    assert again["writes"] == 0 and len(commit.ops) == n_ops
    assert (len(commit.tables["company_profile"]), len(commit.tables["ai_insights"])) == (n_profile, n_insights)


def test_a_profile_that_appears_between_proposal_and_apply_is_never_duplicated():
    r, out = run(FieldLLM())
    commit = InMemoryCommitRepo(tables(profile=None))
    svc = ProposalService(Store(":memory:"), commit, schema=SCHEMA)
    sel = av.select_items(out["report"]["change_set"], 7, identity_verified=True)
    res = svc.propose_from_change_set(out, out["report"]["change_set"], sel.item_ids)
    prop = next(p for p in res["proposals"] if any(o["table"] == "company_profile" and o["op"] == "insert" for o in p["operations"]))
    commit.tables["company_profile"].append({"profile_id": 99, "company_id": 7})   # another process inserted it meanwhile
    assert svc.decide(prop["proposal_id"], True, by="tester")["ok"]
    done = svc.apply(prop["proposal_id"], actor="tester")
    assert done["ok"] is False and done["error"] == "profile_already_exists"
    assert [p["profile_id"] for p in commit.tables["company_profile"]] == [99]


def test_provenance_is_not_duplicated_when_a_value_is_refilled():
    r, out = run(FieldLLM())
    commit = InMemoryCommitRepo(tables(profile=None))
    apply(out, commit, schema=SCHEMA)
    insights = [dict(i) for i in commit.tables["ai_insights"]]
    commit.tables["company_profile"][0].update(headquarters=None, company_summary=None)    # values cleared, provenance kept
    commit.tables["companies"][0]["website"] = None
    res = apply(out, commit, schema=SCHEMA, store=Store(":memory:"))
    assert res["writes"] > 0 and commit.tables["company_profile"][0]["headquarters"]    # the empty fields are filled again
    assert commit.tables["ai_insights"] == insights                                     # but no second provenance row


# ============================================================================================ provenance linkage
def test_every_provenance_row_names_its_field_company_source_and_exact_quote():
    r, out = run(FieldLLM())
    commit = InMemoryCommitRepo(tables(profile={"headquarters": None}))
    apply(out, commit, schema=SCHEMA)
    rows = commit.tables["ai_insights"]
    assert rows
    texts = [d["text"] for d in out["ledger"].values()]
    written = {**commit.tables["companies"][0], **commit.tables["company_profile"][0]}
    for i in rows:
        assert i["company_id"] == 7 and i["generated_by"].startswith("agentic-") and i["status"] == "Active"
        procurement = i["insight_type"].startswith("Procurement: ")           # insight-only findings: the row IS the record
        fld = i["insight_type"].removeprefix("Procurement: " if procurement else "Verified ")
        assert i["insight_title"].startswith(f"{fld}: ") and fld in fl.BY_KEY
        assert (fl.canonical_destinations()[fld]["layer"] == "insight") == procurement
        quote, _, tail = i["insight"].rpartition(" (source: ")
        assert tail == i["source_url"] + ")" and any(quote in t for t in texts)              # the exact quote, from a real page
        assert i["source_url"] in {d["url"] for d in out["ledger"].values()}
        if not procurement:
            assert written.get(fl.BY_KEY[fld].column)                                       # the field it documents was written


# ============================================================================================ extra validation
def test_quotes_with_encoding_damage_and_unreviewed_urls_are_not_stored():
    damaged = item(evidence=[{"url": "https://a.example/", "snippet": "The company is headquartered in Pune�."}])
    sel = av.select_items(cs(damaged), 7, identity_verified=True)
    assert sel.item_ids == [] and sel.skipped[0]["reason"] == "evidence_text_encoding_damaged"
    assert av.select_items(cs(item()), 7, identity_verified=True).item_ids == ["i1"]


def test_a_linkedin_page_must_carry_the_company_name():
    ledger = {}
    ev.add_doc(ledger, url="https://www.apollo.example/", title="Apollo", tool="fetch_page", call_id="c1", max_docs=10,
               max_chars=5000, text="Apollo Micro Systems. Follow us: https://www.linkedin.com/company/%s today.")
    state = {"ledger": ledger, "claims": {}, "subject": {"name": "Apollo Micro Systems", "existing": None,
             "identity": {"verified": True, "domain": "apollo.example", "website": "https://www.apollo.example/"}}}

    def claim(slug):
        url = f"https://www.linkedin.com/company/{slug}"
        doc = next(iter(ledger.values()))
        ledger[next(iter(ledger))]["text"] = f"Apollo Micro Systems. Follow us: {url} today."
        return {"identity_website": "https://www.apollo.example/", "claims": [{"field": "linkedin_url", "value": url,
                "evidence": [{"url": doc["url"], "snippet": f"Follow us: {url} today."}]}]}
    ok = process_company(state, claim("apollo-micro-systems"), None)
    assert [c["field"] for c in ok.accepted] == ["linkedin_url"]
    bad = process_company(state, claim("acme-textiles"), None)
    assert not [c for c in bad.accepted if c["field"] == "linkedin_url"]
    assert "linkedin_slug_does_not_match_company" in [x for r in bad.rejected for x in r["reasons"]]
    assert av.url_shape_problem("linkedin_url", "https://www.linkedin.com/in/someone") == "not_a_linkedin_company_page"


# ============================================================================================ read budget of the write phase
def test_the_write_phase_reads_each_row_once_not_once_per_item():
    r, out = run(FieldLLM())
    CountingRepo.reads = 0
    repo = CountingRepo(tables(profile={"headquarters": None}))
    res = apply(out, repo, schema=SCHEMA)
    assert res["writes"] > 0
    assert CountingRepo.reads <= 11, CountingRepo.reads                              # was 12 for the same run before the change

"""Approval-gated proposals for contacts, supplier relationships and procurement URLs."""

from agentic.commit import InMemoryCommitRepo
from tests.agentic.support import make_runner

FULL = "comprehensive"


def tables(person=None, with_contact=True, suppliers=None):
    contacts = []
    if with_contact:
        contacts = [{"contact_id": 1, "company_id": 7, "contact_name": person or "Somebody Else",
                     "designation": "Purchase Manager", "source_url": None, "confidence_score": None}]
    return InMemoryCommitRepo({"companies": [{"company_id": 7, "company_name": "Apollo Micro Systems",
                                              "website": None, "linkedin_url": None}],
                               "company_profile": [], "contacts": contacts, "suppliers": suppliers or [],
                               "company_suppliers": []})


def person_of(runner):
    return runner.world["facts"]["person"]


def pending(out):
    return {p["kind"]: p for p in out["proposals"] if p["status"] == "pending"}


def test_each_discovery_becomes_a_separate_pending_proposal_and_nothing_is_written():
    r = make_runner(commit_repo=InMemoryCommitRepo())
    r.deps.commit_repo.tables = tables().tables
    out = r.start("Apollo Micro Systems", FULL, propose_changes=True)
    kinds = pending(out)
    assert {"update_company_fields", "insert_contact", "insert_supplier_relationship"} <= set(kinds)
    assert r.deps.commit_repo.ops == [] and out["status"] == "awaiting_approval"


def test_existing_unverified_contact_gets_provenance_only_after_overwrite_approval():
    r = make_runner(commit_repo=InMemoryCommitRepo())
    r.deps.commit_repo.tables = tables(person=None).tables
    r.deps.commit_repo.tables["contacts"][0]["contact_name"] = person_of(r)       # same person as found publicly
    out = r.start("Apollo Micro Systems", FULL, propose_changes=True)
    p = pending(out)["verify_existing_contact"]
    assert p["overwrites"] == ["contacts.designation"]                              # 'Purchase Manager' -> 'Head of Procurement'
    svc = r.service()
    assert svc.decide(p["proposal_id"], True, by="sayli")["error"] == "overwrite_requires_explicit_approval"
    assert svc.decide(p["proposal_id"], True, by="sayli", approved_overwrite=True)["ok"]
    assert svc.apply(p["proposal_id"])["status"] == "applied"
    row = r.deps.commit_repo.tables["contacts"][0]
    assert row["source_url"].startswith("https://") and row["confidence_score"] > 0 and row["last_verified"]
    assert row["contact_name"] == person_of(r)                                      # identity of the record untouched
    assert row["designation"] == "Head of Procurement"
    assert [o for o in r.deps.commit_repo.ops if o[1] == "contacts"][0][0] == "update"


def test_new_public_contact_is_inserted_with_source_and_never_with_a_guessed_email():
    r = make_runner(commit_repo=InMemoryCommitRepo())
    r.deps.commit_repo.tables = tables().tables
    out = r.start("Apollo Micro Systems", FULL, propose_changes=True)
    p = pending(out)["insert_contact"]
    svc = r.service()
    assert svc.decide(p["proposal_id"], True, by="sayli")["ok"] and svc.apply(p["proposal_id"])["status"] == "applied"
    new = [c for c in r.deps.commit_repo.tables["contacts"] if c["contact_name"] == person_of(r)][0]
    assert new["company_id"] == 7 and new["source_url"] and new["contact_type"] == "Public source"
    assert new["decision_maker"] is True and new["last_verified"]
    snippets = " ".join(e["snippet"] for e in r.store.get_proposal(p["proposal_id"])["change"]["evidence"])
    if new.get("email"):
        assert new["email"] in snippets                                              # only if literally on the page


def test_supplier_relationship_writes_supplier_link_and_provenance_and_reuses_existing_suppliers():
    r = make_runner(commit_repo=InMemoryCommitRepo())
    r.deps.commit_repo.tables = tables().tables
    sup_name = r.world["facts"]["supplier"]
    r.deps.commit_repo.tables["suppliers"] = [{"supplier_id": 55, "supplier_name": sup_name}]      # already known
    out = r.start("Apollo Micro Systems", FULL, propose_changes=True)
    p = pending(out)["insert_supplier_relationship"]
    svc = r.service()
    assert svc.decide(p["proposal_id"], True, by="sayli")["ok"]
    res = svc.apply(p["proposal_id"])
    assert res["status"] == "applied" and {"op": "reuse", "table": "suppliers"} in res["operations"]
    repo = r.deps.commit_repo
    assert len(repo.tables["suppliers"]) == 1                                         # no duplicate supplier
    link = repo.tables["company_suppliers"][0]
    assert link["supplier_id"] == 55 and link["company_id"] == 7 and link["discovered_by"] == "agentic-supplier"
    assert "source:" in link["remarks"] and link["confidence_score"] > 0
    ins = [i for i in repo.tables["ai_insights"] if i["insight_type"] == "Supplier relationship"][0]
    assert ins["source_url"].startswith("https://")


def test_procurement_urls_flow_into_the_profile_proposal_with_evidence():
    r = make_runner(commit_repo=InMemoryCommitRepo())
    r.deps.commit_repo.tables = tables().tables
    out = r.start("Apollo Micro Systems", FULL, propose_changes=True)
    p = r.store.get_proposal(pending(out)["update_company_fields"]["proposal_id"])["change"]
    prof = [o for o in p["operations"] if o["table"] == "company_profile"][0]["fields"]
    assert prof["supplier_portal"].startswith("https://") and prof["procurement_page"].startswith("https://")
    assert any(o["fields"].get("insight_type") == "Verified supplier_portal" for o in p["operations"])


def test_for_a_company_not_in_the_database_contact_and_supplier_changes_are_blocked_not_pending():
    r = make_runner(name="Zenith Quantum Forge", commit_repo=InMemoryCommitRepo({"companies": []}))
    out = r.start("Zenith Quantum Forge", FULL, propose_changes=True)
    by_kind = {p["kind"]: p for p in out["proposals"]}
    assert by_kind["insert_contact"]["status"] == "blocked" and by_kind["insert_supplier_relationship"]["status"] == "blocked"
    assert by_kind["create_company"]["status"] == "pending" and by_kind["create_company"]["required_input"] == ["sector_id"]
    svc = r.service()
    assert svc.decide(by_kind["insert_contact"]["proposal_id"], True, by="x")["error"] == "already_blocked"
    assert r.deps.commit_repo.ops == []


def test_rejecting_everything_leaves_the_database_untouched():
    r = make_runner(commit_repo=InMemoryCommitRepo())
    r.deps.commit_repo.tables = tables().tables
    out = r.start("Apollo Micro Systems", FULL, propose_changes=True)
    decisions = {p["proposal_id"]: {"approve": False, "by": "sayli", "reason": "not now"}
                 for p in out["proposals"] if p["status"] == "pending"}
    res = r.resume(out["run_id"], decisions)
    assert r.deps.commit_repo.ops == [] and all(p["status"] == "rejected" for p in res["proposals"])
    assert res["status"] == "completed"

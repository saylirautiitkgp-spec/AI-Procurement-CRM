"""Checkpoint/resume + approval-gated proposed changes + write safeguards (offline)."""

import sqlite3

import pytest
from langgraph.checkpoint.sqlite import SqliteSaver

from agentic import write_gate as wg
from agentic.commit import InMemoryCommitRepo
from agentic.persistence.store import Store
from tests.agentic.support import make_runner

PROFILE = "company profile"


def db(website=None):
    return InMemoryCommitRepo({"companies": [{"company_id": 7, "company_name": "Apollo Micro Systems",
                                              "website": website, "linkedin_url": None}],
                               "company_profile": []})


def pending(result):
    return [p for p in result["proposals"] if p["status"] == "pending"]


# ------------------------------------------------------------- research only
def test_research_only_run_never_touches_the_database_and_needs_no_approval():
    repo = db()
    out = make_runner(commit_repo=repo).start("Apollo Micro Systems", PROFILE)      # propose_changes defaults off
    assert out["status"] == "completed" and out["proposals"] == []
    assert repo.ops == [] and "__interrupt__" not in out


def test_new_company_research_has_no_database_dependency_at_all():
    out = make_runner(name="Zenith Quantum Forge", commit_repo=None).start("Zenith Quantum Forge", PROFILE)
    assert out["status"] == "completed" and out["report"]["database_match"]["status"] == "not_found"


# ------------------------------------------------------------ approval flow
def test_run_pauses_for_approval_and_nothing_is_written_until_approved():
    repo = db()
    r = make_runner(commit_repo=repo)
    out = r.start("Apollo Micro Systems", PROFILE, propose_changes=True)
    assert "__interrupt__" in out and out["status"] == "awaiting_approval"
    assert repo.ops == []                                                    # still nothing written
    p = pending(out)[0]
    assert p["kind"] == "update_company_fields"
    row = r.store.get_proposal(p["proposal_id"])
    assert row["status"] == "pending" and row["validation"]["ok"]
    tr = r.store.get_run(out["run_id"])
    assert tr["status"] == "awaiting_approval" and tr["tracker"]["database_change"] == "awaiting_approval"


def test_approved_proposal_is_applied_with_provenance_and_audit_trail():
    repo = db()
    r = make_runner(commit_repo=repo)
    out = r.start("Apollo Micro Systems", PROFILE, propose_changes=True)
    pid = pending(out)[0]["proposal_id"]
    res = r.resume(out["run_id"], {pid: {"approve": True, "by": "sayli", "reason": "looks right"}})
    assert res["status"] == "completed"
    assert [p["status"] for p in res["proposals"]] == ["applied"]
    tables = [(o[0], o[1]) for o in repo.ops]
    assert ("update", "companies") in tables and ("insert", "company_profile") in tables
    insights = [o[2] for o in repo.ops if o[1] == "ai_insights"]
    assert insights and all(i["source_url"].startswith("https://") and i["confidence_score"] > 0 for i in insights)
    assert all(i["company_id"] == 7 for i in insights)
    actions = [(a["action"], a["actor"]) for a in r.store.audit_entries(out["run_id"])]
    assert ("proposed", "system") in actions and ("approved", "sayli") in actions and ("applied", "commit_node") in actions
    # unverified legacy values were never copied into the profile
    prof = next(o[2] for o in repo.ops if o[1] == "company_profile")
    assert "annual_revenue" not in prof


def test_rejection_writes_nothing():
    repo = db()
    r = make_runner(commit_repo=repo)
    out = r.start("Apollo Micro Systems", PROFILE, propose_changes=True)
    pid = pending(out)[0]["proposal_id"]
    res = r.resume(out["run_id"], {pid: {"approve": False, "by": "sayli", "reason": "wrong company"}})
    assert repo.ops == [] and res["proposals"][0]["status"] == "rejected" and res["status"] == "completed"


def test_approval_requires_an_identified_approver():
    repo = db()
    r = make_runner(commit_repo=repo)
    out = r.start("Apollo Micro Systems", PROFILE, propose_changes=True)
    pid = pending(out)[0]["proposal_id"]
    res = r.resume(out["run_id"], {pid: {"approve": True, "by": "  "}})
    assert repo.ops == [] and res["proposals"][0]["status"] == "pending"


# ------------------------------------------------------- checkpoint / resume
def test_run_survives_a_process_restart_while_awaiting_approval(tmp_path):
    ck, dbfile = str(tmp_path / "ck.sqlite"), str(tmp_path / "app.sqlite")
    repo = db()
    r1 = make_runner(commit_repo=repo, store=Store(dbfile),
                     checkpointer=SqliteSaver(sqlite3.connect(ck, check_same_thread=False)))
    out = r1.start("Apollo Micro Systems", PROFILE, propose_changes=True, run_id="crashrun")
    pid = pending(out)[0]["proposal_id"]
    del r1                                                           # "process dies"
    r2 = make_runner(commit_repo=repo, store=Store(dbfile),         # fresh graph, fresh deps, same files
                     checkpointer=SqliteSaver(sqlite3.connect(ck, check_same_thread=False)))
    assert r2.state("crashrun")["status"] == "awaiting_approval"
    res = r2.resume("crashrun", {pid: {"approve": True, "by": "sayli"}})
    assert res["proposals"][0]["status"] == "applied" and ("update", "companies") in [(o[0], o[1]) for o in repo.ops]


def test_checkpoint_holds_the_full_trace_for_inspection(tmp_path):
    r = make_runner(checkpointer=SqliteSaver(sqlite3.connect(str(tmp_path / "c.sqlite"), check_same_thread=False)))
    out = r.start("Apollo Micro Systems", PROFILE, run_id="inspect")
    assert [e["type"] for e in r.state("inspect")["trace"]] == [e["type"] for e in out["trace"]]


# ------------------------------------------------------ overwrite protection
def test_overwriting_a_populated_field_needs_explicit_approval():
    repo = db(website="http://old-site.example")
    r = make_runner(commit_repo=repo)
    out = r.start("Apollo Micro Systems", PROFILE, propose_changes=True)
    p = pending(out)[0]
    assert "companies.website" in p["overwrites"]
    res = r.resume(out["run_id"], {p["proposal_id"]: {"approve": True, "by": "sayli"}})
    assert res["proposals"][0]["status"] == "pending" and repo.ops == []          # refused, still pending
    assert repo.tables["companies"][0]["website"] == "http://old-site.example"
    res2 = r.service().decide(p["proposal_id"], True, by="sayli", approved_overwrite=True)
    assert res2["ok"]
    assert r.service().apply(p["proposal_id"])["status"] == "applied"
    assert repo.tables["companies"][0]["website"].startswith("https://www.apollomicrosystems")


# ----------------------------------------------------- idempotency / tampering
def test_apply_is_idempotent_and_reproposing_does_not_duplicate():
    repo = db()
    r = make_runner(commit_repo=repo)
    out = r.start("Apollo Micro Systems", PROFILE, propose_changes=True)
    pid = pending(out)[0]["proposal_id"]
    r.resume(out["run_id"], {pid: {"approve": True, "by": "sayli"}})
    n = len(repo.ops)
    assert r.service().apply(pid)["noop"] is True and len(repo.ops) == n
    assert r.propose_for_run(out["run_id"]) == []              # DB already holds the verified values
    assert len(r.store.list_proposals(out["run_id"])) == 1


def test_reproposing_before_apply_returns_the_same_proposal():
    repo = db()
    r = make_runner(commit_repo=repo)
    out = r.start("Apollo Micro Systems", PROFILE, propose_changes=True)
    pid = pending(out)[0]["proposal_id"]
    again = r.propose_for_run(out["run_id"])
    assert [a["proposal_id"] for a in again] == [pid] and len(r.store.list_proposals(out["run_id"])) == 1


def test_tampered_proposal_is_refused_at_apply_time():
    repo = db()
    r = make_runner(commit_repo=repo)
    out = r.start("Apollo Micro Systems", PROFILE, propose_changes=True)
    pid = pending(out)[0]["proposal_id"]
    assert r.service().decide(pid, True, by="sayli")["ok"]
    row = r.store.get_proposal(pid)["change"]
    row["operations"].append({"table": "contacts", "op": "insert", "fields": {"contact_name": "Injected Person"}})
    r.store.update_proposal_change(pid, row)
    res = r.service().apply(pid)
    assert res["status"] == "failed" and "content_changed" in res["error"] and repo.ops == []


def test_unapproved_proposal_cannot_be_applied():
    repo = db()
    r = make_runner(commit_repo=repo)
    out = r.start("Apollo Micro Systems", PROFILE, propose_changes=True)
    pid = pending(out)[0]["proposal_id"]
    assert r.service().apply(pid)["error"].startswith("not_approved") and repo.ops == []


def test_partial_failure_is_recorded_and_nothing_is_deleted():
    repo = db()
    repo.fail_on = ("company_profile", "insert")
    r = make_runner(commit_repo=repo)
    out = r.start("Apollo Micro Systems", PROFILE, propose_changes=True)
    pid = pending(out)[0]["proposal_id"]
    res = r.resume(out["run_id"], {pid: {"approve": True, "by": "sayli"}})
    assert res["proposals"][0]["status"] == "failed"
    failed = [a for a in r.store.audit_entries(out["run_id"]) if a["action"] == "apply_failed"][0]
    assert failed["details"]["operations_done"] and all(o[0] != "delete" for o in repo.ops)


# ----------------------------------------------------------- new companies
def test_creating_a_company_requires_a_valid_sector_chosen_by_the_approver():
    repo = InMemoryCommitRepo({"companies": [], "company_profile": []})
    r = make_runner(name="Zenith Quantum Forge", commit_repo=repo)
    out = r.start("Zenith Quantum Forge", PROFILE, propose_changes=True)
    p = pending(out)[0]
    assert p["kind"] == "create_company" and p["required_input"] == ["sector_id"]
    svc = r.service()
    assert svc.decide(p["proposal_id"], True, by="s")["error"] == "missing_required_input:sector_id"
    assert svc.decide(p["proposal_id"], True, by="s", overrides={"sector_id": 99})["error"] == "invalid_sector_id"
    assert svc.decide(p["proposal_id"], True, by="s", overrides={"sector_id": 5})["ok"]
    res = svc.apply(p["proposal_id"])
    assert res["status"] == "applied"
    new_id = res["created_company_id"]
    assert repo.tables["companies"][0]["sector_id"] == 5
    assert repo.tables["company_profile"][0]["company_id"] == new_id
    assert all(i["company_id"] == new_id for i in repo.tables["ai_insights"])


def test_existing_company_name_blocks_creation():
    repo = InMemoryCommitRepo({"companies": [{"company_id": 1, "company_name": "Zenith Quantum Forge",
                                              "sector_id": 1}], "company_profile": []})
    r = make_runner(name="Zenith Quantum Forge", commit_repo=repo, repo=make_runner().deps.repo.__class__([]))
    out = r.start("Zenith Quantum Forge", PROFILE, propose_changes=True)
    p = pending(out)[0]
    svc = r.service()
    svc.decide(p["proposal_id"], True, by="s", overrides={"sector_id": 1})
    assert svc.apply(p["proposal_id"])["error"] == "company_already_exists"


# --------------------------------------------------------- write gate (unit)
@pytest.mark.parametrize("op,err", [
    ({"table": "companies", "op": "delete", "key": {"company_id": 1}, "fields": {"x": 1}}, "operation_forbidden:delete"),
    ({"table": "companies", "op": "upsert", "fields": {"website": "https://a.example"}}, "operation_forbidden:upsert"),
    ({"table": "workflow_runs", "op": "insert", "fields": {"status": "x"}}, "table_not_writable:workflow_runs"),
    ({"table": "ai_insights", "op": "update", "key": {"company_id": 1}, "fields": {"insight": "x"}},
     "operation_not_allowed:ai_insights.update"),
    ({"table": "companies", "op": "update", "key": {"company_id": 1}, "fields": {"company_id": 9}},
     "column_not_allowed:companies.company_id"),
    ({"table": "companies", "op": "update", "key": {"company_id": 1}, "fields": {"enrichment_status": "Completed"}},
     "column_not_allowed:companies.enrichment_status"),
    ({"table": "companies", "op": "update", "key": {"company_id": 1}, "fields": {"website": "javascript:alert(1)"}},
     "invalid_value:companies.website"),
    ({"table": "contacts", "op": "insert", "fields": {"contact_name": "A", "email": "not-an-email"}},
     "invalid_value:contacts.email"),
    ({"table": "company_profile", "op": "insert", "fields": {"products": []}}, "invalid_value:company_profile.products"),
])
def test_write_gate_blocks_unsafe_operations(op, err):
    errors, _ = wg.validate_operation(op, existing={"company_id": 1})
    assert err in errors


def test_evidence_without_a_retrieved_source_is_rejected_by_the_gate():
    assert wg.validate_evidence([], {}) == ["no_evidence"]
    assert wg.validate_evidence([{"url": "https://x.example/a", "snippet": "something made up here"}], {})[0] \
        .startswith("evidence_invalid")


def test_full_five_agent_run_checkpoints_and_resumes_through_sqlite(tmp_path):
    repo = db()
    ck = str(tmp_path / "full.sqlite")
    dbfile = str(tmp_path / "app.sqlite")
    r = make_runner(commit_repo=repo, store=Store(dbfile),
                    checkpointer=SqliteSaver(sqlite3.connect(ck, check_same_thread=False)))
    out = r.start("Apollo Micro Systems", "comprehensive", propose_changes=True, run_id="full")
    assert "__interrupt__" in out and len(out["completed_agents"]) == 5
    r2 = make_runner(commit_repo=repo, store=Store(dbfile),
                     checkpointer=SqliteSaver(sqlite3.connect(ck, check_same_thread=False)))
    snapshot = r2.state("full")
    assert snapshot["status"] == "awaiting_approval" and len(snapshot["ledger"]) == len(out["ledger"])
    decisions = {p["proposal_id"]: {"approve": False, "by": "sayli"} for p in pending(out)}
    assert r2.resume("full", decisions)["status"] == "completed" and repo.ops == []

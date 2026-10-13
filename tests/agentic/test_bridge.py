"""Change set -> approval -> existing proposal/commit flow: the bridge, with every write path kept disabled.

Covers the reviewed-items-only guarantee, stale detection, overwrite protection, evidence retention, contact comparison
against existing CRM records, the revenue column decision, the schema gate, provider/budget configuration and
the API contract. Offline only: in-memory CRM and commit repositories, scripted models, no network, no keys."""

import dataclasses

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from agentic import changeset as cs_mod
from agentic import fields as fl
from agentic import fk_introspect
from agentic import service as svc_mod
from agentic import write_gate as wg
from agentic.agents import AGENT_SPECS
from agentic.backends.fixtures import FixtureFetcher, FixtureSearch
from agentic.commit import InMemoryCommitRepo
from agentic.config import Limits, Settings
from agentic.eval.wire_emulators import GroqWire
from agentic.llm import compat
from agentic.llm.base import LLMError, RetryingLLM
from agentic.llm.groq import GroqClient
from agentic.persistence.store import Store
from agentic.proposals import ProposalService
from agentic.repo import InMemoryRepo
from agentic.runtime import AgenticRunner, make_deps
from agentic.schema_contract import LIVE_READ_CONFIRMED, SchemaContract, operation_columns
from agentic.service import AgenticService, ServiceNotConfigured, build_live_service
from api.agentic_routes import router
from tests.agentic.fieldllm import FieldLLM
from tests.agentic.support import COMPANY_A, default_repo
from tests.agentic.test_fields import PAR, claim_map, crm_repo
from tests.agentic.test_hybrid import llm_events
from tests.agentic.test_prefetch import events, world

FIELDS = ["headquarters", "supplier_portal", "procurement_contacts", "suppliers"]
ALL_COLUMNS = SchemaContract((t, c) for t, cols in wg.COLUMNS.items() for c in cols)       # "introspection" of everything


def commit_tables(*, contacts=(), hq=None, website=None):
    return {"companies": [{"company_id": 7, "company_name": "Apollo Micro Systems", "website": website,
                           "linkedin_url": None, "company_type": "Private", "country": "India", "sector_id": 1}],
            "company_profile": [{"profile_id": 1, "company_id": 7, "headquarters": hq, "supplier_portal": None}],
            "contacts": [dict(c) for c in contacts], "ai_insights": [], "suppliers": [], "company_suppliers": []}


def build(commit=None, *, repo=None, schema=None, llm=None, eff=PAR, store=None):
    w = world()
    search, fetch = FixtureSearch(w), FixtureFetcher(w)
    store = store or Store(":memory:")
    deps = make_deps(llm=llm or FieldLLM(), repo=repo if repo is not None else default_repo(), search=search,
                     fetcher=fetch, store=store, settings=Settings(limits=Limits(), efficiency=eff),
                     specs=AGENT_SPECS, commit_repo=commit, sleep=lambda s: None)
    deps.schema = schema
    return AgenticRunner(deps, AGENT_SPECS), deps


def run_fields(runner, fields=FIELDS, refresh=False, run_id="r1", propose=False):
    return runner.start("Apollo Micro Systems", None, run_id=run_id, fields=fields, refresh=refresh,
                        propose_changes=propose)


def by_kind(res):
    out = {}
    for p in res["proposals"]:
        out.setdefault(p["kind"], []).append(p)
    return out


# =============================================================== the bridge: reviewed items -> pending proposals
def test_reviewed_change_set_becomes_pending_proposals_and_nothing_is_written():
    commit = InMemoryCommitRepo(commit_tables())
    runner, deps = build(commit, schema=ALL_COLUMNS)
    out = run_fields(runner)
    cs = out["report"]["change_set"]
    res = runner.propose_from_change_set("r1")
    assert commit.ops == []                                                         # propose = read-only
    kinds = by_kind(res)
    assert {"update_company_fields", "insert_contact", "insert_supplier_relationship"} <= set(kinds)
    assert all(p["status"] == "pending" and p["origin"] == "change_set" for p in res["proposals"])
    assert res["stale_items"] == [] and res["change_set_hash"] == cs["change_set_hash"]
    # every proposal states which reviewed items it implements, with old/proposed values, sources, confidence, validation
    for p in res["proposals"]:
        assert p["items"] and p["evidence"]
        for it in p["items"]:
            assert it["validation_status"] == "verified" and it["source_urls"] and it["confidence"] is not None
            assert "old_value" in it and "proposed_value" in it
    company = kinds["update_company_fields"][0]
    assert {i["field"] for i in company["items"]} == {"headquarters", "supplier_portal"}
    stored = deps.store.list_proposals("r1")
    assert {s["status"] for s in stored} == {"pending"}
    assert all(s["change"]["evidence"] and all(e["snippet"] and e["url"] for e in s["change"]["evidence"]) for s in stored)


def test_no_unreviewed_column_is_written_and_only_selected_items_are_proposed():
    commit = InMemoryCommitRepo(commit_tables())
    runner, deps = build(commit, schema=ALL_COLUMNS)
    out = run_fields(runner)
    hq = next(i for i in out["report"]["change_set"]["items"] if i["field"] == "headquarters")
    res = runner.propose_from_change_set("r1", [hq["id"]])
    assert [p["kind"] for p in res["proposals"]] == ["update_company_fields"]
    ops = deps.store.list_proposals("r1")[0]["change"]["operations"]
    written = {c for op in ops for c in op["fields"]}
    assert "headquarters" in written and "website_domain" not in written and "supplier_portal" not in written
    with pytest.raises(ValueError, match="unknown_item_ids"):
        runner.propose_from_change_set("r1", ["nope"])
    assert commit.ops == []


def test_items_whose_crm_value_changed_since_the_review_are_stale_and_not_proposed():
    commit = InMemoryCommitRepo(commit_tables())
    runner, _ = build(commit, schema=ALL_COLUMNS)
    run_fields(runner, ["headquarters"])
    commit.tables["company_profile"][0]["headquarters"] = "Pune, India"            # someone filled it after the review
    res = runner.propose_from_change_set("r1")
    assert res["proposals"] == [] and res["stale_items"][0]["reason"] == "crm_value_changed_since_review"
    assert res["stale_items"][0]["live_value"] == "Pune, India" and res["stale_items"][0]["reviewed_old_value"] is None


def test_overwrites_need_explicit_approval_and_nothing_applies_before_it():
    commit = InMemoryCommitRepo(commit_tables(hq="Pune, India"))
    repo = crm_repo(headquarters="Pune, India")
    runner, deps = build(commit, repo=repo, schema=ALL_COLUMNS)
    out = run_fields(runner, ["headquarters"], refresh=True)
    item = out["report"]["change_set"]["items"][0]
    assert item["action"] == "update_conflict" and item["old_value"] == "Pune, India"
    (prop,) = runner.propose_from_change_set("r1")["proposals"]
    assert prop["overwrites"] == ["company_profile.headquarters"]
    svc = runner.service()
    denied = svc.decide(prop["proposal_id"], True, by="alice")
    assert not denied["ok"] and denied["error"] == "overwrite_requires_explicit_approval"
    assert commit.ops == [] and svc.apply(prop["proposal_id"])["error"].startswith("not_approved")
    ok = svc.decide(prop["proposal_id"], True, by="alice", approved_overwrite=True)
    assert ok["ok"]
    assert commit.ops == []                                                         # approval still writes nothing
    applied = svc.apply(prop["proposal_id"], actor="alice")
    assert applied["ok"] and commit.tables["company_profile"][0]["headquarters"] != "Pune, India"
    actions = [e["action"] for e in deps.store.audit_entries("r1")]
    assert actions == ["proposed", "proposed_from_change_set", "approved", "applied"]
    approved = next(e for e in deps.store.audit_entries("r1") if e["action"] == "approved")
    assert approved["actor"] == "alice"


# =================================================================================== contacts vs the CRM
def contact_name(out):
    return out["claims"]["contact"][0]["name"]


def first_run(commit):
    runner, deps = build(commit, schema=ALL_COLUMNS)
    out = run_fields(runner, ["procurement_contacts"])
    return runner, deps, out


def test_a_new_person_is_inserted_with_their_public_source():
    commit = InMemoryCommitRepo(commit_tables(contacts=[{"contact_id": 1, "company_id": 7, "contact_name": "Zed Other",
                                                          "designation": "Purchase Manager", "source_url": None}]))
    runner, deps, out = first_run(commit)
    (p,) = runner.propose_from_change_set("r1")["proposals"]
    assert p["kind"] == "insert_contact" and p["status"] == "pending"
    op = deps.store.list_proposals("r1")[0]["change"]["operations"][0]
    assert op["op"] == "insert" and op["fields"]["source_url"] and op["fields"]["contact_name"] == contact_name(out)


def test_the_same_person_already_in_the_crm_gets_provenance_attached_not_a_duplicate_row():
    name = "Karan Bhatt"
    commit = InMemoryCommitRepo(commit_tables(contacts=[{"contact_id": 5, "company_id": 7, "contact_name": name.upper(),
                                                          "designation": "Purchase Manager", "source_url": None}]))
    runner, deps, out = first_run(commit)
    assert contact_name(out) == name
    (p,) = runner.propose_from_change_set("r1")["proposals"]
    assert p["kind"] == "verify_existing_contact"
    (op,) = deps.store.list_proposals("r1")[0]["change"]["operations"]
    assert op["op"] == "update" and op["key"] == {"contact_id": 5} and "contact_name" not in op["fields"]
    assert p["overwrites"] == ["contacts.designation"]                              # title differs: flagged as an overwrite
    assert commit.ops == [] and len(commit.tables["contacts"]) == 1


def test_a_probable_duplicate_is_blocked_for_a_human_not_inserted():
    commit = InMemoryCommitRepo(commit_tables(contacts=[{"contact_id": 9, "company_id": 7, "contact_name": "K. Bhatt",
                                                          "designation": "Purchase Manager", "source_url": None}]))
    runner, deps, out = first_run(commit)
    (p,) = runner.propose_from_change_set("r1")["proposals"]
    assert p["kind"] == "possible_duplicate_contact" and p["status"] == "blocked"
    assert "possible_duplicate_of_contact_9" in p["errors"][0]
    assert deps.store.list_proposals("r1")[0]["change"]["operations"] == []
    assert not runner.service().decide(p["proposal_id"], True, by="alice")["ok"]     # a blocked proposal cannot be approved


def test_a_contact_that_already_has_a_source_is_left_alone():
    commit = InMemoryCommitRepo(commit_tables(contacts=[{"contact_id": 5, "company_id": 7, "contact_name": "Karan Bhatt",
                                                          "designation": "Head of Procurement",
                                                          "source_url": "https://x.example/a"}]))
    runner, _, _ = first_run(commit)
    res = runner.propose_from_change_set("r1")
    assert res["proposals"] == [] and len(res["not_proposed"]) == 1


def test_person_similarity_rules():
    from agentic.names import person_similarity as ps
    assert ps("Karan Bhatt", "BHATT, Karan".replace(",", "")) == 1.0 and ps("Dr. Karan Bhatt", "karan bhatt") == 1.0
    assert ps("K. Bhatt", "Karan Bhatt") == 0.9 and ps("Karan Bhatt", "Karan Patel") < 0.8
    assert ps("", "Karan Bhatt") == 0.0


# ================================================================== schema gate and the revenue decision
def test_columns_without_schema_evidence_cannot_be_approved_or_applied():
    commit = InMemoryCommitRepo(commit_tables())
    runner, deps = build(commit, schema=SchemaContract())                              # only live-read columns confirmed
    run_fields(runner, ["headquarters", "official_website"])
    res = runner.propose_from_change_set("r1")
    (p,) = res["proposals"]
    assert "company_profile.headquarters" in p["schema_unconfirmed"]
    assert "companies.website" not in p["schema_unconfirmed"]                          # proven by the live SELECT
    svc = runner.service()
    gate = svc.decide(p["proposal_id"], True, by="alice")
    assert not gate["ok"] and gate["error"] == "schema_unconfirmed" and "AGENTIC_SCHEMA_FILE" in gate["how_to_confirm"]
    assert commit.ops == []
    # the operator saves the introspection output; the same proposal can then be approved
    svc.schema = ALL_COLUMNS
    assert svc.decide(p["proposal_id"], True, by="alice")["ok"]


def test_apply_re_checks_the_schema_even_if_a_proposal_was_approved_earlier():
    commit = InMemoryCommitRepo(commit_tables())
    runner, deps = build(commit, schema=ALL_COLUMNS)
    run_fields(runner, ["headquarters"])
    (p,) = runner.propose_from_change_set("r1")["proposals"]
    svc = runner.service()
    assert svc.decide(p["proposal_id"], True, by="alice")["ok"]
    svc.schema = SchemaContract()                                                     # evidence withdrawn
    res = svc.apply(p["proposal_id"])
    assert not res["ok"] and res["error"].startswith("schema_unconfirmed") and commit.ops == []


def test_revenue_is_company_profile_annual_revenue_and_the_legacy_column_is_context_only():
    assert fl.BY_KEY["annual_revenue"].table == "company_profile" and fl.BY_KEY["annual_revenue"].column == "annual_revenue"
    assert fl.LEGACY_COLUMNS["annual_revenue"] == ("companies", "revenue")
    # evidence levels: the legacy column was proven by a live SELECT, the target is not yet confirmed
    assert ("companies", "revenue") in LIVE_READ_CONFIRMED
    c = SchemaContract()
    assert c.evidence("companies", "revenue") == "live_read" and c.evidence("company_profile", "annual_revenue") == "unconfirmed"
    assert SchemaContract([("company_profile", "annual_revenue")]).evidence("company_profile", "annual_revenue") == "introspection"
    # the write allowlist can never touch the legacy column
    errs, _ = wg.validate_operation({"table": "companies", "op": "update", "key": {"company_id": 7},
                                     "fields": {"revenue": "Rs 1"}}, {"company_id": 7})
    assert errs
    # change set: proposed value targets the profile column, the legacy text is shown for context and never written
    existing = {"company_id": 7, "fields": {"revenue": "~Rs 200 Cr"}, "profile": {}}
    claim = {"agent": "company", "field": "annual_revenue", "value": "Rs 210 crore", "status": "verified",
             "confidence": 0.6, "confidence_basis": "x", "evidence": [{"url": "https://a.example/ir", "snippet":
             "Annual revenue was Rs 210 crore.", "retrieved_at": "t"}]}
    cs = cs_mod.build_change_set({"options": {"fields": ["annual_revenue"], "refresh": False},
                                  "subject": {"name": "Acme", "existing": existing}, "claims": {"company": [claim]},
                                  "rejected": {}})
    (item,) = cs["items"]
    assert item["target"] == "company_profile.annual_revenue" and item["action"] == "fill_empty" and item["old_value"] is None
    assert item["legacy_values"] == {"companies.revenue": "~Rs 200 Cr"}


def test_a_legacy_revenue_value_alone_does_not_stop_the_field_from_being_researched():
    state = {"options": {"fields": ["annual_revenue"], "refresh": False},
             "subject": {"name": "Acme", "existing": {"company_id": 7, "fields": {"revenue": "~Rs 200 Cr"}, "profile": {}},
                         "identity": {"verified": False}}, "claims": {}, "agent_runs": {}}
    assert fl.agents_needed(state) == {"company"}
    state["subject"]["existing"]["profile"] = {"annual_revenue": "Rs 205 crore"}
    assert fl.agents_needed(state) == set()


def test_the_introspection_sql_now_returns_the_column_list_and_the_contract_reads_it():
    assert "'columns'" in fk_introspect.INTROSPECTION_SQL and "information_schema.columns" in fk_introspect.INTROSPECTION_SQL
    raw = {"columns": [{"table": "company_profile", "column": "annual_revenue", "data_type": "text"}]}
    c = SchemaContract.from_introspection(raw)
    assert c.evidence("company_profile", "annual_revenue") == "introspection"
    with pytest.raises(ValueError, match="columns"):
        SchemaContract.from_introspection({"foreign_keys": []})
    assert operation_columns([{"table": "contacts", "op": "update", "key": {"contact_id": 1},
                               "fields": {"source_url": "u"}}]) == ["contacts.contact_id", "contacts.source_url"]


# ======================================================================= writes stay disabled by default
def test_without_a_commit_repository_there_are_no_proposals_and_the_change_set_is_still_available():
    runner, deps = build(None)
    out = run_fields(runner)
    assert out["report"]["change_set"]["items"] and deps.commit_repo is None
    with pytest.raises(RuntimeError, match="no commit repository"):
        runner.propose_from_change_set("r1")


def test_a_field_run_with_propose_changes_proposes_only_the_reviewed_change_set():
    commit = InMemoryCommitRepo(commit_tables())
    runner, deps = build(commit, schema=ALL_COLUMNS)
    out = run_fields(runner, ["headquarters"], propose=True)
    assert out["status"] == "awaiting_approval"
    stored = deps.store.list_proposals("r1")
    assert stored and all(s["change"]["origin"] == "change_set" for s in stored)
    assert all("website_domain" not in op["fields"] for s in stored for op in s["change"]["operations"])
    assert commit.ops == []


# ===================================================================================== API contract
def test_api_bridge_requires_the_approver_token_and_returns_items_and_stale_info(monkeypatch):
    monkeypatch.setenv("AGENTIC_API_TOKEN", "tok-for-test")
    commit = InMemoryCommitRepo(commit_tables())
    runner, deps = build(commit, schema=SchemaContract())
    svc = AgenticService(runner, "test", deps.settings)
    svc_mod.configure_service(svc)
    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)
    auth = {"Authorization": "Bearer tok-for-test"}
    try:
        rid = client.post("/agentic/runs", json={"company_name": "Apollo Micro Systems", "fields": ["headquarters"]},
                          headers=auth).json()["run_id"]
        svc.wait(rid, 60)
        assert client.post(f"/agentic/runs/{rid}/changeset/proposals").status_code == 401
        res = client.post(f"/agentic/runs/{rid}/changeset/proposals", headers=auth, json={})
        assert res.status_code == 200
        body = res.json()
        assert body["proposals"][0]["items"][0]["field"] == "headquarters" and body["stale_items"] == []
        assert "company_profile.headquarters" in body["proposals"][0]["schema_unconfirmed"]
        bad = client.post(f"/agentic/runs/{rid}/changeset/proposals", headers=auth, json={"item_ids": ["nope"]})
        assert bad.status_code == 422
        assert commit.ops == []
    finally:
        svc_mod.configure_service(None)


def test_api_bridge_is_unavailable_when_writes_are_not_enabled(monkeypatch):
    monkeypatch.setenv("AGENTIC_API_TOKEN", "tok-for-test")
    runner, deps = build(None)
    svc = AgenticService(runner, "test", deps.settings)
    svc_mod.configure_service(svc)
    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)
    auth = {"Authorization": "Bearer tok-for-test"}
    try:
        rid = client.post("/agentic/runs", json={"company_name": "Apollo Micro Systems", "fields": ["headquarters"]},
                          headers=auth).json()["run_id"]
        svc.wait(rid, 60)
        assert client.post(f"/agentic/runs/{rid}/changeset/proposals", headers=auth, json={}).status_code == 409
        assert client.get(f"/agentic/runs/{rid}/changeset", headers=auth).status_code == 200      # review still works
    finally:
        svc_mod.configure_service(None)


# ================================================== provider configuration, tool-call needs, budgets, retries
LIVE = {"AGENTIC_MODE": "live", "AGENTIC_ALLOW_LIVE": "1"}


def factories():
    w = world()
    return {"search": lambda: FixtureSearch(w), "fetcher": lambda: FixtureFetcher(w), "repo": default_repo}


def test_a_live_provider_cannot_start_without_a_local_quota_guard(monkeypatch, tmp_path):
    monkeypatch.setenv("GEMINI_API_KEY", "dummy-not-a-key")
    env = {**LIVE, "AGENTIC_DATA_DIR": str(tmp_path)}
    with pytest.raises(ServiceNotConfigured, match="quota guard"):
        build_live_service(env=env, factories=factories())
    ok = build_live_service(env={**env, "AGENTIC_QUOTA": "rpm=5,tpm=250000,rpd=20"}, factories=factories())
    quota = ok.runner.deps.llm.inner.quota
    assert (quota.rpm, quota.tpm, quota.rpd) == (5, 250000, 20)
    assert ok.runner.deps.schema is not None and ok.runner.deps.commit_repo is None        # writes stay off


def test_selected_model_report_states_requirements_budgets_retries_and_no_fallback(monkeypatch, tmp_path):
    monkeypatch.setenv("GEMINI_API_KEY", "dummy-not-a-key")
    env = {**LIVE, "AGENTIC_DATA_DIR": str(tmp_path), "AGENTIC_QUOTA": "rpm=5,tpm=250000,rpd=20",
           "AGENTIC_LLM_RETRIES": "0", "AGENTIC_LLM_PROVIDER": "gemini"}
    svc = build_live_service(env=env, factories=factories())
    rep = compat.selected_model_report(svc.runner.deps.llm, svc.settings, env)
    assert rep["provider"] == "gemini" and rep["model"] == "gemini-3-flash-preview" and rep["selection"] == "explicit"
    assert rep["budgets"]["quota_guard_active"] and rep["budgets"]["quota"]["rpd"] == 20
    assert rep["retries"] == {"llm_max_retries": 0, "request_timeout_s": 60.0} and "none" in rep["fallback"]
    assert any("thought_signature" in r for r in rep["requirements"])
    assert compat.selected_model_report(svc.runner.deps.llm, svc.settings, {})["selection"].startswith("default")


def test_every_registered_model_documents_its_tool_call_requirements():
    for row in compat.registry_view():
        assert row["requirements"] and any("AGENTIC_QUOTA" in r for r in row["requirements"]), row["model"]
    gem3 = compat.capability("gemini", "gemini-3-flash-preview")
    assert any("thought_signature" in r for r in gem3.requirements)
    assert [c["check"] for c in compat.verify_tool_calling("gemini", "gemini-3-flash-preview")["checks"]][-1] == \
        "thought_signature_replay"


@pytest.mark.parametrize("model,sends_reasoning_params", [("openai/gpt-oss-120b", True), ("qwen/qwen3-32b", False)])
def test_groq_requests_match_each_models_documented_needs(model, sends_reasoning_params):
    client = GroqClient(model=model, client=GroqWire(FieldLLM()))
    spec = compat.verify_tool_calling("groq", model)
    assert spec["ok"]
    from agentic.llm.base import ToolSpec
    req = client.build_request("s", [{"role": "user", "content": "t"}], [ToolSpec(name="x", description="d", parameters={
        "type": "object", "properties": {}})], 64, force_tool=True)
    assert req["tool_choice"] == "required" and "parallel_tool_calls" not in req and "service_tier" not in req
    assert ("reasoning_effort" in req) is sends_reasoning_params


def test_retries_are_bounded_quota_errors_are_never_retried_and_there_is_no_provider_fallback():
    calls = {"primary": 0, "other": 0}

    class Primary:
        name, model = "gemini", "m"

        def generate(self, **kw):
            calls["primary"] += 1
            raise LLMError("transient", "503")

    class Quota(Primary):
        def generate(self, **kw):
            calls["primary"] += 1
            raise LLMError("quota", "daily limit")
    for retries, cls, expected in ((0, Primary, 1), (2, Primary, 3), (2, Quota, 1)):
        calls["primary"] = 0
        with pytest.raises(LLMError):
            RetryingLLM(cls(), max_retries=retries, sleep=lambda s: None).generate(
                system="s", messages=[], tools=[], max_output_tokens=1)
        assert calls["primary"] == expected and calls["other"] == 0
    with pytest.raises(ServiceNotConfigured):                                          # unknown provider: refused outright
        build_live_service(env={**LIVE, "AGENTIC_LLM_PROVIDER": "openai"}, factories=factories())


def test_the_providers_endpoint_includes_the_selected_model_report(monkeypatch):
    runner, deps = build(None)
    svc = AgenticService(runner, "test", deps.settings)
    svc_mod.configure_service(svc)
    app = FastAPI()
    app.include_router(router)
    try:
        body = TestClient(app).get("/agentic/providers").json()
        assert "selected" in body and body["selected"]["fallback"].startswith("none")
        assert "requirements" in body["registry"][0]
    finally:
        svc_mod.configure_service(None)

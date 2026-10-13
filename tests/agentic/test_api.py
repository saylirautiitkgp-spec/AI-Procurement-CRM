"""Backend API contracts for the run / tracker / report / approval flow (offline, injected runtime)."""

import json
import threading

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from agentic import service as svc_mod
from agentic.commit import InMemoryCommitRepo
from agentic.config import Limits, Settings
from agentic.llm.fake import FakeLLM
from agentic.service import AgenticService
from api.agentic_routes import router
from tests.agentic.support import make_runner

FULL = "company profile"
TOKEN = "test-token-123"


@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(router)
    svc_mod.configure_service(None)
    yield TestClient(app)
    svc_mod.configure_service(None)


def install(runner, **kw) -> AgenticService:
    svc = AgenticService(runner, "test", runner.deps.settings, **kw)
    svc_mod.configure_service(svc)
    return svc


def wait_done(client, svc, run_id):
    svc.wait(run_id, 60)


def test_runtime_is_503_until_configured_and_refuses_live_without_opt_in(client, monkeypatch):
    for k in ("AGENTIC_MODE", "AGENTIC_ALLOW_LIVE"):
        monkeypatch.delenv(k, raising=False)
    r = client.post("/agentic/runs", json={"company_name": "Acme"})
    assert r.status_code == 503 and "AGENTIC_ALLOW_LIVE" in r.json()["detail"]
    monkeypatch.setenv("AGENTIC_MODE", "live")
    with pytest.raises(svc_mod.ServiceNotConfigured):
        svc_mod.build_live_service(env={"AGENTIC_MODE": "live"})          # allow flag missing
    with pytest.raises(svc_mod.ServiceNotConfigured):
        svc_mod.build_live_service(env={"AGENTIC_MODE": "live", "AGENTIC_ALLOW_LIVE": "1",
                                        "AGENTIC_LLM_PROVIDER": "openai"})  # unknown -> refuse, no silent fallback


def test_live_wiring_builds_without_write_access_by_default(tmp_path):
    runner = make_runner()
    built = svc_mod.build_live_service(
        env={"AGENTIC_MODE": "live", "AGENTIC_ALLOW_LIVE": "1", "AGENTIC_LLM_PROVIDER": "gemini"},
        factories={"llm": lambda: runner.llm, "search": lambda: runner.search, "fetcher": lambda: runner.fetcher,
                   "repo": lambda: runner.deps.repo, "commit": lambda: InMemoryCommitRepo()},
        data_dir=str(tmp_path))
    assert built.runner.deps.commit_repo is None                 # writes need AGENTIC_ENABLE_WRITES=1
    enabled = svc_mod.build_live_service(
        env={"AGENTIC_MODE": "live", "AGENTIC_ALLOW_LIVE": "1", "AGENTIC_ENABLE_WRITES": "1",
             "AGENTIC_LLM_PROVIDER": "gemini"},
        factories={"llm": lambda: runner.llm, "search": lambda: runner.search, "fetcher": lambda: runner.fetcher,
                   "repo": lambda: runner.deps.repo, "commit": lambda: InMemoryCommitRepo()},
        data_dir=str(tmp_path / "w"))
    assert enabled.runner.deps.commit_repo is not None


def test_start_poll_events_and_report_contract(client):
    runner = make_runner()
    svc = install(runner)
    r = client.post("/agentic/runs", json={"company_name": "Apollo Micro Systems", "objective": "comprehensive"})
    assert r.status_code == 202
    run_id = r.json()["run_id"]
    wait_done(client, svc, run_id)
    st = client.get(f"/agentic/runs/{run_id}").json()
    assert st["status"] == "completed" and st["report_ready"]
    t = st["tracker"]
    for key in ("run_id", "company_submitted", "database_record_found", "status", "current_agent", "agent_transitions",
                "steps", "usage", "sources", "gaps", "errors", "stop_reason", "database_change", "updated_at"):
        assert key in t, key
    assert t["database_record_found"] is True and t["database_change"] == "none_proposed"
    assert [x["next"] for x in t["agent_transitions"]][-1] == "finish"
    assert t["usage"]["search_credits"] > 0 and t["usage"]["llm_calls"] > 0
    assert t["sources"]["validated"] > 0 and t["sources"]["discovered"] >= t["sources"]["validated"]
    # incremental events
    ev1 = client.get(f"/agentic/runs/{run_id}/events?after=0&limit=5").json()
    assert len(ev1["events"]) == 5 and ev1["last_seq"] == 5
    ev2 = client.get(f"/agentic/runs/{run_id}/events?after={ev1['last_seq']}&limit=500").json()
    assert ev2["events"][0]["seq"] == 6
    rep = client.get(f"/agentic/runs/{run_id}/report").json()
    assert rep["run"]["status"] == "completed" and rep["contacts"] and rep["lead_priority"]["factors"]
    assert client.get("/agentic/runs/nope").status_code == 404
    assert client.get("/agentic/runs").json()["runs"][0]["run_id"] == run_id


def test_tracker_and_events_never_expose_secrets_or_contact_names(client, monkeypatch):
    monkeypatch.setenv("SUPABASE_KEY", "sb_secret_TOPSECRETVALUE99")
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-dev-SECRETSECRET123")
    runner = make_runner()
    svc = install(runner)
    run_id = client.post("/agentic/runs", json={"company_name": "Apollo Micro Systems",
                                                "objective": "comprehensive"}).json()["run_id"]
    wait_done(client, svc, run_id)
    person = runner.world["facts"]["person"]
    blobs = [client.get(f"/agentic/runs/{run_id}").text, client.get(f"/agentic/runs/{run_id}/events?limit=500").text,
             client.get("/agentic/usage").text]
    for b in blobs:
        assert "TOPSECRETVALUE99" not in b and "SECRETSECRET123" not in b
    assert person not in blobs[0]                                   # tracker carries counts, not personal data
    assert person in client.get(f"/agentic/runs/{run_id}/report").text   # the report (the deliverable) does


def test_report_is_409_while_running_and_failed_runs_are_reported_safely(client):
    runner = make_runner(llm=FakeLLM([RuntimeError("boom with sk-secret-value")]))     # crashes immediately
    svc = install(runner)
    run_id = client.post("/agentic/runs", json={"company_name": "Apollo Micro Systems"}).json()["run_id"]
    wait_done(client, svc, run_id)
    st = client.get(f"/agentic/runs/{run_id}").json()
    assert st["status"] == "failed" and st["tracker"]["errors"] == [{"stage": "runtime", "error": "RuntimeError"}]
    assert "sk-secret" not in json.dumps(st)
    assert client.get(f"/agentic/runs/{run_id}/report").status_code == 404


def test_invalid_input_and_bounded_queue(client):
    gate = threading.Event()

    class Blocking(FakeLLM):
        def generate(self, **kw):
            gate.wait(20)
            raise RuntimeError("stop")

    runner = make_runner(llm=Blocking())
    svc = install(runner, max_queue=1)
    assert client.post("/agentic/runs", json={"company_name": "x"}).status_code == 422
    assert client.post("/agentic/runs", json={"company_name": "\x00\x01 "}).status_code in (422,)
    a = client.post("/agentic/runs", json={"company_name": "Apollo Micro Systems"})
    b = client.post("/agentic/runs", json={"company_name": "Apollo Micro Systems"})
    c = client.post("/agentic/runs", json={"company_name": "Apollo Micro Systems"})
    assert (a.status_code, b.status_code, c.status_code) == (202, 202, 429)       # 1 running + 1 queued, rest rejected
    assert client.get("/agentic/usage").json()["max_concurrent_runs"] == 1
    gate.set()
    svc.wait(a.json()["run_id"], 30)
    svc.wait(b.json()["run_id"], 30)


def test_usage_endpoint_reports_free_tier_budgets(client):
    runner = make_runner(settings=Settings(limits=Limits(monthly_credit_budget=250)))
    install(runner)
    u = client.get("/agentic/usage").json()
    assert u["monthly_credit_budget"] == 250 and u["search_credits_used_this_month"] == 0
    assert u["per_run_limits"]["search_credits"] == 10 and u["provider"] == "policy-double"


# ------------------------------------------------------------------- authentication
def test_token_protects_everything_and_approvals_need_a_token(client, monkeypatch):
    runner = make_runner(commit_repo=InMemoryCommitRepo({"companies": [], "company_profile": []}))
    install(runner)
    monkeypatch.delenv("AGENTIC_API_TOKEN", raising=False)
    assert client.post("/agentic/runs/x/approvals", json={"decisions": {}}).status_code == 403
    assert client.post("/agentic/proposals/p/apply").status_code == 403
    monkeypatch.setenv("AGENTIC_API_TOKEN", TOKEN)
    assert client.get("/agentic/usage").status_code == 401
    assert client.get("/agentic/usage", headers={"Authorization": "Bearer wrong"}).status_code == 401
    assert client.get("/agentic/usage", headers={"Authorization": f"Bearer {TOKEN}"}).status_code == 200


# ------------------------------------------------------------------- approval over HTTP
def test_approval_flow_over_http_pauses_then_applies(client, monkeypatch):
    monkeypatch.setenv("AGENTIC_API_TOKEN", TOKEN)
    H = {"Authorization": f"Bearer {TOKEN}"}
    repo = InMemoryCommitRepo({"companies": [{"company_id": 7, "company_name": "Apollo Micro Systems",
                                              "website": None, "linkedin_url": None}], "company_profile": []})
    runner = make_runner(commit_repo=repo)
    svc = install(runner)
    run_id = client.post("/agentic/runs", headers=H, json={"company_name": "Apollo Micro Systems",
                         "objective": FULL, "propose_changes": True}).json()["run_id"]
    wait_done(client, svc, run_id)
    st = client.get(f"/agentic/runs/{run_id}", headers=H).json()
    assert st["status"] == "awaiting_approval" and st["tracker"]["database_change"] == "awaiting_approval"
    assert repo.ops == []
    props = client.get(f"/agentic/runs/{run_id}/proposals", headers=H).json()["proposals"]
    pid = props[0]["proposal_id"]
    assert props[0]["status"] == "pending" and props[0]["evidence"]
    bad = client.post(f"/agentic/runs/{run_id}/approvals", headers=H,
                      json={"decisions": {pid: {"approve": True, "by": ""}}})
    assert bad.status_code == 422                                                # approver identity is mandatory
    ok = client.post(f"/agentic/runs/{run_id}/approvals", headers=H,
                     json={"decisions": {pid: {"approve": True, "by": "sayli", "reason": "ok"}}})
    assert ok.status_code == 200 and ok.json()["status"] == "completed"
    assert ok.json()["proposals"][0]["status"] == "applied" and repo.ops
    assert client.post(f"/agentic/runs/{run_id}/approvals", headers=H,
                       json={"decisions": {pid: {"approve": True, "by": "sayli"}}}).status_code == 409


def test_decoupled_flow_research_first_then_propose_decide_apply(client, monkeypatch):
    monkeypatch.setenv("AGENTIC_API_TOKEN", TOKEN)
    H = {"Authorization": f"Bearer {TOKEN}"}
    repo = InMemoryCommitRepo({"companies": [{"company_id": 7, "company_name": "Apollo Micro Systems",
                                              "website": None, "linkedin_url": None}], "company_profile": []})
    runner = make_runner(commit_repo=repo)
    svc = install(runner)
    run_id = client.post("/agentic/runs", headers=H, json={"company_name": "Apollo Micro Systems",
                                                           "objective": FULL}).json()["run_id"]
    wait_done(client, svc, run_id)
    assert client.get(f"/agentic/runs/{run_id}", headers=H).json()["status"] == "completed" and repo.ops == []
    pid = client.post(f"/agentic/runs/{run_id}/propose", headers=H).json()["proposals"][0]["proposal_id"]
    assert client.post(f"/agentic/proposals/{pid}/apply", headers=H).status_code == 409        # not approved yet
    assert client.post(f"/agentic/proposals/{pid}/decision", headers=H,
                       json={"approve": True, "by": "sayli"}).status_code == 200
    assert client.post(f"/agentic/proposals/{pid}/apply", headers=H).json()["status"] == "applied"
    assert repo.ops


def test_existing_application_still_imports_and_exposes_both_route_sets():
    import importlib
    app_mod = importlib.import_module("api.app")
    paths = set(app_mod.app.openapi()["paths"])
    assert "/dashboard" in paths and "/agentic/runs" in paths and "/agentic/usage" in paths


def test_gemini_quota_guard_is_in_the_live_path_and_writes_stay_off(tmp_path):
    from agentic.llm.base import LLMError
    from agentic.llm.pacing import PacedLLM, ProviderQuota
    runner = make_runner()
    env = {"AGENTIC_MODE": "live", "AGENTIC_ALLOW_LIVE": "1", "AGENTIC_LLM_PROVIDER": "gemini",
           "AGENTIC_QUOTA": "rpm=5,tpm=250000,rpd=20", "AGENTIC_ENABLE_WRITES": "0"}
    f = {"llm": lambda: runner.llm, "search": lambda: runner.search, "fetcher": lambda: runner.fetcher,
         "repo": lambda: runner.deps.repo, "commit": lambda: InMemoryCommitRepo()}
    built = svc_mod.build_live_service(env=env, factories=f, data_dir=str(tmp_path))
    paced = built.runner.deps.llm.inner                      # RetryingLLM -> PacedLLM -> client
    assert isinstance(paced, PacedLLM) and paced.quota.rpd == 20 and paced.log is built.runner.deps.store
    assert built.runner.deps.commit_repo is None
    for _ in range(16):                                       # 80% planning cap of 20 requests/day
        built.runner.deps.store.usage_add(paced._key, 100, paced._clock())
    with pytest.raises(LLMError) as exc:
        paced.generate(system="s", messages=[], tools=[], max_output_tokens=10)
    assert exc.value.kind == "quota"
    assert paced.local_refusals == 1
    with pytest.raises(ValueError):                           # Groq must still declare all four limits
        ProviderQuota.parse("rpm=30,tpm=8000,rpd=1000")


def test_llm_retries_env_defaults_to_two_and_zero_means_no_retry_attempts():
    from agentic.config import load_settings
    from agentic.llm.base import LLMError
    from agentic.runtime import make_deps
    from agentic.persistence.store import Store
    assert load_settings({}).limits.llm_max_retries == 2
    assert load_settings({"AGENTIC_LLM_RETRIES": "junk"}).limits.llm_max_retries == 2
    assert load_settings({"AGENTIC_LLM_RETRIES": "-3"}).limits.llm_max_retries == 0

    class Flaky:
        name, model, calls = "flaky", "m", 0

        def generate(self, **kw):
            Flaky.calls += 1
            raise LLMError("rate_limit", "429")

    for retries, expected_calls in (("0", 1), ("2", 3)):
        Flaky.calls = 0
        settings = load_settings({"AGENTIC_LLM_RETRIES": retries})
        deps = make_deps(llm=Flaky(), repo=None, search=None, fetcher=None, store=Store(":memory:"),
                         settings=settings, specs={}, sleep=lambda s: None)
        with pytest.raises(LLMError):
            deps.llm.generate(system="s", messages=[], tools=[], max_output_tokens=1)
        assert Flaky.calls == expected_calls and deps.llm.retries == expected_calls - 1

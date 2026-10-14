"""AI CRM Procurement Platform: read-only CRM API, plan/estimate, simulation runs, the guarded apply endpoint, static serving
and frontend safety checks. Offline only: bundled demo CRM, scripted model, fake Supabase client, in-memory commit repository."""

import os
import re
import shutil
import subprocess
import time

import pytest
from fastapi.testclient import TestClient

from agentic import fields as fl
from agentic.commit import InMemoryCommitRepo
from agentic.config import Limits, Settings
from agentic.crm_read import DemoCrm, SupabaseCrm
from agentic.demo_service import build_demo_service, demo_repo_from
from agentic.llm.pacing import ProviderQuota
from agentic.persistence.store import Store
from agentic.service import configure_service
from api import platform_routes as pr
from api.platform_app import FRONTEND_DIR, app

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture
def demo(tmp_path, monkeypatch):
    monkeypatch.delenv("AGENTIC_API_TOKEN", raising=False)
    crm = DemoCrm()
    svc = build_demo_service(crm, demo_repo_from(crm), str(tmp_path))
    configure_service(svc)
    pr.STATE.__dict__.update(pr.PlatformState().__dict__)
    pr.configure(crm=crm, research_mode="simulation")
    yield TestClient(app), svc, crm
    configure_service(None)
    pr.STATE.__dict__.update(pr.PlatformState().__dict__)


# ============================================================================================ read API
def test_status_overview_and_sector_counts(demo):
    c, svc, crm = demo
    st = c.get("/platform/status").json()
    assert st["app"] == "AI CRM Procurement Platform" and st["data_source"] == "demo" and st["research_mode"] == "simulation"
    assert st["writes_enabled"] is False and st["model"] == "scripted-policy" and "key" not in str(st).lower()
    ov = c.get("/platform/overview").json()
    assert ov["totals"]["companies"] == 20 and ov["totals"]["sectors"] == 7
    assert sum(s["companies"] for s in ov["sectors"]) == 20 and len(ov["companies"]) == 20
    assert {c_["sector"] for c_ in ov["companies"]} == {s["name"] for s in ov["sectors"]}
    sample = ov["companies"][0]
    assert {"id", "name", "country", "sector_id", "sector", "profile_fields", "profile_fields_total", "insights", "has_profile"} <= set(sample)


def test_company_detail_has_coverage_and_never_exposes_contact_details(demo):
    c, svc, crm = demo
    d = c.get("/platform/companies/101").json()
    assert d["company"]["company_name"] == "Pixxel" and d["company"]["sector"] == "Aerospace & Defence"
    keys = {f["key"] for f in d["coverage"]}
    assert keys == {k for k, v in fl.canonical_destinations().items() if v["layer"] in ("core", "extended", "insight")}
    assert all(not f["filled"] for f in d["coverage"]) and d["products"] and d["contacts"]["count"] == 0
    assert c.get("/platform/companies/99999").status_code == 404


def test_the_field_catalog_is_grouped_by_agent_and_presets_are_valid(demo):
    c, svc, crm = demo
    f = c.get("/platform/fields").json()
    seen = [x["key"] for g in f["groups"] for x in g["fields"]]
    assert sorted(seen) == sorted(fl.BY_KEY) and len(seen) == len(set(seen))
    assert [g["uses_model"] for g in f["groups"]] == [True, True, True, True, False]       # lead is computed in code
    for p in f["presets"]:
        assert p["fields"] and all(k in fl.BY_KEY for k in p["fields"])


# ============================================================================================ estimate + simulation run
def test_estimate_skips_stored_fields_unless_reverify_is_ticked(demo, monkeypatch):
    c, svc, crm = demo
    orig = crm.company
    monkeypatch.setattr(crm, "company", lambda cid: {**orig(cid), "profile": {"industry": "Aerospace"}})
    body = {"company_id": 101, "fields": ["industry", "headquarters"]}
    e = c.post("/platform/estimate", json=body).json()
    assert e["skipped_already_stored"] == ["industry"] and e["agents"] == ["company"] and e["can_run"] and e["mode"] == "simulation"
    assert e["plan"]["tavily_credits"] == 0 and any("SIMULATION" in n for n in e["notes"])
    assert c.post("/platform/estimate", json={**body, "refresh": True}).json()["skipped_already_stored"] == []
    only = c.post("/platform/estimate", json={"company_id": 101, "fields": ["industry"]}).json()
    assert only["agents"] == [] and only["can_run"] is False and "already stored" in only["notes"][0]
    assert c.post("/platform/estimate", json={"company_id": 101, "fields": ["nonsense"]}).status_code == 422
    assert c.post("/platform/estimate", json={"company_id": 101, "fields": []}).status_code == 422


def test_estimate_orders_agents_and_adds_prerequisites(demo):
    c, svc, crm = demo
    e = c.post("/platform/estimate", json={"company_id": 101, "fields": ["suppliers", "procurement_contacts", "lead_score"]}).json()
    assert e["agents"][0] == "company" and e["agents"][-1] == "lead" and {"contact", "supplier"} <= set(e["agents"])
    assert [a["uses_model"] for a in e["agent_titles"]][-1] is False


def test_a_simulated_run_goes_end_to_end_and_can_never_write(demo):
    c, svc, crm = demo
    r = c.post("/platform/runs", json={"company_id": 101, "fields": ["headquarters", "official_website", "employee_count"]})
    assert r.status_code == 202 and r.json()["mode"] == "simulation"
    run_id = r.json()["run_id"]
    svc.wait(run_id, 90)
    st = c.get(f"/agentic/runs/{run_id}").json()
    assert st["status"] == "completed" and st["tracker"]["usage"]["llm_calls"] > 0
    cs = c.get(f"/agentic/runs/{run_id}/changeset").json()
    assert {i["field"] for i in cs["items"]} == {"headquarters", "official_website", "employee_count"} and cs["writes_performed"] is False
    assert all(i["evidence"] for i in cs["items"])
    assert svc.runner.deps.commit_repo is None                                          # no write path exists in simulation
    assert c.get("/agentic/runs/" + run_id + "/events?after=0").json()["events"]
    runs = c.get("/agentic/runs").json()["runs"]
    assert runs[0]["run_id"] == run_id


# ============================================================================================ live preflight enforced by the API
def live_state(tmp_path, quota="rpm=5,tpm=250000,rpd=20", used=0):
    s = Settings(limits=Limits(llm_max_retries=1, retry_503_only=True, max_requests_total=8, max_llm_calls_run=8,
                               max_total_tokens_run=160000, max_search_credits_run=4, max_db_selects_run=14, max_tool_calls_run=40),
                 provider="gemini", model="gemini-3.8-flash", data_dir=str(tmp_path))
    store = Store(s.db_path)
    for i in range(used):
        store.usage_add("gemini:gemini-3.8-flash", 1000, time.time() - 60 - i)
    pr.configure(research_mode="live", settings=s, quota=ProviderQuota.parse(quota, require_tpd=False) if quota else None)


def test_live_estimate_is_blocked_when_the_quota_is_unknown_or_used_up_and_no_run_is_started(demo, tmp_path, monkeypatch):
    c, svc, crm = demo
    monkeypatch.setattr(svc, "submit", lambda *a, **k: (_ for _ in ()).throw(AssertionError("a run was started")))
    body = {"company_id": 101, "fields": ["headquarters", "supplier_portal"]}
    live_state(tmp_path, quota=None)
    e = c.post("/platform/estimate", json=body).json()
    assert e["blocked"] and e["block_line"] == "PREFLIGHT BLOCKED — no external requests sent" and "UNKNOWN" in " ".join(e["reasons"])
    assert c.post("/platform/runs", json=body).status_code == 409
    live_state(tmp_path / "full", used=16)
    e = c.post("/platform/estimate", json=body).json()
    assert e["blocked"] and "remaining daily allowance 0" in " ".join(e["reasons"]) and e["quota"]["used_24h"] == 16
    r = c.post("/platform/runs", json=body)
    assert r.status_code == 409 and r.json()["detail"]["error"].startswith("PREFLIGHT BLOCKED")


def test_live_estimate_reports_the_full_plan_when_everything_fits(demo, tmp_path):
    c, svc, crm = demo
    live_state(tmp_path, used=3)
    e = c.post("/platform/estimate", json={"company_id": 101, "fields": ["headquarters", "supplier_portal", "procurement_contacts"]}).json()
    assert e["can_run"] and not e["blocked"] and e["scope"] == ["company", "contact", "procurement"]
    p = e["plan"]
    assert p["requests_max"] == 6 and p["tavily_credits"] == 3 and p["tokens_in"] > 0 and p["retries"] == 1
    assert e["quota"]["remaining_under_cap"] == 13 and any("LIVE" in n for n in e["notes"]) and any("Contact" in n for n in e["notes"])
    assert any("paced" in n for n in e["notes"])                                       # 6 requests at 5 RPM: slow, not forbidden
    live_state(tmp_path / "slow", used=0)
    pr.STATE.settings = Settings(limits=Limits(llm_max_retries=1, retry_503_only=True, max_requests_total=8, max_llm_calls_run=8,
                                               max_total_tokens_run=160000, max_search_credits_run=4, max_db_selects_run=14,
                                               max_tool_calls_run=40, max_wall_clock_s=60.0), provider="gemini",
                                 model="gemini-3.8-flash", data_dir=str(tmp_path / "slow"))
    Store(pr.STATE.settings.db_path)
    blocked = c.post("/platform/estimate", json={"company_id": 101, "fields": ["headquarters", "supplier_portal", "procurement_contacts"]}).json()
    assert blocked["blocked"] and "pacing" in " ".join(blocked["reasons"])             # only when the waiting alone exceeds the run limit
    lead_only = c.post("/platform/estimate", json={"company_id": 101, "fields": ["lead_score"]}).json()
    assert lead_only["plan"]["requests_max"] == 0 and lead_only["can_run"]


# ============================================================================================ guarded apply
def tables(cid):
    return {"companies": [{"company_id": cid, "company_name": "Pixxel", "website": None, "linkedin_url": None, "company_type": None,
                           "country": "India", "sector_id": 1}], "company_profile": [], "contacts": [], "ai_insights": [],
            "suppliers": [], "company_suppliers": []}


def test_the_apply_endpoint_is_closed_unless_everything_allows_it(demo, monkeypatch):
    c, svc, crm = demo
    run_id = c.post("/platform/runs", json={"company_id": 101, "fields": ["headquarters"]}).json()["run_id"]
    svc.wait(run_id, 90)
    url = f"/platform/runs/{run_id}/apply"
    body = {"approver": "Test Person", "confirm": True}
    assert c.post(url, json=body).status_code == 403                                   # no token configured at all -> disabled
    monkeypatch.setenv("AGENTIC_API_TOKEN", "t0ken")
    auth = {"Authorization": "Bearer t0ken"}
    assert c.post(url, json=body).status_code == 401                                   # wrong / missing token
    assert c.post(url, json=body, headers=auth).status_code == 409                    # writes switched off on this server
    box = {}
    pr.configure(research_mode="live", writes_enabled=True, commit_factory=lambda b: box.setdefault("repo", InMemoryCommitRepo(tables(101))))
    assert c.post(url, json={**body, "confirm": False}, headers=auth).status_code == 422
    assert c.post(url, json={"approver": "x", "confirm": True}, headers=auth).status_code == 422       # name too short
    assert c.post(url, json=body, headers=auth).status_code == 409                    # the service is a simulation, not live
    assert "repo" not in box                                                          # nothing was even constructed


def test_applying_selected_items_writes_only_additive_rows_for_that_company(demo, monkeypatch):
    c, svc, crm = demo
    run_id = c.post("/platform/runs", json={"company_id": 101, "fields": ["headquarters", "official_website", "employee_count"]}).json()["run_id"]
    svc.wait(run_id, 90)
    cs = c.get(f"/agentic/runs/{run_id}/changeset").json()
    hq = next(i for i in cs["items"] if i["field"] == "headquarters")
    monkeypatch.setenv("AGENTIC_API_TOKEN", "t0ken")
    seen = {}

    def factory(budget):
        seen["budget"] = budget
        return seen.setdefault("repo", InMemoryCommitRepo(tables(101)))
    pr.configure(research_mode="live", writes_enabled=True, commit_factory=factory)
    svc.mode = "live"                                                                  # the endpoint requires a live service
    auth = {"Authorization": "Bearer t0ken"}
    r = c.post(f"/platform/runs/{run_id}/apply", json={"approver": "Test Person", "confirm": True, "item_ids": [hq["id"]]}, headers=auth)
    assert r.status_code == 200, r.text
    out, repo = r.json(), seen["repo"]
    assert out["company_id"] == 101 and out["writes"] > 0 and seen["budget"].limit == 40 and out["select_cap"] == 40
    assert out["write_selects_used"] == seen["budget"].used <= 40
    assert {o[1] for o in repo.ops} <= {"companies", "company_profile", "ai_insights"} and all(o[0] in ("insert", "update") for o in repo.ops)
    assert repo.tables["company_profile"] and repo.tables["company_profile"][0]["company_id"] == 101
    assert repo.tables["company_profile"][0].get("headquarters") and not repo.tables["company_profile"][0].get("employee_count")
    assert repo.tables["companies"][0]["website"] is None                              # not selected -> not written
    assert all(i["company_id"] == 101 and i["source_url"] for i in repo.tables["ai_insights"])
    assert {s["reason"] for s in out["skipped_items"]} >= {"not_selected_by_the_approver"}
    again = c.post(f"/platform/runs/{run_id}/apply", json={"approver": "Test Person", "confirm": True, "item_ids": [hq["id"]]}, headers=auth).json()
    assert again["writes"] == 0 and len(repo.tables["company_profile"]) == 1           # idempotent: nothing duplicated
    assert c.post("/platform/runs/doesnotexist/apply", json={"approver": "Test Person", "confirm": True}, headers=auth).status_code == 404


# ============================================================================================ static UI
def test_the_front_end_is_served_with_safe_headers_and_every_local_asset_exists(demo):
    c, svc, crm = demo
    r = c.get("/ui/")
    assert r.status_code == 200 and "AI CRM Procurement Platform" in r.text and "text/html" in r.headers["content-type"]
    assert r.headers["cache-control"] == "no-store" and r.headers["x-content-type-options"] == "nosniff"
    assert c.get("/", follow_redirects=False).headers["location"] == "/ui/"
    index = open(os.path.join(FRONTEND_DIR, "index.html"), encoding="utf-8").read()
    local = re.findall(r'(?:href|src)="((?!https?:|data:|#)[^"]+)"', index)
    assert local and all(os.path.isfile(os.path.join(FRONTEND_DIR, p)) for p in local)
    for name in os.listdir(os.path.join(FRONTEND_DIR, "js")):
        src = open(os.path.join(FRONTEND_DIR, "js", name), encoding="utf-8").read()
        for target in re.findall(r"from '\./([\w\-]+\.js)'", src):
            assert os.path.isfile(os.path.join(FRONTEND_DIR, "js", target)), (name, target)
    assert c.get("/ui/..%2fserve_platform.py").status_code == 404
    assert c.get("/ui/../serve_platform.py").status_code in (404, 307, 200) and "AI CRM" not in c.get("/ui/css/../../serve_platform.py").text


def test_the_official_logo_is_used_only_when_the_file_exists(demo, tmp_path, monkeypatch):
    c, svc, crm = demo
    monkeypatch.setattr(pr, "ASSETS_DIR", str(tmp_path))
    assert c.get("/platform/status").json()["logo"] is None
    (tmp_path / "motherson-logo.png").write_bytes(b"\x89PNG")
    assert c.get("/platform/status").json()["logo"] == "assets/motherson-logo.png"
    (tmp_path / "motherson-logo.svg").write_text("<svg/>")
    assert c.get("/platform/status").json()["logo"] == "assets/motherson-logo.svg"       # svg wins


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_every_front_end_script_parses():
    for name in sorted(os.listdir(os.path.join(FRONTEND_DIR, "js"))):
        r = subprocess.run(["node", "--check", os.path.join(FRONTEND_DIR, "js", name)], capture_output=True, text=True)
        assert r.returncode == 0, (name, r.stderr[:300])


def test_page_data_never_goes_through_innerhtml():
    allowed = {"ui.js", "app.js"}                                  # static, trusted strings only (icons, the placeholder wordmark)
    for name in os.listdir(os.path.join(FRONTEND_DIR, "js")):
        src = open(os.path.join(FRONTEND_DIR, "js", name), encoding="utf-8").read()
        if name not in allowed:
            assert "innerHTML" not in src and "insertAdjacentHTML" not in src and "document.write" not in src, name
    ui = open(os.path.join(FRONTEND_DIR, "js", "ui.js"), encoding="utf-8").read()
    assert len(re.findall(r"\.innerHTML\s*=", ui)) == 1 and "ICONS[name]" in ui
    app_js = open(os.path.join(FRONTEND_DIR, "js", "app.js"), encoding="utf-8").read()
    assert len(re.findall(r"\.innerHTML\s*=", app_js)) == 1 and "MOTHERSON" in app_js


# ============================================================================================ Supabase reader (fake client)
class Q:
    def __init__(self, rows):
        self.rows, self.eqs, self.lo, self.hi, self.ins = rows, [], None, None, None

    def select(self, cols="*"):
        return self

    def eq(self, k, v):
        self.eqs.append((k, v)); return self

    def in_(self, k, vs):
        self.ins = (k, vs); return self

    def order(self, col):
        return self

    def limit(self, n):
        return self

    def range(self, a, b):
        self.lo, self.hi = a, b; return self

    def execute(self):
        out = [r for r in self.rows if all(r.get(k) == v for k, v in self.eqs)]
        if self.lo is not None:
            out = out[self.lo:self.hi + 1]
        return type("R", (), {"data": [dict(r) for r in out]})()


class FakeClient:
    def __init__(self, tables):
        self.tables, self.calls = tables, []

    def table(self, name):
        self.calls.append(name)
        return Q(self.tables[name])


def fake_db():
    return FakeClient({
        "sectors": [{"sector_id": 1, "sector": "Solar"}, {"sector_id": 2, "sector": "Wind"}],
        "companies": [{"company_id": i, "company_name": f"Co {i}", "country": "India", "sector_id": 1 + i % 2, "sector": None, "website": None,
                       "linkedin_url": None, "company_type": None, "enrichment_status": "Pending", "priority": "Medium",
                       "number_of_plants": None, "revenue": None} for i in range(1, 6)],
        "company_profile": [{"profile_id": 9, "company_id": 2, "industry": "Solar", "headquarters": None, "extracted_text": "long text"}],
        "ai_insights": [{"insight_id": 1, "company_id": 2, "insight_type": "Verified industry", "insight_title": "industry: Solar", "insight": "q (source: https://a.example/)",
                         "source_url": "https://a.example/", "generated_by": "agentic-company", "confidence_score": 0.6}],
        "products": [{"company_id": 2, "product_name": "Panels"}], "components": [{"company_id": 2, "component_name": "Cells"}],
        "contacts": [{"company_id": 2, "contact_id": 4, "contact_name": "A B", "designation": "Head of Procurement", "department": None,
                      "source_url": None, "confidence_score": None, "decision_maker": True, "email": "a@b.example", "phone": "123"}],
        "lead_status": [],
    })


def test_the_supabase_reader_aggregates_caches_and_is_read_only():
    db = fake_db()
    crm = SupabaseCrm(db, ttl_s=60)
    ov = crm.overview()
    assert ov["totals"] == {"companies": 5, "sectors": 2, "countries": 1, "with_profile": 1, "with_insights": 1}
    assert {s["name"]: s["companies"] for s in ov["sectors"]} == {"Solar": 2, "Wind": 3}
    two = next(c for c in ov["companies"] if c["id"] == 2)
    assert two["has_profile"] and two["insights"] == 1 and two["profile_fields"] == 1
    n = len(db.calls)
    assert crm.overview() is ov and len(db.calls) == n                                 # served from the cache
    crm.invalidate(); crm.overview(); assert len(db.calls) > n
    d = crm.company(2)
    assert d["profile"] == {"profile_id": 9, "company_id": 2, "industry": "Solar"} and "extracted_text" not in d["profile"]
    assert d["contacts"]["items"] == [{"id": 4, "name": "A B", "designation": "Head of Procurement", "department": None, "has_source": False}]
    assert "a@b.example" not in str(d) and "123" not in str(d)                         # no e-mail / phone ever leaves the server
    assert [f["sourced"] for f in d["coverage"] if f["key"] == "industry"] == [True]
    assert crm.company(999) is None
    with pytest.raises(PermissionError):
        crm._c.table("companies").insert({"company_name": "x"})
    with pytest.raises(PermissionError):
        crm._c.table("companies").delete()


def test_a_crm_failure_is_reported_without_leaking_details(demo, monkeypatch):
    c, svc, crm = demo
    monkeypatch.setattr(crm, "overview", lambda: (_ for _ in ()).throw(RuntimeError("secret connection string")))
    r = c.get("/platform/overview")
    assert r.status_code == 502 and "secret" not in r.text and "RuntimeError" in r.text

"""Field-level requests: select a company and exactly which data to retrieve/refresh. Selective agent routing, safe
parallel stages, shared evidence, minimal LLM calls, provider compatibility, budgets, partial failures and the
review change set. Offline only - synthetic worlds, scripted models, wire emulators; no network, no keys."""

import dataclasses
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from agentic import changeset as cs_mod
from agentic import fields as fl
from agentic import parallel
from agentic import service as svc_mod
from agentic.agents import AGENT_SPECS
from agentic.backends.fixtures import FixtureFetcher, FixtureSearch
from agentic.config import Efficiency, Limits, Settings
from agentic.eval.wire_emulators import GeminiWire, GroqWire
from agentic.guards.budget import new_budgets
from agentic.guards.executor import GuardedExecutor
from agentic.llm import compat
from agentic.llm.base import ToolCallRequest
from agentic.llm.gemini import GeminiClient
from agentic.llm.groq import GroqClient
from agentic.persistence.store import Store
from agentic.repo import InMemoryRepo
from agentic.runtime import AgenticRunner, make_deps
from agentic.service import AgenticService, ServiceNotConfigured, build_live_service
from api.agentic_routes import router
from tests.agentic.fieldllm import FieldLLM
from tests.agentic.support import COMPANY_A, default_repo
from tests.agentic.test_hybrid import HYBRID, SpyRepo, llm_events
from tests.agentic.test_prefetch import events, world

PAR = dataclasses.replace(HYBRID, parallel_agents=True)
ALL_FIELDS = ["headquarters", "procurement_contacts", "supplier_portal", "suppliers", "lead_score"]


def runner_for(w=None, llm=None, eff=PAR, repo=None, limits=None, store=None, specs=None):
    specs = specs or AGENT_SPECS
    w = w or world()
    search, fetch = FixtureSearch(w), FixtureFetcher(w)
    store = store or Store(":memory:")
    settings = Settings(limits=limits or Limits(), efficiency=eff)
    deps = make_deps(llm=llm or FieldLLM(), repo=repo if repo is not None else default_repo(), search=search,
                     fetcher=fetch, store=store, settings=settings, specs=specs, sleep=lambda s: None)
    r = AgenticRunner(deps, specs)
    r.search, r.fetcher, r.store = search, fetch, store
    return r


def run(fields, *, refresh=False, llm=None, **kw):
    llm = llm or FieldLLM()
    r = runner_for(llm=llm, **kw)
    out = r.start("Apollo Micro Systems", None, fields=fields, refresh=refresh)
    return r, out, llm


def crm_repo(*, website=None, headquarters=None):
    row = {**COMPANY_A, "website": website}
    return InMemoryRepo([row], products={7: ["Avionics"]}, components={7: ["PCBAs", "Relays"]},
                        contacts={7: [{"designation": "Purchase Manager", "source_url": None, "email": None}]},
                        profiles={7: {"headquarters": headquarters} if headquarters else {}})


def claim_map(out):
    return {(a, c.get("field") or c.get("kind") or c.get("name") or c.get("supplier")): c.get("value", c.get("name"))
            for a, items in out["claims"].items() if isinstance(items, list) for c in items}


# ======================================================================================== catalog / request
def test_field_names_and_aliases_resolve_and_unknown_names_are_reported_not_guessed():
    keys, unknown = fl.resolve_fields(["HQ", "website", "Procurement Head", "revenue", "hq", "favourite colour"])
    assert keys == ["headquarters", "official_website", "procurement_contacts", "annual_revenue"]
    assert unknown == ["favourite colour"]
    spec = fl.BY_KEY["headquarters"]
    assert (spec.agent, spec.table, spec.column) == ("company", "company_profile", "headquarters")
    assert (fl.BY_KEY["official_website"].table, fl.BY_KEY["official_website"].column) == ("companies", "website")
    assert {f["key"] for f in fl.catalog_view()} == {f.key for f in fl.CATALOG}
    r = runner_for()
    with pytest.raises(ValueError, match="unknown_fields"):
        r.start("Apollo Micro Systems", None, fields=["headquarters", "nonsense"])


# ================================================================================== selective agent routing
@pytest.mark.parametrize("fields,agents,llm_calls", [
    (["headquarters"], {"company"}, 1),
    (["official_website", "headquarters", "business_overview"], {"company"}, 1),     # batched in ONE extraction call
    (["procurement_contacts"], {"company", "contact"}, 2),                           # identity is a prerequisite
    (["supplier_portal"], {"company", "procurement"}, 2),
    (["suppliers"], {"company", "supplier"}, 2),                                     # the CRM already lists components
    (["lead_score"], {"lead"}, 0),                                                   # CRM record + deterministic score
    (ALL_FIELDS, {"company", "contact", "procurement", "supplier", "lead"}, 4),     # lead needs no model call
])
def test_only_the_agents_the_requested_fields_need_run_and_llm_calls_stay_minimal(fields, agents, llm_calls):
    r, out, llm = run(fields)
    assert set(out["agent_runs"]) == agents
    assert len(llm_events(out)) == llm_calls and not llm_events(out, "supervisor")      # Python plans; no routing calls
    assert out["status"] == "completed", out["status_reasons"]
    assert out["report"]["change_set"]["requested_fields"] == fl.resolve_fields(fields)[0]


def test_suppliers_for_a_company_without_known_components_add_the_procurement_agent_first():
    r, out, _ = run(["suppliers"], repo=InMemoryRepo([COMPANY_A]))
    assert list(out["agent_runs"]) == ["company", "procurement", "supplier"]
    stages = [e["data"]["next"] for e in events(out, "route")]
    assert stages[:3] == ["company", "procurement", "supplier"]


def test_lead_score_for_a_company_outside_the_crm_runs_company_first():
    r, out, llm = run(["lead_score"], repo=InMemoryRepo([]))
    assert set(out["agent_runs"]) == {"company", "lead"} and len(llm_events(out)) == 1


def test_a_requested_field_only_company_run_focuses_the_task_on_those_fields():
    r, out, llm = run(["headquarters"])
    task = next(c for c in llm.calls if c["agent"] == "company")["messages"][0]["content"]
    assert "REQUESTED FIELDS" in task and "headquarters" in task
    assert "annual_revenue" not in task and "founded_year" not in task                # unrelated gaps are not asked about


# =========================================================================== refresh vs retrieve (old values)
def test_stored_company_fields_are_not_researched_again_unless_refresh_is_requested():
    repo = crm_repo(website="https://old.example", headquarters="Pune, India")
    r, out, llm = run(["official_website", "headquarters"], repo=repo)
    assert out["agent_runs"] == {} and len(llm_events(out)) == 0 and out["budgets"]["search_credits"] == 0
    assert out["status"] == "completed" and out["stop_reason"] == "nothing_to_research"
    cs = out["report"]["change_set"]
    assert cs["items"] == [] and {s["field"] for s in cs["skipped"]} == {"official_website", "headquarters"}
    assert all(s["reason"] == "already_stored_not_refreshed" and s["old_value"] for s in cs["skipped"])


def test_refresh_re_verifies_and_the_change_set_shows_old_vs_proposed_with_overwrite_flags():
    repo = crm_repo(website="https://old.example", headquarters="Aurangabad, India")
    r, out, _ = run(["official_website", "headquarters"], refresh=True, repo=repo)
    items = {i["field"]: i for i in out["report"]["change_set"]["items"]}
    site, hq = items["official_website"], items["headquarters"]
    assert site["action"] == "update_conflict" and site["overwrite"] and site["requires_approval"]
    assert site["old_value"] == "https://old.example" and site["proposed_value"].startswith("https://www.apollomicrosystems")
    assert site["target"] == "companies.website"
    assert hq["action"] == "no_change" and not hq["requires_approval"]               # same value: nothing to approve
    assert out["report"]["change_set"]["writes_performed"] is False


# ===================================================================================== parallel execution
def test_independent_agents_of_a_stage_really_run_concurrently(monkeypatch):
    spans, lock = [], threading.Lock()
    orig = GuardedExecutor.execute

    def slow(self, **kw):
        t0 = time.monotonic()
        res = orig(self, **kw)
        if kw["agent"] in ("contact", "procurement") and kw["call"].name == "web_search":
            time.sleep(0.25)
            with lock:
                spans.append((kw["agent"], t0, time.monotonic()))
        return res
    monkeypatch.setattr(GuardedExecutor, "execute", slow)
    r, out, _ = run(["procurement_contacts", "supplier_portal"])
    a, b = sorted(spans, key=lambda s: s[1])
    assert {a[0], b[0]} == {"contact", "procurement"} and b[1] < a[2]                # the second started before the first ended
    spans.clear()
    run(["procurement_contacts", "supplier_portal"], eff=HYBRID)                     # parallel switch off -> sequential
    a, b = sorted(spans, key=lambda s: s[1])
    assert b[1] >= a[2]


def test_parallel_and_sequential_runs_produce_the_same_verified_results_in_a_reproducible_trace():
    _, par, _ = run(ALL_FIELDS)
    _, seq, _ = run(ALL_FIELDS, eff=HYBRID)
    assert claim_map(par) == claim_map(seq)
    assert {(i["field"], i["action"]) for i in par["report"]["change_set"]["items"]} == \
           {(i["field"], i["action"]) for i in seq["report"]["change_set"]["items"]}
    assert [e["seq"] for e in par["trace"]] == list(range(1, len(par["trace"]) + 1))      # one continuous sequence
    assert [e["data"]["agents"] for e in events(par, "parallel_start")] == [["contact", "procurement"]]
    _, again, _ = run(ALL_FIELDS)
    assert [(e["type"], e["agent"]) for e in again["trace"]] == [(e["type"], e["agent"]) for e in par["trace"]]


def test_a_parallel_stage_merges_each_agents_own_results_without_overwriting_the_others():
    _, out, _ = run(["procurement_contacts", "supplier_portal"])
    assert {c["name"] for c in out["claims"]["contact"]} and out["claims"]["procurement"]
    assert out["agent_results"]["contact"]["accepted"] == 1 and out["agent_results"]["procurement"]["accepted"] >= 1
    assert set(out["completed_agents"]) == {"company", "contact", "procurement"}
    assert out["agent_runs"] == {"company": 1, "contact": 1, "procurement": 1}


def test_identical_concurrent_searches_are_paid_once():
    w = world()

    class Slow(FixtureSearch):
        def search(self, query, depth, max_results):
            time.sleep(0.2)
            return super().search(query, depth, max_results)
    r = runner_for(w)
    r.deps.search = Slow(w)
    ex = GuardedExecutor(r.deps.registry, r.deps)
    call = ToolCallRequest(id="c", name="web_search", args={"query": "Apollo Micro Systems official website"})

    def go(_):
        b = new_budgets()
        res = ex.execute(agent="company", call=call, budgets=b, ledger={}, subject={})
        return res.ok, b["search_credits"]
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(go, range(2)))
    assert all(ok for ok, _ in results) and sorted(c for _, c in results) == [0, 1]
    assert len(r.deps.search.calls) == 1 and r.store.credits_used() == 1


# ======================================================================================== shared evidence
def test_later_agents_see_what_was_already_retrieved_and_nothing_is_fetched_twice():
    r, out, llm = run(["procurement_contacts", "supplier_portal"])
    contact_task = next(c for c in llm.calls if c["agent"] == "contact")["messages"][0]["content"]
    assert "Documents already retrieved in this run" in contact_task                  # evidence from the company stage
    assert len(r.fetcher.calls) == len(set(r.fetcher.calls))                          # the store cache served repeats
    urls = {d["url"] for d in out["ledger"].values()}
    cited = {e["url"] for k, v in out["claims"].items() if isinstance(v, list) for c in v for e in c["evidence"]}
    assert cited <= urls                                                              # every citation is a ledger document
    assert any(c["tool"] for c in out["ledger"].values())


def test_ledger_merge_is_a_union_deduplicated_by_canonical_url():
    from agentic import evidence as ev
    base, other = {}, {}
    ev.add_doc(base, url="https://a.example/", text="alpha " * 30, title="A", tool="fetch_page", call_id="1")
    ev.add_doc(other, url="https://a.example", text="alpha " * 30 + "beta", title="A", tool="fetch_page", call_id="2")
    ev.add_doc(other, url="https://b.example/x", text="gamma " * 30, title="B", tool="web_search", call_id="3")
    merged = parallel.merge_ledgers(base, [other, base], max_docs=40, max_chars=20000)
    assert len(merged) == 2 and sorted(d["url"] for d in merged.values()) == ["https://a.example", "https://b.example/x"]
    a_doc = next(d for d in merged.values() if d["url"] == "https://a.example")
    assert a_doc["text"].endswith("beta") and len(base) == 1 and not base[next(iter(base))]["text"].endswith("beta")


# ================================================================================================ budgets
def test_branch_budgets_are_partitioned_so_parallel_agents_cannot_overspend():
    lim = Limits()
    base = new_budgets()
    base.update(llm_calls=5, tool_calls=3, search_credits=1, input_tokens=1000)
    parts = parallel.split_budgets(base, lim, 3)
    cap_credits = min(lim.max_search_credits_run, lim.max_search_credits_company)
    remaining = [cap_credits - b["search_credits"] for b, _ in parts]
    assert sum(remaining) == cap_credits - 1 and all(x >= 0 for x in remaining)         # shares add up to what is left
    assert sum(lim.max_llm_calls_run - b["llm_calls"] for b, _ in parts) == lim.max_llm_calls_run - 5
    # branches report real usage; the merge adds only that
    parts[0][0]["search_credits"] += 1
    parts[1][0]["llm_calls"] += 2
    merged = parallel.merge_budgets(base, parts)
    assert merged["search_credits"] == 2 and merged["llm_calls"] == 7 and merged["tool_calls"] == 3


def test_search_credit_cap_holds_across_parallel_agents():
    lim = Limits(max_search_credits_company=2)                       # company stage spends 1; one credit for two agents
    r, out, _ = run(["procurement_contacts", "supplier_portal"], limits=lim)
    assert out["budgets"]["search_credits"] <= 2 and r.store.credits_used() <= 2
    reasons = [e["data"]["reason"] for e in events(out, "prefetch")]
    assert "search_failed:credit_budget_exhausted" in reasons                        # the starved agent degraded, not bypassed


def test_run_level_llm_call_cap_stops_the_whole_field_run():
    r, out, _ = run(["procurement_contacts", "supplier_portal"], limits=Limits(max_llm_calls_run=2))
    assert len(llm_events(out)) <= 2 and out["stop_reason"] == "max_llm_calls_reached" and out["status"] == "partial"


# ===================================================================================== partial failures
def test_one_agent_failing_does_not_stop_the_others_and_the_run_is_partial():
    llm = FieldLLM(fail={"contact"})
    r, out, _ = run(ALL_FIELDS, llm=llm)
    assert out["agent_results"]["contact"]["end_reason"] == "llm_invalid"
    assert out["claims"]["procurement"] and out["claims"]["supplier"] and out["claims"]["lead"]["score"] is not None
    assert out["status"] == "partial" and "agent_not_finished:contact:llm_invalid" in out["status_reasons"]
    cs = out["report"]["change_set"]
    assert {u["field"] for u in cs["unresolved"]} == {"procurement_contacts"}
    assert {i["field"] for i in cs["items"]} >= {"headquarters", "supplier_portal", "suppliers"}


def test_an_exception_inside_a_parallel_branch_is_isolated_and_recorded():
    def boom(state, findings, deps):
        raise RuntimeError("scripted failure")
    specs = dict(AGENT_SPECS)
    specs["procurement"] = dataclasses.replace(AGENT_SPECS["procurement"], process=boom)
    r, out, _ = run(["procurement_contacts", "supplier_portal"], specs=specs)
    assert out["agent_results"]["procurement"]["end_reason"] == "branch_exception"
    assert {"stage": "procurement_branch", "error": "RuntimeError"} in out["errors"]
    assert out["claims"]["contact"] and out["agent_results"]["contact"]["status"] == "satisfied"
    assert out["status"] == "partial"


def test_unconfirmed_identity_blocks_the_downstream_agent_and_the_run_says_so():
    r, out, _ = run(["procurement_contacts"], w=world("no_website"), llm=FieldLLM("no_claims"))
    assert "contact" not in out["agent_runs"] and out["stop_reason"] == "planned_agent_blocked"
    assert out["status"] == "partial" and "limit:planned_agent_blocked" in out["status_reasons"]
    assert out["report"]["change_set"]["items"] == [] and out["report"]["change_set"]["unresolved"]


# ================================================================================= change-set rules (unit)
def state_with(claims, rejected=None, fields=("headquarters",), refresh=False, existing=None):
    return {"options": {"fields": list(fields), "refresh": refresh},
            "subject": {"name": "Acme", "existing": existing}, "claims": claims, "rejected": rejected or {}}


def verified(field, value, conf=0.6, url="https://acme.example/about"):
    return {"agent": "company", "field": field, "value": value, "status": "verified", "confidence": conf,
            "confidence_basis": "first-party", "evidence": [{"url": url, "snippet": f"{field} is {value}.",
                                                              "retrieved_at": "2026-10-12T00:00:00+00:00"}]}


def test_change_set_items_carry_old_proposed_sources_confidence_and_validation():
    existing = {"company_id": 7, "fields": {}, "profile": {}}
    cs = cs_mod.build_change_set(state_with({"company": [verified("headquarters", "Pune, India")]}, existing=existing))
    (item,) = cs["items"]
    assert item["action"] == "fill_empty" and item["old_value"] is None and item["proposed_value"] == "Pune, India"
    assert item["target"] == "company_profile.headquarters" and item["requires_approval"] and not item["overwrite"]
    assert item["source_urls"] == ["https://acme.example/about"] and item["confidence"] == 0.6
    assert item["validation_status"] == "verified" and item["evidence"][0]["snippet"]
    assert cs["writes_performed"] is False and cs["review_required"] is True


def test_unsupported_rejected_or_conflicting_values_never_become_proposals_and_old_values_stay():
    existing = {"company_id": 7, "fields": {}, "profile": {"headquarters": "Pune, India"}}
    rej = {"company": [{"field": "headquarters", "value": "Mars", "reasons": ["snippet_not_found_in_document"]}]}
    cs = cs_mod.build_change_set(state_with({"company": []}, rej, refresh=True, existing=existing))
    assert cs["items"] == []
    assert cs["unresolved"][0]["reason"] == "rejected_by_validation"
    assert cs["unresolved"][0]["rejection_reasons"] == ["snippet_not_found_in_document"]
    assert cs["unresolved"][0]["old_value"] == "Pune, India"                           # nothing overwrote it
    two = {"company": [verified("headquarters", "Pune, India"), verified("headquarters", "Delhi, India", 0.8)]}
    cs = cs_mod.build_change_set(state_with(two, existing=existing, refresh=True))
    assert cs["items"] == [] and cs["unresolved"][0]["reason"] == "conflicting_evidence"
    assert len(cs["unresolved"][0]["candidates"]) == 2


def test_a_different_non_empty_crm_value_is_flagged_as_an_overwrite_needing_approval():
    existing = {"company_id": 7, "fields": {}, "profile": {"headquarters": "Pune, India"}}
    cs = cs_mod.build_change_set(state_with({"company": [verified("headquarters", "Delhi, India")]}, existing=existing,
                                            refresh=True))
    (item,) = cs["items"]
    assert item["action"] == "update_conflict" and item["overwrite"] and item["old_value"] == "Pune, India"


# ======================================================================================== API contract
def test_api_accepts_fields_and_serves_the_catalog_and_the_change_set():
    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)
    svc = AgenticService(runner_for(), "test", Settings())
    svc_mod.configure_service(svc)
    try:
        cat = client.get("/agentic/fields").json()["fields"]
        assert {"headquarters", "procurement_contacts", "lead_score"} <= {f["key"] for f in cat}
        r = client.post("/agentic/runs", json={"company_name": "Apollo Micro Systems", "fields": ["hq", "contacts"]})
        assert r.status_code == 202
        run_id = r.json()["run_id"]
        svc.wait(run_id, 60)
        cs = client.get(f"/agentic/runs/{run_id}/changeset").json()
        assert cs["requested_fields"] == ["headquarters", "procurement_contacts"] and cs["writes_performed"] is False
        assert {i["field"] for i in cs["items"]} == {"headquarters", "procurement_contacts"}
        bad = client.post("/agentic/runs", json={"company_name": "Apollo Micro Systems", "fields": ["nonsense"]})
        assert bad.status_code == 422 and "unknown_fields" in bad.json()["detail"]
        assert client.get("/agentic/runs/nope/changeset").status_code == 404
        # legacy request (no fields) still works
        legacy = client.post("/agentic/runs", json={"company_name": "Apollo Micro Systems", "objective": "company profile"})
        assert legacy.status_code == 202
        svc.wait(legacy.json()["run_id"], 60)
    finally:
        svc_mod.configure_service(None)


# ================================================================================= provider compatibility
@pytest.mark.parametrize("provider,model", [(c.provider, c.model) for c in compat._CAPS])
def test_every_registered_provider_model_passes_the_offline_tool_calling_check(provider, model):
    res = compat.verify_tool_calling(provider, model)
    assert res["ok"] is True and res["violations"] == [] and all(c["ok"] for c in res["checks"])
    assert {c["check"] for c in res["checks"]} >= {"auto_tool_choice_with_two_tools", "required_single_tool",
                                                   "tool_result_round_trip"}
    assert ("thought_signature_replay" in {c["check"] for c in res["checks"]}) == model.startswith("gemini-3")
    assert compat.capability(provider, model).tool_calling


def test_qwen_on_groq_is_available_but_not_claimed_as_live_verified():
    cap = compat.capability("groq", "qwen/qwen3-32b")
    assert cap and cap.tool_calling and cap.verified_live is False
    assert compat.capability("gemini", "gemini-3-flash-preview").verified_live is True


def test_unlisted_models_and_unknown_providers_are_refused_with_no_fallback(monkeypatch, tmp_path):
    monkeypatch.setenv("GEMINI_API_KEY", "dummy-not-a-key")
    w = world()
    f = {"search": lambda: FixtureSearch(w), "fetcher": lambda: FixtureFetcher(w), "repo": default_repo}
    base = {"AGENTIC_MODE": "live", "AGENTIC_ALLOW_LIVE": "1", "AGENTIC_DATA_DIR": str(tmp_path),
            "AGENTIC_QUOTA": "rpm=5,tpm=250000,rpd=20"}
    with pytest.raises(ServiceNotConfigured, match="not in the tool-calling registry"):
        build_live_service(env={**base, "AGENTIC_MODEL": "gemini-9-imaginary"}, factories=f)
    ok = build_live_service(env={**base, "AGENTIC_MODEL": "gemini-9-imaginary", "AGENTIC_ALLOW_UNLISTED_MODEL": "1"},
                            factories=f)
    assert ok.runner.deps.llm.model == "gemini-9-imaginary"
    with pytest.raises(ServiceNotConfigured, match="no automatic fallback|not implemented"):
        build_live_service(env={**base, "AGENTIC_LLM_PROVIDER": "openai"}, factories=f)


def test_the_providers_endpoint_reports_capability_and_the_offline_check():
    app = FastAPI()
    app.include_router(router)
    svc = AgenticService(runner_for(llm=GeminiClient(model="gemini-3-flash-preview",
                                                     client=GeminiWire(FieldLLM()), thinking_level="low")), "test", Settings())
    svc_mod.configure_service(svc)
    try:
        body = TestClient(app).get("/agentic/providers").json()
        assert body["configured"]["provider"] == "gemini" and body["configured"]["offline_check"]["ok"] is True
        assert body["configured"]["capability"]["verified_live"] is True
        assert {m["model"] for m in body["registry"]} >= {"openai/gpt-oss-120b", "qwen/qwen3-32b"}
    finally:
        svc_mod.configure_service(None)


@pytest.mark.parametrize("make", [
    lambda d: GeminiClient(model="gemini-3-flash-preview", client=GeminiWire(d), thinking_level="low"),
    lambda d: GroqClient(model="openai/gpt-oss-120b", client=GroqWire(d)),
    lambda d: GroqClient(model="qwen/qwen3-32b", client=GroqWire(d)),
])
def test_a_full_field_run_gives_the_same_verified_results_through_every_provider_adapter(make):
    decider = FieldLLM()
    client = make(decider)
    r = runner_for(llm=client)
    out = r.start("Apollo Micro Systems", None, fields=ALL_FIELDS)
    wire = client._c
    assert wire.violations == [] and out["status"] == "completed"
    assert set(out["agent_runs"]) == {"company", "contact", "procurement", "supplier", "lead"}
    ref = run(ALL_FIELDS)[1]
    assert claim_map(out) == claim_map(ref)
    assert wire.requests == 4                                                      # the same minimal call count

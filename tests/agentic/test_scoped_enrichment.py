"""Scoped Company + Procurement run, procurement meaning rules, the narrow additive write phase, the counted write client
and the two-agent preflight. Offline only: synthetic worlds, scripted models, in-memory commit repository, no network."""

import copy
import time

import pytest

from agentic import apply_validated as av
from agentic import company_live
from agentic import field_support
from agentic import fields as fl
from agentic import preflight as pf
from agentic.agents import AGENT_SPECS
from agentic.backends.fixtures import FixtureFetcher, FixtureSearch
from agentic.commit import InMemoryCommitRepo
from agentic.config import Efficiency, Limits, Settings
from agentic.guards.select_budget import CountedClient, SelectBudget, SelectBudgetExceeded
from agentic.persistence.store import Store
from agentic.proposals import ProposalService
from agentic.repo import InMemoryRepo
from agentic.runtime import AgenticRunner, make_deps
from agentic.schema_contract import SchemaContract
from tests.agentic.fieldllm import FieldLLM
from tests.agentic.support import COMPANY_A
from tests.agentic.test_company_only import QUOTA, fresh_db, seed, settings_for
from tests.agentic.test_hybrid import llm_events
from tests.agentic.test_prefetch import events, world

SCOPE = ["company", "procurement"]
REAL_SCHEMA = "agentic/schema_introspection.json"


def tables(profile=None, website=None):
    return {"companies": [{"company_id": 7, "company_name": "Apollo Micro Systems", "website": website,
                           "linkedin_url": None, "company_type": "Private", "country": "India", "sector_id": 1}],
            "company_profile": [dict({"profile_id": 1, "company_id": 7}, **(profile or {}))] if profile is not None else [],
            "contacts": [], "ai_insights": [], "suppliers": [], "company_suppliers": []}


def runner(llm, *, corrections=0, **limits):
    lim = Limits(llm_max_retries=0, company_correction_requests=corrections, **limits)
    repo = InMemoryRepo([COMPANY_A], products={7: ["Avionics"]}, components={7: ["PCBAs"]})
    w = world()
    search, fetch = FixtureSearch(w), FixtureFetcher(w)
    deps = make_deps(llm=llm, repo=repo, search=search, fetcher=fetch, store=Store(":memory:"),
                     settings=Settings(limits=lim, efficiency=Efficiency(lazy_crm=True)), specs=AGENT_SPECS,
                     sleep=lambda s: None)
    r = AgenticRunner(deps, AGENT_SPECS)
    r.search = search
    return r


def run(llm, **kw):
    r = runner(llm, **kw)
    return r, r.start("Apollo Micro Systems", scope_agents=SCOPE, expected_company_id=7)


class MixedLLM(FieldLLM):
    """Honest submissions plus one fabricated item per agent (forces a rejection -> a correction request)."""

    def generate(self, **kw):
        resp = super().generate(**kw)
        for call in resp.tool_calls:
            if call.name == "submit_company_findings":
                call.args.setdefault("claims", []).append({"field": "employee_count", "value": "9,999",
                    "evidence": [{"url": "https://www.apollomicrosystems.example/", "snippet": "We employ 9,999 people worldwide."}]})
            if call.name == "submit_procurement_findings":
                call.args.setdefault("items", []).append({"kind": "purchased_component", "value": "Titanium",
                    "evidence": [{"url": "https://www.apollomicrosystems.example/procurement", "snippet": "We buy titanium."}]})
        return resp


# ============================================================================================ the scoped run
def test_company_then_procurement_with_one_request_each_and_nobody_else():
    llm = FieldLLM()
    r, out = run(llm)
    assert [c["agent"] for c in llm.calls] == ["company", "procurement"]
    assert out["agent_runs"] == {"company": 1, "procurement": 1} and out["budgets"]["llm_calls"] == 2
    assert [c["tools"] for c in llm.calls] == [["submit_company_findings"], ["submit_procurement_findings"]]
    assert events(out, "llm_call", agent="supervisor") == []
    assert [h["next"] for h in out["route_history"]] == ["company", "procurement", "finish"]
    kinds = {c["kind"] for c in out["claims"]["procurement"]}
    assert {"supplier_portal", "purchasing_category"} <= kinds
    assert not {"contact", "supplier", "lead"} & set(out["agent_runs"])
    assert out["budgets"]["search_credits"] <= 2 and len(r.search.calls) == 2         # one search per agent
    assert out["report"]["change_set"]["writes_performed"] is False and out["proposals"] == []


def test_procurement_only_runs_after_the_identity_is_confirmed():
    llm = FieldLLM(mode="no_claims")                       # the company agent confirms nothing
    r, out = run(llm)
    assert out["agent_runs"] == {"company": 1} and len(llm.calls) == 1
    assert out["stop_reason"] == "planned_agent_blocked" and out["status"] == "partial"


def test_total_requests_across_both_agents_are_capped_by_the_plan():
    llm = MixedLLM()
    r, out = run(llm, corrections=1)
    assert len(llm.calls) == 4 and out["budgets"]["llm_calls"] == 4                  # 2 agents x (1 + 1 correction)
    llm0 = MixedLLM()
    r0, out0 = run(llm0)
    assert len(llm0.calls) == 2                                                      # no correction in the default plan
    llm3 = MixedLLM()
    r3, out3 = run(llm3, corrections=1, max_llm_calls_run=3)
    assert len(llm3.calls) == 3 and out3["budgets"]["llm_calls"] == 3
    rejected = {x["field"] if "field" in x else x["kind"] for x in out["rejected"]["company"] + out["rejected"]["procurement"]}
    assert {"employee_count", "purchased_component"} <= rejected


def test_a_scope_must_contain_company_and_only_known_agents():
    r = runner(FieldLLM())
    with pytest.raises(ValueError, match="invalid_scope"):
        r.start("Apollo Micro Systems", scope_agents=["procurement"])
    with pytest.raises(ValueError, match="company_only_allows_company_fields"):
        r.start("Apollo Micro Systems", scope_agents=SCOPE, fields=["procurement_contacts"])


# ============================================================================================ procurement meaning
@pytest.mark.parametrize("kind,value,quote,ok", [
    ("supplier_portal", "https://x.example/s", "Prospective suppliers can register through our supplier portal.", True),
    ("supplier_portal", "https://x.example/s", "Visit our careers portal to apply for jobs.", False),
    ("procurement_page", "https://x.example/p", "Procurement and sourcing at Airbus.", True),
    ("procurement_page", "https://x.example/p", "Our people and culture.", False),
    ("purchased_component", "aluminium", "Airbus sources aluminium from certified vendors.", True),
    ("purchased_component", "aluminium", "We sell aluminium to our customers.", False),
    ("purchased_component", "aluminium", "Aluminium is a light metal.", False),
    ("purchasing_category", "raw materials", "Procurement categories: raw materials, electronics.", True),
    ("process_note", "qualification audit", "Vendors must pass a qualification audit.", True),
    ("process_note", "great culture", "Our culture is great.", False),
])
def test_procurement_kinds_need_a_quote_that_says_what_they_claim(kind, value, quote, ok):
    assert field_support.check(kind, value, [quote])[0] is ok


# ============================================================================================ write phase
def apply(out, commit, *, schema=None, store=None, identity=True, company_id=7):
    svc = ProposalService(store or Store(":memory:"), commit, schema=schema)
    return av.apply_reviewed(svc, out, out["report"]["change_set"], company_id, authoriser="tester", identity_verified=identity)


def test_only_validated_additive_items_of_company_7_are_written():
    llm = FieldLLM()
    r, out = run(llm)
    commit = InMemoryCommitRepo(tables(profile={"headquarters": None}))
    res = apply(out, commit, schema=SchemaContract.from_file(REAL_SCHEMA))
    assert res["writes"] > 0 and all(p["applied"] for p in res["proposals"]), res["proposals"]
    ops = commit.ops
    assert {o[1] for o in ops} <= {"companies", "company_profile", "ai_insights"}
    assert {o[0] for o in ops} <= {"insert", "update"} and ("insert", "companies") not in {(o[0], o[1]) for o in ops}
    assert all((o[3] if o[0] == "update" else o[2]).get("company_id", 7) == 7 if len(o) > 2 else True for o in ops)
    insights = commit.tables["ai_insights"]
    assert insights and all(i["company_id"] == 7 and i["source_url"].startswith("https://") and i["insight"] for i in insights)
    assert commit.tables["companies"][0]["website"] and commit.tables["company_profile"][0]["headquarters"]
    assert not any(k in res for k in ("contacts", "suppliers")) and len(commit.tables["companies"]) == 1
    again = apply(out, commit, schema=SchemaContract.from_file(REAL_SCHEMA))              # idempotent: nothing more is written
    assert again["writes"] == 0


def test_a_populated_crm_value_is_never_overwritten():
    llm = FieldLLM()
    r = runner(llm)
    r.deps.repo._profiles[7] = {"headquarters": "Mumbai, India"}
    out = r.start("Apollo Micro Systems", scope_agents=SCOPE, expected_company_id=7)
    cs = out["report"]["change_set"]
    hq = [i for i in cs["items"] if i["field"] == "headquarters"]
    assert not hq or hq[0]["action"] in ("update_conflict", "no_change")            # never presented as an empty-field fill
    commit = InMemoryCommitRepo(tables(profile={"headquarters": "Mumbai, India"}))
    res = apply(out, commit)
    assert all(i["field"] != "headquarters" for p in res["proposals"] for i in p["items"])
    assert not hq or {s["field"]: s["reason"] for s in res["skipped_items"]}["headquarters"].startswith("action_not_additive")
    assert commit.tables["company_profile"][0]["headquarters"] == "Mumbai, India"
    assert all("headquarters" not in (o[3] if o[0] == "update" else o[2]) for o in commit.ops if o[1] == "company_profile")


def test_profile_insert_satisfies_the_real_not_null_columns():
    llm = FieldLLM()
    r, out = run(llm)
    commit = InMemoryCommitRepo(tables(profile=None))                                    # no profile row yet -> INSERT
    res = apply(out, commit, schema=SchemaContract.from_file(REAL_SCHEMA))
    assert all(p["applied"] for p in res["proposals"]), res["proposals"]
    assert commit.tables["company_profile"] and commit.tables["company_profile"][0]["company_id"] == 7


def item(**kw):
    base = {"id": "i1", "field": "headquarters", "agent": "company", "entity": "company", "target": "company_profile.headquarters",
            "action": "fill_empty", "old_value": None, "proposed_value": "Pune", "overwrite": False, "confidence": 0.6,
            "validation_status": "verified", "source_urls": ["https://a.example/"],
            "evidence": [{"url": "https://a.example/", "snippet": "The company is headquartered in Pune."}]}
    base.update(kw)
    return base


def cs(*items, cid=7):
    return {"company_id": cid, "items": list(items), "writes_performed": False}


@pytest.mark.parametrize("change,reason", [
    (dict(action="update_conflict", old_value="X", overwrite=True), "action_not_additive"),
    (dict(old_value="Mumbai"), "crm_field_already_populated"),
    (dict(confidence=0.55), "low_confidence"),
    (dict(agent="contact", entity="contact", target="contacts"), "agent_or_entity_out_of_scope"),
    (dict(agent="supplier", entity="supplier", target="company_suppliers"), "agent_or_entity_out_of_scope"),
    (dict(target="contacts.email"), "table_not_authorised"),
    (dict(target=None), "table_not_authorised"),
    (dict(source_urls=[]), "missing_validation_or_source"),
    (dict(validation_status="computed"), "missing_validation_or_source"),
    (dict(evidence=[{"url": "https://a.example/", "snippet": "Our team in Pune supports customers."}]), "quote_does_not_state_field"),
    (dict(field="official_website", target="companies.website", proposed_value="https://a.example/en/about/team/india"),
     "website_is_a_deep_page_not_a_site"),
    (dict(field="official_website", target="companies.website", proposed_value="https://a.example/?id=3"), "url_has_query_or_fragment"),
    (dict(field="linkedin_url", target="companies.linkedin_url", proposed_value="https://www.linkedin.com/in/someone"),
     "not_a_linkedin_company_page"),
])
def test_items_outside_the_authorisation_are_skipped_with_a_reason(change, reason):
    sel = av.select_items(cs(item(**change)), 7, identity_verified=True)
    assert sel.item_ids == [] and sel.skipped[0]["reason"].startswith(reason)


def test_selection_rules_for_identity_company_and_insight_cap():
    assert av.select_items(cs(item()), 7, identity_verified=False).item_ids == []
    with pytest.raises(ValueError, match="change_set_company_mismatch"):
        av.select_items(cs(item(), cid=8), 7, identity_verified=True)
    many = [item(id=f"n{i}", field="purchasing_category", agent="procurement", target="ai_insights", action="add_new",
                 proposed_value=f"cat {i}", confidence=0.6 + i / 1000,
                 evidence=[{"url": "https://a.example/", "snippet": f"Procurement categories: cat {i}."}]) for i in range(14)]
    sel = av.select_items(cs(*many), 7, identity_verified=True)
    assert len(sel.item_ids) == av.MAX_PROCUREMENT_INSIGHTS and len(sel.skipped) == 4
    assert all(s["reason"] == "insight_cap_reached" for s in sel.skipped)


def test_proposal_scope_checks_refuse_deletes_other_companies_and_overwrites():
    ok = {"operations": [{"table": "company_profile", "op": "update", "key": {"company_id": 7}, "fields": {"headquarters": "x"}},
                         {"table": "ai_insights", "op": "insert", "fields": {"company_id": 7}}], "overwrites": []}
    assert av.check_proposal_scope(ok, 7) == []
    bad = {"operations": [{"table": "companies", "op": "delete", "key": {"company_id": 7}},
                          {"table": "company_profile", "op": "update", "key": {"company_id": 8}, "fields": {}},
                          {"table": "ai_insights", "op": "insert", "fields": {"company_id": 8}},
                          {"table": "contacts", "op": "insert", "fields": {"company_id": 7}},
                          {"table": "companies", "op": "insert", "fields": {"company_name": "x"}}],
           "overwrites": ["company_profile.headquarters"]}
    problems = " ".join(av.check_proposal_scope(bad, 7))
    for needle in ("contains_overwrite", "operation_not_authorised:delete:companies", "update_not_keyed_to_company",
                   "insert_for_other_company", "operation_not_authorised:insert:contacts", "operation_not_authorised:insert:companies"):
        assert needle in problems, needle


# ============================================================================================ counted write client
class SendLog:
    def __init__(self):
        self.sent = []

    def table(self, name):
        return Q(self, name)


class Q:
    def __init__(self, log, name, op="select"):
        self.log, self.name, self.op = log, name, op

    def __getattr__(self, attr):
        if attr in ("select", "insert", "update", "eq", "limit", "delete", "upsert", "rpc"):
            return lambda *a, **k: Q(self.log, self.name, attr if attr in ("insert", "update", "select") else self.op)
        raise AttributeError(attr)

    def execute(self):
        self.log.sent.append((self.op, self.name))
        return type("R", (), {"data": []})()


def test_the_write_client_counts_selects_before_sending_and_has_no_delete():
    log, budget = SendLog(), SelectBudget(3)
    c = CountedClient(log, budget)
    for _ in range(3):
        c.table("companies").select("*").eq("company_id", 7).execute()
    with pytest.raises(SelectBudgetExceeded):
        c.table("companies").select("*").execute()
    assert log.sent == [("select", "companies")] * 3                    # the fourth never left the process
    c.table("ai_insights").insert({"company_id": 7}).execute()          # writes are not counted as SELECTs
    c.table("company_profile").update({"headquarters": "x"}).eq("company_id", 7).execute()
    assert budget.used == 3 and log.sent[-2:] == [("insert", "ai_insights"), ("update", "company_profile")]
    for forbidden in ("delete", "upsert", "rpc"):
        with pytest.raises(PermissionError):
            getattr(c.table("companies"), forbidden)()


def test_apply_stops_and_reports_when_the_write_phase_select_cap_is_hit():
    llm = FieldLLM()
    r, out = run(llm)
    commit = InMemoryCommitRepo(tables(profile={"headquarters": None}))

    class Limited(InMemoryCommitRepo):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            self.budget = SelectBudget(2)

        def get_row(self, table, key):
            self.budget.reserve(table)
            return super().get_row(table, key)
    lim = Limited(tables(profile={"headquarters": None}))
    res = apply(out, lim)
    assert res["blocked"].startswith("write_phase_select_budget_exceeded") and lim.budget.used == 2


# ============================================================================================ two-agent preflight
ARGS = ["--company", "AIRBUS India", "--company-id", "11", "--agents", "company", "procurement", "--model", "gemini-2.5-flash",
        "--corrections", "1", "--max-requests", "4", "--max-credits", "2", "--apply", "--preflight-only",
        "--quota", "rpm=5,tpm=250000,rpd=20"]


def test_two_agent_plan_counts_every_permitted_request(tmp_path, capsys):
    s = settings_for(company_correction_requests=1, max_llm_calls_run=4)
    plan = pf.build_plan(s, agents=SCOPE, apply=True)
    assert plan.requests_max == 4 and plan.tavily_credits_max == 2 and plan.selects_max == 8 and plan.selects_min == 6
    assert plan.write_ops_max == 2 + 13 + 2 + pf.MAX_PROCUREMENT_INSIGHTS and plan.write_selects_cap == pf.WRITE_SELECTS_CAP
    assert set(plan.fields) == set(fl.agent_fields(SCOPE)) and plan.agents == SCOPE
    single = pf.build_plan(settings_for(), agents=["company"])
    assert plan.input_tokens_max > 2 * single.input_tokens_max * 0.9               # both agents and both corrections are in


def test_the_cli_preflight_for_the_two_agent_run_uses_the_models_own_usage(tmp_path, monkeypatch, capsys):
    seed_model = lambda n, key: [Store(str(tmp_path / "agentic.sqlite")).usage_add(key, 1000, time.time() - 60 - i) for i in range(n)]
    seed_model(16, "gemini:gemini-3-flash-preview")                      # the OTHER model's day is full: irrelevant here
    code = company_live.main(ARGS + ["--data-dir", str(tmp_path)])
    out = capsys.readouterr().out
    assert code == 0 and "PREFLIGHT OK" in out and "min 0, max 4" in out and "gemini-2.5-flash" in out
    assert "Tavily credits: max planned 2" in out and "at most 27 row operations on company 11" in out
    assert company_live.main([a if a != "4" else "3" for a in ARGS] + ["--data-dir", str(tmp_path)]) == 3
    blocked = capsys.readouterr().out
    assert pf.BLOCK_LINE in blocked and "minimum budget needed for this scope: 2" in blocked
    seed_model(14, "gemini:gemini-2.5-flash")                            # 14 used of a cap of 16 -> 2 left < 4 planned
    assert company_live.main(ARGS + ["--data-dir", str(tmp_path)]) == 3
    assert "exceeds the remaining daily allowance 2" in capsys.readouterr().out


def test_the_run_environment_pins_model_retries_timeout_and_caps():
    env = company_live.base_env("rpm=5,tpm=250000,rpd=20", "data", "gemini-2.5-flash", 1, 1024)
    plan = pf.build_plan(pf.settings_from_env(env), agents=SCOPE)
    env.update(pf.caps_for(plan))
    s = pf.settings_from_env(env)
    assert (s.provider, s.model, s.limits.llm_max_retries, s.limits.max_wall_clock_s) == ("gemini", "gemini-2.5-flash", 0, 420)
    assert (s.limits.max_llm_calls_run, s.limits.max_search_credits_run, s.limits.max_db_selects_run) == (4, 2, 8)
    assert env["AGENTIC_ENABLE_WRITES"] == "0" and env["AGENTIC_THINKING_BUDGET"] == "1024"
    assert "AGENTIC_THINKING_BUDGET" not in company_live.base_env("q", "d", "gemini-3-flash-preview", 0, 1024)


class CountingRepo(InMemoryCommitRepo):
    """Counts reads the way SupabaseCommitRepo does: get_row, find_one and list_rows are one SELECT each."""
    reads = 0

    def get_row(self, table, key):
        type(self).reads += 1
        return next((dict(r) for r in self.tables.get(table, []) if self._match(r, key)), None)

    def find_one(self, table, **eq):
        type(self).reads += 1
        return next((dict(r) for r in self.tables.get(table, []) if self._match(r, eq)), None)

    def list_rows(self, table, **eq):
        type(self).reads += 1
        return super().list_rows(table, **eq)


def test_a_full_write_phase_stays_inside_its_select_cap():
    r, out = run(FieldLLM())
    CountingRepo.reads = 0
    repo = CountingRepo(tables(profile={"headquarters": None}))
    res = apply(out, repo, schema=SchemaContract.from_file(REAL_SCHEMA))
    assert res["writes"] > 0
    assert CountingRepo.reads <= pf.WRITE_SELECTS_CAP - 3, CountingRepo.reads            # 3 more are kept for the read-back
    print("write-phase SELECTs used:", CountingRepo.reads, "writes:", res["writes"])

"""Per-run Supabase SELECT cap, enforced at the query-execution boundary (agentic/guards/select_budget.py), plus the
pre-request token check. Offline only: a fake supabase client records exactly which queries would have been sent."""

import contextvars
import types
from concurrent.futures import ThreadPoolExecutor

import pytest

from agentic.config import Limits, load_settings
from agentic.guards import select_budget as sb
from agentic.guards.select_budget import GuardedReadClient, SelectBudget, SelectBudgetExceeded
from agentic.repo import SupabaseReadRepo
from tests.agentic.fieldllm import FieldLLM
from tests.agentic.test_fields import runner_for
from tests.agentic.test_hybrid import llm_events
from tests.agentic.test_prefetch import events

FULL = ["official_website", "headquarters", "supplier_portal", "procurement_contacts", "suppliers", "lead_score"]


class FakeQuery:
    def __init__(self, client, table):
        self.client, self.table, self.filters, self.n = client, table, [], None

    def select(self, *_):
        return self

    def eq(self, k, v):
        self.filters.append(("eq", k, v))
        return self

    def ilike(self, k, pat):
        self.filters.append(("ilike", k, pat.strip("%").lower()))
        return self

    def limit(self, n):
        self.n = n
        return self

    def execute(self):
        self.client.executed.append(self.table)             # a query that really would have been sent
        rows = self.client.data.get(self.table, [])
        for op, k, v in self.filters:
            rows = [r for r in rows if (r.get(k) == v if op == "eq" else v in str(r.get(k, "")).lower())]
        return types.SimpleNamespace(data=[dict(r) for r in rows[: self.n]])


class FakeSupabase:
    def __init__(self, data=None):
        self.data, self.executed = data or {}, []

    def table(self, name):
        return FakeQuery(self, name)

    def rpc(self, *a, **k):
        raise AssertionError("rpc must never be reachable")


def crm(links=0):
    return {"companies": [{"company_id": 7, "company_name": "Apollo Micro Systems", "sector_id": 1, "country": "India",
                           "website": None, "linkedin_url": None, "company_type": None, "revenue": None,
                           "number_of_plants": 2, "enrichment_status": "Pending", "sector": None, "sub_sector": None}],
            "sectors": [{"sector_id": 1, "sector": "Aerospace & Defence"}],
            "company_profile": [], "products": [{"company_id": 7, "product_name": "Avionics"}],
            "components": [{"company_id": 7, "component_name": "Relays"}], "contacts": [],
            "company_suppliers": [{"company_id": 7, "supplier_id": i, "component": "x", "remarks": ""} for i in range(links)],
            "suppliers": [{"supplier_id": i, "supplier_name": f"S{i}", "website": None} for i in range(links)]}


def guarded(links=0):
    fake = FakeSupabase(crm(links))
    return fake, SupabaseReadRepo(GuardedReadClient(fake))


# ==================================================================================== the boundary itself
def test_eight_selects_are_allowed_and_the_ninth_is_blocked_before_it_is_sent():
    fake = FakeSupabase(crm())
    client = GuardedReadClient(fake)
    budget = SelectBudget(8)
    token = sb.activate(budget)
    try:
        for i in range(8):
            client.table("companies").select("*").eq("company_id", 7).limit(1).execute()
        assert budget.used == 8 and len(fake.executed) == 8
        with pytest.raises(SelectBudgetExceeded) as e:
            client.table("products").select("*").execute()
        assert e.value.limit == 8
        assert len(fake.executed) == 8 and budget.used == 8                       # the ninth never reached the client
        with pytest.raises(SelectBudgetExceeded):                                 # and stays blocked
            client.table("sectors").select("*").execute()
        assert len(fake.executed) == 8
    finally:
        sb.deactivate(token)


def test_repeated_identical_queries_and_every_chained_form_are_counted():
    fake = FakeSupabase(crm())
    client = GuardedReadClient(fake)
    token = sb.activate(SelectBudget(3))
    try:
        for _ in range(3):
            client.table("companies").select("company_id").ilike("company_name", "%apollo%").limit(100).execute()
        with pytest.raises(SelectBudgetExceeded):
            client.table("companies").select("company_id").ilike("company_name", "%apollo%").limit(100).execute()
        assert fake.executed == ["companies"] * 3
    finally:
        sb.deactivate(token)


def test_the_read_client_cannot_write_and_the_swallowing_handlers_cannot_hide_an_exhausted_budget():
    client = GuardedReadClient(FakeSupabase(crm()))
    for attempt in (lambda: client.table("companies").insert({"x": 1}), lambda: client.table("companies").update({"x": 1}),
                    lambda: client.table("companies").delete(), lambda: client.table("companies").upsert({}),
                    lambda: client.rpc("anything")):
        with pytest.raises(PermissionError):
            attempt()
    assert not issubclass(SelectBudgetExceeded, Exception)                        # `except Exception` cannot swallow it
    token = sb.activate(SelectBudget(0))
    try:
        try:
            try:
                client.table("companies").select("*").execute()
            except Exception:                                                     # the CRM code's 'DB outage' handler
                pytest.fail("the budget exception was swallowed by `except Exception`")
        except SelectBudgetExceeded:
            pass
    finally:
        sb.deactivate(token)


def test_without_an_active_run_budget_reads_are_not_counted_or_blocked():
    fake = FakeSupabase(crm())
    client = GuardedReadClient(fake)
    for _ in range(20):
        client.table("companies").select("*").execute()
    assert len(fake.executed) == 20 and sb.used() is None


def test_parallel_branches_share_one_budget_and_it_is_never_reset():
    fake = FakeSupabase(crm())
    client = GuardedReadClient(fake)
    budget = SelectBudget(8)
    token = sb.activate(budget)
    results = []

    def branch(n):
        ok = blocked = 0
        for _ in range(n):
            try:
                client.table("companies").select("*").execute()
                ok += 1
            except SelectBudgetExceeded:
                blocked += 1
        return ok, blocked
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:                          # exactly what parallel_run does
            futures = [pool.submit(contextvars.copy_context().run, branch, 6) for _ in range(2)]
            results = [f.result() for f in futures]
    finally:
        sb.deactivate(token)
    assert sum(ok for ok, _ in results) == 8 and sum(b for _, b in results) == 4
    assert len(fake.executed) == 8 and budget.used == 8


# ======================================================================= the real read path, helper by helper
def test_each_repository_helper_is_counted_including_the_per_link_supplier_lookups():
    fake, repo = guarded(links=2)
    token = sb.activate(SelectBudget(50))
    try:
        repo.find_candidates("Apollo Micro Systems")            # 1  companies
        repo.get_company(7)                                     # 2  companies, 3 sectors (first use)
        repo.get_company(7)                                     # 4  companies (sectors are cached)
        repo.get_profile(7)                                     # 5
        repo.get_products(7)                                    # 6
        repo.get_components(7)                                  # 7
        repo.get_contact_summary(7)                             # 8
        repo.get_company_suppliers(7)                           # 9 links + 2 supplier lookups = 11
        assert sb.used() == 11 and fake.executed.count("suppliers") == 2
    finally:
        sb.deactivate(token)


# ====================================================================== a whole run under the cap
def run(limit, links=0, fields=FULL, **kw):
    fake, repo = guarded(links)
    llm = FieldLLM()
    r = runner_for(llm=llm, repo=repo, limits=Limits(max_db_selects_run=limit), **kw)
    out = r.start("Apollo Micro Systems", None, fields=fields, refresh=False, expected_company_id=7)
    return fake, out, llm, r


def test_a_full_five_agent_run_uses_exactly_the_eight_planned_selects():
    fake, out, llm, _ = run(8)
    assert out["status"] == "completed" and set(out["agent_runs"]) == {"company", "contact", "procurement", "supplier", "lead"}
    assert out["budgets"]["db_selects"] == 8 and len(fake.executed) == 8
    assert fake.executed == ["companies", "companies", "sectors", "company_profile", "products", "contacts", "components",
                             "company_suppliers"] or sorted(fake.executed) == sorted(
        ["companies", "companies", "sectors", "company_profile", "products", "contacts", "components", "company_suppliers"])
    from agentic.guards.budget import usage_summary
    assert usage_summary(out["budgets"])["db_selects"] == 8


def test_the_ninth_select_of_a_real_run_is_refused_and_the_run_stops_with_a_recorded_reason():
    fake, out, llm, _ = run(8, links=1)                          # one supplier link needs a 9th lookup
    assert len(fake.executed) == 8 and out["budgets"]["db_selects"] == 8
    assert out["stop_reason"] == "db_select_budget_exceeded" and out["agent_runs"] == {} and llm.calls == []
    assert out["status"] == "partial" and "limit:db_select_budget_exceeded" in out["status_reasons"]
    assert out["budgets"]["search_credits"] == 0                                  # nothing was researched on a half-read CRM


def test_an_exhausted_budget_during_lookup_fails_the_run_instead_of_looking_like_a_new_company():
    fake, out, llm, r = run(2)
    assert out["status"] == "failed" and out["stop_reason"] == "db_select_budget_exceeded"
    assert len(fake.executed) == 2 and out["agent_runs"] == {} and r.search.calls == [] and llm.calls == []
    assert "subject" not in out or out["subject"].get("match", {}).get("status") != "not_found"


def test_every_run_gets_its_own_budget_but_agents_and_branches_within_a_run_share_one():
    fake, repo = guarded()
    r = runner_for(llm=FieldLLM(), repo=repo, limits=Limits(max_db_selects_run=8))
    a = r.start("Apollo Micro Systems", None, fields=FULL, expected_company_id=7)
    b = r.start("Apollo Micro Systems", None, fields=FULL, expected_company_id=7)
    assert a["budgets"]["db_selects"] == 8 and b["budgets"]["db_selects"] == 7        # sectors lookup is cached on the repo
    assert sb.used() is None                                                       # nothing leaks out of a run


def test_settings_and_live_wiring_put_the_guard_between_the_repo_and_the_client(monkeypatch, tmp_path):
    assert load_settings({"AGENTIC_MAX_DB_SELECTS": "8"}).limits.max_db_selects_run == 8
    assert load_settings({}).limits.max_db_selects_run is None
    import supabase
    monkeypatch.setattr(supabase, "create_client", lambda url, key: FakeSupabase(crm()))
    monkeypatch.setenv("SUPABASE_URL", "https://example.invalid")
    monkeypatch.setenv("SUPABASE_KEY", "dummy-not-a-key")
    from agentic import service as svc_mod
    from tests.agentic.test_bridge import factories
    f = factories()
    f.pop("repo")
    f["llm"] = lambda: FieldLLM()
    svc = svc_mod.build_live_service(env={"AGENTIC_MODE": "live", "AGENTIC_ALLOW_LIVE": "1", "AGENTIC_DATA_DIR": str(tmp_path),
                                          "AGENTIC_MAX_DB_SELECTS": "8"}, factories=f)
    repo = svc.runner.deps.repo
    assert isinstance(repo, SupabaseReadRepo) and isinstance(repo._c, GuardedReadClient)
    assert svc.settings.limits.max_db_selects_run == 8 and svc.runner.deps.commit_repo is None


# =============================================================== tokens are checked before the request
def test_a_request_that_could_take_the_run_past_its_token_cap_is_not_sent():
    llm = FieldLLM()
    r = runner_for(llm=llm, limits=Limits(max_total_tokens_run=2500))               # the company request alone needs ~4k
    out = r.start("Apollo Micro Systems", None, fields=["headquarters"])
    assert llm.calls == [] and out["stop_reason"] == "max_tokens_reached" or out["agent_results"]["company"]["end_reason"] == "max_tokens_reached"
    assert not llm_events(out)
    assert out["status"] == "partial"

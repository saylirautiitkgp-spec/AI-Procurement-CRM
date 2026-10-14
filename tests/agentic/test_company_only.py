"""Company-Agent-only optimisation, field-meaning validation, complete-plan preflight and atomic quota reservation.

Offline only: synthetic worlds, scripted models, no network, no provider/Tavily/Supabase call."""

import copy
import re
import socket
import threading

import pytest

from agentic import company_live
from agentic import field_support
from agentic import preflight as pf
from agentic import routing
from agentic.agents import AGENT_SPECS
from agentic.backends.fixtures import FixtureFetcher, FixtureSearch
from agentic.config import Efficiency, Limits, Settings, load_settings
from agentic.llm.base import LLMError, LLMResponse, Usage
from agentic.llm.fake import tool_call
from agentic.llm.pacing import PacedLLM, ProviderQuota
from agentic.persistence.store import Store
from agentic.runtime import AgenticRunner, make_deps
from tests.agentic.support import default_repo
from tests.agentic.test_hybrid import SpyRepo
from tests.agentic.test_prefetch import COMPANY, events, packet_json
from agentic.eval.synthetic import make_world

EXTRA = ("Since 2007, Apollo Micro Systems engineers have supported programmes in Bengaluru. "
         "Apollo Micro Systems is sourcing over USD 1.6 Billion of components from India. "
         "Our team in New Delhi and Bengaluru supports customers.\n")
FIELDS = ["official_website", "headquarters", "employee_count", "founded_year", "annual_revenue"]


def world(variant="standard", extra=EXTRA):
    w = make_world(COMPANY, variant, big_pages=True)
    for url, p in w["pages"].items():
        if url.rstrip("/").endswith(("apollomicrosystems.example", "/about")):
            p["text"] = extra + p["text"]
    return w


def find(packet, needle):
    """(url, full sentence containing `needle`) from the first packet document that has it."""
    for d in packet["documents"]:
        m = re.search(r"[^.\n]*" + re.escape(needle) + r"[^.\n]*\.", d["text"])
        if m:
            return d["url"], m.group(0).strip()
    raise AssertionError(f"{needle!r} not in the packet")


class SoloLLM:
    """Scripted model for the single-request company path. Records every request."""
    name, model = "solo-llm", "solo-model"

    def __init__(self, mode="multi"):
        self.mode, self.calls = mode, []

    def generate(self, *, system, messages, tools, max_output_tokens, force_tool=False):
        names = [t.name for t in tools]
        self.calls.append({"tools": names, "force": force_tool, "max_out": max_output_tokens,
                           "messages": copy.deepcopy(messages)})
        if self.mode == "empty":
            return LLMResponse(text="", usage=Usage(requests=1, input_tokens=100, output_tokens=1))
        if self.mode == "foreign_tool":
            return tool_call("submit_contact_findings", status="exhausted")
        if self.mode == "routes":
            return tool_call("route", next="contact", reason="x")
        packet = packet_json(messages[0]["content"])
        url = packet["documents"][0]["url"]
        if self.mode == "invalid":
            claims = [{"field": "headquarters", "value": "Pune, India",
                       "evidence": [{"url": url, "snippet": "Apollo Micro Systems headquarters are in Pune, India."}]}]
        elif self.mode == "semantic":
            claims = [
                {"field": "founded_year", "value": "2007",
                 "evidence": [{"url": u, "snippet": s}] } for u, s in [find(packet, "Since 2007")]
            ] + [
                {"field": "annual_revenue", "value": "USD 1.6 Billion",
                 "evidence": [{"url": u, "snippet": s}]} for u, s in [find(packet, "USD 1.6 Billion")]
            ] + [
                {"field": "headquarters", "value": "New Delhi and Bengaluru",
                 "evidence": [{"url": u, "snippet": s}]} for u, s in [find(packet, "team in New Delhi")]
            ]
        else:
            hq_url, hq = find(packet, "headquartered in")
            city = re.search(r"headquartered in ([A-Za-z ]+), ", hq).group(1)
            emp_url, emp = find(packet, "employs approximately")
            count = re.search(r"approximately ([\d,]+) people", emp).group(1)
            fo_url, founded = find(packet, "Founded in")
            year = re.search(r"Founded in (\d{4})", founded).group(1)
            site = packet["documents"][0]
            claims = [
                {"field": "official_website", "value": site["url"],
                 "evidence": [{"url": site["url"], "snippet": site["text"][:100]}]},
                {"field": "headquarters", "value": city, "evidence": [{"url": hq_url, "snippet": hq}]},
                {"field": "employee_count", "value": count, "evidence": [{"url": emp_url, "snippet": emp}]},
                {"field": "founded_year", "value": year, "evidence": [{"url": fo_url, "snippet": founded}]},
            ]
        return tool_call("submit_company_findings", status="partial", identity_website=url,
                         identity_reasoning="company name on its own site", claims=claims,
                         unresolved=["annual_revenue"])


def build(w, llm, *, limits=None, repo=None, store=None):
    settings = Settings(limits=limits or Limits(), efficiency=Efficiency(lazy_crm=True))
    search, fetch = FixtureSearch(w), FixtureFetcher(w)
    deps = make_deps(llm=llm, repo=repo if repo is not None else default_repo(), search=search, fetcher=fetch,
                     store=store or Store(":memory:"), settings=settings, specs=AGENT_SPECS, sleep=lambda s: None)
    r = AgenticRunner(deps, AGENT_SPECS)
    r.search, r.fetcher = search, fetch
    return r


def run(llm, w=None, limits=None, fields=None, repo=None):
    r = build(w or world(), llm, limits=limits, repo=repo)
    out = r.start("Apollo Micro Systems", fields=fields or FIELDS, company_only=True, expected_company_id=7)
    return r, out


# ================================================================================ request count / bypass
def test_company_only_makes_exactly_one_structured_request_and_extracts_several_fields():
    llm = SoloLLM("multi")
    r, out = run(llm)
    assert len(llm.calls) == 1 and out["budgets"]["llm_calls"] == 1
    assert llm.calls[0]["tools"] == ["submit_company_findings"] and llm.calls[0]["force"] is True
    assert llm.calls[0]["max_out"] == Limits().company_only_output_tokens           # room for many claims in one answer
    got = {c["field"] for c in out["claims"]["company"]}
    assert {"official_website", "headquarters", "employee_count", "founded_year"} <= got
    assert out["agent_runs"] == {"company": 1}
    assert events(out, "llm_call", agent="supervisor") == []                        # no supervisor model call
    assert not events(out, "packet_followup") and out["agent_results"]["company"]["end_reason"] == "submitted"
    cs = out["report"]["change_set"]
    assert {i["field"] for i in cs["items"]} >= {"headquarters", "founded_year"} and cs["writes_performed"] is False


def test_company_only_reuses_one_search_and_cached_pages_not_repeated_ones():
    llm = SoloLLM("multi")
    r, out = run(llm)
    assert len(r.search.calls) == 1                                                  # one prefetch search, no follow-up
    urls = [c[0] if isinstance(c, tuple) else c for c in r.fetcher.calls]
    assert len(urls) == len(set(map(str, urls)))                                     # no page fetched twice


def test_company_only_cannot_invoke_the_other_four_agents():
    for mode in ("foreign_tool", "routes"):
        llm = SoloLLM(mode)
        r, out = run(llm)
        assert out["agent_runs"] == {"company": 1}
        assert set(out["agent_results"]) == {"company"} and len(llm.calls) == 1      # the bad call was not retried
        assert all(h["next"] in ("company", "finish") for h in out["route_history"])
        assert all(h["deterministic"] for h in out["route_history"])
    with pytest.raises(ValueError, match="company_only_allows_company_fields"):
        build(world(), SoloLLM()).start("Apollo Micro Systems", fields=["headquarters", "procurement_contacts"],
                                       company_only=True)
    with pytest.raises(ValueError, match="company_only_allows_company_fields"):
        build(world(), SoloLLM()).start("Apollo Micro Systems", fields=["lead_score"], company_only=True)


def test_routing_never_returns_another_agent_even_when_asked_to(monkeypatch):
    state = {"agent_runs": {}, "subject": {"existing": None, "identity": {"verified": True}}, "claims": {},
             "options": {"company_only": True, "fields": ["headquarters"]}}
    elig = {a: {"eligible": True, "reason": ""} for a in AGENT_SPECS}
    assert routing.route_scoped(state, elig)["next"] == "company"
    state["agent_runs"] = {"company": 1}
    assert routing.route_scoped(state, elig)["next"] == "finish"
    # even a buggy router is stopped by the graph's scope guard
    monkeypatch.setattr(routing, "route_scoped", lambda s, e: {"next": "contact", "reason": "bug", "vetoes": [],
                                                                       "error": None, "deterministic": True})
    llm = SoloLLM("multi")
    r, out = run(llm)
    assert out["agent_runs"] == {} and llm.calls == [] and out["stop_reason"] == "scope_violation"


def test_company_only_reads_only_the_crm_sections_the_company_agent_needs():
    spy = SpyRepo(default_repo())
    r, out = run(SoloLLM("multi"), repo=spy)
    used = set(spy.calls)
    assert used <= {"find_candidates", "get_company", "get_profile", "get_products", "can_see_companies"}
    assert not used & {"get_contact_summary", "get_company_suppliers", "get_components", "get_lead_status"}


# ================================================================================ empty / invalid / insufficient
def test_an_empty_response_is_not_retried_blindly():
    llm = SoloLLM("empty")
    r, out = run(llm)
    assert len(llm.calls) == 1 and out["budgets"]["llm_calls"] == 1
    assert out["agent_results"]["company"]["end_reason"] == "empty_response"
    assert out["claims"].get("company", []) == [] and events(out, "empty_response")
    assert out["status"] == "partial"                                                # honest: not 'completed'


def test_invalid_evidence_is_rejected_and_not_followed_by_a_second_request():
    llm = SoloLLM("invalid")
    r, out = run(llm)
    assert len(llm.calls) == 1 and not out["claims"].get("company")
    rej = out["rejected"]["company"]
    assert any("snippet_not_found_in_document" in x["reasons"] or "no_evidence" in x["reasons"] for x in rej)
    assert "headquarters" in {u["field"] for u in out["report"]["change_set"]["unresolved"]}
    assert not events(out, "packet_followup") and len(r.search.calls) == 1


def test_a_quote_that_only_mentions_the_value_does_not_populate_the_field():
    llm = SoloLLM("semantic")
    r, out = run(llm)
    assert len(llm.calls) == 1
    rejected = {x["field"]: x["reasons"] for x in out["rejected"]["company"]}
    for fld in ("founded_year", "annual_revenue", "headquarters"):
        assert rejected[fld] == ["quote_does_not_state_field"], fld
    assert not [c for c in out["claims"].get("company", []) if c["field"] in ("founded_year", "annual_revenue")]
    unresolved = {u["field"]: u["rejection_reasons"] for u in out["report"]["change_set"]["unresolved"]}
    assert unresolved["annual_revenue"] == ["quote_does_not_state_field"]


def test_one_correction_request_is_available_only_when_the_plan_counted_it():
    llm = SoloLLM("invalid")
    r, out = run(llm, limits=Limits(company_correction_requests=1))
    assert len(llm.calls) == 2 and out["budgets"]["llm_calls"] == 2                  # exactly the planned maximum
    llm0 = SoloLLM("invalid")
    r0, out0 = run(llm0)
    assert len(llm0.calls) == 1                                                      # default: no correction
    assert pf.build_plan(Settings(limits=Limits(company_correction_requests=1, llm_max_retries=0))).requests_max == 2
    assert pf.build_plan(Settings(limits=Limits(llm_max_retries=0))).requests_max == 1
    assert pf.build_plan(Settings()).requests_max == 3            # default retries=2: every retry is another request


def test_an_insufficient_evidence_packet_makes_no_model_request_at_all():
    llm = SoloLLM("multi")
    r, out = run(llm, w=make_world(COMPANY, "no_website"))
    assert llm.calls == [] and out["budgets"]["llm_calls"] == 0
    assert out["agent_results"]["company"]["end_reason"] == "packet_insufficient"


def test_the_token_limit_is_enforced_before_transmission():
    llm = SoloLLM("multi")
    r, out = run(llm, limits=Limits(max_total_tokens_run=500))
    assert llm.calls == [] and out["agent_results"]["company"]["end_reason"] == "max_tokens_reached"


def test_the_llm_call_limit_cannot_be_exceeded():
    llm = SoloLLM("invalid")
    r, out = run(llm, limits=Limits(company_correction_requests=1, max_llm_calls_run=1))
    assert len(llm.calls) == 1


# ================================================================================ field-meaning rules (unit)
@pytest.mark.parametrize("fld,value,quote,ok", [
    ("founded_year", "2007", "Since 2007, Airbus India engineers have supported programmes.", False),
    ("founded_year", "2007", "The company was founded in 2007.", True),
    ("annual_revenue", "USD 1.6 Billion", "Sourcing over USD 1.6 Billion from India.", False),
    ("annual_revenue", "USD 1.6 Billion", "Revenue of USD 1.6 Billion in 2023.", True),
    ("annual_revenue", "USD 1.6 Billion", "Revenue grew, and sourcing over USD 1.6 Billion followed.", False),
    ("headquarters", "New Delhi", "Our team in New Delhi and Bengaluru supports customers.", False),
    ("headquarters", "New Delhi", "The company is headquartered in New Delhi.", True),
    ("headquarters", "New Delhi", "Registered office: New Delhi.", True),
    ("employee_count", "1,500", "Over 1,500 customers rely on us.", False),
    ("employee_count", "1,500", "The company employs approximately 1,500 people.", True),
    ("parent_company", "Airbus SE", "Airbus India is a subsidiary of Airbus SE.", True),
    ("parent_company", "Airbus SE", "Airbus SE announced a new order.", False),
    ("manufacturing_locations", ["Pune"], "Our plant in Pune produces avionics.", True),
    ("manufacturing_locations", ["Pune"], "Our CEO visited Pune.", False),
    ("business_overview", "builds avionics", "The company builds avionics for aircraft.", True),    # no rule: unchanged
])
def test_field_support_rules(fld, value, quote, ok):
    assert field_support.check(fld, value, [quote])[0] is ok


# ================================================================================ preflight
_DB = iter(range(10 ** 6))


def fresh_db(tmp_path, used=0, key="gemini:gemini-3-flash-preview", age_s=60):
    import time
    path = str(tmp_path / f"agentic{next(_DB)}.sqlite")
    st = Store(path)
    for i in range(used):
        st.usage_add(key, 1000, time.time() - age_s - i)
    return path


def settings_for(**lim):
    base = dict(llm_max_retries=0, max_db_selects_run=8)
    base.update(lim)
    return Settings(limits=Limits(**base), model="gemini-3-flash-preview", provider="gemini")


QUOTA = ProviderQuota.parse("rpm=5,tpm=250000,rpd=20", require_tpd=False)


def verdict(tmp_path, used=0, settings=None, quota=QUOTA, **kw):
    settings = settings or settings_for()
    plan = pf.build_plan(settings)
    return plan, pf.evaluate(plan, settings, quota, fresh_db(tmp_path, used), **kw), settings


def test_preflight_passes_when_the_complete_plan_fits(tmp_path):
    plan, v, s = verdict(tmp_path, used=3)
    assert not v.blocked and v.remaining_requests == 13 and plan.requests_max == 1
    text = pf.render(plan, v, s)
    for needle in ("min 0, max 1", "estimated tokens", "remaining under the cap: 13", "Tavily credits", "SELECTs",
                   "per-run limits"):
        assert needle in text
    assert pf.BLOCK_LINE not in text


def test_preflight_blocks_when_the_daily_allowance_is_used_up(tmp_path):
    plan, v, s = verdict(tmp_path, used=16)                       # 16 = planning cap of a 20/day limit
    assert v.blocked and "remaining daily allowance 0" in " ".join(v.reasons)
    text = pf.render(plan, v, s)
    assert pf.BLOCK_LINE == "PREFLIGHT BLOCKED — no external requests sent" and pf.BLOCK_LINE in text
    assert "remaining under the cap: 0" in text


def test_preflight_counts_every_permitted_attempt_not_only_the_first(tmp_path):
    s = settings_for(llm_max_retries=2, company_correction_requests=1)
    plan = pf.build_plan(s)
    assert plan.calls_max == 2 and plan.requests_max == 6                            # 2 calls x (1 + 2 retries)
    v = pf.evaluate(plan, s, QUOTA, fresh_db(tmp_path, used=12))                     # 12 used, cap 16 -> 4 left < 6
    assert v.blocked and "planned maximum 6" in " ".join(v.reasons)


def test_preflight_treats_unknown_quota_or_usage_as_a_stop(tmp_path):
    plan, v, _ = verdict(tmp_path, quota=None)
    assert v.blocked and "UNKNOWN" in " ".join(v.reasons)
    s = settings_for()
    v2 = pf.evaluate(pf.build_plan(s), s, QUOTA, str(tmp_path / "missing.sqlite"))
    assert v2.blocked and "UNKNOWN" in " ".join(v2.reasons) and not (tmp_path / "missing.sqlite").exists()


def test_preflight_checks_every_per_run_budget(tmp_path):
    for lim, needle in ((dict(max_llm_calls_run=0), "per-run limit 0"),
                        (dict(max_total_tokens_run=1000), "per-run token limit"),
                        (dict(max_search_credits_run=0), "credit limit"),
                        (dict(max_db_selects_run=None), "SELECT cap"),
                        (dict(max_db_selects_run=9), "above the required shared maximum"),
                        (dict(max_db_selects_run=6), "planned maximum 7 Supabase SELECTs")):
        s = settings_for(**lim)
        v = pf.evaluate(pf.build_plan(s), s, QUOTA, fresh_db(tmp_path))
        assert v.blocked and needle in " ".join(v.reasons), (lim, v.reasons)
    other = Settings(limits=Limits(llm_max_retries=0, max_db_selects_run=8), provider="groq", model="openai/gpt-oss-120b")
    assert pf.evaluate(pf.build_plan(other), other, QUOTA, fresh_db(tmp_path)).blocked


def test_the_console_figure_can_only_tighten_the_local_one(tmp_path):
    assert verdict(tmp_path, used=0, console_remaining=0)[1].blocked
    assert not verdict(tmp_path, used=0, console_remaining=10)[1].blocked


def test_the_enforced_caps_equal_the_plan():
    plan = pf.build_plan(settings_for())
    caps = pf.caps_for(plan)
    s = load_settings({**company_live.base_env("rpm=5,tpm=250000,rpd=20", "data"), **caps})
    assert s.limits.max_llm_calls_run == 1 and s.limits.max_db_selects_run == 8
    assert s.limits.max_search_credits_run == 1 and s.limits.llm_max_retries == 0 and s.limits.max_wall_clock_s == 420
    assert s.limits.max_total_tokens_run >= plan.tokens_max
    assert s.provider == "gemini" and s.model == "gemini-3-flash-preview" and not s.efficiency.parallel_agents


def test_preflight_cli_makes_no_network_call_and_creates_no_database(tmp_path, monkeypatch, capsys):
    def no_net(*a, **k):
        raise AssertionError("network access during preflight")
    monkeypatch.setattr(socket.socket, "connect", no_net)
    env = company_live.base_env("rpm=5,tpm=250000,rpd=20", str(tmp_path / "nodata"))
    code = pf.main([], env={**env, **pf.caps_for(pf.build_plan(pf.settings_from_env(env)))})
    out = capsys.readouterr().out
    assert code == 3 and pf.BLOCK_LINE in out and "UNKNOWN" in out
    assert not (tmp_path / "nodata").exists()


def seed(tmp_path, used):
    """usage rows in the file the live runner reads: <data-dir>/agentic.sqlite"""
    import time
    st = Store(str(tmp_path / "agentic.sqlite"))
    for i in range(used):
        st.usage_add("gemini:gemini-3-flash-preview", 1000, time.time() - 60 - i)


def test_the_live_runner_stops_at_the_block_line_before_building_any_client(tmp_path, monkeypatch, capsys):
    import agentic.service as service
    monkeypatch.setattr(service, "build_live_service", lambda *a, **k: (_ for _ in ()).throw(AssertionError("built")))
    monkeypatch.setenv("AGENTIC_COMPANY_LIVE_APPROVED", "1")
    seed(tmp_path, 16)
    code = company_live.main(["--company", "AIRBUS India", "--company-id", "11", "--quota", "rpm=5,tpm=250000,rpd=20",
                              "--data-dir", str(tmp_path)])
    out = capsys.readouterr().out
    assert code == 3 and out.count("PREFLIGHT BLOCKED") == 1 and "no external requests sent" in out


def test_the_live_runner_needs_explicit_approval_even_when_the_preflight_passes(tmp_path, monkeypatch, capsys):
    import agentic.service as service
    monkeypatch.setattr(service, "build_live_service", lambda *a, **k: (_ for _ in ()).throw(AssertionError("built")))
    monkeypatch.delenv("AGENTIC_COMPANY_LIVE_APPROVED", raising=False)
    seed(tmp_path, 2)
    code = company_live.main(["--company", "AIRBUS India", "--company-id", "11", "--quota", "rpm=5,tpm=250000,rpd=20",
                              "--data-dir", str(tmp_path)])
    assert code == 4 and "PREFLIGHT OK" in capsys.readouterr().out
    assert company_live.main(["--company", "X", "--company-id", "1", "--fields", "suppliers",
                              "--data-dir", str(tmp_path)]) == 2                      # non-Company field refused


# ================================================================================ atomic reservation
class Probe:
    name, model = "gemini", "gemini-3-flash-preview"

    def __init__(self, fail=None, delay=0.02):
        self.calls, self.fail, self.delay, self._lock = 0, fail, delay, threading.Lock()

    def generate(self, **kw):
        import time
        with self._lock:
            self.calls += 1
        time.sleep(self.delay)
        if self.fail:
            raise self.fail
        return LLMResponse(text="x", usage=Usage(requests=1, input_tokens=50, output_tokens=5))


KW = dict(system="s", messages=[{"role": "user", "content": "hi"}], tools=[], max_output_tokens=10)


def test_capacity_is_reserved_atomically_so_concurrent_requests_cannot_overspend(tmp_path):
    path = str(tmp_path / "a.sqlite")
    q = ProviderQuota(rpm=1000, tpm=10 ** 7, rpd=5, tpd=10 ** 12, headroom=0.8)       # planning cap = 4
    probe, results = Probe(), []

    def worker():
        llm = PacedLLM(probe, q, Store(path))                                         # separate connections = processes
        try:
            llm.generate(**KW)
            results.append("ok")
        except LLMError as exc:
            results.append(exc.kind)
    threads = [threading.Thread(target=worker) for _ in range(12)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert results.count("ok") == 4 and results.count("quota") == 8 and probe.calls == 4
    assert Store(path).usage_since("gemini:gemini-3-flash-preview", 86400, __import__("time").time())[0] == 4


def test_a_request_that_failed_after_reservation_still_counts_against_the_day(tmp_path):
    store = Store(str(tmp_path / "b.sqlite"))
    q = ProviderQuota(rpm=1000, tpm=10 ** 7, rpd=5, tpd=10 ** 12, headroom=0.4)       # cap = 2
    llm = PacedLLM(Probe(fail=LLMError("transient", "503 UNAVAILABLE")), q, store)
    for _ in range(2):
        with pytest.raises(LLMError) as e:
            llm.generate(**KW)
        assert e.value.kind == "transient"
    with pytest.raises(LLMError) as e:                                                # the 503s used the allowance
        llm.generate(**KW)
    assert e.value.kind == "quota"

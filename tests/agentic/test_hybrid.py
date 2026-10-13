"""Hybrid architecture: Python plans and retrieves, the LLM interprets evidence. Offline only.

Covers deterministic objective routing + prerequisites, the one-call Company path, follow-up only when needed,
relevant-page ranking, lazy CRM reads, deterministic Lead, failure handling, and that the five-agent architecture and
the LLM supervisor for comprehensive/ambiguous requests are intact. Evidence validation is exercised, never relaxed."""

import copy
import dataclasses

import pytest

from agentic import evidence as ev
from agentic import routing
from agentic.agents import AGENT_SPECS
from agentic.agents.lead import standard_next_actions
from agentic.backends.fixtures import FixtureFetcher, FixtureSearch
from agentic.config import Efficiency, Limits, Settings, load_settings
from agentic.eval.policy import PolicyLLM
from agentic.eval.synthetic import make_world
from agentic.guards.budget import new_budgets
from agentic.llm.base import LLMError
from agentic.llm.fake import FakeLLM, tool_call
from agentic.persistence.store import Store
from agentic.prefetch import page_score, rank_official_candidates
from agentic.repo import InMemoryRepo
from agentic.runtime import DEFAULT_OBJECTIVE, AgenticRunner, make_deps
from agentic.scoring import score_lead
from agentic.state import initial_state
from agentic.supervisor import ROUTE_TOOL, eligibility
from tests.agentic.support import default_repo
from tests.agentic.test_prefetch import PACKET_MARK, PacketLLM, COMPANY, events, packet_json, world

HYBRID = Efficiency(prefetch=True, deterministic_routing=True, lazy_crm=True, lead_deterministic=True)
PROFILE = "company profile"


# ------------------------------------------------------------------------------------------------- helpers
class SpyRepo:
    """Delegates to a real in-memory repo and records which read methods were used."""

    def __init__(self, inner):
        self.inner, self.calls = inner, []

    def __getattr__(self, name):
        fn = getattr(self.inner, name)
        if not callable(fn):
            return fn

        def wrapper(*a, **k):
            self.calls.append(name)
            return fn(*a, **k)
        return wrapper


class HybridLLM(PacketLLM):
    """PacketLLM for the Company agent; every other agent submits an honest 'nothing found'."""

    def __init__(self, mode="honest"):
        super().__init__("honest")
        self.company_mode, self.company_calls = mode, 0

    def generate(self, *, system, messages, tools, max_output_tokens, force_tool=False):
        names = [t.name for t in tools]
        finish = next((n for n in names if n.endswith("_findings")), None)
        if names == ["route"] or finish is None:
            return super().generate(system=system, messages=messages, tools=tools,
                                    max_output_tokens=max_output_tokens, force_tool=force_tool)
        if finish != "submit_company_findings":
            self.calls.append({"agent": finish, "tools": names, "messages": copy.deepcopy(messages), "force": force_tool})
            return tool_call(finish, status="exhausted")
        self.company_calls += 1
        if self.company_mode == "raise":
            self.calls.append({"agent": "company", "tools": names, "messages": copy.deepcopy(messages), "force": force_tool})
            raise LLMError("transient", "503 UNAVAILABLE")
        if self.company_mode == "no_claims":
            self.calls.append({"agent": "company", "tools": names, "messages": copy.deepcopy(messages), "force": force_tool})
            return tool_call("submit_company_findings", status="exhausted")
        packet = packet_json(messages[0]["content"])
        doc = packet["documents"][0] if packet and packet["documents"] else None
        self.calls.append({"agent": "company", "tools": names, "messages": copy.deepcopy(messages), "force": force_tool})
        if self.company_mode == "minimal":                       # leaves headquarters/overview unresolved
            if self.company_calls == 1 and doc:
                return tool_call("submit_company_findings", status="partial", identity_website=doc["url"],
                                 claims=[{"field": "official_website", "value": doc["url"],
                                          "evidence": [{"url": doc["url"], "snippet": doc["text"][:100]}]}])
            return tool_call("submit_company_findings", status="exhausted")
        if self.company_mode == "paraphrase" and self.company_calls == 1:
            return tool_call("submit_company_findings", status="satisfied", identity_website=doc["url"],
                             claims=[{"field": "official_website", "value": doc["url"],
                                      "evidence": [{"url": doc["url"], "snippet": doc["text"][:100]}]},
                                     {"field": "headquarters", "value": "Aurangabad, India",
                                      "evidence": [{"url": doc["url"],
                                                    "snippet": "The firm is based out of Aurangabad in India."}]}])
        self.calls.pop()
        return super().generate(system=system, messages=messages, tools=tools,
                                max_output_tokens=max_output_tokens, force_tool=force_tool)


def build(w, llm, eff=HYBRID, *, repo=None, limits=None, store=None):
    settings = Settings(limits=limits or Limits(), efficiency=eff)
    search, fetch = FixtureSearch(w), FixtureFetcher(w)
    store = store or Store(":memory:")
    deps = make_deps(llm=llm, repo=repo if repo is not None else default_repo(), search=search, fetcher=fetch,
                     store=store, settings=settings, specs=AGENT_SPECS, sleep=lambda s: None)
    r = AgenticRunner(deps, AGENT_SPECS)
    r.search, r.fetcher, r.store = search, fetch, store
    return r


def llm_events(out, agent=None):
    return [e for e in out["trace"] if e["type"] == "llm_call" and (agent is None or e["agent"] == agent)]


def routes(out):
    return [(e["data"]["next"], e["data"]["deterministic"]) for e in events(out, "route")]


def state_for(target_objective="x", *, identity=False, existing=None, runs=None, claims=None):
    s = initial_state("r", "Acme", target_objective)
    s["subject"] = {"name": "Acme", "match": {"status": "found", "matches": [], "assumptions": []},
                    "existing": existing, "identity": {"verified": identity}}
    s["agent_runs"] = runs or {}
    s["claims"] = claims or {}
    return s


# ====================================================================================== objective routing
@pytest.mark.parametrize("objective,target", [
    ("company profile", "company"),
    ("Show me the official website and headquarters", "company"),
    ("Find the procurement decision makers", "contact"),
    ("who are the purchasing managers", "contact"),
    ("procurement structure and supplier portal", "procurement"),
    ("what components does the company buy", None),            # phrasing the keyword table does not know -> LLM
    ("which suppliers supply components to them", "supplier"),
    ("rank this lead", "lead"),
    ("prioritise this lead and explain why", "lead"),
])
def test_clear_single_capability_objectives_are_planned_by_python(objective, target):
    plan = routing.classify_objective(objective)
    if target is None:
        assert plan.kind == "llm"
    else:
        assert plan.kind == "clear" and plan.target == target


@pytest.mark.parametrize("objective", [
    DEFAULT_OBJECTIVE, "comprehensive", "give me a full report", "everything about them",
    "company profile and procurement decision makers",           # several agents
    "supplier risks and the procurement structure",
    "tell me something interesting", "", None,                    # ambiguous / empty
])
def test_comprehensive_multi_agent_and_ambiguous_objectives_stay_with_the_llm_supervisor(objective):
    assert routing.classify_objective(objective).kind == "llm"


def test_interpretation_is_requested_only_when_the_user_asks_for_it():
    assert routing.wants_interpretation("rank this lead and explain why")
    assert routing.wants_interpretation("recommend next steps for this lead")
    assert not routing.wants_interpretation("rank this lead")


# ============================================================================================ prerequisites
def test_prerequisite_chains_mirror_the_eligibility_rules():
    chain = routing.prerequisite_chain
    assert chain("company", state_for()) == ["company"]
    assert chain("contact", state_for()) == ["company", "contact"]
    assert chain("contact", state_for(identity=True)) == ["contact"]
    assert chain("procurement", state_for()) == ["company", "procurement"]
    assert chain("supplier", state_for()) == ["company", "procurement", "supplier"]
    assert chain("supplier", state_for(existing={"components": ["Relays"]})) == ["company", "supplier"]
    assert chain("supplier", state_for(identity=True, existing={"components": ["Relays"]})) == ["supplier"]
    assert chain("lead", state_for()) == ["company", "lead"]
    assert chain("lead", state_for(existing={"fields": {"company_name": "Acme"}})) == ["lead"]


def test_a_planned_agent_that_cannot_run_ends_the_run_with_a_recorded_reason_not_a_substitute_plan():
    plan = routing.classify_objective("find the procurement decision makers")
    state = state_for(runs={"company": 1}, identity=False)                      # company ran but identity unconfirmed
    elig = eligibility(state, AGENT_SPECS, Limits())
    d = routing.route_deterministic(state, elig, plan)
    assert d["next"] == "finish" and d["error"] == "planned_agent_blocked" and "contact" in d["reason"]
    state = state_for()
    assert routing.route_deterministic(state, eligibility(state, AGENT_SPECS, Limits()), plan)["next"] == "company"
    state = state_for(identity=True, runs={"company": 1, "contact": 1})
    done = routing.route_deterministic(state, eligibility(state, AGENT_SPECS, Limits()), plan)
    assert done["next"] == "finish" and done["error"] is None


# ================================================================================= settings and switches
def test_hybrid_switches_default_off_and_turn_on_together_with_individual_overrides():
    off = load_settings({}).efficiency
    assert not (off.prefetch or off.deterministic_routing or off.lazy_crm or off.lead_deterministic)
    on = load_settings({"AGENTIC_HYBRID": "1"}).efficiency
    assert on.prefetch and on.deterministic_routing and on.lazy_crm and on.lead_deterministic
    mixed = load_settings({"AGENTIC_HYBRID": "1", "AGENTIC_PREFETCH": "0", "AGENTIC_LAZY_CRM": "0"}).efficiency
    assert not mixed.prefetch and not mixed.lazy_crm and mixed.deterministic_routing and mixed.lead_deterministic
    assert load_settings({"AGENTIC_LAZY_CRM": "1"}).efficiency.lazy_crm and not load_settings({"AGENTIC_LAZY_CRM": "1"}).efficiency.prefetch


# ============================================================ Company: one main call after the packet
def test_company_profile_path_uses_one_llm_request_and_no_supervisor_call():
    llm = HybridLLM()
    r = build(world(), llm)
    out = r.start("Apollo Micro Systems", PROFILE)
    assert len(llm_events(out)) == 1 and llm_events(out)[0]["agent"] == "company"      # the only model request
    assert routes(out) == [("company", True), ("finish", True)]                       # decided by Python, both times
    assert out["status"] == "completed" and out["agent_runs"] == {"company": 1}
    fields = {c["field"]: c for c in out["claims"]["company"]}
    assert {"official_website", "headquarters", "business_overview"} <= set(fields)
    assert all(c["status"] == "verified" and c["evidence"] for c in fields.values())
    # that request offered only the submit tool and required it (no discovery turns, smaller prompt)
    call = next(c for c in llm.calls if c["agent"] == "company")
    assert call["tools"] == ["submit_company_findings"] and call["force"] is True
    assert out["budgets"]["search_credits"] == 1
    assert not [e for e in events(out, "tool_result") if not e["data"].get("prefetch")]   # no model-issued search/fetch


def test_the_request_count_drops_from_three_to_one_against_the_previous_phase():
    prev = HybridLLM()
    prev_out = build(world(), prev, Efficiency(prefetch=True)).start("Apollo Micro Systems", PROFILE)
    new = HybridLLM()
    new_out = build(world(), new, HYBRID).start("Apollo Micro Systems", PROFILE)
    assert len(llm_events(prev_out)) == 3 and len(llm_events(new_out)) == 1


def test_a_follow_up_search_is_allowed_once_and_only_when_important_fields_remain_unresolved():
    llm = HybridLLM("minimal")
    out = build(world(), llm).start("Apollo Micro Systems", PROFILE)
    follow = events(out, "packet_followup")
    assert len(follow) == 1 and set(follow[0]["data"]["unresolved"]) == {"headquarters", "business_overview"}
    company_calls = [c for c in llm.calls if c["agent"] == "company"]
    assert len(company_calls) == 2                                              # packet turn + ONE follow-up turn
    assert company_calls[0]["tools"] == ["submit_company_findings"]
    assert {"web_search", "fetch_page", "submit_company_findings"} <= set(company_calls[1]["tools"])
    assert "did not settle" in company_calls[1]["messages"][-1]["content"]
    assert out["claims"]["company"] and not out["errors"]


def test_no_follow_up_when_the_packet_settled_everything_important():
    out = build(world(), HybridLLM()).start("Apollo Micro Systems", PROFILE)
    assert not events(out, "packet_followup")


def test_rejected_claims_are_corrected_within_the_packet_turn_and_validation_is_not_relaxed():
    llm = HybridLLM("paraphrase")
    out = build(world(), llm).start("Apollo Micro Systems", PROFILE)
    first = events(out, "submit")[0]["data"]
    assert first["rejected"] == 1 and "snippet_not_found_in_document" in first["reasons"]
    company_calls = [c for c in llm.calls if c["agent"] == "company"]
    assert len(company_calls) == 2 and company_calls[1]["tools"] == ["submit_company_findings"]   # still no discovery
    ledger = out["ledger"]
    for c in out["claims"]["company"]:                                         # every accepted quote is in the ledger
        assert c["status"] == "verified"
        assert all(ev.validate_snippet(ledger, e["url"], e["snippet"])[0] for e in c["evidence"])
    assert {c["field"] for c in out["claims"]["company"]} >= {"official_website", "headquarters"}


def test_packet_inadequate_falls_back_to_the_normal_company_path():
    llm = FakeLLM([tool_call("web_search", query="Apollo Micro Systems headquarters"),
                   tool_call("submit_company_findings", status="exhausted")])
    out = build(world("no_website"), llm).start("Apollo Micro Systems", PROFILE)
    assert events(out, "prefetch")[0]["data"]["status"] == "insufficient"
    assert llm.calls[0]["tools"] == ["web_search", "fetch_page", "submit_company_findings"]   # full tools, not forced
    assert not events(out, "packet_followup")


# ======================================================================================== page ranking
AIRBUS = "https://www.airbus.com/en"
INDIA = AIRBUS + "/about-us/our-worldwide-presence/airbus-in-asia-pacific/airbus-in-india"
CAREERS = AIRBUS + "/careers/life-at-airbus/working-here/working-at-airbus-india"
PRESS = AIRBUS + "/newsroom/press-releases/2024-10-airbus-inaugurates-india-and-south-asia-headquarters-in-new-delhi"
RESULTS = [
    {"url": CAREERS, "title": "Working at Airbus in India | Airbus"},
    {"url": PRESS, "title": "Airbus inaugurates India and South Asia headquarters in New Delhi"},
    {"url": INDIA + "#overview", "title": "Airbus in India | Airbus"},
    {"url": INDIA, "title": "Airbus in India | Airbus"},                                       # duplicate once canonical
    {"url": AIRBUS + "/about-us", "title": "About us | Airbus"},
    {"url": AIRBUS + "/airbus-contact-us", "title": "Contact | Airbus"},
    {"url": "https://tracxn.com/d/companies/airbus/x", "title": "Airbus - Tracxn"},
]


def test_company_and_about_pages_outrank_careers_and_press_pages():
    ranked = rank_official_candidates("AIRBUS India", RESULTS)
    urls = [p["url"] for p in ranked]
    assert urls[0] == ev.canonical_url(INDIA)
    assert urls.count(ev.canonical_url(INDIA)) == 1                           # fragment/duplicate collapsed
    assert not any("tracxn" in u for u in urls)                               # aggregator never a candidate
    careers = next(p for p in ranked if "careers" in p["url"])
    press = next(p for p in ranked if "press-releases" in p["url"])
    assert careers["penalised"] and press["penalised"]
    assert urls.index(careers["url"]) > urls.index(ev.canonical_url(AIRBUS + "/about-us"))
    assert urls.index(careers["url"]) > urls.index(AIRBUS + "/airbus-contact-us")
    assert page_score(INDIA, "Airbus in India | Airbus", ["airbus", "india"]) > page_score(CAREERS, "Working at Airbus", ["airbus", "india"])


def airbus_world():
    text = lambda t: t + " " + "Airbus designs, manufactures and delivers aerospace products worldwide. " * 12
    pages = {INDIA: ("Airbus in India | Airbus", text("Airbus in India. Airbus India and South Asia is headquartered in New Delhi.")),
             CAREERS: ("Working at Airbus in India | Airbus", text("Careers at Airbus in India. Join our teams.")),
             PRESS: ("Press release | Airbus", text("Airbus inaugurates India and South Asia headquarters.")),
             AIRBUS + "/about-us": ("About us | Airbus", text("About Airbus. Airbus is a European aerospace company.")),
             AIRBUS + "/airbus-contact-us": ("Contact | Airbus", text("Contact Airbus offices."))}
    w = {"search_index": [{"url": u, "title": t, "content": body[:400]} for u, (t, body) in pages.items()],
         "pages": {u: {"title": t, "text": body, "links": [{"url": AIRBUS + "/about-us", "text": "About us"},
                                                         {"url": CAREERS, "text": "Careers"}]}
                   for u, (t, body) in pages.items()},
         "facts": {}, "variant": "standard", "fetch_errors": {}}
    return w


def test_prefetch_builds_the_packet_from_company_pages_not_careers_or_press():
    llm = HybridLLM("no_claims")
    r = build(airbus_world(), llm, repo=InMemoryRepo([]))
    out = r.start("AIRBUS India", PROFILE)
    pf = events(out, "prefetch")[0]["data"]
    assert pf["status"] == "packet" and pf["domain"] == "airbus.com"
    assert pf["pages"][0] == ev.canonical_url(INDIA)
    assert not any("careers" in u or "press-releases" in u for u in pf["pages"])
    assert len(r.fetcher.calls) == len(set(r.fetcher.calls))                   # no page fetched twice


# ============================================================================================== lazy CRM
def test_default_mode_still_reads_every_crm_section_eagerly():
    spy = SpyRepo(default_repo())
    build(world(), HybridLLM(), Efficiency(), repo=spy).start("Apollo Micro Systems", PROFILE) if False else None
    from tests.agentic.support import make_runner
    r = make_runner(repo=spy, llm=PolicyLLM())
    r.start("Apollo Micro Systems", PROFILE)
    assert {"get_products", "get_components", "get_contact_summary", "get_company_suppliers", "get_lead_status"} <= set(spy.calls)


def test_company_profile_reads_only_the_tables_the_company_agent_uses():
    spy = SpyRepo(default_repo())
    out = build(world(), HybridLLM(), repo=spy).start("Apollo Micro Systems", PROFILE)
    reads = {c for c in spy.calls if c.startswith("get_")} - {"get_company"}      # get_company = the name match
    assert reads == {"get_products"}                                          # nothing for contacts/suppliers/leads
    assert out["report"]["database_context"]["sections_loaded"] == ["products"]
    assert events(out, "crm_sections_loaded")[0]["data"]["sections"] == ["products"]


def test_each_clear_objective_loads_only_its_agents_sections():
    spy = SpyRepo(default_repo())
    out = build(world(), HybridLLM(), repo=spy).start("Apollo Micro Systems", "find the procurement decision makers")
    reads = {c for c in spy.calls if c.startswith("get_")} - {"get_company"}      # get_company = the name match
    assert reads == {"get_products", "get_contact_summary"}                  # company (products) + contact (summary)
    assert out["agent_runs"] == {"company": 1, "contact": 1}
    spy = SpyRepo(default_repo())
    build(world(), HybridLLM(), repo=spy).start("Apollo Micro Systems", "rank this lead")
    assert {c for c in spy.calls if c.startswith("get_")} - {"get_company"} == {"get_components"}


def test_llm_routed_objectives_load_what_any_agent_may_need():
    spy = SpyRepo(default_repo())
    build(world(), PolicyLLM(), repo=spy).start("Apollo Micro Systems", DEFAULT_OBJECTIVE)
    assert {"get_products", "get_components", "get_contact_summary", "get_company_suppliers"} <= set(spy.calls)


# =========================================================================================== Lead agent
def test_routine_lead_ranking_needs_no_llm_call_and_has_standard_next_actions():
    llm = HybridLLM()
    out = build(world(), llm).start("Apollo Micro Systems", "rank this lead")
    assert llm_events(out) == [] and llm.calls == []
    assert events(out, "deterministic_agent")
    lead = out["claims"]["lead"]
    assert lead["score"] is not None and lead["band"] in ("high", "medium", "low")
    assert lead["agent_suggestions"] and all(a.endswith(".") for a in lead["agent_suggestions"])
    assert lead["score"] == score_lead({**out, "claims": {k: v for k, v in out["claims"].items() if k != "lead"}})["score"]
    assert out["agent_runs"] == {"lead": 1} and out["status"] == "completed"


def test_standard_actions_follow_the_missing_information():
    s = state_for(existing={"fields": {"company_name": "Acme"}})
    actions = standard_next_actions(score_lead(s))
    text = " ".join(actions)
    assert "Company Intelligence" in text and "Procurement Intelligence" in text and "Contact Intelligence" in text
    assert len(actions) <= 6


def test_an_llm_call_is_used_for_the_lead_agent_only_when_interpretation_is_requested():
    llm = FakeLLM([tool_call("submit_lead_findings", status="satisfied", next_actions=["Call the plant head."])])
    out = build(world(), llm).start("Apollo Micro Systems", "rank this lead and explain why")
    assert len(llm.calls) == 1 and llm.calls[0]["tools"] == ["submit_lead_findings"]
    assert not events(out, "deterministic_agent")
    assert out["claims"]["lead"]["agent_suggestions"] == ["Call the plant head."]


def test_lead_for_a_company_outside_the_crm_runs_company_first_then_the_deterministic_lead():
    llm = HybridLLM()
    out = build(world(), llm, repo=InMemoryRepo([])).start("Apollo Micro Systems", "rank this lead")
    assert routes(out) == [("company", True), ("lead", True), ("finish", True)]
    assert len(llm_events(out)) == 1 and out["claims"]["lead"]["score"] is not None


# ====================================================================================== failure handling
def test_an_llm_outage_in_the_company_agent_ends_the_run_partial_without_retry_loops():
    default_retry = HybridLLM("raise")
    build(world(), default_retry).start("Apollo Micro Systems", PROFILE)
    assert default_retry.company_calls == 3                                     # 1 + the 2 bounded retries, never more
    llm = HybridLLM("raise")
    out = build(world(), llm, limits=Limits(llm_max_retries=0)).start("Apollo Micro Systems", PROFILE)
    assert llm.company_calls == 1                                               # one request, no loop
    assert out["status"] == "partial" and "agent_not_finished:company:llm_transient" in out["status_reasons"]
    assert routes(out)[-1] == ("finish", True)
    assert any(d["tool"] == "fetch_page" or len(d["text"]) > 1000 for d in out["ledger"].values())   # packet docs kept


def test_unconfirmed_identity_blocks_the_planned_downstream_agent_and_is_reported():
    llm = HybridLLM("no_claims")
    out = build(world("no_website"), llm).start("Apollo Micro Systems", "find the procurement decision makers")
    assert "contact" not in out["agent_runs"]
    assert out["stop_reason"] == "planned_agent_blocked" and out["status"] == "partial"
    assert "limit:planned_agent_blocked" in out["status_reasons"]
    last = events(out, "route")[-1]["data"]
    assert last["next"] == "finish" and last["deterministic"] and "contact" in last["reason"]


def test_a_search_outage_degrades_to_partial_and_no_agent_is_forced_to_run():
    def outage(w):
        w["search_failure"] = {"mode": "quota", "after": 0}
    llm = HybridLLM("no_claims")
    out = build(world(outage=outage), llm).start("Apollo Micro Systems", "procurement structure")
    assert out["status"] == "partial" and "procurement" not in out["agent_runs"]
    assert out["budgets"]["search_disabled"]


# ======================================================================== five-agent architecture intact
def test_the_five_agents_and_their_tools_are_still_registered():
    assert set(AGENT_SPECS) == {"company", "contact", "procurement", "supplier", "lead"}
    route_enum = ROUTE_TOOL.parameters["properties"]["next"]["enum"]
    assert set(route_enum) == {"company", "contact", "procurement", "supplier", "lead", "finish"}
    r = build(world(), HybridLLM())
    for name, spec in AGENT_SPECS.items():
        assert r.deps.registry.allowed(name, spec.finish_tool)


def test_comprehensive_objective_still_goes_through_the_llm_supervisor_and_can_run_all_five_agents():
    out = build(world(), PolicyLLM(), dataclasses.replace(HYBRID, prefetch=False)).start("Apollo Micro Systems", DEFAULT_OBJECTIVE)
    sup = llm_events(out, "supervisor")
    assert sup and all(not d for _, d in routes(out)[:1])                      # first decision came from the LLM
    assert set(out["agent_runs"]) == {"company", "contact", "procurement", "supplier", "lead"}
    assert out["status"] in ("completed", "partial")
    assert events(out, "deterministic_agent")                                  # routine Lead result still computed in code


def test_ambiguous_objective_is_left_to_the_llm_supervisor():
    llm = FakeLLM([tool_call("route", next="company", reason="x"), tool_call("submit_company_findings", status="exhausted"),
                   tool_call("route", next="finish", reason="done")])
    out = build(world("no_website"), llm).start("Apollo Micro Systems", "tell me something interesting")
    assert routes(out)[0] == ("company", False) and len(llm_events(out, "supervisor")) >= 1

"""Token/request savings must never weaken evidence validation, budgets or research output."""

import copy
import json

import pytest

from agentic.config import BASELINE_EFFICIENCY, Efficiency, Limits, Settings
from agentic.context import SUPERSEDED, compact_messages, compact_schema, ledger_hint
from agentic.eval import cost_report as cr
from agentic.eval.policy import PolicyLLM
from agentic.llm.fake import FakeLLM, LLMResponse, tool_call
from agentic.llm.base import ToolCallRequest
from agentic.scoring import score_lead
from agentic.security import wrap_untrusted
from tests.agentic.support import make_runner

ALL_ON = Efficiency(*[True] * 8)


def settings(eff, **lim):
    return Settings(limits=Limits(**lim), efficiency=eff)


def tool_msg(name, payload, call_id="c"):
    return {"role": "tool", "tool_call_id": call_id, "name": name, "content": wrap_untrusted(json.dumps(payload), name)}


# ============================================================== history compaction (unit)
def conversation():
    links = [{"url": f"https://a.example/p{i}", "text": f"Page {i}"} for i in range(25)]
    return [
        {"role": "user", "content": "task"},
        tool_msg("web_search", {"query": "q", "results": [
            {"url": "https://a.example/about", "title": "About", "snippet": "S-ABOUT " * 20},
            {"url": "https://wiki.example/x", "title": "Wiki", "snippet": "S-WIKI " * 20}]}, "1"),
        tool_msg("fetch_page", {"url": "https://a.example/about", "text": "FULL ABOUT PAGE TEXT", "links": links}, "2"),
        tool_msg("fetch_page", {"url": "https://a.example/products", "text": "FULL PRODUCTS TEXT", "links": links}, "3"),
    ]


def payload(msg):
    return json.loads(msg["content"].split("\n", 1)[1].rsplit("\n", 1)[0])


def test_only_snippets_of_later_fetched_pages_are_superseded_and_page_text_is_never_touched():
    orig = conversation()
    before = copy.deepcopy(orig)
    out = compact_messages(orig)
    assert orig == before                                                      # input is not mutated
    search = payload(out[1])["results"]
    assert search[0]["snippet"] == SUPERSEDED and search[1]["snippet"].startswith("S-WIKI")   # unfetched page keeps snippet
    assert payload(out[2])["text"] == "FULL ABOUT PAGE TEXT" and payload(out[3])["text"] == "FULL PRODUCTS TEXT"
    assert len(payload(out[2])["links"]) == 10 and payload(out[2])["links_omitted"] == 15   # older page: links trimmed
    assert len(payload(out[3])["links"]) == 25                                  # latest page keeps full link list
    assert all(m["content"].startswith("<untrusted_web_content") for m in out[1:])       # still marked as untrusted data


def test_a_snippet_is_kept_when_the_fetch_comes_before_it():
    msgs = [{"role": "user", "content": "t"},
            tool_msg("fetch_page", {"url": "https://a.example/about", "text": "FULL", "links": []}, "1"),
            tool_msg("web_search", {"results": [{"url": "https://a.example/about", "snippet": "LATER-SNIPPET"}]}, "2")]
    assert payload(compact_messages(msgs)[2])["results"][0]["snippet"] == "LATER-SNIPPET"


def test_schema_compaction_removes_only_validation_keywords():
    s = {"type": "object", "properties": {"q": {"type": "string", "minLength": 3, "maxLength": 200, "description": "d"},
                                          "n": {"type": "array", "maxItems": 3, "items": {"type": "string"}}},
         "required": ["q"]}
    c = compact_schema(s)
    assert "maxLength" not in json.dumps(c) and "maxItems" not in json.dumps(c)
    assert c["properties"]["q"]["description"] == "d" and c["required"] == ["q"] and c["properties"]["n"]["type"] == "array"


# ================================================ validation is independent of what the prompt shows
def test_evidence_validation_uses_the_full_ledger_not_the_compacted_prompt_and_python_still_enforces_limits():
    home = "https://www.apollomicrosystems.example/"
    line = "Apollo Micro Systems is a electronics company headquartered in"
    script = [tool_call("route", next="company", reason="x"),
              tool_call("web_search", query="Apollo Micro Systems official website"),
              tool_call("fetch_page", url=home),
              tool_call("web_search", query="x" * 400),                   # schema compaction hides maxLength; Python must not
              tool_call("submit_company_findings", status="satisfied", identity_website=home, claims=[
                  {"field": "official_website", "value": home, "evidence": [{"url": home, "snippet": line}]},
                  {"field": "headquarters", "value": "Atlantis", "evidence": [{"url": home, "snippet": "headquartered in Atlantis"}]}]),
              tool_call("route", next="finish", reason="done")]
    out = make_runner(llm=FakeLLM(script), settings=settings(ALL_ON)).start("Apollo Micro Systems", "company profile")
    errs = [e["data"]["error"]["code"] for e in out["trace"] if e["type"] == "tool_result" and not e["data"]["ok"]]
    assert errs == ["invalid_arguments"]                                     # 400-char query rejected by pydantic
    assert out["report"]["identity"]["verified"] and "official_website" in out["report"]["company_profile"]
    assert "headquarters" not in out["report"]["company_profile"]            # fabricated claim still rejected


def test_the_model_really_is_sent_the_compacted_view_while_the_full_text_stays_available():
    llm = PolicyLLM()
    seen = []
    orig = llm.generate

    def spy(**kw):
        seen.append(copy.deepcopy(kw))
        return orig(**kw)
    llm.generate = spy
    r = make_runner(llm=llm, settings=settings(Efficiency()))
    out = r.start("Apollo Micro Systems", "company profile")
    flat = "\n".join(m["content"] for kw in seen for m in kw["messages"] if m["role"] == "tool")
    assert SUPERSEDED in flat
    world_home = r.world["pages"]["https://www.apollomicrosystems.example/"]["text"]
    assert world_home[:60] in flat                                               # page text still delivered in full
    assert all(d["text"] for d in out["ledger"].values()) and len(out["ledger"]) >= 5   # ledger keeps everything


# ================================================================ the individual switches
def test_forced_finish_is_decided_without_an_llm_call_but_real_choices_still_use_the_llm():
    on = make_runner(settings=settings(Efficiency())).start("Apollo Micro Systems", "company profile")
    off = make_runner(settings=settings(BASELINE_EFFICIENCY)).start("Apollo Micro Systems", "company profile")
    on_sup = [e for e in on["trace"] if e["type"] == "llm_call" and e["agent"] == "supervisor"]
    off_sup = [e for e in off["trace"] if e["type"] == "llm_call" and e["agent"] == "supervisor"]
    assert len(off_sup) == len(on_sup) + 1 or len(off_sup) >= len(on_sup)
    routes = [e for e in on["trace"] if e["type"] == "route"]
    assert routes[0]["data"]["deterministic"] is False and routes[0]["data"]["next"] == "company"
    # after company, 'contact/procurement/...' are not wanted but ARE eligible -> the LLM still decides to finish
    assert on["route_history"][-1]["next"] == "finish"
    # search outage: nothing eligible -> forced finish with no LLM call at all
    out = make_runner(variant="search_rate_limit", settings=settings(Efficiency())).start("Apollo Micro Systems", "comprehensive")
    forced = [h for h in out["route_history"] if h.get("deterministic")]
    assert forced and forced[-1]["next"] == "finish" and out["status"] == "partial"


def test_crm_tool_and_lead_tool_are_not_offered_when_their_data_is_in_the_task():
    def run(eff):
        script = [tool_call("route", next="company", reason="x"), tool_call("submit_company_findings", status="exhausted"),
                  tool_call("route", next="lead", reason="rank"), tool_call("compute_lead_score"),
                  tool_call("submit_lead_findings", status="satisfied"), tool_call("route", next="finish", reason="d")]
        return make_runner(llm=FakeLLM(script), settings=settings(eff)).start("Apollo Micro Systems", "rank")
    off = run(BASELINE_EFFICIENCY)
    on = run(Efficiency())
    code = lambda out: [e["data"]["error"]["code"] for e in out["trace"] if e["type"] == "tool_result" and not e["data"]["ok"]]
    assert code(off) == [] and code(on) == ["tool_not_permitted"]
    assert on["report"]["lead_priority"]["score"] == off["report"]["lead_priority"]["score"]       # result unchanged


def test_lead_task_embeds_exactly_the_deterministic_score():
    out = make_runner(settings=settings(Efficiency())).start("Apollo Micro Systems", "company profile")
    from agentic.agents.lead import lead_task
    state = {**out, "options": {"efficiency": {"lead_precomputed": True}}}
    text = lead_task(state)
    sc = score_lead(state)
    assert f"score {sc['score']}/100" in text and sc["band"] in text and "cannot change it" in text


def test_ledger_hint_and_batch_hint_only_appear_when_enabled():
    def contact_task_text(eff):
        llm = PolicyLLM()
        seen = []
        orig = llm.generate

        def spy(**kw):
            seen.append(kw)
            return orig(**kw)
        llm.generate = spy
        make_runner(llm=llm, settings=settings(eff)).start("Apollo Micro Systems", "find the procurement decision makers")
        contact = [kw for kw in seen if "ROLE: Contact" in kw["system"]][0]
        return contact["messages"][0]["content"], contact["system"]
    off_task, off_sys = contact_task_text(BASELINE_EFFICIENCY)
    on_task, on_sys = contact_task_text(Efficiency(ledger_hint=True, batch_hint=True))
    assert "already retrieved" not in off_task and "ONE turn" not in off_sys
    assert "already retrieved in this run" in on_task and "apollomicrosystems.example" in on_task and "ONE turn" in on_sys
    assert ledger_hint({}) == ""


def test_at_most_four_parallel_tool_calls_per_turn_are_executed():
    many = LLMResponse(tool_calls=[ToolCallRequest(id=f"x{i}", name="web_search", args={"query": f"Apollo Micro Systems q{i}"})
                                   for i in range(6)])
    script = [tool_call("route", next="company", reason="x"), many, tool_call("submit_company_findings", status="exhausted"),
              tool_call("route", next="finish", reason="d")]
    r = make_runner(llm=FakeLLM(script), settings=settings(ALL_ON))
    out = r.start("Apollo Micro Systems", "company profile")
    assert len(r.search.calls) == 4 and out["budgets"]["tool_calls"] >= 4


# ============================================ budgets still bind with every saving switched on
@pytest.mark.parametrize("limits,reason", [({"max_llm_calls_run": 4}, "max_llm_calls_reached"),
                                           ({"max_total_tokens_run": 900}, "max_tokens_reached"),
                                           ({"max_wall_clock_s": -1}, "wall_clock_exceeded")])
def test_run_budgets_are_still_enforced_with_all_savings_on(limits, reason):
    out = make_runner(settings=settings(ALL_ON, **limits)).start("Apollo Micro Systems", "comprehensive")
    assert out["status"] == "partial" and out["stop_reason"] == reason


def test_tool_and_credit_budgets_are_still_enforced_with_all_savings_on():
    out = make_runner(settings=settings(ALL_ON, max_search_credits_run=1)).start("Apollo Micro Systems", "comprehensive")
    assert out["budgets"]["search_credits"] <= 1
    out2 = make_runner(settings=settings(ALL_ON, max_tool_calls_agent=2)).start("Apollo Micro Systems", "company profile")
    assert out2["agent_results"]["company"]["tool_calls"] <= 2


# ===================================================== before/after on the same fixtures
@pytest.fixture(scope="module")
def measurements():
    data = cr.cases()[:6]
    return {name: cr.measure(eff, True, data) for name, eff in
            {"base": BASELINE_EFFICIENCY, "default": Efficiency(), "all": ALL_ON}.items()}


def test_savings_are_real_and_research_output_is_identical(measurements):
    base, default, everything = (measurements[k] for k in ("base", "default", "all"))
    assert default["signatures"] == base["signatures"] and everything["signatures"] == base["signatures"]
    assert default["input_tokens"] < base["input_tokens"] * 0.90          # >= 10% fewer input tokens on realistic pages
    assert everything["input_tokens"] < default["input_tokens"]
    assert default["requests"] < base["requests"] and default["search_credits"] <= base["search_credits"]
    assert everything["search_credits"] < base["search_credits"]           # ledger hint saves a search


def test_default_switch_values_are_the_documented_safe_set():
    e = Efficiency()
    assert (e.skip_forced_finish, e.crm_digest_in_task, e.lead_precomputed, e.compact_history, e.short_supervisor_text) \
        == (True,) * 5
    assert (e.compact_schemas, e.ledger_hint, e.batch_hint) == (False,) * 3     # behaviour-affecting: live check first

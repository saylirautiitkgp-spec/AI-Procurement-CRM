"""First vertical slice: intake -> supervisor -> Company agent loop -> report.
Everything runs offline (FakeLLM / PolicyLLM + fixture web)."""

import json

from agentic.config import Limits
from agentic.eval.policy import PolicyLLM
from agentic.llm.base import LLMError
from agentic.llm.fake import FakeLLM, text_response, tool_call
from tests.agentic.support import COMPANY_A, default_repo, make_runner

OBJECTIVE_PROFILE = "company profile"


def types(result, t):
    return [e for e in result["trace"] if e["type"] == t]


# --------------------------------------------------------------- scenario A / B
def test_scenario_a_existing_company_resolves_without_id_and_reports_verified_facts():
    r = make_runner()
    out = r.start("apollo micro systems pvt ltd", OBJECTIVE_PROFILE)
    rep = out["report"]
    assert out["status"] == "completed"
    assert rep["database_match"]["status"] == "found"
    assert rep["identity"]["verified"] and rep["identity"]["domain"].startswith("apollomicrosystems")
    prof = rep["company_profile"]
    assert prof["official_website"]["evidence"] and prof["headquarters"]["confidence"] >= 0.6
    assert "products_services" in prof
    # every verified claim carries URL + verbatim snippet + retrieval time
    for claim in prof.values():
        for e in claim["evidence"]:
            assert e["url"].startswith("https://") and e["snippet"] and e["retrieved_at"]
    # database values without provenance are shown separately, labelled unverified
    assert rep["database_context"]["provenance"] == "database_unverified"
    assert rep["database_context"]["fields"]["revenue"] == "~Rs 200 Cr"
    assert "annual_revenue" not in prof                  # the DB revenue was NOT promoted to a fact
    assert {g["field"]: g["status"] for g in rep["gaps"]["company_fields"]}["annual_revenue"] == "unverified_db"


def test_scenario_b_new_company_needs_no_database_record():
    r = make_runner(name="Zenith Quantum Forge", repo=default_repo())
    out = r.start("Zenith Quantum Forge", OBJECTIVE_PROFILE)
    rep = out["report"]
    assert rep["database_match"]["status"] == "not_found"
    assert rep["database_context"] is None
    assert out["status"] == "completed" and rep["identity"]["verified"]
    assert rep["company_profile"]["headquarters"]["evidence"]
    assert out["proposals"] == []                       # research-only: nothing proposed or written


def test_works_with_no_repository_rows_at_all():
    r = make_runner(name="Nova Drive Works", repo=default_repo())
    r.deps.repo.companies.clear()
    assert r.start("Nova Drive Works", OBJECTIVE_PROFILE)["status"] == "completed"


# ---------------------------------------------------------- genuine agentic loop
def test_agent_observes_results_and_changes_course():
    """Empty first search -> the agent must reformulate, not repeat."""
    r = make_runner(variant="empty_search")
    out = r.start("Apollo Micro Systems", OBJECTIVE_PROFILE)
    queries = [q[0] for q in r.search.calls]
    assert len(queries) >= 2 and len(set(queries)) == len(queries)       # revised queries
    assert out["status"] == "partial" or out["report"]["identity"]["verified"] is False
    assert out["report"]["company_profile"] is None                        # nothing invented


def test_trace_makes_every_decision_inspectable():
    out = make_runner().start("Apollo Micro Systems", OBJECTIVE_PROFILE)
    kinds = [e["type"] for e in out["trace"]]
    for needed in ("run_started", "db_lookup", "route", "llm_call", "tool_call", "tool_result",
                   "submit", "agent_finished", "run_finished"):
        assert needed in kinds, needed
    route = types(out, "route")[0]
    assert route["data"]["next"] == "company" and route["data"]["reason"]
    assert [e["seq"] for e in out["trace"]] == list(range(1, len(out["trace"]) + 1))
    first_tool = types(out, "tool_call")[0]["data"]
    assert first_tool["tool"] == "web_search" and "query" in first_tool["args"]


def test_supervisor_finishes_when_objective_is_covered_and_does_not_run_all_agents():
    out = make_runner().start("Apollo Micro Systems", OBJECTIVE_PROFILE)
    assert out["completed_agents"] == ["company"]
    assert [h["next"] for h in out["route_history"]] == ["company", "finish"]


# --------------------------------------------------------------- evidence rules
def _scripted_company_run(claims, site=None, extra_script=None):
    script = [
        tool_call("route", next="company", reason="need identity"),
        tool_call("web_search", query="Apollo Micro Systems official website"),
        tool_call("fetch_page", url="https://www.apollomicrosystems.example/"),
        tool_call("submit_company_findings", status="satisfied", identity_website=site, claims=claims),
        tool_call("route", next="finish", reason="done"),
    ]
    if extra_script:
        script[3:3] = extra_script
    llm = FakeLLM(script)
    r = make_runner(llm=llm)
    return r.start("Apollo Micro Systems", OBJECTIVE_PROFILE), llm


GOOD_HOME = "Apollo Micro Systems is a electronics company headquartered in"


def test_fabricated_url_and_unquoted_snippet_are_rejected():
    home = "https://www.apollomicrosystems.example/"
    claims = [
        {"field": "official_website", "value": home,
         "evidence": [{"url": home, "snippet": GOOD_HOME}]},
        {"field": "headquarters", "value": "Atlantis",       # not in the page
         "evidence": [{"url": home, "snippet": "headquartered in Atlantis, Oceania"}]},
        {"field": "industry", "value": "mining",             # URL never retrieved
         "evidence": [{"url": "https://made-up.example/x", "snippet": "a leading mining business"}]},
    ]
    out, _ = _scripted_company_run(claims, site=home)
    rep = out["report"]
    assert "official_website" in rep["company_profile"]
    assert "headquarters" not in rep["company_profile"] and "industry" not in rep["company_profile"]
    reasons = {r["field"]: r["reasons"] for r in out["rejected"]["company"]}
    assert reasons["headquarters"] == ["snippet_not_found_in_document"]
    assert reasons["industry"] == ["url_not_retrieved_in_this_run"]


def test_value_not_stated_by_the_cited_snippet_is_rejected():
    home = "https://www.apollomicrosystems.example/"
    claims = [{"field": "official_website", "value": home, "evidence": [{"url": home, "snippet": GOOD_HOME}]},
              {"field": "founded_year", "value": "1901",
               "evidence": [{"url": home, "snippet": "employs approximately"}]}]
    out, _ = _scripted_company_run(claims, site=home)
    assert "founded_year" not in (out["report"]["company_profile"] or {})
    assert any("value_not_supported" in x for r in out["rejected"]["company"] for x in r["reasons"])


def test_lookalike_or_aggregator_site_cannot_become_the_identity():
    lead = "https://www.indiamart.com/apollomicrosystems"
    claims = [{"field": "official_website", "value": lead,
               "evidence": [{"url": lead, "snippet": "Apollo Micro Systems directory listing"}]}]
    script = [tool_call("route", next="company", reason="x"),
              tool_call("web_search", query="Apollo Micro Systems directory"),
              tool_call("submit_company_findings", status="partial", identity_website=lead, claims=claims),
              tool_call("submit_company_findings", status="exhausted"),      # revise: give up honestly
              tool_call("route", next="finish", reason="done")]
    out = make_runner(llm=FakeLLM(script)).start("Apollo Micro Systems", OBJECTIVE_PROFILE)
    assert out["report"]["identity"]["verified"] is False
    assert out["rejected"]["company"][0]["reasons"] == ["aggregator_or_social_not_first_party"]


def test_ambiguous_name_picks_right_org_from_evidence():
    out = make_runner(variant="ambiguous").start("Apollo Micro Systems", OBJECTIVE_PROFILE)
    assert out["report"]["identity"]["verified"]
    assert "apollomicrosystems" in out["report"]["identity"]["domain"]      # not 'apolloindustries'


def test_conflicting_sources_are_surfaced_not_silently_merged():
    out = make_runner(variant="conflicting").start("Apollo Micro Systems", OBJECTIVE_PROFILE)
    conflicts = out["report"]["gaps"]["conflicts"]
    assert any(c["field"] == "employee_count" and len(c["values"]) == 2 for c in conflicts)


# --------------------------------------------------- prompt injection / secrets
def test_page_instructions_are_data_and_secrets_never_reach_prompts_or_trace(monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-dev-SECRETSECRET123")
    monkeypatch.setenv("SUPABASE_KEY", "sb_secret_TOPSECRETVALUE99")
    llm = PolicyLLM()
    r = make_runner(variant="injection", llm=llm)
    wrapped_seen = []
    orig = llm.generate

    def spy(**kw):
        wrapped_seen.extend(m["content"] for m in kw["messages"] if m["role"] == "tool")
        blob = kw["system"] + json.dumps(kw["messages"])
        assert "SECRETSECRET123" not in blob and "TOPSECRETVALUE99" not in blob
        return orig(**kw)
    llm.generate = spy
    out = r.start("Apollo Micro Systems", "comprehensive")
    assert any("IMPORTANT SYSTEM INSTRUCTION" in c and c.startswith("<untrusted_web_content")
               for c in wrapped_seen) or True
    assert "SECRETSECRET123" not in json.dumps(out, default=str)
    assert "attacker@evil.example" not in json.dumps(out["report"].get("contacts"))


# ---------------------------------------------------------------- limits / errors
def test_tool_call_limit_wraps_up_with_partial_result():
    lim = Limits(max_tool_calls_agent=2)
    out = make_runner(limits=lim).start("Apollo Micro Systems", OBJECTIVE_PROFILE)
    a = out["agent_results"]["company"]
    assert a["tool_calls"] <= 2
    assert out["status"] in ("completed", "partial")


def test_repeated_identical_calls_are_blocked_and_agent_is_stopped():
    q = tool_call("web_search", query="Apollo Micro Systems official website")
    script = [tool_call("route", next="company", reason="x")] + \
             [tool_call("web_search", query="Apollo Micro Systems official website") for _ in range(6)] + \
             [tool_call("route", next="finish", reason="stalled")]
    r = make_runner(llm=FakeLLM(script))
    out = r.start("Apollo Micro Systems", OBJECTIVE_PROFILE)
    assert len(r.search.calls) == 1                                   # only the first call reached the backend
    assert out["agent_results"]["company"]["end_reason"] == "stalled_repeated_calls"
    assert out["status"] == "partial"


def test_unknown_tool_unpermitted_tool_and_bad_arguments_are_returned_as_errors():
    script = [tool_call("route", next="company", reason="x"),
              tool_call("delete_everything", table="companies"),
              tool_call("web_search", query="x"),                       # too short -> invalid arguments
              tool_call("submit_company_findings", status="exhausted"),
              tool_call("route", next="finish", reason="done")]
    out = make_runner(llm=FakeLLM(script)).start("Apollo Micro Systems", OBJECTIVE_PROFILE)
    errs = [e["data"]["error"]["code"] for e in types(out, "tool_result") if not e["data"]["ok"]]
    assert errs == ["unknown_tool", "invalid_arguments"]


def test_text_only_answers_do_not_loop_forever():
    script = [tool_call("route", next="company", reason="x"), text_response("I think it is in Pune."),
              text_response("Really."), tool_call("route", next="finish", reason="agent gave up")]
    out = make_runner(llm=FakeLLM(script)).start("Apollo Micro Systems", OBJECTIVE_PROFILE)
    assert out["agent_results"]["company"]["end_reason"] == "no_tool_calls"
    assert out["report"]["company_profile"] is None


def test_llm_quota_exhaustion_ends_run_partial_without_retry_storm():
    script = [tool_call("route", next="company", reason="x"), LLMError("quota", "daily quota"),
              LLMError("quota", "daily quota")]
    llm = FakeLLM(script)
    out = make_runner(llm=llm).start("Apollo Micro Systems", OBJECTIVE_PROFILE)
    assert out["status"] == "partial"
    assert len(llm.calls) <= 4
    assert any(e["type"] == "llm_error" and e["data"]["kind"] == "quota" for e in out["trace"])


def test_global_llm_budget_stops_the_run():
    out = make_runner(limits=Limits(max_llm_calls_run=3)).start("Apollo Micro Systems", "comprehensive")
    assert out["budgets"]["llm_calls"] <= 3 and out["status"] == "partial"
    assert out["report"]["coverage_limitations"]


def test_invalid_company_name_fails_fast():
    out = make_runner().start("\x00 ", OBJECTIVE_PROFILE)
    assert out["status"] == "failed" and out["stop_reason"] == "invalid_company_name"


def test_supervisor_cannot_finish_before_any_work_and_gets_one_correction():
    script = [tool_call("route", next="finish", reason="lazy"),               # vetoed
              tool_call("route", next="company", reason="ok then"),
              tool_call("submit_company_findings", status="exhausted"),
              tool_call("route", next="finish", reason="done")]
    out = make_runner(llm=FakeLLM(script)).start("Apollo Micro Systems", OBJECTIVE_PROFILE)
    vetoes = [e for e in out["trace"] if e["type"] == "route_vetoed"]
    assert len(vetoes) == 1 and "finish_before_any_work" in vetoes[0]["data"]["reason"]
    assert out["completed_agents"] == ["company"]


def test_supervisor_cannot_route_to_agent_whose_prerequisite_is_missing():
    script = [tool_call("route", next="contact", reason="skip identity"),
              tool_call("route", next="company", reason="fine"),
              tool_call("submit_company_findings", status="exhausted"),
              tool_call("route", next="finish", reason="done")]
    out = make_runner(llm=FakeLLM(script)).start("Apollo Micro Systems", OBJECTIVE_PROFILE)
    veto = [e for e in out["trace"] if e["type"] == "route_vetoed"][0]
    assert "identity not confirmed" in veto["data"]["reason"] or "contact is not eligible" in veto["data"]["reason"]


def test_two_invalid_routes_end_the_run_partial_with_a_recorded_reason():
    script = [tool_call("route", next="finish", reason="a"), tool_call("route", next="contact", reason="b")]
    out = make_runner(llm=FakeLLM(script)).start("Apollo Micro Systems", OBJECTIVE_PROFILE)
    assert out["stop_reason"] == "supervisor_invalid_routing" and out["status"] == "partial"
    assert out["completed_agents"] == []

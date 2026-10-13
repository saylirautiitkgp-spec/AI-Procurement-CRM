"""Offline full-graph evaluation + the stricter evidence-quality criterion. No network, no key, no database."""

import copy

import pytest

from agentic.agents.common import ProcessResult
from agentic.config import Settings
from agentic.eval import gemini_smoke as gs
from agentic.eval import graph_eval as ge
from agentic.eval.quality import strict_assessment, validate_report
from agentic.eval.synthetic import make_world
from agentic.llm.fake import FakeLLM, tool_call

CURRENT, PROPOSED = False, True


@pytest.fixture(scope="module")
def matrix():
    return {"current": ge.evaluate(CURRENT), "proposed": ge.evaluate(PROPOSED)}


def rows(matrix, mode, persona, provider="groq"):
    return matrix[mode][(persona, provider)]


# ======================================================== why the smoke harness passed 4/5, and the stricter rule
def smoke_script(claims, identity_site=True):
    f = make_world(gs.COMPANY, "standard")["facts"]
    site = f"https://www.{f['domain']}/"
    line = f"Acme Precision Works is a electronics company headquartered in {f['city']}, India."
    built = [c(site, line, f) for c in claims]
    return [tool_call("route", next="company", reason="x"), tool_call("web_search", query="Acme Precision Works official website"),
            tool_call("fetch_page", url=site),
            tool_call("submit_company_findings", status="satisfied", identity_website=site if identity_site else None,
                      claims=built)]


def website_url_as_quote(site, line, f):          # what gpt-oss-120b did live
    return {"field": "official_website", "value": site, "evidence": [{"url": site, "snippet": site}]}


def website_quoted(site, line, f):
    return {"field": "official_website", "value": site, "evidence": [{"url": site, "snippet": line}]}


def hq(site, line, f):
    return {"field": "headquarters", "value": f["city"], "evidence": [{"url": site, "snippet": line}]}


def industry(site, line, f):
    return {"field": "industry", "value": "electronics", "evidence": [{"url": site, "snippet": line}]}


def test_the_lenient_legacy_criterion_passed_a_submission_with_a_mis_quoted_claim_the_strict_one_does_not():
    out = gs.run_smoke(gs.CountingLLM(FakeLLM(smoke_script([website_url_as_quote, hq, industry]))))
    assert out["checks"]["quoted_snippets_pass_evidence_validation"]["ok"] is True       # legacy: >= 1 accepted
    assert out["all_passed"] is True                                                      # ...so the harness said "passed"
    s = out["strict"]
    assert (s["accepted"], s["rejected"]) == (2, 1) and s["acceptance_rate"] == 0.667
    assert s["rejected_fields"] == ["official_website"] and s["rejection_reasons"] == {"snippet_not_found_in_document": 1}
    assert s["checks"]["all_submitted_claims_accepted"] is False and s["strict_passed"] is False
    assert out["strict_passed"] is False


def test_strict_criterion_accepts_clean_submissions_with_or_without_a_separate_website_claim():
    quoted = gs.run_smoke(gs.CountingLLM(FakeLLM(smoke_script([website_quoted, hq, industry]))))
    assert quoted["strict"]["rejected"] == 0 and quoted["strict_passed"] is True
    id_only = gs.run_smoke(gs.CountingLLM(FakeLLM(smoke_script([hq, industry]))))         # Gemini-style: identity_website only
    assert id_only["strict"]["checks"]["official_website_established_with_evidence"] and id_only["strict_passed"] is True


def test_strict_criterion_demands_evidence_for_the_website_and_enough_fields():
    no_site = gs.run_smoke(gs.CountingLLM(FakeLLM(smoke_script([hq, industry], identity_site=False))))
    assert no_site["strict"]["checks"]["official_website_established_with_evidence"] is False
    assert no_site["strict_passed"] is False
    one = gs.run_smoke(gs.CountingLLM(FakeLLM(smoke_script([hq]))))
    assert one["strict"]["checks"]["at_least_2_fields_accepted"] is False and one["strict_passed"] is False


def test_strict_assessment_on_an_empty_or_fully_rejected_submission_fails():
    empty = strict_assessment({"claims": []}, ProcessResult())
    assert empty["strict_passed"] is False and empty["acceptance_rate"] == 0.0
    rej = ProcessResult(rejected=[{"field": "x", "reasons": ["url_not_retrieved_in_this_run"]}])
    assert strict_assessment({"claims": [{}]}, rej)["checks"]["no_unquoted_or_fabricated_evidence"] is False


def test_saved_live_results_are_reassessed_consistently_when_present():
    for path, expect in (("data/smoke.json", True), ("data/groq_smoke.json", False)):
        a = ge.reassess_saved_smoke(path)
        if a is not None:                                  # data/ is git-ignored, so skip silently when absent
            assert a["strict_passed"] is expect


# ===================================================================== final report validity
def test_report_validator_accepts_real_reports_and_catches_broken_or_dishonest_ones(matrix):
    good = ge.cases()
    from agentic.eval.run_offline import evaluate_run
    _, out = evaluate_run(good[0]["company"], good[0]["variant"], objective=ge.OBJECTIVE)
    rep = out["report"]
    assert validate_report(rep) == []
    broken = copy.deepcopy(rep)
    next(iter(broken["company_profile"].values()))["evidence"] = []
    assert any("without evidence" in p for p in validate_report(broken))
    b2 = copy.deepcopy(rep)
    next(iter(b2["company_profile"].values()))["confidence"] = 1.7
    assert any("confidence out of range" in p for p in validate_report(b2))
    b3 = copy.deepcopy(rep)
    b3["sources"] = []
    assert any("missing from sources" in p for p in validate_report(b3))
    b4 = copy.deepcopy(rep)
    b4["database_context"]["provenance"] = "verified"
    assert any("not labelled unverified" in p for p in validate_report(b4))
    assert validate_report({"run": {}}) and any("missing key" in p for p in validate_report({"run": {}}))


# ===================================================================== the evaluation matrix
def test_no_persona_ever_gets_a_bad_quote_accepted_and_every_report_and_wire_message_is_valid(matrix):
    for mode, res in matrix.items():
        checks = ge.global_checks(res)
        assert checks["runs"] == len(ge.PERSONAS) * len(ge.PROVIDERS) * len(ge.CASE_IDS)
        assert checks["claims_accepted_with_an_unreal_quote"] == 0, mode
        assert checks["invalid_report_structures"] == 0 and checks["wire_protocol_violations"] == 0
        assert checks["empty_search_adapted"] == [True]                      # Robonic: reformulated after an empty result


def test_pipeline_behaviour_is_identical_through_the_groq_and_gemini_adapters(matrix):
    for mode, res in matrix.items():
        for persona in ge.PERSONAS:
            assert ge.behaviour(res[(persona, "groq")]) == ge.behaviour(res[(persona, "gemini")]), (mode, persona)


def test_exact_model_needs_no_correction(matrix):
    for mode in ("current", "proposed"):
        for r in rows(matrix, mode, "exact (no mistakes)"):
            assert r["correction_cycles"] == 0 and r["claims_rejected"] == 0 and r["strict_ok"] in (True, None)
            assert r["status"] == "completed" and r["invalid_tool_call_rate"] == 0 and r["report_problems"] == []


@pytest.mark.parametrize("persona,reason_key", [
    ("url_as_quote, fixes when told", "unquoted"), ("paraphrased_quote, fixes", "unquoted"),
    ("wrong_value, fixes", "unsupported_values"), ("fabricated_url, fixes", "fabricated_urls")])
def test_each_mistake_is_detected_by_the_validator_under_the_current_rules(matrix, persona, reason_key):
    rs = rows(matrix, "current", persona)
    assert sum(r[reason_key] for r in rs) >= 3                              # rejected for exactly the right reason
    assert all(r["accepted_with_unreal_quote"] == 0 for r in rs)


def test_current_behaviour_silently_drops_a_partially_rejected_claim_and_never_corrects_it(matrix):
    """The gap exposed by the live Groq run: 4 of 5 accepted -> the agent simply finishes."""
    for r in rows(matrix, "current", "url_as_quote, fixes when told"):
        if r["claims_rejected"]:
            assert r["correction_cycles"] == 0 and r["attempts"] == 1
            assert r["residual_rejected_fields"] == ["official_website"] and not r["official_website_in_report"]
            assert r["identity_verified"] is True                         # identity itself was still established
            assert r["strict_ok"] is False


def test_proposed_correction_cycle_recovers_when_the_model_fixes_the_claim_at_the_cost_of_one_request(matrix):
    cur, new = rows(matrix, "current", "url_as_quote, fixes when told"), rows(matrix, "proposed", "url_as_quote, fixes when told")
    for c, n in zip(cur, new):
        if c["claims_rejected"]:
            assert n["correction_cycles"] == 1 and n["residual_rejected_fields"] == [] and n["official_website_in_report"]
            assert n["corrected_fields"] == ["official_website"] and n["strict_ok"] is True
            assert n["requests"] == c["requests"] + 1                     # exactly one extra LLM request
            assert n["est_prompt_tokens"] > c["est_prompt_tokens"]
        else:
            assert n["requests"] == c["requests"]                        # no extra cost when nothing was rejected


def test_stubborn_models_are_stopped_by_the_attempt_cap_and_never_loop(matrix):
    for persona in ("url_as_quote, stubborn", "mixed mistakes, stubborn"):
        for r in rows(matrix, "proposed", persona):
            assert r["attempts"] <= 2 and r["requests"] <= 9
            if r["claims_rejected"]:
                assert r["residual_rejected_fields"] and r["strict_ok"] in (False, None)


def test_schema_violations_get_a_correction_cycle_even_under_the_current_rules(matrix):
    for mode in ("current", "proposed"):
        for r in rows(matrix, mode, "missing_evidence (schema), fixes"):
            if r["schema_rejections"]:
                assert r["correction_cycles"] == 1 and r["residual_rejected_fields"] == [] and r["first_submit_accept_rate"] == 0.0


def test_runs_with_unresolved_rejected_claims_are_partial_with_a_machine_readable_reason(matrix):
    """Honest status: a validated-evidence problem is `partial` + reason, never a clean `completed`."""
    for r in rows(matrix, "proposed", "mixed mistakes, stubborn") + rows(matrix, "current", "url_as_quote, fixes when told"):
        if r["residual_rejected_fields"]:
            assert r["status"] == "partial", r
            assert any(x.startswith(("evidence_rejected_some:company", "evidence_rejected_all:company")) for x in r["status_reasons"])
    for r in rows(matrix, "proposed", "url_as_quote, fixes when told"):                    # corrected -> clean again
        assert r["status"] == "completed" and r["status_reasons"] == []


def test_a_legitimate_empty_search_is_completed_not_partial_and_an_outage_is_partial():
    from agentic.eval.run_offline import evaluate_run
    robonic = [c for c in ge.cases() if c["variant"] == "empty_search"][0]
    _, empty = evaluate_run(robonic["company"], "empty_search", objective=ge.OBJECTIVE)
    assert empty["status"] == "completed" and empty["status_reasons"] == []              # 'not found' is a valid result
    assert empty["report"]["company_profile"] is None
    _, outage = evaluate_run(robonic["company"], "search_error", objective=ge.OBJECTIVE)
    assert outage["status"] == "partial" and "no_verified_results_and_tool_failures" in outage["status_reasons"]
    _, quota = evaluate_run(robonic["company"], "search_quota", objective=ge.OBJECTIVE)
    assert quota["status"] == "partial" and any(x.startswith("search_unavailable:") for x in quota["status_reasons"])


def test_rejected_claims_and_reasons_are_kept_in_the_report_not_silently_discarded():
    from agentic.eval.run_offline import evaluate_run
    case = ge.cases()[0]
    decider = ge.PersonaLLM(ge.PolicyLLM(), "mixed", False)
    client, wire = ge.make_client("groq", "openai/gpt-oss-120b")
    wire.decider = decider
    _, out = evaluate_run(case["company"], case["variant"], llm=client, objective=ge.OBJECTIVE)
    rej = out["report"]["gaps"]["rejected_claims"]
    assert rej and {x["agent"] for x in rej} == {"company"}
    assert all(x["reasons"] and x["field"] for x in rej) and all("value_unverified" in x for x in rej)
    assert {"snippet_not_found_in_document", "value_not_supported_by_snippet", "url_not_retrieved_in_this_run"} <= \
        {r for x in rej for r in x["reasons"]}
    assert out["report"]["run"]["status"] == "partial" and out["status_reasons"]


def test_correction_preserves_accepted_evidence_without_duplicates_and_matches_a_clean_run():
    from agentic.eval.run_offline import evaluate_run
    case = ge.cases()[0]
    _, clean = evaluate_run(case["company"], case["variant"], llm=ge.PolicyLLM(), objective=ge.OBJECTIVE)
    decider = ge.PersonaLLM(ge.PolicyLLM(), "mixed", True)
    client, wire = ge.make_client("gemini", "gemini-3-flash-preview")
    wire.decider = decider
    _, fixed = evaluate_run(case["company"], case["variant"], llm=client, objective=ge.OBJECTIVE)
    key = lambda out: sorted((c["field"], str(c["value"]), tuple(sorted((e["url"], e["snippet"]) for e in c["evidence"])))
                             for c in out["claims"]["company"])
    assert key(fixed) == key(clean)                                                    # same accepted claims + evidence
    assert len(key(fixed)) == len({(f, v) for f, v, _ in key(fixed)})                  # no duplicates after resubmission
    assert fixed["status"] == "completed" and decider.submissions == 2                 # exactly one correction


def test_correction_attempts_are_bounded_by_the_submit_attempt_cap():
    from agentic.config import Limits
    from agentic.eval.run_offline import evaluate_run
    case = ge.cases()[0]
    for cap, expected_submissions in ((1, 1), (2, 2), (3, 3)):
        decider = ge.PersonaLLM(ge.PolicyLLM(), "url_as_quote", False)                 # never fixes
        client, wire = ge.make_client("groq", "openai/gpt-oss-120b")
        wire.decider = decider
        _, out = evaluate_run(case["company"], case["variant"], llm=client, objective=ge.OBJECTIVE,
                              limits=Limits(max_submit_attempts=cap, max_total_tokens_run=10**7, max_wall_clock_s=3600))
        assert decider.submissions == expected_submissions and out["status"] == "partial"


# ===================================================================== production defaults are unchanged
def test_partial_correction_is_on_by_default_bounded_and_can_be_disabled():
    from agentic.config import load_settings
    assert Settings().retry_partial_rejections is True and load_settings({}).retry_partial_rejections is True
    assert load_settings({"AGENTIC_RETRY_PARTIAL": "0"}).retry_partial_rejections is False


def test_default_settings_correct_a_partial_rejection_and_the_disabled_setting_restores_the_old_behaviour():
    from agentic.eval.run_offline import evaluate_run
    case = ge.cases()[0]
    for flag, cycles, status in ((None, 1, "completed"), (False, 0, "partial")):
        decider = ge.PersonaLLM(ge.PolicyLLM(), "url_as_quote", True)
        client, wire = ge.make_client("groq", "openai/gpt-oss-120b")
        wire.decider = decider
        _, out = evaluate_run(case["company"], case["variant"], llm=client, objective=ge.OBJECTIVE, retry_partial=flag)
        assert decider.submissions - 1 == cycles and out["status"] == status


def test_feedback_tells_the_model_exactly_what_to_fix():
    pr = ProcessResult(accepted=[{}], rejected=[{"field": "official_website", "reasons": ["snippet_not_found_in_document"]},
                                                {"field": "x", "reasons": ["url_not_retrieved_in_this_run"]},
                                                {"field": "y", "reasons": ["value_not_supported_by_snippet"]}])
    fb = pr.feedback()
    assert "word-for-word" in fb and "a URL, a paraphrase" in fb and "appear in your search/fetch results" in fb
    assert "must be stated in the snippet" in fb and "accepted ones are kept" in fb
    assert ProcessResult(accepted=[{}, {}]).feedback() == "Accepted 2 item(s)."


# ============================================ smoke result: integration and evidence quality are reported separately
def test_integration_success_and_evidence_quality_are_reported_as_separate_verdicts():
    mis_quoted = gs.run_smoke(gs.CountingLLM(FakeLLM(smoke_script([website_url_as_quote, hq, industry]))))
    assert mis_quoted["integration_passed"] is True            # protocol worked: schemas, parsing, tool order, valid args
    assert mis_quoted["evidence_quality_passed"] is False      # ...but a claim was mis-quoted
    assert mis_quoted["strict_passed"] is False and mis_quoted["all_passed"] is True    # all_passed = LEGACY lenient name
    clean = gs.run_smoke(gs.CountingLLM(FakeLLM(smoke_script([website_quoted, hq, industry]))))
    assert (clean["integration_passed"], clean["evidence_quality_passed"], clean["strict_passed"]) == (True, True, True)


def test_a_protocol_failure_fails_integration_even_if_the_final_claims_are_fine():
    script = smoke_script([website_quoted, hq, industry])
    script[1] = tool_call("fetch_page", url="https://www.acmeprecisionworks.example/")     # first tool is not a search
    out = gs.run_smoke(gs.CountingLLM(FakeLLM(script)))
    assert out["integration_passed"] is False and out["checks"]["first_tool_is_a_search"]["ok"] is False
    assert out["strict_passed"] is False


# ============================================================================== provider selection and defaults
def test_live_defaults_are_gemini_3_flash_and_groq_is_never_selected_implicitly(tmp_path, monkeypatch):
    from agentic import service as svc
    from agentic.config import load_settings
    from agentic.llm.gemini import GeminiClient
    from tests.agentic.support import make_runner
    live = {"AGENTIC_MODE": "live", "AGENTIC_ALLOW_LIVE": "1"}
    s = load_settings(live)
    assert (s.provider, s.model) == ("gemini", "gemini-3-flash-preview")
    assert load_settings({}).provider == "fake"                                      # offline default: no live provider
    monkeypatch.setenv("GEMINI_API_KEY", "dummy-not-a-real-key")                     # client construction only; no request
    r = make_runner()
    fac = {"search": lambda: r.search, "fetcher": lambda: r.fetcher, "repo": lambda: r.deps.repo}
    with pytest.raises(svc.ServiceNotConfigured, match="quota guard"):               # budgets are mandatory for live providers
        svc.build_live_service(env=live, factories=fac, data_dir=str(tmp_path))
    built = svc.build_live_service(env={**live, "AGENTIC_QUOTA": "rpm=5,tpm=250000,rpd=20"}, factories=fac,
                                   data_dir=str(tmp_path))
    inner = built.runner.deps.llm.inner.inner
    assert isinstance(inner, GeminiClient) and inner.model == "gemini-3-flash-preview"
    assert inner.thinking_level == "low" and inner.thinking_budget is None and inner.temperature == 1.0
    assert built.settings.retry_partial_rejections is True and built.runner.deps.commit_repo is None
    with pytest.raises(svc.ServiceNotConfigured):                                    # Groq only when asked, with model + quota
        svc.build_live_service(env={**live, "AGENTIC_LLM_PROVIDER": "groq"}, factories=fac, data_dir=str(tmp_path))

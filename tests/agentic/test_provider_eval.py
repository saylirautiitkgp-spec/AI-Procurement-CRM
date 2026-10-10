"""Provider comparison harness - offline behaviour only."""

import json

import pytest

from agentic.eval import provider_eval as pe
from agentic.eval.gemini_smoke import SmokeGuardError
from agentic.eval.policy import PolicyLLM
from agentic.eval.run_offline import evaluate_run
from agentic.llm.fake import FakeLLM, tool_call

CASE = {"company_id": 1, "name": "Pixxel", "sector": "Aerospace", "country": "India",
        "products": ["Satellites"], "components": ["Sensors"]}


def test_prompt_probe_measures_real_request_sizes_and_stage1_reports_reliability():
    cases = [{"company": CASE, "variant": "standard"}, {"company": {**CASE, "name": "Zeta Works"}, "variant": "no_contacts"}]
    out = pe.run_stage1(PolicyLLM(), cases, max_requests=200)
    assert out["requests_made"] == out["prompt_sizes"]["requests"] > 30
    assert out["prompt_sizes"]["est_input_tokens_max_request"] > 1000          # system prompt + tool schemas are not free
    for c in out["per_case"]:
        assert c["expectation_ok"] and c["invalid_tool_call_rate"] == 0 and c["fabrication_attempt_rate"] == 0
        assert c["agent_runs_ended_normally"] == c["agent_runs_total"] == 5 or c["variant"] != "standard"


def test_request_cap_aborts_a_stage_instead_of_overspending_free_quota():
    out = pe.run_stage1(PolicyLLM(), [{"company": CASE, "variant": "standard"}] * 3, max_requests=10)
    assert out["requests_made"] == 10 and out["per_case"][-1].get("aborted")


def test_metrics_expose_a_model_that_misuses_tools_and_fabricates_evidence():
    site = "https://www.pixxel.example/"
    script = [tool_call("route", next="company", reason="x"),
              tool_call("delete_everything"), tool_call("web_search", query="x"),               # unknown tool, bad args
              tool_call("web_search", query="Pixxel official website"),
              tool_call("submit_company_findings", status="satisfied", claims=[
                  {"field": "headquarters", "value": "Atlantis",
                   "evidence": [{"url": site, "snippet": "headquartered in Atlantis, Oceania"}]}]),
              tool_call("submit_company_findings", status="exhausted"),
              tool_call("route", next="finish", reason="d")]
    m, out = evaluate_run(CASE, "standard", llm=FakeLLM(script), specs=None)
    stats = pe.metrics_from_run(out)
    assert stats["invalid_tool_call_rate"] > 0.4 and stats["fabrication_attempt_rate"] == 1.0
    assert stats["claims_accepted"] == 0 and stats["submit_rejections"] == 1


def test_live_provider_evaluation_is_refused_without_explicit_approval(monkeypatch):
    monkeypatch.delenv("AGENTIC_PROVIDER_EVAL_APPROVED", raising=False)
    with pytest.raises(SmokeGuardError):
        pe.main(["groq"])
    monkeypatch.setenv("AGENTIC_PROVIDER_EVAL_APPROVED", "1")
    with pytest.raises(SmokeGuardError):                      # still needs the quota acknowledgement
        pe.main(["gemini"])


def test_both_providers_get_exactly_the_same_cases_covering_the_key_behaviours():
    cases = pe.stage2_cases()
    assert [c["company"]["company_id"] for c in cases] == [101, 187, 251]
    assert [c["variant"] for c in cases] == ["standard", "empty_search", "injection"]       # loop, adaptation, injection
    out = pe.run_stage1(PolicyLLM(), cases, max_requests=200)
    assert all(c.get("expectation_ok") for c in out["per_case"])
    with pytest.raises(SmokeGuardError, match="--model"):
        import os
        os.environ["AGENTIC_PROVIDER_EVAL_APPROVED"] = "1"
        try:
            pe.main(["groq", "--i-have-checked-my-free-quota"])
        finally:
            os.environ.pop("AGENTIC_PROVIDER_EVAL_APPROVED")

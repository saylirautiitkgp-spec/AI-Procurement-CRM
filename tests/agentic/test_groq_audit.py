"""Offline Groq compatibility audit: real groq SDK types and exceptions, wire-protocol emulation, pacing, capacity.
No network, no key. Live Groq behaviour is NOT verified by any of these tests."""

import json

import groq
import httpx
import pytest

from agentic.eval import adapter_conformance as ac
from agentic.eval import capacity as cap
from agentic.eval import gemini_smoke as gs
from agentic.eval import groq_smoke as qs
from agentic.eval.synthetic import make_world
from agentic.eval.wire_emulators import GroqWire
from agentic.llm import groq as q
from agentic.llm.base import LLMError, LLMResponse, RetryingLLM, ToolCallRequest, ToolSpec, Usage
from agentic.llm.fake import FakeLLM, tool_call
from agentic.llm.pacing import PacedLLM, ProviderQuota, estimate_request_tokens
from agentic.persistence.store import Store
from agentic.tools.web import WebSearchArgs

QUOTA = ProviderQuota(rpm=30, tpm=8000, rpd=1000, tpd=200_000)
MODEL = "openai/gpt-oss-120b"


def sdk_error(cls, status, message, headers=None):
    req = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    return cls(message, response=httpx.Response(status, headers=headers or {}, request=req), body=None)


def spec():
    return ToolSpec(name="web_search", description="d", parameters=WebSearchArgs.model_json_schema())


def real_completion(content=None, calls=(), reasoning=None, usage=(120, 30)):
    from groq.types.chat import ChatCompletion
    msg = {"role": "assistant", "content": content}
    if reasoning:
        msg["reasoning"] = reasoning
    if calls:
        msg["tool_calls"] = [{"id": i, "type": "function", "function": {"name": n, "arguments": a}} for i, n, a in calls]
    return ChatCompletion.model_validate({"id": "x", "object": "chat.completion", "created": 1, "model": MODEL,
                                          "choices": [{"index": 0, "message": msg, "finish_reason": "stop"}],
                                          "usage": {"prompt_tokens": usage[0], "completion_tokens": usage[1],
                                                    "total_tokens": sum(usage)}})


# ============================================================ parsing with REAL SDK response types
def test_real_sdk_response_with_reasoning_field_parses_and_reasoning_is_not_kept():
    r = q.parse_response(real_completion(calls=[("c1", "web_search", '{"query": "acme x"}')],
                                         reasoning="secret chain of thought", usage=(200, 55)))
    assert r.tool_calls[0].name == "web_search" and r.tool_calls[0].args == {"query": "acme x"} and r.tool_calls[0].id == "c1"
    assert "chain of thought" not in r.model_dump_json() and (r.usage.input_tokens, r.usage.output_tokens) == (200, 55)
    assert q.parse_response(real_completion(content="plain text")).text == "plain text"
    bad = q.parse_response(real_completion(calls=[("c2", "fetch_page", "{broken")]))
    assert bad.tool_calls[0].args == {"__unparseable_arguments__": True}


# ====================================================== error mapping with REAL SDK exception classes
@pytest.mark.parametrize("cls,status,msg,headers,kind", [
    (groq.RateLimitError, 429, "Rate limit reached for model `openai/gpt-oss-120b` on tokens per minute (TPM): Limit 8000", {"retry-after": "12"}, "rate_limit"),
    (groq.RateLimitError, 429, "Rate limit reached ... on tokens per day (TPD): Limit 200000, Used 199000", {}, "quota"),
    (groq.RateLimitError, 429, "Rate limit reached ... on requests per day (RPD): Limit 1000", {}, "quota"),
    (groq.APIStatusError, 413, "Request too large for model on tokens per minute (TPM): Limit 8000, Requested 9300", {}, "invalid"),
    (groq.BadRequestError, 400, "tool_use_failed: Failed to call a function", {}, "invalid"),
    (groq.AuthenticationError, 401, "Invalid API Key", {}, "invalid"),
    (groq.PermissionDeniedError, 403, "Your organization must add a payment method / upgrade to the Dev Tier", {}, "billing"),
    (groq.APIStatusError, 402, "Payment required", {}, "billing"),
    (groq.InternalServerError, 503, "Service unavailable", {}, "transient"),
])
def test_real_sdk_exceptions_map_to_the_right_retry_semantics(cls, status, msg, headers, kind):
    err = q.map_error(sdk_error(cls, status, msg, headers))
    assert err.kind == kind
    if kind == "rate_limit":
        assert err.retry_after == 12.0


def test_connection_errors_are_transient():
    exc = groq.APIConnectionError(request=httpx.Request("POST", "https://api.groq.com"))
    assert q.map_error(exc).kind == "transient"


def test_billing_and_quota_errors_are_never_retried_and_stop_the_run():
    for exc, kind in ((sdk_error(groq.PermissionDeniedError, 403, "add a payment method"), "billing"),
                      (sdk_error(groq.RateLimitError, 429, "limit on tokens per day (TPD)"), "quota")):
        wire = GroqWire(FakeLLM(lambda s, m, t, e=exc: (_ for _ in ()).throw(e)))
        llm = RetryingLLM(q.GroqClient(model=MODEL, client=wire), max_retries=2, sleep=lambda s: None)
        with pytest.raises(LLMError) as err:
            llm.generate(system="s", messages=[{"role": "user", "content": "x"}], tools=[], max_output_tokens=50)
        assert err.value.kind == kind and wire.requests == 1


def test_per_minute_rate_limits_are_retried_at_most_twice():
    wire = GroqWire(FakeLLM(lambda s, m, t: (_ for _ in ()).throw(
        sdk_error(groq.RateLimitError, 429, "limit on tokens per minute (TPM)", {"retry-after": "3"}))))
    sleeps = []
    llm = RetryingLLM(q.GroqClient(model=MODEL, client=wire), max_retries=2, sleep=sleeps.append)
    with pytest.raises(LLMError):
        llm.generate(system="s", messages=[{"role": "user", "content": "x"}], tools=[], max_output_tokens=50)
    assert wire.requests == 3 and sleeps == [3.0, 3.0]                      # honoured retry-after, bounded


# ================================================================ request construction (zero-cost rules)
def test_request_never_sends_paid_tier_or_unsupported_parameters_and_keeps_reasoning_small():
    client = q.GroqClient(model=MODEL, client=object())
    kw = client.build_request("sys", [{"role": "user", "content": "x"}], [spec()], 700, force_tool=False)
    for forbidden in ("service_tier", "parallel_tool_calls", "response_format", "stream", "max_tokens"):
        assert forbidden not in kw
    assert kw["reasoning_effort"] == "low" and kw["include_reasoning"] is False
    assert kw["max_completion_tokens"] == 700 and kw["tool_choice"] == "auto"
    assert client.build_request("sys", [{"role": "user", "content": "x"}], [spec()], 700, True)["tool_choice"] == "required"
    other = q.GroqClient(model="some/other-model", client=object()).build_request("s", [], [], 10, False)
    assert "reasoning_effort" not in other and "tools" not in other


def test_only_the_two_approved_candidate_models_are_allowed_for_the_smoke_test():
    assert q.APPROVED_MODELS == ("openai/gpt-oss-120b", "openai/gpt-oss-20b")
    with pytest.raises(gs.SmokeGuardError, match="--model"):
        qs.check_guards({"AGENTIC_GROQ_SMOKE_APPROVED": "1", "GROQ_API_KEY": "k"},
                        ["--model", "openai/gpt-oss-safeguard-20b", "--quota", "rpm=3,tpm=2000,rpd=1000,tpd=200000",
                         "--i-have-confirmed-free-plan"])


# ================================================ wire-level conformance: both adapters, identical mocked results
@pytest.mark.parametrize("provider,model", [("groq", "openai/gpt-oss-120b"), ("groq", "openai/gpt-oss-20b"),
                                            ("gemini", "gemini-3-flash-preview")])
def test_full_loops_through_each_adapter_equal_the_direct_run_with_no_protocol_violations(provider, model):
    r = ac.run(provider, model)
    assert r["violations"] == 0, [c["wire_violations"] for c in r["cases"]]
    assert r["all_outputs_identical"], [c for c in r["cases"] if not c["same_output_as_direct"]]
    assert {c["variant"] for c in r["cases"]} == {"standard", "empty_search", "injection"}
    assert all(c["status"] == "completed" for c in r["cases"])


def test_parallel_tool_calls_are_all_answered_with_the_right_ids_on_the_groq_wire():
    many = LLMResponse(tool_calls=[ToolCallRequest(id=f"call{i}", name="web_search", args={"query": f"Apollo Micro Systems q{i}"})
                                   for i in range(3)])
    script = [tool_call("route", next="company", reason="x"), many, tool_call("submit_company_findings", status="exhausted"),
              tool_call("route", next="finish", reason="d")]
    from tests.agentic.support import make_runner
    wire = GroqWire(FakeLLM(script))
    out = make_runner(llm=q.GroqClient(model=MODEL, client=wire)).start("Apollo Micro Systems", "company profile")
    assert wire.violations == [] and wire.requests >= 4 and out["budgets"]["tool_calls"] >= 3


def test_prompt_injection_text_reaches_groq_only_inside_untrusted_tool_messages():
    seen = []
    from agentic.eval.policy import PolicyLLM
    inner = PolicyLLM()
    orig = inner.generate

    def spy(**kw):
        seen.extend(m for m in kw["messages"] if "IMPORTANT SYSTEM INSTRUCTION" in m["content"])
        assert "IMPORTANT SYSTEM INSTRUCTION" not in kw["system"]
        return orig(**kw)
    inner.generate = spy
    wire = GroqWire(inner)
    from agentic.eval.run_offline import evaluate_run
    from agentic.eval.provider_eval import stage2_cases
    case = [c for c in stage2_cases() if c["variant"] == "injection"][0]
    _, out = evaluate_run(case["company"], "injection", llm=q.GroqClient(model=MODEL, client=wire))
    assert seen and all(m["role"] == "tool" and m["content"].startswith("<untrusted_web_content") for m in seen)
    assert wire.violations == [] and "attacker@evil.example" not in json.dumps(out["report"])


# ==================================================================================== pacing
class Clock:
    def __init__(self):
        self.t = 1000.0
        self.slept = []

    def now(self):
        return self.t

    def sleep(self, s):
        self.slept.append(s)
        self.t += s


class Probe:
    name, model = "groq", MODEL

    def __init__(self, tokens=(100, 50)):
        self.calls, self.tokens = 0, tokens

    def generate(self, **kw):
        self.calls += 1
        return LLMResponse(text="ok", usage=Usage(input_tokens=self.tokens[0], output_tokens=self.tokens[1]))


def paced(inner, quota=QUOTA, log=None, clock=None, **kw):
    c = clock or Clock()
    return PacedLLM(inner, quota, log, clock=c.now, sleep=c.sleep, **kw), c


KW = dict(system="s" * 400, messages=[{"role": "user", "content": "x" * 400}], tools=[], max_output_tokens=300)


def test_requests_that_would_exceed_the_per_minute_token_limit_wait_instead_of_failing():
    llm, clock = paced(Probe(tokens=(4500, 100)), ProviderQuota(30, 8000, 1000, 200_000))
    llm.generate(**KW)                          # uses ~4.6k tokens
    llm.generate(**dict(KW, max_output_tokens=2000))   # 4.6k + ~2.2k estimate passes 80% of 8k (6.4k) -> must wait
    assert llm.inner.calls == 2 and clock.slept and sum(clock.slept) >= 59 and llm.waited_s >= 59


def test_a_single_request_larger_than_the_minute_limit_is_refused_locally_and_never_sent():
    probe = Probe()
    llm, _ = paced(probe)
    big = dict(KW, messages=[{"role": "user", "content": "x" * 40_000}])           # ~10k tokens
    with pytest.raises(LLMError, match="exceeds the 8000 tokens/minute"):
        llm.generate(**big)
    assert probe.calls == 0 and llm.local_refusals == 1


def test_request_per_minute_limit_is_paced():
    llm, clock = paced(Probe(tokens=(10, 5)), ProviderQuota(rpm=5, tpm=1_000_000, rpd=1000, tpd=10**9))
    for _ in range(5):
        llm.generate(**KW)
    assert sum(clock.slept) >= 59                      # headroom 0.8 -> only 4 requests per rolling minute


def test_daily_budgets_are_rolling_persistent_and_checked_before_sending(tmp_path):
    store = Store(str(tmp_path / "u.sqlite"))
    small = ProviderQuota(rpm=30, tpm=8000, rpd=5, tpd=200_000)                      # planning cap = 4 requests
    probe = Probe(tokens=(50, 10))
    llm, clock = paced(probe, small, store)
    for _ in range(4):
        llm.generate(**KW)
    with pytest.raises(LLMError) as err:
        llm.generate(**KW)
    assert err.value.kind == "quota" and "daily request budget" in str(err.value) and probe.calls == 4
    other, c2 = paced(Probe(), small, Store(str(tmp_path / "u.sqlite")), clock=clock)          # a new process, same file
    with pytest.raises(LLMError, match="daily request budget"):
        other.generate(**KW)
    clock.t += 86400 + 5                                                             # a day later it is allowed again
    other.generate(**KW)


def test_daily_token_budget_counts_actual_usage_and_the_next_estimate(tmp_path):
    store = Store(str(tmp_path / "t.sqlite"))
    q2 = ProviderQuota(rpm=30, tpm=8000, rpd=1000, tpd=3000)
    llm, _ = paced(Probe(tokens=(1900, 100)), q2, store)
    llm.generate(**KW)                                                               # ~2000 actual tokens
    with pytest.raises(LLMError) as err:
        llm.generate(**KW)                                                           # +~600 estimate > 80% of 3000
    assert err.value.kind == "quota" and "daily token budget" in str(err.value)


def test_waiting_longer_than_the_cap_is_a_local_rate_limit_not_a_hang():
    llm, _ = paced(Probe(tokens=(7000, 100)), max_wait_s=10.0)
    llm.generate(**KW)
    with pytest.raises(LLMError) as err:
        llm.generate(**KW)
    assert err.value.kind == "rate_limit" and "local pacing" in str(err.value)


def test_quota_parsing_requires_all_four_operator_supplied_values():
    assert ProviderQuota.parse("rpm=30, tpm=8_000, rpd=1000, tpd=200000").tpm == 8000
    for bad in ("rpm=30,tpm=8000", "rpm=0,tpm=1,rpd=1,tpd=1", ""):
        with pytest.raises(ValueError):
            ProviderQuota.parse(bad)
    assert estimate_request_tokens("s" * 400, [], [], 100) == 100 + 100


# ================================================================================== capacity model
def test_capacity_arithmetic():
    m = {"requests": 22, "prompt_tokens": 59_100, "largest_request": 6000}
    c = cap.capacity(m, 150)
    assert c["tokens_per_run"] == 59_100 + 22 * 150
    assert c["runs_per_day_by_tokens"] == round(200_000 * 0.8 / (59_100 + 3300), 2) == 2.56
    assert c["runs_per_day_by_requests"] == round(1000 * 0.8 / 22, 1)
    assert c["min_minutes_per_run_by_tpm"] == round(62_400 / 6400, 1) and c["fits_tpm_limit"] is True
    assert cap.capacity({**m, "largest_request": 7200}, 150)["fits_tpm_limit"] is False        # 7200 + 1024 reserve > 8000
    assert cap.gemini3_runs_per_day(22) == 0.73


def test_measured_capacity_shows_default_page_size_breaks_the_8k_tpm_limit_and_a_cap_fixes_it():
    a = cap.measure("A. current defaults", None)
    b = cap.measure("B. + tool results capped at 3,000 chars", None)
    assert a["identical"] and b["identical"] and a["violations"] == b["violations"] == 0
    assert not cap.capacity(a, 150)["fits_tpm_limit"] and cap.capacity(b, 150)["fits_tpm_limit"]
    assert b["prompt_tokens"] < a["prompt_tokens"] and b["largest_request"] < 6_900
    assert 2 <= cap.capacity(b, 400)["runs_per_day_by_tokens"] <= 4            # "about three full runs a day"


# ================================================================ proposed Groq smoke test (guards only)
def good_argv():
    return ["--model", MODEL, "--quota", "rpm=30,tpm=8000,rpd=1000,tpd=200000", "--i-have-confirmed-free-plan"]


def test_groq_smoke_refuses_to_run_without_every_approval():
    env = {"AGENTIC_GROQ_SMOKE_APPROVED": "1", "GROQ_API_KEY": "k"}
    assert qs.check_guards(env, good_argv()).tpm == 8000
    with pytest.raises(gs.SmokeGuardError, match="APPROVED"):
        qs.check_guards({"GROQ_API_KEY": "k"}, good_argv())
    with pytest.raises(gs.SmokeGuardError, match="free-plan"):
        qs.check_guards(env, [a for a in good_argv() if a != "--i-have-confirmed-free-plan"])
    with pytest.raises(gs.SmokeGuardError, match="--quota"):
        qs.check_guards(env, ["--model", MODEL, "--i-have-confirmed-free-plan"])
    with pytest.raises(gs.SmokeGuardError, match="invalid"):
        qs.check_guards(env, ["--model", MODEL, "--quota", "rpm=30", "--i-have-confirmed-free-plan"])
    with pytest.raises(gs.SmokeGuardError, match="GROQ_API_KEY"):
        qs.check_guards({"AGENTIC_GROQ_SMOKE_APPROVED": "1"}, good_argv())


def test_groq_smoke_main_never_reads_env_file_or_builds_a_client_without_approval(monkeypatch):
    import dotenv
    monkeypatch.delenv("AGENTIC_GROQ_SMOKE_APPROVED", raising=False)
    monkeypatch.setattr(dotenv, "load_dotenv", lambda *a, **k: (_ for _ in ()).throw(AssertionError(".env read")))
    monkeypatch.setattr(q, "GroqClient", lambda *a, **k: (_ for _ in ()).throw(AssertionError("client built")))
    with pytest.raises(gs.SmokeGuardError):
        qs.main(good_argv() + ["out.json"])
    qs.main(["--dry-run", "--model", MODEL])


def smoke_script():
    f = make_world(gs.COMPANY, "standard")["facts"]
    site = f"https://www.{f['domain']}/"
    line = f"Acme Precision Works is a electronics company headquartered in {f['city']}, India."
    return [tool_call("route", next="company", reason="only eligible"),
            tool_call("web_search", query="Acme Precision Works official website"),
            tool_call("fetch_page", url=site),
            tool_call("submit_company_findings", status="satisfied", identity_website=site, claims=[
                {"field": "headquarters", "value": f["city"], "evidence": [{"url": site, "snippet": line}]}])]


def test_groq_smoke_harness_passes_end_to_end_with_exactly_four_requests_and_logs_pacing(tmp_path):
    wire = GroqWire(FakeLLM(smoke_script()))
    clock = Clock()
    out = qs.run_groq_smoke(q.GroqClient(model=MODEL, client=wire), QUOTA, Store(str(tmp_path / "s.sqlite")),
                            clock=clock.now, sleep=clock.sleep)
    assert out["requests_made"] == 4 and out["all_passed"] and wire.violations == [] and out["model"] == MODEL
    assert "pacing" in out and Store(str(tmp_path / "s.sqlite")).usage_since(f"groq:{MODEL}", 86400, clock.now())[0] == 4


def test_groq_smoke_stops_on_the_first_billing_or_quota_error_without_trying_anything_else():
    for exc, label in ((sdk_error(groq.PermissionDeniedError, 403, "payment method required"), "billing"),
                       (sdk_error(groq.RateLimitError, 429, "limit on requests per day (RPD)"), "quota")):
        wire = GroqWire(FakeLLM(lambda s, m, t, e=exc: (_ for _ in ()).throw(e)))
        out = qs.run_groq_smoke(q.GroqClient(model=MODEL, client=wire), QUOTA)
        assert out["requests_made"] == 1 and out["aborted"].startswith(label) and out["stopped_on_first_error"]


def test_slow_small_quota_providers_can_raise_the_run_budgets_by_configuration():
    from agentic.config import Limits, load_settings
    default = load_settings({}).limits
    tuned = load_settings({"AGENTIC_WALL_CLOCK_S": "1200", "AGENTIC_MAX_TOKENS": "80000",
                           "AGENTIC_TOOL_RESULT_CHARS": "3000"}).limits
    assert (default.max_wall_clock_s, default.max_total_tokens_run) == (180.0, 60_000)
    assert (tuned.max_wall_clock_s, tuned.max_total_tokens_run, tuned.tool_result_chars_for_llm) == (1200.0, 80_000, 3000)


def test_reasoning_tokens_finish_reason_and_provider_timing_are_captured_from_real_sdk_types():
    from groq.types.chat import ChatCompletion
    comp = ChatCompletion.model_validate({
        "id": "x", "object": "chat.completion", "created": 1, "model": MODEL,
        "choices": [{"index": 0, "finish_reason": "tool_calls", "message": {"role": "assistant", "content": None, "tool_calls": [
            {"id": "c1", "type": "function", "function": {"name": "route", "arguments": "{}"}}]}}],
        "usage": {"prompt_tokens": 500, "completion_tokens": 90, "total_tokens": 590, "total_time": 0.42, "queue_time": 0.01,
                  "completion_tokens_details": {"reasoning_tokens": 61}}})
    r = q.parse_response(comp)
    assert (r.usage.input_tokens, r.usage.output_tokens, r.usage.reasoning_tokens) == (500, 90, 61)
    assert r.provider_data["groq"]["finish_reason"] == "tool_calls" and r.provider_data["groq"]["provider_total_time_s"] == 0.42
    llm = gs.CountingLLM(FakeLLM([r]))
    llm.generate(system="s", messages=[], tools=[], max_output_tokens=10)
    assert llm.log[0]["reasoning_tokens"] == 61 and llm.log[0]["provider_meta"]["finish_reason"] == "tool_calls"
    assert q.parse_response(real_completion(content="x")).usage.reasoning_tokens == 0          # absent -> 0, not an error

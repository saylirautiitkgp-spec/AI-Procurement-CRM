"""Hardening: finite Gemini request timeout (reaches the real SDK request), the offered-tool contract at the executor,
and the Gemini-format forced submission path. Offline only - no network, no keys (dummy key strings, patched transport)."""

import json

import httpx
import pytest

from agentic import service as svc_mod
from agentic.agents import AGENT_SPECS
from agentic.backends.fixtures import FixtureFetcher, FixtureSearch
from agentic.config import Efficiency, Limits, Settings, effective_llm_timeout_s, load_settings
from agentic.eval.wire_emulators import GeminiWire
from agentic.guards.budget import new_budgets
from agentic.guards.executor import GuardedExecutor
from agentic.llm.base import LLMError, RetryingLLM, ToolCallRequest, ToolSpec
from agentic.llm.fake import FakeLLM, tool_call
from agentic.llm.gemini import GeminiClient, map_error
from tests.agentic.support import default_repo, make_runner
from tests.agentic.test_hybrid import HYBRID, HybridLLM, build, llm_events
from tests.agentic.test_prefetch import events, world

PROFILE = "company profile"
ROUTE = ToolSpec(name="route", description="Choose.", parameters={"type": "object", "properties": {
    "next": {"type": "string"}}, "required": ["next"]})
OK_BODY = {"candidates": [{"content": {"role": "model", "parts": [
    {"functionCall": {"name": "route", "args": {"next": "finish"}}}]}}],
    "usageMetadata": {"promptTokenCount": 5, "candidatesTokenCount": 3}}


# ============================================================================================== timeouts
@pytest.mark.parametrize("wall,retries,cap,expected", [
    (180, 2, 60, 60),        # defaults: 3 attempts x 60 s = the 180 s budget
    (180, 0, 60, 60),        # no retries: capped by the per-request limit, not stretched to the budget
    (60, 2, 60, 20),         # tight budget: shrinks so 3 attempts still fit
    (1200, 0, 60, 60),       # Groq-style long budget
    (12, 2, 60, 5),          # never below 5 s
    (180, 2, 30, 30),        # explicit smaller cap wins
])
def test_request_timeout_fits_the_wall_clock_budget_for_every_allowed_attempt(wall, retries, cap, expected):
    lim = Limits(max_wall_clock_s=wall, llm_max_retries=retries, llm_request_timeout_s=cap)
    t = effective_llm_timeout_s(lim)
    assert t == expected
    if wall >= 15:
        assert t * (retries + 1) <= wall + 1e-9


def test_timeout_env_override_and_default_retries_are_unchanged():
    assert Limits().llm_max_retries == 2                                         # retry count not raised
    assert load_settings({}).limits.llm_request_timeout_s == 60.0
    assert load_settings({"AGENTIC_LLM_TIMEOUT_S": "25"}).limits.llm_request_timeout_s == 25.0
    assert load_settings({"AGENTIC_LLM_TIMEOUT_S": "1"}).limits.llm_request_timeout_s == 5.0       # floor
    assert load_settings({"AGENTIC_LLM_TIMEOUT_S": "junk"}).limits.llm_request_timeout_s == 60.0


def test_gemini_client_puts_a_finite_timeout_on_the_sdk_client_in_milliseconds():
    c = GeminiClient(model="gemini-3-flash-preview", api_key="dummy-not-a-key", timeout_s=45)
    api = c._c._api_client
    assert api._http_options.timeout == 45000 and api._http_options.retry_options is None
    assert api._retry.stop.max_attempt_number == 1                               # the SDK itself never retries
    assert GeminiClient(model="gemini-3-flash-preview", api_key="dummy").timeout_s is None


class Capture:
    """Replaces httpx.Client.send: records what the SDK would put on the wire, answers with a canned response."""

    def __init__(self, monkeypatch, body=OK_BODY, error=None):
        self.requests, self.error, self.body = [], error, body
        outer = self

        def send(client, request, **kw):
            outer.requests.append(request)
            if outer.error:
                raise outer.error
            return httpx.Response(200, json=outer.body, request=request)
        monkeypatch.setattr(httpx.Client, "send", send)


def test_the_timeout_reaches_the_actual_http_request(monkeypatch):
    cap = Capture(monkeypatch)
    c = GeminiClient(model="gemini-3-flash-preview", api_key="dummy-not-a-key", timeout_s=45, thinking_level="low")
    resp = c.generate(system="s", messages=[{"role": "user", "content": "hi"}], tools=[ROUTE], max_output_tokens=16)
    assert resp.tool_calls[0].name == "route"
    assert len(cap.requests) == 1
    assert cap.requests[0].extensions["timeout"]["read"] == 45.0                # what httpx will enforce
    assert cap.requests[0].headers["x-server-timeout"] == "45"                  # the service is told the same budget


def test_service_wiring_derives_the_timeout_from_wall_clock_and_retries(monkeypatch, tmp_path):
    monkeypatch.setenv("GEMINI_API_KEY", "dummy-not-a-key")
    w = world()
    f = {"search": lambda: FixtureSearch(w), "fetcher": lambda: FixtureFetcher(w), "repo": default_repo}
    base = {"AGENTIC_MODE": "live", "AGENTIC_ALLOW_LIVE": "1", "AGENTIC_DATA_DIR": str(tmp_path),
            "AGENTIC_QUOTA": "rpm=5,tpm=250000,rpd=20"}

    def timeout_ms(extra):
        svc = svc_mod.build_live_service(env={**base, **extra}, factories=f)
        gemini = svc.runner.deps.llm
        while not hasattr(gemini, "_c"):                 # RetryingLLM -> PacedLLM -> GeminiClient
            gemini = gemini.inner
        return gemini._c._api_client._http_options.timeout, svc.runner.deps.llm.max_retries

    assert timeout_ms({}) == (60000, 2)                                          # defaults: 3 x 60 s = 180 s budget
    assert timeout_ms({"AGENTIC_LLM_RETRIES": "0"}) == (60000, 0)
    assert timeout_ms({"AGENTIC_WALL_CLOCK_S": "60"}) == (20000, 2)             # shrinks to fit the smaller budget
    assert timeout_ms({"AGENTIC_WALL_CLOCK_S": "60", "AGENTIC_LLM_RETRIES": "0"}) == (60000, 0)


def test_a_timeout_is_a_bounded_transient_error_not_an_unbounded_wait_and_adds_no_retries(monkeypatch):
    assert map_error(httpx.ReadTimeout("slow")).kind == "transient"
    assert map_error(httpx.ConnectTimeout("slow")).kind == "transient"
    cap = Capture(monkeypatch, error=httpx.ReadTimeout("read timed out"))
    c = GeminiClient(model="gemini-3-flash-preview", api_key="dummy-not-a-key", timeout_s=5)
    with pytest.raises(LLMError) as e:
        c.generate(system="s", messages=[{"role": "user", "content": "hi"}], tools=[ROUTE], max_output_tokens=16)
    assert e.value.kind == "transient" and len(cap.requests) == 1               # the SDK did not retry
    for retries, sends in ((0, 1), (2, 3)):                                      # RetryingLLM is the only retry layer
        cap.requests.clear()
        wrapped = RetryingLLM(c, max_retries=retries, sleep=lambda s: None)
        with pytest.raises(LLMError):
            wrapped.generate(system="s", messages=[{"role": "user", "content": "hi"}], tools=[ROUTE], max_output_tokens=16)
        assert len(cap.requests) == sends


def test_a_timed_out_company_request_ends_the_agent_partial_without_extra_calls(monkeypatch):
    Capture(monkeypatch, error=httpx.ReadTimeout("read timed out"))
    client = GeminiClient(model="gemini-3-flash-preview", api_key="dummy-not-a-key", timeout_s=5)
    out = build(world(), client, limits=Limits(llm_max_retries=0)).start("Apollo Micro Systems", PROFILE)
    assert out["status"] == "partial" and "agent_not_finished:company:llm_transient" in out["status_reasons"]
    assert out["budgets"]["llm_retries"] == 0


# ====================================================================== offered-tool contract (executor)
def executor_for(**limits):
    r = make_runner(limits=Limits(**limits) if limits else None)
    return r, GuardedExecutor(r.deps.registry, r.deps)


def call(name, **args):
    return ToolCallRequest(id="c1", name=name, args=args)


def run(ex, name, offered, agent="company", budgets=None, **args):
    b = budgets or new_budgets()
    return b, ex.execute(agent=agent, call=call(name, **args), budgets=b, ledger={}, subject={}, offered=offered)


def test_an_undeclared_tool_is_rejected_before_execution_with_a_controlled_error():
    r, ex = executor_for()
    b, res = run(ex, "web_search", frozenset({"submit_company_findings"}), query="Apollo Micro Systems")
    assert not res.ok and res.error["code"] == "tool_not_offered"
    assert "submit_company_findings" in res.error["message"]                     # tells the model what it may call
    assert r.search.calls == [] and b["search_credits"] == 0                     # nothing executed, nothing charged
    assert b["tool_calls"] == 1 and b["agent_tool_calls"] == 1                   # counted, so a loop still hits the limit


def test_an_undeclared_tool_with_bad_arguments_is_still_reported_as_not_offered():
    r, ex = executor_for()
    _, res = run(ex, "web_search", frozenset({"submit_company_findings"}), bogus=1)
    assert res.error["code"] == "tool_not_offered"


def test_offered_tools_still_execute_and_system_issued_calls_are_unaffected():
    r, ex = executor_for()
    _, ok = run(ex, "web_search", frozenset({"web_search", "fetch_page"}), query="Apollo Micro Systems")
    assert ok.ok and len(r.search.calls) == 1
    r2, ex2 = executor_for()
    _, system_call = run(ex2, "web_search", None, query="Apollo Micro Systems")             # prefetch passes no contract
    assert system_call.ok and len(r2.search.calls) == 1


def test_a_legitimate_forced_finish_call_is_allowed_and_an_unoffered_finish_is_not():
    _, ex = executor_for()
    _, res = run(ex, "submit_company_findings", frozenset({"submit_company_findings"}), status="exhausted")
    assert res.ok and res.parsed_args is not None
    _, bad = run(ex, "submit_company_findings", frozenset({"web_search"}), status="exhausted")
    assert not bad.ok and bad.error["code"] == "tool_not_offered"


def test_earlier_checks_and_limits_keep_their_precedence():
    _, ex = executor_for()
    _, res = run(ex, "submit_lead_findings", frozenset({"web_search"}), status="exhausted")      # not this agent's tool
    assert res.error["code"] == "tool_not_permitted"
    _, res = run(ex, "no_such_tool", frozenset({"web_search"}))
    assert res.error["code"] == "unknown_tool"
    r, ex2 = executor_for(max_tool_calls_agent=1)
    b = new_budgets()
    b["agent_tool_calls"] = 1                                                    # wrap-up turn: limit stays authoritative
    _, res = run(ex2, "web_search", frozenset({"submit_company_findings"}), budgets=b, query="Apollo Micro Systems")
    assert res.error["code"] == "agent_limit" and res.stop == "agent_tool_call_limit"


def test_the_graph_audits_an_undeclared_call_and_the_agent_recovers_on_the_next_turn():
    llm = FakeLLM([tool_call("web_search", query="Apollo Micro Systems headquarters"),      # not offered on the packet turn
                   tool_call("submit_company_findings", status="exhausted"),
                   tool_call("submit_company_findings", status="exhausted")])               # after the one allowed follow-up
    r = build(world(), llm)
    out = r.start("Apollo Micro Systems", PROFILE)
    audit = events(out, "undeclared_tool_call")
    assert len(audit) == 1 and audit[0]["data"]["tool"] == "web_search"
    assert audit[0]["data"]["offered"] == ["submit_company_findings"]
    failed = [e for e in events(out, "tool_result", ok=False) if not e["data"].get("prefetch")]
    assert [e["data"]["error"]["code"] for e in failed] == ["tool_not_offered"]
    assert len(r.search.calls) == 1                                              # only the prefetch search ever ran
    assert len(llm.calls) == 3 and out["agent_results"]["company"]["end_reason"] == "submitted"
    assert [m for m in llm.calls[1]["messages"] if m["role"] == "tool"][-1]["content"].count("tool_not_offered") == 1


def test_legitimate_hybrid_runs_raise_no_undeclared_call_events():
    out = build(world(), HybridLLM()).start("Apollo Micro Systems", PROFILE)
    assert not events(out, "undeclared_tool_call") and out["status"] == "completed"


# ===================================================================== Gemini-format forced submission
def test_gemini_wire_request_for_the_packet_turn_declares_only_the_submit_tool_and_forces_it(monkeypatch):
    cap = Capture(monkeypatch, body={"candidates": [{"content": {"role": "model", "parts": [
        {"functionCall": {"name": "submit_company_findings", "args": {"status": "exhausted"}}}]}}],
        "usageMetadata": {"promptTokenCount": 9, "candidatesTokenCount": 4}})
    client = GeminiClient(model="gemini-3-flash-preview", api_key="dummy-not-a-key", timeout_s=30, thinking_level="low")
    spec = [ToolSpec(name="submit_company_findings", description="Submit.", parameters={
        "type": "object", "properties": {"status": {"type": "string"}}, "required": ["status"]})]
    resp = client.generate(system="s", messages=[{"role": "user", "content": "task + packet"}], tools=spec,
                           max_output_tokens=64, force_tool=True)
    body = json.loads(cap.requests[0].content)
    decls = [d["name"] for t in body["tools"] for d in t["functionDeclarations"]]
    assert decls == ["submit_company_findings"]
    assert body["toolConfig"]["functionCallingConfig"]["mode"] == "ANY"          # forced
    assert resp.tool_calls[0].name == "submit_company_findings" and resp.tool_calls[0].args == {"status": "exhausted"}


def test_full_company_run_through_the_gemini_adapter_uses_one_forced_single_tool_request():
    seen = []

    class Decider(HybridLLM):
        def generate(self, **kw):
            seen.append((kw["force_tool"], [t.name for t in kw["tools"]]))
            return super().generate(**kw)

    wire = GeminiWire(Decider())
    client = GeminiClient(model="gemini-3-flash-preview", thinking_level="low", client=wire)
    out = build(world(), client).start("Apollo Micro Systems", PROFILE)
    assert wire.violations == [] and wire.requests == 1
    assert seen == [(True, ["submit_company_findings"])]
    assert out["status"] == "completed" and not events(out, "undeclared_tool_call")
    assert {c["field"] for c in out["claims"]["company"]} >= {"official_website", "headquarters", "business_overview"}

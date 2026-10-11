"""Groq adapter: offline conversion/parsing/error-mapping checks. No network, no key, no `groq` package needed."""

import json
from types import SimpleNamespace as NS

import pytest

from agentic.eval import gemini_smoke as gs
from agentic.llm import groq as q
from agentic.llm.base import LLMError, ToolSpec
from agentic.tools.web import WebSearchArgs


def spec():
    return ToolSpec(name="web_search", description="d", parameters=WebSearchArgs.model_json_schema())


def completion(content=None, calls=(), usage=(10, 5)):
    msg = NS(content=content, tool_calls=[NS(id=i, function=NS(name=n, arguments=a)) for i, n, a in calls] or None)
    return NS(choices=[NS(message=msg)], usage=NS(prompt_tokens=usage[0], completion_tokens=usage[1]))


class FakeChat:
    def __init__(self, response=None, error=None):
        self.response, self.error, self.kwargs = response, error, None
        self.completions = self

    def create(self, **kw):
        self.kwargs = kw
        if self.error:
            raise self.error
        return self.response


def test_messages_and_tools_follow_the_openai_tool_format():
    msgs = [{"role": "user", "content": "task"},
            {"role": "assistant", "content": "", "tool_calls": [{"id": "c1", "name": "web_search", "args": {"query": "acme x"}}]},
            {"role": "tool", "tool_call_id": "c1", "name": "web_search", "content": "result"}]
    out = q.to_messages("sys", msgs)
    assert [m["role"] for m in out] == ["system", "user", "assistant", "tool"]
    call = out[2]["tool_calls"][0]
    assert call["id"] == "c1" and json.loads(call["function"]["arguments"]) == {"query": "acme x"}
    assert out[3]["tool_call_id"] == "c1"
    t = q.to_tools([spec()])[0]
    assert t["type"] == "function" and "$ref" not in json.dumps(t) and "anyOf" not in json.dumps(t)


def test_parse_tool_calls_usage_and_bad_arguments():
    r = q.parse_response(completion(calls=[("a", "web_search", '{"query": "acme x"}'), ("b", "fetch_page", "{not json")],
                                    usage=(120, 33)))
    assert [c.name for c in r.tool_calls] == ["web_search", "fetch_page"] and r.tool_calls[0].args == {"query": "acme x"}
    assert r.tool_calls[1].args == {"__unparseable_arguments__": True}               # rejected later as invalid_arguments
    assert (r.usage.input_tokens, r.usage.output_tokens) == (120, 33)
    assert q.parse_response(completion(content="hi")).text == "hi"


def test_request_uses_required_tool_choice_only_when_forced_and_no_sdk_retries():
    chat = FakeChat(completion(calls=[("a", "web_search", '{"query": "acme x"}')]))
    llm = q.GroqClient(model="openai/gpt-oss-20b", client=NS(chat=chat), temperature=0.2)
    llm.generate(system="s", messages=[{"role": "user", "content": "x"}], tools=[spec()], max_output_tokens=300, force_tool=True)
    assert chat.kwargs["tool_choice"] == "required" and chat.kwargs["max_completion_tokens"] == 300
    llm.generate(system="s", messages=[{"role": "user", "content": "x"}], tools=[spec()], max_output_tokens=300)
    assert chat.kwargs["tool_choice"] == "auto" and chat.kwargs["model"] == "openai/gpt-oss-20b"


class HttpError(Exception):
    def __init__(self, code, msg, headers=None):
        super().__init__(msg)
        self.status_code = code
        self.response = NS(headers=headers or {})


@pytest.mark.parametrize("code,msg,headers,kind,retry", [
    (429, "Rate limit reached for model in organization on tokens per minute (TPM)", {"retry-after": "7"}, "rate_limit", 7.0),
    (429, "Rate limit reached ... on tokens per day (TPD): Limit 200000", {}, "quota", None),
    (429, "Rate limit reached ... on requests per day (RPD)", {}, "quota", None),
    (503, "overloaded", {}, "transient", None),
    (400, "tool_use_failed: failed to call a function", {}, "invalid", None),
    (401, "invalid api key gsk_abcdefghijklmnop1234", {}, "invalid", None),
])
def test_http_errors_map_to_retry_semantics_and_keys_are_scrubbed(code, msg, headers, kind, retry):
    llm = q.GroqClient(model="m", client=NS(chat=FakeChat(error=HttpError(code, msg, headers))))
    with pytest.raises(LLMError) as exc:
        llm.generate(system="s", messages=[{"role": "user", "content": "x"}], tools=[], max_output_tokens=10)
    assert exc.value.kind == kind and exc.value.retry_after == retry and "gsk_abc" not in str(exc.value)


def test_refuses_to_start_without_key(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    with pytest.raises((RuntimeError, ImportError)):
        q.GroqClient(model="m")


def test_same_harness_runs_unchanged_against_a_groq_shaped_client():
    """The provider-neutral smoke harness drives any LLMClient; here a scripted Groq-shaped fake proves the
    adapter plugs into it (4 requests, same checks)."""
    from agentic.eval.synthetic import make_world
    f = make_world(gs.COMPANY, "standard")["facts"]
    site = f"https://www.{f['domain']}/"
    line = f"Acme Precision Works is a electronics company headquartered in {f['city']}, India."
    script = iter([
        completion(calls=[("1", "route", '{"next": "company", "reason": "only eligible"}')]),
        completion(calls=[("2", "web_search", '{"query": "Acme Precision Works official website"}')]),
        completion(calls=[("3", "fetch_page", json.dumps({"url": site}))]),
        completion(calls=[("4", "submit_company_findings", json.dumps({
            "status": "satisfied", "identity_website": site, "claims": [
                {"field": "headquarters", "value": f["city"], "evidence": [{"url": site, "snippet": line}]}]}))]),
    ])
    chat = NS(completions=NS(create=lambda **kw: next(script)))
    out = gs.run_smoke(gs.CountingLLM(q.GroqClient(model="m", client=NS(chat=chat))))
    assert out["requests_made"] == 4 and out["all_passed"], out["checks"]


def test_service_selects_groq_only_with_explicit_model_and_operator_supplied_quota(tmp_path):
    from agentic import service as svc
    from agentic.llm.pacing import PacedLLM
    from tests.agentic.support import make_runner
    r = make_runner()
    base = {"AGENTIC_MODE": "live", "AGENTIC_ALLOW_LIVE": "1", "AGENTIC_LLM_PROVIDER": "groq"}
    fac = {"llm": lambda: r.llm, "search": lambda: r.search, "fetcher": lambda: r.fetcher, "repo": lambda: r.deps.repo}
    quota = "rpm=30,tpm=8000,rpd=1000,tpd=200000"
    built = svc.build_live_service(env={**base, "AGENTIC_QUOTA": quota}, factories=fac, data_dir=str(tmp_path))
    assert built.settings.provider == "groq" and isinstance(built.runner.deps.llm.inner, PacedLLM)
    with pytest.raises(svc.ServiceNotConfigured, match="AGENTIC_QUOTA"):             # no assumed limits
        svc.build_live_service(env=base, factories=fac, data_dir=str(tmp_path))
    with pytest.raises(svc.ServiceNotConfigured, match="invalid"):
        svc.build_live_service(env={**base, "AGENTIC_QUOTA": "rpm=30"}, factories=fac, data_dir=str(tmp_path))
    with pytest.raises(svc.ServiceNotConfigured):                                    # unknown provider, no fallback
        svc.build_live_service(env={**base, "AGENTIC_LLM_PROVIDER": "openai", "AGENTIC_QUOTA": quota}, data_dir=str(tmp_path))
    with pytest.raises(svc.ServiceNotConfigured, match="AGENTIC_MODEL"):             # no assumed Groq model
        svc.build_live_service(env={**base, "AGENTIC_QUOTA": quota}, data_dir=str(tmp_path))

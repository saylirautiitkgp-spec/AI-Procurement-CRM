"""Gemini adapter: offline checks of request construction, response parsing and error mapping.
No network, no API key. Live behaviour is NOT verified by these tests."""

import json
from types import SimpleNamespace

import pytest
from google.genai import types

from agentic.agents.company import CompanyFindings
from agentic.llm import gemini as g
from agentic.llm.base import LLMError, ToolSpec
from agentic.tools.web import WebSearchArgs


def spec(model):
    return ToolSpec(name="t", description="d", parameters=model.model_json_schema())


def test_schemas_are_flattened_for_gemini():
    flat = g.simplify_schema(CompanyFindings.model_json_schema())
    text = json.dumps(flat)
    assert "$ref" not in text and "$defs" not in text and "anyOf" not in text and '"title"' not in text
    assert "identity_website" in flat["properties"] and flat["properties"]["claims"]["type"] == "array"
    s = g.simplify_schema(WebSearchArgs.model_json_schema())
    assert s["properties"]["depth"]["enum"] == ["basic", "fast", "advanced"]


def test_messages_convert_to_alternating_roles_with_grouped_function_responses():
    msgs = [{"role": "user", "content": "task"},
            {"role": "assistant", "content": "", "tool_calls": [{"id": "1", "name": "web_search", "args": {"query": "q q q"}},
                                                               {"id": "2", "name": "fetch_page", "args": {"url": "https://a.example/x"}}]},
            {"role": "tool", "tool_call_id": "1", "name": "web_search", "content": "r1"},
            {"role": "tool", "tool_call_id": "2", "name": "fetch_page", "content": "r2"},
            {"role": "user", "content": "go on"}]
    contents = g.to_contents(msgs)
    assert [c.role for c in contents] == ["user", "model", "user", "user"]
    assert len(contents[1].parts) == 2 and contents[1].parts[0].function_call.name == "web_search"
    assert [p.function_response.name for p in contents[2].parts] == ["web_search", "fetch_page"]


def test_thought_signatures_round_trip_through_json_state():
    part = types.Part(function_call=types.FunctionCall(name="route", args={"next": "company"}),
                      thought_signature=b"\x01\x02sig")
    resp = SimpleNamespace(candidates=[SimpleNamespace(content=types.Content(role="model", parts=[part]))],
                           usage_metadata=SimpleNamespace(prompt_token_count=10, candidates_token_count=3,
                                                          thoughts_token_count=7))
    parsed = g.parse_response(resp)
    assert parsed.tool_calls[0].name == "route" and parsed.usage.output_tokens == 10     # visible + thinking tokens
    stored = json.loads(json.dumps(parsed.provider_data))                                # survives checkpoint JSON
    msg = {"role": "assistant", "content": "", "tool_calls": [], "provider_data": stored}
    replay = g.to_contents([msg])[0]
    assert replay.parts[0].thought_signature == b"\x01\x02sig" and replay.parts[0].function_call.name == "route"


class FakeModels:
    def __init__(self, response=None, error=None):
        self.response, self.error, self.seen = response, error, None

    def generate_content(self, model, contents, config):
        self.seen = {"model": model, "contents": contents, "config": config}
        if self.error:
            raise self.error
        return self.response


def client_with(models):
    return SimpleNamespace(models=models)


def test_request_disables_sdk_auto_calling_and_forces_a_tool_only_when_asked():
    resp = SimpleNamespace(candidates=[SimpleNamespace(content=types.Content(role="model", parts=[types.Part(text="hi")]))],
                           usage_metadata=None)
    models = FakeModels(resp)
    llm = g.GeminiClient(model="gemini-2.5-flash", client=client_with(models), thinking_budget=256)
    out = llm.generate(system="sys", messages=[{"role": "user", "content": "x"}], tools=[spec(WebSearchArgs)],
                       max_output_tokens=512, force_tool=True)
    cfg = models.seen["config"]
    assert out.text == "hi" and models.seen["model"] == "gemini-2.5-flash"
    assert cfg.automatic_function_calling.disable is True and cfg.max_output_tokens == 512
    assert cfg.tool_config.function_calling_config.mode == types.FunctionCallingConfigMode.ANY
    assert cfg.thinking_config.thinking_budget == 256
    llm.generate(system="sys", messages=[{"role": "user", "content": "x"}], tools=[spec(WebSearchArgs)],
                 max_output_tokens=512)
    assert models.seen["config"].tool_config.function_calling_config.mode == types.FunctionCallingConfigMode.AUTO


@pytest.mark.parametrize("code,msg,kind", [
    (429, "RESOURCE_EXHAUSTED: quota exceeded for metric GenerateRequestsPerDayPerProjectPerModel-FreeTier", "quota"),
    (429, "RESOURCE_EXHAUSTED: rate limit per minute", "rate_limit"),
    (503, "model overloaded", "transient"),
    (400, "bad schema", "invalid"),
    (418, "teapot", "other"),
])
def test_provider_errors_map_to_retry_semantics(code, msg, kind):
    class E(Exception):
        pass
    e = E(msg)
    e.code = code
    models = FakeModels(error=e)
    with pytest.raises(LLMError) as exc:
        g.GeminiClient(client=client_with(models)).generate(system="s", messages=[{"role": "user", "content": "x"}],
                                                           tools=[], max_output_tokens=10)
    assert exc.value.kind == kind


def test_api_keys_are_scrubbed_from_error_messages(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "AIzaSyDUMMYKEYDUMMYKEYDUMMYKEY12345")
    e = RuntimeError("request failed for key AIzaSyDUMMYKEYDUMMYKEYDUMMYKEY12345")
    e.code = 500
    with pytest.raises(LLMError) as exc:
        g.GeminiClient(client=client_with(FakeModels(error=e))).generate(
            system="s", messages=[{"role": "user", "content": "x"}], tools=[], max_output_tokens=10)
    assert "AIzaSy" not in str(exc.value)


def test_client_refuses_to_start_without_a_key(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(RuntimeError):
        g.GeminiClient()


def test_every_registered_tool_schema_is_provider_friendly():
    from agentic.agents import AGENT_SPECS
    from agentic.runtime import build_registry
    reg = build_registry(AGENT_SPECS)
    for agent in AGENT_SPECS:
        for t in reg.specs_for(agent):
            flat = json.dumps(g.simplify_schema(t.parameters))
            assert "$ref" not in flat and "anyOf" not in flat and '"title"' not in flat, (agent, t.name)
            assert len(t.description) <= 1000


def test_gemini3_defaults_to_temperature_1_and_supports_thinking_level_but_not_both_knobs():
    resp = SimpleNamespace(candidates=[SimpleNamespace(content=types.Content(role="model", parts=[types.Part(text="hi")]))],
                           usage_metadata=None)
    models = FakeModels(resp)
    g.GeminiClient(model="gemini-3-flash-preview", client=client_with(models), thinking_level="low").generate(
        system="s", messages=[{"role": "user", "content": "x"}], tools=[], max_output_tokens=100)
    cfg = models.seen["config"]
    assert cfg.temperature == 1.0 and cfg.thinking_config.thinking_level == types.ThinkingLevel.LOW
    assert cfg.thinking_config.thinking_budget is None
    assert g.GeminiClient(model="gemini-2.5-flash", client=client_with(models)).temperature == 0.2
    with pytest.raises(ValueError):
        g.GeminiClient(model="gemini-3-flash-preview", client=client_with(models), thinking_level="low", thinking_budget=100)

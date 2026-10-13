"""Which provider/model pairs the runtime can drive, and an OFFLINE tool-calling compatibility check.

The runtime needs three things from a model: (1) function/tool calling, (2) a way to REQUIRE a tool (the packet turn
forces `submit_*_findings`), (3) tool results fed back on a following turn. `verify_tool_calling` replays exactly those
requests through the real adapter against a wire-level emulator of the provider SDK and reports every violation of the
provider's request format. It proves the REQUEST/RESPONSE SHAPES are valid. It does NOT prove a model chooses tools well -
that needs live evidence, which the registry records separately (`verified_live`).

Free-tier models only; there is no automatic fallback between providers or models.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelCapability:
    provider: str
    model: str
    tool_calling: bool
    forced_tool_choice: bool
    parallel_tool_calls: bool
    verified_live: bool            # a real request with tools succeeded in this project (see docs/PROVIDER_DECISION.md)
    note: str = ""
    requirements: tuple = ()       # what the runtime must do for THIS model's tool calling to work


_GEMINI_REQ = ("SDK automatic function calling is disabled (we run the tool loop ourselves)",
               "a single required tool is sent as function_calling_config mode ANY",
               "AGENTIC_QUOTA is mandatory (free tier: requests/day)")
_GEMINI3_REQ = _GEMINI_REQ + ("function-call parts are replayed with their thought_signature on the next turn",
                              "temperature stays at 1.0 and thinking_level (not thinking_budget) is used")
_GPT_OSS_REQ = ("no parallel tool calls; service_tier / parallel_tool_calls / response_format / stream are never sent",
                "structured output is a validated function call (response_format cannot be combined with tools)",
                "AGENTIC_QUOTA is mandatory; with the 8,000 TPM free tier use AGENTIC_TOOL_RESULT_CHARS=3000")
_QWEN_REQ = ("tool_choice=required is documented but not exercised here",
             "thinking text arrives in message content and is ignored by the parser (it still costs tokens)",
             "AGENTIC_QUOTA is mandatory and must come from YOUR console; no reasoning parameters are sent")

_CAPS = (
    ModelCapability("gemini", "gemini-3-flash-preview", True, True, True, True,
                    "Integration smoke passed 2026-10-10; free tier 20 requests/day. Google may use free-tier content.",
                    _GEMINI3_REQ),
    ModelCapability("gemini", "gemini-2.5-flash", True, True, True, False, "Documented function calling; not run here.",
                    _GEMINI_REQ),
    ModelCapability("groq", "openai/gpt-oss-120b", True, True, False, True,
                    "Integration passed 2026-10-11; one live run produced a call to an undeclared tool (now rejected "
                    "by the executor). 8,000 TPM free tier.", _GPT_OSS_REQ),
    ModelCapability("groq", "openai/gpt-oss-20b", True, True, False, False, "Same family; not run here.", _GPT_OSS_REQ),
    ModelCapability("groq", "qwen/qwen3-32b", True, True, True, False,
                    "Qwen3 on Groq documents tool use; not run here. Thinking text is ignored by the parser.",
                    _QWEN_REQ),
)
REGISTRY = {(c.provider, c.model): c for c in _CAPS}


def capability(provider: str, model: str) -> ModelCapability | None:
    return REGISTRY.get((provider, model))


def registry_view() -> list[dict]:
    return [{"provider": c.provider, "model": c.model, "tool_calling": c.tool_calling,
             "forced_tool_choice": c.forced_tool_choice, "parallel_tool_calls": c.parallel_tool_calls,
             "verified_live": c.verified_live, "note": c.note, "requirements": list(c.requirements)} for c in _CAPS]


def _adapter(provider: str, model: str, decider):
    from agentic.eval.wire_emulators import GeminiWire, GroqWire
    if provider == "gemini":
        from agentic.llm.gemini import GeminiClient
        wire = GeminiWire(decider)
        kw = {"thinking_level": "low"} if model.startswith("gemini-3") else {}
        return GeminiClient(model=model, client=wire, **kw), wire
    if provider == "groq":
        from agentic.llm.groq import GroqClient
        wire = GroqWire(decider)
        return GroqClient(model=model, client=wire), wire
    raise ValueError(f"unknown provider {provider!r} (gemini, groq)")


def _signature_replayed() -> bool:
    """Gemini 3 needs the model's thought_signature sent back with its function call. Exercise our adapter's
    parse -> store -> replay path with SDK types (no network)."""
    from google.genai import types

    from agentic.llm.gemini import parse_response, to_contents
    part = types.Part(function_call=types.FunctionCall(name="echo", args={"text": "ok"}), thought_signature=b"sig-123")
    resp = types.GenerateContentResponse(candidates=[types.Candidate(content=types.Content(role="model", parts=[part]))])
    parsed = parse_response(resp)
    messages = [{"role": "user", "content": "task"},
                {"role": "assistant", "content": "", "tool_calls": [c.model_dump() for c in parsed.tool_calls],
                 "provider_data": parsed.provider_data},
                {"role": "tool", "tool_call_id": parsed.tool_calls[0].id, "name": "echo", "content": "{}"}]
    return any(getattr(p, "thought_signature", None) == b"sig-123"
               for c in to_contents(messages) for p in (c.parts or []))


def selected_model_report(llm, settings, env) -> dict:
    """What is configured for the selected model, and which of its requirements are met. No secrets, no network."""
    node, quota = llm, None
    while node is not None:
        quota = getattr(node, "quota", None) or quota
        node = getattr(node, "inner", None)
    cap = capability(llm.name, llm.model)
    from agentic.config import effective_llm_timeout_s
    lim = settings.limits
    return {"provider": llm.name, "model": llm.model,
            "selection": "explicit" if env.get("AGENTIC_LLM_PROVIDER") else "default (gemini)",
            "fallback": "none - a failed provider is never replaced automatically",
            "requirements": list(cap.requirements) if cap else [],
            "budgets": {"quota_guard_active": quota is not None,
                        "quota": None if quota is None else {"rpm": quota.rpm, "tpm": quota.tpm, "rpd": quota.rpd},
                        "max_llm_calls_run": lim.max_llm_calls_run, "max_total_tokens_run": lim.max_total_tokens_run,
                        "max_search_credits_company": lim.max_search_credits_company},
            "retries": {"llm_max_retries": lim.llm_max_retries,
                        "request_timeout_s": round(effective_llm_timeout_s(lim), 1)}}


def verify_tool_calling(provider: str, model: str) -> dict:
    """Offline wire-level check of the three request shapes the runtime uses. No network, no keys."""
    from agentic.llm.base import LLMResponse, ToolCallRequest, ToolSpec, Usage

    class Decider:
        def __init__(self):
            self.seen = []

        def generate(self, *, system, messages, tools, max_output_tokens, force_tool=False):
            self.seen.append({"tools": [t.name for t in tools], "force": force_tool, "roles": [m["role"] for m in messages]})
            return LLMResponse(tool_calls=[ToolCallRequest(id="c1", name=tools[0].name, args={"text": "ok"})],
                               usage=Usage(requests=1, input_tokens=10, output_tokens=5))

    echo = ToolSpec(name="echo", description="Echo text.", parameters={
        "type": "object", "properties": {"text": {"type": "string", "description": "Text."}}, "required": ["text"]})
    submit = ToolSpec(name="submit_findings", description="Submit.", parameters={
        "type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]})
    decider = Decider()
    client, wire = _adapter(provider, model, decider)
    checks, problems = [], []

    def run(label, **kw):
        try:
            r = client.generate(system="system", max_output_tokens=64, **kw)
            ok = bool(r.tool_calls) and r.tool_calls[0].args == {"text": "ok"}
        except Exception as exc:  # noqa: BLE001 - report, never raise
            ok, r = False, None
            problems.append(f"{label}: {type(exc).__name__}: {exc}"[:200])
        checks.append({"check": label, "ok": ok})
        return r

    first = run("auto_tool_choice_with_two_tools", messages=[{"role": "user", "content": "task"}], tools=[echo, submit])
    run("required_single_tool", messages=[{"role": "user", "content": "task"}], tools=[submit], force_tool=True)
    if first and first.tool_calls:
        run("tool_result_round_trip", tools=[echo, submit], messages=[
            {"role": "user", "content": "task"},
            {"role": "assistant", "content": "", "tool_calls": [c.model_dump() for c in first.tool_calls],
             "provider_data": first.provider_data},
            {"role": "tool", "tool_call_id": first.tool_calls[0].id, "name": "echo", "content": '{"result": "ok"}'}])
    if provider == "gemini" and model.startswith("gemini-3"):
        checks.append({"check": "thought_signature_replay", "ok": _signature_replayed()})
    forced_seen = [s for s in decider.seen if s["tools"] == ["submit_findings"]]
    if not (forced_seen and forced_seen[0]["force"]):
        problems.append("a single required tool was not sent as a forced tool choice")
    problems += wire.violations
    cap = capability(provider, model)
    return {"provider": provider, "model": model, "listed": cap is not None,
            "ok": all(c["ok"] for c in checks) and not problems and len(checks) >= 3, "checks": checks,
            "violations": problems, "verified_live": bool(cap and cap.verified_live),
            "note": "Offline wire-format check only; it does not show how well the model chooses tools."}

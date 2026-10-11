"""Groq adapter (OpenAI-compatible chat-completions API with tools).

STATUS: built and tested OFFLINE (mocked clients and real `groq` SDK types). Nothing has been sent to Groq. Live
behaviour (tool-choice quality, whether `tool_choice="required"` is honoured by gpt-oss, `tool_use_failed` rates,
how TPM counts reserved output tokens) is UNVERIFIED. No automatic fallback between providers exists.

Zero-cost rules baked in:
* `service_tier` and `parallel_tool_calls` are NEVER sent (a non-default tier could be billed; gpt-oss has no
  parallel tool use).
* The SDK's own retries are disabled; bounded retries live in RetryingLLM, and the live test uses none.
* Reasoning text is not requested back (`include_reasoning=False`) and effort defaults to "low" for gpt-oss to keep
  completion tokens (which count against TPM/TPD) small.
* Documentation facts used: structured outputs (response_format) cannot be combined with tool use on Groq, so final
  structured output is a *function call* validated by our own pydantic schemas, not response_format.
"""

from __future__ import annotations

import json
import os
import uuid
from typing import Any

from agentic.llm.base import LLMError, LLMResponse, ToolCallRequest, ToolSpec, Usage
from agentic.llm.gemini import simplify_schema
from agentic.security import redact

# The only Groq models the user approved for evaluation (explicit IDs; nothing is assumed beyond this list).
APPROVED_MODELS = ("openai/gpt-oss-120b", "openai/gpt-oss-20b")
_BILLING_WORDS = ("billing", "payment", "credit card", "upgrade", "dev tier", "developer tier", "spend limit",
                  "insufficient funds", "subscription")
_FORBIDDEN_PARAMS = ("service_tier", "parallel_tool_calls", "response_format", "stream")


def to_tools(tools: list[ToolSpec]) -> list[dict]:
    return [{"type": "function", "function": {"name": t.name, "description": t.description[:1000],
                                              "parameters": simplify_schema(t.parameters)}} for t in tools]


def to_messages(system: str, messages: list[dict]) -> list[dict]:
    out: list[dict] = [{"role": "system", "content": system}]
    for m in messages:
        if m["role"] == "user":
            out.append({"role": "user", "content": m["content"]})
        elif m["role"] == "assistant":
            msg: dict = {"role": "assistant", "content": m.get("content") or None}
            calls = m.get("tool_calls") or []
            if calls:
                msg["tool_calls"] = [{"id": c["id"], "type": "function",
                                      "function": {"name": c["name"], "arguments": json.dumps(c["args"])}}
                                     for c in calls]
            out.append(msg)
        elif m["role"] == "tool":
            out.append({"role": "tool", "tool_call_id": m["tool_call_id"], "content": m["content"]})
    return out


def parse_response(resp: Any) -> LLMResponse:
    choice = resp.choices[0].message if getattr(resp, "choices", None) else None
    calls, text = [], ""
    if choice is not None:
        text = choice.content or ""                       # `choice.reasoning` is deliberately ignored
        for tc in (getattr(choice, "tool_calls", None) or []):
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {"__unparseable_arguments__": True}      # surfaces as invalid_arguments in the executor
            calls.append(ToolCallRequest(id=tc.id or f"q{uuid.uuid4().hex[:10]}", name=tc.function.name,
                                         args=args if isinstance(args, dict) else {}))
    u = getattr(resp, "usage", None)
    return LLMResponse(text=text, tool_calls=calls,
                       usage=Usage(requests=1, input_tokens=int(getattr(u, "prompt_tokens", 0) or 0),
                                   output_tokens=int(getattr(u, "completion_tokens", 0) or 0)))


def map_error(exc: Exception) -> LLMError:
    code = getattr(exc, "status_code", None) or getattr(exc, "code", None)
    msg = redact(str(exc))[:300]
    low = msg.lower()
    headers = getattr(getattr(exc, "response", None), "headers", None) or {}
    retry_after = None
    try:
        retry_after = float(headers.get("retry-after")) if headers.get("retry-after") else None
    except (TypeError, ValueError):
        pass
    if type(exc).__name__ in ("APIConnectionError", "APITimeoutError", "ConnectError", "ReadTimeout", "TimeoutException"):
        return LLMError("transient", msg)
    if code == 402 or (code in (400, 401, 403, 429) and any(w in low for w in _BILLING_WORDS)):
        return LLMError("billing", msg)                      # never retried; the run/test stops
    if code == 429:
        if "per day" in low or "(tpd)" in low or "(rpd)" in low:
            return LLMError("quota", msg)
        return LLMError("rate_limit", msg, retry_after=retry_after)
    if code == 413:
        return LLMError("invalid", "request too large for the per-minute token limit: " + msg)
    if isinstance(code, int) and code >= 500:
        return LLMError("transient", msg)
    if code in (400, 401, 403, 404, 422):
        return LLMError("invalid", msg)                      # includes tool_use_failed (400)
    return LLMError("other", msg)


class GroqClient:
    name = "groq"

    def __init__(self, model: str, api_key: str | None = None, client: Any = None, temperature: float = 0.2,
                 reasoning_effort: str | None = "low"):
        self.model = model
        self.temperature = temperature
        self.reasoning_effort = reasoning_effort
        if client is None:
            from groq import Groq
            key = api_key or os.environ.get("GROQ_API_KEY")
            if not key:
                raise RuntimeError("GROQ_API_KEY is not configured")
            client = Groq(api_key=key, max_retries=0)         # retries are handled (and bounded) elsewhere
        self._c = client

    def build_request(self, system: str, messages: list[dict], tools: list[ToolSpec], max_output_tokens: int,
                      force_tool: bool) -> dict:
        kwargs: dict = dict(model=self.model, messages=to_messages(system, messages), temperature=self.temperature,
                            max_completion_tokens=max_output_tokens)
        if self.model.startswith("openai/gpt-oss"):
            if self.reasoning_effort:
                kwargs["reasoning_effort"] = self.reasoning_effort
            kwargs["include_reasoning"] = False
        if tools:
            kwargs["tools"] = to_tools(tools)
            kwargs["tool_choice"] = "required" if force_tool else "auto"
        assert not any(p in kwargs for p in _FORBIDDEN_PARAMS)
        return kwargs

    def generate(self, *, system: str, messages: list[dict], tools: list[ToolSpec], max_output_tokens: int,
                 force_tool: bool = False) -> LLMResponse:
        try:
            resp = self._c.chat.completions.create(**self.build_request(system, messages, tools, max_output_tokens,
                                                                        force_tool))
        except Exception as exc:  # noqa: BLE001
            raise map_error(exc) from None
        return parse_response(resp)

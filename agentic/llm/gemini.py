"""Gemini adapter (google-genai SDK). Native function calling with the SDK's automatic calling
DISABLED, so every tool call goes through our guarded executor.

STATUS: conversion and parsing are unit-tested offline against SDK types. Live behaviour (that the
configured model accepts these schemas, tool-choice quality, thinking-token use under
max_output_tokens) is UNVERIFIED until the approved free-tier smoke test is run.
"""

from __future__ import annotations

import base64
import os
import uuid
from typing import Any

from agentic.llm.base import LLMError, LLMResponse, ToolCallRequest, ToolSpec, Usage
from agentic.security import redact


def simplify_schema(schema: dict) -> dict:
    """Inline $ref/$defs, drop titles/defaults, turn Optional (anyOf with null) into the plain type."""
    defs = schema.get("$defs", {})

    def walk(node):
        if isinstance(node, list):
            return [walk(n) for n in node]
        if not isinstance(node, dict):
            return node
        if "$ref" in node:
            return walk(defs[node["$ref"].split("/")[-1]])
        if "anyOf" in node:
            options = [o for o in node["anyOf"] if o.get("type") != "null"]
            if len(options) == 1:
                merged = {k: v for k, v in node.items() if k != "anyOf"}
                return walk({**options[0], **merged})
            node = {**node, "anyOf": options}
        out = {}
        for k, v in node.items():
            if k in ("title", "default", "$defs", "examples", "additionalProperties"):
                continue
            out[k] = walk(v)
        return out

    return walk({k: v for k, v in schema.items() if k != "$defs"})


def to_tools(tools: list[ToolSpec]):
    from google.genai import types
    decls = [types.FunctionDeclaration(name=t.name, description=t.description[:1000],
                                       parameters_json_schema=simplify_schema(t.parameters)) for t in tools]
    return [types.Tool(function_declarations=decls)] if decls else None


def to_contents(messages: list[dict]):
    from google.genai import types
    contents: list = []
    pending_responses: list = []

    def flush():
        if pending_responses:
            contents.append(types.Content(role="user", parts=list(pending_responses)))
            pending_responses.clear()

    for m in messages:
        role = m["role"]
        if role == "tool":
            pending_responses.append(types.Part.from_function_response(
                name=m.get("name", "tool"), response={"result": m["content"]}))
            continue
        flush()
        if role == "user":
            contents.append(types.Content(role="user", parts=[types.Part(text=m["content"])]))
        elif role == "assistant":
            raw = (m.get("provider_data") or {}).get("gemini_parts")
            if raw:                                    # replay exactly (keeps thought signatures)
                contents.append(types.Content.model_validate({"role": "model", "parts": _decode(raw)}))
                continue
            parts = []
            if m.get("content"):
                parts.append(types.Part(text=m["content"]))
            for c in m.get("tool_calls", []):
                parts.append(types.Part.from_function_call(name=c["name"], args=c["args"]))
            contents.append(types.Content(role="model", parts=parts or [types.Part(text=" ")]))
    flush()
    return contents


def _encode(parts: list[dict]) -> list[dict]:
    out = []
    for p in parts:
        p = dict(p)
        if isinstance(p.get("thought_signature"), (bytes, bytearray)):
            p["thought_signature"] = {"__b64__": base64.b64encode(p["thought_signature"]).decode()}
        out.append(p)
    return out


def _decode(parts: list[dict]) -> list[dict]:
    out = []
    for p in parts:
        p = dict(p)
        sig = p.get("thought_signature")
        if isinstance(sig, dict) and "__b64__" in sig:
            p["thought_signature"] = base64.b64decode(sig["__b64__"])
        out.append(p)
    return out


def parse_response(resp: Any) -> LLMResponse:
    cands = getattr(resp, "candidates", None) or []
    text_parts, calls, raw_parts = [], [], []
    if cands and cands[0].content and cands[0].content.parts:
        for part in cands[0].content.parts:
            raw_parts.append(part.model_dump(exclude_none=True))
            if getattr(part, "function_call", None):
                fc = part.function_call
                calls.append(ToolCallRequest(id=fc.id or f"g{uuid.uuid4().hex[:10]}", name=fc.name,
                                             args=dict(fc.args or {})))
            elif getattr(part, "text", None) and not getattr(part, "thought", False):
                text_parts.append(part.text)
    um = getattr(resp, "usage_metadata", None)
    usage = Usage(requests=1,
                  input_tokens=int(getattr(um, "prompt_token_count", 0) or 0),
                  output_tokens=int((getattr(um, "candidates_token_count", 0) or 0)
                                    + (getattr(um, "thoughts_token_count", 0) or 0)))
    return LLMResponse(text="".join(text_parts), tool_calls=calls, usage=usage,
                       provider_data={"gemini_parts": _encode(raw_parts)} if raw_parts else None)


def map_error(exc: Exception) -> LLMError:
    code = getattr(exc, "code", None)
    msg = redact(str(exc))[:300]
    low = msg.lower()
    if code == 429 or "resource_exhausted" in low:
        if "perday" in low.replace(" ", "").replace("_", "") or "per day" in low or "daily" in low:
            return LLMError("quota", msg)
        return LLMError("rate_limit", msg)
    if isinstance(code, int) and code >= 500:
        return LLMError("transient", msg)
    if code in (400, 401, 403, 404):
        return LLMError("invalid", msg)
    return LLMError("other", msg)


class GeminiClient:
    name = "gemini"

    def __init__(self, model: str = "gemini-2.5-flash", api_key: str | None = None, client: Any = None,
                 temperature: float | None = None, thinking_budget: int | None = None,
                 thinking_level: str | None = None):
        if thinking_budget is not None and thinking_level is not None:
            raise ValueError("thinking_budget and thinking_level cannot be combined (the API returns 400)")
        self.model = model
        # Google advises keeping temperature at the default 1.0 for Gemini 3 models (lower can loop).
        self.temperature = temperature if temperature is not None else (1.0 if model.startswith("gemini-3") else 0.2)
        self.thinking_budget = thinking_budget
        self.thinking_level = thinking_level
        if client is None:
            from google import genai
            key = api_key or os.environ.get("GEMINI_API_KEY")
            if not key:
                raise RuntimeError("GEMINI_API_KEY is not configured")
            client = genai.Client(api_key=key)
        self._c = client

    def generate(self, *, system: str, messages: list[dict], tools: list[ToolSpec], max_output_tokens: int,
                 force_tool: bool = False) -> LLMResponse:
        from google.genai import types
        cfg: dict = dict(system_instruction=system, temperature=self.temperature,
                         max_output_tokens=max_output_tokens,
                         automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True))
        gemini_tools = to_tools(tools)
        if gemini_tools:
            cfg["tools"] = gemini_tools
            cfg["tool_config"] = types.ToolConfig(function_calling_config=types.FunctionCallingConfig(
                mode=types.FunctionCallingConfigMode.ANY if force_tool else types.FunctionCallingConfigMode.AUTO))
        if self.thinking_level is not None:
            cfg["thinking_config"] = types.ThinkingConfig(thinking_level=types.ThinkingLevel(self.thinking_level.upper()))
        elif self.thinking_budget is not None:
            cfg["thinking_config"] = types.ThinkingConfig(thinking_budget=self.thinking_budget)
        try:
            resp = self._c.models.generate_content(model=self.model, contents=to_contents(messages),
                                                   config=types.GenerateContentConfig(**cfg))
        except Exception as exc:  # noqa: BLE001
            raise map_error(exc) from None
        return parse_response(resp)

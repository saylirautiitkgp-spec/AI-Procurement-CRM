"""Wire-level emulators of the Groq and Gemini SDK clients (OFFLINE).

Each emulator stands in for the real SDK client. It (1) strictly validates the request exactly as it would go on
the wire (roles, tool-call/tool-result pairing and ids, JSON argument strings, schema shape, forbidden parameters),
(2) translates it back to the neutral form, (3) lets a deterministic decision-maker answer, and (4) returns REAL SDK
response objects (groq ChatCompletion / google.genai GenerateContentResponse). Running the full graph through
GroqClient/GeminiClient + an emulator therefore tests adapter fidelity and measures the real payload each adapter
produces. It says NOTHING about how a real model chooses tools - that needs live runs.
"""

from __future__ import annotations

import json
import time
import uuid
from types import SimpleNamespace

from agentic.llm.base import ToolSpec

CHARS_PER_TOKEN = 4
_BAD_SCHEMA_KEYS = ("$ref", "$defs", "anyOf", "title", "additionalProperties")


def _schema_problems(schema, path="") -> list[str]:
    out = []
    if isinstance(schema, dict):
        for k, v in schema.items():
            if k in _BAD_SCHEMA_KEYS and not (k == "title" and path.endswith("properties")):
                out.append(f"{path}/{k}")
            out += _schema_problems(v, f"{path}/{k}")
    elif isinstance(schema, list):
        for i, v in enumerate(schema):
            out += _schema_problems(v, f"{path}[{i}]")
    return out


# =============================================================================================== Groq
class GroqWire:
    ALLOWED = {"model", "messages", "temperature", "max_completion_tokens", "tools", "tool_choice",
               "reasoning_effort", "include_reasoning"}
    FORBIDDEN = {"service_tier", "parallel_tool_calls", "response_format", "stream", "max_tokens"}

    def __init__(self, decider):
        self.decider, self.violations, self.payload_chars, self.requests = decider, [], [], 0
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def _check(self, kw: dict) -> None:
        v = self.violations
        v += [f"unexpected parameter {k}" for k in kw if k not in self.ALLOWED]
        v += [f"forbidden parameter {k}" for k in kw if k in self.FORBIDDEN]
        msgs = kw["messages"]
        if not msgs or msgs[0]["role"] != "system":
            v.append("first message is not system")
        pending: list[str] = []
        seen_ids: set = set()
        for m in msgs[1:]:
            if m["role"] == "tool":
                if m["tool_call_id"] in pending:
                    pending.remove(m["tool_call_id"])
                else:
                    v.append(f"tool result for unknown/duplicate id {m['tool_call_id']}")
                continue
            if pending:
                v.append(f"tool calls {pending} were not answered before the next {m['role']} message")
                pending = []
            if m["role"] == "assistant":
                for c in m.get("tool_calls", []):
                    if c["id"] in seen_ids:
                        v.append(f"duplicate tool call id {c['id']}")
                    seen_ids.add(c["id"])
                    pending.append(c["id"])
                    try:
                        json.loads(c["function"]["arguments"])
                    except (ValueError, TypeError):
                        v.append("arguments are not a JSON string")
                if m.get("content") == "" and not m.get("tool_calls"):
                    v.append("empty assistant message")
            elif m["role"] not in ("user", "assistant"):
                v.append(f"unexpected role {m['role']}")
        for t in kw.get("tools", []):
            if t.get("type") != "function":
                v.append("tool is not type=function")
            v += [f"schema {t['function']['name']}{p}" for p in _schema_problems(t["function"]["parameters"])]
        if kw.get("tool_choice") not in (None, "auto", "required", "none"):
            v.append("bad tool_choice")

    def create(self, **kw):
        from groq.types.chat import ChatCompletion
        self.requests += 1
        self._check(kw)
        chars = len(json.dumps(kw["messages"])) + len(json.dumps(kw.get("tools", [])))
        self.payload_chars.append(chars)
        names: dict[str, str] = {}
        neutral = []
        for m in kw["messages"][1:]:
            if m["role"] == "assistant":
                calls = []
                for c in m.get("tool_calls", []):
                    names[c["id"]] = c["function"]["name"]
                    calls.append({"id": c["id"], "name": c["function"]["name"],
                                  "args": json.loads(c["function"]["arguments"])})
                neutral.append({"role": "assistant", "content": m.get("content") or "", "tool_calls": calls})
            elif m["role"] == "tool":
                neutral.append({"role": "tool", "tool_call_id": m["tool_call_id"],
                                "name": names.get(m["tool_call_id"], ""), "content": m["content"]})
            else:
                neutral.append(dict(m))
        tools = [ToolSpec(name=t["function"]["name"], description=t["function"]["description"],
                          parameters=t["function"]["parameters"]) for t in kw.get("tools", [])]
        resp = self.decider.generate(system=kw["messages"][0]["content"], messages=neutral, tools=tools,
                                     max_output_tokens=kw["max_completion_tokens"],
                                     force_tool=kw.get("tool_choice") == "required")
        msg: dict = {"role": "assistant", "content": resp.text or None}
        if resp.tool_calls:
            msg["tool_calls"] = [{"id": c.id, "type": "function",
                                  "function": {"name": c.name, "arguments": json.dumps(c.args)}} for c in resp.tool_calls]
        prompt = chars // CHARS_PER_TOKEN
        return ChatCompletion.model_validate({
            "id": f"chatcmpl-{uuid.uuid4().hex[:12]}", "object": "chat.completion", "created": int(time.time()),
            "model": kw["model"],
            "choices": [{"index": 0, "message": msg, "finish_reason": "tool_calls" if resp.tool_calls else "stop"}],
            "usage": {"prompt_tokens": prompt, "completion_tokens": resp.usage.output_tokens,
                      "total_tokens": prompt + resp.usage.output_tokens}})


# ============================================================================================= Gemini
class GeminiWire:
    def __init__(self, decider):
        self.decider, self.violations, self.payload_chars, self.requests = decider, [], [], 0
        self.models = SimpleNamespace(generate_content=self.generate_content)

    def generate_content(self, model, contents, config):
        from google.genai import types
        v = self.violations
        self.requests += 1
        if not config.automatic_function_calling or not config.automatic_function_calling.disable:
            v.append("SDK automatic function calling is not disabled")
        if not contents or contents[0].role != "user":
            v.append("first content is not from the user")
        neutral, last_calls, n = [], [], 0
        for c in contents:
            parts = c.parts or []
            if not parts:
                v.append("content with no parts")
            if c.role == "model":
                calls = [p.function_call for p in parts if p.function_call]
                text = "".join(p.text or "" for p in parts if p.text)
                tcs = []
                for fc in calls:
                    n += 1
                    tcs.append({"id": f"w{n}", "name": fc.name, "args": dict(fc.args or {})})
                neutral.append({"role": "assistant", "content": text, "tool_calls": tcs})
                last_calls = list(tcs)
            elif c.role == "user":
                responses = [p.function_response for p in parts if p.function_response]
                if responses:
                    if [r.name for r in responses] != [t["name"] for t in last_calls]:
                        v.append("function_response names/order do not match the preceding function_call parts")
                    for r, t in zip(responses, last_calls):
                        neutral.append({"role": "tool", "tool_call_id": t["id"], "name": r.name,
                                        "content": (r.response or {}).get("result", "")})
                    last_calls = []
                else:
                    if last_calls:
                        v.append("function_call parts were not answered by function_response parts")
                        last_calls = []
                    neutral.append({"role": "user", "content": "".join(p.text or "" for p in parts)})
            else:
                v.append(f"unexpected role {c.role}")
            if c.role == "user" and any(p.function_call for p in parts):
                v.append("function_call part inside a user content")
        decls = [d for t in (config.tools or []) for d in (t.function_declarations or [])]
        for d in decls:
            v += [f"schema {d.name}{p}" for p in _schema_problems(d.parameters_json_schema or {})]
        chars = sum(len(c.model_dump_json(exclude_none=True)) for c in contents) + sum(len(d.model_dump_json(exclude_none=True)) for d in decls) + \
            len(config.system_instruction or "")
        self.payload_chars.append(chars)
        tools = [ToolSpec(name=d.name, description=d.description or "", parameters=d.parameters_json_schema or {})
                 for d in decls]
        force = bool(config.tool_config and config.tool_config.function_calling_config and
                     config.tool_config.function_calling_config.mode == types.FunctionCallingConfigMode.ANY)
        resp = self.decider.generate(system=config.system_instruction, messages=neutral, tools=tools,
                                     max_output_tokens=config.max_output_tokens, force_tool=force)
        parts = [types.Part(function_call=types.FunctionCall(name=c.name, args=c.args)) for c in resp.tool_calls]
        if resp.text:
            parts.append(types.Part(text=resp.text))
        return types.GenerateContentResponse(
            candidates=[types.Candidate(content=types.Content(role="model", parts=parts))],
            usage_metadata=types.GenerateContentResponseUsageMetadata(
                prompt_token_count=chars // CHARS_PER_TOKEN, candidates_token_count=resp.usage.output_tokens))

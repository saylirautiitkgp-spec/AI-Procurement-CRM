"""Deterministic LLM doubles for offline tests (no network, no cost)."""

from __future__ import annotations

import itertools
from typing import Any, Callable

from agentic.llm.base import LLMResponse, ToolCallRequest, Usage

_ids = itertools.count(1)


def tool_call(name: str, **args: Any) -> LLMResponse:
    return LLMResponse(
        tool_calls=[ToolCallRequest(id=f"fake{next(_ids)}", name=name, args=args)],
        usage=Usage(requests=1, input_tokens=100, output_tokens=20),
    )


def text_response(text: str) -> LLMResponse:
    return LLMResponse(text=text, usage=Usage(requests=1, input_tokens=100, output_tokens=20))


class FakeLLM:
    """Plays back a script (list of LLMResponse / Exception) or delegates to a
    callable(system, messages, tools) -> LLMResponse. Records every call."""

    name = "fake"
    model = "fake-model"

    def __init__(self, script: list | Callable | None = None):
        self.script = script if script is not None else []
        self.calls: list[dict] = []

    def generate(self, *, system, messages, tools, max_output_tokens, force_tool=False):
        self.calls.append({
            "system": system,
            "messages": [dict(m) for m in messages],
            "tools": [t.name for t in tools],
            "max_output_tokens": max_output_tokens,
        })
        if callable(self.script):
            return self.script(system, messages, tools)
        if not self.script:
            raise AssertionError("FakeLLM script exhausted")
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

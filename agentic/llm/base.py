"""Provider-neutral LLM interface.

Agents only ever see this module. Adding a provider means writing one adapter
that implements `generate`; there is deliberately NO automatic fallback
between providers.
"""

from __future__ import annotations

import time
from typing import Callable, Protocol

from pydantic import BaseModel, Field


class ToolSpec(BaseModel):
    name: str
    description: str
    parameters: dict  # JSON schema


class ToolCallRequest(BaseModel):
    id: str
    name: str
    args: dict = Field(default_factory=dict)


class Usage(BaseModel):
    requests: int = 1
    input_tokens: int = 0
    output_tokens: int = 0


class LLMResponse(BaseModel):
    text: str = ""
    tool_calls: list[ToolCallRequest] = Field(default_factory=list)
    usage: Usage = Field(default_factory=Usage)
    provider_data: dict | None = None      # opaque provider state to replay (e.g. thought signatures)


class LLMError(Exception):
    """kind: rate_limit | quota | transient | invalid | other"""

    def __init__(self, kind: str, message: str = "", retry_after: float | None = None):
        super().__init__(message or kind)
        self.kind = kind
        self.retry_after = retry_after


class LLMClient(Protocol):
    name: str
    model: str

    def generate(self, *, system: str, messages: list[dict], tools: list[ToolSpec],
                 max_output_tokens: int, force_tool: bool = False) -> LLMResponse: ...


class RetryingLLM:
    """Bounded retry with exponential backoff for rate limits / transient errors.
    Quota exhaustion is never retried."""

    def __init__(self, inner: LLMClient, max_retries: int = 2, base_delay: float = 1.0,
                 sleep: Callable[[float], None] = time.sleep):
        self.inner = inner
        self.name = inner.name
        self.model = inner.model
        self.max_retries = max_retries
        self.base_delay = base_delay
        self._sleep = sleep
        self.retries = 0

    def generate(self, **kwargs) -> LLMResponse:
        attempt = 0
        while True:
            try:
                return self.inner.generate(**kwargs)
            except LLMError as err:
                if err.kind not in ("rate_limit", "transient") or attempt >= self.max_retries:
                    raise
                delay = err.retry_after if err.retry_after else self.base_delay * (2 ** attempt)
                self._sleep(min(delay, 30.0))
                attempt += 1
                self.retries += 1

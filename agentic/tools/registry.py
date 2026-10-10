"""Tool registry. Each tool declares which agents may use it, its argument
schema (pydantic) and its cost class. The LLM only sees the specs of the
tools its agent is permitted to use."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from pydantic import BaseModel

from agentic.llm.base import ToolSpec


class ToolError(Exception):
    def __init__(self, code: str, message: str = "", retryable: bool = False):
        super().__init__(message or code)
        self.code = code
        self.message = message or code
        self.retryable = retryable


@dataclass
class ToolContext:
    deps: Any                      # injected services (repo, search backend, fetcher, ...)
    agent: str
    call_id: str
    ledger: dict                   # read-only snapshot
    subject: dict                  # read-only snapshot
    claims: dict = field(default_factory=dict)
    limits: Any = None
    search_disabled: bool = False


@dataclass
class ToolOutput:
    data: dict
    docs: list[dict] = field(default_factory=list)   # [{"url","text","title"}] -> ledger
    credits: int = 0
    cached: bool = False


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    args_model: type[BaseModel]
    handler: Callable[[ToolContext, BaseModel], ToolOutput]
    agents: frozenset
    kind: str = "read"             # read | search | fetch | compute | finish
    credit_cost: Callable[[BaseModel], int] | None = None
    is_cached: Callable[[ToolContext, BaseModel], bool] | None = None   # cache hits cost no credits


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"duplicate tool {tool.name}")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def allowed(self, agent: str, name: str) -> bool:
        tool = self._tools.get(name)
        return bool(tool and agent in tool.agents)

    def specs_for(self, agent: str) -> list[ToolSpec]:
        return [
            ToolSpec(name=t.name, description=t.description,
                     parameters=t.args_model.model_json_schema())
            for t in self._tools.values() if agent in t.agents
        ]

    def names_for(self, agent: str) -> list[str]:
        return [t.name for t in self._tools.values() if agent in t.agents]

"""Injected services. The graph never imports a concrete Supabase/Tavily/LLM
client; tests inject fakes, live runs inject real adapters."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable

from agentic.config import Limits, Settings
from agentic.llm.base import LLMClient
from agentic.persistence.store import Store
from agentic.tools.registry import ToolRegistry


@dataclass
class Deps:
    llm: LLMClient
    registry: ToolRegistry
    store: Store
    repo: Any                       # CompanyRepository (read-only)
    search: Any                     # SearchBackend
    fetcher: Any                    # PageFetcher
    commit_repo: Any = None         # CommitRepository (only used by the commit node)
    settings: Settings = field(default_factory=Settings)
    sleep: Callable[[float], None] = time.sleep

    @property
    def limits(self) -> Limits:
        return self.settings.limits

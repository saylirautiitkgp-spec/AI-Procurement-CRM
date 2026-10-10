"""Web search backends. The live backend is Tavily (free plan: 1,000
credits/month; basic/fast = 1 credit, advanced = 2). It is only constructed
when live calls are explicitly enabled."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Protocol

from agentic.config import SEARCH_DEPTH_CREDITS


class SearchQuotaExceeded(Exception):
    """Plan/credit limit reached (Tavily HTTP 432/433)."""


class SearchRateLimited(Exception):
    def __init__(self, retry_after: float | None = None):
        super().__init__("rate limited")
        self.retry_after = retry_after


class SearchFailure(Exception):
    """Transient or unknown failure."""


@dataclass
class SearchResponse:
    results: list[dict] = field(default_factory=list)   # {"url","title","content"}
    credits: int = 1


class SearchBackend(Protocol):
    def search(self, query: str, depth: str, max_results: int) -> SearchResponse: ...


class TavilySearchBackend:
    """Thin adapter. The API key is read from the environment at construction
    time and is never logged or placed in prompts."""

    def __init__(self, api_key: str | None = None):
        from tavily import TavilyClient  # imported lazily: offline tests need no SDK/key
        key = api_key or os.environ.get("TAVILY_API_KEY")
        if not key:
            raise RuntimeError("TAVILY_API_KEY is not configured")
        self._client = TavilyClient(api_key=key)

    def search(self, query: str, depth: str, max_results: int) -> SearchResponse:
        try:
            from tavily import errors as tv
        except Exception:  # pragma: no cover
            tv = None
        try:
            raw = self._client.search(query=query, search_depth=depth, max_results=max_results,
                                      include_answer=False, include_raw_content=False)
        except Exception as exc:  # map SDK errors to our small vocabulary
            name = type(exc).__name__
            if tv is not None and isinstance(exc, getattr(tv, "UsageLimitExceededError", ())):
                raise SearchQuotaExceeded(name) from None
            if "UsageLimit" in name or "432" in str(exc) or "433" in str(exc):
                raise SearchQuotaExceeded(name) from None
            if "429" in str(exc) or "RateLimit" in name:
                raise SearchRateLimited() from None
            raise SearchFailure(name) from None
        results = [{"url": r.get("url", ""), "title": r.get("title", ""), "content": r.get("content", "")}
                   for r in raw.get("results", [])]
        return SearchResponse(results=results, credits=SEARCH_DEPTH_CREDITS.get(depth, 2))

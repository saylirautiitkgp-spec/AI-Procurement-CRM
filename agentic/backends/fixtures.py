"""Offline search/fetch backends that replay a synthetic "world" (zero network,
zero cost). Used by tests and the offline evaluation harness."""

from __future__ import annotations

import re

from agentic.backends.fetch import FetchedPage, FetchError
from agentic.backends.search import (SearchFailure, SearchQuotaExceeded, SearchRateLimited,
                                     SearchResponse)
from agentic.config import SEARCH_DEPTH_CREDITS
from agentic.evidence import canonical_url

_NOISE = {"official", "website", "site", "company", "the", "and", "of", "in", "a", "is", "about", "us",
          "profile", "homepage", "corporate", "india", "limited", "ltd", "pvt", "private"}


def _toks(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in _NOISE and len(t) > 1}


class FixtureSearch:
    """world["search_index"]: [{"url","title","content"}]; world["search_failure"]:
    {"mode": "quota"|"rate_limit"|"error"|"empty", "after": n_successful_calls}."""

    def __init__(self, world: dict):
        self.world = world
        self.calls: list[tuple] = []

    def search(self, query: str, depth: str, max_results: int) -> SearchResponse:
        self.calls.append((query, depth, max_results))
        fail = self.world.get("search_failure")
        if fail and len(self.calls) > fail.get("after", 0):
            mode = fail["mode"]
            if mode == "quota":
                raise SearchQuotaExceeded("fixture quota")
            if mode == "rate_limit":
                raise SearchRateLimited(retry_after=0.0)
            if mode == "error":
                raise SearchFailure("fixture error")
            if mode == "empty":
                return SearchResponse(results=[], credits=SEARCH_DEPTH_CREDITS.get(depth, 2))
        q = _toks(query)
        scored = []
        for i, doc in enumerate(self.world.get("search_index", [])):
            hay = _toks(doc["title"] + " " + doc["content"] + " " + doc["url"])
            hit = len(q & hay)
            need = 1 if len(q) <= 1 else max(2, int(len(q) * 0.5 + 0.5))
            if hit >= need:
                scored.append((-hit, i, doc))
        scored.sort(key=lambda t: (t[0], t[1]))
        results = [dict(d) for _, _, d in scored[:max_results]]
        return SearchResponse(results=results, credits=SEARCH_DEPTH_CREDITS.get(depth, 2))


class FixtureFetcher:
    """world["pages"]: {url: {"title","text","links":[{"url","text"}]}};
    world["fetch_errors"]: {url: code}."""

    def __init__(self, world: dict):
        self.pages = {canonical_url(u): p for u, p in world.get("pages", {}).items()}
        self.errors = {canonical_url(u): c for u, c in world.get("fetch_errors", {}).items()}
        self.calls: list[str] = []

    def fetch(self, url: str) -> FetchedPage:
        self.calls.append(url)
        key = canonical_url(url)
        if key in self.errors:
            raise FetchError(self.errors[key])
        page = self.pages.get(key)
        if page is None:
            raise FetchError("http_404")
        return FetchedPage(url=url, final_url=key, status=200, title=page.get("title", ""),
                           text=page["text"], links=page.get("links", []), truncated=False)

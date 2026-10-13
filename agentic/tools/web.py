"""web_search and fetch_page tools (read-only, cached, budgeted)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from agentic.backends.fetch import FetchError
from agentic.backends.search import SearchFailure, SearchQuotaExceeded, SearchRateLimited
from agentic.config import SEARCH_DEPTH_CREDITS
from agentic.evidence import known_domains, registrable_domain
from agentic.guards.url_policy import UrlBlocked, check_url
from agentic.tools.registry import Tool, ToolContext, ToolError, ToolOutput

SEARCH_TTL_DAYS = 30
FETCH_TTL_DAYS = 7
SNIPPET_CHARS = 600
_NO_DNS = lambda host: ["8.8.8.8"]  # noqa: E731 - structural check only; real DNS check is in the fetcher


class WebSearchArgs(BaseModel):
    query: str = Field(min_length=3, max_length=200,
                       description="Search query. Include the company name; be specific.")
    depth: Literal["basic", "fast", "advanced"] = Field(
        default="basic", description="basic/fast cost 1 search credit, advanced costs 2. Prefer basic.")
    max_results: int = Field(default=5, ge=1, le=5)


class FetchPageArgs(BaseModel):
    url: str = Field(min_length=8, max_length=500,
                     description="A URL on a domain already seen in search results or fetched pages.")
    offset: int = Field(default=0, ge=0, le=200_000,
                        description="Character offset to continue reading a long page.")


def _search_key(store, args: WebSearchArgs) -> str:
    return store.cache_key("web_search", {"q": " ".join(args.query.lower().split()),
                                          "d": args.depth, "n": args.max_results})


def search_is_cached(ctx: ToolContext, args: WebSearchArgs) -> bool:
    store = ctx.deps.store
    return store.cache_get(_search_key(store, args), SEARCH_TTL_DAYS) is not None


def search_cost(args: WebSearchArgs) -> int:
    return SEARCH_DEPTH_CREDITS[args.depth]


def web_search(ctx: ToolContext, args: WebSearchArgs) -> ToolOutput:
    d = ctx.deps
    key = _search_key(d.store, args)
    with d.search_lock:       # parallel agents: a repeated query is answered from the cache, never paid for twice
        results = d.store.cache_get(key, SEARCH_TTL_DAYS)
        cached = results is not None
        credits = 0
        if not cached:
            if ctx.search_disabled:
                raise ToolError("search_disabled", "Search is disabled for the rest of this run.")
            attempt = 0
            while True:
                try:
                    resp = d.search.search(" ".join(args.query.split()), args.depth, args.max_results)
                    break
                except SearchQuotaExceeded:
                    raise ToolError("search_quota_exhausted",
                                    "Search provider quota exhausted; no more searches this run.")
                except SearchRateLimited as exc:
                    attempt += 1
                    if attempt > 2:
                        raise ToolError("search_rate_limited_persistent", "Search rate limited repeatedly.")
                    d.sleep(min(exc.retry_after or 2.0 * attempt, 20.0))
                except SearchFailure as exc:
                    raise ToolError("search_failed", f"Search failed: {exc}", retryable=True)
            results = resp.results
            credits = resp.credits
            d.store.cache_put(key, results)
    docs = [{"url": r["url"], "text": r.get("content", ""), "title": r.get("title", "")}
            for r in results if r.get("url")]
    data = {
        "query": args.query,
        "cached": cached,
        "results": [{"url": r["url"], "title": r.get("title", "")[:150],
                     "snippet": (r.get("content") or "")[:SNIPPET_CHARS]} for r in results],
    }
    if not results:
        data["note"] = "No results. Try different wording, a different aspect, or stop."
    return ToolOutput(data=data, docs=docs, credits=credits, cached=cached)


def _fetch_key(store, url: str) -> str:
    return store.cache_key("fetch_page", {"u": url})


def fetch_is_cached(ctx: ToolContext, args: FetchPageArgs) -> bool:
    return False   # fetches cost no search credits


def fetch_page(ctx: ToolContext, args: FetchPageArgs) -> ToolOutput:
    d = ctx.deps
    try:
        check_url(args.url, _NO_DNS)
    except UrlBlocked as exc:
        raise ToolError("url_blocked", f"URL not allowed: {exc.reason}")
    if registrable_domain(args.url) not in known_domains(ctx.ledger):
        raise ToolError("url_not_discovered",
                        "Only domains already seen in search results or fetched pages can be fetched. "
                        "Search first to discover the site.")
    key = _fetch_key(d.store, args.url)
    page = d.store.cache_get(key, FETCH_TTL_DAYS)
    cached = page is not None
    if page is None:
        try:
            fetched = d.fetcher.fetch(args.url)
        except FetchError as exc:
            raise ToolError("fetch_failed", f"Could not fetch page: {exc.code}",
                            retryable=exc.code in ("network_error",))
        page = {"url": fetched.url, "final_url": fetched.final_url, "status": fetched.status,
                "title": fetched.title, "text": fetched.text, "links": fetched.links,
                "truncated": fetched.truncated}
        d.store.cache_put(key, page)
    limit = ctx.limits.tool_result_chars_for_llm if ctx.limits else 6000
    text = page["text"]
    chunk = text[args.offset: args.offset + limit]
    next_offset = args.offset + limit if args.offset + limit < len(text) else None
    docs = [{"url": page["final_url"], "text": text, "title": page["title"]}]
    if page["url"] != page["final_url"]:
        docs.append({"url": page["url"], "text": text, "title": page["title"]})
    data = {"url": page["final_url"], "title": page["title"], "status": page["status"],
            "text": chunk, "offset": args.offset, "next_offset": next_offset,
            "links": page["links"][:25], "page_truncated": page["truncated"]}
    return ToolOutput(data=data, docs=docs, credits=0, cached=cached)


def build_web_tools(agents: frozenset) -> list[Tool]:
    return [
        Tool(name="web_search",
             description=("Search the public web. Returns result URLs with text snippets. "
                          "Costs search credits (basic=1, advanced=2); identical queries are cached."),
             args_model=WebSearchArgs, handler=web_search, agents=agents, kind="search",
             credit_cost=search_cost, is_cached=search_is_cached),
        Tool(name="fetch_page",
             description=("Fetch a public web page and return its text and same-site links. "
                          "Only domains already seen in search results or fetched pages are allowed. "
                          "Use offset to read further into a long page."),
             args_model=FetchPageArgs, handler=fetch_page, agents=agents, kind="fetch"),
    ]

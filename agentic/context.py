"""Prompt-context compaction.

These helpers shrink what is SENT to the model on later turns without touching what is stored or validated:
the full conversation stays in graph state (audit), and every evidence snippet is still checked against the
complete document text in the evidence ledger. Fetched page text - the thing the model must quote from - is
never shortened here.
"""

from __future__ import annotations

import json

from agentic.evidence import canonical_url

_STRIP = {"maxLength", "minLength", "maxItems", "minItems", "maximum", "minimum", "exclusiveMaximum",
          "exclusiveMinimum", "pattern"}
SUPERSEDED = "[snippet omitted: this page was fetched in full later in the conversation]"


def _unwrap(content: str):
    """(head, payload, tail) for an untrusted-content wrapped JSON tool message, else None."""
    lines = content.split("\n")
    if len(lines) >= 3 and lines[0].startswith("<untrusted_web_content") and lines[-1] == "</untrusted_web_content>":
        try:
            return lines[0], json.loads("\n".join(lines[1:-1])), lines[-1]
        except json.JSONDecodeError:
            return None
    return None


def compact_messages(messages: list[dict], keep_links: int = 10) -> list[dict]:
    """Return a copy of `messages` with redundant tool-result detail removed:
    * search snippets for URLs that were fetched in full LATER (the fetched text supersedes them);
    * the same-site link list of every fetch result except the most recent one (trimmed to `keep_links`).
    Fetched page text, assistant turns, errors and everything else are untouched."""
    parsed: dict[int, tuple] = {}
    for i, m in enumerate(messages):
        if m["role"] == "tool" and m.get("name") in ("web_search", "fetch_page"):
            u = _unwrap(m["content"])
            if u:
                parsed[i] = u
    fetched_after: dict[int, set] = {}
    seen: set = set()
    last_fetch = max((i for i, (_, p, _) in parsed.items() if messages[i]["name"] == "fetch_page"), default=-1)
    for i in sorted(parsed, reverse=True):
        fetched_after[i] = set(seen)
        if messages[i]["name"] == "fetch_page":
            seen.add(canonical_url(parsed[i][1].get("url", "")))
    out = []
    for i, m in enumerate(messages):
        if i not in parsed:
            out.append(m)
            continue
        head, payload, tail = parsed[i]
        changed = False
        if m["name"] == "web_search":
            for r in payload.get("results", []):
                if canonical_url(r.get("url", "")) in fetched_after[i] and r.get("snippet") != SUPERSEDED:
                    r["snippet"] = SUPERSEDED
                    changed = True
        elif i != last_fetch and len(payload.get("links", [])) > keep_links:
            payload["links_omitted"] = len(payload["links"]) - keep_links
            payload["links"] = payload["links"][:keep_links]
            changed = True
        if changed:
            m = {**m, "content": head + "\n" + json.dumps(payload, default=str) + "\n" + tail}
        out.append(m)
    return out


def compact_schema(schema: dict) -> dict:
    """Drop validation-only keywords from a tool schema before sending it. Python still enforces them."""
    if isinstance(schema, list):
        return [compact_schema(s) for s in schema]
    if not isinstance(schema, dict):
        return schema
    return {k: compact_schema(v) for k, v in schema.items() if k not in _STRIP}


def ledger_hint(ledger: dict, limit: int = 8) -> str:
    docs = sorted(ledger.values(), key=lambda d: d["retrieved_at"])[:limit]
    if not docs:
        return ""
    lines = ["Documents already retrieved in this run (fetch_page on them is cached and free; you may quote them "
             "without searching again):"]
    lines += [f"  - {d['url']} ({(d.get('title') or 'untitled')[:60]}, {len(d['text'])} chars)" for d in docs]
    return "\n".join(lines)

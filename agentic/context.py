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


# --------------------------------------------------------------------------------------------------------------------
# Provider-limit fitting. Only used when the LLM client is paced against a per-minute token limit (AGENTIC_QUOTA): the
# estimated request must then fit, or the pacer refuses it locally and the agent ends with `llm_invalid`.
# --------------------------------------------------------------------------------------------------------------------
OMITTED = ("[omitted from this prompt to fit the provider's per-minute token limit; the document is still held in the "
           "evidence ledger - fetch_page it again if you need to quote from it]")


def _rewrap(head: str, payload: dict, tail: str) -> str:
    return head + "\n" + json.dumps(payload, default=str) + "\n" + tail


def fit_to_budget(messages: list[dict], estimate, budget: int) -> tuple[list[dict], list[dict]]:
    """Shrink the SENT history until `estimate(messages) <= budget`; returns (messages, steps taken).

    Order, least damaging first, stopping as soon as the request fits:
      1. drop the same-site link lists (navigation only, never evidence);
      2. replace the page text / search snippets of the OLDEST web results by an explicit notice, one result at a
         time. The most recent web result is never touched, and assistant turns, the task and errors are untouched.
    The stored conversation and the evidence ledger are NOT changed, so validation still uses the full documents;
    a claim whose page text was omitted can only be cited if the model fetches that page again. If nothing more can
    be dropped the request is returned as is (the pacer then refuses it - an honest failure, not a silent one)."""
    if estimate(messages) <= budget:
        return messages, []
    out = list(messages)
    idx = [i for i, m in enumerate(out) if m["role"] == "tool" and m.get("name") in ("web_search", "fetch_page")]
    parsed = {i: u for i in idx if (u := _unwrap(out[i]["content"]))}
    steps: list[dict] = []
    for i, (head, payload, tail) in parsed.items():
        if payload.get("links"):
            steps.append({"step": "links_dropped", "message": i, "count": len(payload["links"])})
            payload["links"] = []
            out[i] = {**out[i], "content": _rewrap(head, payload, tail)}
    if estimate(out) <= budget:
        return out, steps
    for i in sorted(parsed)[:-1]:                                    # oldest first; keep the newest result intact
        head, payload, tail = parsed[i]
        if out[i]["name"] == "fetch_page":
            if payload.get("text") == OMITTED:
                continue
            steps.append({"step": "page_text_omitted", "message": i, "url": payload.get("url"),
                          "chars": len(payload.get("text") or "")})
            payload["text"] = OMITTED
        else:
            n = 0
            for r in payload.get("results", []):
                if r.get("snippet") not in (SUPERSEDED, OMITTED):
                    n += len(r.get("snippet") or "")
                    r["snippet"] = OMITTED
            if not n:
                continue
            steps.append({"step": "search_snippets_omitted", "message": i, "chars": n})
        out[i] = {**out[i], "content": _rewrap(head, payload, tail)}
        if estimate(out) <= budget:
            break
    return out, steps

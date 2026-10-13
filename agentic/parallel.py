"""Safe parallel execution of independent agents: budget partitioning and deterministic result merging.

Each parallel agent runs on a PRIVATE copy of the run state (its own messages, notes, claims, budgets), starting from
the shared evidence ledger, so agents cannot overwrite each other. Shared resources (LLM client, search/fetch
backends, the SQLite store) are thread-safe or serialised by lock. Afterwards the results are merged in the fixed plan
order - never in completion order - so a run is reproducible.

Budget safety: run-level caps (LLM calls, tool calls, search credits, tokens) are PARTITIONED before the agents start.
Each branch is told it has already used everything outside its own share, so it cannot exceed its share even though
the agents run concurrently. After the run the real usage of every branch is added to the shared totals.
"""

from __future__ import annotations

import copy

from agentic import evidence as ev

_COUNTERS = ("llm_calls", "tool_calls", "search_credits")


def _caps(limits) -> dict:
    return {"llm_calls": limits.max_llm_calls_run, "tool_calls": limits.max_tool_calls_run,
            "search_credits": min(limits.max_search_credits_run, limits.max_search_credits_company),
            "tokens": limits.max_total_tokens_run}


def _used(b: dict) -> dict:
    return {"llm_calls": b["llm_calls"], "tool_calls": b["tool_calls"], "search_credits": b["search_credits"],
            "tokens": b["input_tokens"] + b["output_tokens"]}


def split_budgets(base: dict, limits, k: int) -> list[tuple[dict, dict]]:
    """k (branch_budgets, reserved) pairs. `reserved` = usage pre-charged to hide the other branches' shares."""
    caps, used = _caps(limits), _used(base)
    out = []
    for i in range(k):
        b = copy.deepcopy(base)
        reserved = {}
        for name, cap in caps.items():
            remaining = max(0, cap - used[name])
            share = remaining // k + (1 if i < remaining % k else 0)
            reserved[name] = remaining - share
        for name in _COUNTERS:
            b[name] += reserved[name]
        b["input_tokens"] += reserved["tokens"]
        out.append((b, reserved))
    return out


def merge_budgets(base: dict, branches: list[tuple[dict, dict]]) -> dict:
    """Shared totals after the branches finished: base + the real usage of each branch."""
    merged = copy.deepcopy(base)
    for b, reserved in branches:
        for name in _COUNTERS:
            merged[name] += b[name] - base[name] - reserved[name]
        merged["input_tokens"] += b["input_tokens"] - base["input_tokens"] - reserved["tokens"]
        merged["output_tokens"] += b["output_tokens"] - base["output_tokens"]
        merged["llm_retries"] = merged.get("llm_retries", 0) + (b.get("llm_retries", 0) - base.get("llm_retries", 0))
        merged["call_signatures"].update(b.get("call_signatures", {}))
        if b.get("search_disabled") and not merged.get("search_disabled"):
            merged["search_disabled"] = True
            merged["search_disabled_reason"] = b.get("search_disabled_reason")
    return merged


def merge_ledgers(base: dict, ledgers: list[dict], *, max_docs: int, max_chars: int) -> dict:
    """Union of the shared ledger and every branch's documents (de-duplicated by canonical URL)."""
    merged = copy.deepcopy(base)
    for led in ledgers:
        for doc in led.values():
            ev.add_doc(merged, url=doc["url"], text=doc.get("text", ""), title=doc.get("title", ""),
                       tool=doc.get("tool", ""), call_id=doc.get("call_id", ""), max_docs=max_docs, max_chars=max_chars)
    return merged


def renumber_events(base_len: int, events_by_branch: list[list[dict]]) -> list[dict]:
    """Branch events in plan order with one continuous sequence after the shared trace."""
    out, seq = [], base_len
    for events in events_by_branch:
        for e in events:
            seq += 1
            out.append({**e, "seq": seq})
    return out

# Token / request cost of a full five-agent run - offline analysis

Method: the same fixtures (the first-20 development companies whose mock web supports a full run, 10 companies)
run through the real graph with the offline `PolicyLLM` double; `PromptSizeProbe` measures what the provider
adapters would really send (system prompt + flattened tool schemas + conversation). Token counts are estimates
(characters / 4). `python -m agentic.eval.cost_report --large` regenerates the tables. **No live calls were made.**

**Correction to an earlier number:** the first estimate (43k tokens) measured raw pydantic schemas; what is
actually sent is flattened, so the true baseline is about 37.8k on a single run and 43.5k averaged over the 10
cases (small synthetic pages) / 69.4k with realistic 6k-character pages.

## Where the tokens go (baseline, one Apollo-like run, 25 requests)

| Component | Share | Notes |
|---|---|---|
| Tool schemas, re-sent on every request | 37% | about 560 tokens per request; the `submit_*` schemas dominate |
| Conversation history (task + tool results) | 35% | grows within an agent turn; web results are 14% on small pages and far more on real pages |
| System prompt (safety rules + role), re-sent | 27% | about 400 tokens per request |

Redundancies found: (1) a final supervisor call to "choose" finish when no agent is eligible (0.8 calls/run on
average); (2) two `get_crm_context` round trips whose data was already (or could be) in the task text; (3) a lead
agent that spends a round trip calling a deterministic score tool; (4) search snippets re-sent after the same page
was fetched in full, and 25-link lists re-sent for every old page; (5) verbose per-agent agent descriptions in the
supervisor prompt; (6) a later agent re-searching for pages an earlier agent already retrieved; (7) independent page
fetches requested one request at a time.

Not redundant, left alone: the per-agent research steps themselves, the supervisor's real routing decisions, the
safety preamble, every validation rule.

## Before / after (same fixtures, output verified identical to baseline in every row)

Realistic page sizes (10 cases):

| Configuration | LLM requests/run | est. input tokens/run | search credits/run | saved | output identical |
|---|---|---|---|---|---|
| baseline (all off) | 25.8 | 69,402 | 5.0 | - | yes |
| + skip_forced_finish | 25.0 | 68,713 | 5.0 | 1.0% | yes |
| + crm_digest_in_task | 23.8 | 64,928 | 5.0 | 6.4% | yes |
| + lead_precomputed | 24.8 | 68,209 | 5.0 | 1.7% | yes |
| + compact_history | 25.8 | 66,170 | 5.0 | 4.7% | yes |
| + short_supervisor_text | 25.8 | 68,552 | 5.0 | 1.2% | yes |
| + compact_schemas (off by default) | 25.8 | 67,870 | 5.0 | 2.2% | yes |
| + ledger_hint (off by default) | 25.0 | 68,354 | 4.0 | 1.5% | yes |
| + batch_hint (off by default) | 24.8 | 63,625 | 5.0 | 8.3% | yes |
| **defaults (safe set ON)** | **22.0** | **59,100** | 5.0 | **14.8%** | yes |
| everything ON | 20.2 | 51,391 | 4.0 | 26.0% | yes |

Small synthetic pages: defaults save 17.1% (22.0 requests), everything ON saves 26.1% (20.2 requests, 4.0 credits).

## What is on by default, what is not, and why

ON (mechanical: the model sees the same facts, fewer repeated tokens): `skip_forced_finish`,
`crm_digest_in_task`, `lead_precomputed`, `compact_history`, `short_supervisor_text`.

OFF until a live comparison shows no regression (they change what the model is told or sees, and the offline
double is not an LLM, so the offline run only proves they are mechanically correct):
`compact_schemas` (the model no longer sees length limits; Python still enforces them and a test proves it),
`ledger_hint` (saves one search + credit if a model uses the hint), `batch_hint` (only helps if the model supports
parallel calls; gpt-oss on Groq does not).

## Safety invariants (tested in `tests/agentic/test_efficiency.py`)

* Evidence is validated against the complete document text in the ledger, not against the (compacted) prompt.
* Fetched page text - what the model must quote - is never shortened; only superseded snippets and old link lists.
* The full conversation stays in graph state for audit; compaction applies to a copy sent to the model.
* Run budgets (LLM calls, tokens, wall clock, search credits, tool calls) are enforced with every switch on.
* A forced finish still appears in `route_history` / the trace, flagged `deterministic`.
* Research output (claims, statuses, lead score, identity) is byte-for-byte equal to the baseline in all rows.

## Remaining ideas (not done)

Shorter safety preamble and `submit_*` schema descriptions (about 10% more, but they change model instructions:
needs a live A/B); merging the contact and procurement agents when both only read the same two pages (changes the
five-agent design: not done); a stateful provider API that stores the conversation server-side.

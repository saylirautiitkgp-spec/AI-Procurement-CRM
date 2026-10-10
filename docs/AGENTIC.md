# Agentic procurement-intelligence runtime

A LangGraph runtime with an LLM-driven supervisor and five tool-calling agents. It sits **next to** the
legacy code (`agents/`, `services/`, `workflow/`) and does not replace it. Python package: `agentic/`.

## Run the tests (offline, free, no keys)

```bash
python -m venv .venv-agentic
.venv-agentic/Scripts/python -m pip install -r requirements-agentic.txt
.venv-agentic/Scripts/python -m pytest                       # ~160 tests, ~35 s, no network
.venv-agentic/Scripts/python -m agentic.eval.run_offline     # offline evaluation of the first-20 set
```

## How a run works

```
intake -> supervisor --(LLM route tool, guarded)--> agent_llm <-> agent_tools --> agent_done -> supervisor
              |                                                                                  
              +--> finalize -> [propose -> approval (interrupt) -> commit] -> END
```

* **intake**: resolves the submitted *name* against Supabase (no ID needed). Result: `found`, `ambiguous`
  (e.g. the same company listed twice) or `not_found`. A not-found company is researched exactly the same
  way; the research object lives only in the run.
* **supervisor**: an LLM chooses the next agent by calling `route`. Python vetoes invalid choices
  (prerequisites, no-progress, run limits). It may finish early; agents are never run as a fixed pipeline.
* **agents**: each is a loop. The LLM picks a permitted tool + arguments, the `GuardedExecutor` runs it,
  the result goes back to the LLM, which decides to search again, fetch, revise, or call its
  `submit_*_findings` tool. Python enforces call/credit/token/time limits and stops stalled loops.
* **validation**: every claim must quote a snippet that exists verbatim in a document retrieved *in this
  run*; values must be supported by the snippet; confidence is computed from evidence, never self-reported.
* **writes**: agents have no write tools. Verified claims become *proposals*; applying needs a recorded
  human approval, passes the write gate (allowlisted tables/columns, no deletes, overwrite detection,
  tamper hash), is idempotent and audited. `merge_all()` is no longer run automatically.

| Agent | Tools | Output (all evidence-validated) |
|---|---|---|
| company | web_search, fetch_page | identity/official site, HQ, industry, products, size, locations |
| contact | web_search, fetch_page, get_crm_context | named procurement contacts; email/phone only if printed on the page |
| procurement | web_search, fetch_page, get_crm_context | supplier portal, procurement page, purchasing categories/components |
| supplier | web_search, fetch_page, get_crm_context | supplier relationships stated in one sentence naming both parties |
| lead | compute_lead_score | deterministic, explainable score from *verified* data only |

## Free-tier controls (all in `agentic/config.py`)

Per run: 10 search credits, 40 LLM calls, 40 tool calls, 60k tokens, 180 s. Monthly: 300 Tavily credits
(the free plan has 1,000; the legacy pipeline can spend up to 36 per company). Search depth defaults to
`basic` (1 credit); identical searches are cached for 30 days; one concurrent run; bounded queue.
Quota/rate-limit errors end the run as **partial** with a recorded reason - no retry storms.

## Enabling live runs (needs explicit opt-in)

```
AGENTIC_MODE=live  AGENTIC_ALLOW_LIVE=1  AGENTIC_LLM_PROVIDER=gemini  AGENTIC_MODEL=gemini-2.5-flash
AGENTIC_API_TOKEN=<long random>          # required for approvals; protects every endpoint
AGENTIC_ENABLE_WRITES=1                  # only if you want proposals to be appliable
```
Without these the API answers 503. **No live LLM/Tavily call has been made while building this.**

## Verified vs not verified

Verified offline (tests): routing/vetoes, tool loop, evidence validation, limits, SSRF/size/redirect rules,
secret redaction, approval + resume across restarts (SQLite), write-gate rules, API contracts, Gemini
request/response conversion against SDK types.

**Not verified (needs approved free-tier calls):** real Gemini tool-calling quality and schema acceptance,
thinking-token usage under `max_output_tokens`, real Tavily result shapes, real-page extraction accuracy,
Supabase write path (`SupabaseCommitRepo` is only tested against a fake), RLS/FK-delete behaviour of the
live database. The offline evaluation uses **synthetic** `.example` pages and a **test-double policy**
(`PolicyLLM`); it measures the harness, not real-world accuracy or any productivity gain (no baseline exists).

## Known limitations

* Supabase REST is not transactional: a multi-operation proposal can fail part-way (recorded as `failed`
  with the operations done). A server-side RPC is the proper fix (additive migration, needs approval).
* Provenance is stored in `ai_insights` (existing columns) because `company_profile`/`contacts` have no
  per-field source columns; a dedicated evidence table is a possible additive migration.
* Merge execution is not implemented; only detection exists. A reviewed dry-run merge plan (FK impact for
  all 12 tables that reference `companies`) must come first.
* Lead-score weights are a provisional heuristic and need stakeholder review.
* Groq is not implemented; there is no provider fallback by design.

## Dry-run merge planner (read-only)

```bash
# offline, from the committed structural snapshot (no database access):
python -m agentic.merge_plan --snapshot tests/fixtures/duplicate_groups.json data/merge_plan
# refresh the snapshot from Supabase (SELECT only; contact names are hashed in memory, never stored):
python -m agentic.merge_sources tests/fixtures/duplicate_groups.json
```
Produces `merge_plan.json` + `merge_plan.md`: candidate pairs, category (`likely_duplicate`,
`same_name_different_sector`, `near_name_match`, `distinct_entities_likely`), evidence and rationale, the Golden
Record (unchanged legacy `score_company`), field-by-field decisions (legacy `MergeEngine` rule shown), dependent
rows for all 12 tables that reference `companies`, and the rows the legacy `ForeignKeyService` would lose.
The planner has no write path; every pair is `ready: false` until a transactional executor exists.

## Gemini smoke test (prepared, NOT run)

`python -m agentic.eval.gemini_smoke --dry-run --model <id>` prints the plan. A live run needs
`AGENTIC_SMOKE_APPROVED=1`, `GEMINI_API_KEY` in the process environment (`.env` is not read unless `--load-dotenv`),
`--model <exact id>` and `--i-have-checked-my-free-quota`; at most 4 requests (synthetic inputs, no Tavily, no
database, no retries, 1024 output tokens each). Token/request savings: see `docs/TOKEN_COST_ANALYSIS.md`.

# Agentic procurement-intelligence runtime - architecture, setup, limits

A LangGraph runtime with an LLM-driven supervisor and five tool-calling agents. It sits next to the legacy code
(`agents/`, `services/`, `workflow/`) and does not replace it. Python package: `agentic/`. Hand-over notes for the next
engineer/agent: `CLAUDE.md`. Provider choice and evidence: `docs/PROVIDER_DECISION.md`.

## Setup and tests (offline, free, no keys)

```bash
python -m venv .venv-agentic                    # git-ignored; do NOT install into the git-tracked .venv
.venv-agentic/Scripts/python -m pip install -r requirements-agentic.txt
.venv-agentic/Scripts/python -m pytest          # ~300 tests, ~2 min, no network
.venv-agentic/Scripts/python -m agentic.eval.graph_eval      # full-graph evaluation with simulated model mistakes
.venv-agentic/Scripts/python -m agentic.eval.cost_report     # token/request savings
.venv-agentic/Scripts/python -m agentic.eval.capacity        # free-tier capacity model
.venv-agentic/Scripts/python -m uvicorn api.app:app --port 8000   # API (agentic routes answer 503 until configured)
```

## How a run works

`intake -> supervisor -> agent_llm <-> agent_tools -> agent_done -> supervisor ... -> finalize -> [propose -> approval -> commit]`

* **intake** resolves the submitted *name* against Supabase (no ID needed): `found`, `ambiguous`, `not_found`. A company
  that is not in the database is researched identically; persistence is a separate, approval-gated proposal.
* **supervisor**: an LLM calls `route`; Python vetoes ineligible choices (identity must be confirmed before
  contact/procurement/supplier; no re-run without new information). Early `finish` is allowed.
* **agents** (company, contact, procurement, supplier, lead) are tool-calling loops run through the `GuardedExecutor`
  (permissions, argument validation, call/credit/token/time limits, repeat detection). They finish by calling their own
  `submit_*_findings` tool.
* **evidence validation (never weakened):** every claim must quote a snippet that exists verbatim in a document retrieved
  in this run; the value must be supported by the snippet; confidence is computed from evidence. Emails/phones only if
  printed literally.
* **correction cycle (default ON, bounded):** if a submission has rejected claims, the model gets one correction turn
  (`Limits.max_submit_attempts = 2`; `AGENTIC_RETRY_PARTIAL=0` disables). Accepted claims and their evidence are kept, no
  duplicates are created. Costs one extra LLM request per corrected submission.
* **honest run status:** `completed` only when nothing is unresolved. `partial` carries `status_reasons`:
  `limit:*`, `search_unavailable:*`, `no_verified_results_and_tool_failures`, `agent_not_finished:*`,
  `evidence_rejected_all:<agent>`, `evidence_rejected_some:<agent>:<fields>`. A legitimate "searched, found nothing"
  result stays `completed`. `failed` = technical failure. Rejected claims (agent, field, reasons, unverified value) are
  listed in `report.gaps.rejected_claims`.
* **writes:** agents have no write tools. Verified claims become proposals; applying needs a recorded human approval and
  passes the write gate (allowlists, no deletes, overwrite detection, tamper hash, idempotent, audited).

## Provider configuration (all offline-safe by default)

| Variable | Meaning |
|---|---|
| `AGENTIC_MODE=live` + `AGENTIC_ALLOW_LIVE=1` | required for live research; otherwise the API answers 503 |
| `AGENTIC_LLM_PROVIDER` | `gemini` (default in live mode) or `groq`; there is **no automatic fallback** |
| `AGENTIC_MODEL` | default `gemini-3-flash-preview`; **mandatory** for Groq (approved: `openai/gpt-oss-120b`, `openai/gpt-oss-20b`) |
| `AGENTIC_QUOTA=rpm=..,tpm=..,rpd=..,tpd=..` | **mandatory for Groq**; your console values; enables client-side pacing + rolling-24h budgets |
| `AGENTIC_WALL_CLOCK_S`, `AGENTIC_MAX_TOKENS`, `AGENTIC_TOOL_RESULT_CHARS` | raise/lower run budgets for slow, small-quota providers (Groq: 1200 / 80000 / 3000) |
| `AGENTIC_THINKING_LEVEL` | Gemini 3 thinking level (default `low`) |
| `AGENTIC_RETRY_PARTIAL=0` | disable the correction cycle for partial rejections |
| `AGENTIC_API_TOKEN` | protects every endpoint; approvals are disabled without it |
| `AGENTIC_ENABLE_WRITES=1` | creates the commit repository so approved proposals can be applied |
| `GEMINI_API_KEY` / `GROQ_API_KEY` | read from the process environment (or `.env` via `--load-dotenv` for the smoke scripts) |

## Free-tier limits you must plan around (your console values)

* Gemini 3 Flash: 5 RPM, 20 RPD, 250k TPM: under one full run per day, about two company-only runs per day.
* Groq gpt-oss-120b: 30 RPM, 1,000 RPD, 8,000 TPM, 200,000 TPD: about 2 to 3 full runs/day, at least 8 to 10 minutes per
  full run, tool results must be capped (~3,000 chars) so single requests stay under 8,000 tokens.
* Tavily free plan: 1,000 credits/month; the runtime spends ~5 per full run (monthly cap 300 by default).

## Known limitations

* No real-model research accuracy has been measured; live evidence is two 4-request smoke tests (n=1 each).
* Offline evaluations use synthetic `.example` pages and a deterministic decision-maker: they test the pipeline, not models.
* Supabase writes (`SupabaseCommitRepo`) are tested only against a fake; RLS / FK delete rules are unknown (run
  `agentic.fk_introspect.INTROSPECTION_SQL`). Merge execution does not exist (planner only; `docs/MERGE_EXECUTOR_DESIGN.md`).
* REST writes are not transactional; provenance is stored in `ai_insights` (existing columns).
* The Next.js dashboard is unchanged and not wired to the new API; `next build` was not run (it needs Google Fonts over the
  network); `tsc --noEmit` and ESLint pass.
* Lead-score weights are a provisional heuristic. Legacy bugs listed in `CLAUDE.md` section 10 remain.

## Related documents

`docs/PROVIDER_DECISION.md` (decision + evidence) | `docs/GRAPH_EVAL_REPORT.md` (strict criterion, simulated-mistake matrix) |
`docs/GROQ_OFFLINE_AUDIT.md` (Groq audit, capacity, console checklist) | `docs/TOKEN_COST_ANALYSIS.md` (efficiency switches) |
`docs/PROVIDER_COMPARISON_PLAN.md` (original staged plan) | `docs/MERGE_EXECUTOR_DESIGN.md` (design only; nothing applied).
Dry-run merge planner: `python -m agentic.merge_plan --snapshot tests/fixtures/duplicate_groups.json data/merge_plan`.
Smoke tests (guarded, need explicit approval): `agentic.eval.gemini_smoke`, `agentic.eval.groq_smoke`; each reports
`integration_passed` and `evidence_quality_passed` separately.

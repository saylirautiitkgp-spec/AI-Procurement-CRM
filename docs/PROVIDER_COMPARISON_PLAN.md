# Provider comparison plan: Gemini vs Groq

**Status: plan only. No live request has been made to either provider. No provider is recommended yet.**
Gemini (`agentic/llm/gemini.py`) stays the working default. Groq (`agentic/llm/groq.py`) exists for offline
evaluation only. There is no automatic fallback; the provider is chosen by `AGENTIC_LLM_PROVIDER`.

## 1. Evidence so far (documentation, with confidence)

| Topic | Gemini (2.5 Flash) | Groq | Confidence |
|---|---|---|---|
| Free tier exists | Yes, per Google pricing page | Yes (free plan) | Official docs |
| Free limits (RPM/TPM/RPD) | **Not published**; shown per project in AI Studio | Official table is labelled "Developer plan"; third-party sites quote ~30 RPM / 1,000 RPD / 8K TPM / 200K TPD for gpt-oss and Qwen | Gemini: unknown. Groq: **unconfirmed** |
| Tool calling | Native function calling (AUTO/ANY/NONE/VALIDATED, parallel + compositional) | Tool use on gpt-oss-120b/20b, Llama 3.3 70B, Llama 3.1 8B, Qwen3.8-27B; gpt-oss has **no parallel tool calls** | Official docs; reliability unmeasured |
| Model status | `gemini-2.5-flash` listed for the key, no announced shutdown | gpt-oss and Llama listed as Production; Qwen and MiniMax are **Preview** (may vanish). Llama models are marked "enterprise pricing", so they may not be on the free plan | Official docs; free-plan eligibility unverified |
| Data use on free tier | Google states free-tier content **is used to improve its products** | Inference data not retained by default; abuse-monitoring logs up to 30 days unless Zero Data Retention is enabled; no explicit "not used for training" statement found for free users | Official docs |
| Integration effort | Done and unit-tested | Done offline (adapter + error mapping, tests) | Code |
| Structured output | Flattened JSON schemas accepted by SDK types; live acceptance unverified | JSON mode listed; strict schema enforcement unverified | Unverified |

## 2. Measured offline: how much quota one run needs (real prompts, mock web)

`PromptSizeProbe` measures the real prompt (system prompt + tool schemas + conversation) of every request. For a
full five-agent run on mock pages: **about 25 LLM requests and about 38,000 to 43,000 input tokens**, largest request about
3,000 tokens, mean about 1,700. See `docs/TOKEN_COST_ANALYSIS.md` for the breakdown and the savings now built in
(about 15 to 17% fewer tokens with the safe defaults; verified output-identical offline). Synthetic pages are shorter than real ones (a fetched page can add about 1,500
tokens per call), so treat these as a lower bound; a company-only run is roughly a quarter of that.

What that implies if the *third-party* Groq numbers are right (8K TPM, 200K TPD): a full run needs 15+ minutes
because of the per-minute cap, and only about 3 to 4 full runs fit in a day. Gemini's real allowance is unknown
until you read your AI Studio page. These are estimates, not findings.

## 3. Test procedure (same cases for both providers, mock tools, no Tavily, no database)

| Stage | What | Requests | Needs approval |
|---|---|---|---|
| 0 | Offline conformance of each adapter (already in CI) | 0 | no |
| 1 | `gemini_smoke` 4-request schema/tool-call check (`run_smoke` is provider-neutral) | 4 | **yes, per provider** |
| 2 | `provider_eval` full-graph run on the SAME 3 cases for each provider: #101 Pixxel (standard full loop), #187 Robonic (empty search: must adapt), #251 Gannon Dunkerley (prompt injection) | about 22 per case, capped (default 60) | **yes, per provider** |
| 3 | Only if 1 and 2 are good: 1 real company with real Tavily (about 5 credits) | about 25 + 5 credits | **yes** |

Never run two providers "in parallel" against one task; run each separately and compare logs.

### Metrics recorded (all from the run trace and provider usage fields)

| Criterion | Metric |
|---|---|
| Tool-calling reliability | `invalid_tool_call_rate` (unknown tool, bad arguments, wrong agent, repeated call); route vetoes |
| Adaptation | does the query change after an empty result (existing `empty_search` case) |
| Correct stopping | agent runs that end with a normal submit vs limit/stall |
| Structured output | submit rejections; JSON parse failures (Groq `__unparseable_arguments__`, `tool_use_failed`) |
| Evidence grounding | `fabrication_attempt_rate` = rejected claims whose reason is a missing URL/snippet/unsupported value; **accepted** fabricated claims must always be 0 |
| Injection resistance | `injection` case: no fetch/search toward attacker text, no leak |
| Latency | mean and max per-request latency, wall time per case (full loop) |
| Consumption | requests, input/output tokens per case, vs the quota you confirm |
| Errors | 429 (rate vs daily quota), 5xx, 400; retries used |
| Reproducibility | same case twice at temperature 0.2: do tool choices match? |

### Decision rules (fixed before looking at results)

A provider is **eligible** only if: no accepted fabricated claim; `invalid_tool_call_rate` <= 10%;
>= 80% of agent runs end normally; schema acceptance with no HTTP 400 on our tool set; and the confirmed free quota
allows at least 5 full runs per day. Among eligible providers prefer, in order: (1) fewer fabrication attempts,
(2) data-handling terms you are comfortable with for scraped company/contact text, (3) full-loop latency,
(4) tool-call accuracy, (5) integration effort. If neither is eligible or the evidence is close, report the
uncertainty and run one more targeted stage rather than guess. Do not switch providers automatically.

## 4. Information I need from you before any live stage

1. Gemini: the RPM, TPM and RPD shown for `gemini-2.5-flash` in AI Studio (the "Rate limit" page).
2. Groq (only if you want it evaluated): your console limits page for the exact model, and which model to use
   (`AGENTIC_MODEL` is mandatory for Groq; no default is assumed).
3. Whether Google's free-tier data-use term is acceptable for prompts containing scraped company and contact text.
4. Written approval for the specific stage and request cap.

## 5. How to run (when approved)

```bash
# Stage 1 (Gemini)
AGENTIC_SMOKE_APPROVED=1 python -m agentic.eval.gemini_smoke --i-have-checked-my-free-quota data/smoke_gemini.json
# Stage 2
AGENTIC_PROVIDER_EVAL_APPROVED=1 AGENTIC_MODEL=<model> AGENTIC_PROVIDER_EVAL_MAX_REQUESTS=60 \
  python -m agentic.eval.provider_eval gemini --i-have-checked-my-free-quota data/eval_gemini.json
```
Both refuse to run without the approval variables and the quota acknowledgement.

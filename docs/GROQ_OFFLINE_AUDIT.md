# Groq offline compatibility audit (no live Groq request has been made)

Status date: 2026-10-11. Gemini stays the default provider; nothing here switches providers or adds fallback.
Evidence is kept in four separate classes: **D** documented by the provider, **O** measured offline here,
**E** estimate/assumption, **U** unverified until a live test.

## 1. Models and limits

| Item | Value | Class |
|---|---|---|
| Candidate 1 | `openai/gpt-oss-120b` (131,072 ctx, 65,536 max output; tool use, JSON object + JSON-schema modes, reasoning) | D |
| Candidate 2 | `openai/gpt-oss-20b` (same limits and capabilities) | D |
| `openai/gpt-oss-safeguard-20b` | 3 RPM / 2,000 TPM: unusable for this loop; excluded | D (user screenshot) |
| Limits shown in your console for both gpt-oss models | 30 RPM, 1,000 RPD, 8,000 TPM, 200,000 TPD | user-reported |
| **Do those limits apply to the Free Plan?** | Groq's limits page has separate "Free Plan" and "Developer Plan" tabs, and the figures I could read are labelled **Developer base limits**. They equal your screenshot, but the page does not say they are the Free Plan's. Confirm with the checklist below. | D + U |
| Does Free Plan usage ever bill? | The docs I read do not say. Confirm on the console (checklist). | U |
| How TPM counts tokens (input+output? reserved `max_completion_tokens`?) | not stated in the docs | U (planned for conservatively) |
| Single request larger than TPM | HTTP 413 in practice; the docs only mention 429 | U |
| Structured outputs | `response_format` JSON-schema works on gpt-oss (strict and best-effort) BUT **cannot be combined with tool use in one request**. Our final structured output is a *function call* validated by our own pydantic schemas, so `response_format` is not used. | D |
| Parallel tool calls | gpt-oss: not supported (so `batch_hint` is useless there; the adapter never sends `parallel_tool_calls`) | D |
| Reasoning | `reasoning_effort` low/medium/high; reasoning text returned in `message.reasoning`; `include_reasoning=false` omits it. Whether reasoning tokens count toward TPM/TPD and whether they must be echoed back in multi-turn tool calls is not stated | D / U |

## 2. Adapter issues found and fixed (all verified offline against the real `groq` 1.7.0 SDK types)

| # | Issue | Fix | Test |
|---|---|---|---|
| 1 | Nothing prevented a request from carrying `service_tier` (can select non-default, possibly billed tiers), `parallel_tool_calls`, `response_format`, `stream` | `build_request` never sets them and asserts it | `test_request_never_sends_paid_tier...` |
| 2 | gpt-oss reasoning could inflate completion tokens (they count against TPM/TPD) | `reasoning_effort="low"`, `include_reasoning=False` for gpt-oss only (parameter names confirmed in the SDK signature) | same |
| 3 | Billing/payment errors were not distinguishable and would have looked like generic failures | new `billing` error kind (HTTP 402, or 400/401/403/429 with billing words), never retried, run ends partial (`llm_billing`) | `test_billing_and_quota_errors_are_never_retried...` |
| 4 | HTTP 413 and connection errors were generic | 413 -> explicit "request too large for TPM"; connection/timeouts -> transient | `test_real_sdk_exceptions_...` |
| 5 | No protection against an 8,000-TPM limit: a 429 is a wasted request | `PacedLLM`: waits before sending to stay under 80% of RPM/TPM, refuses locally when one request exceeds TPM, rolling-24 h persistent RPD/TPD budgets, limits supplied by the operator via `AGENTIC_QUOTA` (none assumed) | `test_requests_that_would_exceed...`, daily-budget and persistence tests |
| 6 | Realistic pages make single requests ~7.4k tokens (+1,024 output reservation) which exceeds 8,000 | `AGENTIC_TOOL_RESULT_CHARS` (existing setting) must be lowered for Groq; see section 4 | `test_measured_capacity_shows...` |
| 7 | Default per-run limits (180 s wall clock, 60k tokens) cannot fit a paced Groq run | new env overrides `AGENTIC_WALL_CLOCK_S`, `AGENTIC_MAX_TOKENS` | `test_slow_small_quota_providers...` |
| 8 | Service allowed Groq without a model or quota | Groq now requires `AGENTIC_MODEL` and `AGENTIC_QUOTA` | `test_service_selects_groq_only_with_explicit...` |

Checked and found compatible (no change needed): tool-call selection and argument parsing (including unparseable
argument strings, which become `invalid_arguments`), tool-result continuation (every assistant tool-call id is
answered exactly once, asserted by the wire emulator), parallel calls answered with the right ids, supervisor
routing through `route`, evidence validation (identical output to the direct run), error mapping with real SDK
exception classes, message metadata (ids preserved; reasoning deliberately dropped).

## 3. Fair offline comparison (O): same three cases, identical mocked tool results

Cases: #101 Pixxel (full loop), #187 Robonic (empty search), #251 Gannon Dunkerley (prompt injection). The same
deterministic decision-maker answers behind a wire-level emulator of each SDK, so any difference would be an
adapter bug. Reproduce: `python -m agentic.eval.adapter_conformance`.

| Adapter (model) | wire violations | output identical to direct run | requests (3 cases) | est. prompt tokens, small / realistic pages | largest request |
|---|---|---|---|---|---|
| Groq `openai/gpt-oss-120b` | 0 | yes | 50 | 82,098 / 125,710 | 3,217 / 8,245 |
| Groq `openai/gpt-oss-20b` | 0 | yes | 50 | 82,098 / 125,710 | 3,217 / 8,245 |
| Gemini `gemini-3-flash-preview` | 0 | yes | 50 | 78,383 / 121,988 | 3,089 / 8,117 |

Same requests and same research output, so offline there is **no adapter-level reason to prefer either**.
Payloads are within ~5% of each other (Gemini slightly smaller; both emulators now measure the serialized request
symmetrically - an earlier version of this table wrongly showed Gemini larger because of a measurement artifact).
The two Groq models are indistinguishable offline. Every real difference (tool-choice accuracy, adaptation to
empty/conflicting results, injection resistance of the *model*, latency, real token use incl. reasoning) is
**unmeasurable without live runs**. No winner is selected on this evidence; payload size is not a selection criterion.

## 4. Capacity recalculation (O for tokens/requests, E for output tokens, user-reported limits)

Limits used: 30 RPM / 1,000 RPD / 8,000 TPM / 200,000 TPD, planning headroom 80% (cap: 160,000 tokens/day,
6,400 tokens/minute). Prompt tokens are measured through the Groq adapter with realistic (6k-character) pages;
output tokens per request are an **assumption** (150 / 400 / 1,000) because reasoning-token use is unknown.
Reproduce: `python -m agentic.eval.capacity`.

| Scenario | Run | LLM requests | prompt tokens | largest request | fits 8K TPM incl. 1,024 reserve | full runs/day by TPD (150 / 400 / 1,000 out tokens per request) | min minutes/run by TPM @400 |
|---|---|---|---|---|---|---|---|
| A. current defaults (6,000-char tool results) | full five-agent | 21 | 54,554 | 7,453 | **NO** | 2.77 / 2.54 / 2.12 | 9.8 |
| A | company only | 8 | 29,800 | 7,394 | **NO** | 5.16 / 4.85 / 4.23 | 5.2 |
| B. tool results capped at 3,000 chars | full five-agent | 21 | 47,648 | 5,754 | yes | 3.15 / 2.85 / 2.33 | 8.8 |
| B | company only | 8 | 24,677 | 5,698 | yes | 6.18 / 5.74 / 4.90 | 4.4 |
| C. B + compact_schemas | full five-agent | 21 | 46,383 | 5,674 | yes | 3.23 / 2.92 / 2.37 | 8.6 |
| D. C + 2,000-char tool results | full five-agent | 21 | 43,383 | 4,924 | yes | 3.44 / 3.09 / 2.49 | 8.1 |

Reading: your estimate of "about three full runs a day" holds only with tool results capped around 3,000 chars and
modest reasoning output (~2.1 to 3.4 runs/day across scenarios). Requests per day (1,000) is never the constraint;
**tokens per day and tokens per minute are**. One full run takes at least 8 to 10 minutes (TPM), so
`AGENTIC_WALL_CLOCK_S` must be raised (e.g. 1200) and `AGENTIC_MAX_TOKENS` to about 80,000. With default 6,000-char
tool results a single late request can be refused locally (it would exceed 8,000 TPM) so the cap is mandatory.
Capping tool results raises the chance a fact sits beyond the first chunk (the model can page with `offset`, at the
price of extra requests): **this trade-off is unmeasured**. For comparison, Gemini 3 Flash on your free tier
(20 RPD, 80% headroom) allows 16 / 21 = **0.76 full runs/day** but **about 2 company-only runs/day**.
A company-only run on Groq is ~5 runs/day at 3,000-char caps.

## 5. Unresolved uncertainties (all need a live test or your console)

1. Do the displayed limits apply to your Free Plan project (not only Developer)? 2. Can Free Plan usage ever bill?
3. Does `tool_choice="required"` work with gpt-oss on Groq (forced routing/submit steps use it)?
4. `tool_use_failed` rate and tool-choice quality on our 14-tool schemas. 5. How TPM counts reserved output tokens.
6. Real reasoning-token consumption at `reasoning_effort="low"`. 7. Whether reasoning must be echoed in multi-turn
tool calls (the adapter drops it). 8. Free-tier data-use terms for Groq (docs: no retention by default; 30-day abuse
logs unless Zero Data Retention; no explicit "not used for training" statement found).

## 6. Checklist for you (console only - never share a key, never open or paste `.env`)

1. console.groq.com -> Settings -> **Billing / Plan**: the plan label reads **Free**; no payment method on file; no
   credits purchased; no auto-recharge or spend limit configured.
2. Settings -> **Limits**: open the **Free Plan** view; for `openai/gpt-oss-120b` and `openai/gpt-oss-20b` confirm
   RPM, RPD, TPM, TPD equal 30 / 1,000 / 8,000 / 200,000 (or write down the real numbers).
3. Confirm both model IDs appear in the console model list / Playground and respond to a manual "hello" typed by you
   (optional, your own usage, not ours).
4. Usage page: note today's usage (should be zero) so we can compare after the test.
5. Confirm your project/organisation name is the one whose API key is in your `.env` (the key lives only in `.env`
   or your shell environment; I never need to see it).
6. Tell me: the confirmed four numbers, which model first (default `openai/gpt-oss-120b`), and "approved".

## 7. Proposed smallest Groq smoke test (NOT RUN; needs your explicit approval after the checklist)

Same four steps as the Gemini smoke test; synthetic inputs and mock tools only; no Tavily, no Supabase, no real data;
**at most 4 LLM requests, no retries**; client-side pacing; the first provider error of any kind (quota, rate limit,
billing, 4xx/5xx) ends the test; never enables billing or changes a plan; `service_tier` is never sent.

```
AGENTIC_GROQ_SMOKE_APPROVED=1 python -m agentic.eval.groq_smoke --model openai/gpt-oss-120b \
  --quota "rpm=30,tpm=8000,rpd=1000,tpd=200000" --i-have-confirmed-free-plan --load-dotenv data/groq_smoke.json
```
(`--load-dotenv` loads the key inside the process without displaying it; omit it if `GROQ_API_KEY` is already set in
your shell. Replace the quota numbers with what your console actually shows.) Dry run, no request:
`python -m agentic.eval.groq_smoke --dry-run --model openai/gpt-oss-120b`.

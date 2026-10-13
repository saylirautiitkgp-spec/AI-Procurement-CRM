# Provider decision (2026-10-11): keep Gemini 3 Flash as default; Groq is configurable but NOT proven better

No new live call was made for this decision. Evidence classes: **[LIVE]** measured in the two approved 4-request smoke
tests (synthetic inputs, mock tools, n = 1 per provider); **[SIM]** offline simulation (real graph + real adapters over
wire-level SDK emulators, deterministic decision-maker, simulated model mistakes); **[EST]** estimates; **[DOC]** provider
documentation / your console.

## Evidence

| Criterion | Gemini `gemini-3-flash-preview` | Groq `openai/gpt-oss-120b` | Class |
|---|---|---|---|
| Tool-call correctness and compatibility | forced `route`, `web_search`, `fetch_page`, forced submit all parsed and valid | same four steps all valid (`tool_choice="required"` accepted) | LIVE (n=1 each) |
| Adapter protocol fidelity | 0 wire violations, output identical to the direct run | 0 wire violations, identical | SIM |
| Structured output validity | submit args validated against our schema | validated | LIVE |
| **Evidence acceptance / quotation accuracy** | **5/5 accepted; strict criterion PASS** | **4/5 accepted; strict FAIL** (URL used as the quote for `official_website`) | LIVE (n=1 each) |
| What the pipeline does when a model makes such a mistake | identical for both: rejected for the right reason, never accepted; corrected once (default), then reported `partial` if unresolved | same | SIM |
| Adaptation after empty / incomplete search | not tested live | not tested live. The pipeline reformulates the query in the empty-search case, but the decision-maker there is a test double | SIM only |
| Prompt-injection resistance | not tested live | not tested live. Pipeline property verified: page text reaches either model only inside `<untrusted_web_content>`; no fetch/search toward attacker text; no leak | SIM (harness, not model) |
| Latency | 1.97 / 1.88 / 1.50 / 3.02 s per request (8.4 s for 4) | 1.66 / 0.50 / 0.70 / 1.39 s (4.25 s for 4); provider-side time 0.14 to 1.06 s | LIVE |
| Tokens (4 requests) | 4,779 in / 479 out (thinking included) | 4,629 in / 659 out (74 reasoning) | LIVE |
| Full-loop tokens | ~21 to 22 requests, ~47k to 55k prompt tokens per full five-agent run; company-only ~8 requests, ~14k to 17k | same loop, payload within ~5% | SIM + EST |
| Free-tier capacity | 5 RPM / 20 RPD / 250k TPM: **0.76 full runs/day, about 2 company-only runs/day** | 30 RPM / 1,000 RPD / 8,000 TPM / 200,000 TPD: **about 2.1 to 3.4 full runs/day**, >= 8 to 10 min per full run, about 5 to 6 company-only runs/day; needs tool results capped at 3,000 chars | DOC + EST |
| Error handling | rate-limit / quota / invalid / transient mapped; bounded retry | same plus `billing` kind, client-side pacing, rolling-24h budgets (all enforced and tested offline) | SIM / unit tests |
| Data handling on the free tier | Google may use free-tier content to improve its products | no inference-data retention by default; 30-day abuse logs unless Zero Data Retention; no explicit "not used for training" statement | DOC |

## Decision

**Keep Gemini 3 Flash as the default provider.** Rule applied: Groq is preferred only if existing verified evidence
supports the switch. It does not: on the one metric that measures research quality, evidence acceptance, the single live
sample favours Gemini (5/5 vs 4/5), and n=1 is far too small to call that a difference either way. Groq's advantages
(capacity about 3x more full runs per day, lower latency, looser data terms) are real but are not evidence of
correct research. **Groq has not been proven better, and has not been proven worse.**

Practical consequence you should know: with 20 requests/day Gemini 3 Flash cannot complete even one full five-agent run
per day on the free tier. It can do about two company-only runs a day. For volume today the configured alternative is Groq
(`AGENTIC_LLM_PROVIDER=groq`, `AGENTIC_MODEL=openai/gpt-oss-120b`, `AGENTIC_QUOTA=...`); that is a deliberate operator
choice with unproven evidence quality, not an automatic fallback (none exists).

## What evidence is still missing (smallest set that would justify a switch either way)

1. Many more live company-only runs per provider (not n=1), judged with the strict criterion: e.g. 10 each. Quota reality:
   about 8 requests and 17k tokens each, so Gemini needs ~5 days at 2/day and Groq fits ~10/day.
2. One live prompt-injection case per provider (the smoke tests contain none).
3. One live empty-search case per provider (does the real model reformulate?).
4. One live full five-agent loop on Groq under pacing to observe real reasoning-token use and the 8,000 TPM behaviour.
5. How each model responds to the correction feedback (does it fix the claim?). The pipeline is tested; the models are not.

None of these is run. Each needs your explicit approval.

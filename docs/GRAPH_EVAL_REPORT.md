# Offline full-graph evaluation and stricter evidence criterion

> UPDATE (same day, finalisation pass): proposals A and B below were ADOPTED after the offline tests showed them bounded
> and safe. Partial-rejection correction is now ON by default (`AGENTIC_RETRY_PARTIAL=0` disables); runs with unresolved
> rejected claims are `partial` with `status_reasons`. In the tables, "current" = the previous behaviour (correction
> disabled), "proposed" = the new default. The smoke harness now reports `integration_passed` and `evidence_quality_passed`
> separately. See `docs/PROVIDER_DECISION.md`.

Date: 2026-10-11. **No live Groq/Gemini call, no Tavily/Supabase access, no database access was made for this work.**
Gemini remains the default; no provider fallback exists. Reproduce: `python -m agentic.eval.graph_eval`.

## 1. Why the Groq smoke test "passed" with 4 of 5 claims accepted

The harness check was `accepted >= 1` (see `quoted_snippets_pass_evidence_validation` in `gemini_smoke.py`). I wrote it
deliberately lenient to prove the validator plumbing works; it was never an evidence-quality gate. The rejected claim
(`official_website`, quote = the URL string, reason `snippet_not_found_in_document`) therefore did not fail the test.
Separately, inspecting the graph showed what would happen in a real run:

1. **Partial rejections are not corrected.** The graph sends a submission back to the model only when *every* claim was
   rejected (`not pr.accepted`). With 4/5 accepted the agent finishes, the feedback is never seen by the model, and the
   rejected claim is silently dropped (identity is still established through `identity_website`, so the run continues).
2. **Run status ignores rejections.** If an agent submits and every claim is rejected, the run is still `completed`.
   (Not changed; proposal B below.)

## 2. Stricter criterion (evaluation tooling only; production acceptance rules unchanged)

`agentic/eval/quality.py::strict_assessment` judges a submission on evidence quality. It passes only if ALL hold:
1. every submitted claim was accepted (zero rejections, at least one accepted);
2. no rejection for an unquoted/fabricated reason (`snippet_not_found_in_document`, `url_not_retrieved_in_this_run`,
   `value_not_supported_by_snippet`, non-literal email/phone, bad snippet length);
3. the official website is established with evidence - validator-verified identity **or** an accepted `official_website`
   claim (Gemini put the site in `identity_website` only; requiring a duplicate claim would be unfair);
4. at least two distinct fields accepted.
`gemini_smoke.run_smoke` (also used by the Groq smoke) now reports **both** the legacy `all_passed` and the new
`strict` block / `strict_passed`. `validate_report` additionally checks the final structured report (required keys, final
status, every claim has a complete evidence record whose URL is listed in `sources`, confidence in [0,1], database values
labelled `database_unverified`).

Re-assessment of the two saved live smoke results (offline, synthetic data): **Gemini 3 Flash 5/5 accepted, strict PASS;
Groq gpt-oss-120b 4/5 accepted, strict FAIL** (`official_website`, `snippet_not_found_in_document`).

## 3. What the offline matrix is (and is not)

The real graph runs company-only (`"company profile"`) on 5 synthetic cases - Pixxel (standard), Robonic (empty search),
Gannon Dunkerley (prompt injection), Unmanned Life (ambiguous name), Huawei Digital Power India (conflicting sources) -
through the real Groq (`openai/gpt-oss-120b`) and Gemini (`gemini-3-flash-preview`) adapters via wire-level SDK emulators,
with identical mocked web evidence. A deterministic decision-maker answers; **simulated personas then corrupt its final
submission** in specific ways, including the exact mistake Groq made live. The matrix therefore measures how the
**pipeline** reacts to model mistakes. It does **not** measure how often a real model makes them. 90 runs per setting.

## 4. Results (offline; prompt tokens are estimates = chars/4, output tokens are not measured offline)

Aggregates over all 180 runs (both settings, both adapters): **0 claims accepted with a quote that is not in the evidence;
0 invalid final reports; 0 wire-protocol violations; behaviour identical through the Groq and Gemini adapters for every
persona; the empty-search case adapted (reformulated the query)**; max invalid-tool-call rate 16.7% (only the deliberate
schema-violation persona).

Groq adapter, per-persona means over the cases that submitted claims (Robonic's empty search submits none):

| Persona (simulated mistake) | Setting | first-submit accepted | final accepted | correction cycles/run | residual rejected fields | requests/run | est. prompt tokens/run | strict-ok runs |
|---|---|---|---|---|---|---|---|---|
| none (exact) | current | 100% | 100% | 0 | 0 | 7.6 | 14,037 | 4/4 |
| **URL used as the quote (seen live)** | **current** | 84% | 83% | **0** | **4** | 7.6 | 14,043 | **0/4** |
| URL used as the quote | proposed | 84% | **100%** | 0.8 | **0** | **8.4** | 16,898 | **4/4** |
| URL as quote, model repeats the mistake | proposed | 84% | 83% | 0.8 | 4 | 8.4 | 16,898 | 0/4 (stopped by the 2-attempt cap) |
| paraphrased quote, fixes when told | current -> proposed | 84% | 83% -> 100% | 0 -> 0.8 | 4 -> 0 | 7.6 -> 8.4 | 14,043 -> 16,914 | 0/4 -> 4/4 |
| value not in its snippet, fixes | current -> proposed | 84% | 83% -> 100% | 0 -> 0.8 | 4 -> 0 | 7.6 -> 8.4 | 14,043 -> 16,906 | 0/4 -> 4/4 |
| fabricated evidence URL, fixes | current -> proposed | 84% | 83%* -> 100% | 0 -> 0.8 | 3 -> 0 | 7.6 -> 8.4 | 14,043 -> 16,905 | 1/4 -> 4/4 |
| schema violation (empty evidence), fixes | both | 0% | 100% | 0.8 | 0 | 8.4 | 16,846 | 4/4 |
| all three evidence mistakes, fixes | current -> proposed | 52% | 54% -> 100% | 0 -> 0.8 | 11 -> 0 | 7.6 -> 8.4 | 14,043 -> 16,951 | 0/4 -> 4/4 |

(*one case had a second, valid claim for the same field, so the field still counted as accepted.)

Findings:
* **The validator itself is sound:** every simulated mistake was rejected for the correct reason and none was ever
  accepted (also confirmed by mutation tests that fail when the quote check is removed).
* **The weak point is what happens afterwards:** under today's rules a partially rejected submission is never corrected, so
  a model that makes one evidence mistake permanently loses that claim (here: the `official_website` section of the report).
* **Schema violations already get a correction cycle** (the tool error is returned to the model and it can retry).
* **Cost of the proposed fix, measured offline:** exactly +1 LLM request and about +3.6k estimated prompt tokens per
  corrected submission (+10% requests, +20% prompt tokens on a company-only run; zero when nothing is rejected). The cap of
  two submissions means a stubborn model cannot loop.

## 5. Proposals - NOT applied to production defaults (need your decision)

* **A. Correct partial rejections** (`Settings.retry_partial_rejections`, implemented, **default off = unchanged
  behaviour**). Consequence: +1 request and ~3.6k prompt tokens per corrected submission, which matters under Groq's
  8,000 TPM / 200k TPD (one correction is ~2% of the 160k planning token cap per day) and especially under
  Gemini 3 Flash's 20 requests/day (one extra request is 5% of the daily allowance). Benefit: no silently lost claims; recovers the website claim in the simulation.
  Validation rules are untouched.
* **B. Report `partial` when an agent's every claim was rejected** (not implemented; currently reported `completed`).
  Consequence: some runs that look complete would be labelled partial - more honest, may surface more "partial" in the tracker.
* **C. Use the strict criterion for any further live smoke/evaluation gating** (implemented in the harness alongside the
  legacy check; production code does not use it).
* Already applied (wording only): correction feedback now says a quote must be copied word-for-word, that a URL or summary
  is not a quote, which URLs are citable, and that only corrected items need re-submitting.

## 6. Fair comparison with the existing live baselines (n = 1 each; NOT a provider ranking)

| | Gemini 3 Flash (live smoke) | Groq gpt-oss-120b (live smoke) |
|---|---|---|
| Evidence-validator acceptance | 5/5 | 4/5 |
| Strict criterion | pass | fail (URL used as a quote) |
| Requests | 4 | 4 |
| Input / output tokens (measured) | 4,779 / 479 | 4,629 / 659 (74 reasoning) |
| Request latencies (measured) | 1.97, 1.88, 1.50, 3.02 s | 1.66, 0.50, 0.70, 1.39 s (total 4.25 s) |
| Free-tier headroom | 20 requests/day | 8,000 tokens/min, 200k tokens/day |
| Offline pipeline behaviour (same mocked evidence, simulated mistakes) | identical to Groq | identical to Gemini |

Measured = from the two live smoke runs. Offline = pipeline simulation. Estimates = prompt tokens (chars/4). **Cannot be
concluded:** which model makes fewer evidence mistakes over many runs, real latency over a full loop, injection resistance of
either model, real output/reasoning-token use in longer loops. Those need an approved live evaluation with many more samples.

## 7. Suggested next (needs your explicit approval; nothing was run)

Decide A and B. If A is approved to enable for live testing, the smallest informative live step is the existing 4-request
smoke test repeated on `openai/gpt-oss-120b` judged with `strict_passed` (and a second attempt on Gemini only if you want a
matched pair); a company-only full-graph live run (about 8 requests, about 14k to 17k prompt tokens) is the next size up and
would be the first real measurement of correction behaviour.

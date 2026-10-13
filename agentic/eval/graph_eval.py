"""Offline full-graph evaluation (company-only) with SIMULATED model failure modes.

WHAT THIS IS: the whole LangGraph pipeline - supervisor, company agent, guarded tools, evidence validator, correction
cycle, report - run through the real Groq and Gemini adapters (wire emulators) on identical synthetic inputs and
identical mocked web evidence. A deterministic decision-maker answers; "personas" then corrupt its final submission
in specific, realistic ways (e.g. the URL used as a quote - the exact mistake gpt-oss-120b made live).

WHAT THIS IS NOT: a measurement of how often a real model makes these mistakes. It measures how the PIPELINE reacts
when a model does. Provider quality can only be measured with live runs. Output tokens are not measured offline;
prompt tokens are estimates (chars/4) of the real request payloads.

    python -m agentic.eval.graph_eval
"""

from __future__ import annotations

import copy
import json
import statistics
from collections import Counter

from agentic import evidence as ev
from agentic.config import Limits
from agentic.eval.adapter_conformance import make_client
from agentic.eval.policy import PolicyLLM
from agentic.eval.provider_eval import metrics_from_run
from agentic.eval.quality import UNQUOTED, validate_report
from agentic.eval.run_offline import evaluate_run
from agentic.eval.synthetic import make_world

SUBMIT = "submit_company_findings"
CASE_IDS = (101, 187, 251, 205, 414)       # Pixxel, Robonic (empty search), Gannon (injection), Unmanned Life (ambiguous), Huawei DP (conflicting)
PROVIDERS = (("gemini", "gemini-3-flash-preview"), ("groq", "openai/gpt-oss-120b"))
OBJECTIVE = "company profile"


def cases(path: str = "tests/fixtures/first20.json") -> list[dict]:
    by_id = {c["company_id"]: c for c in json.load(open(path, encoding="utf-8"))["companies"]}
    return [{"company": by_id[i], "variant": by_id[i]["web_variant"]} for i in CASE_IDS]


# ------------------------------------------------------------------------------------------- personas
def _claim(args, field):
    return next((c for c in args.get("claims", []) if c["field"] == field), None)


def corrupt(persona: str, args: dict) -> dict:
    a = copy.deepcopy(args)
    if persona == "url_as_quote":                       # observed live with gpt-oss-120b
        if c := _claim(a, "official_website"):
            c["evidence"][0]["snippet"] = c["evidence"][0]["url"]
    elif persona == "paraphrased_quote":
        if c := _claim(a, "headquarters"):
            c["evidence"][0]["snippet"] = c["evidence"][0]["snippet"].replace(" is a ", " operates as a ") + " (summary)"
    elif persona == "wrong_value":
        if c := _claim(a, "headquarters"):
            c["value"] = "Atlantis"
    elif persona == "fabricated_url":
        if c := _claim(a, "employee_count"):
            c["evidence"][0]["url"] = "https://made-up.example/about"
    elif persona == "missing_evidence":                 # schema violation (evidence is required)
        if c := _claim(a, "founded_year"):
            c["evidence"] = []
    elif persona == "mixed":
        for p in ("url_as_quote", "wrong_value", "fabricated_url"):
            a = corrupt(p, a)
    elif persona != "exact":
        raise ValueError(persona)
    return a


class PersonaLLM:
    """Wraps the decision-maker. On the FIRST submission it applies the persona's mistake. If the pipeline sends
    feedback and `fixes` is True it re-submits the clean claims; if False it repeats the same mistake (stubborn)."""

    name, model = "persona", "persona"

    def __init__(self, inner, persona: str, fixes: bool):
        self.inner, self.persona, self.fixes = inner, persona, fixes
        self.clean: dict | None = None
        self.submissions = 0

    def generate(self, **kw):
        resp = self.inner.generate(**kw)
        for call in resp.tool_calls:
            if call.name != SUBMIT:
                continue
            self.submissions += 1
            feedback_seen = any(m["role"] == "tool" and m.get("name") == SUBMIT for m in kw["messages"])
            if not feedback_seen or self.clean is None:
                self.clean = copy.deepcopy(call.args)
                call.args = corrupt(self.persona, call.args)
            else:
                call.args = copy.deepcopy(self.clean) if self.fixes else corrupt(self.persona, self.clean)
        return resp


PERSONAS = {                    # name -> (persona key, fixes after feedback)
    "exact (no mistakes)": ("exact", True),
    "url_as_quote, fixes when told": ("url_as_quote", True),
    "url_as_quote, stubborn": ("url_as_quote", False),
    "paraphrased_quote, fixes": ("paraphrased_quote", True),
    "wrong_value, fixes": ("wrong_value", True),
    "fabricated_url, fixes": ("fabricated_url", True),
    "missing_evidence (schema), fixes": ("missing_evidence", True),
    "mixed mistakes, fixes": ("mixed", True),
    "mixed mistakes, stubborn": ("mixed", False),
}


# ------------------------------------------------------------------------------------------- one run
def run_one(provider: str, model: str, case: dict, persona_key: str, fixes: bool, retry_partial: bool) -> dict:
    decider = PersonaLLM(PolicyLLM(), persona_key, fixes)
    client, wire = make_client(provider, model)
    wire.decider = decider                                     # the wire emulator asks the persona-wrapped decision-maker
    limits = Limits(max_total_tokens_run=1_000_000, max_wall_clock_s=3600)
    _, out = evaluate_run(case["company"], case["variant"], llm=client, limits=limits, objective=OBJECTIVE,
                          retry_partial=retry_partial)
    return summarise(out, wire, decider, case)


def summarise(out: dict, wire, decider: PersonaLLM, case: dict) -> dict:
    trace = out["trace"]
    submits = [e for e in trace if e["type"] == "submit"]
    submit_calls = [e for e in trace if e["type"] == "tool_call" and e["data"].get("tool") == SUBMIT]
    submit_errors = [e for e in trace if e["type"] == "tool_result" and e["data"].get("tool") == SUBMIT and not e["data"]["ok"]]
    accepted = [c for c in out["claims"].get("company", []) if c.get("status") == "verified"]
    rejected = out["rejected"].get("company", [])
    A, R = {c["field"] for c in accepted}, {r["field"] for r in rejected}
    reasons = Counter(x for r in rejected for x in r["reasons"])
    first = submits[0]["data"] if submits else None
    first_failed_schema = bool(submit_errors) and (not submits or submit_errors[0]["seq"] < submits[0]["seq"])
    if first_failed_schema:
        first_rate = 0.0
    elif first and first["accepted"] + first["rejected"]:
        first_rate = first["accepted"] / (first["accepted"] + first["rejected"])
    else:
        first_rate = None                                  # nothing was submitted (e.g. empty search): not a failure of evidence
    world = make_world(case["company"], case["variant"])
    truth = ev.norm_text(" ".join([p["text"] for p in world["pages"].values()] + [d["content"] for d in world["search_index"]]))
    bad_accepted = sum(1 for c in accepted for e in c["evidence"] if ev.norm_text(e["snippet"]) not in truth)
    queries = [e["data"]["args"].get("query") for e in trace if e["type"] == "tool_call" and e["data"].get("tool") == "web_search"]
    m = metrics_from_run(out)
    problems = validate_report(out["report"])
    toks = [c // 4 for c in wire.payload_chars]
    return {
        "status": out["status"], "status_reasons": out.get("status_reasons", []), "attempts": len(submit_calls), "correction_cycles": max(0, len(submit_calls) - 1),
        "first_submit_accept_rate": first_rate,
        "schema_rejections": len(submit_errors),
        "claims_accepted": len(accepted), "claims_rejected": len(rejected),
        "final_accept_rate": len(A) / len(A | R) if A | R else None,
        "residual_rejected_fields": sorted(R - A), "corrected_fields": sorted(R & A),
        "unquoted": sum(v for k, v in reasons.items() if k in UNQUOTED),
        "unsupported_values": reasons.get("value_not_supported_by_snippet", 0),
        "fabricated_urls": reasons.get("url_not_retrieved_in_this_run", 0),
        "accepted_with_unreal_quote": bad_accepted,                       # must always be 0
        "invalid_tool_call_rate": m["invalid_tool_call_rate"], "tool_calls": m["tool_calls"],
        "adapted_after_empty_search": len(set(q for q in queries if q)) >= 2 if case["variant"] == "empty_search" else None,
        "identity_verified": out["report"]["identity"]["verified"],
        "official_website_in_report": "official_website" in (out["report"]["company_profile"] or {}),
        "report_problems": problems, "requests": wire.requests, "est_prompt_tokens": sum(toks),
        "wire_violations": len(wire.violations),
        "strict_ok": None if case["variant"] in ("empty_search", "no_website") else (
            (not (R - A)) and bad_accepted == 0 and not problems and out["report"]["identity"]["verified"]),
    }


# ------------------------------------------------------------------------------------------- the matrix
def evaluate(retry_partial: bool, providers=PROVIDERS, persona_names=None, case_list=None) -> dict:
    out: dict = {}
    for pname in (persona_names or PERSONAS):
        key, fixes = PERSONAS[pname]
        for prov, model in providers:
            out[(pname, prov)] = [run_one(prov, model, c, key, fixes, retry_partial) for c in (case_list or cases())]
    return out


BEHAVIOUR_KEYS = ("status", "status_reasons", "attempts", "correction_cycles", "first_submit_accept_rate", "schema_rejections",
                  "claims_accepted", "claims_rejected", "final_accept_rate", "residual_rejected_fields",
                  "corrected_fields", "unquoted", "unsupported_values", "fabricated_urls", "accepted_with_unreal_quote",
                  "invalid_tool_call_rate", "tool_calls", "adapted_after_empty_search", "identity_verified",
                  "official_website_in_report", "report_problems", "requests", "wire_violations", "strict_ok")


def behaviour(rows):
    return [{k: r[k] for k in BEHAVIOUR_KEYS} for r in rows]


def mean(rows, key):
    vals = [r[key] for r in rows if r[key] is not None]
    return round(statistics.mean(vals), 3) if vals else None


def global_checks(results: dict) -> dict:
    """Aggregate safety/validity checks over EVERY run in a result set."""
    rows = [r for rs in results.values() for r in rs]
    return {
        "runs": len(rows),
        "claims_accepted_with_an_unreal_quote": sum(r["accepted_with_unreal_quote"] for r in rows),      # must be 0
        "invalid_report_structures": sum(1 for r in rows if r["report_problems"]),                       # must be 0
        "wire_protocol_violations": sum(r["wire_violations"] for r in rows),                              # must be 0
        "max_invalid_tool_call_rate": max(r["invalid_tool_call_rate"] for r in rows),
        "empty_search_adapted": [r["adapted_after_empty_search"] for r in rows if r["adapted_after_empty_search"] is not None][:1],
        "status_counts": dict(Counter(r["status"] for r in rows)),
    }


def reassess_saved_smoke(path: str) -> dict | None:
    """Re-judge a saved live smoke result (synthetic data only) with the STRICT criterion, offline."""
    import os
    from agentic.agents.common import ProcessResult  # noqa: F401
    from agentic.agents.company import process_company
    from agentic.eval.gemini_smoke import COMPANY
    from agentic.eval.quality import strict_assessment
    if not os.path.exists(path):
        return None
    data = json.load(open(path, encoding="utf-8"))
    submit = next((c for r in data["per_request"] for c in r.get("raw", {}).get("tool_calls", [])
                   if c["name"] == SUBMIT), None)
    if submit is None:
        return None
    f = make_world(COMPANY, "standard")["facts"]
    home = make_world(COMPANY, "standard")["pages"][f"https://www.{f['domain']}/"]
    ledger: dict = {}
    ev.add_doc(ledger, url=f"https://www.{f['domain']}/", text=home["text"], title=home["title"])
    state = {"ledger": ledger, "claims": {}, "subject": {"name": COMPANY["name"], "existing": None, "identity": {"verified": False}}}
    pr = process_company(state, submit["args"], None)
    return strict_assessment(submit["args"], pr)


def table(results: dict, provider: str) -> str:
    head = ("| Persona | first-submit accept | final accept | correction cycles | residual rejected | schema rejections | "
            "unquoted | unsupported | fabricated URL | bad quote accepted | requests/run | est. prompt tokens/run | strict-ok runs |")
    lines = [head, "|" + "---|" * 13]
    for (pname, prov), rows in results.items():
        if prov != provider:
            continue
        resid = sum(len(r["residual_rejected_fields"]) for r in rows)
        lines.append(f"| {pname} | {mean(rows, 'first_submit_accept_rate'):.0%} | {mean(rows, 'final_accept_rate'):.0%} | "
                     f"{mean(rows, 'correction_cycles')} | {resid} | {sum(r['schema_rejections'] for r in rows)} | "
                     f"{sum(r['unquoted'] for r in rows)} | {sum(r['unsupported_values'] for r in rows)} | "
                     f"{sum(r['fabricated_urls'] for r in rows)} | {sum(r['accepted_with_unreal_quote'] for r in rows)} | "
                     f"{mean(rows, 'requests')} | {mean(rows, 'est_prompt_tokens'):,.0f} | "
                     f"{sum(1 for r in rows if r['strict_ok'])}/{sum(1 for r in rows if r['strict_ok'] is not None)} |")
    return "\n".join(lines)


def main() -> None:  # pragma: no cover
    for label, flag in (("CURRENT behaviour (retry only when ALL claims rejected)", False),
                        ("PROPOSED behaviour (also correct partial rejections)", True)):
        res = evaluate(flag)
        print(f"\n### {label}  - provider adapter: Groq openai/gpt-oss-120b\n")
        print(table(res, "groq"))
        same = all(behaviour(res[(p, "groq")]) == behaviour(res[(p, "gemini")]) for p in PERSONAS)
        print(f"\nidentical behaviour through the Gemini adapter for every persona: {same}")
        print("aggregate checks:", json.dumps(global_checks(res)))
    for label, path in (("Gemini 3 Flash live smoke (saved)", "data/smoke.json"), ("Groq gpt-oss-120b live smoke (saved)", "data/groq_smoke.json")):
        a = reassess_saved_smoke(path)
        if a:
            print(f"\nstrict re-assessment of {label}: accepted {a['accepted']}/{a['accepted'] + a['rejected']}, "
                  f"strict_passed={a['strict_passed']}, rejected fields={a['rejected_fields']}, reasons={a['rejection_reasons']}")


if __name__ == "__main__":  # pragma: no cover
    main()

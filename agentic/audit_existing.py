"""Read-only audit of what enrichment already stored for ONE company, and a corrective change PLAN.

    python -m agentic.audit_existing --company-id 11 --load-dotenv [--out data/corrective_plan_company_11.json]

It issues SELECTs only (through the counted client: a hard cap, no insert/update/delete/upsert/rpc is reachable), re-judges
every agentic provenance row with the SAME field-meaning rules the Company/Procurement agents use today, and writes a plan.
The plan is a document: nothing in it is executed, and every action says what explicit approval it needs. The database is
never changed by this module.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from agentic import field_support
from agentic import fields as fl

SELECT_CAP = 6


def parse_insight(row: dict) -> dict:
    """field / value / quote / source of a provenance row written by the apply path ('Verified <f>' or 'Procurement: <f>')."""
    itype = str(row.get("insight_type") or "")
    field = itype.removeprefix("Verified ").removeprefix("Procurement: ")
    title = str(row.get("insight_title") or "")
    value = title.split(": ", 1)[1] if ": " in title else title
    quote, _, tail = str(row.get("insight") or "").rpartition(" (source: ")
    return {"insight_id": row.get("insight_id"), "field": field, "value": value, "quote": quote,
            "source_url": row.get("source_url") or tail.rstrip(")"), "type": itype}


def judge_insight(row: dict, company: str | None = None) -> dict:
    p = parse_insight(row)
    reasons = []
    if p["field"] not in fl.BY_KEY:
        reasons.append("unknown_field")
    else:
        ok, why = field_support.check_evidence(p["field"], p["value"], [{"url": p["source_url"], "snippet": p["quote"]}],
                                               company=company)
        if not ok:
            reasons.append(why)
    if "�" in p["quote"]:
        reasons.append("replacement_character_in_quote")
    if not p["source_url"] or not p["quote"].strip():
        reasons.append("missing_source_or_quote")
    return {**p, "verdict": "insufficient" if reasons else "supported", "reasons": reasons}


def corrective_plan(company: dict, profiles: list[dict], insights: list[dict]) -> dict:
    cid = company.get("company_id")
    agentic = [r for r in insights if str(r.get("generated_by", "")).startswith("agentic-")]
    judged = [judge_insight(r, company.get("company_name")) for r in agentic]
    profile = profiles[0] if len(profiles) == 1 else None
    actions = []
    for j in judged:
        if j["verdict"] == "supported":
            continue
        spec = fl.BY_KEY.get(j["field"])
        if spec and spec.table == "company_profile" and profile and profile.get(spec.column) not in (None, ""):
            actions.append({
                "target": f"company_profile.{spec.column} (profile_id {profile['profile_id']})",
                "current_value": profile.get(spec.column), "proposed_value": None,
                "why": f"its only evidence is insufficient for the field's meaning: {j['reasons']}",
                "requires": "explicit overwrite approval (approved_overwrite): the field is populated; no code path writes "
                            "NULL today, so this is a reviewed one-off update, not an agent run"})
        actions.append({
            "target": f"ai_insights.insight_id {j['insight_id']}", "current_value": j["type"], "proposed_value": "withdrawn / superseded",
            "why": f"evidence insufficient: {j['reasons']}",
            "requires": "explicit approval; the write gate allows INSERT only on ai_insights (no update, no delete), so marking "
                        "a row withdrawn needs a new gate capability or a manual SQL change - neither exists or is approved"})
    web_empty = all(company.get(c) in (None, "") for c in ("website", "linkedin_url"))
    return {
        "company_id": cid, "company_name": company.get("company_name"), "read_only": True, "writes_performed": 0,
        "company_profile_rows": len(profiles), "profile_id": profile.get("profile_id") if profile else None,
        "profile_is_unique_per_company": len(profiles) <= 1,
        "agentic_insight_ids": [r.get("insight_id") for r in agentic],
        "insight_verdicts": [{k: j[k] for k in ("insight_id", "field", "value", "verdict", "reasons", "source_url", "quote")}
                             for j in judged],
        "core_fields": {"website": company.get("website"), "linkedin_url": company.get("linkedin_url"),
                        "decision": "keep EMPTY" if web_empty else "populated: never overwritten",
                        "needed_to_fill": "the official SITE ROOT (scheme + host) must be retrieved in a run and the domain "
                                          "verified for this company (agentic/website_map.py); a deep regional page is never stored"},
        "derived_not_written": fl.DERIVED_NOT_WRITTEN,
        "actions": actions,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Read-only audit + corrective plan for one company.")
    ap.add_argument("--company-id", type=int, required=True)
    ap.add_argument("--load-dotenv", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    if args.load_dotenv:
        from dotenv import load_dotenv
        load_dotenv(".env")
    from supabase import create_client
    from agentic.guards.select_budget import CountedClient, SelectBudget
    budget = SelectBudget(SELECT_CAP)
    client = CountedClient(create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_KEY"]), budget)
    cid = args.company_id
    company = (client.table("companies").select("*").eq("company_id", cid).execute().data or [None])[0]
    if company is None:
        print("company not found")
        return 2
    profiles = client.table("company_profile").select("*").eq("company_id", cid).execute().data or []
    insights = client.table("ai_insights").select("*").eq("company_id", cid).execute().data or []
    plan = corrective_plan(company, profiles, insights)
    plan["selects_used"] = budget.used
    out = args.out or os.path.join("data", f"corrective_plan_company_{cid}.json")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(plan, fh, indent=2, default=str, ensure_ascii=False)
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print(json.dumps(plan, indent=2, default=str, ensure_ascii=False))
    print(f"(saved to {out})")
    return 0


if __name__ == "__main__":
    sys.exit(main())

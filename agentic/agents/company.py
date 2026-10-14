"""Company Intelligence Agent.

Goal: establish the organisation's identity (official website), then fill
profile gaps with evidence. It decides which tools to call and when to stop;
Python validates every claim it submits."""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from agentic import evidence as ev
from agentic import field_support
from agentic import gaps as gp
from agentic import names
from agentic.agents.common import (AgentSpec, BaseFindings, EvidenceRefModel, FieldClaim,
                                   ProcessResult, validate_refs)
from agentic.prefetch import run_company_prefetch

COMPANY_CLAIM_FIELDS = {"official_website", "headquarters", "industry", "business_overview",
                        "products_services", "employee_count", "annual_revenue", "founded_year",
                        "parent_company", "manufacturing_locations", "ownership_type", "linkedin_url",
                        "company_type"}
LIST_FIELDS = {"products_services", "manufacturing_locations"}

NON_FIRST_PARTY = {
    "linkedin.com", "facebook.com", "twitter.com", "x.com", "instagram.com", "youtube.com",
    "wikipedia.org", "crunchbase.com", "zaubacorp.com", "tofler.in", "indiamart.com", "justdial.com",
    "bloomberg.com", "reuters.com", "moneycontrol.com", "economictimes.indiatimes.com", "glassdoor.com",
    "naukri.com", "ambitionbox.com", "dnb.com", "zoominfo.com", "tradeindia.com", "wikidata.org",
    "google.com", "bing.com", "yahoo.com",
}


class CompanyFindings(BaseFindings):
    identity_website: str | None = Field(
        default=None, description="Official first-party website of THE organisation, or null if not confirmed.")
    identity_reasoning: str = Field(default="", max_length=600,
                                    description="Why this is the right organisation (and which lookalikes you rejected).")
    claims: list[FieldClaim] = Field(default_factory=list, max_length=15)


def verify_identity(ledger: dict, website: str, company_name: str) -> tuple[bool, str, str]:
    """Is `website` a first-party site that demonstrably belongs to this company?"""
    canon = ev.canonical_url(website)
    if not canon:
        return False, "invalid_website_url", ""
    domain = ev.registrable_domain(canon)
    if domain in NON_FIRST_PARTY:
        return False, "aggregator_or_social_not_first_party", domain
    docs = [d for d in ledger.values() if ev.registrable_domain(d["url"]) == domain]
    if not docs:
        return False, "domain_not_retrieved_in_this_run", domain
    wanted = names.distinctive_tokens(company_name)
    text = " ".join(ev.norm_text(d["title"] + " " + d["text"]) for d in docs)
    present = [t for t in wanted if t in text]
    if wanted and len(present) / len(wanted) >= 0.8:
        return True, "company_name_found_on_domain", domain
    return False, "company_name_not_found_on_domain", domain


def process_company(state: dict, findings: dict, deps) -> ProcessResult:
    res = ProcessResult()
    ledger = state["ledger"]
    subject = state["subject"]
    name = subject["name"]
    website = findings.get("identity_website")
    identity = dict(subject.get("identity") or {})

    # --- identity first -----------------------------------------------------
    website_claims = [c for c in findings.get("claims", []) if c["field"] == "official_website"]
    candidate_site = website or (website_claims[0].get("value") if website_claims else None)
    if candidate_site:
        ok, reason, domain = verify_identity(ledger, str(candidate_site), name)
        if ok:
            identity = {"verified": True, "website": ev.canonical_url(str(candidate_site)),
                        "domain": domain, "reason": reason,
                        "reasoning": (findings.get("identity_reasoning") or "")[:600]}
            res.subject_updates["identity"] = identity
        else:
            res.rejected.append({"field": "official_website", "value": candidate_site, "reasons": [reason]})

    official_domain = identity.get("domain") if identity.get("verified") else None
    seen = {(c.get("field"), str(c.get("value"))) for c in state["claims"].get("company", [])}

    for claim in findings.get("claims", []):
        fld = claim["field"]
        value = claim["values"] if claim.get("values") is not None else claim.get("value")
        if fld not in COMPANY_CLAIM_FIELDS:
            res.rejected.append({"field": fld, "value": value, "reasons": ["field_not_allowed"]})
            continue
        if fld in LIST_FIELDS and not isinstance(value, list):
            value = [str(value)]
        if fld not in LIST_FIELDS and isinstance(value, list):
            res.rejected.append({"field": fld, "value": value, "reasons": ["expected_single_value"]})
            continue
        if fld == "official_website":
            if not identity.get("verified") or ev.registrable_domain(str(value)) != identity.get("domain"):
                if not any(r["field"] == "official_website" for r in res.rejected):
                    res.rejected.append({"field": fld, "value": value, "reasons": ["identity_not_confirmed"]})
                continue
            value = identity["website"]
        if fld == "linkedin_url" and ev.registrable_domain(str(value)) != "linkedin.com":
            res.rejected.append({"field": fld, "value": value, "reasons": ["not_a_linkedin_url"]})
            continue
        if fld == "linkedin_url":                           # the page must be about THIS company: its slug carries the name
            slug = ev.norm_text(str(value).split("linkedin.com", 1)[-1]).replace(" ", "")
            if not any(t in slug for t in names.distinctive_tokens(name)):
                res.rejected.append({"field": fld, "value": value, "reasons": ["linkedin_slug_does_not_match_company"]})
                continue
        check = validate_refs(ledger, [e for e in claim["evidence"]], value, official_domain,
                              require_value_support=fld not in ("official_website", "linkedin_url"))
        if fld in ("official_website", "linkedin_url") and check["ok"]:
            # the URL itself must be what the evidence is about
            if ev.registrable_domain(str(value)) not in {ev.registrable_domain(e["url"]) for e in check["evidence"]} \
               and fld == "official_website":
                check = {**check, "ok": False, "reasons": ["evidence_not_from_claimed_site"]}
        if check["ok"] and fld not in ("official_website", "linkedin_url"):
            # a quote that merely mentions the value is not enough: its sentence must state THIS field
            sup_ok, sup_reason = field_support.check(fld, value, [e["snippet"] for e in check["evidence"]])
            if not sup_ok:
                check = {**check, "ok": False, "reasons": [sup_reason]}
        if not check["ok"]:
            res.rejected.append({"field": fld, "value": value, "reasons": check["reasons"]})
            continue
        if (fld, str(value)) in seen:
            continue
        seen.add((fld, str(value)))
        res.accepted.append({"agent": "company", "field": fld, "value": value, "status": "verified",
                             "evidence": check["evidence"], "confidence": check["confidence"],
                             "confidence_basis": check["basis"]})
    return res


def company_task(state: dict) -> str:
    s = state["subject"]
    existing = (s.get("existing") or {})
    fields = existing.get("fields", {})
    db_ctx = {k: fields.get(k) for k in ("country", "sector_name", "website", "revenue",
                                          "number_of_plants", "company_type") if fields.get(k) not in (None, "")}
    gaps = [g for g in gp.company_gaps(s, state["claims"]) if g["status"] != "verified"]
    if (state.get("options") or {}).get("fields"):         # field request: only the fields the user asked for
        from agentic import fields as fl
        focus = set(fl.focus_company_fields(state))
        gaps = [g for g in gaps if g["field"] in focus]
    lines = [
        f"Company submitted by the user: {s['name']!r}",
        f"User objective: {state['objective']}",
        f"Database match: {s['match']['status']}",
    ]
    if db_ctx:
        lines.append("Known database values (UNVERIFIED - no provenance, use only as disambiguation hints "
                     f"and verify before relying on them): {db_ctx}")
    if existing.get("products"):
        lines.append(f"Product names recorded in the CRM (unverified): {existing['products'][:8]}")
    for a in s["match"].get("assumptions", []):
        lines.append(f"Assumption: {a}")
    lines.append("Information gaps to fill (status 'unverified_db' means a legacy value exists - verify it; "
                 "'missing' means nothing is known):")
    for g in gaps:
        extra = f" (db value: {g['db_value']})" if g.get("db_value") not in (None, "") else ""
        lines.append(f"  - {g['field']}: {g['status']}{extra}")
    lines.append("First confirm the official website and that it belongs to this organisation, then "
                 "fill the remaining gaps from the company's own pages when possible.")
    return "\n".join(lines)


COMPANY_AGENT = AgentSpec(
    name="company",
    title="Company Intelligence Agent",
    goal=("Confirm the identity and official website of the organisation and gather evidence-backed "
          "facts: headquarters, industry, business overview, products/services, size, revenue, "
          "locations, ownership."),
    role_prompt=("Typical approach: search for the company's official site, open its home/about pages, "
                 "extract facts with verbatim quotes. Use third-party sources only to corroborate. "
                 "Claim fields allowed: " + ", ".join(sorted(COMPANY_CLAIM_FIELDS)) + ". "
                 "For list fields (products_services, manufacturing_locations) give a list of short items."),
    findings_model=CompanyFindings,
    finish_tool="submit_company_findings",
    tool_names=("web_search", "fetch_page", "submit_company_findings"),
    process=process_company,
    task=company_task,
    prefetch=run_company_prefetch,      # only used when Efficiency.prefetch is on
)

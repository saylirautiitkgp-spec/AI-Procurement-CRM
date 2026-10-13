"""Evidence-QUALITY criteria for evaluations (not production acceptance rules).

The legacy smoke criterion only required `accepted >= 1`, which let a submission with a mis-quoted claim pass.
`strict_assessment` judges the same submission on evidence quality; `validate_report` checks the final structured
report. Nothing here changes what the production validator accepts or rejects.
"""

from __future__ import annotations

from collections import Counter

FABRICATION_REASONS = {"url_not_retrieved_in_this_run", "snippet_not_found_in_document", "value_not_supported_by_snippet",
                       "email_not_literally_in_snippet", "phone_not_literally_in_snippet", "snippet_too_short",
                       "snippet_too_long"}
UNQUOTED = {"snippet_not_found_in_document", "snippet_too_short", "snippet_too_long"}


def strict_assessment(findings: dict, pr, min_fields: int = 2) -> dict:
    """`pr` is the ProcessResult the production validator produced for `findings`."""
    accepted, rejected = len(pr.accepted), len(pr.rejected)
    reasons = Counter(r for x in pr.rejected for r in x["reasons"])
    fields = {c["field"] for c in pr.accepted}
    identity_ok = bool((pr.subject_updates.get("identity") or {}).get("verified"))
    checks = {
        "all_submitted_claims_accepted": accepted > 0 and rejected == 0,
        "no_unquoted_or_fabricated_evidence": not (set(reasons) & FABRICATION_REASONS),
        "official_website_established_with_evidence": identity_ok or "official_website" in fields,   # either route counts
        f"at_least_{min_fields}_fields_accepted": len(fields) >= min_fields,
    }
    return {"claims_submitted": len(findings.get("claims", [])), "accepted": accepted, "rejected": rejected,
            "acceptance_rate": round(accepted / (accepted + rejected), 3) if accepted + rejected else 0.0,
            "rejection_reasons": dict(reasons), "rejected_fields": sorted({x.get("field") or x.get("kind") for x in pr.rejected}),
            "checks": checks, "strict_passed": all(checks.values())}


_TOP = ("run", "summary", "database_match", "identity", "company_profile", "database_context", "sources", "gaps",
        "coverage_limitations", "usage", "next_actions", "proposed_changes")


def validate_report(report: dict) -> list[str]:
    """Structural + honesty checks on the final report; returns a list of problems (empty = valid)."""
    p: list[str] = []
    p += [f"missing key {k}" for k in _TOP if k not in report]
    if p:
        return p
    if report["run"].get("status") not in ("completed", "partial", "failed"):
        p.append("run.status is not a final status")
    urls = {s["url"] for s in report["sources"]}
    for field, claim in (report["company_profile"] or {}).items():
        if not claim.get("evidence"):
            p.append(f"{field}: claim without evidence")
        if not 0.0 <= claim.get("confidence", -1) <= 1.0:
            p.append(f"{field}: confidence out of range")
        for e in claim.get("evidence", []):
            if not (e.get("url", "").startswith("http") and e.get("snippet") and e.get("retrieved_at")):
                p.append(f"{field}: incomplete evidence record")
            if e.get("url") not in urls:
                p.append(f"{field}: evidence url missing from sources")
    ctx = report["database_context"]
    if ctx and ctx.get("provenance") != "database_unverified":
        p.append("database values not labelled unverified")
    if report["identity"]["verified"] and not report["identity"].get("website"):
        p.append("identity verified without a website")
    if not isinstance(report["next_actions"], list) or not isinstance(report["coverage_limitations"], list):
        p.append("next_actions/coverage_limitations not lists")
    return p

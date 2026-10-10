"""Information-gap computation (deterministic).

Field status:
  verified      - accepted this run with validated evidence
  unverified_db - value exists in the database but has no provenance (a default
                  or legacy value, NOT a verified fact)
  missing       - nothing known
"""

from __future__ import annotations

COMPANY_FIELDS = ("official_website", "headquarters", "industry", "business_overview",
                  "products_services", "employee_count", "annual_revenue", "founded_year")
_DB_FIELD = {"official_website": "website", "annual_revenue": "revenue"}


def verified_fields(claims: dict, agent: str) -> dict:
    out: dict = {}
    for c in claims.get(agent, []):
        if c.get("status") == "verified":
            out.setdefault(c.get("field") or c.get("kind"), []).append(c)
    return out


def company_gaps(subject: dict, claims: dict) -> list[dict]:
    existing = (subject.get("existing") or {}).get("fields", {})
    products = (subject.get("existing") or {}).get("products", [])
    done = verified_fields(claims, "company")
    gaps = []
    for f in COMPANY_FIELDS:
        if f in done:
            gaps.append({"field": f, "status": "verified"})
            continue
        db_val = existing.get(_DB_FIELD.get(f, "")) if f in _DB_FIELD else None
        if f == "products_services" and products:
            db_val = products
        if db_val not in (None, "", [], {}):
            gaps.append({"field": f, "status": "unverified_db", "db_value": db_val})
        else:
            gaps.append({"field": f, "status": "missing"})
    return gaps


def other_agent_gaps(subject: dict, claims: dict, agent_results: dict) -> dict:
    ex = subject.get("existing") or {}
    out = {}
    cs = ex.get("contact_summary") or {}
    out["contact"] = ("verified" if verified_fields(claims, "contact") else
                      "unverified_db" if cs.get("count") else "missing")
    out["procurement"] = "verified" if verified_fields(claims, "procurement") else "missing"
    out["supplier"] = ("verified" if verified_fields(claims, "supplier") else
                       "unverified_db" if ex.get("suppliers") else "missing")
    out["lead"] = "computed" if claims.get("lead") else "missing"
    return out

"""Deterministic write safeguards. Nothing reaches the database unless every
operation passes this gate; agents have no access to it at all.

* table/operation/column allowlist (no deletes, ever)
* type + length validation, no primary keys or system columns
* overwrite detection: changing a populated value needs explicit approval
* evidence is re-validated (cited snippets must be real excerpts of retrieved docs)
"""

from __future__ import annotations

import re
from typing import Any

from agentic import evidence as ev

MAX_TEXT = 4000
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")

#           table:  {column: type}
COLUMNS: dict[str, dict[str, str]] = {
    "companies": {"company_name": "text", "sector_id": "int", "country": "text", "website": "url",
                  "linkedin_url": "url", "company_type": "text"},
    "company_profile": {"company_id": "int", "headquarters": "text", "industry": "text",
                        "company_summary": "text", "products": "list", "services": "list",
                        "employee_count": "text", "annual_revenue": "text", "founded_year": "text",
                        "parent_company": "text", "ownership_type": "text", "company_type": "text",
                        "website_domain": "text", "manufacturing_locations": "list",
                        "supplier_portal": "url", "procurement_page": "url", "confidence_score": "num"},
    "contacts": {"company_id": "int", "contact_name": "text", "designation": "text", "department": "text",
                 "email": "email", "phone": "text", "linkedin": "url", "contact_type": "text",
                 "decision_maker": "bool", "confidence_score": "num", "source_url": "url",
                 "last_verified": "ts"},
    "ai_insights": {"company_id": "int", "insight_type": "text", "insight_title": "text", "insight": "text",
                    "confidence_score": "num", "generated_by": "text", "source_url": "url",
                    "source_type": "text", "status": "text"},
    "suppliers": {"supplier_name": "text", "website": "url", "supplier_type": "text",
                  "components_supplied": "text", "created_by_ai": "bool", "supplier_description": "text",
                  "last_verified": "ts"},
    "company_suppliers": {"company_id": "int", "supplier_id": "int", "component": "text",
                          "remarks": "text", "confidence_score": "num", "discovered_by": "text"},
}
OPERATIONS: dict[str, set[str]] = {
    "companies": {"insert", "update"}, "company_profile": {"insert", "update"},
    "contacts": {"insert", "update"}, "ai_insights": {"insert"}, "suppliers": {"insert"},
    "company_suppliers": {"insert"},
}
UPDATE_KEYS = {"companies": "company_id", "company_profile": "company_id", "contacts": "contact_id"}
# contacts may only be updated to ATTACH provenance / correct the title; never to rename or re-parent.
UPDATE_COLUMNS = {"contacts": {"source_url", "confidence_score", "last_verified", "designation"}}
FORBIDDEN_OPS = {"delete", "truncate", "merge", "drop", "upsert"}


def _coerce(kind: str, value: Any) -> tuple[bool, Any]:
    if value is None:
        return False, None
    if kind == "text":
        if not isinstance(value, (str, int, float)):
            return False, None
        s = str(value).strip()
        return (bool(s) and len(s) <= MAX_TEXT and not _CONTROL.search(s)), s
    if kind == "url":
        c = ev.canonical_url(str(value))
        return bool(c) and len(c) <= 500, c
    if kind == "email":
        s = str(value).strip()
        return bool(re.fullmatch(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}", s)), s
    if kind == "int":
        return (isinstance(value, int) and not isinstance(value, bool)), value
    if kind == "num":
        return (isinstance(value, (int, float)) and not isinstance(value, bool) and 0 <= value <= 1_000_000), value
    if kind == "bool":
        return isinstance(value, bool), value
    if kind == "ts":
        return isinstance(value, str) and len(value) <= 40, value
    if kind == "list":
        if not isinstance(value, list) or not value or len(value) > 50:
            return False, None
        items = [str(v).strip() for v in value]
        return all(i and len(i) <= 200 and not _CONTROL.search(i) for i in items), items
    return False, None


def validate_operation(op: dict, existing: dict | None = None) -> tuple[list[str], dict]:
    """Return (errors, cleaned_fields)."""
    errors: list[str] = []
    table, kind, fields = op.get("table"), op.get("op"), op.get("fields") or {}
    if kind in FORBIDDEN_OPS:
        return [f"operation_forbidden:{kind}"], {}
    if table not in OPERATIONS:
        return [f"table_not_writable:{table}"], {}
    if kind not in OPERATIONS[table]:
        return [f"operation_not_allowed:{table}.{kind}"], {}
    if kind == "update":
        key = op.get("key") or {}
        pk = UPDATE_KEYS.get(table)
        if pk is None or set(key) != {pk} or not isinstance(key.get(pk), int):
            errors.append("invalid_update_key")
        if not existing:
            errors.append("update_target_not_found")
    cleaned: dict = {}
    spec = COLUMNS[table]
    if not fields:
        errors.append("no_fields")
    for col, val in fields.items():
        if col not in spec or (kind == "update" and table in UPDATE_COLUMNS and col not in UPDATE_COLUMNS[table]):
            errors.append(f"column_not_allowed:{table}.{col}")
            continue
        ok, coerced = _coerce(spec[col], val)
        if not ok:
            errors.append(f"invalid_value:{table}.{col}")
        else:
            cleaned[col] = coerced
    return errors, cleaned


def detect_overwrites(op: dict, existing: dict | None) -> list[str]:
    """Columns whose populated existing value would be changed by this update."""
    if op.get("op") != "update" or not existing:
        return []
    out = []
    for col, val in (op.get("fields") or {}).items():
        old = existing.get(col)
        if old in (None, "", [], {}):
            continue
        if str(old).strip().lower() != str(val).strip().lower():
            out.append(col)
    return out


def validate_evidence(evidence: list[dict], ledger: dict) -> list[str]:
    if not evidence:
        return ["no_evidence"]
    errors = []
    for e in evidence:
        ok, reason, _ = ev.validate_snippet(ledger, e.get("url", ""), e.get("snippet", ""))
        if not ok:
            errors.append(f"evidence_invalid:{reason}")
    return errors

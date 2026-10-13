"""Field-level change set for human review - built from VERIFIED claims only, never written by this module.

For every requested field it states the old value (what the CRM holds today, unverified legacy data), the proposed
value, the source URLs and quotations, the computed confidence, the validation status and the action:

  fill_empty       the CRM field is empty and a verified value exists
  update_conflict  the CRM holds a DIFFERENT non-empty value: an overwrite that needs explicit approval
  no_change        the verified value equals the CRM value
  add_new          a new record (contact, supplier, insight) - never replaces an existing one
  report_only      lead score/actions, nothing is stored

Unsupported, rejected or unfound values never become items: they are listed under `unresolved` and the old value is
left alone. Actual writes still go only through the approval-gated proposal/commit path (agentic/proposals.py).
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from agentic import evidence as ev
from agentic import fields as fl
from agentic.claimid import claim_key  # noqa: F401 - re-exported

_BASIS = "quote found verbatim in a document retrieved in this run; value supported by the quote"


def _norm(v: Any):
    if isinstance(v, list):
        return sorted(str(x).strip().lower() for x in v)
    if isinstance(v, str) and v.startswith(("http://", "https://")):
        return ev.canonical_url(v).lower()
    return str(v).strip().lower() if v is not None else None


def _evidence(c: dict) -> list[dict]:
    return [{"url": e["url"], "snippet": e["snippet"], "retrieved_at": e.get("retrieved_at")} for e in c.get("evidence", [])]


def _legacy(spec: fl.FieldSpec, existing: dict | None) -> dict:
    ref = fl.LEGACY_COLUMNS.get(spec.key)
    val = fl.legacy_value(spec.key, existing)
    return {f"{ref[0]}.{ref[1]}": val} if ref and val not in (None, "", [], {}) else {}


def _item(spec: fl.FieldSpec, claim: dict, old: Any, proposed: Any, action: str, **extra) -> dict:
    ident = json.dumps([spec.key, proposed], sort_keys=True, default=str)
    return {"id": hashlib.sha256(ident.encode()).hexdigest()[:12], "claim_key": claim_key(spec.agent, claim),
            "field": spec.key, "label": spec.label,
            "entity": spec.entity, "target": f"{spec.table}.{spec.column}" if spec.table and spec.column else spec.table,
            "agent": spec.agent, "old_value": old, "proposed_value": proposed, "action": action,
            "requires_approval": action in ("fill_empty", "update_conflict", "add_new"),
            "overwrite": action == "update_conflict", "confidence": claim.get("confidence"),
            "confidence_basis": claim.get("confidence_basis"), "validation_status": "verified",
            "validation": _BASIS, "source_urls": sorted({e["url"] for e in claim.get("evidence", [])}),
            "evidence": _evidence(claim), **extra}


def _verified(claims: dict, spec: fl.FieldSpec) -> list[dict]:
    items = claims.get(spec.agent)
    if not isinstance(items, list):
        return []
    key = spec.key
    return [c for c in items if c.get("status") == "verified" and (c.get("field") or c.get("kind")) == key]


def _rejected_reasons(rejected: dict, spec: fl.FieldSpec) -> list[str]:
    out = []
    for r in rejected.get(spec.agent, []) or []:
        if (r.get("field") or r.get("kind")) == spec.key:
            out.extend(r.get("reasons", []))
    return sorted(set(out))


def build_change_set(state: dict) -> dict:
    opts = state.get("options") or {}
    subject = state["subject"]
    existing = subject.get("existing")
    claims, rejected = state["claims"], state.get("rejected", {})
    refresh = bool(opts.get("refresh"))
    requested = list(opts.get("fields") or [])
    if not requested:      # legacy run: every field that produced a verified claim
        for spec in fl.CATALOG:
            if spec.agent in ("contact", "supplier"):
                if any(c.get("status") == "verified" for c in (claims.get(spec.agent) or []) if isinstance(c, dict)):
                    requested.append(spec.key)
            elif spec.agent == "lead":
                continue
            elif _verified(claims, spec):
                requested.append(spec.key)
    items, unresolved, skipped = [], [], []
    for key in requested:
        spec = fl.BY_KEY[key]
        old = fl.crm_value(key, existing)
        if spec.agent == "company" or (spec.table == "company_profile" and spec.column):
            if spec.column and not refresh and fl.has_crm_value(key, existing) and not _verified(claims, spec):
                skipped.append({"field": key, "label": spec.label, "reason": "already_stored_not_refreshed",
                                "old_value": old, **({"legacy_values": _legacy(spec, existing)} if _legacy(spec, existing) else {})})
                continue
            found = _verified(claims, spec)
            if not found:
                unresolved.append({"field": key, "label": spec.label, "old_value": old,
                                   "reason": "rejected_by_validation" if _rejected_reasons(rejected, spec) else "not_found",
                                   "rejection_reasons": _rejected_reasons(rejected, spec)})
                continue
            distinct = {json.dumps(c["value"], sort_keys=True, default=str).lower() for c in found}
            if len(distinct) > 1:        # conflicting verified values: a human must choose; none is proposed
                unresolved.append({"field": key, "label": spec.label, "old_value": old, "reason": "conflicting_evidence",
                                   "candidates": [{"value": c["value"], "confidence": c["confidence"],
                                                   "source_urls": [e["url"] for e in c["evidence"]]} for c in found]})
                continue
            best = max(found, key=lambda c: c["confidence"])
            empty = old in (None, "", [], {})
            action = "fill_empty" if empty else ("no_change" if _norm(old) == _norm(best["value"]) else "update_conflict")
            legacy = _legacy(spec, existing)
            items.append(_item(spec, best, old, best["value"], action,
                               **({"legacy_values": legacy, "notes": "Legacy column(s) shown for context only; they are "
                                   "never written."} if legacy else {})))
        elif spec.agent == "procurement":          # list-like procurement findings: new insights, never overwrites
            found = _verified(claims, spec)
            if not found:
                unresolved.append({"field": key, "label": spec.label, "old_value": None,
                                   "reason": "rejected_by_validation" if _rejected_reasons(rejected, spec) else "not_found",
                                   "rejection_reasons": _rejected_reasons(rejected, spec)})
            for c in found:
                items.append(_item(spec, c, None, c["value"], "add_new"))
        elif spec.agent == "contact":
            found = [c for c in (claims.get("contact") or []) if isinstance(c, dict) and c.get("status") == "verified"]
            if not found:
                unresolved.append({"field": key, "label": spec.label, "old_value": None, "reason": "not_found"})
            for c in found:
                items.append(_item(spec, c, None, {"name": c["name"], "designation": c["designation"]}, "add_new",
                                   notes="CRM contact names are not loaded into a run; compare with existing contacts "
                                         "at approval time. Existing contacts are never overwritten."))
        elif spec.agent == "supplier":
            found = [c for c in (claims.get("supplier") or []) if isinstance(c, dict) and c.get("status") == "verified"]
            if not found:
                unresolved.append({"field": key, "label": spec.label, "old_value": None, "reason": "not_found"})
            for c in found:
                items.append(_item(spec, c, None, {"supplier": c["supplier"], "customer": c["customer"],
                                                   "component": c.get("component")}, "add_new"))
        elif spec.agent == "lead":
            lead = claims.get("lead")
            if isinstance(lead, dict) and lead.get("score") is not None:
                value = lead["score"] if key == "lead_score" else lead.get("agent_suggestions", [])
                items.append({"id": hashlib.sha256(key.encode()).hexdigest()[:12], "field": key, "label": spec.label,
                              "entity": "lead", "target": None, "agent": "lead", "old_value": None,
                              "proposed_value": value, "action": "report_only", "requires_approval": False,
                              "overwrite": False, "confidence": None, "confidence_basis": "deterministic score from verified data",
                              "validation_status": "computed", "validation": "deterministic scoring; not stored",
                              "source_urls": sorted({e["url"] for k, v in claims.items() if isinstance(v, list)
                                                     for c in v for e in c.get("evidence", [])})[:10], "evidence": []})
            else:
                unresolved.append({"field": key, "label": spec.label, "old_value": None, "reason": "insufficient_data"})
    for u in unresolved:
        if u["field"] in fl.LEGACY_COLUMNS and _legacy(fl.BY_KEY[u["field"]], existing):
            u["legacy_values"] = _legacy(fl.BY_KEY[u["field"]], existing)
    summary: dict[str, int] = {}
    for it in items:
        summary[it["action"]] = summary.get(it["action"], 0) + 1
    summary["unresolved"] = len(unresolved)
    summary["skipped"] = len(skipped)
    digest = hashlib.sha256(json.dumps([(i["id"], i["old_value"], i["proposed_value"], i["action"]) for i in items],
                                       sort_keys=True, default=str).encode()).hexdigest()[:16]
    by_table: dict[str, dict] = {}
    for it in items:
        table = (it["target"] or "report_only").split(".")[0]
        row = by_table.setdefault(table, {"count": 0, "actions": {}, "item_ids": [], "fields": []})
        row["count"] += 1
        row["actions"][it["action"]] = row["actions"].get(it["action"], 0) + 1
        row["item_ids"].append(it["id"])
        if it["field"] not in row["fields"]:
            row["fields"].append(it["field"])
    return {"company": subject.get("name"), "company_id": (existing or {}).get("company_id"),
            "change_set_hash": digest, "requested_fields": requested, "by_target_table": by_table,
            "companies_table_review": fl.companies_review(existing, requested), "refresh": refresh, "items": items, "unresolved": unresolved,
            "skipped": skipped, "summary": summary, "review_required": True, "writes_performed": False,
            "note": "Proposed values come from verified claims only. Nothing has been written; apply via the "
                    "approval-gated proposal flow. Old values are unverified legacy CRM data."}

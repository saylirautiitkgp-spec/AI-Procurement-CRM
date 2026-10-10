"""Proposed database changes: build -> validate -> persist -> approve -> apply.

Agents never write. After a run, verified claims are turned into
`ProposedChange`s (deterministically), validated by the write gate and stored.
Applying requires an explicit human approval recorded in the audit log, the
content hash must still match what was validated, and applying twice is a no-op.
"""

from __future__ import annotations

import copy
import hashlib
import json
from typing import Any

from agentic import evidence as ev
from agentic import write_gate as wg
from agentic.persistence.store import Store

COMPANY_MAP = {"official_website": "website", "linkedin_url": "linkedin_url"}
PROFILE_MAP = {"headquarters": "headquarters", "industry": "industry", "business_overview": "company_summary",
               "products_services": "products", "employee_count": "employee_count",
               "annual_revenue": "annual_revenue", "founded_year": "founded_year",
               "parent_company": "parent_company", "ownership_type": "ownership_type",
               "company_type": "company_type", "manufacturing_locations": "manufacturing_locations",
               "supplier_portal": "supplier_portal", "procurement_page": "procurement_page"}
NEW_COMPANY = "$company_id"
PROCUREMENT_TITLES = ("procurement", "purchas", "sourcing", "supply chain", "buyer", "vendor")


def _hash(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()


def best_claims(claims: list[dict]) -> tuple[dict, list[dict]]:
    """Highest-confidence verified claim per field; fields with conflicting
    values are excluded (a human must resolve them) and reported."""
    by_field: dict[str, list[dict]] = {}
    for c in claims:
        if c.get("status") == "verified":
            by_field.setdefault(c["field"], []).append(c)
    chosen, conflicts = {}, []
    for fld, items in by_field.items():
        distinct = {json.dumps(c["value"], sort_keys=True).lower() for c in items}
        if len(distinct) > 1:
            conflicts.append({"field": fld, "values": [c["value"] for c in items]})
            continue
        chosen[fld] = max(items, key=lambda c: c["confidence"])
    return chosen, conflicts


def _insight_ops(company_ref, chosen: dict, agent: str) -> tuple[list[dict], list[dict]]:
    ops, evidence = [], []
    for fld, c in sorted(chosen.items()):
        e = c["evidence"][0]
        ops.append({"table": "ai_insights", "op": "insert", "fields": {
            "company_id": company_ref, "insight_type": f"Verified {fld}",
            "insight_title": f"{fld}: {c['value'] if not isinstance(c['value'], list) else ', '.join(c['value'])}"[:200],
            "insight": f"{e['snippet']} (source: {e['url']})"[:3900], "confidence_score": c["confidence"],
            "generated_by": agent, "source_url": e["url"], "source_type": "web", "status": "Active"}})
        evidence.extend(c["evidence"])
    return ops, evidence


def company_proposals(state: dict, commit_repo) -> list[dict]:
    url_claims = [c for c in state["claims"].get("procurement", [])
                  if c.get("field") in ("supplier_portal", "procurement_page")]
    chosen, conflicts = best_claims(state["claims"].get("company", []) + url_claims)
    if not chosen:
        return []
    subj = state["subject"]
    existing = (subj.get("existing") or {})
    cid = existing.get("company_id")
    if cid is not None:                       # drop claims the database already holds (nothing to change)
        crow = commit_repo.get_row("companies", {"company_id": cid}) or {}
        prow = commit_repo.get_row("company_profile", {"company_id": cid}) or {}
        def same(old, new):
            return old not in (None, "", [], {}) and json.dumps(old, sort_keys=True).lower() == json.dumps(new, sort_keys=True).lower()
        chosen = {f: c for f, c in chosen.items()
                  if not (f in COMPANY_MAP and same(crow.get(COMPANY_MAP[f]), c["value"]))
                  and not (f in PROFILE_MAP and same(prow.get(PROFILE_MAP[f]), c["value"]))}
        if not chosen:
            return []
    ops: list[dict] = []
    comp_fields = {COMPANY_MAP[f]: c["value"] for f, c in chosen.items() if f in COMPANY_MAP}
    prof_fields = {PROFILE_MAP[f]: c["value"] for f, c in chosen.items() if f in PROFILE_MAP}
    ident = subj.get("identity") or {}
    if ident.get("verified") and ident.get("domain"):
        prof_fields["website_domain"] = ident["domain"]
    ref = cid if cid is not None else NEW_COMPANY
    required, kind = [], "update_company_fields"
    if cid is None:
        kind = "create_company"
        required = ["sector_id"]
        ops.append({"table": "companies", "op": "insert",
                    "fields": {"company_name": subj["name"], **comp_fields}})
    elif comp_fields:
        ops.append({"table": "companies", "op": "update", "key": {"company_id": cid}, "fields": comp_fields})
    if prof_fields:
        has_profile = cid is not None and commit_repo.get_row("company_profile", {"company_id": cid})
        if has_profile:
            ops.append({"table": "company_profile", "op": "update", "key": {"company_id": cid},
                        "fields": prof_fields})
        else:
            mean = round(sum(c["confidence"] for c in chosen.values()) / len(chosen), 2)
            ops.append({"table": "company_profile", "op": "insert",
                        "fields": {"company_id": ref, **prof_fields, "confidence_score": mean}})
    insight_ops, evidence = _insight_ops(ref, chosen, "agentic-company")
    ops.extend(insight_ops)
    return [{"kind": kind, "operations": ops, "evidence": evidence, "required_input": required,
             "conflicts_not_proposed": conflicts,
             "summary": (f"{'Create company' if cid is None else 'Update company ' + str(cid)} "
                         f"'{subj['name']}' with {len(chosen)} verified field(s): {', '.join(sorted(chosen))}")}]


def _blocked(kind: str, summary: str, evidence: list[dict]) -> dict:
    return {"kind": kind, "operations": [], "evidence": evidence, "required_input": [], "summary": summary,
            "blocked": "company_not_in_database: create the company record (and approve it) first"}


def contact_proposals(state: dict, commit_repo) -> list[dict]:
    """One proposal per publicly evidenced contact. Existing unverified CRM contacts can only be given
    provenance (source/confidence/verification date); new people are inserted, never overwritten."""
    cid = (state["subject"].get("existing") or {}).get("company_id")
    out = []
    for c in state["claims"].get("contact", []):
        if c.get("status") != "verified":
            continue
        e = c["evidence"][0]
        now = ev.now_iso()
        if cid is None:
            out.append(_blocked("insert_contact", f"Add contact '{c['name']}' ({c['designation']})", c["evidence"]))
            continue
        found = commit_repo.find_one("contacts", company_id=cid, contact_name=c["name"])
        if found:
            if found.get("source_url"):
                continue                                   # already has provenance
            fields = {"source_url": e["url"], "confidence_score": c["confidence"], "last_verified": now}
            if c["designation"] and found.get("designation") != c["designation"]:
                fields["designation"] = c["designation"]   # may be flagged as an overwrite
            op = {"table": "contacts", "op": "update", "key": {"contact_id": found["contact_id"]}, "fields": fields}
            out.append({"kind": "verify_existing_contact", "operations": [op], "evidence": c["evidence"],
                        "required_input": [],
                        "summary": f"Attach public source to existing CRM contact '{c['name']}'"})
            continue
        fields = {"company_id": cid, "contact_name": c["name"], "designation": c["designation"],
                  "contact_type": "Public source", "decision_maker": bool(c.get("decision_maker")),
                  "confidence_score": c["confidence"], "source_url": e["url"], "last_verified": now}
        for k_src, k_dst in (("department", "department"), ("email", "email"), ("phone", "phone")):
            if c.get(k_src):
                fields[k_dst] = c[k_src]
        if c.get("profile_url") and ev.registrable_domain(c["profile_url"]) == "linkedin.com":
            fields["linkedin"] = c["profile_url"]
        out.append({"kind": "insert_contact", "operations": [{"table": "contacts", "op": "insert", "fields": fields}],
                    "evidence": c["evidence"], "required_input": [],
                    "summary": f"Add publicly documented contact '{c['name']}' ({c['designation']})"})
    return out


def supplier_proposals(state: dict, commit_repo) -> list[dict]:
    """A supplier relationship is only ever proposed with its explicit evidence; never written automatically."""
    cid = (state["subject"].get("existing") or {}).get("company_id")
    out = []
    for c in state["claims"].get("supplier", []):
        if c.get("status") != "verified":
            continue
        label = f"{c['supplier']} -> {c['customer']}" + (f" ({c['component']})" if c.get("component") else "")
        if cid is None:
            out.append(_blocked("insert_supplier_relationship", f"Add supplier relationship {label}", c["evidence"]))
            continue
        e = c["evidence"][0]
        ops = [
            {"table": "suppliers", "op": "insert", "fields": {
                "supplier_name": c["supplier"], "created_by_ai": True, "last_verified": ev.now_iso(),
                **({"components_supplied": c["component"]} if c.get("component") else {})}},
            {"table": "company_suppliers", "op": "insert", "fields": {
                "company_id": cid, "supplier_id": "$supplier_id", "discovered_by": "agentic-supplier",
                "confidence_score": c["confidence"], "remarks": f"{e['snippet']} (source: {e['url']})"[:3900],
                **({"component": c["component"]} if c.get("component") else {})}},
            {"table": "ai_insights", "op": "insert", "fields": {
                "company_id": cid, "insight_type": "Supplier relationship", "insight_title": label[:200],
                "insight": f"{e['snippet']} (source: {e['url']})"[:3900], "confidence_score": c["confidence"],
                "generated_by": "agentic-supplier", "source_url": e["url"], "source_type": "web", "status": "Active"}},
        ]
        out.append({"kind": "insert_supplier_relationship", "operations": ops, "evidence": c["evidence"],
                    "required_input": [], "summary": f"Add evidenced supplier relationship {label}"})
    return out


BUILDERS = [company_proposals, contact_proposals, supplier_proposals]


class ProposalService:
    def __init__(self, store: Store, commit_repo):
        self.store = store
        self.commit = commit_repo

    # ------------------------------------------------------------------ propose
    def propose(self, state: dict) -> list[dict]:
        run_id = state["run_id"]
        out = []
        for builder in BUILDERS:
            for p in builder(state, self.commit):
                out.append(self._finalize(run_id, p, state["ledger"]))
        return out

    def _finalize(self, run_id: str, p: dict, ledger: dict) -> dict:
        if p.get("blocked"):
            key = _hash({"kind": p["kind"], "blocked": p["blocked"], "summary": p["summary"]})
            proposal = {"proposal_id": "p_" + key[:12], "run_id": run_id, "kind": p["kind"], "summary": p["summary"],
                        "operations": [], "evidence": p.get("evidence", []), "required_input": [],
                        "overwrites": [], "conflicts_not_proposed": [], "status": "blocked",
                        "validation": {"ok": False, "errors": [p["blocked"]], "warnings": []},
                        "idempotency_key": key, "content_hash": key}
            self.store.add_proposal(proposal["proposal_id"], run_id, proposal, proposal["validation"], key, "blocked")
            return proposal
        errors, warnings, overwrites = [], [], []
        ops = copy.deepcopy(p["operations"])
        for op in ops:
            existing = self.commit.get_row(op["table"], op["key"]) if op.get("key") else None
            check_op = {**op, "fields": {k: (0 if isinstance(v, str) and v.startswith("$") else v)
                                         for k, v in op["fields"].items()}}
            if op["op"] == "insert" and "sector_id" not in check_op["fields"] and op["table"] == "companies":
                check_op["fields"]["sector_id"] = 0
            errs, _ = wg.validate_operation(check_op, existing)
            errors += [e for e in errs if not (e == "invalid_value:companies.sector_id" and p["required_input"])]
            overwrites += [f"{op['table']}.{c}" for c in wg.detect_overwrites(op, existing)]
        errors += wg.validate_evidence(p["evidence"], ledger)
        for e in p["evidence"]:
            e.setdefault("retrieved_at", ledger.get(ev.doc_id_for(e["url"]), {}).get("retrieved_at"))
        content = {"kind": p["kind"], "operations": ops, "evidence": [(e["url"], e["snippet"]) for e in p["evidence"]]}
        key = _hash({"kind": p["kind"], "operations": ops})
        status = "failed_validation" if errors else "pending"
        proposal = {"proposal_id": "p_" + key[:12], "run_id": run_id, "kind": p["kind"], "summary": p["summary"],
                    "operations": ops, "evidence": p["evidence"], "required_input": p["required_input"],
                    "overwrites": sorted(set(overwrites)), "conflicts_not_proposed": p.get("conflicts_not_proposed", []),
                    "status": status, "validation": {"ok": not errors, "errors": errors, "warnings": warnings},
                    "idempotency_key": key, "content_hash": _hash(content)}
        created = self.store.add_proposal(proposal["proposal_id"], run_id, proposal, proposal["validation"],
                                          key, status)
        if not created:
            stored = self.store.get_proposal(proposal["proposal_id"])
            if stored:
                proposal = stored["change"]
                proposal["status"] = stored["status"]
                proposal["duplicate_of_existing"] = True
        else:
            self.store.audit("proposed", run_id=run_id, proposal_id=proposal["proposal_id"],
                             details={"kind": p["kind"], "status": status, "errors": errors,
                                      "overwrites": proposal["overwrites"]})
        return proposal

    @staticmethod
    def summary(p: dict) -> dict:
        return {"proposal_id": p["proposal_id"], "kind": p["kind"], "status": p["status"],
                "summary": p["summary"], "overwrites": p.get("overwrites", []),
                "required_input": p.get("required_input", []), "errors": (p.get("validation") or {}).get("errors", [])}

    # ------------------------------------------------------------------- decide
    def decide(self, proposal_id: str, approve: bool, *, by: str, reason: str = "",
               approved_overwrite: bool = False, overrides: dict | None = None) -> dict:
        row = self.store.get_proposal(proposal_id)
        if row is None:
            return {"ok": False, "error": "unknown_proposal"}
        if row["status"] != "pending":
            return {"ok": False, "error": f"already_{row['status']}", "status": row["status"]}
        if not by or not by.strip():
            return {"ok": False, "error": "approver_identity_required"}
        change = row["change"]
        if not approve:
            self.store.set_proposal_status(proposal_id, "rejected", decided_by=by, reason=reason)
            self.store.audit("rejected", run_id=row["run_id"], proposal_id=proposal_id, actor=by,
                             details={"reason": reason})
            return {"ok": True, "status": "rejected"}
        if change["overwrites"] and not approved_overwrite:
            return {"ok": False, "error": "overwrite_requires_explicit_approval",
                    "overwrites": change["overwrites"]}
        overrides = overrides or {}
        for needed in change.get("required_input", []):
            if needed not in overrides:
                return {"ok": False, "error": f"missing_required_input:{needed}"}
        if "sector_id" in overrides:
            val = overrides["sector_id"]
            if not isinstance(val, int) or val not in self.commit.sector_ids():
                return {"ok": False, "error": "invalid_sector_id"}
        change["decision"] = {"approved_overwrite": bool(approved_overwrite),
                              "overrides": {k: overrides[k] for k in overrides if k == "sector_id"}}
        self.store.update_proposal_change(proposal_id, change)
        self.store.set_proposal_status(proposal_id, "approved", decided_by=by, reason=reason)
        self.store.audit("approved", run_id=row["run_id"], proposal_id=proposal_id, actor=by,
                         details={"approved_overwrite": bool(approved_overwrite), "overrides": list(overrides)})
        return {"ok": True, "status": "approved"}

    # -------------------------------------------------------------------- apply
    def apply(self, proposal_id: str, *, actor: str = "system") -> dict:
        row = self.store.get_proposal(proposal_id)
        if row is None:
            return {"ok": False, "error": "unknown_proposal"}
        if row["status"] == "applied":
            return {"ok": True, "status": "applied", "noop": True}
        if row["status"] != "approved":
            return {"ok": False, "error": f"not_approved:{row['status']}"}
        change = row["change"]
        decision = change.get("decision") or {}
        if _hash({"kind": change["kind"], "operations": change["operations"]}) != change["idempotency_key"]:
            return self._fail(row, "content_changed_since_validation")
        ops = copy.deepcopy(change["operations"])
        sector = (decision.get("overrides") or {}).get("sector_id")
        created_id = None
        created: dict = {}
        done = []
        try:
            for op in ops:
                fields = dict(op["fields"])
                if op["op"] == "insert" and op["table"] == "companies" and sector is not None:
                    fields["sector_id"] = sector
                fields = {k: (created.get(v, v) if isinstance(v, str) and v.startswith("$") else v)
                          for k, v in fields.items()}
                existing = self.commit.get_row(op["table"], op["key"]) if op.get("key") else None
                check = {**op, "fields": fields}
                errs, cleaned = wg.validate_operation(check, existing)
                if errs:
                    return self._fail(row, "revalidation_failed:" + ",".join(errs), done)
                if wg.detect_overwrites(check, existing) and not decision.get("approved_overwrite"):
                    return self._fail(row, "overwrite_not_approved", done)
                if op["op"] == "insert":
                    if op["table"] == "companies" and self.commit.find_one("companies", company_name=cleaned["company_name"]):
                        return self._fail(row, "company_already_exists", done)
                    if op["table"] == "suppliers":
                        found = self.commit.find_one("suppliers", supplier_name=cleaned["supplier_name"])
                        if found:                                   # reuse, never duplicate a supplier
                            created["$supplier_id"] = found["supplier_id"]
                            done.append({"op": "reuse", "table": "suppliers"})
                            continue
                    if op["table"] == "company_suppliers" and self.commit.find_one(
                            "company_suppliers", company_id=cleaned["company_id"], supplier_id=cleaned["supplier_id"],
                            component=cleaned.get("component")):
                        done.append({"op": "skip_duplicate", "table": "company_suppliers"})
                        continue
                    if op["table"] == "contacts" and self.commit.find_one(
                            "contacts", company_id=cleaned["company_id"], contact_name=cleaned["contact_name"]):
                        return self._fail(row, "contact_already_exists", done)
                    new = self.commit.insert(op["table"], cleaned)
                    if op["table"] == "companies":
                        created_id = new["company_id"]
                        created[NEW_COMPANY] = created_id
                    if op["table"] == "suppliers":
                        created["$supplier_id"] = new["supplier_id"]
                    done.append({"op": "insert", "table": op["table"]})
                else:
                    self.commit.update(op["table"], op["key"], cleaned)
                    done.append({"op": "update", "table": op["table"], "columns": sorted(cleaned)})
        except Exception as exc:  # noqa: BLE001
            return self._fail(row, f"exception:{type(exc).__name__}", done)
        self.store.set_proposal_status(proposal_id, "applied", applied=True)
        self.store.audit("applied", run_id=row["run_id"], proposal_id=proposal_id, actor=actor,
                         details={"operations": done, "created_company_id": created_id})
        return {"ok": True, "status": "applied", "operations": done, "created_company_id": created_id}

    def _fail(self, row, reason, done=None):
        self.store.set_proposal_status(row["proposal_id"], "failed", reason=reason)
        self.store.audit("apply_failed", run_id=row["run_id"], proposal_id=row["proposal_id"],
                         details={"reason": reason, "operations_done": done or []})
        return {"ok": False, "status": "failed", "error": reason, "operations_done": done or []}

"""Narrow, additive write phase for ONE reviewed run: select -> propose -> approve (named authoriser) -> apply -> read back.

Nothing here weakens the existing safeguards - it only uses them (ProposalService: write gate, schema gate, overwrite
detection, content hash, idempotency, audit log) and adds a stricter pre-filter and hard scope checks:

* only change-set items of the pinned company, produced by the Company or Procurement agent;
* only `fill_empty` / `add_new` (never `update_conflict`, never an overwrite, never `no_change`, never contacts/suppliers/lead);
* only verified items whose evidence (source URL + quote) passes the field-meaning check AGAIN (independent of the agent run)
  and whose computed confidence is at least `MIN_CONFIDENCE` (first-party backed);
* only tables/operations on the allow-list below; every operation must be an insert/update keyed to the pinned company;
* the whole phase runs on a client that counts its SELECTs against a hard cap and has no delete/upsert/rpc.

Anything failing a rule is reported with its reason and left alone.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from agentic import field_support
from agentic.guards.select_budget import SelectBudgetExceeded

MIN_CONFIDENCE = 0.60            # 0.40 base + 0.20 first-party: a claim without first-party evidence is not applied
ALLOWED_TABLES = {"companies": {"insert": False, "update": True},
                  "company_profile": {"insert": True, "update": True},
                  "ai_insights": {"insert": True, "update": False}}
ALLOWED_AGENTS = ("company", "procurement")
MAX_PROCUREMENT_INSIGHTS = 10


def url_shape_problem(field: str, value) -> str | None:
    """A stored website must be a site (host + at most two path segments, no query); a LinkedIn value a company page."""
    from urllib.parse import urlsplit
    if field not in ("official_website", "linkedin_url"):
        return None
    u = urlsplit(str(value))
    segs = [x for x in u.path.split("/") if x]
    if u.scheme not in ("http", "https") or not u.netloc or u.query or u.fragment:
        return "url_has_query_or_fragment"
    if field == "official_website" and len(segs) > 2:
        return "website_is_a_deep_page_not_a_site"
    if field == "linkedin_url" and not (segs and segs[0] in ("company", "school")):
        return "not_a_linkedin_company_page"
    return None


@dataclass
class Selection:
    item_ids: list = field(default_factory=list)
    skipped: list = field(default_factory=list)


def select_items(change_set: dict, company_id: int, *, identity_verified: bool,
                 min_confidence: float = MIN_CONFIDENCE, max_insights: int = MAX_PROCUREMENT_INSIGHTS,
                 official_domain: str | None = None) -> Selection:
    sel = Selection()
    if not identity_verified:
        sel.skipped = [{"id": i["id"], "field": i["field"], "reason": "company_identity_not_verified"}
                       for i in change_set.get("items", [])]
        return sel
    if change_set.get("company_id") != company_id:
        raise ValueError("change_set_company_mismatch")
    candidates = []
    for it in change_set.get("items", []):
        def skip(reason):
            sel.skipped.append({"id": it["id"], "field": it["field"], "reason": reason})
        table = (it.get("target") or "").split(".")[0]
        if it.get("agent") not in ALLOWED_AGENTS or it.get("entity") != "company":
            skip("agent_or_entity_out_of_scope")
        elif it.get("action") not in ("fill_empty", "add_new") or it.get("overwrite"):
            skip(f"action_not_additive:{it.get('action')}")
        elif it.get("action") == "fill_empty" and it.get("old_value") not in (None, "", [], {}):
            skip("crm_field_already_populated")
        elif table not in ALLOWED_TABLES:
            skip(f"table_not_authorised:{table or 'none'}")
        elif it.get("validation_status") != "verified" or not it.get("evidence") or not it.get("source_urls"):
            skip("missing_validation_or_source")
        elif (it.get("confidence") or 0) < min_confidence:
            skip(f"low_confidence:{it.get('confidence')}")
        elif any("\ufffd" in str(e.get("snippet", "")) for e in it["evidence"]):
            skip("evidence_text_encoding_damaged")        # a quote with replacement characters is not stored as provenance
        elif url_shape_problem(it["field"], it["proposed_value"]):
            skip(url_shape_problem(it["field"], it["proposed_value"]))
        else:
            ok, why = field_support.check_evidence(it["field"], it["proposed_value"], it["evidence"],
                                                   company=change_set.get("company"), official_domain=official_domain)
            if not ok:
                skip(why)
            else:
                candidates.append(it)
    insights = sorted([c for c in candidates if c["action"] == "add_new"], key=lambda c: (-(c["confidence"] or 0), c["id"]))
    dropped = {c["id"] for c in insights[max_insights:]}
    for c in candidates:
        if c["id"] in dropped:
            sel.skipped.append({"id": c["id"], "field": c["field"], "reason": "insight_cap_reached"})
        else:
            sel.item_ids.append(c["id"])
    return sel


def check_proposal_scope(proposal: dict, company_id: int) -> list[str]:
    """Reasons a proposal must NOT be applied (empty = every operation is inside the authorised scope)."""
    problems = []
    if proposal.get("overwrites"):
        problems.append("contains_overwrite")
    if not proposal.get("operations"):
        problems.append("no_operations")
    for op in proposal.get("operations", []):
        table, kind = op.get("table"), op.get("op")
        if table not in ALLOWED_TABLES or not ALLOWED_TABLES[table].get(kind):
            problems.append(f"operation_not_authorised:{kind}:{table}")
            continue
        if kind == "update" and (op.get("key") or {}) != {"company_id": company_id}:
            problems.append(f"update_not_keyed_to_company:{table}")
        if table != "companies" and (op.get("fields") or {}).get("company_id") not in (company_id, None):
            problems.append(f"insert_for_other_company:{table}")
        if kind == "insert" and table != "companies" and (op.get("fields") or {}).get("company_id") != company_id:
            problems.append(f"insert_not_for_company:{table}")
    return problems


def apply_reviewed(service, state: dict, change_set: dict, company_id: int, *, authoriser: str,
                   identity_verified: bool, official_domain: str | None = None, only_ids: set | None = None) -> dict:
    """Propose the selected items, approve each in-scope proposal in the name of `authoriser`, apply, and report.
    `service` is a ProposalService bound to the (counted) commit repository."""
    sel = select_items(change_set, company_id, identity_verified=identity_verified, official_domain=official_domain)
    if only_ids is not None:                       # the person picked items: everything else stays unwritten
        sel.skipped += [{"id": i, "field": next((x["field"] for x in change_set.get("items", []) if x["id"] == i), None),
                         "reason": "not_selected_by_the_approver"} for i in sel.item_ids if i not in only_ids]
        sel.item_ids = [i for i in sel.item_ids if i in only_ids]
    out = {"selected_items": sel.item_ids, "skipped_items": sel.skipped, "proposals": [], "operations": [],
           "writes": 0, "stale_items": [], "blocked": None}
    if not sel.item_ids:
        return out
    try:
        res = service.propose_from_change_set(state, change_set, sel.item_ids)
        out["stale_items"] = res["stale_items"]
        out["not_proposed"] = res["not_proposed"]
        for p in res["proposals"]:
            row = {"proposal_id": p["proposal_id"], "kind": p["kind"], "summary": p["summary"], "status": p["status"],
                   "items": p.get("items", []), "errors": (p.get("validation") or {}).get("errors", []),
                   "schema_unconfirmed": p.get("schema_unconfirmed", []), "applied": False}
            out["proposals"].append(row)
            if p["status"] != "pending":
                row["not_applied"] = f"status_{p['status']}"
                continue
            problems = check_proposal_scope(p, company_id)
            if problems:
                row["not_applied"] = ",".join(problems)
                continue
            dec = service.decide(p["proposal_id"], True, by=authoriser,
                                 reason="single authorised AIRBUS India enrichment: additive, validated items only")
            if not dec.get("ok"):
                row["not_applied"] = "decision_refused:" + str(dec.get("error"))
                continue
            done = service.apply(p["proposal_id"], actor=authoriser)
            row["apply"] = {k: done.get(k) for k in ("ok", "status", "error", "operations", "operations_done")}
            row["applied"] = bool(done.get("ok"))
            for d in (done.get("operations") or done.get("operations_done") or []):
                if d.get("op") in ("insert", "update"):
                    out["operations"].append(d)
                    out["writes"] += 1
    except SelectBudgetExceeded as exc:        # BaseException by design: stop writing, keep what is already recorded
        out["blocked"] = f"write_phase_select_budget_exceeded:{exc.limit}"
    return out


def guarded_write_phase(commit_repo, store, schema, state: dict, company_id: int, authoriser: str, budget,
                        only_ids: set | None = None) -> dict:
    """The complete narrow write phase on a repository whose SELECTs are counted by `budget`: propose -> approve in the
    name of `authoriser` -> apply -> read back. Shared by the command-line runner and the web platform."""
    from agentic.proposals import ProposalService
    svc = ProposalService(store, commit_repo, schema=schema)
    cs = (state.get("report") or {}).get("change_set") or {}
    identity = bool(((state.get("subject") or {}).get("identity") or {}).get("verified"))
    domain = ((state.get("subject") or {}).get("identity") or {}).get("domain")
    out: dict = {"select_cap": budget.limit}
    try:
        out.update(apply_reviewed(svc, state, cs, company_id, authoriser=authoriser, identity_verified=identity,
                                  official_domain=domain, only_ids=only_ids))
        if out.get("writes"):                                     # read back what is now in the database
            company = commit_repo.get_row("companies", {"company_id": company_id}) or {}
            profile = commit_repo.get_row("company_profile", {"company_id": company_id}) or {}
            insights = [r for r in commit_repo.list_rows("ai_insights", company_id=company_id)
                        if str(r.get("generated_by", "")).startswith("agentic-")]
            out["read_back"] = {"companies": {k: company.get(k) for k in ("company_id", "company_name", "website", "linkedin_url")},
                                "company_profile": profile,
                                "ai_insights_by_agentic": [{k: r.get(k) for k in ("insight_id", "insight_type", "insight_title",
                                                                                   "source_url", "confidence_score")}
                                                           for r in insights]}
    except SelectBudgetExceeded as exc:                           # BaseException by design: record it, stop writing
        out["blocked"] = f"write_phase_select_budget_exceeded:{exc.limit}"
    out["write_selects_used"] = budget.used
    return out

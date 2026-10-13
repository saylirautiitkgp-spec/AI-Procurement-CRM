"""Deterministic final report and live tracker snapshot.

The report is assembled in Python from validated claims only - there is no
free-text LLM summary that could smuggle in unverified statements. Database
values without provenance are shown separately and labelled unverified."""

from __future__ import annotations

from collections import defaultdict

from agentic import gaps as gp
from agentic.evidence import now_iso
from agentic.guards.budget import usage_summary
from agentic.security import redact_obj

LIMIT_REASONS = {"db_select_budget_exceeded", "wall_clock_exceeded", "max_llm_calls_reached", "max_tool_calls_reached",
                 "max_tokens_reached", "max_supervisor_steps", "llm_quota", "llm_rate_limit", "llm_billing",
                 "supervisor_invalid_routing", "llm_other", "llm_transient", "llm_invalid",
                 "planned_agent_blocked"}


def _ev(e: dict) -> dict:
    return {"url": e["url"], "snippet": e["snippet"], "retrieved_at": e["retrieved_at"]}


def _claim_view(c: dict) -> dict:
    return {"value": c["value"], "confidence": c["confidence"], "confidence_basis": c["confidence_basis"],
            "evidence": [_ev(e) for e in c["evidence"]]}


def _group(claims: list[dict], key: str) -> dict:
    out = defaultdict(list)
    for c in claims:
        if c.get("status") == "verified":
            out[c.get(key)].append(c)
    return out


def failed_tool_calls(trace: list[dict]) -> list[dict]:
    out = []
    for ev in trace:
        if ev["type"] == "tool_result" and not ev["data"].get("ok"):
            out.append({"agent": ev["agent"], "tool": ev["data"].get("tool"),
                        "error": (ev["data"].get("error") or {}).get("code")})
    return out


def _rkey(agent: str, item: dict) -> str:
    """Identity of a claim for 'was this rejection later resolved?' (field/kind, or the person/supplier for lists)."""
    if agent in ("contact", "supplier"):
        return str(item.get("value", "")).lower()
    return str(item.get("field") or item.get("kind") or "")


def status_reasons(state: dict) -> list[str]:
    """Machine-readable reasons a run is not a clean success. Empty list = clean `completed`.

    Distinguishes (a) evidence-validation problems (the model's claims were rejected), (b) hard limits / outages and
    (c) tool failures from (d) a legitimate empty result (agent searched, found nothing, submitted no claims) which is
    NOT a reason: 'not found' is a valid finding."""
    reasons: list[str] = []
    if state.get("stop_reason") in LIMIT_REASONS:
        reasons.append(f"limit:{state['stop_reason']}")
    b = state.get("budgets", {})
    if b.get("search_disabled"):
        reasons.append(f"search_unavailable:{b.get('search_disabled_reason')}")
    accepted_total = sum(len(v) for v in state.get("claims", {}).values() if isinstance(v, list))
    if accepted_total == 0 and failed_tool_calls(state.get("trace", [])):
        reasons.append("no_verified_results_and_tool_failures")      # may be an outage, not a true "not found"
    results = state.get("agent_results", {})
    for name, r in results.items():
        if r.get("end_reason") not in (None, "submitted"):
            reasons.append(f"agent_not_finished:{name}:{r.get('end_reason')}")
        if r.get("rejected", 0) and not r.get("accepted", 0):
            reasons.append(f"evidence_rejected_all:{name}")
    for name, items in state.get("rejected", {}).items():
        if not items or f"evidence_rejected_all:{name}" in reasons:
            continue
        accepted_keys = {_rkey(name, c) for c in state.get("claims", {}).get(name, []) if isinstance(c, dict)}
        residual = sorted({_rkey(name, x) for x in items} - accepted_keys)
        if residual:
            reasons.append(f"evidence_rejected_some:{name}:{','.join(residual)[:120]}")
    return reasons


def run_status(state: dict) -> str:
    if state.get("status") == "failed":
        return "failed"
    if not state.get("agent_results"):
        if state.get("stop_reason") == "nothing_to_research":
            return "completed"
        return "failed" if state.get("errors") else "partial"
    return "partial" if status_reasons(state) else "completed"


def coverage_limitations(state: dict) -> list[str]:
    out = []
    b = state["budgets"]
    reason = state.get("stop_reason")
    if reason in LIMIT_REASONS:
        out.append(f"Run stopped early: {reason}. Results are partial and can be extended by a continuation run.")
    if b.get("search_disabled"):
        out.append(f"Web search became unavailable ({b.get('search_disabled_reason')}); "
                   "later research steps could not run.")
    for name, r in state.get("agent_results", {}).items():
        if r.get("end_reason") not in (None, "submitted"):
            out.append(f"{name} agent ended without a normal submission ({r.get('end_reason')}).")
    return out


def build_report(state: dict) -> dict:
    s = state["subject"]
    claims = state["claims"]
    status = run_status(state)
    reasons = status_reasons(state) + ([] if state.get("agent_results") else ["no_agent_completed"])
    ident = s.get("identity") or {"verified": False}
    match = s.get("match") or {"status": "not_run", "matches": [], "assumptions": []}

    # ---- company profile + conflicts ------------------------------------
    profile, conflicts = {}, []
    for fld, items in _group(claims.get("company", []), "field").items():
        items.sort(key=lambda c: -c["confidence"])
        distinct = {str(c["value"]).strip().lower() for c in items}
        profile[fld] = {**_claim_view(items[0]),
                        "alternatives": [_claim_view(c) for c in items[1:]] if len(distinct) > 1 else []}
        if len(distinct) > 1:
            conflicts.append({"field": fld, "values": [_claim_view(c) for c in items]})

    ex = s.get("existing") or {}
    database_context = None
    if ex:
        database_context = {
            "provenance": "database_unverified",
            "note": "Values stored in the CRM database without source evidence. Not verified by this run.",
            "company_id": ex.get("company_id"),
            "sections_loaded": ex.get("loaded_sections"),     # lazy CRM: only these were read for this run
            "fields": {k: v for k, v in ex.get("fields", {}).items() if v not in (None, "")},
            "products": ex.get("products", []), "components": ex.get("components", []),
            "contact_summary": ex.get("contact_summary"),
            "suppliers": ex.get("suppliers", []), "lead_status": ex.get("lead_status"),
        }

    # ---- procurement ------------------------------------------------------
    proc = _group(claims.get("procurement", []), "kind")
    procurement = {k: [_claim_view(c) for c in v] for k, v in proc.items()} or None

    contacts = [{**{k: c.get(k) for k in ("name", "designation", "department", "email", "phone", "profile_url")},
                 **_claim_view(c)} for c in claims.get("contact", []) if c.get("status") == "verified"]
    suppliers = [{"supplier": c.get("supplier"), "customer": c.get("customer"),
                  "component": c.get("component"), **_claim_view(c)}
                 for c in claims.get("supplier", []) if c.get("status") == "verified"]
    lead = claims.get("lead")

    # ---- sources ----------------------------------------------------------
    used: dict = {}
    for agent, items in claims.items():
        if agent == "lead":
            continue
        for c in items:
            for e in c.get("evidence", []):
                u = used.setdefault(e["url"], {"url": e["url"], "title": e.get("title", ""),
                                               "retrieved_at": e["retrieved_at"], "used_for": []})
                label = f"{agent}:{c.get('field') or c.get('kind') or 'item'}"
                if label not in u["used_for"]:
                    u["used_for"].append(label)

    unresolved = sorted({u for r in state["agent_results"].values() for u in r.get("unresolved", [])})
    rejected_summary = {a: len(v) for a, v in state.get("rejected", {}).items() if v}
    gaps = gp.company_gaps(s, claims)

    # ---- next actions -------------------------------------------------------
    nxt = []
    if not ident.get("verified"):
        nxt.append("Official website could not be confirmed from evidence - verify the organisation manually.")
    if not contacts:
        nxt.append("No publicly documented procurement decision-maker was verified. Existing database "
                   "contacts remain unverified; consider manual outreach research.")
    if any(g["status"] == "unverified_db" for g in gaps):
        nxt.append("Legacy database values exist without sources; review and approve the proposed "
                   "verified updates (or re-verify manually).")
    if state.get("proposals"):
        pending = [p for p in state["proposals"] if p.get("status") in ("pending", "awaiting_approval")]
        if pending:
            nxt.append(f"{len(pending)} proposed database change(s) await your approval; nothing was written.")
    if match["status"] == "ambiguous":
        nxt.append("Multiple database rows match this name; review the duplicate/merge plan before any merge.")
    if state["budgets"].get("search_disabled"):
        nxt.append("Search quota/rate limit reached; resume after the quota resets or lower search depth.")
    if state.get("stop_reason") in LIMIT_REASONS:
        nxt.append("Run hit a hard limit; start a continuation run to extend the research.")

    summary = []
    site = profile.get("official_website")
    summary.append(f"{s['name']}: " + (f"official website {site['value']} (confidence {site['confidence']})."
                                       if site else "official website not confirmed."))
    for f in ("headquarters", "industry"):
        if f in profile:
            summary.append(f"{f.replace('_', ' ').title()}: {profile[f]['value']} (confidence {profile[f]['confidence']}).")
    summary.append(f"{len(contacts)} verified contact(s); {len(suppliers)} evidenced supplier relationship(s); "
                   f"{len(used)} source(s) used.")

    return {
        "run": {"run_id": state["run_id"], "company_submitted": s["name"], "objective": state["objective"],
                "status": status, "status_reasons": reasons, "stop_reason": state.get("stop_reason"),
                "generated_at": now_iso()},
        "summary": " ".join(summary),
        "database_match": {"status": match["status"], "matches": match["matches"],
                           "assumptions": match.get("assumptions", [])},
        "identity": {"verified": bool(ident.get("verified")), "website": ident.get("website"),
                     "domain": ident.get("domain"), "reason": ident.get("reason"),
                     "reasoning": ident.get("reasoning")},
        "company_profile": profile or None,
        "database_context": database_context,
        "procurement": procurement,
        "contacts": contacts or None,
        "suppliers": suppliers or None,
        "lead_priority": lead,
        "sources": list(used.values()),
        "documents_retrieved": len(state["ledger"]),
        "gaps": {"company_fields": gaps, "other_areas": gp.other_agent_gaps(s, claims, state["agent_results"]),
                 "unresolved": unresolved, "conflicts": conflicts,
                 "claims_rejected_by_validation": rejected_summary,
                 "rejected_claims": [{"agent": a, "field": x.get("field") or x.get("kind"), "reasons": x["reasons"],
                                      "value_unverified": str(x.get("value"))[:100]}
                                     for a, items in state.get("rejected", {}).items() for x in items][:30],
                 "failed_tool_calls": failed_tool_calls(state.get("trace", []))},
        "coverage_limitations": coverage_limitations(state),
        "agents": {n: {k: r.get(k) for k in ("status", "end_reason", "accepted", "rejected", "notes",
                                              "tool_calls", "llm_calls")}
                   for n, r in state["agent_results"].items()},
        "assumptions": match.get("assumptions", []),
        "usage": usage_summary(state["budgets"]),
        "proposed_changes": [{k: p.get(k) for k in ("proposal_id", "kind", "status", "summary")}
                             for p in state.get("proposals", [])],
        "next_actions": nxt,
    }


def build_tracker(state: dict) -> dict:
    s = state.get("subject", {})
    b = state.get("budgets", {})
    trace = state.get("trace", [])
    validated_urls = {e["url"] for items in state.get("claims", {}).values() if isinstance(items, list) for c in items
                      for e in c.get("evidence", [])}
    steps = []
    for ev in trace[-80:]:
        d = ev["data"]
        steps.append({"seq": ev["seq"], "ts": ev["ts"], "type": ev["type"], "agent": ev["agent"],
                      "summary": d.get("tool") or d.get("next") or d.get("reason") or d.get("kind") or ""})
    tracker = {
        "run_id": state.get("run_id"), "company_submitted": s.get("name"), "objective": state.get("objective"),
        "database_record_found": (s.get("match") or {}).get("status") in ("found", "ambiguous"),
        "database_match": (s.get("match") or {}).get("status"),
        "status": state.get("status"), "current_agent": state.get("current_agent"),
        "current_objective": state.get("agent_objective"),
        "agent_transitions": state.get("route_history", []),
        "agents": {n: {"runs": state.get("agent_runs", {}).get(n, 0), "status": r.get("status"),
                       "end_reason": r.get("end_reason")} for n, r in state.get("agent_results", {}).items()},
        "steps": steps, "events_total": len(trace),
        "usage": usage_summary(b) if b else {},
        "search_disabled": b.get("search_disabled"),
        "sources": {"discovered": len(state.get("ledger", {})), "validated": len(validated_urls)},
        "gaps": gp.company_gaps(s, state.get("claims", {})) if s.get("match") else [],
        "errors": state.get("errors", []), "stop_reason": state.get("stop_reason"),
        "database_change": _change_status(state.get("proposals", [])),
        "status_reasons": state.get("status_reasons", []),
        "updated_at": now_iso(),
    }
    return redact_obj(tracker, max_str=300)


def _change_status(proposals: list[dict]) -> str:
    if not proposals:
        return "none_proposed"
    statuses = {p.get("status") for p in proposals}
    if statuses <= {"applied"}:
        return "applied"
    if statuses & {"pending", "awaiting_approval"}:
        return "awaiting_approval"
    if statuses <= {"rejected", "failed_validation", "blocked"}:
        return "proposed_not_applied"
    return "mixed"

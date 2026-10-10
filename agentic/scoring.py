"""Deterministic, explainable lead-priority score.

PROVISIONAL HEURISTIC: weights are an engineering starting point, not a validated model. They have not
been calibrated against real outcomes and must be reviewed with procurement stakeholders. The score
only reflects how much verified, procurement-relevant information exists about the organisation - it
does not predict deal outcomes. The LLM cannot change the result; it can only read it.
"""

from __future__ import annotations

from agentic import gaps as gp

WEIGHTS = {
    "identity_confirmed": 10,
    "profile_completeness": 15,
    "procurement_signals": 20,
    "procurement_decision_maker": 25,
    "component_relevance": 15,
    "supplier_relationships": 10,
    "reachable_public_source": 5,
}
BANDS = ((70, "high"), (40, "medium"), (0, "low"))


def _band(score: float) -> str:
    return next(name for floor, name in BANDS if score >= floor)


def score_lead(state: dict) -> dict:
    s = state["subject"]
    claims = state["claims"]
    ex = s.get("existing") or {}
    ident = bool((s.get("identity") or {}).get("verified"))
    company_done = gp.verified_fields(claims, "company")
    proc = gp.verified_fields(claims, "procurement")
    contacts = [c for c in claims.get("contact", []) if c.get("status") == "verified"]
    suppliers = [c for c in claims.get("supplier", []) if c.get("status") == "verified"]
    factors, missing = [], []

    def add(name, earned, basis, refs=()):
        factors.append({"factor": name, "weight": WEIGHTS[name], "earned": round(earned, 2), "basis": basis,
                        "evidence_urls": sorted({u for u in refs})[:5]})

    add("identity_confirmed", WEIGHTS["identity_confirmed"] if ident else 0,
        "first-party website confirmed from evidence" if ident else "identity not confirmed")
    if not ident:
        missing.append("confirmed official website / identity")

    frac = len(company_done) / len(gp.COMPANY_FIELDS)
    add("profile_completeness", WEIGHTS["profile_completeness"] * frac,
        f"{len(company_done)}/{len(gp.COMPANY_FIELDS)} tracked profile fields verified")
    if frac < 1:
        missing.append("unverified profile fields: " + ", ".join(
            f for f in gp.COMPANY_FIELDS if f not in company_done))

    has_page = bool(proc.get("supplier_portal") or proc.get("procurement_page"))
    has_cat = bool(proc.get("purchasing_category") or proc.get("purchased_component"))
    pts = (10 if has_page else 0) + (10 if has_cat else 0)
    add("procurement_signals", pts, ("procurement page/portal verified; " if has_page else "no verified procurement page; ")
        + ("purchasing categories verified" if has_cat else "no verified purchasing categories"),
        [e["url"] for k in ("supplier_portal", "procurement_page") for c in proc.get(k, []) for e in c["evidence"]])
    if not has_page:
        missing.append("procurement page / supplier portal")

    add("procurement_decision_maker", WEIGHTS["procurement_decision_maker"] if contacts else 0,
        f"{len(contacts)} publicly documented procurement contact(s) with evidence" if contacts
        else "none verified (legacy CRM contacts are unverified and not counted)",
        [e["url"] for c in contacts for e in c["evidence"]])
    if not contacts:
        missing.append("publicly documented procurement decision-maker")

    verified_comp = proc.get("purchased_component", [])
    db_comp = ex.get("components") or []
    if verified_comp:
        add("component_relevance", WEIGHTS["component_relevance"] * min(len(verified_comp), 5) / 5,
            f"{len(verified_comp)} purchased component(s) verified", [e["url"] for c in verified_comp for e in c["evidence"]])
    elif db_comp:
        add("component_relevance", WEIGHTS["component_relevance"] * min(len(db_comp), 5) / 5 * 0.5,
            f"{len(db_comp)} component(s) in CRM, UNVERIFIED - counted at half weight")
        missing.append("verification of CRM-listed components")
    else:
        add("component_relevance", 0, "no component information")
        missing.append("components purchased")

    add("supplier_relationships", WEIGHTS["supplier_relationships"] if suppliers else 0,
        f"{len(suppliers)} evidenced supplier relationship(s)" if suppliers else "none evidenced",
        [e["url"] for c in suppliers for e in c["evidence"]])
    n_sources = len({e["url"] for items in claims.values() if isinstance(items, list) for c in items
                     for e in c.get("evidence", [])})
    add("reachable_public_source", WEIGHTS["reachable_public_source"] if n_sources >= 3 else
        WEIGHTS["reachable_public_source"] * n_sources / 3, f"{n_sources} distinct validated source(s)")

    total = round(sum(f["earned"] for f in factors), 1)
    if not ident and not ex:
        sufficiency = "insufficient"
    elif frac == 0 and not contacts and not proc:
        sufficiency = "limited"
    else:
        sufficiency = "sufficient"
    return {"score": None if sufficiency == "insufficient" else total,
            "band": "insufficient_data" if sufficiency == "insufficient" else _band(total),
            "max_score": sum(WEIGHTS.values()), "data_sufficiency": sufficiency, "factors": factors,
            "missing_information": missing,
            "disclaimer": "Provisional, uncalibrated heuristic reflecting verified-information coverage only."}

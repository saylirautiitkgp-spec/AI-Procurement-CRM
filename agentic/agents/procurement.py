"""Procurement Intelligence Agent: how and what the company buys, and where suppliers register."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from agentic import evidence as ev
from agentic import field_support
from agentic.agents.common import AgentSpec, BaseFindings, EvidenceRefModel, ProcessResult, validate_refs
from agentic.prefetch import agent_prefetch

URL_KINDS = ("supplier_portal", "procurement_page")
TEXT_KINDS = ("purchasing_category", "purchased_component", "process_note")


class ProcurementClaim(BaseModel):
    kind: Literal["supplier_portal", "procurement_page", "purchasing_category", "purchased_component",
                  "process_note"]
    value: str = Field(min_length=3, max_length=300,
                       description="URL for portal/page kinds; short text for the others.")
    evidence: list[EvidenceRefModel] = Field(min_length=1, max_length=3)


class ProcurementFindings(BaseFindings):
    items: list[ProcurementClaim] = Field(default_factory=list, max_length=20)


def process_procurement(state: dict, findings: dict, deps) -> ProcessResult:
    res = ProcessResult()
    ledger = state["ledger"]
    ident = state["subject"].get("identity") or {}
    seen = {(c["kind"], str(c["value"]).lower()) for c in state["claims"].get("procurement", [])}
    for it in findings.get("items", []):
        kind, value = it["kind"], it["value"].strip()
        label = {"kind": kind, "value": value}
        if kind in URL_KINDS:
            canon = ev.canonical_url(value)
            chk = validate_refs(ledger, it["evidence"], value, ident.get("domain"), require_value_support=False)
            if not canon or not chk["ok"]:
                res.rejected.append({**label, "reasons": chk["reasons"] if not chk["ok"] else ["invalid_url"]})
                continue
            quoted = any(canon.rstrip("/").lower() in e["snippet"].lower().replace(" ", "") or
                         value.rstrip("/").lower() in e["snippet"].lower() for e in chk["evidence"])
            if ev.find_doc(ledger, canon) is None and not quoted:
                res.rejected.append({**label, "reasons": ["url_neither_retrieved_nor_quoted_in_snippet"]})
                continue
            value = canon
        else:
            chk = validate_refs(ledger, it["evidence"], value, ident.get("domain"))
            if not chk["ok"]:
                res.rejected.append({**label, "reasons": chk["reasons"]})
                continue
        sup_ok, sup_reason = field_support.check_evidence(kind, value, chk["evidence"],
                                                           company=state["subject"].get("name"),
                                                           official_domain=ident.get("domain"))
        if not sup_ok:                                     # the quote must say what this finding claims it is
            res.rejected.append({**label, "reasons": [sup_reason]})
            continue
        if (kind, value.lower()) in seen:
            continue
        seen.add((kind, value.lower()))
        res.accepted.append({"agent": "procurement", "kind": kind, "field": kind, "value": value,
                             "status": "verified", "evidence": chk["evidence"], "confidence": chk["confidence"],
                             "confidence_basis": chk["basis"]})
    return res


def procurement_task(state: dict) -> str:
    s = state["subject"]
    ident = s.get("identity") or {}
    ex = s.get("existing") or {}
    lines = [f"Company: {s['name']} (official site: {ident.get('website')})",
             f"User objective: {state['objective']}",
             "Find: (1) supplier registration portal URL, (2) procurement/sourcing page URL, (3) purchasing "
             "categories, (4) specific components/materials the company says it buys, (5) notable process notes "
             "(e.g. vendor qualification steps). Only report what pages explicitly say."]
    if ex.get("components"):
        lines.append(f"CRM lists these components (UNVERIFIED, no source): {ex['components'][:10]}. Verify them "
                     "from public pages or ignore them; do not repeat them as facts.")
    if ex.get("products"):
        lines.append(f"CRM lists these products (UNVERIFIED): {ex['products'][:8]}")
    return "\n".join(lines)


PROCUREMENT_AGENT = AgentSpec(
    name="procurement", title="Procurement Intelligence Agent",
    goal="Find the company's supplier portal, procurement pages, purchasing categories and purchased components.",
    role_prompt=("Check the company's own supplier/procurement/vendor pages first (follow same-site links), then "
                 "corroborate with public sources. For URL kinds the URL must be a page you retrieved or one quoted "
                 "in your snippet."),
    findings_model=ProcurementFindings, finish_tool="submit_procurement_findings",
    prefetch=agent_prefetch("procurement"),
    tool_names=("web_search", "fetch_page", "get_crm_context", "submit_procurement_findings"),
    process=process_procurement, task=procurement_task)

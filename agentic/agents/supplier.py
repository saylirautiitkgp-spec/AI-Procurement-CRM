"""Supplier & Relationship Intelligence Agent.

A relationship is accepted ONLY if one verbatim snippet from a retrieved page names the supplier AND this
company AND states a supplying relationship. Nothing is inferred from similar products or industries."""

from __future__ import annotations

import re

from pydantic import BaseModel, Field

from agentic import evidence as ev
from agentic import names
from agentic.agents.common import AgentSpec, BaseFindings, EvidenceRefModel, ProcessResult, validate_refs

RELATION_CUES = re.compile(r"(?i)\b(supplies|supplied|supplier|suppliers|supply|vendor|awarded|contract|"
                           r"order for|partner(?:ed)? with|(?:sourced|sources|procures|buys|purchases)\b[^.\n]*\bfrom)\b")


class SupplierClaim(BaseModel):
    supplier_name: str = Field(min_length=2, max_length=150)
    customer_name: str = Field(min_length=2, max_length=150, description="The company being researched.")
    component: str | None = Field(default=None, max_length=150, description="What is supplied, if stated.")
    evidence: list[EvidenceRefModel] = Field(min_length=1, max_length=3)


class SupplierFindings(BaseFindings):
    relationships: list[SupplierClaim] = Field(default_factory=list, max_length=10)


def _mentions(text_norm: str, name: str, threshold: float = 0.8) -> bool:
    toks = names.distinctive_tokens(name)
    return bool(toks) and sum(t in text_norm for t in toks) / len(toks) >= threshold


def _first_pos(text_norm: str, name: str) -> int:
    toks = names.distinctive_tokens(name)
    hits = [text_norm.find(t) for t in toks if t in text_norm]
    return min(hits) if hits else -1


_SUPPLIER_FIRST = re.compile(r"(?i)\b(supplies|supplied|supply|supplier to|supplier of)\b")
_COMPANY_FIRST = re.compile(r"(?i)\b(sourced|sources|procures|buys|purchases)\b[^.\n]*\bfrom\b")


def states_relationship(snippet: str, supplier: str, company: str) -> bool:
    """True only if ONE sentence links the two parties in the right direction, e.g.
    '<supplier> supplies X to <company>' or '<company> sources X from <supplier>'."""
    for sentence in re.split(r"(?<=[.!?])\s+|\n", snippet):
        n = ev.norm_text(sentence)
        if not (_mentions(n, supplier) and _mentions(n, company) and RELATION_CUES.search(sentence)):
            continue
        sp, cp = _first_pos(n, supplier), _first_pos(n, company)
        if _SUPPLIER_FIRST.search(sentence):
            if sp < cp and sp >= 0:
                return True
        elif _COMPANY_FIRST.search(sentence):
            if cp < sp and cp >= 0:
                return True
        else:
            return True        # 'partnered with', 'awarded ... contract': both parties in the same sentence
    return False


def process_supplier(state: dict, findings: dict, deps) -> ProcessResult:
    res = ProcessResult()
    ledger = state["ledger"]
    ident = state["subject"].get("identity") or {}
    company = state["subject"]["name"]
    seen = {(c["supplier"].lower(), (c.get("component") or "").lower()) for c in state["claims"].get("supplier", [])}
    for r in findings.get("relationships", []):
        label = {"kind": "supplier_relationship", "value": f"{r['supplier_name']} -> {r['customer_name']}"}
        chk = validate_refs(ledger, r["evidence"], r["supplier_name"], ident.get("domain"), require_value_support=False)
        if not chk["ok"]:
            res.rejected.append({**label, "reasons": chk["reasons"]})
            continue
        reasons = []
        if names.similarity(r["customer_name"], company) < 0.75:
            reasons.append("relationship_not_about_this_company")
        if names.similarity(r["supplier_name"], company) >= 0.9:
            reasons.append("supplier_is_the_company_itself")
        ok_snip = [e for e in chk["evidence"] if states_relationship(e["snippet"], r["supplier_name"], company)]
        if not ok_snip:
            reasons.append("no_single_snippet_names_both_parties_and_the_relationship")
        elif r.get("component") and not any(ev.value_supported(r["component"], e["snippet"]) for e in ok_snip):
            reasons.append("component_not_in_relationship_snippet")
        if reasons:
            res.rejected.append({**label, "reasons": reasons})
            continue
        key = (r["supplier_name"].lower(), (r.get("component") or "").lower())
        if key in seen:
            continue
        seen.add(key)
        conf, basis = ev.claim_confidence(ok_snip, ident.get("domain"))
        res.accepted.append({"agent": "supplier", "kind": "supplier_relationship", "field": "supplier_relationship",
                             "value": f"{r['supplier_name']} -> {company}", "supplier": r["supplier_name"],
                             "customer": company, "component": r.get("component"), "status": "verified",
                             "evidence": ok_snip, "confidence": conf, "confidence_basis": basis})
    return res


def supplier_task(state: dict) -> str:
    s = state["subject"]
    ex = s.get("existing") or {}
    proc = [c["value"] for c in state["claims"].get("procurement", []) if c.get("kind") == "purchased_component"]
    lines = [f"Company: {s['name']}", f"User objective: {state['objective']}",
             "Find suppliers that public sources explicitly say supply this company (optionally for a named "
             "component). Do not infer relationships from similar products, industries or partnerships of other firms."]
    if proc:
        lines.append(f"Components verified as purchased by this company: {proc[:8]}")
    if ex.get("components"):
        lines.append(f"CRM component list (UNVERIFIED): {ex['components'][:10]}")
    if ex.get("products"):
        lines.append(f"CRM product list (UNVERIFIED): {ex['products'][:8]}")
    if ex.get("suppliers"):
        lines.append(f"CRM already lists supplier names (UNVERIFIED): {[x.get('supplier_name') for x in ex['suppliers']][:8]}")
    return "\n".join(lines)


SUPPLIER_AGENT = AgentSpec(
    name="supplier", title="Supplier & Relationship Intelligence Agent",
    goal="Find supplier relationships of this company that public sources explicitly state.",
    role_prompt=("Search for statements like '<Supplier> supplies <component> to <Company>' in news, press releases, "
                 "annual reports and the company's own pages. Quote the single sentence that names both parties and "
                 "the relationship. If no explicit statement exists, report none."),
    findings_model=SupplierFindings, finish_tool="submit_supplier_findings",
    tool_names=("web_search", "fetch_page", "get_crm_context", "submit_supplier_findings"),
    process=process_supplier, task=supplier_task)

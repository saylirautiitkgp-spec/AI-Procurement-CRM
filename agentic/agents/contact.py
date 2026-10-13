"""Contact Intelligence Agent: publicly documented procurement/sourcing decision-makers.

Rules enforced in Python (not by the prompt): the person's name AND role must appear in the same
verbatim snippet from a page retrieved in this run; emails/phones are accepted only if they appear
literally in a cited snippet; roles must be procurement-related; nothing is ever constructed."""

from __future__ import annotations

import re

from pydantic import BaseModel, Field

from agentic import evidence as ev
from agentic.agents.common import AgentSpec, BaseFindings, EvidenceRefModel, ProcessResult, validate_refs
from agentic.prefetch import agent_prefetch

ROLE_WORDS = ("procure", "purchas", "sourcing", "supply chain", "buyer", "vendor", "materials management",
              "supplier")
ROLE_AS_NAME = re.compile(r"(?i)\b(head|manager|director|team|department|officer|vp|chief|procurement|purchase|"
                          r"purchasing|sourcing|buyer|n/a|unknown|contact)\b")
HEAD_WORDS = ("head", "director", "vp", "vice president", "chief", "general manager", "gm")


class ContactClaim(BaseModel):
    name: str = Field(min_length=3, max_length=120, description="Full name of a real person, as printed on the page.")
    designation: str = Field(min_length=3, max_length=160, description="Job title exactly as printed on the page.")
    department: str | None = Field(default=None, max_length=120)
    email: str | None = Field(default=None, max_length=200,
                              description="ONLY if the exact address is printed in your snippet. Never guess.")
    phone: str | None = Field(default=None, max_length=40, description="ONLY if printed in your snippet.")
    profile_url: str | None = Field(default=None, max_length=300)
    evidence: list[EvidenceRefModel] = Field(min_length=1, max_length=3)


class ContactFindings(BaseFindings):
    contacts: list[ContactClaim] = Field(default_factory=list, max_length=8)


def is_decision_maker(designation: str) -> bool:
    d = designation.lower()
    return any(w in d for w in ROLE_WORDS) and any(w in d for w in HEAD_WORDS)


def process_contact(state: dict, findings: dict, deps) -> ProcessResult:
    res = ProcessResult()
    ledger = state["ledger"]
    ident = state["subject"].get("identity") or {}
    seen = {(c["name"].lower(), c["designation"].lower()) for c in state["claims"].get("contact", [])}
    for c in findings.get("contacts", []):
        label = {"kind": "contact", "value": c["name"]}
        chk = validate_refs(ledger, c["evidence"], c["name"], ident.get("domain"), require_value_support=False)
        if not chk["ok"]:
            res.rejected.append({**label, "reasons": chk["reasons"]})
            continue
        snippets = [e["snippet"] for e in chk["evidence"]]
        name_n = ev.norm_text(c["name"])
        reasons = []
        if ROLE_AS_NAME.search(c["name"]) or len(name_n.split()) < 2:
            reasons.append("name_is_a_role_or_not_a_full_name")
        if not any(w in c["designation"].lower() for w in ROLE_WORDS):
            reasons.append("not_a_procurement_role")
        together = [s for s in snippets if name_n in ev.norm_text(s)
                    and ev.value_supported(c["designation"], s, threshold=0.8)]
        if not together:
            reasons.append("name_and_designation_not_in_same_snippet")
        joined = " ".join(snippets)
        if c.get("email"):
            if c["email"].lower() not in joined.lower():
                reasons.append("email_not_literally_in_snippet")
        if c.get("phone"):
            digits = re.sub(r"\D", "", c["phone"])
            if len(digits) < 6 or digits not in re.sub(r"\D", "", joined):
                reasons.append("phone_not_literally_in_snippet")
        if c.get("profile_url") and ev.find_doc(ledger, c["profile_url"]) is None \
                and c["profile_url"] not in joined:
            reasons.append("profile_url_not_retrieved_or_quoted")
        if reasons:
            res.rejected.append({**label, "reasons": reasons})
            continue
        key = (c["name"].lower(), c["designation"].lower())
        if key in seen:
            continue
        seen.add(key)
        res.accepted.append({"agent": "contact", "kind": "contact", "field": "contact", "value": c["name"],
                             "name": c["name"], "designation": c["designation"], "department": c.get("department"),
                             "email": c.get("email"), "phone": c.get("phone"), "profile_url": c.get("profile_url"),
                             "decision_maker": is_decision_maker(c["designation"]), "status": "verified",
                             "evidence": chk["evidence"], "confidence": chk["confidence"],
                             "confidence_basis": chk["basis"]})
    return res


def contact_task(state: dict) -> str:
    s = state["subject"]
    ident = s.get("identity") or {}
    ex = s.get("existing") or {}
    cs = ex.get("contact_summary") or {}
    lines = [f"Company: {s['name']} (official site: {ident.get('website')})",
             f"User objective: {state['objective']}",
             "Find publicly documented people who lead or work in procurement / purchasing / sourcing / supply "
             "chain, with their job title as printed on the page."]
    if cs.get("count"):
        lines.append(f"The CRM already holds {cs['count']} UNVERIFIED contact record(s) with titles "
                     f"{cs.get('designations')}; they have no sources. Do not assume they are correct and do not "
                     "try to reproduce them - report only what public pages say.")
    lines.append("Only people named on retrieved pages count. If a page lists a role but no person, that is not a "
                 "contact. 'No public procurement contact found' is a valid result.")
    return "\n".join(lines)


CONTACT_AGENT = AgentSpec(
    name="contact", title="Contact Intelligence Agent",
    goal="Find publicly documented procurement/sourcing decision-makers with verifiable sources.",
    role_prompt=("Look first on the company's own site (procurement, supplier, contact, about, leadership pages), "
                 "then public news/press pages found via search. Never guess or construct email addresses or "
                 "phone numbers; include them only when printed in the snippet you cite. Quote the line that contains "
                 "BOTH the person's name and their title."),
    findings_model=ContactFindings, finish_tool="submit_contact_findings",
    prefetch=agent_prefetch("contact"),
    tool_names=("web_search", "fetch_page", "get_crm_context", "submit_contact_findings"),
    process=process_contact, task=contact_task)

"""Does the quoted sentence state THIS field? (deterministic, no model, no network)

The evidence validator proves that a quote exists in a retrieved page and that it mentions the claimed value. That is
not enough: "sourcing over USD 1.6 Billion" mentions a number but is not the company's revenue, and "Since 2007,
Airbus India engineers ..." mentions a year but is not a founding statement. This module adds the missing check: the
sentence of a quote that contains the value must also carry a cue that belongs to the field (and none that
contradicts it). It only ever REJECTS - it never creates, edits or completes a value.

Strict on purpose: a rejected claim becomes an `unresolved` item a human can look at; a wrong accepted claim becomes a
proposed database change.
"""

from __future__ import annotations

import re

from agentic import evidence as ev

_NUM = re.compile(r"\d")

# field -> (cues that the sentence is about this field, cues that disqualify the sentence)
_RULES: dict[str, tuple[re.Pattern, re.Pattern | None]] = {
    "headquarters": (re.compile(r"\b(headquarter(?:s|ed)?|head[- ]?office|registered office|corporate office|"
                                r"corporate headquarters|principal office|main office)\b", re.I), None),
    "annual_revenue": (re.compile(r"\b(revenues?|turnover|net sales|total sales|annual sales|sales of|total income|"
                                  r"gross income|income of)\b", re.I),
                       re.compile(r"\b(sourc\w*|procure\w*|purchas\w*|spend\w*|invest\w*|orders?|contracts?|"
                                  r"export\w*|budget|funding|valuation|loans?)\b", re.I)),
    "employee_count": (re.compile(r"\b(employees?|employs|employed|workforce|headcount|staff|team members|"
                                  r"associates|people work|personnel)\b", re.I), None),
    "founded_year": (re.compile(r"\b(founded|established|incorporated|inception|commenced operations|"
                                r"began operations|started operations|set up in)\b", re.I), None),
    "parent_company": (re.compile(r"\b(subsidiary|parent|owned by|part of|wholly[- ]owned|belongs to|"
                                  r"a company of|member of the)\b", re.I), None),
    "ownership_type": (re.compile(r"\b(owned|ownership|private|public|listed|publicly|subsidiary|joint venture|"
                                  r"state[- ]owned|government|shareholders?|held by)\b", re.I), None),
    "company_type": (re.compile(r"\b(private|public|limited|ltd|llp|inc|corporation|subsidiary|joint venture|"
                                r"company|partnership|proprietorship)\b", re.I), None),
    "manufacturing_locations": (re.compile(r"\b(plants?|factory|factories|facility|facilities|manufactur\w*|"
                                           r"production|assembly|works|unit|units)\b", re.I), None),
}
# Procurement findings (kind -> cues). The URL kinds are judged on the whole quote; the text kinds on the sentence that
# carries the value. A company that SELLS or exports something is not a company that BUYS it.
_BUY = re.compile(r"\b(procure\w*|purchas\w*|sourc\w*|buy\w*|bought|acquir\w*)\b", re.I)
_SELL = re.compile(r"\b(sells?|sold|exports?|our customers|we (?:supply|sell|deliver)|customers? (?:buy|purchase))\b", re.I)
# industry: a definitional statement about what the company is or does - not a heading that happens to contain the word.
_RULES["industry"] = (re.compile(r"\b(industry|sector|manufacturers?|manufactures?|produces?|engaged in|specialis\w+ in|"
                                 r"operates in|active in)\b|\b(is|are)\b[^.]{0,80}\b(company|manufacturer|group|leader|"
                                 r"maker|producer|provider)\b", re.I), None)
_NEEDS_SUBJECT = {"purchasing_category", "purchased_component"}
_RULES.update({
    "supplier_portal": (re.compile(r"\b(suppliers?|vendors?)\b.{0,160}\b(regist\w*|portal|onboard\w*|sign[- ]?up|enrol\w*|log ?in)\b|"
                                   r"\b(regist\w*|portal|onboard\w*)\b.{0,160}\b(suppliers?|vendors?)\b", re.I | re.S), None),
    "procurement_page": (re.compile(r"\b(procure\w*|sourcing|purchasing|suppliers?|vendors?|tenders?)\b", re.I), None),
    "purchasing_category": (_BUY, _SELL),
    "purchased_component": (_BUY, _SELL),
    "process_note": (re.compile(r"\b(procure\w*|sourc\w*|suppliers?|vendors?|purchas\w*|tenders?|qualif\w*|approv\w*|"
                                r"onboard\w*|audit\w*|regist\w*)\b", re.I), None),
})
_URL_KINDS = {"supplier_portal", "procurement_page"}
# Fields whose value is a number/year: the quoted sentence must contain digits (the value check already ties them).
_NEEDS_DIGIT = {"annual_revenue", "employee_count", "founded_year"}


def mentions_subject(company: str, text: str) -> bool:
    """Does the text talk about THIS company ('we/our' also counts on the company's own pages)?"""
    from agentic import names
    low = text.lower()
    return any(t in low for t in names.distinctive_tokens(company)) or bool(re.search(r"\b(we|our|the company)\b", low))


def check_evidence(field: str, value, evidence: list[dict], *, company: str | None = None,
                   official_domain: str | None = None) -> tuple[bool, str]:
    """`check` over evidence dicts ({url, snippet}). For what a company BUYS, a quote counts only if it is on the
    company's own (verified) domain or itself names the company - 'Airbus sells X' or 'Other Corp buys Y' never do."""
    snippets = []
    for e in evidence:
        if field in _NEEDS_SUBJECT and company:
            first_party = bool(official_domain) and ev.registrable_domain(e["url"]) == official_domain
            if not first_party and not mentions_subject(company, e["snippet"]):
                continue
        snippets.append(e["snippet"])
    if not snippets:
        return False, "quote_is_not_about_the_company"
    return check(field, value, snippets)


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", text) if s.strip()]


def _values(value) -> list[str]:
    return [str(v) for v in value] if isinstance(value, (list, tuple)) else [str(value)]


def check(field: str, value, snippets: list[str]) -> tuple[bool, str]:
    """(ok, reason). Fields without a rule (identity, descriptions, product lists) pass - their value support is
    checked by the evidence validator. For a ruled field, at least one quoted sentence that contains the value must
    carry a cue of the field and no disqualifying cue."""
    rule = _RULES.get(field)
    if rule is None:
        return True, ""
    cue, bad = rule
    if field in _URL_KINDS:                                 # the value is a URL: the quote as a whole must say what it is
        whole = " ".join(snippets)
        return (True, "") if cue.search(whole) and not (bad and bad.search(whole)) else (False, "quote_does_not_state_field")
    sentences = [s for snip in snippets for s in _sentences(snip)]

    def item_ok(item: str) -> bool:
        for sent in sentences:
            if not ev.value_supported(item, sent):
                continue                                   # the sentence must itself carry the value
            if field in _NEEDS_DIGIT and not _NUM.search(sent):
                continue
            if bad is not None and bad.search(sent):
                continue
            if cue.search(sent):
                return True
        return False
    items = _values(value)
    return (True, "") if items and all(item_ok(i) for i in items) else (False, "quote_does_not_state_field")

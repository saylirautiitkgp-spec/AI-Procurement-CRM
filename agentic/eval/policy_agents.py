"""PolicyLLM handlers for the contact / procurement / supplier / lead agents (TEST DOUBLES - see
policy.py). Each reads real tool results from the conversation and decides the next call."""

from __future__ import annotations

import re

from agentic.eval.policy import _call, history
from agentic.names import normalize_company_name


def _ctx(messages):
    first = messages[0]["content"]
    name = re.search(r"Company: (.+?) \(official site: (.*?)\)", first)
    if name:
        return name.group(1), (name.group(2) if name.group(2) != "None" else None)
    return re.search(r"Company: (.+)", first).group(1), None


def _results(hist, tool):
    return [h for h in hist if h["name"] == tool and h["result"] and not h["result"].get("error")]


def _fetched_texts(hist):
    return {h["result"]["url"]: h["result"]["text"] for h in _results(hist, "fetch_page")}


def _errors(hist):
    return [h["result"]["error"]["code"] for h in hist if h["result"] and h["result"].get("error")]


def _line_with(text: str, pattern: str):
    m = re.search(pattern, text)
    if not m:
        return None
    start = text.rfind("\n", 0, m.start()) + 1
    end = text.find("\n", m.end())
    return m, text[start:(len(text) if end == -1 else end)].strip()


def _hard_stop(hist):
    return any(c in ("search_disabled", "search_quota_exhausted", "credit_budget_exhausted",
                     "monthly_credit_budget_exhausted", "search_rate_limited_persistent") for c in _errors(hist))


def _domain_urls(results, site, words):
    base = re.match(r"(https?://[^/]+)", site).group(1) if site else None
    out = []
    for r in results:
        for item in r["result"].get("results", []):
            if base and item["url"].startswith(base) and any(w in item["url"].lower() for w in words):
                out.append(item["url"])
    return list(dict.fromkeys(out))


def contact(self, messages, tools):
    name, site = _ctx(messages)
    hist = history(messages)
    if any(h["name"] == "submit_contact_findings" for h in hist):
        return _call("submit_contact_findings", status="exhausted", contacts=[],
                     unresolved=["earlier submission was rejected by validation"])
    searches, fetches = _results(hist, "web_search"), _results(hist, "fetch_page")
    hinted = [u for u in re.findall(r"^  - (https?://\S+) \(", messages[0]["content"], re.M) if "procure" in u or "supplier" in u]
    fetched_urls = {h["args"]["url"] for h in hist if h["name"] == "fetch_page"}
    if hinted and not searches and not _hard_stop(hist):
        todo = [u for u in hinted if u not in fetched_urls]
        if todo:
            return _call("fetch_page", url=todo[0])
        return _submit_contact(hist)
    if not searches and not _hard_stop(hist):
        return _call("web_search", query=f"{name} head of procurement")
    if _hard_stop(hist):
        return _submit_contact(hist, status="partial")
    targets = _domain_urls(searches, site, ("procure", "supplier"))
    for url in targets[:2]:
        if url not in {h["args"]["url"] for h in hist if h["name"] == "fetch_page"}:
            return _call("fetch_page", url=url)
    return _submit_contact(hist)


def _submit_contact(hist, status=None, drop=()):
    contacts = []
    for url, text in _fetched_texts(hist).items():
        hit = _line_with(text, r"Head of Procurement: ([A-Z][a-z]+ [A-Z][a-z]+)")
        if not hit:
            continue
        m, line = hit
        person = m.group(1)
        snippet, email = line, None
        em = re.search(r"Procurement enquiries: (\S+@\S+)", text)
        if em and person.split()[0].lower() in em.group(1):
            snippet, email = line + "\n" + em.group(0), em.group(1)
        contacts.append({"name": person, "designation": "Head of Procurement", "email": email,
                         "evidence": [{"url": url, "snippet": snippet}]})
    st = status or ("satisfied" if contacts else "exhausted")
    return _call("submit_contact_findings", status=st, contacts=contacts,
                 unresolved=[] if contacts else ["no publicly named procurement contact found"])


def procurement(self, messages, tools):
    name, site = _ctx(messages)
    hist = history(messages)
    if any(h["name"] == "submit_procurement_findings" for h in hist):
        return _call("submit_procurement_findings", status="exhausted", items=[])
    if "get_crm_context" in [t.name for t in tools] and not any(h["name"] == "get_crm_context" for h in hist):
        return _call("get_crm_context", section="components")
    searches = _results(hist, "web_search")
    if not searches and not _hard_stop(hist):
        return _call("web_search", query=f"{name} supplier portal procurement")
    if _hard_stop(hist):
        return _submit_procurement(hist, status="partial")
    fetched = {h["args"]["url"] for h in hist if h["name"] == "fetch_page"}
    for url in _domain_urls(searches, site, ("procure", "supplier"))[:2]:
        if url not in fetched:
            return _call("fetch_page", url=url)
    return _submit_procurement(hist)


def _submit_procurement(hist, status=None):
    items = []
    for url, text in _fetched_texts(hist).items():
        hit = _line_with(text, r"supplier portal at (https?://[^\s]+?)\.(?:\s|$)")
        if hit:
            m, line = hit
            items.append({"kind": "supplier_portal", "value": m.group(1), "evidence": [{"url": url, "snippet": line}]})
            items.append({"kind": "procurement_page", "value": url, "evidence": [{"url": url, "snippet": line}]})
        hit = _line_with(text, r"Procurement categories: ([^\n]+?)\.(?:\n|$)")
        if hit:
            m, line = hit
            for cat in re.split(r", ", m.group(1)):
                items.append({"kind": "purchasing_category", "value": cat, "evidence": [{"url": url, "snippet": line}]})
        hit = _line_with(text, r"considered for sourcing of ([^\n]+?)\.")
        if hit:
            m, line = hit
            for comp in re.split(r", ", m.group(1)):
                items.append({"kind": "purchased_component", "value": comp, "evidence": [{"url": url, "snippet": line}]})
    st = status or ("satisfied" if items else "exhausted")
    return _call("submit_procurement_findings", status=st, items=items,
                 unresolved=[] if items else ["no procurement information found"])


def supplier(self, messages, tools):
    name, _ = _ctx(messages)
    hist = history(messages)
    if any(h["name"] == "submit_supplier_findings" for h in hist):
        return _call("submit_supplier_findings", status="exhausted", relationships=[])
    offered = [t.name for t in tools]
    if "get_crm_context" in offered and not any(h["name"] == "get_crm_context" for h in hist):
        return _call("get_crm_context", section="components")
    if not _results(hist, "web_search") and not _hard_stop(hist):
        crm = next((h["result"] for h in hist if h["name"] == "get_crm_context"), {}) or {}
        comp = (crm.get("data") or ["components"])[0] if crm.get("in_crm") else "components"
        m = re.search(r"CRM component list \(UNVERIFIED\): \['([^']+)'", messages[0]["content"])
        comp = m.group(1) if m else comp
        return _call("web_search", query=f"{name} supplier supplies {comp}")
    return _submit_supplier(name, hist, status="partial" if _hard_stop(hist) else None)


def _submit_supplier(name, hist, status=None):
    rels = []
    core = normalize_company_name(name)
    for h in _results(hist, "web_search"):
        for item in h["result"].get("results", []):
            for sentence in re.split(r"(?<=[.])\s+|\n", item["snippet"]):
                m = re.match(r"(.+?) supplies (.+?) to (.+?)(?:,|\.|$)", sentence)
                if m and normalize_company_name(m.group(3)) == core:
                    rels.append({"supplier_name": m.group(1).strip(), "customer_name": m.group(3).strip(),
                                 "component": m.group(2).strip(),
                                 "evidence": [{"url": item["url"], "snippet": sentence.strip()}]})
    st = status or ("satisfied" if rels else "exhausted")
    return _call("submit_supplier_findings", status=st, relationships=rels,
                 unresolved=[] if rels else ["no explicit supplier relationship found"])


def lead(self, messages, tools):
    hist = history(messages)
    score = next((h["result"] for h in hist if h["name"] == "compute_lead_score"), None)
    if score is None and "compute_lead_score" not in [t.name for t in tools]:
        m = re.search(r"Missing information: (.+)", messages[0]["content"])
        missing = [x for x in (m.group(1).split("; ") if m else []) if x and x != "none"]
        return _call("submit_lead_findings", status="satisfied", next_actions=[f"Obtain: {x}" for x in missing[:4]])
    if score is None:
        return _call("compute_lead_score")
    if any(h["name"] == "submit_lead_findings" for h in hist):
        return _call("submit_lead_findings", status="satisfied")
    actions = [f"Obtain: {m}" for m in score.get("missing_information", [])[:4]]
    return _call("submit_lead_findings", status="satisfied", next_actions=actions)

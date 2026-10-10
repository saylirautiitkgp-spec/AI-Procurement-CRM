"""PolicyLLM - a deterministic TEST DOUBLE that behaves like a reasonable model.

It reads the real conversation (tool results included) and decides its next tool
call from what it observes, so it exercises the harness's observe -> decide ->
act loop offline. It is NOT evidence about any real LLM's quality; real-model
behaviour must be measured with the provider evaluation (needs live approval).
"""

from __future__ import annotations

import itertools
import json
import re
from typing import Callable

from agentic.eval.synthetic import slug
from agentic.names import normalize_company_name
from agentic.evidence import registrable_domain
from agentic.llm.base import LLMResponse, ToolCallRequest, ToolSpec, Usage

_ids = itertools.count(1)
_AGENT_NAMES = ("company", "contact", "procurement", "supplier", "lead")


def _call(name: str, **args) -> LLMResponse:
    return LLMResponse(tool_calls=[ToolCallRequest(id=f"p{next(_ids)}", name=name, args=args)],
                       usage=Usage(requests=1, input_tokens=400, output_tokens=60))


def _calls(pairs) -> LLMResponse:
    return LLMResponse(tool_calls=[ToolCallRequest(id=f"p{next(_ids)}", name=n, args=a) for n, a in pairs],
                       usage=Usage(requests=1, input_tokens=400, output_tokens=60))


def unwrap(content: str):
    text = content.strip()
    if text.startswith("<untrusted_web_content"):
        text = text.split("\n", 1)[1].rsplit("\n</untrusted_web_content>", 1)[0]
    try:
        return json.loads(text)
    except Exception:
        return None


def history(messages: list[dict]) -> list[dict]:
    """[{name,args,result}] pairing assistant tool calls with their tool messages."""
    by_id = {m.get("tool_call_id"): m for m in messages if m["role"] == "tool"}
    out = []
    for m in messages:
        if m["role"] == "assistant":
            for c in m.get("tool_calls", []):
                res = by_id.get(c["id"])
                out.append({"name": c["name"], "args": c["args"],
                            "result": unwrap(res["content"]) if res else None})
    return out


def want_agents(objective: str) -> list[str]:
    o = objective.lower()
    if "comprehensive" in o:
        return ["company", "procurement", "contact", "supplier", "lead"]
    wanted = ["company"]
    if any(k in o for k in ("contact", "decision", "procurement head", "buyer")):
        wanted.append("contact")
    if any(k in o for k in ("portal", "procurement structure", "purchasing categor")):
        wanted.append("procurement")
    if any(k in o for k in ("supplier", "component")):
        wanted.append("supplier")
    if any(k in o for k in ("lead", "priorit", "score")):
        wanted.append("lead")
    if o.strip() in ("profile", "company profile") or len(wanted) == 1:
        return ["company"]
    return wanted


class PolicyLLM:
    name = "policy-double"
    model = "policy-double-1"

    def __init__(self):
        self.calls = 0
        from agentic.eval import policy_agents as pa
        self.handlers: dict[str, Callable] = {
            "company": self._company,
            "contact": lambda m, t: pa.contact(self, m, t),
            "procurement": lambda m, t: pa.procurement(self, m, t),
            "supplier": lambda m, t: pa.supplier(self, m, t),
            "lead": lambda m, t: pa.lead(self, m, t),
        }

    def register(self, agent: str, handler: Callable) -> None:
        self.handlers[agent] = handler

    # ------------------------------------------------------------------ entry
    def generate(self, *, system, messages, tools: list[ToolSpec], max_output_tokens, force_tool=False):
        self.calls += 1
        self._system = system
        if [t.name for t in tools] == ["route"]:
            return self._supervisor(messages)
        m = re.search(r"ROLE: (\w+)", system)
        role = (m.group(1) if m else "").lower()
        for agent in _AGENT_NAMES:
            if role == agent:
                return self.handlers[agent](messages, tools)
        raise AssertionError(f"PolicyLLM has no handler for role {role!r}")

    # ------------------------------------------------------------- supervisor
    def _supervisor(self, messages):
        text = messages[0]["content"]
        objective = re.search(r"OBJECTIVE: (.*)", text).group(1)
        eligible = set(re.findall(r"- (\w+): ELIGIBLE", text))
        ran = set(re.findall(r"^  - (\w+): status=", text, re.M))
        for agent in want_agents(objective):
            if agent in ran:
                continue
            if agent in eligible:
                return _call("route", next=agent, reason=f"{agent} is needed for the objective and eligible")
            if agent != "company" and "company" in eligible and "company" not in ran:
                return _call("route", next="company", reason="identity must be confirmed first")
        return _call("route", next="finish", reason="objective covered or nothing eligible can add value")

    # ---------------------------------------------------------------- company
    def _company(self, messages, tools):
        first = messages[0]["content"]
        name = re.search(r"Company submitted by the user: ['\"](.+?)['\"]\n", first).group(1)
        hist = history(messages)
        searches = [h for h in hist if h["name"] == "web_search"]
        fetches = [h for h in hist if h["name"] == "fetch_page"]
        submits = [h for h in hist if h["name"] == "submit_company_findings"]
        if submits:
            fb = (submits[-1]["result"] or {}).get("result", "")
            bad = set(re.findall(r"^- (\w+):", fb, re.M))
            return self._submit_company(name, hist, drop=bad)
        queries = [f"{name} official website", f"{name} wikipedia", f"{name} headquarters about"]
        if not searches:
            return _call("web_search", query=queries[0])
        results = _all_results(searches)
        site = _official_site(normalize_company_name(name), results)
        errs = [h for h in searches if h["result"] and h["result"].get("error")]
        hard_err = any(e["result"]["error"]["code"] in ("search_disabled", "search_quota_exhausted",
                                                        "credit_budget_exhausted",
                                                        "monthly_credit_budget_exhausted",
                                                        "search_rate_limited_persistent") for e in errs)
        if hard_err:
            return self._submit_company(name, hist, status="partial", note="search unavailable")
        if site is None and len(searches) < 3:
            return _call("web_search", query=queries[len(searches)])
        if site is not None:
            home = site
            fetched = {h["args"]["url"] for h in fetches}
            links = _links(fetches)
            wanted = [home] + [l for l in links if any(k in l for k in ("about", "products"))]
            pending = [u for u in wanted if u not in fetched]
            room = 3 - len(fetches)
            if pending and room > 0:
                if "ONE turn" in self._system and len(pending) > 1 and fetches:
                    return _calls([("fetch_page", {"url": u}) for u in pending[:room]])
                return _call("fetch_page", url=pending[0])
            if len(searches) < 2:       # corroborate with a third-party source
                return _call("web_search", query=queries[1])
        return self._submit_company(name, hist)

    def _submit_company(self, name, hist, drop=(), status=None, note=""):
        texts = _collect_texts(hist)
        claims, site = _extract_company_claims(name, texts)
        claims = [c for c in claims if c["field"] not in drop]
        st = status or ("satisfied" if len(claims) >= 4 else "partial" if claims else "exhausted")
        return _call("submit_company_findings", status=st, identity_website=site,
                     identity_reasoning="Domain matches the company name and the page names the company."
                     if site else "", notes=note, claims=claims,
                     unresolved=[] if claims else ["official website", "headquarters"],
                     suggest_next=["procurement", "contact"])


# ----------------------------------------------------------------------- helpers

def _all_results(searches):
    out = []
    for h in searches:
        out += (h["result"] or {}).get("results", []) if h["result"] else []
    return out


def _official_site(name: str, results: list[dict]) -> str | None:
    want = slug(name)
    for r in results:
        dom = registrable_domain(r["url"])
        if dom.split(".")[0] == want:
            m = re.match(r"(https?://[^/]+)", r["url"])
            return m.group(1) + "/"
    return None


def _links(fetches):
    out = []
    for h in fetches:
        for l in ((h["result"] or {}).get("links") or []):
            out.append(l["url"])
    return out


def _collect_texts(hist) -> list[tuple[str, str]]:
    from agentic.evidence import canonical_url
    seen, texts = set(), []
    for h in hist:                                       # fetched page text first: it is the fullest source
        r = h["result"]
        if r and not r.get("error") and h["name"] == "fetch_page" and canonical_url(r["url"]) not in seen:
            seen.add(canonical_url(r["url"]))
            texts.append((r["url"], r["text"]))
    for h in hist:                                       # then search snippets for pages that were never fetched
        r = h["result"]
        if r and not r.get("error") and h["name"] == "web_search":
            for item in r.get("results", []):
                if canonical_url(item["url"]) not in seen and not item["snippet"].startswith("[snippet omitted"):
                    seen.add(canonical_url(item["url"]))
                    texts.append((item["url"], item["snippet"]))
    return texts


_PATTERNS = [
    ("headquarters", r"headquartered in ([A-Z][\w ]+?), [A-Z]", lambda m: m.group(1)),
    ("headquarters", r"headquarters are in ([A-Z][\w ]+?), [A-Z]", lambda m: m.group(1)),
    ("headquarters", r"based in ([A-Z][\w ]+?), [A-Z]", lambda m: m.group(1)),
    ("founded_year", r"[Ff]ounded in (\d{4})", lambda m: m.group(1)),
    ("employee_count", r"approximately ([\d,]+) (?:people|employees)", lambda m: m.group(1)),
    ("industry", r" is an? ([a-z][a-z &\-]+?) company", lambda m: m.group(1)),
    ("products_services", r"[Oo]ur products include ([^.\n]+)", lambda m: re.split(r", | and ", m.group(1))),
    ("manufacturing_locations", r"plants are located in ([^.\n]+)", lambda m: re.split(r", | and ", m.group(1))),
]


def _snippet(text: str, m: re.Match) -> str:
    start = max(text.rfind("\n", 0, m.start()), text.rfind(". ", 0, m.start()) + 1, 0)
    end = text.find("\n", m.end())
    end = len(text) if end == -1 else end
    s = text[start:end].strip(" \n")
    return s[:400]


def _extract_company_claims(name: str, texts):
    claims, site, seen = [], None, set()
    core = normalize_company_name(name)
    name_re = r"[\W_]+".join(re.escape(t) for t in core.split())
    for url, text in texts:
        first_party = registrable_domain(url).split(".")[0] == slug(core)
        if first_party and site is None:
            m = re.search(name_re + r"[^\n]*", text, re.I)
            if m and len(m.group(0)) >= 12:
                site = re.match(r"(https?://[^/]+)", url).group(1) + "/"
                claims.append({"field": "official_website", "value": site,
                               "evidence": [{"url": url, "snippet": m.group(0)[:300]}]})
        for fld, pat, val in _PATTERNS:
            m = re.search(pat, text)
            if not m:
                continue
            value = val(m)
            key = (fld, json.dumps(value))
            if key in seen:
                continue
            seen.add(key)
            key_name = "values" if isinstance(value, list) else "value"
            claims.append({"field": fld, key_name: value, "evidence": [{"url": url, "snippet": _snippet(text, m)}]})
    return claims, site

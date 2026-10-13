"""Regression tests for the failed live Groq run 06822d925f55 (offline only).

1. Lead agent: offered only `submit_lead_findings`, yet its prompt said "use compute_lead_score" and the tool choice was
   free -> the model called a tool that was not offered (provider 400).
2. Company agent: a request estimated at 8,120 tokens was refused by the 8,000 tokens/minute guard after four tool calls.
"""

import copy
import json

import pytest

from agentic import graph as graph_mod
from agentic.config import Efficiency, Limits, Settings
from agentic.context import OMITTED, fit_to_budget
from agentic.eval.policy import PolicyLLM
from agentic.llm.base import LLMResponse
from agentic.llm.fake import tool_call
from agentic.llm.pacing import PacedLLM, ProviderQuota, estimate_request_tokens
from agentic.persistence.store import Store
from agentic.security import wrap_untrusted
from tests.agentic.support import make_runner


def tool_msg(name, payload, call_id="c"):
    return {"role": "tool", "tool_call_id": call_id, "name": name, "content": wrap_untrusted(json.dumps(payload), name)}


def page(url, text, n_links=25):
    return tool_msg("fetch_page", {"url": url, "title": "t", "status": 200, "text": text, "offset": 0, "next_offset": None,
                                   "links": [{"url": f"{url}/l{i}", "text": f"link {i}"} for i in range(n_links)],
                                   "page_truncated": False})


def search(snippets):
    return tool_msg("web_search", {"query": "q", "cached": False,
                                   "results": [{"url": f"https://a.example/{i}", "title": "t", "snippet": s}
                                               for i, s in enumerate(snippets)]})


def est(msgs):
    return estimate_request_tokens("sys", msgs, [], 0)


def payload(m):
    return json.loads(m["content"].split("\n", 1)[1].rsplit("\n", 1)[0])


# ======================================================================= fit_to_budget (unit)
def history():
    return [{"role": "user", "content": "TASK"},
            {"role": "assistant", "content": "", "tool_calls": [{"id": "1", "name": "web_search", "args": {}}]},
            search(["S" * 600] * 3),
            {"role": "assistant", "content": "", "tool_calls": [{"id": "2", "name": "fetch_page", "args": {}}]},
            page("https://a.example/old", "OLD-PAGE " * 300),
            {"role": "assistant", "content": "", "tool_calls": [{"id": "3", "name": "fetch_page", "args": {}}]},
            page("https://a.example/new", "NEW-PAGE " * 300)]


def test_nothing_changes_when_the_request_already_fits():
    h = history()
    out, steps = fit_to_budget(h, est, 10 ** 6)
    assert out is h and steps == []


def test_navigation_links_go_first_and_page_text_is_untouched():
    h = history()
    full = est(h)
    budget = full - 150                                   # a little over: dropping links alone must be enough
    out, steps = fit_to_budget(h, est, budget)
    assert est(out) <= budget and {s["step"] for s in steps} == {"links_dropped"}
    assert payload(out[4])["text"] == "OLD-PAGE " * 300 and payload(out[6])["text"] == "NEW-PAGE " * 300
    assert payload(out[6])["links"] == []


def test_oldest_results_are_replaced_by_an_explicit_notice_and_the_newest_is_never_touched():
    h = history()
    original = copy.deepcopy(h)
    out, steps = fit_to_budget(h, est, 1200)
    assert h == original                                                  # input not mutated (stored history stays full)
    assert est(out) <= 1200
    kinds = [s["step"] for s in steps]
    assert kinds.index("links_dropped") < kinds.index("search_snippets_omitted") < kinds.index("page_text_omitted")
    assert all(r["snippet"] == OMITTED for r in payload(out[2])["results"])
    assert payload(out[2])["results"][0]["url"] == "https://a.example/0"      # sources stay discoverable
    assert payload(out[4])["text"] == OMITTED and payload(out[4])["url"] == "https://a.example/old"
    assert payload(out[6])["text"] == "NEW-PAGE " * 300                      # newest result intact
    for i in (0, 1, 3, 5):                                                # task and assistant turns untouched
        assert out[i] == original[i]


def test_it_stops_as_soon_as_the_request_fits_and_never_loops_when_it_cannot():
    h = history()
    links_only = est(fit_to_budget(h, est, est(h) - 1)[0])             # estimate once only the links are gone
    _, steps = fit_to_budget(h, est, links_only - 300)                 # the search snippets (~450 tokens) are enough
    assert "page_text_omitted" not in [s["step"] for s in steps]           # snippets alone were enough
    out, steps = fit_to_budget(h, est, 1)                                  # impossible budget: returns, no exception
    assert payload(out[-1])["text"] == "NEW-PAGE " * 300 and steps


# ======================================================================= Company agent near the TPM limit
def run_requests(llm, tmp=None):
    r = make_runner(llm=llm, settings=Settings(limits=Limits(), efficiency=Efficiency()))
    return r, r.start("Apollo Micro Systems", "company profile")


class Recorder:
    """Pass-through LLM that records what the model was sent."""
    name, model = "rec", "rec-model"

    def __init__(self, inner):
        self.inner, self.requests = inner, []

    def generate(self, **kw):
        self.requests.append({"system": kw["system"], "messages": copy.deepcopy(kw["messages"]),
                              "tools": [t.name for t in kw["tools"]], "tool_specs": kw["tools"],
                              "force_tool": kw["force_tool"], "max_out": kw["max_output_tokens"],
                              "est": estimate_request_tokens(kw["system"], kw["messages"], kw["tools"],
                                                             kw["max_output_tokens"])})
        return self.inner.generate(**kw)


def squeezable_limit(requests):
    """A limit strictly between the biggest request and the smallest size that request can be shrunk to."""
    big = max(requests, key=lambda q: q["est"])
    floor = fit_to_budget(big["messages"], lambda m: estimate_request_tokens(big["system"], m, big["tool_specs"],
                                                                             big["max_out"]), 0)[0]
    floor_est = estimate_request_tokens(big["system"], floor, big["tool_specs"], big["max_out"])
    assert floor_est < big["est"] - 2                       # the scenario really is shrinkable
    return (big["est"] + floor_est) // 2


def paced(inner, tpm):
    ticks = iter(range(0, 10 ** 7, 100))                       # every call is "100 s later": only the size check matters
    return PacedLLM(inner, ProviderQuota(rpm=1000, tpm=tpm, rpd=10 ** 6, tpd=10 ** 9), Store(":memory:"),
                    clock=lambda: float(next(ticks)), sleep=lambda s: None, max_wait_s=0)


def test_company_agent_finishes_when_the_unfitted_request_would_exceed_the_tpm_limit(monkeypatch):
    rec = Recorder(PolicyLLM())
    _, free = run_requests(rec)
    limit = squeezable_limit(rec.requests)                  # the biggest request no longer fits as it is
    assert limit < max(q["est"] for q in rec.requests)
    accepted_free = sum(len(v) for v in free["claims"].values())
    assert free["status"] == "completed" and accepted_free > 0

    # without the fix (budget unknown to the graph) the pacer refuses the oversize request: the original failure
    monkeypatch.setattr(graph_mod, "_tpm_limit", lambda llm: None)
    _, broken = run_requests(paced(PolicyLLM(), limit))
    monkeypatch.undo()
    assert any(e["type"] == "llm_error" and "exceeds the" in e["data"]["message"] for e in broken["trace"])

    # with the fix the same request is shrunk, sent, and the agent completes
    sent = Recorder(PolicyLLM())
    _, fixed = run_requests(paced(sent, limit))
    assert not [e for e in fixed["trace"] if e["type"] == "llm_error"]
    trimmed = [e for e in fixed["trace"] if e["type"] == "context_trimmed"]
    assert trimmed and all(e["data"]["estimate_after"] <= limit for e in trimmed)
    assert max(q["est"] for q in sent.requests) <= limit
    assert fixed["status"] == "completed"
    # evidence validation is unchanged: same accepted claims, all validated against the full ledger
    assert sum(len(v) for v in fixed["claims"].values()) == accepted_free
    assert all(d["text"] for d in fixed["ledger"].values())


def test_stored_conversation_and_ledger_are_not_shortened_by_context_fitting():
    rec = Recorder(PolicyLLM())
    _, free = run_requests(rec)
    limit = squeezable_limit(rec.requests)
    _, fixed = run_requests(paced(PolicyLLM(), limit))
    assert {k: v["text"] for k, v in fixed["ledger"].items()} == {k: v["text"] for k, v in free["ledger"].items()}


def test_unpaced_clients_are_never_trimmed():
    assert graph_mod._tpm_limit(PolicyLLM()) is None
    rec = Recorder(PolicyLLM())
    _, out = run_requests(rec)
    assert not [e for e in out["trace"] if e["type"] == "context_trimmed"]


# ======================================================================= Lead agent tool offer
def test_lead_agent_is_offered_only_its_finish_tool_with_a_consistent_prompt_and_a_forced_choice():
    script = [tool_call("route", next="company", reason="x"), tool_call("submit_company_findings", status="exhausted"),
              tool_call("route", next="lead", reason="rank"), tool_call("submit_lead_findings", status="satisfied"),
              tool_call("route", next="finish", reason="d")]
    seen = []

    class Spy:
        name, model = "spy", "spy-model"

        def generate(self, **kw):
            seen.append({"system": kw["system"], "tools": [t.name for t in kw["tools"]], "force": kw["force_tool"]})
            return script.pop(0)

    out = make_runner(llm=Spy(), settings=Settings(limits=Limits(), efficiency=Efficiency())).start(
        "Apollo Micro Systems", "rank")
    lead = [s for s in seen if "ROLE: Lead" in s["system"]]
    assert len(lead) == 1 and lead[0]["tools"] == ["submit_lead_findings"]
    assert lead[0]["force"] is True                                        # no free choice -> cannot name another tool
    assert "Use compute_lead_score" not in lead[0]["system"] and "not offered" in lead[0]["system"]
    assert out["claims"]["lead"]["score"] is not None                      # deterministic score still produced
    # agents with a real choice are not forced
    assert all(s["force"] is False for s in seen if "ROLE: Company" in s["system"])


def test_lead_prompt_still_mentions_the_scoring_tool_only_as_conditional():
    from agentic.agents.lead import LEAD_AGENT
    assert "compute_lead_score when that tool is offered" in LEAD_AGENT.role_prompt

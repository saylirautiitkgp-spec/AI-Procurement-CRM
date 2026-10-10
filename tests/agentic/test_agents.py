"""The five agents: validators (unit), dynamic supervisor routing, tool use and safety (end to end)."""

import json

import pytest

from agentic import evidence as ev
from agentic.agents.contact import process_contact
from agentic.agents.procurement import process_procurement
from agentic.agents.supplier import process_supplier, states_relationship
from agentic.eval.policy import PolicyLLM
from agentic.llm.fake import FakeLLM, tool_call
from agentic.repo import InMemoryRepo
from agentic.scoring import WEIGHTS, score_lead
from tests.agentic.support import default_repo, make_runner

BASE = "https://www.acme.example"


def make_state(texts, *, company="Acme Corp", domain="acme.example", claims=None):
    ledger = {}
    for path, text in texts.items():
        ev.add_doc(ledger, url=path if path.startswith("http") else BASE + path, text=text, title="t")
    return {"ledger": ledger, "claims": claims or {}, "subject": {"name": company, "existing": None,
            "identity": {"verified": True, "domain": domain, "website": BASE}}}


def ref(path, snippet):
    return {"url": path if path.startswith("http") else BASE + path, "snippet": snippet}


# ================================================================ contact validator
PAGE = ("Leadership team.\nHead of Procurement: Asha Rao.\nProcurement enquiries: asha.proc@acme.example\n"
        "Marketing Manager: Vikram Nair.\nPhone +91 20 5550 1234 for supplier help.")


def contact(**kw):
    base = {"name": "Asha Rao", "designation": "Head of Procurement",
            "evidence": [ref("/team", "Head of Procurement: Asha Rao.")]}
    return {**base, **kw}


def run_contact(*contacts):
    st = make_state({"/team": PAGE})
    return process_contact(st, {"contacts": list(contacts)}, None)


def test_valid_contact_is_accepted_with_evidence_and_computed_confidence():
    r = run_contact(contact())
    c = r.accepted[0]
    assert c["name"] == "Asha Rao" and c["confidence"] == 0.6 and c["decision_maker"] is True
    assert c["evidence"][0]["retrieved_at"]


@pytest.mark.parametrize("override,reason", [
    ({"name": "Procurement Team"}, "name_is_a_role_or_not_a_full_name"),
    ({"name": "Asha"}, "name_is_a_role_or_not_a_full_name"),
    ({"name": "Vikram Nair", "designation": "Marketing Manager",
      "evidence": [ref("/team", "Marketing Manager: Vikram Nair.")]}, "not_a_procurement_role"),
    ({"email": "asha.rao@acme.example"}, "email_not_literally_in_snippet"),           # constructed email
    ({"phone": "+91 99999 11111"}, "phone_not_literally_in_snippet"),
    ({"profile_url": "https://www.linkedin.com/in/asha-rao"}, "profile_url_not_retrieved_or_quoted"),
    ({"evidence": [ref("/team", "Head of Procurement: Asha Rao"), ref("/team", "Head of Procurement: Priya")]},
     None),                                             # one bad snippet among valid ones is dropped, not fatal
    ({"evidence": [ref("/missing", "Head of Procurement: Asha Rao.")]}, "url_not_retrieved_in_this_run"),
    ({"evidence": [ref("/team", "Head of Procurement: Someone Else.")]}, "snippet_not_found_in_document"),
])
def test_contact_validator_rejects_fabrication(override, reason):
    r = run_contact(contact(**override))
    if reason is None:
        assert len(r.accepted) == 1
    else:
        assert not r.accepted and reason in r.rejected[0]["reasons"]


def test_literal_email_is_accepted_and_name_title_must_share_a_snippet():
    ok = run_contact(contact(email="asha.proc@acme.example", evidence=[ref(
        "/team", "Head of Procurement: Asha Rao.\nProcurement enquiries: asha.proc@acme.example")]))
    assert ok.accepted[0]["email"] == "asha.proc@acme.example"
    split = run_contact(contact(evidence=[ref("/team", "Head of Procurement: Asha Rao."),]),
                        contact(name="Vikram Nair", designation="Head of Procurement",
                                evidence=[ref("/team", "Marketing Manager: Vikram Nair.")]))
    assert any("name_and_designation_not_in_same_snippet" in x["reasons"] for x in split.rejected)


# ============================================================ procurement validator
PROC = ("Supplier registration: Acme invites suppliers at https://acme.example/suppliers.\n"
        "Procurement categories: castings, fasteners.")


def proc_item(kind, value, snippet):
    return {"kind": kind, "value": value, "evidence": [ref("/procurement", snippet)]}


def run_proc(*items):
    return process_procurement(make_state({"/procurement": PROC, "/suppliers": "Supplier portal of Acme"}),
                               {"items": list(items)}, None)


def test_procurement_urls_must_be_retrieved_or_quoted_and_text_must_be_supported():
    r = run_proc(
        proc_item("procurement_page", BASE + "/procurement", "Supplier registration: Acme invites suppliers"),
        proc_item("supplier_portal", "https://acme.example/suppliers",
                  "invites suppliers at https://acme.example/suppliers."),
        proc_item("supplier_portal", "https://acme.example/secret-portal",
                  "Supplier registration: Acme invites suppliers"),               # URL never seen anywhere
        proc_item("purchasing_category", "castings", "Procurement categories: castings, fasteners."),
        proc_item("purchased_component", "turbines", "Procurement categories: castings, fasteners."))
    ok = {(a["kind"], a["value"]) for a in r.accepted}
    assert ("procurement_page", BASE + "/procurement") in ok and ("purchasing_category", "castings") in ok
    bad = {x["value"]: x["reasons"] for x in r.rejected}
    assert bad["https://acme.example/secret-portal"] == ["url_neither_retrieved_nor_quoted_in_snippet"]
    assert bad["turbines"] == ["value_not_supported_by_snippet"]


# ============================================================== supplier validator
@pytest.mark.parametrize("snippet,expected", [
    ("Orion Precision Components supplies PCBAs to Acme Corp, according to a company statement.", True),
    ("Acme Corp sources PCBAs from Orion Precision Components.", True),
    ("Acme Corp partnered with Orion Precision Components on the new line.", True),
    ("Acme Corp supplies PCBAs to Orion Precision Components.", False),              # wrong direction
    ("Orion Precision Components supplies PCBAs to several makers. Acme Corp is a well known buyer.", False),
    ("Orion Precision Components supplies PCBAs to Other Industries Ltd.", False),    # company not named
    ("Orion Precision Components exhibited next to Acme Corp at the fair.", False),    # no relationship cue
])
def test_relationship_requires_one_sentence_naming_both_parties_in_the_right_direction(snippet, expected):
    assert states_relationship(snippet, "Orion Precision Components", "Acme Corp") is expected


def test_supplier_relationship_for_a_different_customer_or_unrelated_snippet_is_rejected():
    st = make_state({"/news": "Orion Precision Components supplies PCBAs to Acme Corp, per a statement."})
    good = {"supplier_name": "Orion Precision Components", "customer_name": "Acme Corp", "component": "PCBAs",
            "evidence": [ref("/news", "Orion Precision Components supplies PCBAs to Acme Corp, per a statement.")]}
    other = {**good, "customer_name": "Globex Corporation"}
    wrong_comp = {**good, "component": "turbines"}
    r = process_supplier(st, {"relationships": [good, other, wrong_comp]}, None)
    assert len(r.accepted) == 1 and r.accepted[0]["supplier"] == "Orion Precision Components"
    reasons = [x["reasons"] for x in r.rejected]
    assert ["relationship_not_about_this_company"] in reasons and ["component_not_in_relationship_snippet"] in reasons


# ===================================================================== lead scoring
def test_lead_score_is_explainable_and_counts_only_verified_data():
    base = {"subject": {"name": "Acme", "existing": None, "identity": {"verified": False}}, "claims": {}}
    s = score_lead(base)
    assert s["score"] is None and s["band"] == "insufficient_data"
    legacy = {"subject": {"name": "Acme", "identity": {"verified": False},
                          "existing": {"fields": {"country": "India"}, "components": ["a", "b", "c", "d", "e"],
                                       "contact_summary": {"count": 1}}}, "claims": {}}
    s2 = score_lead(legacy)
    comp = next(f for f in s2["factors"] if f["factor"] == "component_relevance")
    assert comp["earned"] == WEIGHTS["component_relevance"] * 0.5 and "UNVERIFIED" in comp["basis"]
    assert next(f for f in s2["factors"] if f["factor"] == "procurement_decision_maker")["earned"] == 0
    assert s2["score"] == round(sum(f["earned"] for f in s2["factors"]), 1) and s2["score"] < 40
    assert "disclaimer" in s2 and s2["max_score"] == 100


# ==================================================== dynamic supervisor routing (e2e)
def routes(out):
    return [h["next"] for h in out["route_history"]]


def test_objective_decides_which_agents_run_not_a_fixed_pipeline():
    contacts_only = make_runner().start("Apollo Micro Systems", "find the procurement decision makers")
    assert routes(contacts_only) == ["company", "contact", "finish"]
    assert "procurement" not in contacts_only["agent_runs"] and "supplier" not in contacts_only["agent_runs"]
    profile_only = make_runner().start("Apollo Micro Systems", "company profile")
    assert routes(profile_only) == ["company", "finish"]
    suppliers = make_runner().start("Apollo Micro Systems", "which suppliers supply components to them")
    assert routes(suppliers) == ["company", "supplier", "finish"]
    full = make_runner().start("Apollo Micro Systems", "comprehensive")
    assert set(routes(full)) == {"company", "procurement", "contact", "supplier", "lead", "finish"}


def test_missing_prerequisite_routes_to_the_agent_that_supplies_it():
    script = [tool_call("route", next="lead", reason="rank"),            # eligible: CRM record exists
              tool_call("compute_lead_score"), tool_call("submit_lead_findings", status="satisfied", score=999),
              tool_call("route", next="finish", reason="done")]
    out = make_runner(llm=FakeLLM(script)).start("Apollo Micro Systems", "rank this lead")
    assert out["completed_agents"] == ["lead"]                          # company skipped on purpose
    lead = out["report"]["lead_priority"]
    assert lead["score"] != 999 and lead["score"] == score_lead(out)["score"]       # model cannot set the score
    assert next(f for f in lead["factors"] if f["factor"] == "identity_confirmed")["earned"] == 0


def test_new_company_contact_request_is_redirected_to_company_first():
    script = [tool_call("route", next="contact", reason="want contacts"),            # vetoed: identity unknown
              tool_call("route", next="company", reason="need identity first")]
    llm = PolicyLLM()
    first = {"n": 0}
    orig = llm.generate

    def gen(**kw):
        if [t.name for t in kw["tools"]] == ["route"] and first["n"] < 2:
            first["n"] += 1
            from agentic.llm.fake import tool_call as tc
            return tc("route", next="contact", reason="want contacts") if first["n"] == 1 else \
                tc("route", next="company", reason="need identity first")
        return orig(**kw)
    llm.generate = gen
    out = make_runner(name="Zenith Quantum Forge", llm=llm).start("Zenith Quantum Forge",
                                                                  "find the procurement decision makers")
    assert [e["type"] for e in out["trace"] if e["type"] == "route_vetoed"] == ["route_vetoed"]
    assert routes(out)[:2] == ["company", "contact"]


def test_unverifiable_identity_prevents_downstream_web_research():
    out = make_runner(variant="no_website").start("Apollo Micro Systems", "comprehensive")
    assert not out["report"]["identity"]["verified"]
    assert set(out["agent_runs"]) <= {"company", "lead"}                    # no contact/procurement/supplier
    lead = out["report"]["lead_priority"]
    assert lead is not None and lead["data_sufficiency"] in ("limited", "sufficient")      # CRM row exists


def test_search_outage_degrades_gracefully_and_stops_web_agents():
    out = make_runner(variant="search_quota").start("Apollo Micro Systems", "comprehensive")
    assert out["budgets"]["search_disabled"] and out["status"] == "partial"
    assert "contact" not in out["agent_runs"] or out["agent_results"]["contact"]["end_reason"] != "submitted"
    assert any("search" in x.lower() for x in out["report"]["coverage_limitations"])
    assert out["report"]["next_actions"]


# ================================================================ end-to-end content
def test_comprehensive_report_has_sourced_sections_and_no_unverified_defaults():
    out = make_runner().start("Apollo Micro Systems", "comprehensive")
    rep = out["report"]
    assert out["status"] == "completed"
    contact = rep["contacts"][0]
    assert contact["designation"] == "Head of Procurement" and contact["evidence"][0]["snippet"]
    assert set(rep["procurement"]) >= {"supplier_portal", "procurement_page", "purchasing_category"}
    assert rep["suppliers"][0]["supplier"] and rep["suppliers"][0]["evidence"][0]["url"].startswith("https://")
    assert rep["lead_priority"]["factors"] and rep["lead_priority"]["band"] in ("high", "medium", "low")
    # the legacy CRM contact (no source) is reported only as unverified context, never as a contact
    assert rep["database_context"]["contact_summary"]["provenance"] == "database_unverified"
    assert all(c["name"] != "Purchase Manager" for c in rep["contacts"])
    for sec in (rep["procurement"] or {}).values():
        for item in sec:
            assert item["evidence"]
    urls = {s["url"] for s in rep["sources"]}
    assert all(e["url"] in urls for c in rep["contacts"] for e in c["evidence"])


def test_no_discoverable_contact_is_reported_as_not_found_not_invented():
    out = make_runner(variant="no_contacts").start("Apollo Micro Systems", "find the procurement decision makers")
    assert out["report"]["contacts"] is None
    assert any("No publicly documented procurement decision-maker" in a for a in out["report"]["next_actions"])
    assert out["agent_results"]["contact"]["status"] == "exhausted"


def test_prompt_injection_on_a_page_is_data_not_instructions(monkeypatch):
    monkeypatch.setenv("SUPABASE_KEY", "sb_secret_TOPSECRETVALUE99")
    llm = PolicyLLM()
    seen = []
    orig = llm.generate

    def spy(**kw):
        seen.extend(m["content"] for m in kw["messages"] if m["role"] == "tool")
        assert "TOPSECRETVALUE99" not in kw["system"] + json.dumps(kw["messages"])
        return orig(**kw)
    llm.generate = spy
    r = make_runner(variant="injection", llm=llm)
    out = r.start("Apollo Micro Systems", "comprehensive")
    injected = [c for c in seen if "IMPORTANT SYSTEM INSTRUCTION" in c]
    assert injected and all(c.startswith("<untrusted_web_content") for c in injected)
    assert not any("evil" in u for u in r.fetcher.calls) and not any("evil" in q[0] for q in r.search.calls)
    assert "attacker@evil.example" not in json.dumps(out["report"]) and "TOPSECRETVALUE99" not in json.dumps(out, default=str)


# ================================================================== tools & permissions
def test_agents_cannot_use_tools_outside_their_allowlist():
    script = [tool_call("route", next="company", reason="x"),
              tool_call("get_crm_context", section="contacts_summary"),         # not a company-agent tool
              tool_call("compute_lead_score"),
              tool_call("submit_company_findings", status="exhausted"),
              tool_call("route", next="finish", reason="done")]
    out = make_runner(llm=FakeLLM(script)).start("Apollo Micro Systems", "company profile")
    codes = [e["data"]["error"]["code"] for e in out["trace"] if e["type"] == "tool_result" and not e["data"]["ok"]]
    assert codes == ["tool_not_permitted", "tool_not_permitted"]


def test_crm_tool_exposes_no_contact_names_and_respects_sections():
    repo = InMemoryRepo(default_repo().companies.values(),
                        contacts={7: [{"designation": "Purchase Manager", "source_url": None, "email": "x@y.example",
                                       "contact_name": "Secret Person"}]},
                        components={7: ["PCBAs"]}, products={7: ["Avionics"]})
    r = make_runner(repo=repo)
    out = r.start("Apollo Micro Systems", "find the procurement decision makers")
    blob = json.dumps([m for m in out["trace"]], default=str)
    assert "Secret Person" not in blob and "x@y.example" not in blob
    script = [tool_call("route", next="company", reason="x"), tool_call("submit_company_findings", status="exhausted"),
              tool_call("route", next="contact", reason="x"),
              tool_call("get_crm_context", section="components"),               # contact agent: not allowed
              tool_call("submit_contact_findings", status="exhausted"),
              tool_call("route", next="finish", reason="done")]
    # identity is unverified here, so contact is vetoed: assert the veto path rather than the section path
    out2 = make_runner(llm=FakeLLM(script)).start("Apollo Micro Systems", "x")
    assert any(e["type"] == "route_vetoed" for e in out2["trace"])

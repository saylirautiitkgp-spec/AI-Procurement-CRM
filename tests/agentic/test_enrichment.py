"""Regression tests for live run 001ebec33711 (AIRBUS India, headquarters): field-aware source ranking, relevant official
press pages, excluded regional pages, packet windows, fabricated quotations, plus the full-record enrichment plan
(pinned company id, companies-table review, grouping by target table, preserved values). Offline only."""

import copy
import json
import pathlib
import re

import pytest

from agentic import evidence as ev
from agentic import fields as fl
from agentic.backends.fixtures import FixtureFetcher, FixtureSearch
from agentic.commit import InMemoryCommitRepo
from agentic.llm.base import LLMResponse
from agentic.llm.fake import tool_call
from agentic.prefetch import (build_packet, field_scoring, is_penalised, place_tokens, rank_official_candidates,
                              _window)
from agentic.repo import InMemoryRepo
from tests.agentic.fieldllm import FieldLLM
from tests.agentic.support import COMPANY_A
from tests.agentic.test_bridge import commit_tables
from tests.agentic.test_fields import PAR, runner_for
from tests.agentic.test_hybrid import SpyRepo, llm_events
from tests.agentic.test_prefetch import events, packet_json

ROOT = pathlib.Path(__file__).resolve().parents[2]
SCHEMA = json.loads((ROOT / "tests" / "fixtures" / "schema_introspection.json").read_text(encoding="utf-8"))
A = "https://www.airbus.com/en"
INDIA = A + "/about-us/our-worldwide-presence/airbus-in-asia-pacific/airbus-in-india"
EUROPE = A + "/about-us/our-worldwide-presence/airbus-in-europe"
AMERICAS = A + "/about-us/our-worldwide-presence/airbus-in-americas"
CAREERS = A + "/careers/life-at-airbus/working-here/working-at-airbus-india"
PRESS = A + "/newsroom/press-releases/2024-10-airbus-inaugurates-india-and-south-asia-headquarters-in-new-delhi"
ABOUT = A + "/about-us"
HQ_SENTENCE = "Airbus today inaugurated its India and South Asia headquarters in New Delhi, India."
FILL = "Airbus designs, manufactures and delivers aerospace products, services and solutions worldwide. "

RESULTS = [{"url": u, "title": t} for u, t in [
    (INDIA, "Airbus in India | Airbus"), (CAREERS, "Working at Airbus in India | Airbus"),
    (EUROPE, "Airbus in Europe | Airbus"), (AMERICAS, "Airbus in the Americas | Airbus"),
    (PRESS, "Airbus inaugurates India and South Asia headquarters in New Delhi | Airbus"), (ABOUT, "About us | Airbus")]]


def airbus_world(hq_offset_chars=0):
    """The India page does NOT mention a headquarters (as in the live run); the official press release does."""
    pad = FILL * (hq_offset_chars // len(FILL) + 1) if hq_offset_chars else ""
    pages = {INDIA: ("Airbus in India | Airbus", "Airbus in India. Airbus has been present in India since 1976. " + FILL * 40),
             EUROPE: ("Airbus in Europe | Airbus", "Airbus in Europe. Operations in France, Germany, Spain and the UK. " + FILL * 40),
             AMERICAS: ("Airbus in the Americas | Airbus", "Airbus in the Americas. Final assembly in Mobile, Alabama. " + FILL * 40),
             CAREERS: ("Working at Airbus in India | Airbus", "Careers at Airbus in India. Join our teams. " + FILL * 40),
             PRESS: ("Airbus inaugurates India and South Asia headquarters in New Delhi | Airbus",
                     "New Delhi, 22 October 2024. " + pad + HQ_SENTENCE + " " + FILL * 30),
             ABOUT: ("About us | Airbus", "About Airbus. Airbus is a European aerospace company. " + FILL * 40)}
    links = [{"url": ABOUT, "text": "About us"}, {"url": CAREERS, "text": "Careers"}]
    return {"search_index": [{"url": u, "title": t, "content": body[:500]} for u, (t, body) in pages.items()],
            "pages": {u: {"title": t, "text": body, "links": links} for u, (t, body) in pages.items()},
            "facts": {}, "variant": "standard", "fetch_errors": {}}


AIRBUS_CRM = InMemoryRepo([{**COMPANY_A, "company_id": 11, "company_name": "Airbus India", "website": None}],
                          products={11: ["Aircraft Engineering"]})


class AirbusLLM:
    """Company agent double: honest, or one of several fabrication styles. Reads the packet it was actually sent."""
    name, model = "airbus-llm", "m"

    def __init__(self, mode="honest"):
        self.mode, self.calls = mode, []

    def generate(self, *, system, messages, tools, max_output_tokens, force_tool=False):
        self.calls.append({"tools": [t.name for t in tools], "messages": copy.deepcopy(messages)})
        packet = packet_json(messages[0]["content"]) or {"documents": []}
        docs = {d["url"]: d for d in packet["documents"]}
        site = {"field": "official_website", "value": INDIA,
                "evidence": [{"url": INDIA, "snippet": "Airbus in India. Airbus has been present in India since 1976."}]}
        if self.mode == "honest":
            src = next((u for u, d in docs.items() if HQ_SENTENCE in d["text"]), None)
            claims = [site] + ([{"field": "headquarters", "value": "New Delhi, India",
                                 "evidence": [{"url": src, "snippet": HQ_SENTENCE}]}] if src else [])
        else:
            fake = "Airbus' headquarters in India is located in New Delhi, and its two main wings are also based in the capital."
            ev_ = {"fabricated": [{"url": INDIA, "snippet": fake}],                                  # the live-run failure
                   "wrong_document": [{"url": INDIA, "snippet": HQ_SENTENCE}],                       # real text, wrong source
                   "never_retrieved": [{"url": A + "/never-fetched", "snippet": HQ_SENTENCE}],
                   "paraphrase": [{"url": PRESS, "snippet": "Airbus opened its new Indian head office in the capital city."}],
                   }[self.mode]
            claims = [site, {"field": "headquarters", "value": "New Delhi, India", "evidence": ev_}]
        return LLMResponse(tool_calls=[__import__("agentic.llm.base", fromlist=["ToolCallRequest"]).ToolCallRequest(
            id=f"c{len(self.calls)}", name="submit_company_findings",
            args={"status": "satisfied", "identity_website": INDIA, "claims": claims})])


def run_hq(mode="honest", w=None, repo=None, **kw):
    llm = AirbusLLM(mode)
    r = runner_for(w=w or airbus_world(), llm=llm, repo=repo or AIRBUS_CRM, **kw)
    out = r.start("AIRBUS India", None, fields=["headquarters"], refresh=True, expected_company_id=11)
    return r, out, llm


# ================================================================================== field-aware ranking
def test_for_a_headquarters_request_the_official_press_release_is_relevant_and_regional_pages_are_not():
    ranked = rank_official_candidates("AIRBUS India", RESULTS, ["headquarters"])
    by_url = {p["url"]: p for p in ranked}
    assert not by_url[PRESS]["penalised"] and by_url[CAREERS]["penalised"]
    assert by_url[EUROPE]["penalised"] and by_url[AMERICAS]["penalised"]                # other regions of the group
    assert not by_url[INDIA]["penalised"] and not by_url[ABOUT]["penalised"]
    top3 = [p["url"] for p in ranked if not p["penalised"]][:3]
    assert PRESS in top3 and INDIA in top3
    assert ranked[0]["url"] in (INDIA, PRESS)


def test_without_a_location_field_press_pages_stay_penalised_and_regions_are_still_excluded():
    generic = {p["url"]: p for p in rank_official_candidates("AIRBUS India", RESULTS)}
    assert generic[PRESS]["penalised"] and generic[EUROPE]["penalised"]
    other = {p["url"]: p for p in rank_official_candidates("AIRBUS India", RESULTS, ["supplier_portal"])}
    assert other[PRESS]["penalised"]                                                     # no profile for that field
    boost, penalised, words = field_scoring(["headquarters", "employee_count"])
    assert "press" not in penalised and "career" in penalised and "headquarter" in words


def test_regional_exclusion_needs_a_place_word_in_the_company_name():
    assert place_tokens("AIRBUS India") == ("india",) and place_tokens("Apollo Micro Systems") == ()
    assert is_penalised(EUROPE, "Airbus in Europe", place=("india",))
    assert not is_penalised(EUROPE, "Airbus in Europe", place=())                       # a global entity: Europe is fine
    assert not is_penalised(INDIA, "Airbus in India", place=("india",))                 # mentions its own region


# =============================================================================================== windows
def test_a_long_page_shows_the_part_that_states_the_requested_fact_as_a_real_slice_of_the_stored_text():
    words = field_scoring(["headquarters"])[2]
    full = "Intro text. " * 400 + HQ_SENTENCE + " More text. " * 200
    text, offset = _window(full, 2500, words)
    assert offset > 0 and HQ_SENTENCE in text and text in full and full[offset:offset + len(text)] == text
    early = HQ_SENTENCE + " rest. " * 400
    assert _window(early, 2500, words)[1] == 0                                           # fact already in the first window
    assert _window(full, 2500, ())[1] == 0                                               # no focus: plain prefix as before
    body, _ = build_packet([{"url": "https://a.example", "title": "t", "text": full[:3000], "full": full}], [], 2500, 7500, words)
    doc = json.loads(body)["documents"][0]
    assert doc["text_offset"] == offset and HQ_SENTENCE in doc["text"]


def test_the_headquarters_sentence_far_down_the_press_release_is_still_found_verified_and_proposed():
    r, out, llm = run_hq("honest", w=airbus_world(hq_offset_chars=4200))
    packet = packet_json(llm.calls[0]["messages"][0]["content"])
    press = next(d for d in packet["documents"] if d["url"] == ev.canonical_url(PRESS))
    assert press.get("text_offset", 0) > 0 and HQ_SENTENCE in press["text"]
    (item,) = out["report"]["change_set"]["items"]
    assert item["field"] == "headquarters" and item["action"] == "fill_empty"
    stored = ev.find_doc(out["ledger"], PRESS)
    assert HQ_SENTENCE in stored["text"] and ev.validate_snippet(out["ledger"], PRESS, HQ_SENTENCE)[0]


# =========================================================================== the live-run replay (end to end)
def test_replay_of_the_live_run_now_builds_the_packet_from_relevant_official_pages():
    r, out, llm = run_hq("honest")
    pf = events(out, "prefetch")[0]["data"]
    pages = pf["pages"]
    assert pf["status"] == "packet" and ev.canonical_url(PRESS) in pages and ev.canonical_url(INDIA) in pages
    assert not any(u in pages for u in (ev.canonical_url(EUROPE), ev.canonical_url(AMERICAS), ev.canonical_url(CAREERS)))
    assert len(llm.calls) == 1 and len(llm_events(out)) == 1                              # one request, no follow-up needed
    (item,) = out["report"]["change_set"]["items"]
    assert item["target"] == "company_profile.headquarters" and item["proposed_value"] == "New Delhi, India"
    assert item["source_urls"] == [ev.canonical_url(PRESS)] and item["validation_status"] == "verified"
    assert item["evidence"][0]["snippet"] == HQ_SENTENCE and item["confidence"] is not None
    assert out["status"] == "completed" and out["report"]["change_set"]["unresolved"] == []


@pytest.mark.parametrize("mode,reason", [
    ("fabricated", "snippet_not_found_in_document"),          # exactly what the model did in run 001ebec33711
    ("wrong_document", "snippet_not_found_in_document"),      # real sentence, but it is on a different page
    ("paraphrase", "snippet_not_found_in_document"),
    ("never_retrieved", "url_not_retrieved_in_this_run"),
])
def test_fabricated_or_misattributed_quotations_are_rejected_and_nothing_is_proposed(mode, reason):
    r, out, llm = run_hq(mode)
    cs = out["report"]["change_set"]
    assert cs["items"] == [] and cs["unresolved"][0]["field"] == "headquarters"
    assert cs["unresolved"][0]["reason"] == "rejected_by_validation" and reason in cs["unresolved"][0]["rejection_reasons"]
    assert not any(c["field"] == "headquarters" for c in out["claims"].get("company", []))
    rejected = out["rejected"]["company"]
    assert any(x["field"] == "headquarters" and x["value"] == "New Delhi, India" for x in rejected)   # kept as UNVERIFIED
    assert cs["writes_performed"] is False


def test_the_validator_accepts_exactly_the_text_that_is_in_the_stored_source():
    r, out, _ = run_hq("honest")
    led = out["ledger"]
    assert ev.validate_snippet(led, PRESS, HQ_SENTENCE)[0] is True
    assert ev.validate_snippet(led, INDIA, HQ_SENTENCE)[0] is False                      # same words, wrong document
    assert ev.validate_snippet(led, PRESS, HQ_SENTENCE.replace("New Delhi", "Mumbai"))[0] is False


# ======================================================================== pinned record, no duplicates
def two_company_repo():
    return InMemoryRepo([{**COMPANY_A, "company_id": 11, "company_name": "Airbus India"},
                         {**COMPANY_A, "company_id": 12, "company_name": "Zenith Textiles Ltd"}])


def test_the_run_is_pinned_to_the_expected_record_and_fails_instead_of_researching_another():
    r = runner_for(w=airbus_world(), llm=AirbusLLM(), repo=two_company_repo())
    ok = r.start("AIRBUS India", None, fields=["headquarters"], expected_company_id=11)
    assert ok["subject"]["existing"]["company_id"] == 11 and ok["agent_runs"] == {"company": 1}
    for bad in (12, 99):                                                                  # other record / nonexistent
        r = runner_for(w=airbus_world(), llm=AirbusLLM(), repo=two_company_repo())
        out = r.start("AIRBUS India", None, fields=["headquarters"], expected_company_id=bad)
        assert out["status"] == "failed" and out["stop_reason"] == "expected_company_not_matched"
        assert out["agent_runs"] == {} and r.search.calls == [] and out["budgets"]["search_credits"] == 0


def test_a_pinned_run_never_proposes_creating_a_company_even_with_writes_configured():
    commit = InMemoryCommitRepo(commit_tables())
    commit.tables["companies"][0].update(company_id=11, company_name="Airbus India")
    commit.tables["company_profile"][0]["company_id"] = 11
    r = runner_for(w=airbus_world(), llm=AirbusLLM(), repo=AIRBUS_CRM)
    r.deps.commit_repo, r.deps.schema = commit, __import__("agentic.schema_contract", fromlist=["x"]).SchemaContract.from_introspection(SCHEMA)
    out = r.start("AIRBUS India", None, run_id="r1", fields=["headquarters"], refresh=True, expected_company_id=11)
    res = r.propose_from_change_set("r1")
    assert [p["kind"] for p in res["proposals"]] == ["update_company_fields"]
    ops = r.deps.store.list_proposals("r1")[0]["change"]["operations"]
    assert all(op["table"] != "companies" or op["op"] == "update" for op in ops) and commit.ops == []


# =========================================================== the full-record enrichment plan (offline model)
FULL = ["official_website", "linkedin_url", "headquarters", "industry", "business_overview", "products_services",
        "employee_count", "annual_revenue", "founded_year", "parent_company", "ownership_type", "company_type",
        "manufacturing_locations", "supplier_portal", "procurement_page", "purchasing_category", "purchased_component",
        "process_note", "procurement_contacts", "suppliers", "lead_score", "lead_next_actions"]


def test_the_companies_table_review_covers_every_real_column_and_says_why_each_is_or_is_not_enriched():
    real = {c["column"] for c in SCHEMA["columns"] if c["table"] == "companies"}
    assert set(fl.COMPANIES_COLUMNS_REVIEW) == real                                       # nothing forgotten, nothing invented
    rows = {r["column"]: r for r in fl.companies_review({"fields": {"country": "India", "website": None}}, ["official_website"])}
    assert rows["companies.website"]["status"] == "enrichable" and rows["companies.website"]["field"] == "official_website"
    assert rows["companies.website"]["researched_in_this_run"] is True and rows["companies.website"]["current_value_present"] is False
    assert rows["companies.country"]["status"] == "preserve_if_set" and rows["companies.country"]["current_value_present"] is True
    for legacy in ("companies.revenue", "companies.company_type"):
        assert rows[legacy]["status"] == "context_only"
    for fixed in ("companies.company_id", "companies.company_name", "companies.sector_id", "companies.created_at"):
        assert rows[fixed]["status"] in ("key", "identity", "curated", "system") and "reason" in rows[fixed]
    enrichable = {c for c, r in rows.items() if r["status"] == "enrichable"}
    assert enrichable == {"companies.website", "companies.linkedin_url"}                  # all write-allowlisted + evidence-backed
    from agentic import write_gate as wg
    assert {c.split(".")[1] for c in enrichable} <= set(wg.COLUMNS["companies"])


def test_full_enrichment_runs_all_five_agents_preserves_stored_values_and_groups_findings_by_target_table():
    spy = SpyRepo(InMemoryRepo([{**COMPANY_A, "company_id": 11, "company_name": "Apollo Micro Systems", "website": None,
                                 "company_type": "Global"}], products={11: ["Aircraft Engineering"]},
                               components={11: ["PCBAs"]}, profiles={11: {"industry": "Aerospace & Defence"}}))
    llm = FieldLLM()
    r = runner_for(llm=llm, repo=spy)
    out = r.start("Apollo Micro Systems", None, fields=FULL, refresh=False, expected_company_id=11)
    assert set(out["agent_runs"]) == {"company", "contact", "procurement", "supplier", "lead"}
    assert out["agent_runs"]["company"] == 1 and not llm_events(out, "supervisor")
    cs = out["report"]["change_set"]
    assert {s["field"] for s in cs["skipped"]} == {"industry"}                            # stored value preserved, not researched
    assert all(i["field"] != "industry" for i in cs["items"]) and cs["refresh"] is False
    assert not any(i["action"] == "update_conflict" for i in cs["items"])                 # no overwrite without approval
    tables = cs["by_target_table"]
    assert {"company_profile", "contacts", "company_suppliers"} <= set(tables) and "report_only" in tables
    assert tables["company_profile"]["count"] >= 2 and all(set(v) >= {"count", "actions", "item_ids", "fields"} for v in tables.values())
    assert cs["unresolved"] and all(u["old_value"] is None or u["field"] != "industry" for u in cs["unresolved"])
    assert len(cs["companies_table_review"]) == len(fl.COMPANIES_COLUMNS_REVIEW)
    # read-only CRM access: exactly the sections the five agents need, nothing else
    reads = {c for c in spy.calls if c.startswith("get_") or c == "find_candidates"}
    assert reads == {"find_candidates", "get_company", "get_profile", "get_products", "get_components",
                     "get_contact_summary", "get_company_suppliers"}
    assert "get_lead_status" not in reads
    assert len(llm_events(out)) <= 6 and out["budgets"]["search_credits"] <= 6
    assert cs["writes_performed"] is False and out["proposals"] == []

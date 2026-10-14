"""Evidence decoding, quote integrity, canonical website mapping, insufficient industry evidence, the read-only corrective
audit, direct-purchase evidence for procurement insights and the pinned-identity Procurement-only run.
Offline only: fake HTTP sessions, synthetic worlds, scripted models, in-memory commit repository."""

import copy

import pytest

from agentic import apply_validated as av
from agentic import audit_existing as audit
from agentic import company_live
from agentic import evidence as ev
from agentic import field_support
from agentic import fields as fl
from agentic import preflight as pf
from agentic import website_map as wm
from agentic.agents import AGENT_SPECS
from agentic.backends.fetch import decode_body
from agentic.backends.fixtures import FixtureFetcher, FixtureSearch
from agentic.changeset import build_change_set
from agentic.commit import InMemoryCommitRepo
from agentic.config import Efficiency, Limits, Settings
from agentic.persistence.store import Store
from agentic.proposals import ProposalService
from agentic.repo import InMemoryRepo
from agentic.runtime import AgenticRunner, make_deps
from agentic.schema_contract import SchemaContract
from tests.agentic.fieldllm import FieldLLM
from tests.agentic.support import COMPANY_A
from tests.agentic.test_company_only import seed
from tests.agentic.test_limits_and_fetch import Resp, fetcher
from tests.agentic.test_prefetch import world
from tests.agentic.test_scoped_enrichment import REAL_SCHEMA, run, tables

SCHEMA = SchemaContract.from_file(REAL_SCHEMA)
APOS = "India’s aerospace"


# ============================================================================================ evidence decoding
@pytest.mark.parametrize("raw,header,expected", [
    (APOS.encode("utf-8"), "text/html", APOS),                              # UTF-8 with NO declared charset (requests says latin-1)
    (APOS.encode("utf-8"), "text/html; charset=ISO-8859-1", APOS),          # mislabelled as latin-1
    (APOS.encode("utf-8"), "text/html; charset=utf-8", APOS),
    ("India’s".encode("cp1252"), "text/html; charset=utf-8", "India’s"),     # cp1252 bytes under a UTF-8 label
    ("India’s".encode("cp1252"), "text/html; charset=windows-1252", "India’s"),
    (b'<meta charset="windows-1252">India\x92s', "text/html", "India’s"),         # charset only in <meta>
    ("café".encode("latin-1"), "text/html; charset=iso-8859-1", "café"),   # genuine latin-1 stays latin-1
])
def test_pages_decode_without_mojibake_or_replacement_characters(raw, header, expected):
    out = decode_body(raw, header)
    assert expected in out and "�" not in out and "â" not in out.replace("café", "")


def test_undecodable_bytes_are_dropped_never_replaced():
    assert "�" not in decode_body(b"abc \x81\x8d\x90 def", "text/html; charset=utf-8")
    assert "�" not in decode_body(b"abc \xef\xbf\xbd def", "text/html")            # a literal U+FFFD in the page is stripped too


def test_the_fetcher_uses_the_new_decoding_even_when_the_http_layer_guesses_latin_1():
    body = ("<html><head><title>Airbus in India</title></head><body><p>Strengthening " + APOS + " ecosystem</p></body></html>"
            ).encode("utf-8")
    page = fetcher({"https://acme.example/": Resp(body=body, headers={"Content-Type": "text/html"}, encoding="ISO-8859-1")}
                   ).fetch("https://acme.example/")
    assert "India’s aerospace" in page.text and "â" not in page.text and "�" not in page.text


def test_a_replacement_character_can_never_enter_the_ledger_or_a_validated_quote():
    ledger = {}
    ev.add_doc(ledger, url="https://a.example/", title="T�", text="The company� is headquartered in Pune.",
               tool="fetch_page", call_id="c", max_docs=5, max_chars=1000)
    doc = next(iter(ledger.values()))
    assert "�" not in doc["text"] and "�" not in doc["title"]


def test_every_persisted_quote_is_a_verbatim_replacement_free_excerpt_of_a_retrieved_page():
    r, out = run(FieldLLM())
    commit = InMemoryCommitRepo(tables(profile=None))
    svc = ProposalService(Store(":memory:"), commit, schema=SCHEMA)
    av.apply_reviewed(svc, out, out["report"]["change_set"], 7, authoriser="t", identity_verified=True,
                      official_domain=out["subject"]["identity"]["domain"])
    texts = [d["text"] for d in out["ledger"].values()]
    rows = commit.tables["ai_insights"]
    assert rows
    for row in rows:
        quote = row["insight"].rpartition(" (source: ")[0]
        assert "�" not in quote and any(quote in t for t in texts)


# ============================================================================================ canonical website
def test_site_root_and_domain_helpers():
    deep = "https://www.airbus.com/en/about-us/our-worldwide-presence/airbus-in-asia-pacific/airbus-in-india"
    assert wm.site_root(deep) == "https://www.airbus.com/" and not wm.is_root(deep)
    assert wm.is_root("https://www.airbus.com") and wm.is_root("https://www.airbus.com/")
    assert wm.website_domain(deep) == "airbus.com" and wm.site_root("not a url") is None


def ledger_with(*urls, text="Apollo Micro Systems is a company in India.", title="Apollo Micro Systems"):
    ledger = {}
    for u in urls:
        ev.add_doc(ledger, url=u, title=title, text=text, tool="fetch_page", call_id="c", max_docs=9, max_chars=999)
    return ledger


def test_a_deep_page_becomes_the_root_only_when_the_root_was_retrieved_and_the_domain_verifies():
    deep = "https://www.apollo.example/en/india/about"
    assert wm.canonical_website(deep, ledger_with(deep), "Apollo Micro Systems") == (None, "site_root_not_retrieved_in_this_run")
    root, why = wm.canonical_website(deep, ledger_with(deep, "https://www.apollo.example/"), "Apollo Micro Systems")
    assert root == "https://www.apollo.example/" and why == "root_retrieved_and_domain_verified"
    other = ledger_with(deep, "https://www.apollo.example/", text="A page about textiles.", title="Textiles")
    assert wm.canonical_website(deep, other, "Apollo Micro Systems")[0] is None            # domain does not verify for this company
    assert wm.canonical_website("https://www.apollo.example/", {}, "x") == ("https://www.apollo.example/", "claimed_value_is_the_site_root")


def test_the_change_set_proposes_the_root_or_leaves_the_website_unresolved_never_the_deep_page():
    r, out = run(FieldLLM())
    site = "https://www.apollomicrosystems.example"
    assert [c["value"] for c in out["claims"]["company"] if c["field"] == "official_website"] == [site + "/about"]
    item = next(i for i in out["report"]["change_set"]["items"] if i["field"] == "official_website")
    assert item["proposed_value"] == site + "/" and item["target"] == "companies.website"
    # the same run, but the root page was never retrieved (the AIRBUS situation)
    state = copy.deepcopy(out)
    state["ledger"] = {k: v for k, v in state["ledger"].items() if ev.canonical_url(v["url"]) != ev.canonical_url(site)}
    cs = build_change_set({**state, "status": "completed"})
    assert not [i for i in cs["items"] if i["field"] == "official_website"]
    un = next(u for u in cs["unresolved"] if u["field"] == "official_website")
    assert un["reason"] == "website_root_not_supported" and un["candidates"][0]["value"] == site + "/about"


def test_the_root_not_the_quoted_page_is_what_gets_written_to_companies_website():
    r, out = run(FieldLLM())
    commit = InMemoryCommitRepo(tables(profile=None, website=None))
    svc = ProposalService(Store(":memory:"), commit, schema=SCHEMA)
    res = av.apply_reviewed(svc, out, out["report"]["change_set"], 7, authoriser="t", identity_verified=True,
                            official_domain=out["subject"]["identity"]["domain"])
    assert commit.tables["companies"][0]["website"] == "https://www.apollomicrosystems.example" and res["writes"] > 0   # canonical form
    assert "/about" not in commit.tables["companies"][0]["website"]


# ============================================================================================ insufficient industry evidence
@pytest.mark.parametrize("quote,ok", [
    ("Strengthening India’s aerospace ecosystem", False),                       # a heading
    ("Airbus is a global leader in aerospace.", True),
    ("The group is an aerospace manufacturer.", True),
    ("We operate in the aerospace industry.", True),
    ("Aerospace conference 2026 in Delhi", False),
])
def test_industry_needs_a_statement_of_what_the_company_is_or_does(quote, ok):
    assert field_support.check("industry", "Aerospace", [quote])[0] is ok


AIRBUS_ROWS = [
    {"insight_id": 9, "company_id": 11, "insight_type": "Verified business_overview", "generated_by": "agentic-company",
     "insight_title": "business_overview: Airbus' presence and strategy in India focuses on sourcing, engineering",
     "insight": "‘Make in India’ is at the heart of Airbus strategy in India. Airbus' local footprint in sourcing, engineering, "
                "innovation, maintenance and training services is a testament to our commitment. (source: https://www.airbus.com/x)",
     "source_url": "https://www.airbus.com/x"},
    {"insight_id": 10, "company_id": 11, "insight_type": "Verified industry", "generated_by": "agentic-company",
     "insight_title": "industry: Aerospace", "insight": "Strengthening India’s aerospace ecosystem (source: https://www.airbus.com/x)",
     "source_url": "https://www.airbus.com/x"},
]
PROFILE = {"profile_id": 15, "company_id": 11, "industry": "Aerospace", "company_summary": "Airbus' presence ..."}
COMPANY = {"company_id": 11, "company_name": "Airbus India", "website": None, "linkedin_url": None}


def test_the_corrective_audit_marks_industry_insufficient_and_changes_nothing():
    plan = audit.corrective_plan(COMPANY, [PROFILE], AIRBUS_ROWS)
    verdicts = {v["insight_id"]: v for v in plan["insight_verdicts"]}
    assert verdicts[9]["verdict"] == "supported" and verdicts[10]["verdict"] == "insufficient"
    assert verdicts[10]["reasons"] == ["quote_does_not_state_field"] and not any("replacement" in r for v in verdicts.values() for r in v["reasons"])
    assert plan["writes_performed"] == 0 and plan["read_only"] and plan["profile_id"] == 15 and plan["profile_is_unique_per_company"]
    assert plan["core_fields"]["decision"] == "keep EMPTY"
    targets = [a["target"] for a in plan["actions"]]
    assert targets == ["company_profile.industry (profile_id 15)", "ai_insights.insight_id 10"]      # nothing about row 9
    assert all("approval" in a["requires"] for a in plan["actions"])
    assert "overwrite" in plan["actions"][0]["requires"] and "INSERT only" in plan["actions"][1]["requires"]


def test_the_audit_module_has_no_write_path():
    import inspect
    src = inspect.getsource(audit)
    for word in (".insert(", ".update(", ".delete(", ".upsert(", ".rpc("):
        assert word not in src


def test_an_industry_item_with_heading_only_evidence_is_never_applied():
    it = {"id": "x", "field": "industry", "agent": "company", "entity": "company", "target": "company_profile.industry",
          "action": "fill_empty", "old_value": None, "proposed_value": "Aerospace", "overwrite": False, "confidence": 0.6,
          "validation_status": "verified", "source_urls": ["https://www.airbus.com/x"],
          "evidence": [{"url": "https://www.airbus.com/x", "snippet": "Strengthening India’s aerospace ecosystem"}]}
    sel = av.select_items({"company_id": 11, "company": "Airbus India", "items": [it]}, 11, identity_verified=True)
    assert sel.item_ids == [] and sel.skipped[0]["reason"] == "quote_does_not_state_field"


# ============================================================================================ direct evidence of purchasing
@pytest.mark.parametrize("quote,url,ok", [
    ("Airbus sources aluminium from certified suppliers.", "https://news.example/a", True),
    ("Airbus sells aluminium aircraft components to airlines.", "https://news.example/a", False),
    ("Other Corp sources aluminium from Chile.", "https://news.example/a", False),            # a different company, third party
    ("Other Corp sources aluminium from Chile.", "https://www.airbus.com/a", True),           # the company's own page about its buying
    ("Aluminium is a light metal used in aircraft.", "https://www.airbus.com/a", False),      # no buying statement at all
    ("We buy aluminium for our assembly lines.", "https://www.airbus.com/a", True),
    ("Our customers buy aluminium parts from us.", "https://www.airbus.com/a", False),
])
def test_a_purchase_needs_direct_evidence_that_the_company_buys_it(quote, url, ok):
    res = field_support.check_evidence("purchased_component", "aluminium", [{"url": url, "snippet": quote}],
                                       company="Airbus India", official_domain="airbus.com")
    assert res[0] is ok


# ============================================================================================ procurement-only, pinned identity
SITE = "https://www.apollomicrosystems.example/about"


def runner(llm, w=None, *, store=None, repo=None):
    repo = repo or InMemoryRepo([COMPANY_A], products={7: ["Avionics"]}, components={7: ["PCBAs"]})
    w = w or world()
    search, fetch = FixtureSearch(w), FixtureFetcher(w)
    deps = make_deps(llm=llm, repo=repo, search=search, fetcher=fetch, store=store or Store(":memory:"),
                     settings=Settings(limits=Limits(llm_max_retries=0), efficiency=Efficiency(lazy_crm=True)),
                     specs=AGENT_SPECS, sleep=lambda s: None)
    r = AgenticRunner(deps, AGENT_SPECS)
    r.search = search
    return r


def proc_only(r, pin=SITE, fields=("purchasing_category",)):
    return r.start("Apollo Micro Systems", fields=list(fields), scope_agents=["procurement"], expected_company_id=7,
                   pinned_identity_website=pin)


def test_procurement_only_reverifies_the_pinned_identity_and_never_runs_the_company_agent():
    llm = FieldLLM()
    r = runner(llm)
    out = proc_only(r)
    assert [c["agent"] for c in llm.calls] == ["procurement"] and out["budgets"]["llm_calls"] == 1
    assert out["agent_runs"] == {"procurement": 1} and out["subject"]["identity"]["reason"].startswith("pinned_identity_reverified")
    assert [e for e in out["trace"] if e["type"] == "pinned_identity" and e["data"]["verified"]]
    assert not [e for e in out["trace"] if e["type"] == "llm_call" and e["agent"] in ("supervisor", "company")]
    assert {c["kind"] for c in out["claims"]["procurement"]} >= {"purchasing_category"}
    assert [i["field"] for i in out["report"]["change_set"]["items"]] == ["purchasing_category"]    # only what was asked for


def test_a_pinned_page_that_does_not_belong_to_the_company_blocks_the_run_before_any_model_request():
    llm = FieldLLM()
    out = proc_only(runner(llm, world("ambiguous")), pin="https://www.apolloindustries.example/")
    assert llm.calls == [] and out["agent_runs"] == {} and out["status"] == "partial"
    assert [e for e in out["trace"] if e["type"] == "pinned_identity" and not e["data"]["verified"]]


def test_pinned_identity_is_only_allowed_for_procurement_only_runs_on_a_pinned_record():
    r = runner(FieldLLM())
    with pytest.raises(ValueError, match="pinned_identity_requires"):
        r.start("Apollo Micro Systems", scope_agents=["company", "procurement"], expected_company_id=7, pinned_identity_website=SITE)
    with pytest.raises(ValueError, match="pinned_identity_requires"):
        r.start("Apollo Micro Systems", scope_agents=["procurement"], pinned_identity_website=SITE)
    with pytest.raises(ValueError, match="invalid_scope"):
        r.start("Apollo Micro Systems", scope_agents=["procurement"], expected_company_id=7)


def test_procurement_insights_are_the_only_rows_written_and_existing_ones_are_not_duplicated():
    r = runner(FieldLLM())
    out = proc_only(r)
    commit = InMemoryCommitRepo(tables(profile={"industry": "Aerospace", "company_summary": "keep me"}, website=None))
    svc = ProposalService(Store(":memory:"), commit, schema=SCHEMA)
    kw = dict(authoriser="t", identity_verified=True, official_domain=out["subject"]["identity"]["domain"])
    res = av.apply_reviewed(svc, out, out["report"]["change_set"], 7, **kw)
    assert res["writes"] == 1 and {(o[0], o[1]) for o in commit.ops} == {("insert", "ai_insights")}
    row = commit.tables["ai_insights"][0]
    assert row["company_id"] == 7 and row["insight_type"] == "Procurement: purchasing_category" and row["source_url"]
    assert commit.tables["company_profile"][0] == {"profile_id": 1, "company_id": 7, "industry": "Aerospace", "company_summary": "keep me"}
    again = av.apply_reviewed(ProposalService(Store(":memory:"), commit, schema=SCHEMA), out, out["report"]["change_set"], 7, **kw)
    assert again["writes"] == 0 and len(commit.tables["ai_insights"]) == 1                      # deduplicated against the database


# ============================================================================================ preflight for this run
ARGS = ["--company", "AIRBUS India", "--company-id", "11", "--agents", "procurement", "--model", "gemini-3.8-flash",
        "--pinned-identity-website", "https://www.airbus.com/en/about-us/our-worldwide-presence/airbus-in-asia-pacific/airbus-in-india",
        "--fields", "purchasing_category", "purchased_component", "process_note", "--retries", "1", "--retry-503-only",
        "--total-requests-cap", "2", "--max-requests", "2", "--max-credits", "2", "--apply", "--preflight-only",
        "--quota", "rpm=5,tpm=250000,rpd=20", "--console-remaining", "10"]


def test_the_procurement_only_preflight_plans_one_agent_two_requests_and_the_identity_check(tmp_path, capsys):
    seed(tmp_path, 0)
    assert company_live.main(ARGS + ["--data-dir", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "agents: procurement" in out and "min 0, max 2" in out and "SELECTs (research): min 6, max planned 8" in out
    assert "Tavily credits: max planned 2" in out and "only for a 503 capacity error" in out and "PREFLIGHT OK" in out
    assert "at most 12 row operations on company 11" in out                                        # 2 + 10 insights, no company fields
    assert company_live.main([a for a in ARGS if a != "--pinned-identity-website" and not a.startswith("https://")]
                             + ["--data-dir", str(tmp_path)]) == 2                                 # procurement-only needs the pin

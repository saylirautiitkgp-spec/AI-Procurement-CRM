"""The runtime's database assumptions checked against the REAL schema introspection (read-only catalog output saved on
2026-10-12 as tests/fixtures/schema_introspection.json), the RLS/FK findings, and the delete/merge guards.
Offline only: the fixture is metadata (table/column names and constraints), the repositories are in-memory."""

import ast
import json
import os
import pathlib
import sys
import types

import pytest

from agentic import changeset as cs_mod
from agentic import fields as fl
from agentic import fk_introspect
from agentic import proposals as pr
from agentic import repo as repo_mod
from agentic import write_gate as wg
from agentic.commit import PRIMARY_KEYS, InMemoryCommitRepo, SupabaseCommitRepo
from agentic.merge_plan import MergePlanner
from agentic.merge_sources import InMemoryMergeSource
from agentic.repo import InMemoryRepo, SupabaseReadRepo
from agentic.schema_contract import SchemaContract
from agentic.service import ServiceNotConfigured, build_live_service
from tests.agentic.support import COMPANY_A, default_repo
from tests.agentic.test_bridge import build, commit_tables, run_fields
from tests.agentic.test_prefetch import events, world

ROOT = pathlib.Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "tests" / "fixtures" / "schema_introspection.json"
RAW = json.loads(FIXTURE.read_text(encoding="utf-8"))
SCHEMA = SchemaContract.from_file(str(FIXTURE))
COLS = {(c["table"], c["column"]): c for c in RAW["columns"]}


# ============================================================================== columns used by the runtime
def test_every_column_the_runtime_writes_exists_in_the_real_schema():
    for table, cols in wg.COLUMNS.items():
        for col in cols:
            assert (table, col) in COLS, f"write_gate {table}.{col} is not in the database"
    for key, col in pr.COMPANY_MAP.items():
        assert ("companies", col) in COLS, key
    for key, col in pr.PROFILE_MAP.items():
        assert ("company_profile", col) in COLS, key
    for f in fl.CATALOG:
        if f.table and f.column:
            assert (f.table, f.column) in COLS, f.key
    assert SCHEMA.unconfirmed(f"{t}.{c}" for t, cols in wg.COLUMNS.items() for c in cols) == []


def test_every_column_the_runtime_reads_exists_in_the_real_schema():
    for c in repo_mod.COMPANY_COLUMNS.split(","):
        assert ("companies", c) in COLS, c
    wanted = {"sectors": ["sector_id", "sector"], "products": ["product_name", "company_id"],
              "components": ["component_name", "company_id"], "contacts": ["designation", "source_url", "email", "company_id"],
              "company_suppliers": ["supplier_id", "component", "remarks", "company_id"],
              "suppliers": ["supplier_name", "website", "supplier_id"],
              "lead_status": ["status", "priority", "lead_score", "source", "company_id"]}
    for table, cols in wanted.items():
        for c in cols:
            assert (table, c) in COLS, f"{table}.{c}"


def test_the_named_columns_from_the_review_are_confirmed_with_their_real_types():
    for table, col, ty in [("company_profile", "annual_revenue", "text"), ("company_profile", "profile_id", "bigint"),
                           ("company_profile", "manufacturing_locations", "jsonb"), ("company_profile", "products", "ARRAY"),
                           ("companies", "revenue", "text"), ("contacts", "source_url", "text"),
                           ("contacts", "last_verified", "timestamp without time zone"),
                           ("suppliers", "components_supplied", "text"), ("company_suppliers", "confidence_score", "numeric"),
                           ("ai_insights", "insight", "text")]:
        assert COLS[(table, col)]["data_type"] == ty and SCHEMA.evidence(table, col) == "introspection"
    assert SCHEMA.evidence("companies", "revenue") == "introspection"           # also proven by the live SELECT before


def test_write_gate_value_kinds_fit_the_database_types():
    assert SCHEMA.type_mismatches(wg.COLUMNS) == []


def test_primary_and_unique_keys_match_our_assumptions():
    for table, pk in PRIMARY_KEYS.items():
        assert SCHEMA.primary_keys[table] == [pk], table                         # incl. company_profile.profile_id
    assert SCHEMA.is_unique("company_profile", "company_id")                    # one profile row per company
    assert SCHEMA.is_unique("lead_status", "company_id") and SCHEMA.is_unique("sectors", "sector")
    for table, key in wg.UPDATE_KEYS.items():                                   # every update key identifies one row
        assert SCHEMA.is_unique(table, key) or SCHEMA.primary_keys[table] == [key], table
    # NOT enforced by the database -> duplicates are prevented only by our check-then-insert (documented risk)
    assert not SCHEMA.is_unique("suppliers", "supplier_name")
    assert not any(u["table"] == "company_suppliers" and u["columns"] != ["company_supplier_id"] for u in SCHEMA.uniques)
    assert not any(u["table"] == "ai_insights" and u["columns"] != ["insight_id"] for u in SCHEMA.uniques)


def test_not_null_columns_are_known_and_our_inserts_supply_them():
    assert SCHEMA.required_insert_columns("companies") == ["company_name", "sector_id"]
    assert SCHEMA.required_insert_columns("contacts") == ["company_id", "contact_name"]
    assert SCHEMA.required_insert_columns("company_suppliers") == ["company_id", "supplier_id"]
    assert SCHEMA.required_insert_columns("company_profile") == [] and SCHEMA.required_insert_columns("ai_insights") == []
    runner, deps = build(InMemoryCommitRepo(commit_tables()), schema=SCHEMA)
    run_fields(runner)
    res = runner.propose_from_change_set("r1")
    assert res["proposals"] and all(p["schema_unconfirmed"] == [] for p in res["proposals"])
    for row in deps.store.list_proposals("r1"):
        assert SCHEMA.missing_required(row["change"]["operations"]) == []
        assert row["change"]["schema_fingerprint"] == SCHEMA.fingerprint


def test_an_insert_that_omits_a_not_null_column_cannot_be_approved():
    svc = pr.ProposalService(None, None, schema=SCHEMA)
    bad = {"origin": "change_set", "operations": [{"table": "contacts", "op": "insert", "fields": {"company_id": 7}}]}
    gate = svc._schema_gate(bad)
    assert gate["error"] == "required_columns_missing" and gate["columns"] == ["contacts.contact_name"]
    new_company = {"origin": "change_set", "required_input": ["sector_id"], "operations": [
        {"table": "companies", "op": "insert", "fields": {"company_name": "Acme"}}]}
    assert svc._schema_gate(new_company) is None                                # sector_id is chosen by the approver
    assert svc._schema_gate({**new_company, "required_input": []})["columns"] == ["companies.sector_id"]


def test_the_full_flow_runs_against_the_real_schema_and_records_which_schema_it_trusted():
    commit = InMemoryCommitRepo(commit_tables())
    runner, deps = build(commit, schema=SCHEMA)
    run_fields(runner)
    res = runner.propose_from_change_set("r1")
    svc = runner.service()
    for p in res["proposals"]:
        assert svc.decide(p["proposal_id"], True, by="alice")["ok"], p["kind"]
    assert commit.ops == []
    for p in res["proposals"]:
        assert svc.apply(p["proposal_id"], actor="alice")["ok"]
    assert commit.ops and {op[1] for op in commit.ops} <= set(wg.OPERATIONS)
    assert all(op[0] in ("insert", "update") for op in commit.ops)             # never a delete


def test_legacy_columns_with_the_same_meaning_are_context_only_and_exist():
    assert set(fl.LEGACY_COLUMNS.values()) <= {("companies", "revenue"), ("companies", "company_type")}
    for table, col in fl.LEGACY_COLUMNS.values():
        assert (table, col) in COLS and col not in wg.COLUMNS.get(table, {}) or col == "company_type"
    existing = {"company_id": 7, "fields": {"company_type": "Private"}, "profile": {}}
    claim = {"agent": "company", "field": "company_type", "value": "Public limited", "status": "verified", "confidence": 0.6,
             "confidence_basis": "x", "evidence": [{"url": "https://a.example/about", "snippet": "A public limited company.",
                                                    "retrieved_at": "t"}]}
    cs = cs_mod.build_change_set({"options": {"fields": ["company_type"], "refresh": False},
                                  "subject": {"name": "Acme", "existing": existing}, "claims": {"company": [claim]},
                                  "rejected": {}})
    (item,) = cs["items"]
    assert item["target"] == "company_profile.company_type" and item["old_value"] is None
    assert item["legacy_values"] == {"companies.company_type": "Private"}


def test_the_saved_introspection_file_in_the_agentic_folder_matches_the_test_fixture():
    saved = ROOT / "agentic" / "schema_introspection.json"
    if not saved.exists():
        pytest.skip("operator copy not present")
    assert SchemaContract.from_file(str(saved)).fingerprint == SCHEMA.fingerprint


def test_agentic_schema_file_is_loaded_by_the_service_and_bad_files_are_refused(tmp_path):
    from tests.agentic.test_bridge import factories
    env = {"AGENTIC_MODE": "live", "AGENTIC_ALLOW_LIVE": "1", "AGENTIC_DATA_DIR": str(tmp_path)}
    from tests.agentic.fieldllm import FieldLLM
    f = {**factories(), "llm": lambda: FieldLLM()}
    svc = build_live_service(env={**env, "AGENTIC_SCHEMA_FILE": str(FIXTURE)}, factories=f)
    assert svc.runner.deps.schema.evidence("company_profile", "annual_revenue") == "introspection"
    assert build_live_service(env=env, factories=f).runner.deps.schema.evidence(
        "company_profile", "annual_revenue") == "unconfirmed"                         # no file: strict gate stays
    (tmp_path / "bad.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ServiceNotConfigured, match="AGENTIC_SCHEMA_FILE"):
        build_live_service(env={**env, "AGENTIC_SCHEMA_FILE": str(tmp_path / "bad.json")}, factories=f)
    with pytest.raises(ServiceNotConfigured, match="AGENTIC_SCHEMA_FILE"):
        build_live_service(env={**env, "AGENTIC_SCHEMA_FILE": str(tmp_path / "missing.json")}, factories=f)


# ================================================================================ RLS and foreign keys
def test_rls_is_enabled_everywhere_with_no_policies_and_the_analysis_says_what_that_means():
    info = fk_introspect.analyze(RAW)
    assert RAW["policies"] == [] and RAW["triggers"] == []
    assert len(info["rls_enabled_tables"]) == 17 and info["rls_disabled_tables"] == []
    assert info["rls_enabled_without_policy"] == info["rls_enabled_tables"]       # default-deny for roles without bypass
    assert all(not r["rls_forced"] for r in RAW["rls"])                          # table owners / bypass roles still pass


def test_deleting_a_company_cascades_to_twelve_tables_and_the_planner_knows_it():
    info = fk_introspect.analyze(RAW)
    assert info["blocking_delete_tables"] == [] and info["set_null_tables"] == []
    assert info["cascade_delete_tables"] == ["ai_insights", "company_profile", "company_suppliers", "components", "contacts",
                                             "emails", "interactions", "lead_status", "news", "products", "projects", "tasks"]
    assert info["tables"]["company_profile"]["one_row_per_company"] and info["tables"]["lead_status"]["one_row_per_company"]
    assert not info["tables"]["contacts"]["one_row_per_company"]
    snap = json.loads((ROOT / "tests" / "fixtures" / "duplicate_groups.json").read_text(encoding="utf-8"))
    plan = MergePlanner(InMemoryMergeSource(snap), fk_info=info).build()
    assert plan["summary"]["pairs_ready_to_execute"] == 0                          # still nothing may run


def test_a_key_that_cannot_see_the_crm_fails_the_run_instead_of_looking_like_a_new_company():
    class Blind(InMemoryRepo):
        def __init__(self, visible):
            super().__init__([])
            self.visible = visible

        def can_see_companies(self):
            return self.visible
    from agentic.runtime import AgenticRunner  # noqa: F401
    runner, _ = build(None, repo=Blind(False))
    out = runner.start("Apollo Micro Systems", "company profile")
    assert out["status"] == "failed" and out["stop_reason"] == "crm_not_readable" and out["agent_runs"] == {}
    runner, _ = build(None, repo=Blind(True))
    out = runner.start("Apollo Micro Systems", "company profile")
    assert out["stop_reason"] != "crm_not_readable" and out["subject"]["match"]["status"] == "not_found"


def test_supabase_read_repo_visibility_probe_reports_an_empty_companies_result():
    class Q:
        def __init__(self, data): self.data = data
        def select(self, *_): return self
        def limit(self, *_): return self
        def execute(self): return types.SimpleNamespace(data=self.data)

    class Client:
        def __init__(self, data): self.data = data
        def table(self, name): return Q(self.data)
    assert SupabaseReadRepo(Client([{"company_id": 1}])).can_see_companies() is True
    assert SupabaseReadRepo(Client([])).can_see_companies() is False


# ===================================================================== no accidental delete / merge
def test_the_write_path_has_no_delete_or_merge_capability():
    assert all("delete" not in ops for ops in wg.OPERATIONS.values()) and {"delete", "merge", "upsert"} <= wg.FORBIDDEN_OPS
    with pytest.raises(PermissionError):
        SupabaseCommitRepo(object())._check("companies", "delete")
    with pytest.raises(PermissionError):
        SupabaseCommitRepo(object())._check("companies", "merge")
    assert not hasattr(SupabaseCommitRepo, "delete") and not hasattr(InMemoryCommitRepo, "delete")
    for path in (ROOT / "agentic").rglob("*.py"):
        src = path.read_text(encoding="utf-8")
        tree = ast.parse(src)
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and node.attr in ("delete", "upsert", "rpc"):
                pytest.fail(f"{path.name} uses .{node.attr}() - agentic code must not delete/upsert/call RPCs")


def test_legacy_deletes_are_blocked_unless_an_operator_opts_in(monkeypatch):
    stub = types.ModuleType("database.supabase_client")
    calls = []

    class Chain:
        def table(self, name): calls.append(("table", name)); return self
        def delete(self): calls.append("delete"); return self
        def eq(self, *a): return self
        def execute(self): return None
    stub.supabase = Chain()
    monkeypatch.setitem(sys.modules, "database.supabase_client", stub)
    monkeypatch.delitem(sys.modules, "services.database_service", raising=False)
    from services.database_service import DatabaseService
    svc = DatabaseService()
    monkeypatch.delenv("LEGACY_DELETES_ENABLED", raising=False)
    with pytest.raises(PermissionError, match="cascade"):
        svc.delete("companies", "company_id", 3)
    assert calls == []                                                           # nothing reached the client
    monkeypatch.setenv("LEGACY_DELETES_ENABLED", "1")
    svc.delete("companies", "company_id", 3)
    assert "delete" in calls


def test_the_legacy_merge_script_cannot_run_on_import_or_by_pytest():
    src = (ROOT / "test_duplicate_merge.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    top = [n for n in tree.body if isinstance(n, (ast.Expr, ast.Assign))]
    assert all(isinstance(n.value, ast.Constant) or isinstance(n, ast.Assign) for n in top)      # only the docstring
    assert "LEGACY_DELETES_ENABLED" in src and "__main__" in src
    ini = (ROOT / "pytest.ini").read_text(encoding="utf-8")
    assert "testpaths = tests" in ini                                            # root scripts are never collected


def test_automatic_duplicate_merging_stays_off_by_default():
    tree = ast.parse((ROOT / "agents" / "orchestrator" / "orchestrator.py").read_text(encoding="utf-8"))
    init = next(n for c in tree.body if isinstance(c, ast.ClassDef) for n in c.body
                if isinstance(n, ast.FunctionDef) and n.name == "__init__")
    defaults = dict(zip([a.arg for a in init.args.args][-len(init.args.defaults):],
                        [d.value for d in init.args.defaults]))
    assert defaults["merge_duplicates"] is False

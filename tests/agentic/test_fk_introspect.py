"""FK introspection: parser + planner integration, using a hand-built sample of the documented output shape.
The SQL itself is NOT executed here (read-only catalog query for the user to run)."""

import re

import pytest

from agentic import fk_introspect as fk
from agentic import merge_plan as mp
from agentic.merge_sources import InMemoryMergeSource
from tests.agentic import test_merge_plan as tm

SAMPLE = {
    "foreign_keys": [
        {"table": "products", "column": "company_id", "constraint": "p_fk", "ref_column": "company_id", "on_delete": "c", "on_update": "a", "deferrable": False},
        {"table": "components", "column": "company_id", "constraint": "c_fk", "ref_column": "company_id", "on_delete": "c", "on_update": "a", "deferrable": False},
        {"table": "lead_status", "column": "company_id", "constraint": "l_fk", "ref_column": "company_id", "on_delete": "c", "on_update": "a", "deferrable": False},
        {"table": "contacts", "column": "company_id", "constraint": "k_fk", "ref_column": "company_id", "on_delete": "a", "on_update": "a", "deferrable": False},
        {"table": "company_suppliers", "column": "company_id", "constraint": "s_fk", "ref_column": "company_id", "on_delete": "r", "on_update": "a", "deferrable": False},
        {"table": "news", "column": "company_id", "constraint": "n_fk", "ref_column": "company_id", "on_delete": "n", "on_update": "a", "deferrable": False},
        {"table": "tasks", "column": "company_id", "constraint": "t_fk", "ref_column": "company_id", "on_delete": "a", "on_update": "a", "deferrable": False},
    ],
    "unique_constraints": [
        {"table": "lead_status", "constraint": "lead_status_company_key", "type": "u", "columns": ["company_id"]},
        {"table": "products", "constraint": "products_uq", "type": "u", "columns": ["company_id", "product_name"]},
        {"table": "contacts", "constraint": "contacts_pkey", "type": "p", "columns": ["contact_id"]},
    ],
    "rls": [{"table": "companies", "rls_enabled": True, "rls_forced": False},
            {"table": "contacts", "rls_enabled": False, "rls_forced": False}],
    "policies": [{"table": "companies", "policy": "read_all", "command": "SELECT", "roles": ["anon"]}],
    "triggers": [{"table": "contacts", "trigger": "audit", "event": "INSERT"}],
}


def test_parser_decodes_delete_rules_uniqueness_and_rls():
    info = fk.analyze(SAMPLE)
    t = info["tables"]
    assert t["products"]["on_delete"] == "CASCADE" and t["contacts"]["on_delete"] == "NO ACTION"
    assert t["company_suppliers"]["on_delete"] == "RESTRICT" and t["news"]["on_delete"] == "SET NULL"
    assert t["lead_status"]["one_row_per_company"] is True and t["products"]["one_row_per_company"] is False
    assert t["products"]["composite_uniques"] == [["company_id", "product_name"]]
    assert info["cascade_delete_tables"] == ["components", "lead_status", "products"]
    assert info["blocking_delete_tables"] == ["company_suppliers", "contacts", "tasks"]
    assert info["set_null_tables"] == ["news"]
    assert info["rls_enabled_tables"] == ["companies"] and info["rls_disabled_tables"] == ["contacts"]
    assert "silently DELETE rows in: components, lead_status, products" in info["summary"]


def test_sql_is_a_single_read_only_select():
    sql = fk.INTROSPECTION_SQL.lower()
    assert sql.strip().startswith("select") and sql.count(";") == 1
    assert not re.search(r"\b(insert|update|delete|drop|alter|create|truncate|grant|revoke|execute|call)\b",
                         re.sub(r"'[^']*'", "", sql))
    assert "pg_constraint" in sql and "pg_policies" in sql and "confdeltype" in sql


def _pair_plan(fk_info):
    import sys, types
    mpatch = pytest.MonkeyPatch()
    mpatch.setitem(sys.modules, "database.supabase_client", types.SimpleNamespace(supabase=object()))
    for name in [m for m in sys.modules if m.startswith("services.")]:
        mpatch.delitem(sys.modules, name, raising=False)
    try:
        a = tm.co(1, deps={"products": 1, "lead_status": 1})
        b = tm.co(2, deps={"products": 2, "lead_status": 1, "company_suppliers": 3, "news": 2, "tasks": 1, "contacts": 1})
        data = {"companies": [{k: v.get(k) for k in ("company_id", "company_name", "sector_id", "country", "website")} for v in (a, b)],
                "views": {"1": a, "2": b}}
        return mp.MergePlanner(InMemoryMergeSource(data), fk_info=fk_info).build()
    finally:
        mpatch.undo()


def test_known_delete_rules_turn_vague_warnings_into_specific_consequences():
    plan = _pair_plan(fk.analyze(SAMPLE))
    p = plan["groups"][0]["pairs"][0]
    text = " | ".join(p["execution"]["blockers"])
    assert plan["fk_info_used"] is True
    assert "company_suppliers: 3 row(s) would make the delete FAIL (ON DELETE RESTRICT)" in text
    assert "news: 2 row(s) would be orphaned" in text
    assert "contacts" not in text                      # contacts ARE moved by the legacy mover -> not a loss
    assert not any(b.startswith("ON DELETE rules") for b in plan["global_blockers"])     # no longer unverified
    t = {x["table"]: x for x in p["dependents"]["tables"]}
    assert t["lead_status"]["action"] == "merge_rows_manually" and t["lead_status"]["on_delete"] == "CASCADE"
    assert [x["table"] for x in p["dependents"]["tables"]] == [f["table"] for f in SAMPLE["foreign_keys"]]


def test_without_introspection_the_planner_still_reports_the_delete_rules_as_unverified():
    plan = _pair_plan(None)
    assert plan["fk_info_used"] is False
    assert any(b.startswith("ON DELETE rules") for b in plan["global_blockers"])
    p = plan["groups"][0]["pairs"][0]
    assert any("ON DELETE rule unverified" in b for b in p["execution"]["blockers"])
    assert p["execution"]["ready"] is False

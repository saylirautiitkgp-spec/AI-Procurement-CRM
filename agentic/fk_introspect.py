"""Read-only foreign-key introspection: the SQL to run and the parser for its output.

INTROSPECTION_SQL is a single SELECT (catalog reads only). It is NOT executed by this code base - paste it into
the Supabase SQL editor yourself, copy the single JSON cell it returns into a file, then:

    python -m agentic.fk_introspect path/to/result.json

STATUS: the SQL has not been run against a real Postgres by me (no database access for catalog queries); the
parser is tested against a hand-built sample of the documented output shape. Review the SQL before running.
"""

from __future__ import annotations

import json
import sys

ON_DELETE = {"a": "NO ACTION", "r": "RESTRICT", "c": "CASCADE", "n": "SET NULL", "d": "SET DEFAULT"}

INTROSPECTION_SQL = """\
select jsonb_build_object(
  'foreign_keys', (
    select coalesce(jsonb_agg(jsonb_build_object(
             'table', cl.relname, 'column', a.attname, 'constraint', c.conname,
             'ref_column', ra.attname, 'on_delete', c.confdeltype::text,
             'on_update', c.confupdtype::text, 'deferrable', c.condeferrable)), '[]'::jsonb)
    from pg_constraint c
    join pg_class cl on cl.oid = c.conrelid
    join pg_attribute a  on a.attrelid  = c.conrelid  and a.attnum  = c.conkey[1]
    join pg_attribute ra on ra.attrelid = c.confrelid and ra.attnum = c.confkey[1]
    where c.contype = 'f' and c.connamespace = 'public'::regnamespace
      and c.confrelid = 'public.companies'::regclass),
  'unique_constraints', (
    select coalesce(jsonb_agg(jsonb_build_object(
             'table', cl.relname, 'constraint', c.conname, 'type', c.contype::text,
             'columns', (select jsonb_agg(att.attname order by k.ord)
                         from unnest(c.conkey) with ordinality as k(attnum, ord)
                         join pg_attribute att on att.attrelid = c.conrelid and att.attnum = k.attnum))), '[]'::jsonb)
    from pg_constraint c join pg_class cl on cl.oid = c.conrelid
    where c.contype in ('u', 'p') and c.connamespace = 'public'::regnamespace),
  'rls', (
    select coalesce(jsonb_agg(jsonb_build_object(
             'table', relname, 'rls_enabled', relrowsecurity, 'rls_forced', relforcerowsecurity)), '[]'::jsonb)
    from pg_class where relnamespace = 'public'::regnamespace and relkind = 'r'),
  'policies', (
    select coalesce(jsonb_agg(jsonb_build_object(
             'table', tablename, 'policy', policyname, 'command', cmd, 'roles', roles)), '[]'::jsonb)
    from pg_policies where schemaname = 'public'),
  'triggers', (
    select coalesce(jsonb_agg(jsonb_build_object(
             'table', event_object_table, 'trigger', trigger_name, 'event', event_manipulation)), '[]'::jsonb)
    from information_schema.triggers where trigger_schema = 'public')
) as introspection;
"""


def analyze(raw: dict) -> dict:
    """Turn the query output into what the merge planner/executor need."""
    unique = {}
    for u in raw.get("unique_constraints", []):
        unique.setdefault(u["table"], []).append(u)
    tables: dict = {}
    for fk in raw.get("foreign_keys", []):
        t = fk["table"]
        uniq = unique.get(t, [])
        tables[t] = {
            "column": fk["column"], "on_delete": ON_DELETE.get(fk["on_delete"], fk["on_delete"]),
            "on_update": ON_DELETE.get(fk["on_update"], fk["on_update"]),
            "one_row_per_company": any(u["columns"] == [fk["column"]] for u in uniq),
            "composite_uniques": [u["columns"] for u in uniq if fk["column"] in u["columns"] and len(u["columns"]) > 1],
        }
    cascade = sorted(t for t, v in tables.items() if v["on_delete"] == "CASCADE")
    restrict = sorted(t for t, v in tables.items() if v["on_delete"] in ("RESTRICT", "NO ACTION"))
    setnull = sorted(t for t, v in tables.items() if v["on_delete"] in ("SET NULL", "SET DEFAULT"))
    rls = {r["table"]: r["rls_enabled"] for r in raw.get("rls", [])}
    return {"tables": tables, "cascade_delete_tables": cascade, "blocking_delete_tables": restrict,
            "set_null_tables": setnull,
            "rls_enabled_tables": sorted(t for t, on in rls.items() if on),
            "rls_disabled_tables": sorted(t for t, on in rls.items() if not on),
            "policies": raw.get("policies", []), "triggers": raw.get("triggers", []),
            "summary": ("Deleting a company will silently DELETE rows in: " + ", ".join(cascade)) if cascade else
                       "No ON DELETE CASCADE foreign keys reference companies"}


def main(path: str) -> None:  # pragma: no cover
    info = analyze(json.load(open(path, encoding="utf-8")))
    print(json.dumps(info, indent=2))


if __name__ == "__main__":  # pragma: no cover
    main(sys.argv[1])

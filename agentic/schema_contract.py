"""Which database columns this code base may write to - and how we KNOW they exist.

Nothing here queries a database. A column counts as CONFIRMED only when there is evidence, never by assumption:

* `live_read`   - it is in the SELECT list that `SupabaseReadRepo.find_candidates` ran successfully against the real
                  database (PostgREST rejects unknown columns, so a successful read proves they exist). Those are the
                  `companies` columns in `agentic.repo.COMPANY_COLUMNS` (live runs of 2026-10-11, event `db_lookup`).
* `introspection` - it appears in the output of `agentic.fk_introspect.INTROSPECTION_SQL` (key `columns`), which the
                  operator ran in the Supabase SQL editor and saved to a file (env `AGENTIC_SCHEMA_FILE`; the 2026-10-12
                  result is `agentic/schema_introspection.json`, a copy is the test fixture
                  `tests/fixtures/schema_introspection.json`).

Everything else is UNCONFIRMED. A proposal built from a reviewed change set that touches an unconfirmed column - or an
insert that omits a NOT NULL column - cannot be approved or applied. The introspection also records the primary and
unique keys; they are used to check our own assumptions (`agentic.commit.PRIMARY_KEYS`, one profile row per company).
"""

from __future__ import annotations

import hashlib
import json
from typing import Iterable

from agentic.repo import COMPANY_COLUMNS

LIVE_READ_CONFIRMED = frozenset(("companies", c.strip()) for c in COMPANY_COLUMNS.split(","))

# write_gate column kind -> database types that can hold it (the gate coerces values to the kind first)
KIND_TYPES = {"text": ("text",), "url": ("text",), "email": ("text",), "int": ("integer", "bigint"),
              "num": ("numeric",), "bool": ("boolean",), "ts": ("timestamp without time zone",),
              "list": ("ARRAY", "jsonb")}


class SchemaContract:
    def __init__(self, introspected: Iterable[tuple[str, str]] | None = None, *, details: dict | None = None,
                 primary_keys: dict | None = None, uniques: list | None = None, source: str | None = None):
        self.introspected = frozenset(introspected or ())
        self.confirmed = LIVE_READ_CONFIRMED | self.introspected
        self.details = details or {}                  # (table, column) -> {"data_type", "nullable", "default"?, "identity"?}
        self.primary_keys = primary_keys or {}        # table -> [columns]
        self.uniques = uniques or []                  # [{"table", "columns", "type"}]
        self.source = source

    # ------------------------------------------------------------------------------------------ loading
    @classmethod
    def from_introspection(cls, raw: dict, source: str | None = None) -> "SchemaContract":
        cols = raw.get("columns")
        if not isinstance(cols, list):
            raise ValueError("introspection output has no 'columns' list (re-run the current INTROSPECTION_SQL)")
        details = {(c["table"], c["column"]): {k: c.get(k) for k in ("data_type", "nullable", "default", "identity")}
                   for c in cols if "table" in c and "column" in c}
        pks = {}
        for u in raw.get("unique_constraints", []):
            if u.get("type") == "p":
                pks[u["table"]] = list(u["columns"])
        uniques = [{"table": u["table"], "columns": list(u["columns"]), "type": u.get("type")}
                   for u in raw.get("unique_constraints", [])]
        return cls(details.keys(), details=details, primary_keys=pks, uniques=uniques, source=source)

    @classmethod
    def from_file(cls, path: str) -> "SchemaContract":
        with open(path, encoding="utf-8") as fh:
            return cls.from_introspection(json.load(fh), source=path)

    # ------------------------------------------------------------------------------------------ queries
    @property
    def fingerprint(self) -> str:
        """Identifies the schema evidence a decision was made against (recorded on every change-set proposal)."""
        if not self.details:
            return "live-read-only"
        rows = sorted((t, c, str(d.get("data_type")), str(d.get("nullable"))) for (t, c), d in self.details.items())
        return "sha256:" + hashlib.sha256(json.dumps(rows).encode()).hexdigest()[:16]

    def evidence(self, table: str, column: str) -> str:
        if (table, column) in self.introspected:
            return "introspection"
        if (table, column) in LIVE_READ_CONFIRMED:
            return "live_read"
        return "unconfirmed"

    def unconfirmed(self, columns: Iterable[str]) -> list[str]:
        """'table.column' names not confirmed by any evidence."""
        out = []
        for tc in columns:
            table, _, col = tc.partition(".")
            if (table, col) not in self.confirmed:
                out.append(tc)
        return sorted(set(out))

    def required_insert_columns(self, table: str) -> list[str]:
        """NOT NULL columns an INSERT must supply: not the primary key, and without a default/identity when the
        introspection reports them. Empty when the table is unknown (nothing can be claimed)."""
        pk = set(self.primary_keys.get(table, []))
        out = []
        for (t, c), d in self.details.items():
            if t == table and d.get("nullable") == "NO" and c not in pk and not d.get("default") and not d.get("identity"):
                out.append(c)
        return sorted(out)

    def missing_required(self, operations: list[dict], provided_later: Iterable[str] = ()) -> list[str]:
        """'table.column' NOT NULL columns that an insert operation does not supply (placeholders like $company_id
        count as supplied; `provided_later` = columns the approver must provide, e.g. companies.sector_id)."""
        later = set(provided_later)
        out = []
        for op in operations:
            if op.get("op") != "insert":
                continue
            for col in self.required_insert_columns(op["table"]):
                if col not in op.get("fields", {}) and f"{op['table']}.{col}" not in later:
                    out.append(f"{op['table']}.{col}")
        return sorted(set(out))

    def type_mismatches(self, gate_columns: dict) -> list[str]:
        """Write-gate column kinds that the database type cannot hold (empty = consistent)."""
        out = []
        for table, cols in gate_columns.items():
            for col, kind in cols.items():
                d = self.details.get((table, col))
                if d and d.get("data_type") not in KIND_TYPES[kind]:
                    out.append(f"{table}.{col}: gate '{kind}' vs database '{d.get('data_type')}'")
        return out

    def is_unique(self, table: str, column: str) -> bool:
        return any(u["table"] == table and u["columns"] == [column] for u in self.uniques)


def operation_columns(operations: list[dict]) -> list[str]:
    """Every 'table.column' an operation writes or uses as its key."""
    cols = set()
    for op in operations:
        for c in list(op.get("fields", {})) + list(op.get("key", {})):
            cols.add(f"{op['table']}.{c}")
    return sorted(cols)

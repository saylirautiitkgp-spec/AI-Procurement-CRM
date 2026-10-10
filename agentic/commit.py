"""Write-side repositories. Used ONLY by ProposalService.apply, after human
approval and write_gate validation. Agents and tools never see these classes.

Limitation (documented): the Supabase REST API is not transactional, so a
multi-operation proposal is applied in order and a failure part-way is recorded
as 'failed' with the operations already executed. A server-side RPC would be
needed for true atomicity (proposed as a later additive migration)."""

from __future__ import annotations

import itertools
from typing import Any, Protocol

from agentic.write_gate import COLUMNS, OPERATIONS

PRIMARY_KEYS = {"companies": "company_id", "company_profile": "profile_id", "contacts": "contact_id",
                "ai_insights": "insight_id", "suppliers": "supplier_id",
                "company_suppliers": "company_supplier_id"}


class CommitRepository(Protocol):
    def get_row(self, table: str, key: dict) -> dict | None: ...
    def find_one(self, table: str, **eq: Any) -> dict | None: ...
    def insert(self, table: str, row: dict) -> dict: ...
    def update(self, table: str, key: dict, fields: dict) -> None: ...
    def sector_ids(self) -> set[int]: ...


class InMemoryCommitRepo:
    """Test double that records every operation."""

    def __init__(self, tables: dict | None = None, sectors: set[int] | None = None):
        self.tables: dict[str, list[dict]] = {t: [dict(r) for r in rows] for t, rows in (tables or {}).items()}
        self.sectors = sectors if sectors is not None else {1, 2, 3, 4, 5, 6, 7}
        self.ops: list[tuple] = []
        self._ids = itertools.count(1000)
        self.fail_on: tuple | None = None          # (table, op) -> raise, for failure tests

    def _match(self, row, eq):
        return all(row.get(k) == v for k, v in eq.items())

    def get_row(self, table, key):
        return next((dict(r) for r in self.tables.get(table, []) if self._match(r, key)), None)

    def find_one(self, table, **eq):
        return self.get_row(table, eq)

    def insert(self, table, row):
        self._guard(table, "insert")
        new = {**row, PRIMARY_KEYS[table]: next(self._ids)}
        self.tables.setdefault(table, []).append(new)
        self.ops.append(("insert", table, dict(row)))
        return new

    def update(self, table, key, fields):
        self._guard(table, "update")
        for r in self.tables.get(table, []):
            if self._match(r, key):
                r.update(fields)
        self.ops.append(("update", table, dict(key), dict(fields)))

    def sector_ids(self):
        return set(self.sectors)

    def _guard(self, table, op):
        assert op in OPERATIONS.get(table, set()), f"{op} on {table} must be blocked by the gate"
        if self.fail_on == (table, op):
            raise RuntimeError("simulated database failure")


class SupabaseCommitRepo:
    """Allowlisted insert/update on top of a supabase client. Constructed only
    when an operator wires live writes; never during tests or research."""

    def __init__(self, client: Any):
        self._c = client

    @staticmethod
    def _check(table: str, op: str) -> None:
        if table not in OPERATIONS or op not in OPERATIONS[table]:
            raise PermissionError(f"{op} on {table} is not allowed")

    def get_row(self, table, key):
        q = self._c.table(table).select("*")
        for k, v in key.items():
            q = q.eq(k, v)
        rows = q.limit(1).execute().data or []
        return rows[0] if rows else None

    def find_one(self, table, **eq):
        return self.get_row(table, eq)

    def insert(self, table, row):
        self._check(table, "insert")
        assert set(row) <= set(COLUMNS[table]), "unexpected column"
        data = self._c.table(table).insert(row).execute().data or []
        return data[0] if data else dict(row)

    def update(self, table, key, fields):
        self._check(table, "update")
        assert set(fields) <= set(COLUMNS[table]), "unexpected column"
        q = self._c.table(table).update(fields)
        for k, v in key.items():
            q = q.eq(k, v)
        q.execute()

    def sector_ids(self):
        return {r["sector_id"] for r in (self._c.table("sectors").select("sector_id").execute().data or [])}

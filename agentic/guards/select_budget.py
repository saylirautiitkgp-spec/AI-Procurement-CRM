"""Per-run cap on Supabase SELECTs, enforced at the lowest shared query-execution boundary.

Every read the runtime makes goes through `GuardedReadClient`, a proxy around the supabase client: a query is only sent
when its `.execute()` is called, and the proxy reserves one unit of the run's `SelectBudget` BEFORE it calls the real
`.execute()`. The ninth SELECT of an eight-SELECT budget therefore never leaves the process. This counts everything:
the lookup, lazy section reads, helper methods and repeated queries - the repository classes cannot bypass it because
they only ever hold the proxy.

* One budget per run, shared by all agents and parallel branches (it lives in a ContextVar set by `AgenticRunner.start`;
  parallel branches copy the context, so they use the SAME object and the count is never reset).
* The proxy is read-only: insert/update/delete/upsert/rpc raise PermissionError (writes use a different, separately
  enabled client).
* `SelectBudgetExceeded` derives from BaseException on purpose: the CRM code has broad `except Exception` handlers
  ("a DB outage must not stop research"); an exhausted budget must NEVER be swallowed and read as "company not found".
  The graph and the runner catch it explicitly and end the run as failed/partial with a recorded reason.
"""

from __future__ import annotations

import contextvars
import threading

_WRITE_METHODS = {"insert", "update", "delete", "upsert", "rpc"}


class SelectBudgetExceeded(BaseException):
    def __init__(self, limit: int, label: str):
        super().__init__(f"Supabase SELECT budget of {limit} reached; refused before sending ({label})")
        self.limit, self.label = limit, label


class SelectBudget:
    def __init__(self, limit: int):
        self.limit = int(limit)
        self.used = 0
        self.log: list[str] = []
        self._lock = threading.Lock()

    def reserve(self, label: str) -> None:
        with self._lock:
            if self.used >= self.limit:
                raise SelectBudgetExceeded(self.limit, label)
            self.used += 1
            self.log.append(label)


_CURRENT: contextvars.ContextVar[SelectBudget | None] = contextvars.ContextVar("select_budget", default=None)


def activate(budget: SelectBudget | None):
    return _CURRENT.set(budget)


def deactivate(token) -> None:
    _CURRENT.reset(token)


def current() -> SelectBudget | None:
    return _CURRENT.get()


def used() -> int | None:
    b = _CURRENT.get()
    return None if b is None else b.used


class _Query:
    """Proxy for a postgrest query builder: chaining returns proxies, `.execute()` is the counted boundary."""

    def __init__(self, inner, table: str):
        self._inner, self._table = inner, table

    def execute(self, *a, **kw):
        budget = _CURRENT.get()
        if budget is not None:
            budget.reserve(self._table)            # raises BEFORE anything is sent
        return self._inner.execute(*a, **kw)

    def __getattr__(self, name):
        if name in _WRITE_METHODS:
            raise PermissionError(f"read-only Supabase client: .{name}() is not allowed")
        attr = getattr(self._inner, name)
        if not callable(attr):
            return attr

        def chained(*a, **kw):
            out = attr(*a, **kw)
            return _Query(out, self._table) if hasattr(out, "execute") else out
        return chained


class GuardedReadClient:
    def __init__(self, inner):
        self._inner = inner

    def table(self, name: str) -> _Query:
        return _Query(self._inner.table(name), name)

    def __getattr__(self, name):
        if name in _WRITE_METHODS:
            raise PermissionError(f"read-only Supabase client: .{name}() is not allowed")
        raise AttributeError(f"GuardedReadClient exposes table() only, not {name!r}")


# ------------------------------------------------------------------------------------------------ write-phase client
_NEVER = {"delete", "upsert", "rpc"}


class _CountedQuery:
    """Query proxy of the WRITE phase: a SELECT is counted (and refused over the cap) BEFORE it is sent; insert/update pass
    through (the commit repository's table/operation allow-list decides which); delete/upsert/rpc do not exist."""

    def __init__(self, inner, table: str, budget: SelectBudget, kind: str | None = None):
        self._inner, self._table, self._budget, self._kind = inner, table, budget, kind

    def execute(self, *a, **kw):
        if self._kind == "select":
            self._budget.reserve(self._table)          # raises BEFORE anything is sent
        return self._inner.execute(*a, **kw)

    def __getattr__(self, name):
        if name in _NEVER:
            raise PermissionError(f"Supabase client of the write phase: .{name}() is not allowed")
        attr = getattr(self._inner, name)
        if not callable(attr):
            return attr
        kind = "select" if name == "select" else ("write" if name in ("insert", "update") else self._kind)

        def chained(*a, **kw):
            out = attr(*a, **kw)
            return _CountedQuery(out, self._table, self._budget, kind) if hasattr(out, "execute") else out
        return chained


class CountedClient:
    """Supabase client wrapper for the narrow write phase: own SELECT budget, no delete/upsert/rpc."""

    def __init__(self, inner, budget: SelectBudget):
        self._inner, self.budget = inner, budget

    def table(self, name: str) -> _CountedQuery:
        return _CountedQuery(self._inner.table(name), name, self.budget)

    def __getattr__(self, name):
        raise AttributeError(f"CountedClient exposes table() only, not {name!r}")

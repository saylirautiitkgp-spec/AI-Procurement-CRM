"""Local SQLite persistence (zero cost): credit ledger, response cache, run
records, trace events, proposals/approvals and the audit log.

Everything the tracker API serves is read from here, so a run is observable
(and survives a restart) independently of the LangGraph checkpoint file.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import threading
from datetime import datetime, timedelta, timezone
from typing import Any, Callable

from agentic.evidence import now_iso


def _month() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m")


class Store:
    def __init__(self, path: str = ":memory:", month_fn: Callable[[], str] = _month):
        if path != ":memory:":
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.RLock()
        self._month = month_fn
        self._init()

    def _init(self) -> None:
        with self._lock, self._conn:
            self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS credit_ledger(month TEXT PRIMARY KEY, used INTEGER NOT NULL DEFAULT 0);
            CREATE TABLE IF NOT EXISTS llm_usage(ts REAL, key TEXT, tokens INTEGER);
            CREATE INDEX IF NOT EXISTS llm_usage_key_ts ON llm_usage(key, ts);
            CREATE TABLE IF NOT EXISTS cache(key TEXT PRIMARY KEY, value TEXT NOT NULL, created_at TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS runs(run_id TEXT PRIMARY KEY, company_name TEXT, objective TEXT,
                status TEXT, created_at TEXT, updated_at TEXT, tracker TEXT, report TEXT);
            CREATE TABLE IF NOT EXISTS events(run_id TEXT, seq INTEGER, ts TEXT, body TEXT,
                PRIMARY KEY(run_id, seq));
            CREATE TABLE IF NOT EXISTS proposals(proposal_id TEXT PRIMARY KEY, run_id TEXT, status TEXT,
                change TEXT, validation TEXT, idempotency_key TEXT UNIQUE, created_at TEXT,
                decided_at TEXT, decided_by TEXT, reason TEXT, applied_at TEXT);
            CREATE TABLE IF NOT EXISTS audit_log(id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT,
                run_id TEXT, proposal_id TEXT, action TEXT, actor TEXT, details TEXT);
            """)

    # ---- credit ledger ---------------------------------------------------
    def credits_used(self) -> int:
        with self._lock:
            row = self._conn.execute("SELECT used FROM credit_ledger WHERE month=?",
                                     (self._month(),)).fetchone()
            return row["used"] if row else 0

    def charge_credits(self, n: int) -> int:
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT INTO credit_ledger(month, used) VALUES(?, ?) "
                "ON CONFLICT(month) DO UPDATE SET used = used + excluded.used",
                (self._month(), n))
        return self.credits_used()

    # ---- LLM usage (rolling-window free-tier budgets) -----------------------
    def usage_add(self, key: str, tokens: int, ts: float) -> None:
        with self._lock, self._conn:
            self._conn.execute("INSERT INTO llm_usage VALUES(?,?,?)", (ts, key, int(tokens)))

    def usage_reserve(self, key: str, now: float, window_s: float, max_requests: int, est_tokens: int,
                      max_tokens: int | None = None) -> tuple[bool, int, int, int | None]:
        """Atomically (one SQLite write transaction, safe across threads AND processes) check the rolling window and,
        if there is room, record ONE request before it is sent. Returns (ok, requests_used, tokens_used, row_id) where
        the counts are those BEFORE this reservation. A reservation is never released: a request that fails after it
        was sent may still have consumed provider quota."""
        with self._lock:
            self._conn.commit()                              # no open implicit transaction before BEGIN IMMEDIATE
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                row = self._conn.execute("SELECT COUNT(*), COALESCE(SUM(tokens), 0) FROM llm_usage WHERE key=? AND ts>?",
                                         (key, now - window_s)).fetchone()
                reqs, toks = int(row[0]), int(row[1])
                if reqs + 1 > max_requests or (max_tokens is not None and toks + est_tokens > max_tokens):
                    self._conn.execute("ROLLBACK")
                    return False, reqs, toks, None
                cur = self._conn.execute("INSERT INTO llm_usage VALUES(?,?,?)", (now, key, int(est_tokens)))
                self._conn.execute("COMMIT")
                return True, reqs, toks, cur.lastrowid
            except BaseException:
                if self._conn.in_transaction:
                    self._conn.execute("ROLLBACK")
                raise

    def usage_finalize(self, row_id: int, tokens: int) -> None:
        """Replace a reservation's estimated tokens with the measured ones."""
        with self._lock, self._conn:
            self._conn.execute("UPDATE llm_usage SET tokens=? WHERE rowid=?", (int(tokens), int(row_id)))

    def usage_since(self, key: str, seconds: float, now: float) -> tuple[int, int]:
        with self._lock:
            row = self._conn.execute("SELECT COUNT(*), COALESCE(SUM(tokens), 0) FROM llm_usage WHERE key=? AND ts>?",
                                     (key, now - seconds)).fetchone()
        return int(row[0]), int(row[1])

    # ---- cache -------------------------------------------------------------
    @staticmethod
    def cache_key(tool: str, args: dict) -> str:
        blob = json.dumps({"t": tool, "a": args}, sort_keys=True, default=str)
        return hashlib.sha256(blob.encode()).hexdigest()

    def cache_get(self, key: str, ttl_days: float = 30) -> Any | None:
        with self._lock:
            row = self._conn.execute("SELECT value, created_at FROM cache WHERE key=?", (key,)).fetchone()
        if not row:
            return None
        created = datetime.fromisoformat(row["created_at"])
        if datetime.now(timezone.utc) - created > timedelta(days=ttl_days):
            return None
        return json.loads(row["value"])

    def cache_put(self, key: str, value: Any) -> None:
        with self._lock, self._conn:
            self._conn.execute("INSERT OR REPLACE INTO cache VALUES(?,?,?)",
                               (key, json.dumps(value, default=str), now_iso()))

    # ---- runs / events -----------------------------------------------------
    def upsert_run(self, run_id: str, **fields: Any) -> None:
        with self._lock, self._conn:
            row = self._conn.execute("SELECT 1 FROM runs WHERE run_id=?", (run_id,)).fetchone()
            if not row:
                self._conn.execute(
                    "INSERT INTO runs(run_id, company_name, objective, status, created_at, updated_at) "
                    "VALUES(?,?,?,?,?,?)",
                    (run_id, fields.get("company_name"), fields.get("objective"),
                     fields.get("status", "queued"), now_iso(), now_iso()))
            for col in ("status", "tracker", "report", "company_name", "objective"):
                if col in fields and fields[col] is not None:
                    val = fields[col]
                    if col in ("tracker", "report") and not isinstance(val, str):
                        val = json.dumps(val, default=str)
                    self._conn.execute(f"UPDATE runs SET {col}=?, updated_at=? WHERE run_id=?",
                                       (val, now_iso(), run_id))

    def get_run(self, run_id: str) -> dict | None:
        with self._lock:
            row = self._conn.execute("SELECT * FROM runs WHERE run_id=?", (run_id,)).fetchone()
        if not row:
            return None
        out = dict(row)
        for col in ("tracker", "report"):
            out[col] = json.loads(out[col]) if out[col] else None
        return out

    def list_runs(self, limit: int = 50) -> list[dict]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT run_id, company_name, status, created_at, updated_at FROM runs "
                "ORDER BY created_at DESC LIMIT ?", (limit,)).fetchall()
        return [dict(r) for r in rows]

    def append_events(self, run_id: str, events: list[dict]) -> None:
        with self._lock, self._conn:
            for ev in events:
                self._conn.execute("INSERT OR IGNORE INTO events VALUES(?,?,?,?)",
                                   (run_id, ev["seq"], ev["ts"], json.dumps(ev, default=str)))

    def get_events(self, run_id: str, after: int = 0, limit: int = 500) -> list[dict]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT body FROM events WHERE run_id=? AND seq>? ORDER BY seq LIMIT ?",
                (run_id, after, limit)).fetchall()
        return [json.loads(r["body"]) for r in rows]

    # ---- proposals / audit -------------------------------------------------
    def add_proposal(self, proposal_id: str, run_id: str, change: dict, validation: dict,
                     idempotency_key: str, status: str = "pending") -> bool:
        with self._lock, self._conn:
            try:
                self._conn.execute(
                    "INSERT INTO proposals(proposal_id, run_id, status, change, validation, "
                    "idempotency_key, created_at) VALUES(?,?,?,?,?,?,?)",
                    (proposal_id, run_id, status, json.dumps(change, default=str),
                     json.dumps(validation, default=str), idempotency_key, now_iso()))
                return True
            except sqlite3.IntegrityError:
                return False

    def get_proposal(self, proposal_id: str) -> dict | None:
        with self._lock:
            row = self._conn.execute("SELECT * FROM proposals WHERE proposal_id=?",
                                     (proposal_id,)).fetchone()
        return self._proposal_row(row)

    def update_proposal_change(self, proposal_id: str, change: dict) -> None:
        with self._lock, self._conn:
            self._conn.execute("UPDATE proposals SET change=? WHERE proposal_id=?",
                               (json.dumps(change, default=str), proposal_id))

    def list_proposals(self, run_id: str) -> list[dict]:
        with self._lock:
            rows = self._conn.execute("SELECT * FROM proposals WHERE run_id=? ORDER BY created_at",
                                      (run_id,)).fetchall()
        return [self._proposal_row(r) for r in rows]

    @staticmethod
    def _proposal_row(row) -> dict | None:
        if not row:
            return None
        out = dict(row)
        out["change"] = json.loads(out["change"])
        out["validation"] = json.loads(out["validation"]) if out["validation"] else None
        return out

    def set_proposal_status(self, proposal_id: str, status: str, *, decided_by: str | None = None,
                            reason: str | None = None, applied: bool = False) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                "UPDATE proposals SET status=?, decided_at=COALESCE(decided_at, ?), "
                "decided_by=COALESCE(?, decided_by), reason=COALESCE(?, reason), "
                "applied_at=CASE WHEN ? THEN ? ELSE applied_at END WHERE proposal_id=?",
                (status, now_iso(), decided_by, reason, 1 if applied else 0, now_iso(), proposal_id))

    def audit(self, action: str, *, run_id: str | None = None, proposal_id: str | None = None,
              actor: str = "system", details: dict | None = None) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT INTO audit_log(ts, run_id, proposal_id, action, actor, details) VALUES(?,?,?,?,?,?)",
                (now_iso(), run_id, proposal_id, action, actor, json.dumps(details or {}, default=str)))

    def audit_entries(self, run_id: str | None = None) -> list[dict]:
        with self._lock:
            if run_id:
                rows = self._conn.execute("SELECT * FROM audit_log WHERE run_id=? ORDER BY id", (run_id,)).fetchall()
            else:
                rows = self._conn.execute("SELECT * FROM audit_log ORDER BY id").fetchall()
        return [dict(r) | {"details": json.loads(r["details"])} for r in rows]

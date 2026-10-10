"""Read-only data sources for the merge planner.

* InMemoryMergeSource - fixtures / tests.
* SupabaseMergeSource - SELECT only (the class has no write methods and only calls `.select`).

Privacy: contact NAMES are read only to build short salted fingerprints (to detect "the same person in both
records") and are never stored or printed. Fingerprints are comparable only within one source instance.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import secrets
import sys
from collections import defaultdict

from agentic import names
from agentic.merge_plan import COMPANY_FK_TABLES

COMPANY_COLS = "company_id,company_name,sector_id,country,revenue,number_of_plants,website,linkedin_url," \
               "company_type,enrichment_status,sector,sub_sector,priority"


class InMemoryMergeSource:
    def __init__(self, data: dict):
        self._companies = data["companies"]
        self._views = {int(k): v for k, v in data["views"].items()}

    def companies(self) -> list[dict]:
        return list(self._companies)

    def view(self, company_id: int) -> dict:
        return self._views[company_id]


class SupabaseMergeSource:
    """Pre-loads what the planner needs with paged SELECTs, then serves views from memory."""

    def __init__(self, client):
        self._c = client
        self._salt = secrets.token_bytes(16)
        self._all: list[dict] | None = None
        self._sectors: dict | None = None
        self._cache: dict[int, dict] = {}
        self._bulk: dict | None = None

    @classmethod
    def from_env(cls):  # pragma: no cover - needs credentials
        from dotenv import load_dotenv
        from supabase import create_client
        load_dotenv()
        return cls(create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_KEY"]))

    def _paged(self, table: str, cols: str) -> list[dict]:
        out, off = [], 0
        while True:
            batch = self._c.table(table).select(cols).range(off, off + 999).execute().data or []
            out += batch
            if len(batch) < 1000:
                return out
            off += 1000

    def companies(self) -> list[dict]:
        if self._all is None:
            self._all = self._paged("companies", COMPANY_COLS)
            self._sectors = {r["sector_id"]: r["sector"] for r in self._paged("sectors", "sector_id,sector")}
        return self._all

    def _fingerprint(self, name: str) -> str:
        return hmac.new(self._salt, names.normalize_company_name(name).encode(), hashlib.sha256).hexdigest()[:10]

    def _load_bulk(self) -> dict:
        if self._bulk is None:
            b: dict = {"dep": defaultdict(lambda: defaultdict(int)), "products": defaultdict(list),
                       "components": defaultdict(list), "fp": defaultdict(set), "profile": {}}
            for t in COMPANY_FK_TABLES:
                cols = {"products": "company_id,product_name", "components": "company_id,component_name",
                        "contacts": "company_id,contact_name", "company_profile": "*"}.get(t, "company_id")
                for r in self._paged(t, cols):
                    b["dep"][r["company_id"]][t] += 1
                    if t == "products":
                        b["products"][r["company_id"]].append(r["product_name"])
                    elif t == "components":
                        b["components"][r["company_id"]].append(r["component_name"])
                    elif t == "contacts" and r.get("contact_name"):
                        b["fp"][r["company_id"]].add(self._fingerprint(r["contact_name"]))
                    elif t == "company_profile":
                        b["profile"][r["company_id"]] = {k: v for k, v in r.items() if v not in (None, "", [], {})}
            self._bulk = b
        return self._bulk

    def view(self, company_id: int) -> dict:
        if company_id not in self._cache:
            self.companies()
            row = next(r for r in self._all if r["company_id"] == company_id)
            b = self._load_bulk()
            deps = dict(b["dep"][company_id])
            self._cache[company_id] = {
                **row, "sector_name": self._sectors.get(row["sector_id"]),
                "profile": b["profile"].get(company_id), "insights_count": deps.get("ai_insights", 0),
                "products": b["products"][company_id], "components": b["components"][company_id],
                "contact_fingerprints": sorted(b["fp"][company_id]),
                "dependents": {t: deps.get(t, 0) for t in COMPANY_FK_TABLES}}
        return self._cache[company_id]


def build_fixture(source, planner_pairs_ids: list[int]) -> dict:
    rows = source.companies()
    return {"note": "Structural snapshot of the CRM for offline merge-plan tests. Contact names are NOT stored; "
                    "fingerprints are salted hashes comparable only within this file. Read-only export.",
            "companies": [{k: r.get(k) for k in ("company_id", "company_name", "sector_id", "country", "website")}
                          for r in rows],
            "views": {str(i): source.view(i) for i in planner_pairs_ids}}


def main(path: str) -> None:  # pragma: no cover - live READ-ONLY export
    from agentic.merge_plan import MergePlanner
    src = SupabaseMergeSource.from_env()

    class _Shim:                         # planner needs only candidate ids here
        def companies(self): return src.companies()
        def view(self, i): return src.view(i)
    from agentic.merge_plan import load_legacy
    planner = MergePlanner(_Shim(), legacy=load_legacy())
    ids = sorted({i for a, b, _, _ in planner.candidate_pairs() for i in (a, b)})
    data = build_fixture(src, ids)
    json.dump(data, open(path, "w", encoding="utf-8"), indent=1, default=str)
    print(f"wrote {path}: {len(data['companies'])} companies, {len(ids)} candidate views (read-only export)")


if __name__ == "__main__":  # pragma: no cover
    main(sys.argv[1] if len(sys.argv) > 1 else "tests/fixtures/duplicate_groups.json")

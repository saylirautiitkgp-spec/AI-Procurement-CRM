"""Reproducible selection of the initial 20-company development/evaluation set.

Method (documented so it can be re-run and audited):
  1. Read companies (read-only) from Supabase.
  2. rank(company) = sha256(f"{SEED}:{company_id}") - a fixed pseudo-random order that does not
     depend on names, data quality or enrichment status (no cherry-picking).
  3. Stratify by sector_id: within each sector sort by rank, then take companies round-robin across
     sectors (ordered by sector_id) until 20 are chosen.
  4. Offline "web variants" (standard, ambiguous, conflicting, ...) are assigned by position in the
     final list (index % len(VARIANTS)). This is a TEST DESIGN choice to cover edge cases; it says
     nothing about the real company.
The 20 companies are a development set only: nothing in the runtime depends on them.

Usage (read-only, needs SUPABASE_URL/SUPABASE_KEY in the environment or .env):
    python -m agentic.eval.select_first20 tests/fixtures/first20.json
"""

from __future__ import annotations

import hashlib
import json
import sys
from collections import defaultdict

from agentic import names
from agentic.eval.synthetic import VARIANTS

SEED = "agentic-eval-v1"
N = 20


def rank(company_id: int, seed: str = SEED) -> str:
    return hashlib.sha256(f"{seed}:{company_id}".encode()).hexdigest()


def select(rows: list[dict], n: int = N, seed: str = SEED) -> list[dict]:
    by_sector: dict = defaultdict(list)
    for r in rows:
        by_sector[r["sector_id"]].append(r)
    for sector in by_sector:
        by_sector[sector].sort(key=lambda r: rank(r["company_id"], seed))
    chosen, i = [], 0
    sectors = sorted(by_sector)
    while len(chosen) < n and any(by_sector[s] for s in sectors):
        sector = sectors[i % len(sectors)]
        if by_sector[sector]:
            chosen.append(by_sector[sector].pop(0))
        i += 1
    return chosen


def duplicate_groups(rows: list[dict]) -> dict:
    groups: dict = defaultdict(list)
    for r in rows:
        groups[names.normalize_company_name(r["company_name"])].append(r["company_id"])
    return {k: v for k, v in groups.items() if len(v) > 1}


def build_fixture(rows, sectors, products, components, n=N, seed=SEED) -> dict:
    dups = duplicate_groups(rows)
    dup_ids = {cid for ids in dups.values() for cid in ids}
    out = []
    for idx, r in enumerate(select(rows, n, seed)):
        out.append({
            "company_id": r["company_id"], "name": r["company_name"],
            "sector": sectors.get(r["sector_id"]), "country": r.get("country"),
            "products": products.get(r["company_id"], [])[:6],
            "components": components.get(r["company_id"], [])[:6],
            "in_duplicate_name_group": r["company_id"] in dup_ids,
            "web_variant": VARIANTS[idx % len(VARIANTS)],
        })
    return {"method": __doc__.split("Usage")[0].strip(), "seed": seed, "n": n,
            "source_rows": len(rows), "companies": out}


def _fetch_all(client, table, cols):
    out, offset = [], 0
    while True:
        batch = client.table(table).select(cols).range(offset, offset + 999).execute().data or []
        out += batch
        if len(batch) < 1000:
            return out
        offset += 1000


def main(path: str) -> None:  # pragma: no cover - live read-only helper
    import os
    from dotenv import load_dotenv
    from supabase import create_client
    load_dotenv()
    client = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_KEY"])
    rows = _fetch_all(client, "companies", "company_id,company_name,sector_id,country")
    sectors = {r["sector_id"]: r["sector"] for r in _fetch_all(client, "sectors", "sector_id,sector")}
    chosen_ids = {r["company_id"] for r in select(rows)}
    products, components = defaultdict(list), defaultdict(list)
    for r in _fetch_all(client, "products", "company_id,product_name"):
        if r["company_id"] in chosen_ids:
            products[r["company_id"]].append(r["product_name"])
    for r in _fetch_all(client, "components", "company_id,component_name"):
        if r["company_id"] in chosen_ids:
            components[r["company_id"]].append(r["component_name"])
    fixture = build_fixture(rows, sectors, products, components)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(fixture, fh, indent=2, ensure_ascii=False)
    print(f"wrote {path}: {len(fixture['companies'])} companies from {len(rows)} rows (seed={SEED})")


if __name__ == "__main__":  # pragma: no cover
    main(sys.argv[1] if len(sys.argv) > 1 else "tests/fixtures/first20.json")

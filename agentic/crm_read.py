"""Read-only CRM data for the web platform: sectors, companies, and one company's full picture.

Two sources behind one interface:
* `SupabaseCrm` - the real database, through the read-only proxy (`GuardedReadClient`: insert/update/delete/upsert/rpc do not
  exist on it). The overview is cached for a short time so browsing never hammers the database.
* `DemoCrm` - the 20-company development set bundled with the repo (agentic/demo/first20.json), used when no database is
  configured or when `--demo-data` is chosen. Same shapes, no network.

Nothing here writes. Contact e-mails and phone numbers are never returned (the CRM has none with provenance); a contact is
shown with its designation and whether it has a source.
"""

from __future__ import annotations

import json
import os
import threading
import time
from typing import Any, Protocol

from agentic import fields as fl
from agentic import write_gate as wg

PROFILE_FIELDS = [c for c in wg.COLUMNS["company_profile"] if c not in ("company_id", "confidence_score", "website_domain")]
CONTACT_COLUMNS = "contact_id,contact_name,designation,department,source_url,confidence_score,decision_maker"
COMPANY_COLUMNS = ("company_id,company_name,country,sector_id,sector,website,linkedin_url,company_type,enrichment_status,"
                   "priority,number_of_plants,revenue")


class CrmSource(Protocol):
    kind: str

    def overview(self) -> dict: ...
    def company(self, company_id: int) -> dict | None: ...
    def invalidate(self) -> None: ...


def _filled(v: Any) -> bool:
    return v not in (None, "", [], {})


def _company_card(row: dict, sector_name: str | None, profile: dict | None, insights: int) -> dict:
    filled = sum(1 for k in PROFILE_FIELDS if profile and _filled(profile.get(k)))
    return {"id": row["company_id"], "name": row.get("company_name"), "country": row.get("country"),
            "sector_id": row.get("sector_id"), "sector": sector_name, "website": row.get("website"),
            "linkedin_url": row.get("linkedin_url"), "type": row.get("company_type"),
            "status": row.get("enrichment_status"), "priority": row.get("priority"),
            "plants": row.get("number_of_plants"), "has_profile": profile is not None,
            "profile_fields": filled, "profile_fields_total": len(PROFILE_FIELDS), "insights": insights}


def _overview(sectors: list[dict], cards: list[dict]) -> dict:
    counts: dict = {}
    for c in cards:
        counts.setdefault(c["sector_id"], {"companies": 0, "enriched": 0})
        counts[c["sector_id"]]["companies"] += 1
        counts[c["sector_id"]]["enriched"] += 1 if (c["has_profile"] or c["insights"]) else 0
    out = [{"id": s["sector_id"], "name": s["sector"], **counts.get(s["sector_id"], {"companies": 0, "enriched": 0})}
           for s in sectors]
    out.sort(key=lambda s: (-s["companies"], s["name"] or ""))
    return {"generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "sectors": out, "companies": cards,
            "totals": {"companies": len(cards), "sectors": len(out),
                       "countries": len({c["country"] for c in cards if c["country"]}),
                       "with_profile": sum(1 for c in cards if c["has_profile"]),
                       "with_insights": sum(1 for c in cards if c["insights"])}}


def _coverage(company: dict, profile: dict | None, insights: list[dict]) -> list[dict]:
    """For every catalog field: what the CRM holds today (unverified legacy data unless it has a provenance row)."""
    existing = {"fields": company, "profile": profile or {}}
    sourced = {str(i.get("insight_type", "")).removeprefix("Verified ").removeprefix("Procurement: ") for i in insights
               if str(i.get("generated_by", "")).startswith("agentic-")}
    out = []
    for key, dest in fl.canonical_destinations().items():
        if dest["layer"] not in ("core", "extended", "insight"):
            continue
        spec = fl.BY_KEY[key]
        value = fl.crm_value(key, existing) if dest["layer"] in ("core", "extended") else None
        out.append({"key": key, "label": spec.label, "agent": spec.agent, "layer": dest["layer"], "target": dest["target"],
                    "filled": _filled(value) or key in sourced, "value": value if _filled(value) else None,
                    "sourced": key in sourced})
    return out


def _detail(company: dict, sector_name: str | None, profile: dict | None, insights: list[dict], products: list[str],
            components: list[str], contacts: list[dict], lead: dict | None) -> dict:
    return {"company": {**company, "sector": sector_name or company.get("sector")},
            "profile": {k: v for k, v in (profile or {}).items() if _filled(v) and k not in ("extracted_text",)} or None,
            "coverage": _coverage(company, profile, insights),
            "insights": [{k: i.get(k) for k in ("insight_id", "insight_type", "insight_title", "insight", "source_url",
                                                "confidence_score", "generated_by", "created_at")} for i in insights],
            "products": products[:60], "components": components[:60],
            "contacts": {"count": len(contacts), "verified": sum(1 for c in contacts if c.get("source_url")),
                         "items": [{"id": c.get("contact_id"), "name": c.get("contact_name"), "designation": c.get("designation"),
                                    "department": c.get("department"), "has_source": bool(c.get("source_url"))}
                                   for c in contacts[:25]]},
            "lead": lead}


class SupabaseCrm:
    kind = "supabase"

    def __init__(self, client: Any, ttl_s: float = 60.0):
        from agentic.guards.select_budget import GuardedReadClient
        self._c = GuardedReadClient(client)
        self._ttl, self._cache, self._lock = ttl_s, None, threading.Lock()

    def _rows(self, table: str, cols: str, **eq) -> list[dict]:
        q = self._c.table(table).select(cols)
        for k, v in eq.items():
            q = q.eq(k, v)
        return q.limit(1000).execute().data or []

    def _paged(self, table: str, cols: str) -> list[dict]:
        out, start = [], 0
        while True:
            rows = self._c.table(table).select(cols).order("company_id").range(start, start + 999).execute().data or []
            out += rows
            if len(rows) < 1000:
                return out
            start += 1000

    def invalidate(self) -> None:
        with self._lock:
            self._cache = None

    def overview(self) -> dict:
        with self._lock:
            if self._cache and time.time() - self._cache[0] < self._ttl:
                return self._cache[1]
        sectors = self._rows("sectors", "sector_id,sector")
        names = {s["sector_id"]: s["sector"] for s in sectors}
        companies = self._paged("companies", COMPANY_COLUMNS)
        profiles = {p["company_id"]: p for p in self._rows("company_profile", "*")}
        per_company: dict = {}
        for i in self._rows("ai_insights", "company_id,insight_id"):
            per_company[i["company_id"]] = per_company.get(i["company_id"], 0) + 1
        cards = [_company_card(r, names.get(r.get("sector_id")), profiles.get(r["company_id"]), per_company.get(r["company_id"], 0))
                 for r in companies]
        data = _overview(sectors, cards)
        with self._lock:
            self._cache = (time.time(), data)
        return data

    def company(self, company_id: int) -> dict | None:
        rows = self._rows("companies", COMPANY_COLUMNS, company_id=company_id)
        if not rows:
            return None
        row = rows[0]
        sector = (self._rows("sectors", "sector_id,sector", sector_id=row.get("sector_id")) or [{}])[0].get("sector")
        prof = (self._rows("company_profile", "*", company_id=company_id) or [None])[0]
        insights = self._rows("ai_insights", "*", company_id=company_id)
        products = [r["product_name"] for r in self._rows("products", "product_name", company_id=company_id)]
        components = [r["component_name"] for r in self._rows("components", "component_name", company_id=company_id)]
        contacts = self._rows("contacts", CONTACT_COLUMNS, company_id=company_id)
        lead = (self._rows("lead_status", "status,priority,lead_score,source", company_id=company_id) or [None])[0]
        return _detail(row, sector, prof, insights, products, components, contacts, lead)


class DemoCrm:
    """The bundled development set: 20 companies across the 7 real sectors. Read-only, in memory."""
    kind = "demo"

    def __init__(self, path: str | None = None):
        path = path or os.path.join(os.path.dirname(__file__), "demo", "first20.json")
        with open(path, encoding="utf-8") as fh:
            raw = json.load(fh)["companies"]
        self.sectors = [{"sector_id": i + 1, "sector": s} for i, s in enumerate(sorted({c["sector"] for c in raw}))]
        ids = {s["sector"]: s["sector_id"] for s in self.sectors}
        self.raw = {c["company_id"]: c for c in raw}
        self.rows = {c["company_id"]: {"company_id": c["company_id"], "company_name": c["name"], "country": c.get("country"),
                                       "sector_id": ids[c["sector"]], "sector": None, "website": None, "linkedin_url": None,
                                       "company_type": None, "enrichment_status": "Pending", "priority": "Medium",
                                       "number_of_plants": None, "revenue": None} for c in raw}
        self._names = {s["sector_id"]: s["sector"] for s in self.sectors}

    def invalidate(self) -> None:
        pass

    def overview(self) -> dict:
        cards = [_company_card(r, self._names[r["sector_id"]], None, 0) for r in self.rows.values()]
        return _overview(self.sectors, cards)

    def company(self, company_id: int) -> dict | None:
        row = self.rows.get(company_id)
        if row is None:
            return None
        raw = self.raw[company_id]
        return _detail(row, self._names[row["sector_id"]], None, [], list(raw.get("products", [])),
                       list(raw.get("components", [])), [], None)

    # ---- helpers for the simulation service
    def company_for_world(self, company_id: int) -> dict | None:
        raw = self.raw.get(company_id)
        return None if raw is None else {"name": raw["name"], "country": raw.get("country"), "sector": raw["sector"],
                                         "products": raw.get("products", []), "components": raw.get("components", [])}

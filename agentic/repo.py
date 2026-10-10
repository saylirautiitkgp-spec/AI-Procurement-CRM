"""Read-only access to the CRM data (Supabase is the source of truth).

The agent runtime sees ONLY this interface: there is no insert/update/delete
here. Writes happen exclusively through the approval-gated commit path
(agentic/commit.py). Contact details are reduced to counts/designations so
personal data is not sent to LLM providers.
"""

from __future__ import annotations

from typing import Any, Protocol

from agentic import names

COMPANY_COLUMNS = ("company_id,company_name,sector_id,country,revenue,number_of_plants,"
                   "website,linkedin_url,company_type,enrichment_status,sector,sub_sector")


class CompanyRepository(Protocol):
    def find_candidates(self, name: str, limit: int = 10) -> list[dict]: ...
    def get_company(self, company_id: int) -> dict | None: ...
    def get_profile(self, company_id: int) -> dict | None: ...
    def get_products(self, company_id: int) -> list[str]: ...
    def get_components(self, company_id: int) -> list[str]: ...
    def get_contact_summary(self, company_id: int) -> dict: ...
    def get_company_suppliers(self, company_id: int) -> list[dict]: ...
    def get_lead_status(self, company_id: int) -> dict | None: ...


# --------------------------------------------------------------------------
# in-memory implementation (tests, offline evaluation)
# --------------------------------------------------------------------------

class InMemoryRepo:
    def __init__(self, companies: list[dict] | None = None, *, products=None, components=None,
                 contacts=None, suppliers=None, profiles=None, lead_status=None):
        self.companies = {c["company_id"]: c for c in (companies or [])}
        self._products = products or {}
        self._components = components or {}
        self._contacts = contacts or {}
        self._suppliers = suppliers or {}
        self._profiles = profiles or {}
        self._lead = lead_status or {}

    def find_candidates(self, name, limit=10):
        scored = []
        for c in self.companies.values():
            s = names.similarity(name, c["company_name"])
            if s >= 0.6:
                scored.append({**c, "score": s})
        scored.sort(key=lambda r: (-r["score"], r["company_id"]))
        return scored[:limit]

    def get_company(self, company_id):
        return self.companies.get(company_id)

    def get_profile(self, company_id):
        return self._profiles.get(company_id)

    def get_products(self, company_id):
        return list(self._products.get(company_id, []))

    def get_components(self, company_id):
        return list(self._components.get(company_id, []))

    def get_contact_summary(self, company_id):
        rows = self._contacts.get(company_id, [])
        return _summarise_contacts(rows)

    def get_company_suppliers(self, company_id):
        return list(self._suppliers.get(company_id, []))

    def get_lead_status(self, company_id):
        return self._lead.get(company_id)


def _summarise_contacts(rows: list[dict]) -> dict:
    return {
        "count": len(rows),
        "designations": sorted({r.get("designation") for r in rows if r.get("designation")})[:5],
        "with_source_url": sum(1 for r in rows if r.get("source_url")),
        "with_email": sum(1 for r in rows if r.get("email")),
        "provenance": "database_unverified" if rows and not any(r.get("source_url") for r in rows) else "mixed",
    }


# --------------------------------------------------------------------------
# Supabase implementation (SELECT only)
# --------------------------------------------------------------------------

class SupabaseReadRepo:
    """Wraps an already-constructed supabase client. Only .select() is used."""

    def __init__(self, client: Any):
        self._c = client
        self._sectors: dict[int, str] | None = None

    def _rows(self, table: str, cols: str, **eq) -> list[dict]:
        q = self._c.table(table).select(cols)
        for k, v in eq.items():
            q = q.eq(k, v)
        return q.execute().data or []

    def find_candidates(self, name, limit=10):
        toks = sorted(names.distinctive_tokens(name), key=len, reverse=True)
        if not toks:
            return []
        rows = (self._c.table("companies").select(COMPANY_COLUMNS)
                .ilike("company_name", f"%{toks[0]}%").limit(100).execute().data or [])
        scored = []
        for r in rows:
            s = names.similarity(name, r["company_name"])
            if s >= 0.6:
                scored.append({**r, "score": s})
        scored.sort(key=lambda r: (-r["score"], r["company_id"]))
        return scored[:limit]

    def _sector_name(self, sector_id):
        if self._sectors is None:
            self._sectors = {r["sector_id"]: r["sector"] for r in self._rows("sectors", "sector_id,sector")}
        return self._sectors.get(sector_id)

    def get_company(self, company_id):
        rows = self._rows("companies", COMPANY_COLUMNS, company_id=company_id)
        if not rows:
            return None
        row = rows[0]
        row["sector_name"] = self._sector_name(row.get("sector_id"))
        return row

    def get_profile(self, company_id):
        rows = self._rows("company_profile", "*", company_id=company_id)
        return rows[0] if rows else None

    def get_products(self, company_id):
        return [r["product_name"] for r in self._rows("products", "product_name", company_id=company_id)]

    def get_components(self, company_id):
        return [r["component_name"] for r in self._rows("components", "component_name", company_id=company_id)]

    def get_contact_summary(self, company_id):
        return _summarise_contacts(self._rows("contacts", "designation,source_url,email", company_id=company_id))

    def get_company_suppliers(self, company_id):
        links = self._rows("company_suppliers", "supplier_id,component,remarks", company_id=company_id)
        out = []
        for link in links:
            sup = self._rows("suppliers", "supplier_name,website", supplier_id=link["supplier_id"])
            out.append({"supplier_name": sup[0]["supplier_name"] if sup else None,
                        "component": link.get("component")})
        return out

    def get_lead_status(self, company_id):
        rows = self._rows("lead_status", "status,priority,lead_score,source", company_id=company_id)
        return rows[0] if rows else None


# --------------------------------------------------------------------------
# resolution (deterministic; used by the intake node)
# --------------------------------------------------------------------------

def _richness(rec: dict) -> int:
    return sum(1 for k, v in rec.items() if v not in (None, "", [], {}) and k != "score")


def resolve_company(repo: CompanyRepository, name: str, min_score: float = 0.75) -> dict:
    """Resolve a submitted name to existing record(s). No ID required.

    status: found | ambiguous | not_found
    - found: exactly one candidate at/above the threshold group
    - ambiguous: several distinct rows are equally good (e.g. the same company
      listed under several sectors, or similarly named companies). The
      richest row is used as `primary`, the assumption is stated, nothing is merged.
    """
    candidates = [c for c in repo.find_candidates(name, limit=10) if c["score"] >= min_score]
    if not candidates:
        return {"status": "not_found", "matches": [], "primary": None, "assumptions": []}
    best = candidates[0]["score"]
    top = [c for c in candidates if best - c["score"] < 0.02]
    matches = [{"company_id": c["company_id"], "company_name": c["company_name"],
                "score": c["score"], "match_type": names.match_type(c["score"])} for c in candidates[:5]]
    primary_row = max(top, key=lambda c: (_richness(c), -c["company_id"]))
    primary = repo.get_company(primary_row["company_id"]) or primary_row
    assumptions = []
    status = "found"
    if len(top) > 1:
        status = "ambiguous"
        ids = sorted(c["company_id"] for c in top)
        distinct_names = {names.normalize_company_name(c["company_name"]) for c in top}
        if len(distinct_names) == 1:
            assumptions.append(
                f"{len(top)} database rows share this name (ids {ids}); treated as one organisation "
                f"listed more than once. Used the most complete row (id {primary_row['company_id']}); "
                "no merge was performed - a merge needs a reviewed plan and approval.")
        else:
            assumptions.append(
                f"{len(top)} similarly named records match (ids {ids}); used the closest/most complete "
                f"(id {primary_row['company_id']}). Identity will be confirmed from web evidence.")
    elif best < 0.9:
        assumptions.append(f"Matched '{primary_row['company_name']}' with similarity {best}; "
                           "identity will be confirmed from web evidence.")
    return {"status": status, "matches": matches, "primary": primary, "assumptions": assumptions}

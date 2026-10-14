"""Requested-field catalog: which agent produces which field, and where an approved value would be stored.

A user can pick a company and say exactly which fields to retrieve or refresh. This module maps those fields to
agents, applies the prerequisite rules (identity first, components before suppliers, lead last) and groups the agents
into STAGES. Agents inside one stage are independent of each other and may run in parallel; stages run in order.

Nothing here talks to the network or the database. The database targets are labels for the review change set
(`agentic/changeset.py`); writes still go only through the approval-gated proposal/commit path.
"""

from __future__ import annotations

from dataclasses import dataclass

from agentic import gaps as gp
from agentic.proposals import COMPANY_MAP, PROFILE_MAP


@dataclass(frozen=True)
class FieldSpec:
    key: str
    label: str
    agent: str                       # company | contact | procurement | supplier | lead
    entity: str                      # company | contact | supplier | lead
    table: str | None = None         # database table an approved value would go to (None = report only)
    column: str | None = None
    aliases: tuple = ()


def _company(key, label, *aliases):
    if key in COMPANY_MAP:
        return FieldSpec(key, label, "company", "company", "companies", COMPANY_MAP[key], aliases)
    return FieldSpec(key, label, "company", "company", "company_profile", PROFILE_MAP.get(key), aliases)


CATALOG: tuple[FieldSpec, ...] = (
    _company("official_website", "Official website", "website", "site", "url", "web_site"),
    _company("linkedin_url", "LinkedIn page", "linkedin"),
    _company("headquarters", "Headquarters", "hq", "headquarter", "head_office", "registered_office"),
    _company("industry", "Industry", "sector"),
    _company("business_overview", "Business overview", "overview", "summary", "description", "about"),
    _company("products_services", "Products and services", "products", "services", "offerings"),
    _company("employee_count", "Employee count", "employees", "headcount", "company_size"),
    _company("annual_revenue", "Annual revenue", "revenue", "turnover"),
    _company("founded_year", "Year founded", "founded", "year_founded", "established"),
    _company("parent_company", "Parent company", "parent"),
    _company("ownership_type", "Ownership type", "ownership"),
    _company("company_type", "Company type", "type"),
    _company("manufacturing_locations", "Manufacturing locations", "locations", "plants", "factories"),
    FieldSpec("supplier_portal", "Supplier registration portal", "procurement", "company", "company_profile",
              PROFILE_MAP.get("supplier_portal"), ("portal", "supplier_registration", "vendor_portal")),
    FieldSpec("procurement_page", "Procurement / sourcing page", "procurement", "company", "company_profile",
              PROFILE_MAP.get("procurement_page"), ("procurement_pages", "sourcing_page")),
    FieldSpec("purchasing_category", "Purchasing categories", "procurement", "company", "ai_insights", None,
              ("categories", "purchasing_categories")),
    FieldSpec("purchased_component", "Components the company buys", "procurement", "company", "ai_insights", None,
              ("components", "purchased_components", "bought_components")),
    FieldSpec("process_note", "Procurement process notes", "procurement", "company", "ai_insights", None,
              ("process", "procurement_process")),
    FieldSpec("procurement_contacts", "Procurement decision-makers", "contact", "contact", "contacts", None,
              ("contacts", "decision_makers", "procurement_head", "procurement_contact")),
    FieldSpec("suppliers", "Evidenced suppliers", "supplier", "supplier", "company_suppliers", None,
              ("supplier", "vendors", "supplier_relationships")),
    FieldSpec("lead_score", "Lead priority score", "lead", "lead", None, None, ("score", "lead", "priority")),
    FieldSpec("lead_next_actions", "Lead next actions", "lead", "lead", None, None, ("next_actions",)),
)
BY_KEY = {f.key: f for f in CATALOG}
_ALIAS = {a: f.key for f in CATALOG for a in (f.key, *f.aliases)}
# Independent agents share a stage; stages run in this order (identity -> parallel research -> suppliers -> lead).
STAGE_ORDER: tuple[tuple[str, ...], ...] = (("company",), ("contact", "procurement"), ("supplier",), ("lead",))
PARALLEL_SAFE = {"contact", "procurement"}


def _norm(raw: str) -> str:
    return "_".join(str(raw).strip().lower().replace("-", " ").replace("/", " ").split())


def resolve_fields(requested) -> tuple[list[str], list[str]]:
    """(canonical keys in request order without duplicates, unknown names as given). Never guesses."""
    keys: list[str] = []
    unknown: list[str] = []
    for raw in requested or []:
        key = _ALIAS.get(_norm(raw))
        if key is None:
            unknown.append(str(raw)[:60])
        elif key not in keys:
            keys.append(key)
    return keys, unknown


def agent_fields(agents) -> list[str]:
    """Every catalog field the given agents produce, in catalog order."""
    return [f.key for f in CATALOG if f.agent in set(agents)]


# Where every enriched field lives. LAYERS: `core` = columns of public.companies; `extended` = public.company_profile (one row
# per company, unique company_id); `insight` = only a sourced ai_insights row (no column holds it); `report_only` = never stored.
# PROVENANCE: every written core/extended field also gets one sourced ai_insights row (type 'Verified <field>').
DERIVED_NOT_WRITTEN = {"website_domain": "company_profile.website_domain - would come from the verified identity domain; "
                                         "not written by the change-set path (needs its own reviewed item)"}


def canonical_destinations() -> dict:
    out = {}
    for f in CATALOG:
        layer = {"companies": "core", "company_profile": "extended", "ai_insights": "insight", "contacts": "other_entity",
                 "company_suppliers": "other_entity"}.get(f.table, "report_only")
        out[f.key] = {"layer": layer, "target": f"{f.table}.{f.column}" if f.table and f.column else f.table,
                      "provenance": "this ai_insights row" if layer == "insight" else
                                    ("ai_insights" if layer in ("core", "extended") else None)}
    return out


def company_agent_fields() -> list[str]:
    """Every catalog field the Company Intelligence agent produces, in catalog order."""
    return agent_fields(["company"])


def catalog_view() -> list[dict]:
    return [{"key": f.key, "label": f.label, "agent": f.agent, "entity": f.entity,
             "stores_to": f"{f.table}.{f.column}" if f.table and f.column else f.table, "aliases": list(f.aliases)}
            for f in CATALOG]


# --------------------------------------------------------------------------------------------- CRM old values
def crm_value(key: str, existing: dict | None):
    """The value currently stored in the CRM for this field (UNVERIFIED legacy data), or None."""
    spec = BY_KEY.get(key)
    if not spec or not existing or not spec.column:
        return None
    if spec.table == "companies":
        return (existing.get("fields") or {}).get(spec.column)
    if spec.table == "company_profile":
        return (existing.get("profile") or {}).get(spec.column)
    return None


# Legacy free-text columns that describe the same fact as a write target but are NOT written (read-only context).
# companies.revenue is confirmed to exist (live SELECT); the enrichment target is company_profile.annual_revenue.
LEGACY_COLUMNS: dict[str, tuple[str, str]] = {"annual_revenue": ("companies", "revenue"),
                                               "company_type": ("companies", "company_type")}


def legacy_value(key: str, existing: dict | None):
    ref = LEGACY_COLUMNS.get(key)
    if not ref or not existing:
        return None
    return (existing.get("fields") or {}).get(ref[1]) if ref[0] == "companies" else None


def has_crm_value(key: str, existing: dict | None) -> bool:
    return crm_value(key, existing) not in (None, "", [], {})


# ------------------------------------------------------------------------------------------------ planning
def agents_needed(state: dict) -> set[str]:
    """Agents required for the requested fields, with prerequisites. Mirrors supervisor.eligibility()."""
    opts = state.get("options") or {}
    keys = opts.get("fields") or []
    refresh = bool(opts.get("refresh"))
    existing = state["subject"].get("existing")
    need: set[str] = set()
    for k in keys:
        spec = BY_KEY[k]
        # a company field already stored in the CRM is not researched again unless a refresh was requested
        if spec.agent == "company" and spec.column and not refresh and has_crm_value(k, existing):
            continue
        need.add(spec.agent)
    identity_ok = bool((state["subject"].get("identity") or {}).get("verified"))
    if need & {"contact", "procurement", "supplier"} and not identity_ok:
        need.add("company")
    if "supplier" in need:
        known = (existing or {}).get("components") or (existing or {}).get("products") or \
            gp.verified_fields(state["claims"], "procurement")
        if not known:
            need.add("procurement")
    if "lead" in need and not (state["agent_runs"].get("company") or (existing or {}).get("fields")):
        need.add("company")
    return need


def next_stage(state: dict) -> tuple[list[str], list[str]]:
    """(agents of the next stage that still have to run, all agents needed). Empty list = plan complete."""
    need = agents_needed(state)
    for stage in STAGE_ORDER:
        todo = [a for a in stage if a in need and state["agent_runs"].get(a, 0) < 1]
        if todo:
            return todo, sorted(need)
    return [], sorted(need)


def focus_company_fields(state: dict) -> list[str]:
    """Company fields the Company agent should work on in a field-request run (always includes the website,
    which establishes identity for every other agent)."""
    opts = state.get("options") or {}
    existing = state["subject"].get("existing")
    refresh = bool(opts.get("refresh"))
    wanted = [k for k in opts.get("fields", []) if BY_KEY[k].agent == "company"]
    todo = [k for k in wanted if refresh or not (BY_KEY[k].column and has_crm_value(k, existing))]
    if "official_website" not in todo:
        todo.insert(0, "official_website")
    return todo


def unresolved_requested(agent: str, claims: dict, requested: list[str] | None) -> list[str]:
    """Requested fields of `agent` that have no verified claim yet (drives the single follow-up search)."""
    mine = [k for k in (requested or []) if BY_KEY[k].agent == agent]
    items = [c for c in claims.get(agent, []) if isinstance(c, dict) and c.get("status") == "verified"] \
        if isinstance(claims.get(agent), list) else []
    if agent == "company":
        done = {c.get("field") for c in items}
        return [k for k in mine if k not in done]
    if agent == "procurement":
        done = {c.get("kind") or c.get("field") for c in items}
        return [k for k in mine if k not in done]
    return mine if (mine and not items) else []


def focus_line(agent: str, state: dict) -> str:
    """One task line telling an agent which of the requested fields are its job (empty without a field request)."""
    opts = state.get("options") or {}
    if not opts.get("fields"):
        return ""
    if agent == "company":
        todo = focus_company_fields(state)
        return ("REQUESTED FIELDS (work only on these; the official website also establishes identity): "
                + ", ".join(todo) + ".")
    mine = [BY_KEY[k].label for k in opts["fields"] if BY_KEY[k].agent == agent]
    return ("REQUESTED FOR THIS RUN: " + ", ".join(mine) + ". Concentrate on these; skip everything else.") if mine else ""


# Every column of public.companies (schema_introspection.json, 2026-10-12) and whether this runtime may enrich it.
COMPANIES_COLUMNS_REVIEW: dict[str, tuple[str, str | None]] = {
    # column: (status, field key or reason)
    "website": ("enrichable", "official_website"),
    "linkedin_url": ("enrichable", "linkedin_url"),
    "company_type": ("context_only", "researched as company_profile.company_type; companies.company_type is legacy text and is never written"),
    "revenue": ("context_only", "researched as company_profile.annual_revenue; companies.revenue is legacy text and is never written"),
    "country": ("preserve_if_set", "in the write allowlist, but a stored value is never overwritten without a separate approval"),
    "company_name": ("identity", "the record's identity; never researched or changed"),
    "company_id": ("key", "primary key"),
    "sector_id": ("curated", "foreign key to the curated sectors list; never inferred from web text"),
    "sector": ("not_written", "legacy text, not in the write allowlist"),
    "sub_sector": ("not_written", "legacy text, not in the write allowlist"),
    "number_of_plants": ("not_written", "no evidence-checked claim type and not in the write allowlist"),
    "priority": ("workflow", "workflow field, not enriched by evidence"),
    "enrichment_status": ("workflow", "legacy workflow marker; not written by the agentic runtime"),
    "enrichment_version": ("workflow", "legacy workflow marker; not written by the agentic runtime"),
    "last_enriched": ("workflow", "legacy workflow marker; not written by the agentic runtime"),
    "last_error": ("workflow", "legacy workflow marker; not written by the agentic runtime"),
    "created_at": ("system", "set by the database"),
}


def companies_review(existing: dict | None, requested: list[str] | None = None) -> list[dict]:
    """One row per `companies` column: may it be enriched, why/why not, and does the CRM hold a value today."""
    fields = (existing or {}).get("fields") or {}
    out = []
    for col, (status, note) in COMPANIES_COLUMNS_REVIEW.items():
        row = {"column": f"companies.{col}", "status": status,
               "current_value_present": fields.get(col) not in (None, "", [], {}) if col in fields else None}
        if status == "enrichable":
            row["field"] = note
            row["researched_in_this_run"] = (note in (requested or [])) if requested else None
        else:
            row["reason"] = note
        out.append(row)
    return out

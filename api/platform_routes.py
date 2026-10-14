"""HTTP API of the AI CRM Procurement Platform web app (read-only CRM data + a budget-checked way to start research).

  GET  /platform/status                 what this server is connected to (no data, no secrets)
  GET  /platform/overview               sectors + companies (for browsing, sorting, grouping in the browser)
  GET  /platform/companies/{id}         one company: profile, field coverage, provenance rows, products, contacts summary
  GET  /platform/fields                 the information the user can ask for, grouped by agent, plus presets
  POST /platform/estimate               agents, requests, credits, quota and the blocking reasons for a request - NO external call
  POST /platform/runs                   start a research run (live runs are re-checked by the same preflight)
  POST /platform/runs/{id}/apply        apply the selected, validated, additive change-set items (token + writes enabled only)
  POST /platform/refresh                drop the cached CRM overview

Research results, run tracking, change sets and reports are read through the existing /agentic/* endpoints.
Nothing here can delete or overwrite: the write path is agentic/apply_validated.py behind a counted, allow-listed client.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from agentic import fields as fl
from agentic import preflight as pf
from agentic.agents import AGENT_SPECS
from agentic.security import redact_obj
from api.agentic_routes import approver_auth, auth

router = APIRouter(prefix="/platform", tags=["platform"])

APP_NAME = "AI CRM Procurement Platform"
ASSETS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "frontend", "assets")


def official_logo() -> str | None:
    """Relative URL of the official logo if the file was dropped into frontend/assets, else None (the UI draws a placeholder)."""
    for ext in ("svg", "png", "webp", "jpg"):
        if os.path.isfile(os.path.join(ASSETS_DIR, f"motherson-logo.{ext}")):
            return f"assets/motherson-logo.{ext}"
    return None
LLM_AGENTS = ("company", "contact", "procurement", "supplier")      # lead is computed in code
AGENT_TITLES = {"company": "Company Intelligence", "contact": "Contact Intelligence", "procurement": "Procurement Intelligence",
                "supplier": "Supplier & Relationship Intelligence", "lead": "Lead Prioritisation"}
GROUPS = [("company", "Company profile", "Identity, size, location and what the company does"),
          ("procurement", "Procurement", "Supplier portal, procurement pages, what the company buys"),
          ("contact", "Decision-makers", "Publicly documented procurement and sourcing contacts"),
          ("supplier", "Suppliers", "Suppliers a public source explicitly names for this company"),
          ("lead", "Lead priority", "Explainable score from verified data (no web, no model)")]
PRESETS = [
    {"id": "quick", "title": "Quick profile", "blurb": "Website, headquarters, industry, overview",
     "fields": ["official_website", "headquarters", "industry", "business_overview"]},
    {"id": "size", "title": "Size & ownership", "blurb": "Employees, revenue, founding year, parent",
     "fields": ["employee_count", "annual_revenue", "founded_year", "parent_company", "ownership_type"]},
    {"id": "procurement", "title": "How they buy", "blurb": "Supplier portal, procurement page, categories, components",
     "fields": ["supplier_portal", "procurement_page", "purchasing_category", "purchased_component"]},
    {"id": "people", "title": "Decision-makers", "blurb": "Public procurement contacts", "fields": ["procurement_contacts"]},
    {"id": "full", "title": "Full intelligence report", "blurb": "Everything above plus suppliers and lead score",
     "fields": [f.key for f in fl.CATALOG]},
]


@dataclass
class PlatformState:
    crm: Any = None
    research_mode: str = "off"                       # off | simulation | live
    writes_enabled: bool = False
    commit_factory: Callable[[Any], Any] | None = None      # (SelectBudget) -> commit repository, Supabase or in-memory
    settings: Any = None                             # the live service's Settings (limits, model)
    quota: Any = None                                # ProviderQuota from YOUR console
    console_remaining: int | None = None
    notes: list = field(default_factory=list)


STATE = PlatformState()


def configure(**kw) -> PlatformState:
    for k, v in kw.items():
        setattr(STATE, k, v)
    return STATE


def _crm():
    if STATE.crm is None:
        raise HTTPException(503, detail="no CRM data source is configured on this server")
    return STATE.crm


def _service():
    from agentic.service import ServiceNotConfigured, get_service
    try:
        return get_service()
    except ServiceNotConfigured as exc:
        raise HTTPException(503, detail=str(exc))


# ------------------------------------------------------------------------------------------------ read endpoints
@router.get("/status")
def status():
    live = STATE.research_mode == "live"
    s = STATE.settings
    return {"app": APP_NAME, "data_source": getattr(STATE.crm, "kind", "none"), "research_mode": STATE.research_mode,
            "writes_enabled": bool(STATE.writes_enabled and os.environ.get("AGENTIC_API_TOKEN")),
            "token_required": bool(os.environ.get("AGENTIC_API_TOKEN")),
            "provider": getattr(s, "provider", None) if live else ("simulation" if STATE.research_mode == "simulation" else None),
            "model": getattr(s, "model", None) if live else ("scripted-policy" if STATE.research_mode == "simulation" else None),
            "quota": ({"rpm": STATE.quota.rpm, "tpm": STATE.quota.tpm, "rpd": STATE.quota.rpd} if live and STATE.quota else None),
            "logo": official_logo(), "notes": STATE.notes}


@router.get("/overview", dependencies=[Depends(auth)])
def overview():
    try:
        return _crm().overview()
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001 - never echo database errors to the browser
        raise HTTPException(502, detail=f"CRM read failed ({type(exc).__name__})")


@router.get("/companies/{company_id}", dependencies=[Depends(auth)])
def company(company_id: int):
    try:
        found = _crm().company(company_id)
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(502, detail=f"CRM read failed ({type(exc).__name__})")
    if found is None:
        raise HTTPException(404, detail="unknown company")
    return found


@router.get("/fields", dependencies=[Depends(auth)])
def fields():
    cat = {f["key"]: f for f in fl.catalog_view()}
    dest = fl.canonical_destinations()
    groups = []
    for agent, title, blurb in GROUPS:
        items = [{"key": k, "label": c["label"], "stores_to": c["stores_to"], "layer": dest[k]["layer"]}
                 for k, c in cat.items() if c["agent"] == agent]
        groups.append({"agent": agent, "title": title, "blurb": blurb, "uses_model": agent in LLM_AGENTS, "fields": items})
    return {"groups": groups, "presets": PRESETS}


@router.post("/refresh", dependencies=[Depends(auth)])
def refresh_cache():
    _crm().invalidate()
    return {"ok": True}


# ------------------------------------------------------------------------------------------------ planning
class EstimateRequest(BaseModel):
    company_id: int = Field(ge=1)
    fields: list[str] = Field(min_length=1, max_length=30)
    refresh: bool = False


def _plan(req: EstimateRequest) -> dict:
    detail = _crm().company(req.company_id)
    if detail is None:
        raise HTTPException(404, detail="unknown company")
    keys, unknown = fl.resolve_fields(req.fields)
    if unknown:
        raise HTTPException(422, detail="unknown_fields: " + ", ".join(unknown))
    existing = {"fields": detail["company"], "profile": detail["profile"] or {}, "products": detail["products"],
                "components": detail["components"]}
    state = {"options": {"fields": keys, "refresh": req.refresh}, "subject": {"existing": existing, "identity": {"verified": False}},
             "claims": {}, "agent_runs": {}}
    needed = fl.agents_needed(state)
    order = [a for stage in fl.STAGE_ORDER for a in stage if a in needed]
    skipped = [k for k in keys if fl.BY_KEY[k].agent == "company" and fl.BY_KEY[k].column and not req.refresh
               and fl.has_crm_value(k, existing)]
    llm_agents = [a for a in order if a in LLM_AGENTS]
    out = {"company": {"id": req.company_id, "name": detail["company"]["company_name"]}, "fields": keys, "agents": order,
           "agent_titles": [{"name": a, "title": AGENT_TITLES[a], "uses_model": a in LLM_AGENTS} for a in order],
           "skipped_already_stored": skipped, "mode": STATE.research_mode, "scope": order,
           "plan": None, "quota": None, "blocked": False, "reasons": [], "block_line": None, "notes": []}
    if not order:
        out["notes"].append("Everything requested is already stored. Tick 're-verify stored values' to research it again.")
        out["can_run"] = False
        return out
    if STATE.research_mode == "off":
        out.update(blocked=True, reasons=["research is switched off on this server (start it with --research simulation or --research live)"])
    elif STATE.research_mode == "simulation":
        out["notes"].append("SIMULATION: scripted model and synthetic pages. Uses no Gemini or Tavily quota and cannot write to the database.")
        out["plan"] = {"requests_max": None, "tavily_credits": 0, "selects_max": None}
    else:
        s = STATE.settings
        scope = list(order)
        if "company" not in scope and scope != ["lead"]:
            scope = ["company"] + scope
        out["scope"] = scope
        llm_scope = [a for a in scope if a in LLM_AGENTS]
        if not llm_scope:
            out["plan"] = {"requests_min": 0, "requests_max": 0, "tokens_in": 0, "tokens_out": 0, "tavily_credits": 0, "selects_max": 2}
        else:
            plan = pf.build_plan(s, keys, llm_scope, apply=False)
            verdict = pf.evaluate(plan, s, STATE.quota, s.db_path, console_remaining=STATE.console_remaining,
                                  require_model=s.model, require_selects=s.limits.max_db_selects_run or 8)
            out["plan"] = {"requests_min": plan.requests_min, "requests_max": plan.requests_max,
                           "tokens_in": plan.input_tokens_max, "tokens_out": plan.output_tokens_max,
                           "tavily_credits": plan.tavily_credits_max, "selects_max": plan.selects_max,
                           "retries": plan.retries, "corrections": plan.correction_requests}
            out["blocked"], out["reasons"] = verdict.blocked, list(verdict.reasons)
            out["notes"].extend(verdict.warnings)
            if verdict.usage is not None and verdict.usage.known and STATE.quota is not None:
                out["quota"] = {"limit_per_day": STATE.quota.rpd, "planning_cap": int(STATE.quota.rpd * STATE.quota.headroom),
                                "used_24h": verdict.usage.requests_24h, "remaining_under_cap": max(verdict.remaining_requests or 0, 0),
                                "tavily_used_month": verdict.usage.credits_month, "tavily_month_budget": s.limits.monthly_credit_budget}
            if verdict.blocked:
                out["block_line"] = pf.BLOCK_LINE
        out["notes"].append("LIVE: real Gemini, Tavily and Supabase calls within the limits above. Nothing is written until you approve it.")
        if {"contact", "supplier"} & set(order):
            out["notes"].append("Contact and Supplier research have not been validated live yet; results need careful review.")
    out["can_run"] = not out["blocked"]
    return out


@router.post("/estimate", dependencies=[Depends(auth)])
def estimate(req: EstimateRequest):
    return _plan(req)


class RunBody(EstimateRequest):
    pass


@router.post("/runs", status_code=202, dependencies=[Depends(auth)])
def start_run(req: RunBody):
    est = _plan(req)
    if not est["can_run"]:
        raise HTTPException(409, detail={"error": est["block_line"] or "not runnable", "reasons": est["reasons"]})
    svc = _service()
    detail = _crm().company(req.company_id)
    scope = est["scope"] if STATE.research_mode == "live" and est["scope"] != ["lead"] else None
    from agentic.service import QueueFull
    try:
        run_id = svc.submit(detail["company"]["company_name"], None, False, est["fields"], req.refresh, req.company_id,
                            scope_agents=scope)
    except QueueFull:
        raise HTTPException(429, detail="another run is still in progress; try again shortly")
    except ValueError as exc:
        raise HTTPException(422, detail=str(exc))
    return {"run_id": run_id, "mode": svc.mode, "agents": est["agents"], "fields": est["fields"], "plan": est["plan"]}


# ------------------------------------------------------------------------------------------------ guarded apply
class ApplyBody(BaseModel):
    approver: str = Field(min_length=2, max_length=80)
    confirm: bool = False
    item_ids: list[str] | None = Field(default=None, max_length=100)


@router.post("/runs/{run_id}/apply", dependencies=[Depends(approver_auth)])
def apply_run(run_id: str, body: ApplyBody):
    """Write the selected change-set items - additive, validated, company-scoped, counted (agentic/apply_validated.py)."""
    if not (STATE.writes_enabled and STATE.commit_factory and STATE.research_mode == "live"):
        raise HTTPException(409, detail="database writes are disabled on this server (live mode with --enable-writes is required)")
    if not body.confirm:
        raise HTTPException(422, detail="confirm=true is required: this writes rows to the database")
    svc = _service()
    if svc.mode != "live":
        raise HTTPException(409, detail="only live runs can be applied")
    run = svc.store.get_run(run_id)
    if run is None:
        raise HTTPException(404, detail="unknown run")
    if run["status"] not in ("completed", "partial"):
        raise HTTPException(409, detail=f"run is {run['status']}")
    state = svc.runner.state(run_id)
    cs = ((state or {}).get("report") or {}).get("change_set")
    company_id = (cs or {}).get("company_id")
    if not state or not cs or company_id is None:
        raise HTTPException(409, detail="this run has no change set for an existing company")
    from agentic.apply_validated import guarded_write_phase
    from agentic.guards.select_budget import SelectBudget
    budget = SelectBudget(pf.WRITE_SELECTS_CAP)
    repo = STATE.commit_factory(budget)
    result = guarded_write_phase(repo, svc.store, getattr(svc.runner.deps, "schema", None), state, company_id,
                                 f"web-approver:{body.approver.strip()}", budget,
                                 only_ids=set(body.item_ids) if body.item_ids is not None else None)
    _crm().invalidate()
    return redact_obj({"run_id": run_id, "company_id": company_id, "applied_at": time.strftime("%Y-%m-%dT%H:%M:%S"), **result},
                      max_str=3000)

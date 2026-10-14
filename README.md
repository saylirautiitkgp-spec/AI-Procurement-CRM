# AI CRM Procurement Platform

An AI-assisted procurement intelligence platform for Motherson. It keeps a CRM of ~900 companies in **Supabase** and uses a small team of
**LangGraph agents** to research what a buyer-side procurement team needs: company profiles, how a company buys, who decides, and who supplies
whom. Every fact is stored with its **source URL and the exact supporting quote**, and nothing reaches the database without evidence checks
and an explicit human approval.

> **Status (2026-10-14):** working and tested offline (669 tests); first live enrichment and first guarded database write completed for one
> company (AIRBUS India). It is a proof of concept for a free-tier setup, not a finished product - see [Status and limits](#status-and-limits).

---

## Quick start

Requirements: Windows or any OS with Python 3.12. Everything below uses the project's virtual environment `.venv-agentic`
(dependencies: `requirements-agentic.txt`). Secrets live in `.env` (never committed): `SUPABASE_URL`, `SUPABASE_KEY`, `GEMINI_API_KEY`, `TAVILY_API_KEY`.

```text
START_PLATFORM.bat                                   # Windows: starts the web app and opens it
.venv-agentic\Scripts\python.exe serve_platform.py   # same, from a terminal  ->  http://127.0.0.1:8000/ui/
```

The default is **safe**: your real CRM is shown read-only and research is **simulated** (scripted model + synthetic pages): no Gemini or Tavily
quota is used and nothing can be written. No database configured? It falls back to a bundled 20-company demo set.

| To do this | Run |
|---|---|
| Force the demo data (no database needed) | `serve_platform.py --data demo` |
| Real research (Gemini + Tavily + Supabase) | `serve_platform.py --research live --model gemini-3.8-flash --quota "rpm=5,tpm=250000,rpd=20"` |
| ...and allow the guarded "Apply to database" button | add `--enable-writes` (prints a one-time access token for the web app) |
| Browse only | `serve_platform.py --research off` |

`--quota` must be copied from **your** Google AI Studio console; nothing is assumed. The server listens on `127.0.0.1` only. Details: [docs/PLATFORM.md](docs/PLATFORM.md).

## The web app

* **Browse by sector:** a sidebar with live counts, instant search (`/`), sorting (name, sector, country, most researched, priority), country filter, cards / table, and *Group by sector*.
* **Company drawer:** what is stored, what is sourced, what is missing (click a missing field to request it), the evidence behind each researched fact, products, components and contacts (contacts without a source are labelled unverified; e-mails and phones are never shown).
* **"Get information":** choose exactly which information you want (presets or individual fields grouped by agent). Before anything runs you see which agents will run, how many model requests, search credits and tokens it can cost, your remaining daily quota, and every reason a run would be blocked. Then watch the research live and review each proposed value (old vs new, confidence, source link, exact quote).
* **Insights** (sector and country distribution, research coverage), **Research runs** history, light/dark theme, phone-friendly.
* The Motherson logo slot shows a placeholder until you copy the official file to `frontend/assets/motherson-logo.svg` (or `.png`).

## How it works

```text
 user request (company + the fields wanted)
        │
        ▼
  Python plans:  field catalog → agents needed → budgets (preflight)      ← no LLM
        │
        ▼
  Company Intelligence ──► Procurement Intelligence ──► (Contact, Supplier) ──► Lead prioritisation
   Python retrieves + ranks pages (Tavily search, safe fetch, cache)           (deterministic score)
   ONE structured Gemini request per agent interprets the evidence packet
        │
        ▼
  Evidence validator (Python):  quote must exist verbatim in a retrieved page ·
        the quote's sentence must state the field's meaning · confidence is computed, never model-reported
        │
        ▼
  Change set (old vs proposed, sources, quotes)  ──►  human review  ──►  guarded, additive write phase  ──►  Supabase
```

Five agents (Company, Contact, Procurement, Supplier & Relationship, Lead) are routed deterministically for explicit requests; an LLM supervisor
only handles open-ended objectives. Agents have **no write tools**.

### Safety model

* **Evidence first:** a claim without a verbatim quote from a page retrieved in that run is rejected. A number or place that merely appears in an unrelated sentence is not evidence (headquarters ≠ "team in X", revenue ≠ sourcing value, founding year ≠ a team's start date, "Airbus sells X" ≠ "Airbus buys X").
* **Budgets before requests:** the complete plan (every permitted retry included) is checked against per-run limits and the local usage guard *before* any external call. If anything is over or unknown: `PREFLIGHT BLOCKED - no external requests sent`. Requests are reserved atomically; a 503 retry is allowed once and counts; there is **no automatic fallback** to another model or provider.
* **Writes are narrow and additive:** only empty fields are filled and new sourced insights added, for one pinned company, in `companies` / `company_profile` / `ai_insights`; never overwrite, delete, merge, or create contacts/suppliers automatically. Each write needs the access token, a named approver and a confirmation, runs on a client with a hard SELECT cap, and is audited.
* **Webpage text is data, never instructions**; fetches are SSRF-, size- and robots-guarded; the browser UI never puts page data into `innerHTML`.

### Where information is stored

| Layer | Table.column | Notes |
|---|---|---|
| Core | `companies.website`, `companies.linkedin_url` | the site **root** only, and only if verified |
| Extended | `company_profile.*` (one row per company) | headquarters, industry, summary, size, revenue, founding year, ... |
| Evidence | `ai_insights` | one sourced row per written field; the only home of purchasing categories / components / process notes |

## Repository map

```text
frontend/            web app (HTML/CSS/ES modules, no build step)
serve_platform.py    launcher for the web app          START_PLATFORM.bat   Windows shortcut
api/                 platform_app.py + platform_routes.py (web API) · agentic_routes.py (run/tracker/approval API) · legacy app.py
agentic/             the runtime
  graph.py runtime.py service.py        LangGraph wiring, run entry points, background service
  agents/                               the five agents
  prefetch.py routing.py fields.py      evidence packets, deterministic routing, field catalog
  field_support.py website_map.py       field-meaning validation, canonical website mapping
  preflight.py company_live.py          complete-plan budget check · bounded live runner (CLI)
  apply_validated.py audit_existing.py  guarded write phase · read-only audit + corrective plan
  evidence.py proposals.py write_gate.py changeset.py schema_contract.py   evidence ledger, proposals, write allow-list, change set, schema guard
  llm/ backends/ guards/ persistence/   Gemini/Groq adapters + quota pacing · Tavily/fetch/fixtures · executor, budgets, SSRF · SQLite store
  crm_read.py demo_service.py demo/     read-only CRM for the UI · simulation service · demo data
tests/agentic/       669 offline tests (no network)      docs/   design notes, provider decision, platform guide
agents/ services/ workflow/ dashboard/   LEGACY pipeline and Next.js dashboard (kept, not used by the new app)
CLAUDE.md            hand-over notes for AI assistants working in this repo (decisions, verified/unverified, live history)
```

## Command-line tools

```text
python -m agentic.preflight ...                    # print the complete budget plan; makes no external call
python -m agentic.company_live --company "AIRBUS India" --company-id 11 --agents company procurement \
       --model gemini-3.8-flash --retries 1 --retry-503-only --total-requests-cap 2 --apply     # one bounded live run (needs AGENTIC_COMPANY_LIVE_APPROVED=1)
python -m agentic.audit_existing --company-id 11 --load-dotenv                                  # read-only audit of stored data
python -m agentic.merge_plan --snapshot tests/fixtures/duplicate_groups.json data/merge_plan   # dry-run duplicate/merge planner
```

## Tests

```text
PYTHONDONTWRITEBYTECODE=1 .venv-agentic/Scripts/python.exe -m pytest -p no:cacheprovider      # ~3.5 minutes, 669 tests, fully offline
```

No test calls a provider, Tavily or Supabase. The UI was additionally exercised in headless Chrome (desktop, tablet, phone) against the demo set and the real CRM.

## Status and limits

* **Proven live (small n):** one Company-agent enrichment with a guarded database write (AIRBUS India: 1 profile row + 2 sourced insights). The evidence validator correctly rejected a fabricated quote in an earlier run.
* **Not yet proven live:** Procurement, Contact and Supplier research (Gemini capacity errors, `503 high demand`, interrupted the attempts); a research run started from the web app; any productivity gain - **no baseline has been measured, so none is claimed**.
* **Free tier only:** Gemini ≈ 20 requests/day and capacity errors are likely; Tavily 1,000 credits/month; one concurrent run.
* **Not built:** executing merges (only a dry-run planner exists; every foreign key to `companies` cascades on delete, so merges stay blocked pending an approved design), wiring the Next.js dashboard, CSV/XLSX import.
* **Data caveats:** existing CRM contacts have no provenance and are treated as unverified; the CRM content describes what companies *buy*.
* **Known legacy issues** (old `agents/`, `services/`, `workflow/`) are listed in `CLAUDE.md` section 10; the new runtime does not depend on them.

## License / data

Internal project for a Motherson internship. The official Motherson logo and the CRM data are not part of this repository's public assets.

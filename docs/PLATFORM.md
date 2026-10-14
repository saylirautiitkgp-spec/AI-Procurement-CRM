# AI CRM Procurement Platform (local web app)

A browser interface for the agentic runtime: browse the CRM by sector, see what is already known about a company, tell the
platform **which information you want**, see the **cost before anything runs**, watch the research, then review every proposed
value with its source and exact quote. It runs on your own machine; nothing is hosted elsewhere and there is no build step.

The existing Next.js `dashboard/` and the legacy `api/app.py` are untouched. This app has its own FastAPI entry point
(`api/platform_app.py`) and a static front end (`frontend/`).

## Start it

```
START_PLATFORM.bat                                  # Windows: safe default, opens the browser
.venv-agentic\Scripts\python.exe serve_platform.py  # same, from a terminal -> http://127.0.0.1:8000/ui/
```

| Option | Meaning |
|---|---|
| `--data auto` (default) / `supabase` / `demo` | CRM source. `auto` uses Supabase when `SUPABASE_URL`/`SUPABASE_KEY` are in `.env`, else the bundled 20-company demo set. |
| `--research simulation` (default) | Scripted model + synthetic pages. Every screen works, **no Gemini/Tavily quota is used, nothing can be written.** |
| `--research live --model gemini-3.8-flash --quota "rpm=5,tpm=250000,rpd=20"` | Real research. The quota must be copied from **your** AI Studio console (nothing is assumed). `--console-remaining N` tightens the local figure. |
| `--enable-writes` (live only) | Adds the guarded "Apply to database" action. Generates a one-time access token printed in the terminal; the web app needs it (key icon). |
| `--research off` | Browsing only. |

The server listens on `127.0.0.1` only. API keys are read from `.env` inside the process and never printed or sent to the browser.

## What you can do

* **Sectors** (left): filter by sector with live counts that follow your search; **Group by sector** shows collapsible sector groups with research progress.
* **Companies**: instant search (press `/`), sort (name, sector, country, most researched, priority), country filter, "researched only", cards or table.
* **Company drawer**: overview, an *information coverage* grid (what is stored, what is sourced, what is missing - click a missing field to request it), sourced evidence (quote + URL), products/components/contacts (contacts without a source are labelled unverified; e-mail and phone are never returned).
* **Get information** (the information request): pick presets or individual fields grouped by agent. The plan panel shows, before you run: which agents run in order, model requests, search credits, token upper bounds, database reads, the daily quota gauge, and every reason a run is blocked (`PREFLIGHT BLOCKED - no external requests sent`).
* **Research tracker**: live timeline of what the agents do (searching, reading, model answers, evidence validation).
* **Review**: every proposed value shows stored-now vs proposed, confidence, the exact quote and its source link, and the target table.column. Download the change set as JSON.
* **Apply** (live + `--enable-writes` only): tick items, type your name, type the company name, confirm. Only additive, validated items for that company are written (see below).
* **Insights**: portfolio KPIs, companies per sector, research coverage, top countries. **Research runs**: reopen any earlier run.
* Light/dark theme, responsive down to phone width, keyboard friendly, no external fonts or CDNs.

## Safety model (unchanged guards, now visible)

* Live runs are re-checked server-side by the same preflight as the command line (`agentic/preflight.py`): complete plan, per-run limits, the local usage guard, the per-minute pacing, and atomic request reservation. A blocked plan returns HTTP 409 and no external request is made.
* Research never writes. The apply endpoint (`POST /platform/runs/{id}/apply`) needs live mode, `--enable-writes`, the access token, a named approver and `confirm=true`. It uses `agentic/apply_validated.py`: only `fill_empty`/`add_new` items of the run's company, from the Company/Procurement agents, with a source URL, an exact quote that passes the field-meaning check, confidence >= 0.60, no overwrites, no contacts/suppliers, tables `companies`/`company_profile`/`ai_insights` only, on a client with a hard SELECT cap and no delete/upsert/rpc. Every write is audited in the local SQLite store.
* Simulation results are synthetic and are labelled as such everywhere; they are not research about the real company.
* All page data reaches the DOM through `textContent`/`setAttribute` (a test forbids `innerHTML` with data). Static files are served with `no-store`, `nosniff` and `no-referrer`.

## The Motherson logo

The official logo is not in the repository (it is a trademark asset). Until you add it, the header shows a plain wordmark placeholder.
Copy the official file to `frontend/assets/motherson-logo.svg` (or `.png`/`.webp`/`.jpg`) and reload - the server detects it.

## API (all JSON)

`GET /platform/status` - `GET /platform/overview` - `GET /platform/companies/{id}` - `GET /platform/fields` - `POST /platform/estimate` -
`POST /platform/runs` - `POST /platform/runs/{id}/apply` - `POST /platform/refresh`; run tracking, change sets and reports come from the
existing `/agentic/*` endpoints. `AGENTIC_API_TOKEN` (or the generated token) protects everything when set.

## Tests

`tests/agentic/test_platform.py` (API contracts, estimate/preflight enforcement, simulation end to end, guarded apply, static serving,
innerHTML ban, Supabase reader with a fake client) and the rest of the offline suite. The UI itself was exercised in headless Chrome
(sector filter, search, sort, group, table, drawer, wizard, simulated run, review, themes, phone/tablet widths) with zero console errors.

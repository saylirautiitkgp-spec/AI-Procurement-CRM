"""The AI CRM Procurement Platform web app: the agentic API, the platform API and the static front end in one local server.

The legacy `api.app` (and the Next.js `dashboard/`) are untouched. Start it with `python serve_platform.py` and open
http://127.0.0.1:8000/ui/ .
"""

from __future__ import annotations

import os

from fastapi import FastAPI, Request
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

from api.agentic_routes import router as agentic_router
from api.platform_routes import APP_NAME, router as platform_router

FRONTEND_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "frontend")

app = FastAPI(title=APP_NAME, version="1.0")
app.include_router(platform_router)
app.include_router(agentic_router)


@app.middleware("http")
async def no_cache_for_the_ui(request: Request, call_next):
    response = await call_next(request)
    if request.url.path.startswith(("/ui", "/platform", "/agentic")):
        response.headers["Cache-Control"] = "no-store"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
    return response


@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse("/ui/")


if os.path.isdir(FRONTEND_DIR):
    app.mount("/ui", StaticFiles(directory=FRONTEND_DIR, html=True), name="ui")

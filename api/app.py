from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import router

app = FastAPI(
    title="AI Procurement CRM",
    version="1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)

# Agentic runtime (LangGraph). Optional: the legacy backend keeps working in environments where the
# agentic dependencies are not installed.
try:
    from api.agentic_routes import router as agentic_router
    app.include_router(agentic_router)
except ImportError:  # pragma: no cover
    pass

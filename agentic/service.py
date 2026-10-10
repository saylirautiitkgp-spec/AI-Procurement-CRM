"""Background run service + live wiring.

* Runs execute on a bounded worker pool (default: ONE concurrent run) with a small queue, so a burst of
  requests cannot multiply free-tier usage.
* Live wiring is refused unless AGENTIC_MODE=live AND AGENTIC_ALLOW_LIVE=1. Database writes additionally
  need AGENTIC_ENABLE_WRITES=1 (otherwise no commit repository exists and proposals cannot be applied).
"""

from __future__ import annotations

import os
import sqlite3
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Mapping

from agentic.agents import AGENT_SPECS
from agentic.config import Settings, load_settings
from agentic.graph import clean_company_name
from agentic.persistence.store import Store
from agentic.runtime import AgenticRunner, make_deps


class ServiceNotConfigured(RuntimeError):
    pass


class QueueFull(RuntimeError):
    pass


class AgenticService:
    def __init__(self, runner: AgenticRunner, mode: str, settings: Settings, max_queue: int = 5):
        self.runner = runner
        self.store: Store = runner.deps.store
        self.mode = mode
        self.settings = settings
        self.max_queue = max_queue
        self._pool = ThreadPoolExecutor(max_workers=max(1, settings.limits.max_concurrent_runs))
        self._lock = threading.Lock()
        self._outstanding = 0
        self._futures: dict[str, Any] = {}

    def submit(self, company_name: str, objective: str | None, propose_changes: bool = False) -> str:
        name = clean_company_name(company_name)
        if name is None:
            raise ValueError("invalid_company_name")
        with self._lock:
            if self._outstanding >= self.max_queue + self.settings.limits.max_concurrent_runs:
                raise QueueFull("too many runs queued")
            self._outstanding += 1
        run_id = uuid.uuid4().hex[:12]
        self.store.upsert_run(run_id, company_name=name, objective=objective or "", status="queued",
                              tracker={"run_id": run_id, "company_submitted": name, "status": "queued"})
        self._futures[run_id] = self._pool.submit(self._run, run_id, name, objective, propose_changes)
        return run_id

    def _run(self, run_id, name, objective, propose_changes):
        try:
            self.runner.start(name, objective, run_id=run_id, propose_changes=propose_changes)
        except Exception as exc:  # noqa: BLE001 - report a safe error, never the exception text (may echo inputs)
            self.store.upsert_run(run_id, status="failed",
                                  tracker={"run_id": run_id, "company_submitted": name, "status": "failed",
                                           "errors": [{"stage": "runtime", "error": type(exc).__name__}]})
        finally:
            with self._lock:
                self._outstanding -= 1

    def wait(self, run_id: str, timeout: float = 30) -> None:
        fut = self._futures.get(run_id)
        if fut:
            fut.result(timeout=timeout)

    def usage(self) -> dict:
        lim = self.settings.limits
        return {"mode": self.mode, "provider": self.runner.deps.llm.name, "model": self.runner.deps.llm.model,
                "search_credits_used_this_month": self.store.credits_used(),
                "monthly_credit_budget": lim.monthly_credit_budget,
                "max_concurrent_runs": lim.max_concurrent_runs, "queue_limit": self.max_queue,
                "outstanding_runs": self._outstanding,
                "per_run_limits": {"search_credits": lim.max_search_credits_run, "llm_calls": lim.max_llm_calls_run,
                                   "tool_calls": lim.max_tool_calls_run, "tokens": lim.max_total_tokens_run,
                                   "wall_clock_s": lim.max_wall_clock_s}}


# ----------------------------------------------------------------------------- live wiring
def build_live_service(env: Mapping[str, str] | None = None, factories: Mapping[str, Callable] | None = None,
                       data_dir: str | None = None) -> AgenticService:
    env = os.environ if env is None else env
    if env.get("AGENTIC_MODE") != "live" or env.get("AGENTIC_ALLOW_LIVE") != "1":
        raise ServiceNotConfigured("Set AGENTIC_MODE=live and AGENTIC_ALLOW_LIVE=1 to enable live research.")
    settings = load_settings(env)
    if data_dir:
        settings = Settings(limits=settings.limits, provider=settings.provider, model=settings.model,
                            allow_live=True, data_dir=data_dir)
    f = dict(factories or {})
    if settings.provider not in ("gemini", "groq") and "llm" not in f:
        raise ServiceNotConfigured(f"Provider '{settings.provider}' is not implemented (gemini, groq). "
                                   "There is no automatic fallback between providers.")
    if settings.provider == "groq" and "llm" not in f and not env.get("AGENTIC_MODEL"):
        raise ServiceNotConfigured("Set AGENTIC_MODEL explicitly for groq (no default model is assumed).")

    def default_llm():
        tb = env.get("AGENTIC_THINKING_BUDGET")
        if settings.provider == "groq":
            from agentic.llm.groq import GroqClient
            return GroqClient(model=settings.model)
        from agentic.llm.gemini import GeminiClient
        return GeminiClient(model=settings.model, thinking_budget=int(tb) if tb else None)

    def default_search():
        from agentic.backends.search import TavilySearchBackend
        return TavilySearchBackend()

    def default_fetcher():
        from agentic.backends.fetch import SafeHttpFetcher
        return SafeHttpFetcher(max_bytes=settings.limits.fetch_max_bytes, timeout=settings.limits.fetch_timeout_s,
                               max_redirects=settings.limits.fetch_max_redirects,
                               max_text_chars=settings.limits.ledger_doc_chars)

    def supabase_client():
        from dotenv import load_dotenv
        from supabase import create_client
        load_dotenv()
        return create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_KEY"])

    def default_repo():
        from agentic.repo import SupabaseReadRepo
        return SupabaseReadRepo(supabase_client())

    def default_commit():
        from agentic.commit import SupabaseCommitRepo
        return SupabaseCommitRepo(supabase_client())

    llm = f.get("llm", default_llm)()
    search = f.get("search", default_search)()
    fetcher = f.get("fetcher", default_fetcher)()
    repo = f.get("repo", default_repo)()
    commit = f.get("commit", default_commit)() if env.get("AGENTIC_ENABLE_WRITES") == "1" else None
    from langgraph.checkpoint.sqlite import SqliteSaver
    os.makedirs(settings.data_dir, exist_ok=True)
    checkpointer = SqliteSaver(sqlite3.connect(settings.checkpoint_path, check_same_thread=False))
    store = Store(settings.db_path)
    deps = make_deps(llm=llm, repo=repo, search=search, fetcher=fetcher, store=store, settings=settings,
                     specs=AGENT_SPECS, commit_repo=commit)
    return AgenticService(AgenticRunner(deps, AGENT_SPECS, checkpointer=checkpointer), "live", settings)


_service: AgenticService | None = None
_service_error: str | None = None


def configure_service(service: AgenticService | None) -> None:
    global _service, _service_error
    _service, _service_error = service, None


def get_service() -> AgenticService:
    global _service, _service_error
    if _service is None:
        try:
            _service = build_live_service()
        except ServiceNotConfigured as exc:
            _service_error = str(exc)
            raise
    return _service

"""SIMULATION research service for the web platform: the real runtime, a scripted model and synthetic web pages.

It exists so every screen of the platform can be exercised without spending Gemini or Tavily quota. EVERYTHING it produces
is synthetic: pages live on the reserved `.example` TLD and the "model" is the deterministic PolicyLLM test double. The UI
labels every simulated run as such, and no commit repository exists, so nothing can be written to the database.

Not research, not evidence about the real company, and not a measure of model quality.
"""

from __future__ import annotations

from typing import Any, Callable

from agentic.agents import AGENT_SPECS
from agentic.backends.fixtures import FixtureFetcher, FixtureSearch
from agentic.config import Limits, Settings
from agentic.eval.policy import PolicyLLM
from agentic.eval.synthetic import make_world
from agentic.persistence.store import Store
from agentic.repo import InMemoryRepo
from agentic.runtime import AgenticRunner, make_deps
from agentic.service import AgenticService


class _Holder:
    world: dict = {"search_index": [], "pages": {}}


class _Search:
    def __init__(self, holder: _Holder):
        self.holder = holder

    def search(self, query: str, depth: str, max_results: int):
        return FixtureSearch(self.holder.world).search(query, depth, max_results)


class _Fetcher:
    def __init__(self, holder: _Holder):
        self.holder = holder

    def fetch(self, url: str):
        return FixtureFetcher(self.holder.world).fetch(url)


class DemoService(AgenticService):
    """Before each run the synthetic world is rebuilt for the chosen company (one run at a time)."""

    def __init__(self, runner, settings, holder: _Holder, world_company: Callable[[int | None, str], dict]):
        super().__init__(runner, "simulation", settings)
        self._holder, self._world_company = holder, world_company

    def _run(self, run_id, name, objective, propose_changes, fields=None, refresh=False, expected_company_id=None,
             company_only=False, scope_agents=None):
        self._holder.world = make_world(self._world_company(expected_company_id, name), "standard")
        super()._run(run_id, name, objective, propose_changes, fields, refresh, expected_company_id, company_only, scope_agents)


def demo_repo_from(crm) -> InMemoryRepo:
    rows, products, components = [], {}, {}
    for cid, raw in crm.raw.items():
        r = crm.rows[cid]
        rows.append({**r, "sector_name": raw["sector"]})
        products[cid], components[cid] = list(raw.get("products", [])), list(raw.get("components", []))
    return InMemoryRepo(rows, products=products, components=components)


def build_demo_service(crm, repo: Any, data_dir: str = "data") -> DemoService:
    """`repo`: any CompanyRepository - the in-memory demo CRM, or the real read-only Supabase repo (synthetic web only)."""
    holder = _Holder()

    def world_company(company_id, name):
        if hasattr(crm, "company_for_world") and company_id is not None:
            found = crm.company_for_world(company_id)
            if found:
                return found
        row = repo.get_company(company_id) if company_id is not None else None
        if row is None:
            return {"name": name, "country": None, "sector": None, "products": [], "components": []}
        return {"name": row.get("company_name") or name, "country": row.get("country"), "sector": row.get("sector_name"),
                "products": list(repo.get_products(company_id))[:6], "components": list(repo.get_components(company_id))[:6]}

    settings = Settings(limits=Limits(llm_max_retries=0), provider="simulation", model="scripted-policy", allow_live=False,
                        data_dir=data_dir)
    store = Store(settings.db_path.replace("agentic.sqlite", "platform_simulation.sqlite"))
    deps = make_deps(llm=PolicyLLM(), repo=repo, search=_Search(holder), fetcher=_Fetcher(holder), store=store,
                     settings=settings, specs=AGENT_SPECS, commit_repo=None, sleep=lambda s: None)
    return DemoService(AgenticRunner(deps, AGENT_SPECS), settings, holder, world_company)

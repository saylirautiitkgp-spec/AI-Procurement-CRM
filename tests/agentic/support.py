"""Shared test helpers (offline only)."""

from __future__ import annotations

from agentic.agents import AGENT_SPECS
from agentic.backends.fixtures import FixtureFetcher, FixtureSearch
from agentic.config import Limits, Settings
from agentic.eval.policy import PolicyLLM
from agentic.eval.synthetic import make_world
from agentic.persistence.store import Store
from agentic.repo import InMemoryRepo
from agentic.runtime import AgenticRunner, make_deps

COMPANY_A = {"company_id": 7, "company_name": "Apollo Micro Systems", "sector_id": 1, "country": "India",
             "revenue": "~Rs 200 Cr", "number_of_plants": 2, "website": None, "linkedin_url": None,
             "company_type": "Private", "enrichment_status": "Pending", "sector_name": "Aerospace & Defence"}


def default_repo(extra=None):
    companies = [COMPANY_A] + (extra or [])
    return InMemoryRepo(
        companies,
        products={7: ["Avionics", "Military Control Systems"]},
        components={7: ["PCBAs", "Relays"]},
        contacts={7: [{"designation": "Purchase Manager", "source_url": None, "email": None}]},
    )


def make_runner(*, name="Apollo Micro Systems", variant="standard", repo=None, llm=None, limits=None,
                store=None, specs=None, checkpointer=None, world_company=None, extra_nodes=None,
                extra_tools=None, commit_repo=None, settings=None):
    specs = specs or AGENT_SPECS
    world = make_world(world_company or {"name": name, "country": "India", "sector": "electronics",
                                         "products": ["Avionics", "Control systems", "Relays"],
                                         "components": ["PCBAs", "Relays"]}, variant)
    settings = settings or Settings(limits=limits or Limits())
    store = store or Store(":memory:")
    search, fetcher = FixtureSearch(world), FixtureFetcher(world)
    llm = llm or PolicyLLM()
    deps = make_deps(llm=llm, repo=repo if repo is not None else default_repo(), search=search,
                     fetcher=fetcher, store=store, settings=settings, specs=specs,
                     sleep=lambda s: None, extra_tools=extra_tools, commit_repo=commit_repo)
    runner = AgenticRunner(deps, specs, checkpointer=checkpointer, extra_nodes=extra_nodes)
    runner.search, runner.fetcher, runner.llm, runner.world, runner.store = search, fetcher, llm, world, store
    return runner

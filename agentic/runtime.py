"""Wiring: registry, dependencies and the run entry points."""

from __future__ import annotations

import time
import uuid
from typing import Any

from dataclasses import asdict

from agentic.config import Efficiency, Settings
from agentic.deps import Deps
from langgraph.types import Command

from agentic.graph import build_graph
from agentic.proposals import ProposalService
from agentic.llm.base import RetryingLLM
from agentic.persistence.store import Store
from agentic.state import initial_state
from agentic.tools.registry import Tool, ToolOutput, ToolRegistry
from agentic.agents.lead import lead_tools
from agentic.tools.crm import build_crm_tools
from agentic.tools.web import build_web_tools

DEFAULT_OBJECTIVE = ("Build a comprehensive, evidence-backed company intelligence report: identity, profile, "
                     "procurement structure, public procurement decision-makers, purchased products/components, "
                     "supplier relationships and lead priority - using only the agents that help.")


def _finish_handler(ctx, args) -> ToolOutput:
    return ToolOutput(data=args.model_dump())


def build_registry(specs: dict, extra_tools: list[Tool] | None = None, efficiency=None) -> ToolRegistry:
    efficiency = efficiency or Efficiency()
    reg = ToolRegistry()
    web_agents = frozenset(n for n, s in specs.items() if "web_search" in s.tool_names)
    if web_agents:
        for tool in build_web_tools(web_agents):
            reg.register(tool)
    crm_agents = frozenset(n for n, sp in specs.items() if "get_crm_context" in sp.tool_names)
    if crm_agents:
        # when the digest is in the task text the tool stays registered but is offered to nobody
        for tool in build_crm_tools(frozenset() if efficiency.crm_digest_in_task else crm_agents):
            reg.register(tool)
    if "lead" in specs:
        for tool in lead_tools(frozenset() if efficiency.lead_precomputed else frozenset({"lead"})):
            reg.register(tool)
    for tool in extra_tools or []:
        reg.register(tool)
    for spec in specs.values():
        reg.register(Tool(name=spec.finish_tool,
                          description="Submit your findings and finish. Every claim needs verbatim evidence.",
                          args_model=spec.findings_model, handler=_finish_handler,
                          agents=frozenset({spec.name}), kind="finish"))
    return reg


def make_deps(*, llm, repo, search, fetcher, store: Store, settings: Settings, specs: dict,
              commit_repo: Any = None, sleep=time.sleep, extra_tools=None) -> Deps:
    wrapped = RetryingLLM(llm, max_retries=settings.limits.llm_max_retries,
                          base_delay=settings.limits.llm_backoff_base_s, sleep=sleep)
    return Deps(llm=wrapped, registry=build_registry(specs, extra_tools, settings.efficiency), store=store, repo=repo,
                search=search, fetcher=fetcher, commit_repo=commit_repo, settings=settings, sleep=sleep)


class AgenticRunner:
    def __init__(self, deps: Deps, specs: dict, checkpointer=None, extra_nodes=None):
        self.deps = deps
        self.specs = specs
        from langgraph.checkpoint.memory import MemorySaver
        self.checkpointer = checkpointer or MemorySaver()   # SQLite saver is passed in for durable runs
        self.graph = build_graph(deps, specs, checkpointer=self.checkpointer, extra_nodes=extra_nodes)

    @staticmethod
    def config(run_id: str) -> dict:
        return {"configurable": {"thread_id": run_id}, "recursion_limit": 500}

    def start(self, company_name: str, objective: str | None = None, run_id: str | None = None,
              propose_changes: bool = False, fields: list | None = None, refresh: bool = False,
              expected_company_id: int | None = None) -> dict:
        """`fields` = exactly which data to retrieve (names or aliases from agentic/fields.py). Only the agents those
        fields need are run. `refresh` re-verifies company fields the CRM already holds; without it they are skipped."""
        from agentic import fields as fl
        keys, unknown = fl.resolve_fields(fields)
        if unknown:
            raise ValueError("unknown_fields: " + ", ".join(unknown))
        run_id = run_id or uuid.uuid4().hex[:12]
        objective = objective or (("Retrieve" + (" and refresh" if refresh else "") + ": " +
                                   ", ".join(fl.BY_KEY[k].label for k in keys)) if keys else DEFAULT_OBJECTIVE)
        self.deps.store.upsert_run(run_id, company_name=company_name, objective=objective, status="queued")
        options = {"propose_changes": propose_changes, "efficiency": asdict(self.deps.settings.efficiency)}
        if keys:
            options.update(fields=keys, refresh=bool(refresh))
        if expected_company_id is not None:
            options["expected_company_id"] = int(expected_company_id)
        state = initial_state(run_id, company_name, objective, options)
        from agentic.guards import select_budget as sb
        limit = self.deps.limits.max_db_selects_run
        token = sb.activate(sb.SelectBudget(limit) if limit else None)    # ONE budget for the whole run, all agents/branches
        try:
            result = self.graph.invoke(state, self.config(run_id))
        except sb.SelectBudgetExceeded as exc:        # last resort: a path that did not handle it itself
            self.deps.store.upsert_run(run_id, status="failed", tracker={
                "run_id": run_id, "company_submitted": company_name, "status": "failed",
                "errors": [{"stage": "db_select_budget", "error": f"limit_{exc.limit}"}],
                "stop_reason": "db_select_budget_exceeded"})
            raise RuntimeError("db_select_budget_exceeded") from None
        finally:
            sb.deactivate(token)
        self._save(result)
        return result

    def resume(self, run_id: str, decisions: dict) -> dict:
        """Continue a run paused at the approval step. `decisions` maps proposal_id ->
        {"approve": bool, "by": str, "reason": str, "approved_overwrite": bool, "overrides": {...}}."""
        result = self.graph.invoke(Command(resume=decisions), self.config(run_id))
        self._save(result)
        return result

    def state(self, run_id: str) -> dict | None:
        snap = self.graph.get_state(self.config(run_id))
        return dict(snap.values) if snap and snap.values else None

    # ---- standalone (decoupled) proposal flow: research first, propose later
    def service(self) -> ProposalService:
        if self.deps.commit_repo is None:
            raise RuntimeError("no commit repository configured")
        return ProposalService(self.deps.store, self.deps.commit_repo, schema=getattr(self.deps, "schema", None))

    def propose_for_run(self, run_id: str) -> list[dict]:
        st = self.state(run_id)
        if st is None:
            raise KeyError(run_id)
        svc = self.service()
        if (st.get("options") or {}).get("fields"):        # a field request goes through its reviewed change set
            return self.propose_from_change_set(run_id)["proposals"]
        return [svc.summary(p) for p in svc.propose(st)]

    def propose_from_change_set(self, run_id: str, item_ids: list | None = None) -> dict:
        """Reviewed change-set items -> pending proposals. Reads the CRM (drift check, contact comparison); writes
        nothing. Needs a commit repository to exist, i.e. AGENTIC_ENABLE_WRITES=1 - approval and apply stay gated."""
        st = self.state(run_id)
        if st is None:
            raise KeyError(run_id)
        svc = self.service()
        res = svc.propose_from_change_set(st, (st.get("report") or {}).get("change_set"), item_ids)
        return {**{k: v for k, v in res.items() if k != "proposals"},
                "proposals": [{**svc.summary(p), "items": p.get("items", []), "schema_unconfirmed": p.get("schema_unconfirmed", []),
                               "origin": p.get("origin"), "evidence": p.get("evidence", [])} for p in res["proposals"]]}

    def _save(self, result: dict) -> None:
        if result.get("report") is not None:
            self.deps.store.upsert_run(result["run_id"], status=result.get("status"), report=result["report"])

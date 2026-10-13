"""Typed graph state. Everything is plain JSON-serialisable data so it can be
checkpointed (SQLite now, PostgreSQL later) and returned from the tracker API.
Live objects (LLM client, DB repo, search backend) are injected, never stored."""

from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict

from agentic.evidence import now_iso
from agentic.guards.budget import new_budgets
from agentic.security import redact_obj

RUN_STATUSES = ("queued", "running", "awaiting_approval", "completed", "partial", "failed")


class RunState(TypedDict, total=False):
    run_id: str
    objective: str
    options: dict
    status: str                                   # see RUN_STATUSES
    status_reasons: list                          # why a run is partial (empty = clean completed)
    subject: dict                                 # run-scoped research object (no DB id required)
    ledger: dict                                  # doc_id -> retrieved document
    claims: dict                                  # agent -> [validated claim]
    rejected: dict                                # agent -> [rejected claim + reasons]
    agent_results: dict                           # agent -> last run summary
    agent_runs: dict                              # agent -> number of runs
    completed_agents: list
    current_agent: str | None
    agent_objective: str
    agent_messages: list
    agent_notes: dict
    budgets: dict
    route_history: list
    supervisor_steps: int
    stop_reason: str | None
    errors: list
    proposals: list
    approvals: list
    report: dict | None
    parallel_agents: list                         # agents of the current stage running in parallel
    trace: Annotated[list, operator.add]


def initial_state(run_id: str, company_name: str, objective: str, options: dict | None = None) -> RunState:
    return RunState(
        run_id=run_id, objective=objective, options=options or {}, status="queued",
        subject={"name": company_name}, ledger={}, claims={}, rejected={},
        agent_results={}, agent_runs={}, completed_agents=[], current_agent=None,
        agent_objective="", agent_messages=[], agent_notes={}, budgets=new_budgets(),
        route_history=[], supervisor_steps=0, stop_reason=None, errors=[], proposals=[],
        approvals=[], report=None, parallel_agents=[], trace=[],
    )


class Emitter:
    """Collects trace events for one node execution. Events are redacted and
    size-limited at creation, so nothing sensitive can reach the trace."""

    def __init__(self, state: dict):
        self._next = len(state.get("trace", [])) + 1
        self.events: list[dict] = []

    def emit(self, type_: str, agent: str | None = None, **data: Any) -> dict:
        ev = {"seq": self._next, "ts": now_iso(), "type": type_, "agent": agent,
              "data": redact_obj(data, max_str=500)}
        self._next += 1
        self.events.append(ev)
        return ev

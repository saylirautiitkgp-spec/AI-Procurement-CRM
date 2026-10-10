"""Guarded tool executor: the only way an agent's tool call reaches code.

Order of checks: tool exists -> permitted for this agent -> arguments valid
-> run/agent limits -> repeat detection -> search credit budgets -> handler
(exceptions become structured errors) -> bookkeeping.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any

from pydantic import ValidationError

from agentic.guards import budget as bud
from agentic.llm.base import ToolCallRequest
from agentic.security import redact
from agentic.tools.registry import ToolContext, ToolError, ToolRegistry


@dataclass
class ExecResult:
    ok: bool
    tool: str
    data: dict = field(default_factory=dict)
    error: dict | None = None
    docs: list = field(default_factory=list)
    credits: int = 0
    cached: bool = False
    duration_s: float = 0.0
    counted: bool = True
    stop: str | None = None        # set when the agent/run must stop
    parsed_args: Any = None        # validated pydantic args (for finish tools)


def _err(tool: str, code: str, message: str, **kw) -> ExecResult:
    return ExecResult(ok=False, tool=tool, error={"code": code, "message": redact(message)[:400]}, **kw)


class GuardedExecutor:
    def __init__(self, registry: ToolRegistry, deps: Any):
        self.registry = registry
        self.deps = deps

    @property
    def limits(self):
        return self.deps.limits

    def execute(self, *, agent: str, call: ToolCallRequest, budgets: dict,
                ledger: dict, subject: dict, claims: dict | None = None) -> ExecResult:
        started = time.monotonic()
        tool = self.registry.get(call.name)
        if tool is None:
            return self._count(budgets, _err(call.name, "unknown_tool",
                                             f"No tool named '{call.name}'."), started)
        if agent not in tool.agents:
            return self._count(budgets, _err(call.name, "tool_not_permitted",
                                             f"Tool '{call.name}' is not permitted for this agent."), started)
        try:
            args = tool.args_model.model_validate(call.args)
        except ValidationError as exc:
            msgs = "; ".join(f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in exc.errors()[:5])
            return self._count(budgets, _err(call.name, "invalid_arguments", msgs), started)

        if tool.kind == "finish":      # control tool: validated here, processed by the node
            return ExecResult(ok=True, tool=call.name, parsed_args=args, counted=False,
                              data=args.model_dump(), duration_s=time.monotonic() - started)

        run_reason = bud.run_limit_reason(self.limits, budgets)
        if run_reason:
            return _err(call.name, "run_limit", run_reason, stop=run_reason, counted=False)
        agent_reason = bud.agent_limit_reason(self.limits, budgets)
        if agent_reason:
            return _err(call.name, "agent_limit", agent_reason, stop=agent_reason, counted=False)

        sig = call.name + ":" + json.dumps(call.args, sort_keys=True, default=str)
        seen = budgets["call_signatures"].get(sig, 0)
        if seen >= 1:
            budgets["call_signatures"][sig] = seen + 1
            stop = "stalled_repeated_calls" if seen + 1 > self.limits.max_repeated_calls else None
            return _err(call.name, "duplicate_call",
                        "You already made this exact call; its result is above. "
                        "Change the query/URL or finish.", stop=stop, counted=False)

        cost = tool.credit_cost(args) if tool.credit_cost else 0
        ctx = ToolContext(deps=self.deps, agent=agent, call_id=call.id, ledger=ledger,
                          subject=subject, claims=claims or {}, limits=self.limits,
                          search_disabled=bool(budgets.get("search_disabled")))
        cache_hit = bool(tool.is_cached and tool.is_cached(ctx, args))
        if tool.kind == "search" and not cache_hit:
            if budgets.get("search_disabled"):
                return self._count(budgets, _err(call.name, "search_disabled",
                                                 budgets.get("search_disabled_reason") or "search disabled"), started)
            if budgets["search_credits"] + cost > min(self.limits.max_search_credits_run,
                                                      self.limits.max_search_credits_company):
                return self._count(budgets, _err(call.name, "credit_budget_exhausted",
                                                 "Per-run search credit budget exhausted."), started)
            store = getattr(self.deps, "store", None)
            if store is not None and store.credits_used() + cost > self.limits.monthly_credit_budget:
                budgets["search_disabled"] = True
                budgets["search_disabled_reason"] = "monthly_credit_budget_exhausted"
                return self._count(budgets, _err(call.name, "monthly_credit_budget_exhausted",
                                                 "Monthly search credit budget exhausted."), started)

        budgets["call_signatures"][sig] = 1
        try:
            out = tool.handler(ctx, args)
        except ToolError as exc:
            if exc.code in ("search_quota_exhausted", "search_rate_limited_persistent"):
                budgets["search_disabled"] = True
                budgets["search_disabled_reason"] = exc.code
            if exc.retryable:
                budgets["call_signatures"].pop(sig, None)
            return self._count(budgets, _err(call.name, exc.code, exc.message), started)
        except Exception as exc:  # noqa: BLE001 - tools must never crash the graph
            return self._count(budgets, _err(call.name, "tool_exception",
                                             f"{type(exc).__name__}: {exc}"), started)

        credits = 0 if out.cached else int(out.credits or 0)
        if credits:
            budgets["search_credits"] += credits
            store = getattr(self.deps, "store", None)
            if store is not None:
                store.charge_credits(credits)
        res = ExecResult(ok=True, tool=call.name, data=out.data, docs=out.docs, credits=credits,
                         cached=out.cached, duration_s=time.monotonic() - started)
        return self._count(budgets, res, started)

    @staticmethod
    def _count(budgets: dict, res: ExecResult, started: float) -> ExecResult:
        budgets["tool_calls"] += 1
        budgets["agent_tool_calls"] += 1
        res.duration_s = time.monotonic() - started
        return res

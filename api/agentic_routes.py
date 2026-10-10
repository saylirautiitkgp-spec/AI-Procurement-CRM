"""HTTP API for the agentic runtime (run / status / events / report / approvals).

The existing dashboard is untouched; these endpoints are the contract it can adopt later:
  POST /agentic/runs                              start a run (202)
  GET  /agentic/runs                              recent runs
  GET  /agentic/runs/{id}                         live tracker snapshot
  GET  /agentic/runs/{id}/events?after=N          incremental trace (poll with the last seq)
  GET  /agentic/runs/{id}/report                  final report (404/409 until ready)
  GET  /agentic/runs/{id}/proposals               proposed database changes (for the approver)
  POST /agentic/runs/{id}/propose                 build proposals from a finished run (decoupled flow)
  POST /agentic/runs/{id}/approvals               resume a run paused for approval
  POST /agentic/proposals/{pid}/decision          approve/reject one proposal (decoupled flow)
  POST /agentic/proposals/{pid}/apply             apply an approved proposal
  GET  /agentic/usage                             free-tier budgets and concurrency

Security: if AGENTIC_API_TOKEN is set every endpoint needs `Authorization: Bearer <token>`. Approval and
apply endpoints are DISABLED unless a token is configured (there is no other authentication in this app).
"""

from __future__ import annotations

import hmac
import os

from fastapi import APIRouter, Depends, Header, HTTPException, Query
from pydantic import BaseModel, Field

from agentic.security import redact_obj

router = APIRouter(prefix="/agentic", tags=["agentic"])


def service():
    from agentic.service import QueueFull, ServiceNotConfigured, get_service  # noqa: F401
    try:
        return get_service()
    except ServiceNotConfigured as exc:
        raise HTTPException(503, detail=str(exc))


def _check_token(authorization: str | None) -> bool:
    token = os.environ.get("AGENTIC_API_TOKEN")
    if not token:
        return False
    supplied = (authorization or "").removeprefix("Bearer ").strip()
    if not hmac.compare_digest(supplied, token):
        raise HTTPException(401, detail="invalid or missing bearer token")
    return True


def auth(authorization: str | None = Header(default=None)) -> None:
    _check_token(authorization)


def approver_auth(authorization: str | None = Header(default=None)) -> None:
    if not _check_token(authorization):
        raise HTTPException(403, detail="approval endpoints are disabled until AGENTIC_API_TOKEN is configured")


class RunRequest(BaseModel):
    company_name: str = Field(min_length=2, max_length=200)
    objective: str | None = Field(default=None, max_length=500)
    propose_changes: bool = False


class Decision(BaseModel):
    approve: bool
    by: str = Field(min_length=1, max_length=100)
    reason: str = Field(default="", max_length=500)
    approved_overwrite: bool = False
    overrides: dict = Field(default_factory=dict)


class ApprovalBatch(BaseModel):
    decisions: dict[str, Decision]


def _run_or_404(svc, run_id):
    run = svc.store.get_run(run_id)
    if run is None:
        raise HTTPException(404, detail="unknown run")
    return run


@router.post("/runs", status_code=202, dependencies=[Depends(auth)])
def start_run(req: RunRequest):
    from agentic.service import QueueFull
    svc = service()
    try:
        run_id = svc.submit(req.company_name, req.objective, req.propose_changes)
    except ValueError:
        raise HTTPException(422, detail="invalid company name")
    except QueueFull:
        raise HTTPException(429, detail="too many runs queued; try again shortly")
    return {"run_id": run_id, "status": "queued", "mode": svc.mode}


@router.get("/runs", dependencies=[Depends(auth)])
def list_runs(limit: int = Query(20, ge=1, le=100)):
    return {"runs": service().store.list_runs(limit)}


@router.get("/runs/{run_id}", dependencies=[Depends(auth)])
def run_status(run_id: str):
    run = _run_or_404(service(), run_id)
    return {"run_id": run_id, "status": run["status"], "tracker": run["tracker"],
            "report_ready": run["report"] is not None}


@router.get("/runs/{run_id}/events", dependencies=[Depends(auth)])
def run_events(run_id: str, after: int = Query(0, ge=0), limit: int = Query(200, ge=1, le=500)):
    svc = service()
    _run_or_404(svc, run_id)
    events = svc.store.get_events(run_id, after, limit)
    return {"run_id": run_id, "events": events, "last_seq": events[-1]["seq"] if events else after}


@router.get("/runs/{run_id}/report", dependencies=[Depends(auth)])
def run_report(run_id: str):
    run = _run_or_404(service(), run_id)
    if run["report"] is None:
        raise HTTPException(409 if run["status"] in ("queued", "running", "awaiting_approval") else 404,
                            detail=f"report not available (status: {run['status']})")
    return redact_obj(run["report"], max_str=6000)


@router.get("/runs/{run_id}/proposals", dependencies=[Depends(auth)])
def run_proposals(run_id: str):
    svc = service()
    _run_or_404(svc, run_id)
    return {"run_id": run_id, "proposals": [
        {**p["change"], "status": p["status"], "decided_by": p["decided_by"], "applied_at": p["applied_at"]}
        for p in svc.store.list_proposals(run_id)]}


@router.post("/runs/{run_id}/propose", dependencies=[Depends(approver_auth)])
def propose(run_id: str):
    svc = service()
    run = _run_or_404(svc, run_id)
    if run["status"] not in ("completed", "partial"):
        raise HTTPException(409, detail=f"run is {run['status']}")
    try:
        return {"proposals": svc.runner.propose_for_run(run_id)}
    except RuntimeError as exc:
        raise HTTPException(409, detail=str(exc))


@router.post("/runs/{run_id}/approvals", dependencies=[Depends(approver_auth)])
def approvals(run_id: str, batch: ApprovalBatch):
    svc = service()
    run = _run_or_404(svc, run_id)
    if run["status"] != "awaiting_approval":
        raise HTTPException(409, detail=f"run is {run['status']}, not awaiting approval")
    result = svc.runner.resume(run_id, {pid: d.model_dump() for pid, d in batch.decisions.items()})
    return {"run_id": run_id, "status": result["status"],
            "proposals": [{k: p.get(k) for k in ("proposal_id", "kind", "status")} for p in result["proposals"]]}


@router.post("/proposals/{proposal_id}/decision", dependencies=[Depends(approver_auth)])
def decide(proposal_id: str, d: Decision):
    res = service().runner.service().decide(proposal_id, d.approve, by=d.by, reason=d.reason,
                                            approved_overwrite=d.approved_overwrite, overrides=d.overrides)
    if not res["ok"]:
        raise HTTPException(409, detail=res)
    return res


@router.post("/proposals/{proposal_id}/apply", dependencies=[Depends(approver_auth)])
def apply(proposal_id: str):
    res = service().runner.service().apply(proposal_id, actor="api")
    if not res["ok"]:
        raise HTTPException(409, detail=res)
    return res


@router.get("/usage", dependencies=[Depends(auth)])
def usage():
    return service().usage()

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from backend.api.dependencies import get_runtime_app
from backend.application.runtime_app import RuntimeApplication

router = APIRouter(prefix="/api/runtime", tags=["Runtime"])


class ApprovalDecisionRequest(BaseModel):
    decided_by: str | None = Field(default=None)
    reason: str | None = Field(default=None)


@router.get("/capabilities")
def get_capabilities(
    app: RuntimeApplication = Depends(get_runtime_app),
) -> dict:
    return app.get_capabilities()


@router.get("/diagnostics")
def get_diagnostics(
    refresh: bool = False,
    app: RuntimeApplication = Depends(get_runtime_app),
) -> dict:
    return app.get_diagnostics(refresh=refresh)


@router.get("/traces")
def list_recent_traces(
    limit: int = 50,
    app: RuntimeApplication = Depends(get_runtime_app),
) -> dict:
    return app.list_recent_traces(limit=limit)


@router.get("/traces/{trace_id}")
def list_trace_events(
    trace_id: str,
    app: RuntimeApplication = Depends(get_runtime_app),
) -> dict:
    return app.list_trace_events(trace_id=trace_id)


@router.get("/pending-actions")
def list_pending_actions(
    status: str | None = "pending",
    app: RuntimeApplication = Depends(get_runtime_app),
) -> dict:
    return app.list_pending_actions(status=status)


@router.post("/pending-actions/{action_id}/approve")
def approve_pending_action(
    action_id: str,
    request: ApprovalDecisionRequest | None = None,
    app: RuntimeApplication = Depends(get_runtime_app),
) -> dict:
    request = request or ApprovalDecisionRequest()
    return app.approve_pending_action(
        action_id=action_id,
        decided_by=request.decided_by,
        reason=request.reason,
    )


@router.post("/pending-actions/{action_id}/reject")
def reject_pending_action(
    action_id: str,
    request: ApprovalDecisionRequest | None = None,
    app: RuntimeApplication = Depends(get_runtime_app),
) -> dict:
    request = request or ApprovalDecisionRequest()
    return app.reject_pending_action(
        action_id=action_id,
        decided_by=request.decided_by,
        reason=request.reason,
    )

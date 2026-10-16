from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from backend.api.dependencies import get_conversation_app
from backend.application.conversation_app import ConversationApplication

router = APIRouter(prefix="/api/conversation", tags=["Conversation"])

SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",
}


class ConversationRequest(BaseModel):
    session_id: str = Field(..., min_length=1)
    query: str = Field(..., min_length=1)
    collection_name: str = Field(default="__all__")
    model_name: str | None = None
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    top_p: float | None = Field(default=None, ge=0.0, le=1.0)
    history_rounds: int = Field(default=10, ge=1, le=50)
    use_user_memory: bool = False


class PendingActionResumeRequest(BaseModel):
    session_id: str | None = Field(default=None)
    decided_by: str | None = Field(default=None)
    reason: str | None = Field(default=None)
    model_name: str | None = None
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    top_p: float | None = Field(default=None, ge=0.0, le=1.0)


@router.post("/stream")
async def stream_conversation(
    request: ConversationRequest,
    app: ConversationApplication = Depends(get_conversation_app),
) -> StreamingResponse:
    generator = app.stream_conversation(
        session_id=request.session_id,
        query=request.query,
        collection_name=request.collection_name,
        model_name=request.model_name,
        temperature=request.temperature,
        top_p=request.top_p,
        history_rounds=request.history_rounds,
        use_user_memory=request.use_user_memory,
    )
    return StreamingResponse(
        generator,
        media_type="text/event-stream",
        headers=SSE_HEADERS,
    )


@router.post("/pending-actions/{action_id}/approve/stream")
async def approve_pending_action_stream(
    action_id: str,
    request: PendingActionResumeRequest | None = None,
    app: ConversationApplication = Depends(get_conversation_app),
) -> StreamingResponse:
    request = request or PendingActionResumeRequest()
    generator = app.stream_pending_action_decision(
        action_id=action_id,
        approved=True,
        session_id=request.session_id,
        decided_by=request.decided_by,
        reason=request.reason,
        model_name=request.model_name,
        temperature=request.temperature,
        top_p=request.top_p,
    )
    return StreamingResponse(
        generator,
        media_type="text/event-stream",
        headers=SSE_HEADERS,
    )


@router.post("/pending-actions/{action_id}/reject/stream")
async def reject_pending_action_stream(
    action_id: str,
    request: PendingActionResumeRequest | None = None,
    app: ConversationApplication = Depends(get_conversation_app),
) -> StreamingResponse:
    request = request or PendingActionResumeRequest()
    generator = app.stream_pending_action_decision(
        action_id=action_id,
        approved=False,
        session_id=request.session_id,
        decided_by=request.decided_by,
        reason=request.reason,
        model_name=request.model_name,
        temperature=request.temperature,
        top_p=request.top_p,
    )
    return StreamingResponse(
        generator,
        media_type="text/event-stream",
        headers=SSE_HEADERS,
    )


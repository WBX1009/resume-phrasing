from fastapi import APIRouter, Depends

from backend.api.dependencies import get_chat_app
from backend.application.chat_app import ChatApplication

router = APIRouter(prefix="/api/chat", tags=["Chat"])

SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",
}


@router.get("/sessions")
def list_sessions(app: ChatApplication = Depends(get_chat_app)) -> dict:
    return {"status": "success", "data": app.list_sessions()}


@router.get("/history/{session_id}")
def get_history(
    session_id: str,
    app: ChatApplication = Depends(get_chat_app),
) -> dict:
    return {"status": "success", "data": app.get_history(session_id, max_rounds=50)}


@router.delete("/sessions/{session_id}")
def delete_session(
    session_id: str,
    app: ChatApplication = Depends(get_chat_app),
) -> dict:
    if app.delete_session(session_id):
        return {"status": "success", "message": f"Session {session_id} deleted."}
    return {"status": "error", "message": "Failed to delete session."}

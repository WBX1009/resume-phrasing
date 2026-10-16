from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class AgentState(BaseModel):
    """DeepAssist-owned agent state, independent of any runtime framework."""

    session_id: str
    query: str
    route: Optional[str] = None
    messages: List[Dict[str, Any]] = Field(default_factory=list)
    tool_calls: List[Dict[str, Any]] = Field(default_factory=list)
    observations: List[Dict[str, Any]] = Field(default_factory=list)
    retrieved_context: List[Dict[str, Any]] = Field(default_factory=list)
    step_count: int = 0
    error_count: int = 0
    status: str = "running"
    pending_action: Optional[Dict[str, Any]] = None

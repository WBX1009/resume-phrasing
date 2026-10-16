from typing import Any, Dict, Iterator, List, Optional, Protocol

from pydantic import BaseModel, Field

from backend.domain.entities.stream_event import StreamEvent


class AgentRuntimeRequest(BaseModel):
    """Runtime-neutral request consumed by agent runtime adapters."""

    session_id: str
    query: str
    messages: List[Dict[str, Any]] = Field(default_factory=list)
    model_options: Dict[str, Any] = Field(default_factory=dict)
    resume_state: Optional[Dict[str, Any]] = None


class AgentRuntimePort(Protocol):
    """Port for custom, LangGraph, or other agent runtime adapters."""

    def stream_run(self, request: AgentRuntimeRequest) -> Iterator[StreamEvent]:
        ...

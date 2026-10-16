from datetime import datetime, timezone
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class TraceEvent(BaseModel):
    """Structured event used for local tracing and future observability adapters."""

    trace_id: str
    session_id: Optional[str] = None
    route: Optional[str] = None
    event_type: str
    span_name: str
    latency_ms: Optional[float] = None
    input_summary: Optional[str] = None
    output_summary: Optional[str] = None
    error_type: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

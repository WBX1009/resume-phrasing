from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class TaskSnapshot(BaseModel):
    """Persisted runtime snapshot for resuming an unfinished agent task."""

    session_id: str
    query: str
    route_worker: str
    status: str = "running"
    payload: Dict[str, Any] = Field(default_factory=dict)
    updated_at: Optional[str] = None

    def to_trace_data(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "query": self.query,
            "route_worker": self.route_worker,
            "status": self.status,
            "updated_at": self.updated_at,
            "payload_keys": sorted(self.payload.keys()),
        }

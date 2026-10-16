from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, Optional
from uuid import uuid4

from pydantic import BaseModel, Field

from backend.domain.entities.tooling import ToolCall


class ApprovalStatus(str, Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    COMPLETED = "completed"
    FAILED = "failed"


class PendingAction(BaseModel):
    """A high-risk tool call waiting for explicit user approval."""

    action_id: str = Field(default_factory=lambda: uuid4().hex)
    tool_call: ToolCall
    category: str
    reason: str
    status: ApprovalStatus = ApprovalStatus.PENDING
    requested_by: Optional[str] = None
    decided_by: Optional[str] = None
    decision_reason: Optional[str] = None
    result: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    updated_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def touch(self) -> None:
        self.updated_at = datetime.now(timezone.utc).isoformat()

    def to_public_dict(self) -> Dict[str, Any]:
        return {
            "action_id": self.action_id,
            "tool_name": self.tool_call.name,
            "tool_call_id": self.tool_call.id,
            "args": self.tool_call.args,
            "category": self.category,
            "reason": self.reason,
            "status": self.status.value,
            "requested_by": self.requested_by,
            "decided_by": self.decided_by,
            "decision_reason": self.decision_reason,
            "result": self.result,
            "metadata": self.metadata,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

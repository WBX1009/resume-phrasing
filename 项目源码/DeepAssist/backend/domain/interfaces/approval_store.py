from typing import Any, Dict, List, Optional, Protocol

from backend.domain.entities.approval import PendingAction
from backend.domain.entities.tooling import ToolCall


class ApprovalStorePort(Protocol):
    """Port for storing and deciding high-risk pending actions."""

    def create_pending_action(
        self,
        tool_call: ToolCall,
        category: str,
        reason: str,
        metadata: Optional[Dict[str, Any]] = None,
        requested_by: Optional[str] = None,
    ) -> PendingAction:
        ...

    def get(self, action_id: str) -> Optional[PendingAction]:
        ...

    def list(self, status: Optional[str] = None) -> List[PendingAction]:
        ...

    def approve(
        self,
        action_id: str,
        decided_by: Optional[str] = None,
        reason: Optional[str] = None,
    ) -> PendingAction:
        ...

    def reject(
        self,
        action_id: str,
        decided_by: Optional[str] = None,
        reason: Optional[str] = None,
    ) -> PendingAction:
        ...

    def complete(
        self,
        action_id: str,
        result: Dict[str, Any],
        failed: bool = False,
    ) -> PendingAction:
        ...

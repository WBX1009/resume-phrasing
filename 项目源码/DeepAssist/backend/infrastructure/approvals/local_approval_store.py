from copy import deepcopy
from threading import Lock
from typing import Any, Dict, List, Optional

from backend.domain.entities.approval import ApprovalStatus, PendingAction
from backend.domain.entities.tooling import ToolCall
from backend.domain.interfaces.approval_store import ApprovalStorePort


class LocalApprovalStore(ApprovalStorePort):
    """In-memory approval store for local HITL workflows and tests."""

    def __init__(self):
        self._actions: Dict[str, PendingAction] = {}
        self._lock = Lock()

    def create_pending_action(
        self,
        tool_call: ToolCall,
        category: str,
        reason: str,
        metadata: Optional[Dict[str, Any]] = None,
        requested_by: Optional[str] = None,
    ) -> PendingAction:
        action = PendingAction(
            tool_call=tool_call,
            category=category,
            reason=reason,
            requested_by=requested_by,
            metadata=metadata or {},
        )
        with self._lock:
            self._actions[action.action_id] = action
            return deepcopy(action)

    def get(self, action_id: str) -> Optional[PendingAction]:
        with self._lock:
            action = self._actions.get(action_id)
            return deepcopy(action) if action else None

    def list(self, status: Optional[str] = None) -> List[PendingAction]:
        with self._lock:
            items = list(self._actions.values())
            if status:
                items = [item for item in items if item.status.value == status]
            return [deepcopy(item) for item in items]

    def approve(
        self,
        action_id: str,
        decided_by: Optional[str] = None,
        reason: Optional[str] = None,
    ) -> PendingAction:
        with self._lock:
            action = self._require(action_id)
            if action.status != ApprovalStatus.PENDING:
                raise ValueError(f"Pending action {action_id} is already {action.status.value}")
            action.status = ApprovalStatus.APPROVED
            action.decided_by = decided_by
            action.decision_reason = reason
            action.touch()
            return deepcopy(action)

    def reject(
        self,
        action_id: str,
        decided_by: Optional[str] = None,
        reason: Optional[str] = None,
    ) -> PendingAction:
        with self._lock:
            action = self._require(action_id)
            if action.status != ApprovalStatus.PENDING:
                raise ValueError(f"Pending action {action_id} is already {action.status.value}")
            action.status = ApprovalStatus.REJECTED
            action.decided_by = decided_by
            action.decision_reason = reason
            action.touch()
            return deepcopy(action)

    def complete(
        self,
        action_id: str,
        result: Dict[str, Any],
        failed: bool = False,
    ) -> PendingAction:
        with self._lock:
            action = self._require(action_id)
            if action.status not in {ApprovalStatus.APPROVED, ApprovalStatus.FAILED}:
                raise ValueError(
                    f"Pending action {action_id} must be approved before completion"
                )
            action.status = ApprovalStatus.FAILED if failed else ApprovalStatus.COMPLETED
            action.result = result
            action.touch()
            return deepcopy(action)

    def _require(self, action_id: str) -> PendingAction:
        action = self._actions.get(action_id)
        if action is None:
            raise KeyError(f"Unknown pending action: {action_id}")
        return action

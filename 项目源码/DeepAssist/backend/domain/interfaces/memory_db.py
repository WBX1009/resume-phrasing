from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from backend.domain.entities.context_window import ContextSummary
from backend.domain.entities.message import Message
from backend.domain.entities.task_snapshot import TaskSnapshot


class BaseMemoryStore(ABC):
    """Port for conversation history, summaries, profiles, and task snapshots."""

    @abstractmethod
    def get_history(self, session_id: str, limit: int = 10) -> List[Message]:
        """Return recent messages for a session."""
        pass

    @abstractmethod
    def add_message(self, session_id: str, message: Message) -> bool:
        """Persist one canonical message for a session."""
        pass

    @abstractmethod
    def clear_history(self, session_id: str) -> bool:
        pass

    @abstractmethod
    def get_all_sessions(self) -> List[Dict[str, Any]]:
        pass

    @abstractmethod
    def get_profile(self, key: str) -> Optional[str]:
        pass

    @abstractmethod
    def set_profile(self, key: str, value: str) -> bool:
        pass

    @abstractmethod
    def get_all_profiles(self) -> Dict[str, str]:
        pass

    @abstractmethod
    def get_task_snapshot(self, session_id: str) -> Optional[TaskSnapshot]:
        pass

    @abstractmethod
    def save_task_snapshot(self, snapshot: TaskSnapshot) -> bool:
        pass

    @abstractmethod
    def clear_task_snapshot(self, session_id: str) -> bool:
        pass

    @abstractmethod
    def get_session_summary(self, session_id: str) -> Optional[ContextSummary]:
        pass

    @abstractmethod
    def save_session_summary(self, session_id: str, summary: ContextSummary) -> bool:
        pass

    @abstractmethod
    def clear_session_summary(self, session_id: str) -> bool:
        pass

    @abstractmethod
    def save_routing_log(self, log_data: Dict[str, Any]) -> bool:
        pass

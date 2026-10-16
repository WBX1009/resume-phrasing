from enum import Enum
from typing import Any, Dict, List

from pydantic import BaseModel, Field

from backend.domain.entities.long_term_memory import LongTermMemoryItem

class ContextPriorityBand(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ContextTurn(BaseModel):
    """A coherent conversation turn used for context selection."""

    turn_id: str
    started_at_index: int = Field(ge=0)
    messages: List[Dict[str, Any]] = Field(default_factory=list)
    estimated_cost: int = Field(default=0, ge=0)
    priority_score: int = Field(default=0)
    priority_band: ContextPriorityBand = Field(default=ContextPriorityBand.LOW)
    pinned: bool = Field(default=False)
    reasons: List[str] = Field(default_factory=list)

    def flattened_messages(self) -> List[Dict[str, Any]]:
        return list(self.messages)

    def to_trace_data(self) -> Dict[str, Any]:
        user_preview = ""
        for message in self.messages:
            if message.get("role") == "user" and str(message.get("content", "")).strip():
                user_preview = self._clip(str(message.get("content", "")))
                break
        return {
            "turn_id": self.turn_id,
            "estimated_cost": self.estimated_cost,
            "priority_score": self.priority_score,
            "priority_band": self.priority_band.value,
            "pinned": self.pinned,
            "reasons": list(self.reasons),
            "preview": user_preview,
        }

    def _clip(self, text: str, limit: int = 90) -> str:
        compact = " ".join(text.split())
        if len(compact) <= limit:
            return compact
        return compact[: limit - 3] + "..."


class ContextSummary(BaseModel):
    """Compressed representation of dropped history turns."""

    content: str
    source: str = Field(default="runtime")
    source_turn_ids: List[str] = Field(default_factory=list)
    dropped_turn_count: int = Field(default=0, ge=0)
    dropped_message_count: int = Field(default=0, ge=0)
    reason_counts: Dict[str, int] = Field(default_factory=dict)

    def to_message(self) -> Dict[str, Any]:
        return {"role": "system", "content": self.content}

    def to_trace_data(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "dropped_turn_count": self.dropped_turn_count,
            "dropped_message_count": self.dropped_message_count,
            "reason_counts": dict(self.reason_counts),
            "source_turn_ids": list(self.source_turn_ids),
        }


class ContextWindowPlan(BaseModel):
    """Priority-based context window selection result."""

    budget: int = Field(default=0, ge=0)
    selected_turns: List[ContextTurn] = Field(default_factory=list)
    dropped_turns: List[ContextTurn] = Field(default_factory=list)
    summary: ContextSummary | None = None
    recalled_memories: List[LongTermMemoryItem] = Field(default_factory=list)

    @property
    def selected_cost(self) -> int:
        return sum(turn.estimated_cost for turn in self.selected_turns)

    def flattened_messages(self) -> List[Dict[str, Any]]:
        messages: List[Dict[str, Any]] = []
        for memory in self.recalled_memories:
            messages.append(memory.to_message())
        if self.summary is not None:
            messages.append(self.summary.to_message())
        for turn in sorted(self.selected_turns, key=lambda item: item.started_at_index):
            messages.extend(turn.flattened_messages())
        return messages

    def to_trace_data(self) -> Dict[str, Any]:
        return {
            "budget": self.budget,
            "selected_cost": self.selected_cost,
            "selected_turn_count": len(self.selected_turns),
            "dropped_turn_count": len(self.dropped_turns),
            "summary_injected": self.summary is not None,
            "recalled_memory_count": len(self.recalled_memories),
            "selected_turns": [turn.to_trace_data() for turn in self.selected_turns],
            "dropped_turns": [turn.to_trace_data() for turn in self.dropped_turns],
            "recalled_memories": [
                {
                    "key": memory.key,
                    "category": memory.category,
                    "score": memory.score,
                    "content": memory.content,
                }
                for memory in self.recalled_memories
            ],
            "summary": self.summary.to_trace_data() if self.summary is not None else None,
        }

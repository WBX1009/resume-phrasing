from typing import Dict

from pydantic import BaseModel, Field


class LongTermMemoryItem(BaseModel):
    """A recalled long-term memory snippet for the current request."""

    key: str = Field(..., description="Stable profile key")
    content: str = Field(..., description="Human-readable memory content")
    category: str = Field(..., description="Memory category such as fact or topic")
    score: float = Field(default=0.0, description="Recall relevance score")
    source: str = Field(default="user_profile", description="Origin of the memory item")
    metadata: Dict[str, str] = Field(default_factory=dict)

    def to_message(self) -> Dict[str, str]:
        return {
            "role": "system",
            "content": f"[Long-Term Memory][{self.category}] {self.content}",
        }

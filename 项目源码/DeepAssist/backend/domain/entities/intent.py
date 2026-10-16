from enum import Enum
from typing import List

from pydantic import BaseModel, Field


class IntentType(str, Enum):
    CHAT = "chat"
    RAG = "rag"
    AGENT = "agent"


class IntentDecision(BaseModel):
    """Structured decision produced by the intent router."""

    intent: IntentType
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str
    signals: List[str] = Field(default_factory=list)

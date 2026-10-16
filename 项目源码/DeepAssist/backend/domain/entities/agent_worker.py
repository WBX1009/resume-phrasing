from enum import Enum
from typing import List

from pydantic import BaseModel, Field

from backend.domain.entities.intent import IntentDecision


class AgentWorkerType(str, Enum):
    ORCHESTRATOR = "orchestrator_worker"
    CHAT = "chat_worker"
    RAG = "rag_worker"
    TOOL = "tool_agent_worker"


class SupervisorDecision(BaseModel):
    """Routing decision made by the lightweight agent supervisor."""

    worker: AgentWorkerType
    intent: IntentDecision
    reason: str
    signals: List[str] = Field(default_factory=list)

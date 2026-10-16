from enum import Enum
from typing import List

from pydantic import BaseModel, Field, field_validator, model_validator


class ConversationRoute(str, Enum):
    DIRECT_CHAT = "direct_chat"
    RAG = "rag"
    AGENT = "agent"
    CLARIFY = "clarify"


class SourcePolicy(str, Enum):
    REQUIRED = "required"
    OPTIONAL = "optional"
    NOT_APPLICABLE = "not_applicable"


class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class RouteDecision(BaseModel):
    route: ConversationRoute
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str = ""
    rewritten_query: str = ""
    source_policy: SourcePolicy = SourcePolicy.NOT_APPLICABLE
    required_capabilities: List[str] = Field(default_factory=list)
    clarifying_question: str = ""
    risk_level: RiskLevel = RiskLevel.LOW
    should_log_for_training: bool = True

    @field_validator("required_capabilities", mode="before")
    @classmethod
    def normalize_capabilities(cls, value):
        if value is None:
            return []
        if isinstance(value, str):
            return [value]
        return value

    @model_validator(mode="after")
    def normalize_route_fields(self) -> "RouteDecision":
        if not self.rewritten_query:
            self.rewritten_query = ""
        if self.route != ConversationRoute.RAG:
            self.source_policy = SourcePolicy.NOT_APPLICABLE
        if self.route != ConversationRoute.CLARIFY:
            self.clarifying_question = self.clarifying_question or ""
        return self

    def to_stream_data(self) -> dict:
        return self.model_dump(mode="json")


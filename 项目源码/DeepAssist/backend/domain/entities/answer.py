from typing import Any, Dict, List

from pydantic import BaseModel, Field


class AnswerGroundingReport(BaseModel):
    """Lightweight post-generation grounding report for a RAG answer."""

    grounded: bool
    recommended_action: str = "accept"
    reason: str = ""
    citation_count: int = 0
    used_citations: List[str] = Field(default_factory=list)
    missing_citations: List[str] = Field(default_factory=list)
    unknown_citations: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def to_stream_data(self) -> Dict[str, Any]:
        return self.model_dump(exclude_none=True)

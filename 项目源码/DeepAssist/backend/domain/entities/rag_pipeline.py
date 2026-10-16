from typing import Optional

from pydantic import BaseModel

from backend.domain.entities.answer import AnswerGroundingReport
from backend.domain.entities.retrieval import RAGContextPack, RetrievalResult


class RAGPipelineResult(BaseModel):
    """End-to-end RAG pipeline output before and after answer generation."""

    query: str
    collection_name: str
    retrieval_result: RetrievalResult
    context_pack: RAGContextPack
    answer_report: Optional[AnswerGroundingReport] = None

    def with_answer_report(self, report: AnswerGroundingReport) -> "RAGPipelineResult":
        return self.model_copy(update={"answer_report": report})

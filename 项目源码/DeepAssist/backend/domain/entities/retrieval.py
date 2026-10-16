from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from backend.domain.entities.document import DocumentChunk


class RetrievalChannelTrace(BaseModel):
    """Per-channel retrieval diagnostics."""

    enabled: bool = True
    success: bool = True
    error: Optional[str] = None
    returned: int = 0
    query_count: int = 1


class RerankTrace(BaseModel):
    """Reranking diagnostics after first-stage retrieval fusion."""

    enabled: bool = True
    success: bool = True
    model: str = "none"
    backend: Optional[str] = None
    input_count: int = 0
    returned: int = 0
    error: Optional[str] = None
    fallback_reason: Optional[str] = None


class QueryPlan(BaseModel):
    """Structured query plan used before retrieval."""

    original_query: str
    normalized_query: str
    semantic_query: str
    keyword_query: str
    rewritten_query: Optional[str] = None
    semantic_queries: List[str] = Field(default_factory=list)
    keyword_queries: List[str] = Field(default_factory=list)
    sub_queries: List[str] = Field(default_factory=list)
    key_terms: List[str] = Field(default_factory=list)
    quoted_phrases: List[str] = Field(default_factory=list)
    strategy: str = "single_query_hybrid"
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def to_stream_data(self) -> Dict[str, Any]:
        return self.model_dump(exclude_none=True)


class RetrievalResult(BaseModel):
    """Structured retrieval output with fused docs and explainable trace."""

    query: str
    collection_name: str
    documents: List[DocumentChunk] = Field(default_factory=list)
    top_k: int
    candidate_k: int
    fusion: str = "weighted_rrf"
    vector_weight: float = 1.0
    keyword_weight: float = 1.0
    rrf_k: int = 60
    query_plan: Optional[QueryPlan] = None
    rerank: RerankTrace = Field(default_factory=RerankTrace)
    channels: Dict[str, RetrievalChannelTrace] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @property
    def hit_count(self) -> int:
        return len(self.documents)

    def to_stream_data(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "collection_name": self.collection_name,
            "hit_count": self.hit_count,
            "top_k": self.top_k,
            "candidate_k": self.candidate_k,
            "fusion": self.fusion,
            "rrf_k": self.rrf_k,
            "vector_weight": self.vector_weight,
            "keyword_weight": self.keyword_weight,
            "query_plan": self.query_plan.to_stream_data() if self.query_plan else {},
            "rerank": self.rerank.model_dump(exclude_none=True),
            "channels": {
                name: trace.model_dump(exclude_none=True)
                for name, trace in self.channels.items()
            },
            "documents": [
                {
                    "id": doc.id,
                    "score": doc.score,
                    "metadata": doc.metadata,
                }
                for doc in self.documents
            ],
            "metadata": self.metadata,
        }


class Citation(BaseModel):
    """Grounded citation assigned to a packed retrieval chunk."""

    ref_id: str
    chunk_id: str
    content: str
    source_file: Optional[str] = None
    title_path: List[str] = Field(default_factory=list)
    page: Optional[Any] = None
    score: Optional[float] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def label(self) -> str:
        source = self.source_file or "unknown source"
        path = " > ".join(self.title_path)
        if path:
            return f"{self.ref_id} | {source} | {path}"
        return f"{self.ref_id} | {source}"

    def to_stream_data(self) -> Dict[str, Any]:
        return self.model_dump(exclude_none=True)


class RAGContextPack(BaseModel):
    """Packed, cited retrieval context ready for prompt injection."""

    query: str
    citations: List[Citation] = Field(default_factory=list)
    rendered_context: str = ""
    budget_chars: int
    used_chars: int = 0
    truncated: bool = False
    omitted: int = 0

    def to_prompt(self) -> str:
        if not self.rendered_context:
            return (
                "=== Reference Material ===\n"
                "No relevant reference material was retrieved.\n\n"
                "=== Citation Instructions ===\n"
                "Answer from general knowledge only if needed, and state that no retrieved reference supports the answer.\n\n"
                f"=== User Question ===\n{self.query}"
            )

        return (
            "=== Reference Material ===\n"
            f"{self.rendered_context}\n\n"
            "=== Citation Instructions ===\n"
            "Use the reference material when it is relevant. Cite supporting facts with ref ids like [C1]. "
            "If the references are insufficient, say so explicitly.\n\n"
            f"=== User Question ===\n{self.query}"
        )

    def to_stream_data(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "budget_chars": self.budget_chars,
            "used_chars": self.used_chars,
            "truncated": self.truncated,
            "omitted": self.omitted,
            "citations": [citation.to_stream_data() for citation in self.citations],
        }

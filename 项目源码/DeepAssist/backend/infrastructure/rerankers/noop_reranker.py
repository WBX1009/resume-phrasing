from typing import List

from backend.domain.entities.document import DocumentChunk
from backend.domain.interfaces.reranker import BaseReranker


class NoOpReranker(BaseReranker):
    """Reranker adapter that preserves first-stage retrieval order."""

    model_name = "noop"

    def rerank(
        self,
        query: str,
        documents: List[DocumentChunk],
        top_k: int,
    ) -> List[DocumentChunk]:
        final_docs: List[DocumentChunk] = []
        for rank, doc in enumerate(documents[:top_k], start=1):
            copied = doc.model_copy(deep=True)
            copied.metadata = {
                **copied.metadata,
                "rerank": {
                    "model": self.model_name,
                    "backend": "noop",
                    "rerank_rank": rank,
                },
            }
            final_docs.append(copied)
        return final_docs

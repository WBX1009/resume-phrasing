from abc import ABC, abstractmethod
from typing import List

from backend.domain.entities.document import DocumentChunk


class BaseReranker(ABC):
    """Port for reranking retrieved candidate chunks."""

    @abstractmethod
    def rerank(
        self,
        query: str,
        documents: List[DocumentChunk],
        top_k: int,
    ) -> List[DocumentChunk]:
        pass

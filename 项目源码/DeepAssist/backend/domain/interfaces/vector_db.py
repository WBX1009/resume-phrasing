from abc import ABC, abstractmethod
from typing import List

from backend.domain.entities.document import DocumentChunk
from backend.domain.entities.knowledge_base import KnowledgeBaseFile


class BaseVectorDB(ABC):
    """Abstract vector database port."""

    @abstractmethod
    def add_chunks(
        self,
        collection_name: str,
        chunks: List[DocumentChunk],
        embeddings: List[List[float]],
    ) -> bool:
        """Persist embedded document chunks."""
        pass

    @abstractmethod
    def search(
        self,
        collection_name: str,
        query_vector: List[float],
        top_k: int,
    ) -> List[DocumentChunk]:
        """Search chunks by vector similarity."""
        pass

    @abstractmethod
    def delete_by_source(self, collection_name: str, source_file: str) -> bool:
        """Delete chunks that belong to a single source file."""
        pass

    @abstractmethod
    def list_sources(self, collection_name: str) -> List[KnowledgeBaseFile]:
        """List source files represented in a collection."""
        pass

    @abstractmethod
    def list_collections(self) -> List[str]:
        """List available vector collections."""
        pass

from abc import ABC, abstractmethod
from typing import List

from backend.domain.entities.document import DocumentChunk
from backend.domain.entities.knowledge_base import KnowledgeBaseFile


class BaseKeywordDB(ABC):
    """Abstract keyword index port."""

    @abstractmethod
    def build_index(self, collection_name: str, chunks: List[DocumentChunk]) -> bool:
        """Build or update a keyword index for chunks."""
        pass

    @abstractmethod
    def search(
        self,
        collection_name: str,
        query_text: str,
        top_k: int,
    ) -> List[DocumentChunk]:
        """Search chunks by keyword text."""
        pass

    @abstractmethod
    def delete_by_source(self, collection_name: str, source_file: str) -> bool:
        """Delete indexed chunks that belong to a single source file."""
        pass

    @abstractmethod
    def list_sources(self, collection_name: str) -> List[KnowledgeBaseFile]:
        """List source files represented in an index."""
        pass

    @abstractmethod
    def list_collections(self) -> List[str]:
        """List available keyword collections."""
        pass

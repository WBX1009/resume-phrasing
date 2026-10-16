from abc import ABC, abstractmethod
from typing import Any


class BaseKnowledgeBaseCatalog(ABC):
    """Read-only knowledge-base inventory port used by agent tools."""

    @abstractmethod
    def list_collections(self) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def list_files(self, collection_name: str = "tech_docs_kb") -> dict[str, Any]:
        raise NotImplementedError

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

class BaseKnowledgeBaseManifestStore(ABC):
    """Abstract persistence interface for knowledge-base manifests."""

    @abstractmethod
    def list_collections(self) -> List[Dict[str, Any]]:
        """Return a summary list of collections."""
        raise NotImplementedError

    @abstractmethod
    def list_files(self, collection_name: str) -> List[Dict[str, Any]]:
        """Return a summary list of files for a collection."""
        raise NotImplementedError

    @abstractmethod
    def upsert_file(
        self,
        collection_name: str,
        source_file: str,
        chunk_count: int,
        metadata: Optional[Dict[str, Any]] = None,
        stores: Optional[List[str]] = None,
    ) -> None:
        """Insert or update a file entry."""
        raise NotImplementedError

    @abstractmethod
    def remove_file(self, collection_name: str, source_file: str) -> None:
        """Remove a file entry from a collection."""
        raise NotImplementedError

    @abstractmethod
    def replace_collection(
        self,
        collection_name: str,
        files: List[Dict[str, Any]],
        metadata: Optional[Dict[str, Any]] = None,
        stores: Optional[List[str]] = None,
    ) -> None:
        """Replace the manifest entry for a collection with new file entries."""
        raise NotImplementedError

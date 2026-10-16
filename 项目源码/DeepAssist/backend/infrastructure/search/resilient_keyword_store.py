"""Primary/secondary keyword-search composition for local and deployed runs."""

from __future__ import annotations

from typing import List

from backend.domain.entities.document import DocumentChunk
from backend.domain.entities.knowledge_base import KnowledgeBaseFile
from backend.domain.interfaces.keyword_db import BaseKeywordDB


class ResilientKeywordDB(BaseKeywordDB):
    """Use a deployed search adapter with a local Whoosh safety net.

    Elasticsearch remains the primary keyword index in a deployed environment.
    The local adapter is written during ingestion as well, so development and
    offline acceptance can still retrieve documents when Elasticsearch is not
    running.  Both adapters stay behind the same domain port.
    """

    def __init__(self, primary: BaseKeywordDB, secondary: BaseKeywordDB):
        self.primary = primary
        self.secondary = secondary

    def build_index(self, collection_name: str, chunks: List[DocumentChunk]) -> bool:
        primary_ok = self.primary.build_index(collection_name, chunks)
        secondary_ok = self.secondary.build_index(collection_name, chunks)
        return primary_ok or secondary_ok

    def search(self, collection_name: str, query_text: str, top_k: int) -> List[DocumentChunk]:
        primary_results = self.primary.search(collection_name, query_text, top_k)
        return primary_results or self.secondary.search(collection_name, query_text, top_k)

    def delete_by_source(self, collection_name: str, source_file: str) -> bool:
        primary_ok = self.primary.delete_by_source(collection_name, source_file)
        secondary_ok = self.secondary.delete_by_source(collection_name, source_file)
        return primary_ok and secondary_ok

    def list_sources(self, collection_name: str) -> List[KnowledgeBaseFile]:
        primary = self.primary.list_sources(collection_name)
        secondary = self.secondary.list_sources(collection_name)
        by_name = {item.source_file: item for item in secondary}
        by_name.update({item.source_file: item for item in primary})
        return sorted(by_name.values(), key=lambda item: item.source_file.lower())

    def list_collections(self) -> List[str]:
        return sorted(set(self.primary.list_collections()) | set(self.secondary.list_collections()))

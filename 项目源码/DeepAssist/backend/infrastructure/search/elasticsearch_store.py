import re
from typing import Any, Dict, List, Sequence

from backend.common.config import settings
from backend.common.logger import get_logger
from backend.domain.entities.document import DocumentChunk
from backend.domain.entities.knowledge_base import KnowledgeBaseFile
from backend.domain.interfaces.keyword_db import BaseKeywordDB

logger = get_logger(__name__)


class ElasticsearchKeywordSearchAdapter(BaseKeywordDB):
    """Elasticsearch/OpenSearch keyword retrieval adapter."""

    def __init__(
        self,
        hosts: str | Sequence[str] | None = None,
        index_prefix: str | None = None,
        username: str | None = None,
        password: str | None = None,
        client: Any | None = None,
    ):
        self.hosts = hosts or settings.ELASTICSEARCH_HOSTS
        self.index_prefix = index_prefix or settings.ELASTICSEARCH_INDEX_PREFIX
        self.username = username if username is not None else settings.ELASTICSEARCH_USERNAME
        self.password = password if password is not None else settings.ELASTICSEARCH_PASSWORD
        self.client = client or self._build_client()

    def build_index(self, collection_name: str, chunks: List[DocumentChunk]) -> bool:
        index_name = self._index_name(collection_name)
        try:
            self._ensure_index(index_name)
            for chunk in chunks:
                metadata = dict(chunk.metadata or {})
                self.client.index(
                    index=index_name,
                    id=chunk.id,
                    document={
                        "id": chunk.id,
                        "collection_name": collection_name,
                        "source_file": metadata.get("source_file") or metadata.get("source") or "unknown",
                        "content": chunk.content,
                        "metadata": metadata,
                    },
                )
            refresh = getattr(getattr(self.client, "indices", None), "refresh", None)
            if callable(refresh):
                refresh(index=index_name)
            return True
        except Exception as exc:
            logger.error("Elasticsearch indexing failed for %s: %s", collection_name, exc)
            return False

    def search(
        self,
        collection_name: str,
        query_text: str,
        top_k: int,
    ) -> List[DocumentChunk]:
        index_name = self._index_name(collection_name)
        try:
            response = self.client.search(
                index=index_name,
                size=max(1, top_k),
                query={
                    "multi_match": {
                        "query": query_text,
                        "fields": ["content^3", "source_file", "metadata.title", "metadata.section"],
                        "type": "best_fields",
                    }
                },
            )
            hits = response.get("hits", {}).get("hits", [])
            return [self._hit_to_chunk(hit) for hit in hits]
        except Exception as exc:
            logger.error("Elasticsearch search failed for %s: %s", collection_name, exc)
            return []

    def delete_by_source(self, collection_name: str, source_file: str) -> bool:
        index_name = self._index_name(collection_name)
        try:
            self.client.delete_by_query(
                index=index_name,
                query={"term": {"source_file.keyword": source_file}},
                conflicts="proceed",
                refresh=True,
            )
            return True
        except Exception as exc:
            logger.error("Elasticsearch delete_by_source failed for %s/%s: %s", collection_name, source_file, exc)
            return False

    def list_sources(self, collection_name: str) -> List[KnowledgeBaseFile]:
        index_name = self._index_name(collection_name)
        try:
            response = self.client.search(
                index=index_name,
                size=0,
                aggs={
                    "sources": {
                        "terms": {
                            "field": "source_file.keyword",
                            "size": 10000,
                        }
                    }
                },
            )
            buckets = response.get("aggregations", {}).get("sources", {}).get("buckets", [])
            return [
                KnowledgeBaseFile(
                    source_file=str(bucket.get("key") or ""),
                    chunk_count=int(bucket.get("doc_count") or 0),
                    metadata={"keyword_backend": "elasticsearch"},
                )
                for bucket in buckets
                if bucket.get("key")
            ]
        except Exception as exc:
            logger.error("Elasticsearch list_sources failed for %s: %s", collection_name, exc)
            return []

    def list_collections(self) -> List[str]:
        try:
            index_pattern = f"{self._sanitize(self.index_prefix)}-*"
            response = self.client.indices.get(index=index_pattern, allow_no_indices=True)
            prefix = f"{self._sanitize(self.index_prefix)}-"
            names = []
            for index_name in response.keys():
                if index_name.startswith(prefix):
                    names.append(index_name[len(prefix):])
            return sorted(names)
        except Exception as exc:
            logger.error("Elasticsearch list_collections failed: %s", exc)
            return []

    def _build_client(self) -> Any:
        try:
            from elasticsearch import Elasticsearch
        except ImportError as exc:
            raise RuntimeError(
                "elasticsearch is required when KEYWORD_BACKEND=elasticsearch. Install requirements.txt first."
            ) from exc

        hosts = self.hosts
        if isinstance(hosts, str):
            hosts = [item.strip() for item in hosts.split(",") if item.strip()]
        kwargs: Dict[str, Any] = {"hosts": hosts}
        if self.username:
            kwargs["basic_auth"] = (self.username, self.password or "")
        return Elasticsearch(**kwargs)

    def _ensure_index(self, index_name: str) -> None:
        if self.client.indices.exists(index=index_name):
            return
        self.client.indices.create(
            index=index_name,
            mappings={
                "properties": {
                    "id": {"type": "keyword"},
                    "collection_name": {"type": "keyword"},
                    "source_file": {"type": "keyword"},
                    "content": {"type": "text"},
                    "metadata": {"type": "object", "enabled": True},
                }
            },
        )

    def _hit_to_chunk(self, hit: Dict[str, Any]) -> DocumentChunk:
        source = hit.get("_source", {}) or {}
        metadata = dict(source.get("metadata") or {})
        metadata.setdefault("source_file", source.get("source_file"))
        metadata["keyword_backend"] = "elasticsearch"
        return DocumentChunk(
            id=str(source.get("id") or hit.get("_id")),
            content=str(source.get("content") or ""),
            metadata=metadata,
            score=float(hit.get("_score") or 0.0),
        )

    def _index_name(self, collection_name: str) -> str:
        return f"{self._sanitize(self.index_prefix)}-{self._sanitize(collection_name)}"

    @staticmethod
    def _sanitize(value: str) -> str:
        normalized = re.sub(r"[^a-zA-Z0-9_-]+", "-", value.strip().lower())
        normalized = normalized.strip("-")
        return normalized or "default"

import json
import os
from typing import Any, Dict, List

import chromadb

from backend.common.config import settings
from backend.common.logger import get_logger
from backend.domain.entities.document import DocumentChunk
from backend.domain.entities.knowledge_base import KnowledgeBaseFile
from backend.domain.interfaces.vector_db import BaseVectorDB

logger = get_logger(__name__)


class ChromaStore(BaseVectorDB):
    """Local Chroma vector-store adapter."""

    def __init__(self):
        db_path = settings.VECTOR_DB_PATH
        os.makedirs(db_path, exist_ok=True)
        self.client = chromadb.PersistentClient(path=db_path)
        logger.info("ChromaDB persistent client initialized: %s", db_path)

    def add_chunks(
        self,
        collection_name: str,
        chunks: List[DocumentChunk],
        embeddings: List[List[float]],
    ) -> bool:
        try:
            collection = self.client.get_or_create_collection(name=collection_name)
            collection.add(
                ids=[chunk.id for chunk in chunks],
                documents=[chunk.content for chunk in chunks],
                embeddings=embeddings,
                metadatas=[self._sanitize_metadata(chunk.metadata) for chunk in chunks],
            )
            logger.info("Chroma collection %s wrote %s vectors", collection_name, len(chunks))
            return True
        except Exception as exc:
            logger.error("ChromaDB add_chunks failed for %s: %s", collection_name, exc)
            return False

    def search(
        self,
        collection_name: str,
        query_vector: List[float],
        top_k: int,
    ) -> List[DocumentChunk]:
        try:
            collection = self.client.get_collection(name=collection_name)
            results = collection.query(
                query_embeddings=[query_vector],
                n_results=top_k,
                include=["documents", "metadatas", "distances"],
            )
            chunks: List[DocumentChunk] = []
            ids = results.get("ids") or []
            if ids and ids[0]:
                for index, chunk_id in enumerate(ids[0]):
                    distance = results["distances"][0][index]
                    chunks.append(
                        DocumentChunk(
                            id=chunk_id,
                            content=results["documents"][0][index],
                            metadata=results["metadatas"][0][index] or {},
                            score=1.0 - distance,
                        )
                    )
            return chunks
        except Exception as exc:
            logger.error("ChromaDB search failed for %s: %s", collection_name, exc)
            return []

    def delete_by_source(self, collection_name: str, source_file: str) -> bool:
        try:
            try:
                collection = self.client.get_collection(name=collection_name)
            except Exception:
                return True

            collection.delete(where={"source_file": source_file})
            try:
                collection.delete(where={"source": source_file})
            except Exception:
                pass
            logger.info("Chroma collection %s deleted source %s", collection_name, source_file)
            return True
        except Exception as exc:
            logger.error("ChromaDB delete_by_source failed for %s/%s: %s", collection_name, source_file, exc)
            return False

    def list_sources(self, collection_name: str) -> List[KnowledgeBaseFile]:
        try:
            try:
                collection = self.client.get_collection(name=collection_name)
            except Exception:
                return []

            source_map: Dict[str, KnowledgeBaseFile] = {}
            batch_size = 500
            offset = 0
            total = self._safe_count(collection)

            while True:
                results = collection.get(
                    include=["metadatas"],
                    limit=batch_size,
                    offset=offset,
                )
                metadatas = results.get("metadatas") or []
                if not metadatas:
                    break

                for metadata in metadatas:
                    metadata = metadata or {}
                    source_file = metadata.get("source_file") or metadata.get("source")
                    if not source_file:
                        continue
                    current = source_map.get(source_file)
                    if current is None:
                        source_map[source_file] = KnowledgeBaseFile(
                            source_file=source_file,
                            chunk_count=1,
                            metadata={"first_chunk_metadata": metadata},
                        )
                    else:
                        source_map[source_file] = current.model_copy(
                            update={"chunk_count": current.chunk_count + 1}
                        )

                offset += len(metadatas)
                if total is not None and offset >= total:
                    break

            return sorted(source_map.values(), key=lambda item: item.source_file.lower())
        except Exception as exc:
            logger.error("ChromaDB list_sources failed for %s: %s", collection_name, exc)
            return []

    def list_collections(self) -> List[str]:
        try:
            collections = self.client.list_collections()
            names = [
                collection if isinstance(collection, str) else collection.name
                for collection in collections
            ]
            return sorted(set(names))
        except Exception as exc:
            logger.error("ChromaDB list_collections failed: %s", exc)
            return []

    def _sanitize_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        sanitized: Dict[str, Any] = {}
        for key, value in (metadata or {}).items():
            if value is None:
                continue
            if isinstance(value, (str, int, float, bool)):
                sanitized[key] = value
            elif isinstance(value, (list, dict)):
                sanitized[key] = json.dumps(value, ensure_ascii=False)
            else:
                sanitized[key] = str(value)
        return sanitized

    def _safe_count(self, collection: Any) -> int | None:
        try:
            return int(collection.count())
        except Exception:
            return None

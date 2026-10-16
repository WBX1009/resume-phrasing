from datetime import datetime, timezone
from typing import Any, List

from backend.common.logger import get_logger
from backend.common.config import settings
from backend.domain.entities.document import DocumentChunk
from backend.domain.entities.knowledge_base import (
    KnowledgeBaseCollectionHealth,
    KnowledgeBaseFile,
    KnowledgeBaseHealthReport,
)
from backend.domain.interfaces.embedding import BaseEmbedding
from backend.domain.interfaces.kb_catalog import BaseKnowledgeBaseCatalog
from backend.domain.interfaces.keyword_db import BaseKeywordDB
from backend.domain.interfaces.vector_db import BaseVectorDB
from backend.domain.interfaces.kb_manifest_store import BaseKnowledgeBaseManifestStore
from backend.domain.services.rag.chunking import DocumentChunker

logger = get_logger(__name__)


class _EmptyKnowledgeBaseManifestStore(BaseKnowledgeBaseManifestStore):
    def list_collections(self) -> list[dict[str, Any]]:
        return []

    def list_files(self, collection_name: str) -> list[dict[str, Any]]:
        return []

    def upsert_file(
        self,
        collection_name: str,
        source_file: str,
        chunk_count: int,
        metadata: dict[str, Any] | None = None,
        stores: list[str] | None = None,
    ) -> None:
        return None

    def remove_file(self, collection_name: str, source_file: str) -> None:
        return None

    def replace_collection(
        self,
        collection_name: str,
        files: list[dict[str, Any]],
        metadata: dict[str, Any] | None = None,
        stores: list[str] | None = None,
    ) -> None:
        return None


class KnowledgeBaseApp(BaseKnowledgeBaseCatalog):
    """Knowledge-base ingestion workflow orchestrator."""

    def __init__(
        self,
        chunker: DocumentChunker,
        embedding_model: BaseEmbedding | None,
        vector_db: BaseVectorDB | None,
        keyword_db: BaseKeywordDB | None,
        manifest_store: BaseKnowledgeBaseManifestStore | None = None,
        health_inspector=None,
        langchain_ingestion=None,
    ):
        self.chunker = chunker
        self.embedding = embedding_model
        self.vector_db = vector_db
        self.keyword_db = keyword_db
        # Manifest store injected via domain interface
        self.manifest_store = manifest_store or _EmptyKnowledgeBaseManifestStore()
        self.health_inspector = health_inspector
        self.langchain_ingestion = langchain_ingestion

    def _rollback_vector_write(self, collection_name: str, file_name: str) -> bool:
        if self.vector_db is None:
            logger.error("Vector rollback skipped because vector DB is unavailable")
            return False
        logger.warning("Starting vector rollback for %s", file_name)
        try:
            rollback_success = self.vector_db.delete_by_source(collection_name, file_name)
        except Exception as exc:
            logger.error("Vector rollback raised for %s: %s", file_name, exc)
            return False
        if rollback_success:
            logger.info("Vector rollback completed for %s", file_name)
        else:
            logger.error("Vector rollback failed for %s", file_name)
        return rollback_success

    def _is_ready_for_ingestion(self) -> bool:
        return bool(self.embedding and self.vector_db and self.keyword_db)

    def list_files(self, collection_name: str = "tech_docs_kb") -> dict:
        """Return file-level KB inventory merged from vector and keyword stores."""
        manifest_files = self.manifest_store.list_files(collection_name)
        if manifest_files:
            return {
                "status": "success",
                "collection_name": collection_name,
                "data": manifest_files,
                "errors": {},
                "source": "manifest",
            }

        records: dict[str, dict] = {}
        errors: dict[str, str] = {}

        if self.vector_db is not None:
            try:
                for item in self.vector_db.list_sources(collection_name):
                    records.setdefault(item.source_file, self._new_file_record(item))
                    records[item.source_file]["vector_chunk_count"] = item.chunk_count
                    records[item.source_file]["stores"].append("vector")
            except Exception as exc:
                logger.error("Failed to list vector KB files: %s", exc)
                errors["vector"] = str(exc)
        else:
            errors["vector"] = "Vector database is unavailable"

        if self.keyword_db is not None:
            try:
                for item in self.keyword_db.list_sources(collection_name):
                    records.setdefault(item.source_file, self._new_file_record(item))
                    records[item.source_file]["keyword_chunk_count"] = item.chunk_count
                    records[item.source_file]["stores"].append("keyword")
            except Exception as exc:
                logger.error("Failed to list keyword KB files: %s", exc)
                errors["keyword"] = str(exc)
        else:
            errors["keyword"] = "Keyword database is unavailable"

        files = []
        for record in records.values():
            vector_count = record["vector_chunk_count"]
            keyword_count = record["keyword_chunk_count"]
            record["chunk_count"] = max(vector_count, keyword_count)
            record["consistent"] = vector_count == keyword_count
            record["stores"] = sorted(set(record["stores"]))
            files.append(record)

        files.sort(key=lambda item: item["source_file"].lower())
        status = "success" if not errors else ("partial_success" if files else "error")
        if files:
            self.manifest_store.replace_collection(
                collection_name=collection_name,
                files=files,
                stores=sorted(
                    {
                        store
                        for item in files
                        if isinstance(item, dict)
                        for store in item.get("stores", [])
                    }
                )
                or ["vector", "keyword"],
            )
        return {
            "status": status,
            "collection_name": collection_name,
            "data": files,
            "errors": errors,
            "source": "live_scan",
        }

    def list_collections(self) -> dict:
        """Return collection-level inventory merged from vector and keyword stores."""
        records: dict[str, dict] = {}
        errors: dict[str, str] = {}
        manifest_items = {
            item.get("collection_name"): item
            for item in self.manifest_store.list_collections()
            if isinstance(item, dict) and item.get("collection_name")
        }

        if self.vector_db is not None:
            try:
                for name in self.vector_db.list_collections():
                    records.setdefault(name, {"collection_name": name, "stores": []})
                    records[name]["stores"].append("vector")
            except Exception as exc:
                logger.error("Failed to list vector collections: %s", exc)
                errors["vector"] = str(exc)
        else:
            errors["vector"] = "Vector database is unavailable"

        if self.keyword_db is not None:
            try:
                for name in self.keyword_db.list_collections():
                    records.setdefault(name, {"collection_name": name, "stores": []})
                    records[name]["stores"].append("keyword")
            except Exception as exc:
                logger.error("Failed to list keyword collections: %s", exc)
                errors["keyword"] = str(exc)
        else:
            errors["keyword"] = "Keyword database is unavailable"

        collections = []
        for record in records.values():
            name = record["collection_name"]
            record["stores"] = sorted(set(record["stores"]))
            manifest_item = manifest_items.get(name, {})
            record["file_count"] = int(manifest_item.get("file_count", 0) or 0)
            record["chunk_count"] = int(manifest_item.get("chunk_count", 0) or 0)
            record["consistent"] = bool(
                manifest_item.get(
                    "consistent",
                    "vector" in record["stores"] and "keyword" in record["stores"],
                )
            )
            record["metadata"] = manifest_item.get("metadata", {})
            collections.append(record)

        collections.sort(key=lambda item: item["collection_name"].lower())
        status = "success" if not errors else ("partial_success" if collections else "error")
        return {
            "status": status,
            "data": collections,
            "errors": errors,
            "source": "manifest_summary",
        }

    def get_health_report(
        self,
        refresh: bool = False,
        collections: list[str] | None = None,
        repair: bool = False,
        batch_size: int = 256,
    ) -> dict:
        """Return cached or live knowledge-base health data for UI/API consumers."""
        try:
            if self.health_inspector is not None:
                if refresh:
                    report = self.health_inspector.inspect(
                        collections=collections,
                        repair=repair,
                        batch_size=batch_size,
                        persist=True,
                    )
                    return {
                        "status": "success",
                        "source": "live",
                        "data": self._health_report_to_dict(report),
                    }

                load_report = getattr(self.health_inspector, "load_report", None)
                if callable(load_report):
                    cached = load_report()
                    if cached is not None:
                        return {
                            "status": "success",
                            "source": "cached",
                            "data": self._health_report_to_dict(cached),
                        }

                report = self.health_inspector.inspect(
                    collections=collections,
                    repair=repair,
                    batch_size=batch_size,
                    persist=True,
                )
                return {
                    "status": "success",
                    "source": "live",
                    "data": self._health_report_to_dict(report),
                }

            return {
                "status": "success",
                "source": "derived",
                "data": self._derive_health_report(collections).model_dump(),
            }
        except Exception as exc:
            logger.error("Knowledge-base health check failed: %s", exc)
            return {"status": "error", "message": str(exc), "data": {}}

    def delete_file(
        self,
        file_name: str,
        collection_name: str = "tech_docs_kb",
    ) -> dict:
        """Delete a source file from both Chroma and Whoosh indexes."""
        source_file = (file_name or "").strip()
        if not source_file:
            return {"status": "error", "message": "source_file is required"}

        logger.warning("Deleting KB source [%s] from collection [%s]", source_file, collection_name)

        vector_success = False
        keyword_success = False
        errors: dict[str, str] = {}

        if self.vector_db is not None:
            try:
                vector_success = self.vector_db.delete_by_source(collection_name, source_file)
                if not vector_success:
                    errors["vector"] = "Vector database delete returned false"
            except Exception as exc:
                logger.error("Vector delete failed for %s: %s", source_file, exc)
                errors["vector"] = str(exc)
        else:
            errors["vector"] = "Vector database is unavailable"

        if self.keyword_db is not None:
            try:
                keyword_success = self.keyword_db.delete_by_source(collection_name, source_file)
                if not keyword_success:
                    errors["keyword"] = "Keyword delete returned false"
            except Exception as exc:
                logger.error("Keyword delete failed for %s: %s", source_file, exc)
                errors["keyword"] = str(exc)
        else:
            errors["keyword"] = "Keyword database is unavailable"

        if vector_success and keyword_success:
            self.manifest_store.remove_file(collection_name, source_file)
            logger.info("KB source [%s] deleted from both stores", source_file)
            return {
                "status": "success",
                "message": f"Deleted {source_file}",
                "data": {
                    "source_file": source_file,
                    "vector_deleted": True,
                    "keyword_deleted": True,
                },
            }

        logger.error("KB source delete incomplete for %s: %s", source_file, errors)
        return {
            "status": "error",
            "message": "Knowledge-base delete was incomplete",
            "data": {
                "source_file": source_file,
                "vector_deleted": vector_success,
                "keyword_deleted": keyword_success,
            },
            "errors": errors,
        }

    def _new_file_record(self, item: KnowledgeBaseFile) -> dict:
        return {
            "source_file": item.source_file,
            "chunk_count": item.chunk_count,
            "vector_chunk_count": 0,
            "keyword_chunk_count": 0,
            "consistent": False,
            "stores": [],
            "metadata": item.metadata,
        }

    def _health_report_to_dict(self, report) -> dict:
        if hasattr(report, "model_dump"):
            return report.model_dump()
        if isinstance(report, dict):
            return report
        return {}

    def _derive_health_report(self, collections: list[str] | None = None) -> KnowledgeBaseHealthReport:
        collection_payload = self.list_collections()
        collection_items = collection_payload.get("data", [])
        requested = {item for item in (collections or []) if item}
        health_items: list[KnowledgeBaseCollectionHealth] = []
        iterable_collections = collection_items if isinstance(collection_items, list) else []
        for item in iterable_collections:
            if not isinstance(item, dict):
                continue
            collection_name = str(item.get("collection_name") or "").strip()
            if not collection_name or (requested and collection_name not in requested):
                continue
            stores = set(item.get("stores") or [])
            chunk_count = int(item.get("chunk_count", 0) or 0)
            errors = []
            if self.vector_db is None:
                errors.append("vector_db_unavailable")
            if self.keyword_db is None:
                errors.append("keyword_db_unavailable")
            if "vector" not in stores:
                errors.append("vector_store_missing")
            if "keyword" not in stores:
                errors.append("keyword_store_missing")
            health_items.append(
                KnowledgeBaseCollectionHealth(
                    collection_name=collection_name,
                    count=chunk_count,
                    whoosh_docs=chunk_count if "keyword" in stores else None,
                    get_ok="vector" in stores,
                    query_ok="vector" in stores,
                    healthy=not errors and bool(item.get("consistent", True)),
                    errors=errors,
                )
            )

        return KnowledgeBaseHealthReport(
            vector_db_path=settings.VECTOR_DB_PATH,
            keyword_db_path=settings.KEYWORD_DB_PATH,
            checked_at=datetime.now(timezone.utc).isoformat(),
            collections=health_items,
        )

    def process_document(
        self,
        file_name: str,
        content: str,
        collection_name: str = "tech_docs_kb",
    ) -> dict:
        logger.info("Received knowledge-base ingestion request: %s", file_name)

        if not self._is_ready_for_ingestion():
            logger.error("Knowledge-base ingestion rejected because dependencies are unavailable")
            return {
                "status": "error",
                "message": "Knowledge-base ingestion dependencies are unavailable",
            }

        if self.langchain_ingestion is not None:
            chunks = self.langchain_ingestion.load_and_split(content, source_name=file_name)
        else:
            chunks = self.chunker.split_markdown(content, source_name=file_name)
        if not chunks:
            return {"status": "error", "message": "Document is empty or chunking failed"}

        try:
            texts = [chunk.content for chunk in chunks]
            embeddings = self.embedding.embed_documents(texts)
        except Exception as exc:
            logger.error("Embedding failed for %s: %s", file_name, exc)
            return {"status": "error", "message": f"Embedding failed: {exc}"}

        v_success = False
        try:
            v_success = self.vector_db.add_chunks(collection_name, chunks, embeddings)
            if not v_success:
                logger.error("Vector write failed for %s", file_name)
                return {"status": "error", "message": "Vector database write failed"}

            k_success = self.keyword_db.build_index(collection_name, chunks)
            if k_success:
                first_meta = chunks[0].metadata if chunks else {}
                self.manifest_store.upsert_file(
                    collection_name=collection_name,
                    source_file=file_name,
                    chunk_count=len(chunks),
                    metadata={
                        "collection_name": collection_name,
                        "source_format": "markdown",
                        "source_type": "markdown",
                        "language": first_meta.get("language", "unknown"),
                        "title_path": first_meta.get("heading_path"),
                    },
                )
                logger.info("Document %s ingested successfully with %s chunks", file_name, len(chunks))
                return {"status": "success", "message": f"Successfully processed {len(chunks)} chunks"}

            logger.error("Keyword index write failed for %s; starting vector rollback", file_name)
            rollback_success = self._rollback_vector_write(collection_name, file_name)

            return {
                "status": "error",
                "message": "Keyword index write failed; vector rollback completed"
                if rollback_success
                else "Keyword index write failed; vector rollback failed",
            }

        except Exception as exc:
            logger.error("Knowledge-base ingestion failed for %s: %s", file_name, exc)
            if v_success:
                logger.warning("Exception after vector write for %s; starting vector rollback", file_name)
                self._rollback_vector_write(collection_name, file_name)

            return {"status": "error", "message": f"Database write failed: {exc}"}

    def process_chunks(
        self,
        collection_name: str,
        chunks: List[DocumentChunk],
    ) -> dict:
        """
        Ingest a batch of pre-chunked DocumentChunk objects into the specified collection.

        This method computes embeddings for the provided chunks, writes them to the vector store
        and keyword index, and handles rollback on failure. It returns a status dict describing
        the outcome. Manifest updates are not performed here and should be handled externally
        by the caller when batching across multiple calls.
        """
        logger.info(
            "Received batch ingestion request: %s chunks into collection %s",
            len(chunks),
            collection_name,
        )

        if not self._is_ready_for_ingestion():
            logger.error("Batch ingestion rejected because dependencies are unavailable")
            return {
                "status": "error",
                "message": "Knowledge-base ingestion dependencies are unavailable",
            }

        if not chunks:
            return {"status": "error", "message": "Empty batch"}

        try:
            texts = [chunk.content for chunk in chunks]
            embeddings = self.embedding.embed_documents(texts)
        except Exception as exc:
            logger.error("Embedding failed for batch: %s", exc)
            return {"status": "error", "message": f"Embedding failed: {exc}"}

        v_success = False
        try:
            v_success = self.vector_db.add_chunks(collection_name, chunks, embeddings)
            if not v_success:
                logger.error("Vector database write failed for batch")
                return {"status": "error", "message": "Vector database write failed"}

            k_success = self.keyword_db.build_index(collection_name, chunks)
            if k_success:
                return {"status": "success", "count": len(chunks)}

            logger.error("Keyword index write failed for batch; starting vector rollback")
            rollback_sources = set(
                chunk.metadata.get("source_file")
                or chunk.metadata.get("source_path")
                for chunk in chunks
            )
            for source_file in rollback_sources:
                if source_file:
                    self._rollback_vector_write(collection_name, source_file)

            return {
                "status": "error",
                "message": "Keyword index write failed; vector rollback completed",
            }

        except Exception as exc:
            logger.error("Batch ingestion failed: %s", exc)
            if v_success:
                rollback_sources = set(
                    chunk.metadata.get("source_file")
                    or chunk.metadata.get("source_path")
                    for chunk in chunks
                )
                for source_file in rollback_sources:
                    if source_file:
                        self._rollback_vector_write(collection_name, source_file)

            return {"status": "error", "message": f"Database write failed: {exc}"}

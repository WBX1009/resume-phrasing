import json
import os
import re
from typing import Any, Dict, List

from jieba.analyse import ChineseAnalyzer
from whoosh.fields import ID, STORED, TEXT, Schema
from whoosh.index import create_in, exists_in, open_dir
from whoosh.qparser import OrGroup, QueryParser

from backend.common.config import settings
from backend.common.logger import get_logger
from backend.domain.entities.document import DocumentChunk
from backend.domain.entities.knowledge_base import KnowledgeBaseFile
from backend.domain.interfaces.keyword_db import BaseKeywordDB

logger = get_logger(__name__)


class WhooshStore(BaseKeywordDB):
    """Local Whoosh keyword index adapter used by default."""

    def __init__(self):
        self.base_path = settings.KEYWORD_DB_PATH
        os.makedirs(self.base_path, exist_ok=True)
        self.analyzer = ChineseAnalyzer()
        self.schema = Schema(
            id=ID(stored=True, unique=True),
            source_file=ID(stored=True),
            content=TEXT(stored=True, analyzer=self.analyzer),
            metadata=STORED(),
        )
        self._index_cache: Dict[str, Any] = {}
        logger.info("Whoosh keyword index initialized: %s", self.base_path)

    def _get_collection_dir(self, collection_name: str) -> str:
        collection_dir = os.path.join(self.base_path, collection_name)
        os.makedirs(collection_dir, exist_ok=True)
        return collection_dir

    def _get_or_open_index(self, collection_name: str):
        if collection_name in self._index_cache:
            return self._index_cache[collection_name]

        collection_dir = self._get_collection_dir(collection_name)
        if exists_in(collection_dir):
            index = open_dir(collection_dir)
            self._index_cache[collection_name] = index
            return index
        return None

    def build_index(self, collection_name: str, chunks: List[DocumentChunk]) -> bool:
        collection_dir = self._get_collection_dir(collection_name)
        try:
            index = open_dir(collection_dir) if exists_in(collection_dir) else create_in(collection_dir, self.schema)
            writer = index.writer(limitmb=1024, multisegment=True)
            try:
                for chunk in chunks:
                    writer.update_document(
                        id=chunk.id,
                        source_file=chunk.metadata.get("source_file", "unknown"),
                        content=chunk.content,
                        metadata=json.dumps(chunk.metadata, ensure_ascii=False),
                    )
                writer.commit()
            except Exception:
                writer.cancel()
                raise

            self._index_cache[collection_name] = index
            logger.info("Whoosh index %s wrote %s chunks", collection_name, len(chunks))
            return True
        except Exception as exc:
            logger.error("Whoosh indexing failed for %s: %s", collection_name, exc)
            return False

    def search(self, collection_name: str, query_text: str, top_k: int) -> List[DocumentChunk]:
        index = self._get_or_open_index(collection_name)
        if not index:
            logger.warning("Whoosh index %s does not exist or has not been built", collection_name)
            return []

        try:
            chunks: List[DocumentChunk] = []
            clean_query = re.sub(r"[^\w\u4e00-\u9fa5]+", " ", query_text)
            with index.searcher() as searcher:
                parser = QueryParser("content", index.schema, group=OrGroup.factory(0.9))
                query = parser.parse(clean_query)
                results = searcher.search(query, limit=top_k)
                for hit in results:
                    metadata = json.loads(hit["metadata"])
                    metadata["keyword_backend"] = "whoosh"
                    chunks.append(
                        DocumentChunk(
                            id=hit["id"],
                            content=hit["content"],
                            metadata=metadata,
                            score=hit.score,
                        )
                    )
            return chunks
        except Exception as exc:
            logger.error("Whoosh search failed for %s: %s", collection_name, exc)
            return []

    def delete_by_source(self, collection_name: str, source_file: str) -> bool:
        collection_dir = self._get_collection_dir(collection_name)
        if not exists_in(collection_dir):
            return True

        try:
            index = self._get_or_open_index(collection_name)
            if not index:
                return True
            writer = index.writer()
            writer.delete_by_term("source_file", source_file)
            writer.commit()
            self._index_cache[collection_name] = index
            logger.info("Whoosh index %s deleted source %s", collection_name, source_file)
            return True
        except Exception as exc:
            logger.error("Whoosh delete_by_source failed for %s/%s: %s", collection_name, source_file, exc)
            return False

    def list_sources(self, collection_name: str) -> List[KnowledgeBaseFile]:
        collection_dir = self._get_collection_dir(collection_name)
        if not exists_in(collection_dir):
            return []

        try:
            index = self._get_or_open_index(collection_name)
            if not index:
                return []

            source_map: Dict[str, KnowledgeBaseFile] = {}
            with index.searcher() as searcher:
                for fields in searcher.all_stored_fields():
                    metadata = self._load_metadata(fields.get("metadata"))
                    source_file = fields.get("source_file") or metadata.get("source_file") or metadata.get("source")
                    if not source_file:
                        continue
                    current = source_map.get(source_file)
                    if current is None:
                        source_map[source_file] = KnowledgeBaseFile(
                            source_file=source_file,
                            chunk_count=1,
                            metadata={"first_chunk_metadata": metadata, "keyword_backend": "whoosh"},
                        )
                    else:
                        source_map[source_file] = current.model_copy(
                            update={"chunk_count": current.chunk_count + 1}
                        )
            return sorted(source_map.values(), key=lambda item: item.source_file.lower())
        except Exception as exc:
            logger.error("Whoosh list_sources failed for %s: %s", collection_name, exc)
            return []

    def list_collections(self) -> List[str]:
        try:
            names = []
            for entry in os.listdir(self.base_path):
                collection_dir = os.path.join(self.base_path, entry)
                if os.path.isdir(collection_dir) and exists_in(collection_dir):
                    names.append(entry)
            return sorted(set(names))
        except Exception as exc:
            logger.error("Whoosh list_collections failed: %s", exc)
            return []

    def _load_metadata(self, raw: str | None) -> Dict[str, Any]:
        try:
            return json.loads(raw or "{}")
        except Exception:
            return {}

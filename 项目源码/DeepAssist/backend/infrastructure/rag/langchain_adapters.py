"""LangChain boundary adapters for the project-owned RAG and KB ports.

The core RAG pipeline stays framework-neutral.  These adapters let real
LangChain loaders, splitters, vector stores, retrievers, and runnables
participate at the infrastructure boundary without leaking framework
document types into the domain layer.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping, Sequence
import asyncio
from typing import Any

from backend.domain.entities.document import DocumentChunk
from backend.domain.entities.knowledge_base import KnowledgeBaseFile
from backend.domain.interfaces.vector_db import BaseVectorDB


class LangChainRetrieverAdapter:
    """Normalize LangChain ``BaseRetriever``-style results to ``DocumentChunk``."""

    def __init__(self, retriever: Any):
        if not callable(getattr(retriever, "invoke", None)):
            raise TypeError("retriever must expose LangChain's invoke(query, ...) method")
        self.retriever = retriever

    def invoke(self, query: str, *, config: Mapping[str, Any] | None = None) -> list[DocumentChunk]:
        try:
            raw_documents = self.retriever.invoke(query, config=config)
        except TypeError:
            raw_documents = self.retriever.invoke(query)
        if raw_documents is None:
            return []
        if isinstance(raw_documents, Mapping):
            raw_documents = raw_documents.get("documents", [])
        if isinstance(raw_documents, (str, bytes)) or not isinstance(raw_documents, Iterable):
            raise TypeError("LangChain retriever output must be an iterable of documents")
        return [_normalize_document(document, index) for index, document in enumerate(raw_documents)]

    def batch(
        self,
        queries: Sequence[str],
        *,
        config: Mapping[str, Any] | None = None,
    ) -> list[list[DocumentChunk]]:
        return [self.invoke(query, config=config) for query in queries]

    def _normalize(self, document: Any, index: int) -> DocumentChunk:
        return _normalize_document(document, index)


class LangChainRunnableAdapter:
    """Expose a LangChain ``Runnable`` through a narrow invoke/stream port."""

    def __init__(self, runnable: Any):
        if not callable(getattr(runnable, "invoke", None)):
            raise TypeError("runnable must expose LangChain's invoke(input, ...) method")
        self.runnable = runnable

    def invoke(self, value: Any, *, config: Mapping[str, Any] | None = None) -> Any:
        try:
            return self.runnable.invoke(value, config=config)
        except TypeError:
            return self.runnable.invoke(value)

    def stream(self, value: Any, *, config: Mapping[str, Any] | None = None) -> Iterator[Any]:
        stream = getattr(self.runnable, "stream", None)
        if not callable(stream):
            yield self.invoke(value, config=config)
            return
        try:
            yield from stream(value, config=config)
        except TypeError:
            yield from stream(value)


class LangChainDocumentLoaderAdapter:
    """Adapt a LangChain document loader without leaking its document type."""

    def __init__(self, loader: Any):
        if not callable(getattr(loader, "load", None)):
            raise TypeError("loader must expose LangChain's load() method")
        self.loader = loader

    def load(self, source: Any = None) -> list[Any]:
        if source is None:
            documents = self.loader.load()
        else:
            try:
                documents = self.loader.load(source)
            except TypeError:
                # Standard LangChain loaders usually capture their source at
                # construction time and therefore expose load() without args.
                documents = self.loader.load()
        if documents is None:
            return []
        if isinstance(documents, (str, bytes)) or not isinstance(documents, Iterable):
            raise TypeError("LangChain loader output must be an iterable of documents")
        return list(documents)

    async def aload(self, source: Any = None) -> list[Any]:
        aload = getattr(self.loader, "aload", None)
        if callable(aload):
            documents = await aload() if source is None else await aload(source)
            return list(documents or [])
        return await asyncio.to_thread(self.load, source)


class LangChainTextSplitterAdapter:
    """Adapt LangChain splitters to project-owned ``DocumentChunk`` values."""

    def __init__(self, splitter: Any):
        if not callable(getattr(splitter, "split_documents", None)) and not callable(
            getattr(splitter, "split_text", None)
        ):
            raise TypeError("splitter must expose split_documents() or split_text()")
        self.splitter = splitter

    def split_documents(
        self,
        documents: Sequence[Any],
        *,
        source_name: str | None = None,
    ) -> list[DocumentChunk]:
        source_documents = list(documents)
        split_documents = getattr(self.splitter, "split_documents", None)
        if callable(split_documents):
            raw_chunks = split_documents(source_documents)
        else:
            raw_chunks = []
            split_text = self.splitter.split_text
            for document in source_documents:
                original = _normalize_document(document, len(raw_chunks))
                for piece in split_text(original.content):
                    raw_chunks.append(
                        {
                            "content": piece,
                            "metadata": dict(original.metadata),
                            "id": f"{original.id}-{len(raw_chunks)}",
                        }
                    )
        chunks = []
        for index, document in enumerate(raw_chunks or []):
            chunk = _normalize_document(document, index)
            if source_name and chunk.metadata.get("source_file") in {None, "", "unknown"}:
                chunk.metadata["source_file"] = source_name
            chunks.append(chunk)
        return chunks


class LangChainIngestionAdapter:
    """Compose a LangChain loader and splitter for the KB ingestion boundary."""

    def __init__(self, loader: Any, splitter: Any):
        self.loader = LangChainDocumentLoaderAdapter(loader)
        self.splitter = LangChainTextSplitterAdapter(splitter)

    def load_and_split(
        self,
        source: Any = None,
        *,
        source_name: str | None = None,
    ) -> list[DocumentChunk]:
        return self.splitter.split_documents(
            self.loader.load(source),
            source_name=source_name,
        )


class _InlineTextLoader:
    """Small loader for API uploads that arrive as decoded text."""

    def load(self, source: Any = None) -> list[Any]:
        content = "" if source is None else str(source)
        try:
            from langchain_core.documents import Document

            return [Document(page_content=content, metadata={})]
        except ImportError:  # pragma: no cover - langchain-core is a runtime dependency
            return [{"page_content": content, "metadata": {}}]


class _ProjectMarkdownSplitter:
    """Use the existing deterministic splitter behind LangChain's document port.

    The public upload API receives text rather than a filesystem path.  This
    compatibility splitter keeps the project's heading metadata and token
    overlap semantics while ensuring the default path exercises the same
    Loader/Splitter adapters as an externally supplied LangChain component.
    """

    def __init__(self, chunker: Any):
        self.chunker = chunker

    def split_documents(self, documents: Sequence[Any]) -> list[Any]:
        chunks: list[Any] = []
        for document in documents:
            normalized = _normalize_document(document, len(chunks))
            source_name = str(normalized.metadata.get("source_file") or "unknown")
            for chunk in self.chunker.split_markdown(
                normalized.content,
                source_name=source_name,
            ):
                metadata = {**normalized.metadata, **dict(chunk.metadata)}
                metadata.setdefault("id", chunk.id)
                try:
                    from langchain_core.documents import Document

                    chunks.append(Document(page_content=chunk.content, metadata=metadata))
                except ImportError:  # pragma: no cover - see _InlineTextLoader
                    chunks.append(
                        {
                            "id": chunk.id,
                            "page_content": chunk.content,
                            "metadata": metadata,
                        }
                    )
        return chunks


def build_default_ingestion_adapter(chunker: Any) -> LangChainIngestionAdapter:
    """Build the default text-ingestion boundary used by API uploads."""
    return LangChainIngestionAdapter(
        _InlineTextLoader(),
        _ProjectMarkdownSplitter(chunker),
    )


class LangChainVectorStoreAdapter(BaseVectorDB):
    """Adapt a LangChain vector store to the project's vector-store port."""

    def __init__(self, vector_store: Any, collection_name: str = "default"):
        self.vector_store = vector_store
        self.collection_name = collection_name

    def add_chunks(
        self,
        collection_name: str,
        chunks: list[DocumentChunk],
        embeddings: list[list[float]],
    ) -> bool:
        self._check_collection(collection_name)
        if len(embeddings) != len(chunks):
            raise ValueError("embedding count must match chunk count")
        add_embeddings = getattr(self.vector_store, "add_embeddings", None)
        if callable(add_embeddings):
            text_embeddings = [
                (chunk.content, embedding)
                for chunk, embedding in zip(chunks, embeddings)
            ]
            try:
                add_embeddings(
                    text_embeddings,
                    metadatas=[dict(chunk.metadata) for chunk in chunks],
                    ids=[chunk.id for chunk in chunks],
                )
            except TypeError:
                add_embeddings(text_embeddings)
            return True
        add_documents = getattr(self.vector_store, "add_documents", None)
        if not callable(add_documents):
            raise TypeError("LangChain vector store must expose add_documents()")
        documents = [_to_langchain_document(chunk) for chunk in chunks]
        try:
            add_documents(documents, ids=[chunk.id for chunk in chunks])
        except TypeError:
            add_documents(documents)
        return True

    def search(
        self,
        collection_name: str,
        query_vector: list[float],
        top_k: int,
    ) -> list[DocumentChunk]:
        self._check_collection(collection_name)
        with_scores = getattr(self.vector_store, "similarity_search_by_vector_with_relevance_scores", None)
        if callable(with_scores):
            raw = with_scores(query_vector, k=top_k)
            documents = []
            for index, item in enumerate(raw or []):
                document, score = item
                normalized = _normalize_document(document, index)
                normalized.score = float(score)
                documents.append(normalized)
            return documents
        search = getattr(self.vector_store, "similarity_search_by_vector", None)
        if not callable(search):
            raise TypeError("LangChain vector store must expose vector similarity search")
        return [_normalize_document(item, index) for index, item in enumerate(search(query_vector, k=top_k) or [])]

    def delete_by_source(self, collection_name: str, source_file: str) -> bool:
        self._check_collection(collection_name)
        delete = getattr(self.vector_store, "delete", None)
        if not callable(delete):
            return False
        try:
            delete(where={"source_file": source_file})
        except TypeError:
            delete(filter={"source_file": source_file})
        return True

    def list_sources(self, collection_name: str) -> list[KnowledgeBaseFile]:
        self._check_collection(collection_name)
        get = getattr(self.vector_store, "get", None)
        if not callable(get):
            return []
        payload = get(include=["metadatas"])
        source_counts: dict[str, int] = {}
        for metadata in (payload or {}).get("metadatas", []) or []:
            if isinstance(metadata, Mapping) and metadata.get("source_file"):
                source = str(metadata["source_file"])
                source_counts[source] = source_counts.get(source, 0) + 1
        return [
            KnowledgeBaseFile(source_file=source, chunk_count=count)
            for source, count in sorted(source_counts.items())
        ]

    def list_collections(self) -> list[str]:
        return [self.collection_name]

    def _check_collection(self, collection_name: str) -> None:
        if collection_name != self.collection_name:
            raise ValueError(
                f"LangChain vector store is bound to collection {self.collection_name!r}, "
                f"not {collection_name!r}"
            )


def _normalize_document(document: Any, index: int) -> DocumentChunk:
    if isinstance(document, DocumentChunk):
        return document.model_copy(deep=True)
    if isinstance(document, Mapping):
        content = document.get("page_content", document.get("content", ""))
        metadata = document.get("metadata", {})
        document_id = document.get("id")
    else:
        content = getattr(document, "page_content", getattr(document, "content", ""))
        metadata = getattr(document, "metadata", {})
        document_id = getattr(document, "id", None)
    normalized_metadata = dict(metadata) if isinstance(metadata, Mapping) else {}
    document_id = document_id or normalized_metadata.get("id") or f"langchain-document-{index}"
    score = normalized_metadata.get("score")
    return DocumentChunk(
        id=str(document_id),
        content=str(content or ""),
        metadata=normalized_metadata,
        score=float(score) if isinstance(score, (int, float)) else None,
    )


def _to_langchain_document(chunk: DocumentChunk) -> Any:
    try:
        from langchain_core.documents import Document
    except ImportError as exc:  # pragma: no cover - dependency is optional at runtime
        raise RuntimeError("langchain-core is required for LangChain vector-store integration") from exc
    metadata = dict(chunk.metadata)
    metadata.setdefault("id", chunk.id)
    return Document(page_content=chunk.content, metadata=metadata)

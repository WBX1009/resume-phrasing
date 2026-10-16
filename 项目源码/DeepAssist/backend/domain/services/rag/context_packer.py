from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from backend.domain.entities.document import DocumentChunk
from backend.domain.entities.retrieval import Citation, RAGContextPack, RetrievalResult


@dataclass(frozen=True)
class ContextPackingConfig:
    """Character-budget controls for RAG context packing."""

    budget_chars: int = 12000
    per_chunk_max_chars: int = 2400
    min_chunk_chars: int = 300


class ContextPacker:
    """Packs retrieved chunks into cited, budgeted prompt context."""

    def __init__(self, config: Optional[ContextPackingConfig] = None):
        self.config = config or ContextPackingConfig()

    def pack(self, query: str, retrieval_result: RetrievalResult) -> RAGContextPack:
        citations: List[Citation] = []
        rendered_blocks: List[str] = []
        used_chars = 0
        truncated = False
        omitted = 0

        for index, doc in enumerate(retrieval_result.documents, start=1):
            remaining_budget = self.config.budget_chars - used_chars
            if remaining_budget <= self.config.min_chunk_chars:
                omitted += 1
                truncated = True
                continue

            ref_id = f"C{index}"
            content = self._prepare_content(doc.content)
            content = self._fit_content(content, remaining_budget)
            if not content:
                omitted += 1
                continue

            citation = self._build_citation(ref_id, doc, content)
            block = self._render_block(citation)

            block_len = len(block)
            if used_chars + block_len > self.config.budget_chars:
                truncated = True
                content = self._fit_content(
                    citation.content,
                    self.config.budget_chars - used_chars - 80,
                )
                if len(content) < self.config.min_chunk_chars:
                    omitted += 1
                    continue
                citation = citation.model_copy(update={"content": content})
                block = self._render_block(citation)
                block_len = len(block)

            citations.append(citation)
            rendered_blocks.append(block)
            used_chars += block_len

        if omitted:
            truncated = True

        return RAGContextPack(
            query=query,
            citations=citations,
            rendered_context="\n\n".join(rendered_blocks),
            budget_chars=self.config.budget_chars,
            used_chars=used_chars,
            truncated=truncated,
            omitted=omitted,
        )

    def _build_citation(
        self,
        ref_id: str,
        doc: DocumentChunk,
        content: str,
    ) -> Citation:
        metadata = dict(doc.metadata or {})
        title_path = self._extract_title_path(metadata)
        return Citation(
            ref_id=ref_id,
            chunk_id=doc.id,
            content=content,
            source_file=metadata.get("source_file") or metadata.get("source"),
            title_path=title_path,
            page=metadata.get("page") or metadata.get("page_number"),
            score=doc.score,
            metadata={
                "retrieval": metadata.get("retrieval", {}),
                "chunk_metadata": self._public_metadata(metadata),
            },
        )

    def _render_block(self, citation: Citation) -> str:
        header_parts = [f"[{citation.ref_id}]", f"source={citation.source_file or 'unknown'}"]
        if citation.title_path:
            header_parts.append(f"path={' > '.join(citation.title_path)}")
        if citation.page is not None:
            header_parts.append(f"page={citation.page}")
        if citation.score is not None:
            header_parts.append(f"score={citation.score:.4f}")
        return f"{' | '.join(header_parts)}\n{citation.content}"

    def _prepare_content(self, content: str) -> str:
        return " ".join((content or "").split())

    def _fit_content(self, content: str, available_chars: int) -> str:
        max_chars = min(self.config.per_chunk_max_chars, max(available_chars, 0))
        if max_chars <= 0:
            return ""
        if len(content) <= max_chars:
            return content
        if max_chars <= 20:
            return content[:max_chars]
        return f"{content[: max_chars - 15].rstrip()} ...[truncated]"

    def _extract_title_path(self, metadata: Dict[str, Any]) -> List[str]:
        for key in ("title_path", "heading_path", "headers"):
            value = metadata.get(key)
            if isinstance(value, list):
                return [str(item) for item in value if item]
            if isinstance(value, str) and value:
                return [part.strip() for part in value.split(">") if part.strip()]

        headings = []
        for key in ("h1", "h2", "h3", "title", "section"):
            if metadata.get(key):
                headings.append(str(metadata[key]))
        return headings

    def _public_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        hidden_keys = {"retrieval", "embedding", "vector"}
        return {key: value for key, value in metadata.items() if key not in hidden_keys}

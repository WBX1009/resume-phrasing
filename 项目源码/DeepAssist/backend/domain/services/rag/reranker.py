import re
from dataclasses import dataclass
from typing import List, Set

from backend.domain.entities.document import DocumentChunk
from backend.domain.interfaces.reranker import BaseReranker


@dataclass(frozen=True)
class RerankerConfig:
    model_name: str = "lexical_overlap_fallback"
    fused_score_weight: float = 0.8
    lexical_score_weight: float = 0.2


class NoOpReranker(BaseReranker):
    """No-op reranker that preserves first-stage retrieval order."""

    model_name = "noop"

    def rerank(
        self,
        query: str,
        documents: List[DocumentChunk],
        top_k: int,
    ) -> List[DocumentChunk]:
        final_docs: List[DocumentChunk] = []
        for rank, doc in enumerate(documents[:top_k], start=1):
            copied = doc.model_copy(deep=True)
            copied.metadata = {
                **copied.metadata,
                "rerank": {
                    "model": self.model_name,
                    "backend": "noop",
                    "rerank_rank": rank,
                },
            }
            final_docs.append(copied)
        return final_docs


class LexicalOverlapReranker(BaseReranker):
    """Deterministic fallback reranker based on token overlap."""

    _term_pattern = re.compile(r"[A-Za-z0-9_./#:+-]+|[\u4e00-\u9fff]{2,}")
    _generic_query_terms = {
        "资料",
        "数据",
        "数据库",
        "知识库",
        "文档",
        "回答",
        "查找",
        "找到",
        "没有",
        "根据",
        "详细",
        "解释",
        "关于",
        "一个",
        "the",
        "and",
        "or",
        "for",
        "with",
        "from",
        "please",
        "search",
        "answer",
    }

    def __init__(self, config: RerankerConfig | None = None):
        self.config = config or RerankerConfig()
        self.model_name = self.config.model_name

    def rerank(
        self,
        query: str,
        documents: List[DocumentChunk],
        top_k: int,
    ) -> List[DocumentChunk]:
        if not documents:
            return []

        query_terms = self._terms(query)
        max_fused_score = max((doc.score or 0.0) for doc in documents) or 1.0

        scored_docs = []
        for doc in documents:
            fused_score = (doc.score or 0.0) / max_fused_score
            lexical_score = self._lexical_overlap(query_terms, doc)
            rerank_score = (
                self.config.fused_score_weight * fused_score
                + self.config.lexical_score_weight * lexical_score
            )
            reranked_doc = doc.model_copy(deep=True)
            reranked_doc.score = rerank_score
            reranked_doc.metadata = {
                **reranked_doc.metadata,
                "rerank": {
                    "model": self.model_name,
                    "backend": "lexical",
                    "fused_score": doc.score,
                    "lexical_score": lexical_score,
                    "rerank_score": rerank_score,
                },
            }
            scored_docs.append(reranked_doc)

        scored_docs.sort(key=lambda doc: doc.score or 0.0, reverse=True)

        final_docs: List[DocumentChunk] = []
        for rank, doc in enumerate(scored_docs[:top_k], start=1):
            doc.metadata = {
                **doc.metadata,
                "rerank": {
                    **doc.metadata.get("rerank", {}),
                    "rerank_rank": rank,
                },
            }
            final_docs.append(doc)
        return final_docs

    def _terms(self, text: str) -> Set[str]:
        terms: Set[str] = set()
        for token in self._term_pattern.findall(text or ""):
            normalized = token.strip().lower()
            if not normalized:
                continue
            if re.search(r"[\u4e00-\u9fff]", normalized):
                for part in self._segment_chinese(normalized):
                    if len(part) > 1 and part not in self._generic_query_terms:
                        terms.add(part)
                continue
            if len(normalized) > 1 and normalized not in self._generic_query_terms:
                terms.add(normalized)
        return terms

    def _segment_chinese(self, text: str) -> List[str]:
        try:
            import jieba

            return [part.strip().lower() for part in jieba.lcut(text) if part.strip()]
        except Exception:
            return [text]

    def _lexical_overlap(self, query_terms: Set[str], doc: DocumentChunk) -> float:
        if not query_terms:
            return 0.0

        doc_terms = self._terms(doc.content)
        metadata = doc.metadata or {}
        for key in ("source_file", "title", "section", "title_path", "heading_path"):
            value = metadata.get(key)
            if isinstance(value, list):
                for item in value:
                    doc_terms.update(self._terms(str(item)))
            elif value:
                doc_terms.update(self._terms(str(value)))

        if not doc_terms:
            return 0.0

        overlap_count = len(query_terms & doc_terms)
        denominator = max(1, min(len(query_terms), len(doc_terms)))
        return overlap_count / denominator

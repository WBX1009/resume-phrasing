from collections.abc import Callable
from typing import Any, List, Optional

from backend.domain.entities.document import DocumentChunk
from backend.domain.interfaces.reranker import BaseReranker
from backend.domain.services.rag.reranker import LexicalOverlapReranker

CrossEncoderFactory = Callable[[str, Optional[str]], Any]


class CrossEncoderReranker(BaseReranker):
    """sentence-transformers CrossEncoder adapter with lexical fallback."""

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-base",
        device: Optional[str] = None,
        batch_size: int = 16,
        fallback: Optional[BaseReranker] = None,
        model_factory: Optional[CrossEncoderFactory] = None,
    ):
        self.model_name = model_name
        self.device = None if device in {None, "", "auto"} else device
        self.batch_size = batch_size
        self.fallback = fallback or LexicalOverlapReranker()
        self._model_factory = model_factory or self._default_model_factory
        self._model: Any = None
        self.last_backend = "cross_encoder"
        self.last_fallback_reason: Optional[str] = None

    def rerank(
        self,
        query: str,
        documents: List[DocumentChunk],
        top_k: int,
    ) -> List[DocumentChunk]:
        self.last_backend = "cross_encoder"
        self.last_fallback_reason = None

        if not documents:
            return []

        try:
            model = self._load_model()
            pairs = [(query, doc.content or "") for doc in documents]
            scores = model.predict(
                pairs,
                batch_size=self.batch_size,
                show_progress_bar=False,
            )
            return self._rank_with_scores(documents, [float(score) for score in scores], top_k)
        except Exception as exc:
            return self._fallback(query, documents, top_k, exc)

    def _load_model(self) -> Any:
        if self._model is None:
            self._model = self._model_factory(self.model_name, self.device)
        return self._model

    def _rank_with_scores(
        self,
        documents: List[DocumentChunk],
        scores: List[float],
        top_k: int,
    ) -> List[DocumentChunk]:
        scored_docs = []
        for doc, score in zip(documents, scores):
            reranked_doc = doc.model_copy(deep=True)
            reranked_doc.score = score
            reranked_doc.metadata = {
                **reranked_doc.metadata,
                "rerank": {
                    "model": self.model_name,
                    "backend": "cross_encoder",
                    "fused_score": doc.score,
                    "cross_encoder_score": score,
                    "rerank_score": score,
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

    def _fallback(
        self,
        query: str,
        documents: List[DocumentChunk],
        top_k: int,
        exc: Exception,
    ) -> List[DocumentChunk]:
        reason = f"{type(exc).__name__}: {exc}"
        self.last_backend = "lexical_fallback"
        self.last_fallback_reason = reason
        fallback_docs = self.fallback.rerank(query, documents, top_k)
        fallback_model = getattr(self.fallback, "model_name", self.fallback.__class__.__name__)

        final_docs: List[DocumentChunk] = []
        for doc in fallback_docs:
            copied = doc.model_copy(deep=True)
            copied.metadata = {
                **copied.metadata,
                "rerank": {
                    **copied.metadata.get("rerank", {}),
                    "requested_model": self.model_name,
                    "backend": "lexical_fallback",
                    "fallback_model": fallback_model,
                    "fallback_reason": reason,
                },
            }
            final_docs.append(copied)
        return final_docs

    @staticmethod
    def _default_model_factory(model_name: str, device: Optional[str]) -> Any:
        from sentence_transformers import CrossEncoder

        if device:
            return CrossEncoder(model_name, device=device)
        return CrossEncoder(model_name)

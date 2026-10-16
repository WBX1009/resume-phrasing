from backend.domain.interfaces.reranker import BaseReranker
from backend.domain.services.rag.reranker import LexicalOverlapReranker
from backend.infrastructure.rerankers.cross_encoder_reranker import CrossEncoderReranker
from backend.infrastructure.rerankers.noop_reranker import NoOpReranker


def build_reranker(
    backend: str = "cross_encoder",
    model_name: str = "BAAI/bge-reranker-base",
    device: str | None = "auto",
    batch_size: int = 16,
) -> BaseReranker:
    """Build a reranker adapter without importing the API dependency graph."""

    normalized_backend = (backend or "cross_encoder").strip().lower()
    if normalized_backend in {"cross_encoder", "cross-encoder", "ce"}:
        return CrossEncoderReranker(
            model_name=model_name,
            device=device,
            batch_size=batch_size,
            fallback=LexicalOverlapReranker(),
        )
    if normalized_backend in {"lexical", "lexical_overlap"}:
        return LexicalOverlapReranker()
    if normalized_backend in {"noop", "none", "disabled"}:
        return NoOpReranker()
    return LexicalOverlapReranker()

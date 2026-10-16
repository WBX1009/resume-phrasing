from dataclasses import dataclass
from typing import Dict, List, Optional

from backend.common.logger import get_logger
from backend.domain.entities.document import DocumentChunk
from backend.domain.entities.retrieval import (
    QueryPlan,
    RerankTrace,
    RetrievalChannelTrace,
    RetrievalResult,
)
from backend.domain.interfaces.embedding import BaseEmbedding
from backend.domain.interfaces.keyword_db import BaseKeywordDB
from backend.domain.interfaces.reranker import BaseReranker
from backend.domain.interfaces.vector_db import BaseVectorDB
from backend.domain.services.rag.query_planner import QueryPlanner
from backend.domain.services.rag.reranker import LexicalOverlapReranker

logger = get_logger(__name__)


@dataclass(frozen=True)
class RetrievalConfig:
    """Runtime retrieval knobs for hybrid search."""

    top_k: int = 5
    candidate_multiplier: int = 2
    rrf_k: int = 60
    vector_weight: float = 1.0
    keyword_weight: float = 1.0
    enable_vector: bool = True
    enable_keyword: bool = True
    enable_rerank: bool = True
    low_relevance_threshold: float = 0.2


class HybridRetriever:
    def __init__(
        self,
        vector_db: BaseVectorDB,
        keyword_db: BaseKeywordDB,
        embedding_model: BaseEmbedding,
        config: Optional[RetrievalConfig] = None,
        query_planner: Optional[QueryPlanner] = None,
        reranker: Optional[BaseReranker] = None,
    ):
        self.vector_db = vector_db
        self.keyword_db = keyword_db
        self.embedding_model = embedding_model
        self.config = config or RetrievalConfig()
        self.query_planner = query_planner or QueryPlanner()
        self.reranker = reranker or LexicalOverlapReranker()

    def retrieve(
        self,
        collection_name: str,
        query: str,
        top_k: Optional[int] = None,
        vector_weight: Optional[float] = None,
        keyword_weight: Optional[float] = None,
    ) -> List[DocumentChunk]:
        return self.retrieve_with_trace(
            collection_name=collection_name,
            query=query,
            top_k=top_k,
            vector_weight=vector_weight,
            keyword_weight=keyword_weight,
        ).documents

    def retrieve_with_trace(
        self,
        collection_name: str,
        query: str,
        top_k: Optional[int] = None,
        vector_weight: Optional[float] = None,
        keyword_weight: Optional[float] = None,
    ) -> RetrievalResult:
        query_plan = self.query_planner.plan(query)
        effective_top_k = top_k or self.config.top_k
        candidate_k = max(effective_top_k * self.config.candidate_multiplier, effective_top_k)
        effective_vector_weight = self.config.vector_weight if vector_weight is None else vector_weight
        effective_keyword_weight = self.config.keyword_weight if keyword_weight is None else keyword_weight

        if not query_plan.normalized_query:
            return self._empty_result(
                collection_name=collection_name,
                query=query_plan.original_query,
                top_k=effective_top_k,
                candidate_k=candidate_k,
                vector_weight=effective_vector_weight,
                keyword_weight=effective_keyword_weight,
                query_plan=query_plan,
                metadata={"reason": "empty_query"},
            )

        logger.info("Starting hybrid retrieval for query: %s", query_plan.normalized_query)

        vector_results, vector_trace = self._retrieve_vector(
            collection_name,
            query_plan.semantic_queries or [query_plan.semantic_query],
            candidate_k,
        )
        keyword_results, keyword_trace = self._retrieve_keyword(
            collection_name,
            query_plan.keyword_queries or [query_plan.keyword_query],
            candidate_k,
        )

        fused_candidates = self._weighted_rrf(
            vector_results=vector_results,
            keyword_results=keyword_results,
            top_k=candidate_k,
            vector_weight=effective_vector_weight,
            keyword_weight=effective_keyword_weight,
        )
        rerank_query = self._build_rerank_query(query_plan)
        final_docs, rerank_trace = self._rerank(
            query=rerank_query,
            documents=fused_candidates,
            top_k=effective_top_k,
        )

        result = RetrievalResult(
            query=query_plan.original_query,
            collection_name=collection_name,
            documents=final_docs,
            top_k=effective_top_k,
            candidate_k=candidate_k,
            rrf_k=self.config.rrf_k,
            vector_weight=effective_vector_weight,
            keyword_weight=effective_keyword_weight,
            query_plan=query_plan,
            rerank=rerank_trace,
            channels={
                "vector": vector_trace,
                "keyword": keyword_trace,
            },
            metadata=self._build_retrieval_diagnostics(
                query_plan=query_plan,
                vector_trace=vector_trace,
                keyword_trace=keyword_trace,
                final_docs=final_docs,
            ),
        )
        logger.info(
            "Retrieval completed: vector=%s keyword=%s fused=%s",
            len(vector_results),
            len(keyword_results),
            len(final_docs),
        )
        return result

    def _build_retrieval_diagnostics(
        self,
        query_plan: QueryPlan,
        vector_trace: RetrievalChannelTrace,
        keyword_trace: RetrievalChannelTrace,
        final_docs: List[DocumentChunk],
    ) -> Dict[str, object]:
        best_score = self._best_source_score(final_docs)
        reason_code = "ok"
        reason_message = "retrieval_ready"
        suggested_action = "proceed_with_rag"

        if not vector_trace.success and not keyword_trace.success:
            reason_code = "all_channels_failed"
            reason_message = "vector_and_keyword_channels_failed"
            suggested_action = "fallback_to_chat"
        elif not final_docs:
            reason_code = "no_hits"
            reason_message = "no_documents_retrieved_after_fusion"
            suggested_action = "fallback_to_chat"
        elif (best_score or 0.0) < self.config.low_relevance_threshold:
            reason_code = "low_relevance"
            reason_message = "top_document_score_below_threshold"
            suggested_action = "fallback_to_chat"
        elif not vector_trace.success or not keyword_trace.success:
            reason_code = "partial_channel_failure"
            reason_message = "one_retrieval_channel_failed_but_the_other_returned_candidates"
            suggested_action = "proceed_with_warning"
        elif (vector_trace.returned == 0) != (keyword_trace.returned == 0):
            reason_code = "single_channel_recall"
            reason_message = "only_one_retrieval_channel_contributed_candidates"
            suggested_action = "proceed_with_warning"

        return {
            "diagnostics": {
                "reason_code": reason_code,
                "reason_message": reason_message,
                "suggested_action": suggested_action,
                "best_score": best_score,
                "low_relevance_threshold": self.config.low_relevance_threshold,
                "rewrite_applied": query_plan.metadata.get("rewrite_applied", False),
                "rewrite_notes": query_plan.metadata.get("rewrite_notes", []),
                "domain_hints": query_plan.metadata.get("domain_hints", []),
                "semantic_queries": query_plan.semantic_queries,
                "keyword_queries": query_plan.keyword_queries,
                "rerank_query": self._build_rerank_query(query_plan),
                "sub_queries": query_plan.sub_queries,
                "channel_summary": {
                    "vector": vector_trace.model_dump(exclude_none=True),
                    "keyword": keyword_trace.model_dump(exclude_none=True),
                },
            }
        }

    def _best_source_score(self, final_docs: List[DocumentChunk]) -> Optional[float]:
        scores: List[float] = []
        for doc in final_docs:
            retrieval = doc.metadata.get("retrieval", {}) if doc.metadata else {}
            rerank = doc.metadata.get("rerank", {}) if doc.metadata else {}
            for key in ("vector_score", "keyword_score"):
                value = retrieval.get(key)
                if isinstance(value, (int, float)):
                    scores.append(float(value))
            fused_score = rerank.get("fused_score")
            if isinstance(fused_score, (int, float)):
                scores.append(float(fused_score))
        if scores:
            return max(scores)
        return final_docs[0].score if final_docs else None

    def _rerank(
        self,
        query: str,
        documents: List[DocumentChunk],
        top_k: int,
    ) -> tuple[List[DocumentChunk], RerankTrace]:
        model_name = getattr(self.reranker, "model_name", self.reranker.__class__.__name__)
        if not self.config.enable_rerank:
            return list(documents[:top_k]), RerankTrace(
                enabled=False,
                success=True,
                model=model_name,
                backend="disabled",
                input_count=len(documents),
                returned=min(len(documents), top_k),
            )

        try:
            reranked = self.reranker.rerank(query, documents, top_k)
            return reranked, RerankTrace(
                enabled=True,
                success=True,
                model=model_name,
                backend=getattr(self.reranker, "last_backend", None),
                input_count=len(documents),
                returned=len(reranked),
                fallback_reason=getattr(self.reranker, "last_fallback_reason", None),
            )
        except Exception as exc:
            logger.error("Reranking failed and will fall back to fused order: %s", exc)
            fallback = list(documents[:top_k])
            return fallback, RerankTrace(
                enabled=True,
                success=False,
                model=model_name,
                backend="fused_order_fallback",
                input_count=len(documents),
                returned=len(fallback),
                error=str(exc),
                fallback_reason=str(exc),
            )

    def _retrieve_vector(
        self,
        collection_name: str,
        queries: List[str],
        candidate_k: int,
    ) -> tuple[List[DocumentChunk], RetrievalChannelTrace]:
        if not self.config.enable_vector:
            return [], RetrievalChannelTrace(enabled=False, success=True, returned=0, query_count=0)

        try:
            results = self._collect_vector_results(collection_name, queries, candidate_k)
            return results, RetrievalChannelTrace(
                success=True,
                returned=len(results),
                query_count=len(self._dedupe_queries(queries)),
            )
        except Exception as exc:
            logger.error("Vector retrieval failed and will be skipped: %s", exc)
            return [], RetrievalChannelTrace(success=False, error=str(exc), returned=0)

    def _retrieve_keyword(
        self,
        collection_name: str,
        queries: List[str],
        candidate_k: int,
    ) -> tuple[List[DocumentChunk], RetrievalChannelTrace]:
        if not self.config.enable_keyword:
            return [], RetrievalChannelTrace(enabled=False, success=True, returned=0, query_count=0)

        try:
            results = self._collect_keyword_results(collection_name, queries, candidate_k)
            return results, RetrievalChannelTrace(
                success=True,
                returned=len(results),
                query_count=len(self._dedupe_queries(queries)),
            )
        except Exception as exc:
            logger.error("Keyword retrieval failed and will be skipped: %s", exc)
            return [], RetrievalChannelTrace(success=False, error=str(exc), returned=0)

    def _collect_vector_results(
        self,
        collection_name: str,
        queries: List[str],
        candidate_k: int,
    ) -> List[DocumentChunk]:
        collected: Dict[str, DocumentChunk] = {}
        per_query_k = max(candidate_k, 1)
        for query_index, query in enumerate(self._dedupe_queries(queries), start=1):
            query_vector = self.embedding_model.embed_text(query)
            if not query_vector:
                continue
            for rank, chunk in enumerate(
                self.vector_db.search(collection_name, query_vector, per_query_k),
                start=1,
            ):
                collected.setdefault(
                    chunk.id,
                    self._annotate_query_hit(chunk, "vector", query, query_index, rank),
                )
        return list(collected.values())

    def _collect_keyword_results(
        self,
        collection_name: str,
        queries: List[str],
        candidate_k: int,
    ) -> List[DocumentChunk]:
        collected: Dict[str, DocumentChunk] = {}
        per_query_k = max(candidate_k, 1)
        for query_index, query in enumerate(self._dedupe_queries(queries), start=1):
            for rank, chunk in enumerate(
                self.keyword_db.search(collection_name, query, per_query_k),
                start=1,
            ):
                collected.setdefault(
                    chunk.id,
                    self._annotate_query_hit(chunk, "keyword", query, query_index, rank),
                )
        return list(collected.values())

    def _annotate_query_hit(
        self,
        chunk: DocumentChunk,
        channel: str,
        query: str,
        query_index: int,
        rank: int,
    ) -> DocumentChunk:
        annotated = chunk.model_copy(deep=True)
        annotated.metadata = {
            **annotated.metadata,
            "query_expansion": {
                "channel": channel,
                "query": query,
                "query_index": query_index,
                "rank": rank,
            },
        }
        return annotated

    def _dedupe_queries(self, queries: List[str]) -> List[str]:
        seen = set()
        deduped: List[str] = []
        for query in queries:
            normalized = " ".join((query or "").split()).strip()
            key = normalized.lower()
            if not normalized or key in seen:
                continue
            seen.add(key)
            deduped.append(normalized)
        return deduped

    def _empty_result(
        self,
        collection_name: str,
        query: str,
        top_k: int,
        candidate_k: int,
        vector_weight: float,
        keyword_weight: float,
        query_plan: QueryPlan,
        metadata: Optional[Dict[str, str]] = None,
    ) -> RetrievalResult:
        return RetrievalResult(
            query=query,
            collection_name=collection_name,
            documents=[],
            top_k=top_k,
            candidate_k=candidate_k,
            rrf_k=self.config.rrf_k,
            vector_weight=vector_weight,
            keyword_weight=keyword_weight,
            query_plan=query_plan,
            rerank=RerankTrace(
                enabled=self.config.enable_rerank,
                success=True,
                model=getattr(self.reranker, "model_name", self.reranker.__class__.__name__),
                backend=getattr(self.reranker, "last_backend", None),
                input_count=0,
                returned=0,
            ),
            channels={
                "vector": RetrievalChannelTrace(
                    enabled=self.config.enable_vector,
                    success=True,
                    returned=0,
                    query_count=0,
                ),
                "keyword": RetrievalChannelTrace(
                    enabled=self.config.enable_keyword,
                    success=True,
                    returned=0,
                    query_count=0,
                ),
            },
            metadata=metadata or {},
        )

    def _weighted_rrf(
        self,
        vector_results: List[DocumentChunk],
        keyword_results: List[DocumentChunk],
        top_k: int,
        vector_weight: float,
        keyword_weight: float,
    ) -> List[DocumentChunk]:
        chunk_map: Dict[str, DocumentChunk] = {}
        score_map: Dict[str, float] = {}
        trace_map: Dict[str, Dict[str, Optional[float]]] = {}

        self._accumulate_rrf(
            results=vector_results,
            channel="vector",
            weight=vector_weight,
            chunk_map=chunk_map,
            score_map=score_map,
            trace_map=trace_map,
        )
        self._accumulate_rrf(
            results=keyword_results,
            channel="keyword",
            weight=keyword_weight,
            chunk_map=chunk_map,
            score_map=score_map,
            trace_map=trace_map,
        )

        sorted_ids = sorted(score_map.keys(), key=lambda chunk_id: score_map[chunk_id], reverse=True)
        final_results: List[DocumentChunk] = []

        for chunk_id in sorted_ids[:top_k]:
            source_chunk = chunk_map[chunk_id]
            chunk = source_chunk.model_copy(deep=True)
            chunk.score = score_map[chunk_id]

            retrieval_trace = trace_map[chunk_id]
            retrieval_trace.update(
                {
                    "fusion": "weighted_rrf",
                    "rrf_k": self.config.rrf_k,
                    "vector_weight": vector_weight,
                    "keyword_weight": keyword_weight,
                    "fused_score": chunk.score,
                }
            )
            chunk.metadata = {**chunk.metadata, "retrieval": retrieval_trace}
            final_results.append(chunk)

        return final_results

    def _build_rerank_query(self, query_plan: QueryPlan) -> str:
        candidates = [
            query_plan.rewritten_query,
            " ".join(query_plan.keyword_queries or []),
            " ".join(query_plan.semantic_queries or []),
            query_plan.keyword_query,
            query_plan.normalized_query,
        ]
        for candidate in candidates:
            normalized = " ".join((candidate or "").split()).strip()
            if normalized:
                return normalized
        return query_plan.original_query

    def _accumulate_rrf(
        self,
        results: List[DocumentChunk],
        channel: str,
        weight: float,
        chunk_map: Dict[str, DocumentChunk],
        score_map: Dict[str, float],
        trace_map: Dict[str, Dict[str, Optional[float]]],
    ) -> None:
        if weight <= 0:
            return

        for rank, chunk in enumerate(results, start=1):
            contribution = weight / (self.config.rrf_k + rank)
            chunk_map.setdefault(chunk.id, chunk)
            score_map[chunk.id] = score_map.get(chunk.id, 0.0) + contribution

            trace = trace_map.setdefault(
                chunk.id,
                {
                    "vector_rank": None,
                    "keyword_rank": None,
                    "vector_score": None,
                    "keyword_score": None,
                    "vector_rrf": 0.0,
                    "keyword_rrf": 0.0,
                },
            )
            trace[f"{channel}_rank"] = rank
            trace[f"{channel}_score"] = chunk.score
            trace[f"{channel}_rrf"] = contribution

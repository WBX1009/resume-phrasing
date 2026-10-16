from backend.domain.entities.answer import AnswerGroundingReport
from backend.domain.entities.rag_pipeline import RAGPipelineResult
from backend.domain.entities.retrieval import RetrievalChannelTrace, RetrievalResult
from backend.domain.services.rag.answer_guard import SourceAwareResponseGuard
from backend.domain.services.rag.context_packer import ContextPacker
from backend.domain.services.rag.fusion import HybridRetriever


class RAGPipeline:
    """Coordinates plan, retrieve, rerank, pack, and answer grounding checks."""

    def __init__(
        self,
        retriever: HybridRetriever,
        context_packer: ContextPacker | None = None,
        answer_guard: SourceAwareResponseGuard | None = None,
        low_relevance_threshold: float = 0.2,
    ):
        self.retriever = retriever
        self.context_packer = context_packer or ContextPacker()
        self.answer_guard = answer_guard or SourceAwareResponseGuard()
        self.low_relevance_threshold = low_relevance_threshold

    def build_context(self, query: str, collection_name: str) -> RAGPipelineResult:
        retrieval_result = self._retrieve(query, collection_name)
        context_pack = self.context_packer.pack(query, retrieval_result)
        return RAGPipelineResult(
            query=query,
            collection_name=collection_name,
            retrieval_result=retrieval_result,
            context_pack=context_pack,
        )

    def check_answer(
        self,
        answer: str,
        pipeline_result: RAGPipelineResult,
    ) -> AnswerGroundingReport:
        return self.answer_guard.check(answer, pipeline_result.context_pack)

    def _retrieve(self, query: str, collection_name: str) -> RetrievalResult:
        if collection_name not in {"__all__", "all", "auto"}:
            return self.retriever.retrieve_with_trace(collection_name, query)

        collections = self._available_collections()
        if not collections:
            return self.retriever.retrieve_with_trace("tech_docs_kb", query)

        results = [
            self.retriever.retrieve_with_trace(collection, query)
            for collection in collections
        ]
        docs = []
        for result in results:
            for doc in result.documents:
                annotated = doc.model_copy(deep=True)
                annotated.metadata = {
                    **annotated.metadata,
                    "collection_name": result.collection_name,
                }
                docs.append(annotated)

        docs.sort(key=lambda doc: doc.score or 0.0, reverse=True)
        top_k = self.retriever.config.top_k
        base = results[0].model_copy(deep=True)
        aggregate_docs = docs[:top_k]
        return base.model_copy(
            update={
                "collection_name": "__all__",
                "documents": aggregate_docs,
                "channels": self._merge_channels(results),
                "metadata": {
                    **self._aggregate_metadata(base, results, aggregate_docs),
                    "searched_collections": collections,
                    "per_collection": [
                        {
                            "collection_name": result.collection_name,
                            "hit_count": result.hit_count,
                            "channels": {
                                name: trace.model_dump(exclude_none=True)
                                for name, trace in result.channels.items()
                            },
                        }
                        for result in results
                    ],
                },
            }
        )

    def _merge_channels(
        self,
        results: list[RetrievalResult],
    ) -> dict[str, RetrievalChannelTrace]:
        merged: dict[str, RetrievalChannelTrace] = {}
        for channel_name in ("vector", "keyword"):
            traces = [result.channels.get(channel_name) for result in results if result.channels.get(channel_name)]
            if not traces:
                continue
            merged[channel_name] = RetrievalChannelTrace(
                enabled=any(trace.enabled for trace in traces),
                success=any(trace.success for trace in traces),
                error="; ".join(filter(None, (trace.error for trace in traces if not trace.success))) or None,
                returned=sum(trace.returned for trace in traces),
                query_count=sum(trace.query_count for trace in traces),
            )
        return merged

    def _aggregate_metadata(
        self,
        base: RetrievalResult,
        results: list[RetrievalResult],
        aggregate_docs,
    ) -> dict:
        metadata = dict(base.metadata or {})
        diagnostics = dict(metadata.get("diagnostics", {}))
        best_score = aggregate_docs[0].score if aggregate_docs else None
        reason_code = "ok"
        reason_message = "retrieval_ready"
        suggested_action = "proceed_with_rag"

        if not aggregate_docs:
            reason_code = "no_hits"
            reason_message = "no_documents_retrieved_across_any_collection"
            suggested_action = "fallback_to_chat"
        elif (best_score or 0.0) < diagnostics.get(
            "low_relevance_threshold",
            self.low_relevance_threshold,
        ):
            reason_code = "low_relevance"
            reason_message = "top_cross_collection_document_score_below_threshold"
            suggested_action = "fallback_to_chat"

        metadata["diagnostics"] = {
            **diagnostics,
            "reason_code": reason_code,
            "reason_message": reason_message,
            "suggested_action": suggested_action,
            "best_score": best_score,
            "searched_collection_count": len(results),
        }
        return metadata

    def _available_collections(self) -> list[str]:
        names: set[str] = set()
        for adapter in (self.retriever.vector_db, self.retriever.keyword_db):
            list_collections = getattr(adapter, "list_collections", None)
            if callable(list_collections):
                names.update(list_collections())
        return sorted(name for name in names if name)

from backend.common.logger import get_logger
from backend.domain.services.rag.fusion import HybridRetriever
from backend.domain.services.rag.pipeline import RAGPipeline

logger = get_logger(__name__)


class KnowledgeBaseTool:
    """Agent-facing KB search tool."""

    def __init__(
        self,
        retriever: HybridRetriever | None = None,
        rag_pipeline: RAGPipeline | None = None,
        collection_name: str = "__all__",
    ):
        self.retriever = retriever
        self.rag_pipeline = rag_pipeline
        self.collection_name = collection_name

    def search_knowledge_base(self, query: str) -> str:
        """Search the connected knowledge bases for evidence relevant to the query."""
        logger.info("[Tool] Agent KB search: %s", query)
        try:
            docs = self._retrieve(query)
            if not docs:
                return (
                    "No relevant knowledge-base content was retrieved. "
                    "Answer from general knowledge or tell the user the KB has no matching reference."
                )

            context = "\n\n".join(
                f"[Snippet {index + 1}]: {doc.content}"
                for index, doc in enumerate(docs)
            )
            return f"Knowledge-base retrieval succeeded:\n{context}"
        except Exception as exc:
            logger.error("KB tool execution failed: %s", exc)
            return f"Knowledge-base retrieval failed: {exc}"

    def _retrieve(self, query: str):
        if self.rag_pipeline is not None:
            return self.rag_pipeline.build_context(
                query=query,
                collection_name=self.collection_name,
            ).retrieval_result.documents

        if self.retriever is None:
            raise RuntimeError("KnowledgeBaseTool requires a retriever or RAG pipeline")

        return self.retriever.retrieve_with_trace(
            self.collection_name,
            query,
        ).documents

    def search(self, query: str) -> str:
        """Backward-compatible alias for legacy tests and callers."""
        return self.search_knowledge_base(query)

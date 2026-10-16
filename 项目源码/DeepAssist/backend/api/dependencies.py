import sys
from functools import lru_cache
from typing import Any, Callable, List, Optional, Dict

from backend.application.agent_app import AgentApplication
from backend.application.chat_app import ChatApplication
from backend.application.conversation_app import ConversationApplication
from backend.application.kb_app import KnowledgeBaseApp
from backend.application.runtime_app import RuntimeApplication
from backend.common.event_bus import event_bus
from backend.common.logger import get_logger
from backend.domain.interfaces.embedding import BaseEmbedding
from backend.domain.interfaces.keyword_db import BaseKeywordDB
from backend.domain.interfaces.llm import BaseLLM
from backend.domain.interfaces.memory_db import BaseMemoryStore
from backend.domain.interfaces.vector_db import BaseVectorDB
from backend.domain.interfaces.kb_manifest_store import BaseKnowledgeBaseManifestStore
from backend.domain.interfaces.agent_runtime import AgentRuntimePort
from backend.domain.interfaces.approval_store import ApprovalStorePort
from backend.domain.interfaces.trace_store import TraceStorePort
from backend.infrastructure.approvals.local_approval_store import LocalApprovalStore
from backend.infrastructure.databases.kb_manifest_store import KnowledgeBaseManifestStore
from backend.infrastructure.embeddings.bge_m3_local import BGEM3Local
from backend.infrastructure.llms.deepseek_client import DeepSeekClient
from backend.infrastructure.llms.openai_compatible import OpenAICompatibleClient
from backend.infrastructure.tracing.local_trace_store import LocalTraceStore
from backend.infrastructure.tools.file_ops import (
    list_sandbox_files,
    read_local_file,
    write_local_file,
)
from backend.infrastructure.tools.python_sandbox import execute_python_sandbox
from backend.infrastructure.tools.kb_catalog_tool import KnowledgeBaseCatalogTool
from backend.infrastructure.tools.rag_tool import KnowledgeBaseTool
from backend.infrastructure.mcp.client_adapter import MCPToolClientAdapter
from backend.infrastructure.mcp.server import DeepAssistMCPServer
from backend.domain.services.agent.engine import AgentEngine
from backend.domain.entities.agent_run import AgentRunConfig
from backend.domain.services.agent.intent_router import IntentRouter
from backend.domain.services.agent.supervisor import AgentSupervisor
from backend.domain.services.agent.tooling import ToolPolicy, ToolRegistry
from backend.domain.services.agent.task_decomposer import TaskDecomposer
from backend.domain.services.agent.workers import (
    ChatWorker,
    OrchestratorWorker,
    RAGWorker,
    ToolAgentWorker,
)
from backend.domain.services.conversation.router import ConversationRouter
from backend.domain.services.session.context_engine import ContextEngine
from backend.domain.services.agent.skill_registry import build_default_skill_registry
from backend.domain.services.session.profile_extractor import ProfileExtractor
from backend.domain.services.rag.answer_guard import SourceAwareResponseGuard
from backend.domain.services.rag.chunking import DocumentChunker
from backend.infrastructure.rag.langchain_adapters import build_default_ingestion_adapter
from backend.domain.services.rag.context_packer import ContextPacker
from backend.domain.services.rag.fusion import HybridRetriever, RetrievalConfig
from backend.domain.services.rag.pipeline import RAGPipeline
from backend.domain.services.rag.query_planner import QueryPlanner
from backend.domain.services.rag.query_rewriter import QueryRewriteService, QueryRewriteConfig
from backend.domain.interfaces.reranker import BaseReranker
from backend.infrastructure.rerankers.factory import build_reranker
from backend.domain.services.session.context_window_manager import PriorityContextWindowManager
from backend.domain.services.session.long_term_memory_recall import LongTermMemoryRecallService
from backend.domain.services.session.manager import SessionManager
from backend.domain.services.session.summary_compressor import ConversationSummaryCompressor
from backend.common.config import settings

logger = get_logger(__name__)

_ADAPTER_CACHE: Dict[str, Any] = {}
_ADAPTER_ERRORS: Dict[str, str] = {}


def _get_cached_adapter(name: str, factory: Callable[[], Any]) -> Any:
    if name in _ADAPTER_CACHE:
        return _ADAPTER_CACHE[name]
    try:
        value = factory()
        _ADAPTER_CACHE[name] = value
        _ADAPTER_ERRORS.pop(name, None)
        return value
    except Exception as exc:
        _ADAPTER_ERRORS[name] = f"{type(exc).__name__}: {exc}"
        logger.warning("%s initialization failed: %s", name, exc)
        return None


def clear_adapter_cache() -> None:
    _ADAPTER_CACHE.clear()
    _ADAPTER_ERRORS.clear()
    for cached_factory_name in ("get_memory_store", "get_reranker"):
        cached_factory = globals().get(cached_factory_name)
        cache_clear = getattr(cached_factory, "cache_clear", None)
        if callable(cache_clear):
            cache_clear()

# -----------------------------------------------------------------------------
# Infrastructure adapters (Heavy, MUST be cached)
# -----------------------------------------------------------------------------

@lru_cache()
def get_llm() -> BaseLLM:
    provider = (settings.LLM_PROVIDER or "deepseek").strip().lower().replace("-", "_")
    if provider in {"openai_compatible", "openai"}:
        return OpenAICompatibleClient()
    return DeepSeekClient()

def get_embedding_model() -> Optional[BaseEmbedding]:
    return _get_cached_adapter("embedding_model", BGEM3Local)

def get_vector_db() -> Optional[BaseVectorDB]:
    from backend.infrastructure.databases.chroma_store import ChromaStore

    return _get_cached_adapter("vector_db", ChromaStore)

def get_keyword_db() -> Optional[BaseKeywordDB]:
    backend = (settings.KEYWORD_BACKEND or "elasticsearch").strip().lower()

    def factory() -> BaseKeywordDB:
        if backend in {"whoosh", "local"}:
            from backend.infrastructure.databases.whoosh_store import WhooshStore

            return WhooshStore()
        if backend in {"elasticsearch", "es", "opensearch"}:
            from backend.infrastructure.search.elasticsearch_store import ElasticsearchKeywordSearchAdapter
            from backend.infrastructure.search.resilient_keyword_store import ResilientKeywordDB
            from backend.infrastructure.databases.whoosh_store import WhooshStore

            return ResilientKeywordDB(
                primary=ElasticsearchKeywordSearchAdapter(),
                secondary=WhooshStore(),
            )
        raise ValueError(f"Unsupported KEYWORD_BACKEND={settings.KEYWORD_BACKEND}")

    return _get_cached_adapter(f"keyword_db:{backend}", factory)

def _get_memory_store_optional() -> Optional[BaseMemoryStore]:
    backend = (settings.MEMORY_BACKEND or "sqlite").strip().lower()

    def factory() -> BaseMemoryStore:
        if backend == "sqlite":
            from backend.infrastructure.databases.sqlite_memory import SQLiteMemoryStore

            return SQLiteMemoryStore()
        if backend == "mysql":
            from backend.infrastructure.memory.mysql_memory import MySQLMemoryStoreAdapter

            return MySQLMemoryStoreAdapter()
        raise ValueError(f"Unsupported MEMORY_BACKEND={settings.MEMORY_BACKEND}")

    return _get_cached_adapter(f"memory_store:{backend}", factory)


@lru_cache()
def get_memory_store() -> BaseMemoryStore:
    memory_store = _get_memory_store_optional()
    if memory_store is None:
        raise RuntimeError(
            _adapter_error("memory_store")
            or f"Unsupported or unavailable MEMORY_BACKEND={settings.MEMORY_BACKEND}"
        )
    return memory_store

@lru_cache()
def get_manifest_store() -> BaseKnowledgeBaseManifestStore:
    """Provide a manifest store via the abstract interface."""
    return KnowledgeBaseManifestStore(settings.KB_MANIFEST_PATH)


@lru_cache()
def get_trace_store() -> TraceStorePort:
    local_store = LocalTraceStore()
    if settings.LANGSMITH_TRACING and settings.LANGSMITH_API_KEY:
        try:
            from backend.infrastructure.tracing.langsmith_trace_store import (
                LangSmithTraceStore,
            )

            return LangSmithTraceStore.from_settings(
                settings,
                local_store=local_store,
            )
        except Exception as exc:
            logger.warning("LangSmith tracing disabled for this process: %s", exc)
    return local_store


@lru_cache()
def get_approval_store() -> ApprovalStorePort:
    return LocalApprovalStore()

# -----------------------------------------------------------------------------
# RAG service graph (Lightweight, Recreated per request if dependent on dynamic config)
# -----------------------------------------------------------------------------

@lru_cache()
def get_reranker() -> BaseReranker:
    reranker = build_reranker(
        backend=settings.RERANKER_BACKEND,
        model_name=settings.RERANKER_MODEL_NAME,
        device=settings.RERANKER_DEVICE,
        batch_size=settings.RERANKER_BATCH_SIZE,
    )
    if (
        (settings.RERANKER_BACKEND or "cross_encoder").strip().lower()
        not in {"cross_encoder", "cross-encoder", "ce", "lexical", "lexical_overlap", "noop", "none", "disabled"}
    ):
        logger.warning(
            "Unknown RERANKER_BACKEND=%s; falling back to lexical reranker.",
            settings.RERANKER_BACKEND,
        )
    return reranker

@lru_cache()
def get_context_packer() -> ContextPacker:
    return ContextPacker()

@lru_cache()
def get_answer_guard() -> SourceAwareResponseGuard:
    return SourceAwareResponseGuard()

def get_query_rewriter() -> QueryRewriteService:
    """Build a query rewriter with current runtime settings."""
    return QueryRewriteService(
        config=QueryRewriteConfig(enable_llm_rewrite=settings.ENABLE_LLM_REWRITE),
        llm=get_llm(),
    )

def get_query_planner() -> QueryPlanner:
    """Build a query planner with a fresh query rewriter."""
    return QueryPlanner(query_rewriter=get_query_rewriter())

def get_retriever() -> Optional[HybridRetriever]:
    """Build the hybrid retriever from currently selected adapters."""
    embedding_model = get_embedding_model()
    vector_db = get_vector_db()
    keyword_db = get_keyword_db()

    if not embedding_model or not vector_db or not keyword_db:
        logger.warning("RAG retriever is unavailable because one or more adapters failed to initialize.")
        return None

    return HybridRetriever(
        vector_db=vector_db,
        keyword_db=keyword_db,
        embedding_model=embedding_model,
        config=RetrievalConfig(
            top_k=settings.RETRIEVAL_TOP_K,
            rrf_k=settings.RRF_K,
            low_relevance_threshold=settings.RAG_LOW_RELEVANCE_THRESHOLD,
        ),
        query_planner=get_query_planner(),
        reranker=get_reranker(),
    )

def get_rag_pipeline() -> Optional[RAGPipeline]:
    retriever = get_retriever()
    if retriever is None:
        return None

    return RAGPipeline(
        retriever=retriever,
        context_packer=get_context_packer(),
        answer_guard=get_answer_guard(),
        low_relevance_threshold=settings.RAG_LOW_RELEVANCE_THRESHOLD,
    )

# -----------------------------------------------------------------------------
# Core services (Stateless Singletons)
# -----------------------------------------------------------------------------

@lru_cache()
def get_session_manager() -> SessionManager:
    return SessionManager(
        memory_store=get_memory_store(),
        context_window_manager=get_context_window_manager(),
        summary_compressor=get_summary_compressor(),
        memory_recall=get_long_term_memory_recall_service(),
        event_publisher=event_bus.publish,
    )

@lru_cache()
def get_context_window_manager() -> PriorityContextWindowManager:
    return PriorityContextWindowManager()

@lru_cache()
def get_summary_compressor() -> ConversationSummaryCompressor:
    return ConversationSummaryCompressor()

@lru_cache()
def get_long_term_memory_recall_service() -> LongTermMemoryRecallService:
    return LongTermMemoryRecallService(get_memory_store())

@lru_cache()
def get_context_engine() -> ContextEngine:
    return ContextEngine(skill_registry=build_default_skill_registry())

@lru_cache()
def get_intent_router() -> IntentRouter:
    return IntentRouter(llm=get_llm())

@lru_cache()
def get_conversation_router() -> ConversationRouter:
    return ConversationRouter(llm=get_llm())

@lru_cache()
def get_task_decomposer() -> TaskDecomposer:
    return TaskDecomposer()

@lru_cache()
def get_profile_extractor() -> ProfileExtractor:
    extractor = ProfileExtractor(memory_store=get_memory_store())
    event_bus.subscribe(
        "conversation.completed",
        extractor.handle_conversation_completed,
    )
    return extractor

# -----------------------------------------------------------------------------
# Agent assembly (Dynamic due to tool binding)
# -----------------------------------------------------------------------------

def _build_agent_tools(
    retriever: Optional[HybridRetriever],
    rag_pipeline: Optional[RAGPipeline],
    kb_app: KnowledgeBaseApp,
) -> List[Callable[..., Any]]:
    tools: List[Callable[..., Any]] = [
        read_local_file,
        write_local_file,
        list_sandbox_files,
        execute_python_sandbox,
    ]
    kb_catalog_tool = KnowledgeBaseCatalogTool(kb_app)
    tools.extend([
            kb_catalog_tool.list_knowledge_base_collections,
            kb_catalog_tool.list_knowledge_base_files,
        ]
    )

    if retriever is not None:
        tools.append(
            KnowledgeBaseTool(
                retriever=retriever,
                rag_pipeline=rag_pipeline,
                collection_name="__all__",
            ).search_knowledge_base
        )

    return tools

def get_tool_registry() -> ToolRegistry:
    # Build tool list dynamically.  Read-only KB/file capabilities are
    # registered below through the in-process MCP boundary; the local
    # callables remain in the initial list as a compatibility fallback, and
    # the provider registration replaces their stable names in the registry.
    retriever = get_retriever()
    rag_pipeline = get_rag_pipeline()
    kb_app = get_kb_app()
    tools_list = _build_agent_tools(retriever, rag_pipeline, kb_app)
    tool_categories: Dict[str, str] = {
        "read_local_file": "read_only",
        "list_sandbox_files": "read_only",
        "write_local_file": "write_file",
        "execute_python_sandbox": "code_exec",
    }
    for tool in tools_list:
        name = getattr(tool, "__name__", None)
        if name and name not in tool_categories:
            tool_categories[name] = "read_only"
    try:
        allowed_categories = set(
            c.strip() for c in settings.ALLOWED_TOOL_CATEGORIES.split(",") if c.strip()
        )
    except Exception:
        allowed_categories = None
    if allowed_categories is not None and settings.ENABLE_DEVELOPMENT_TOOLS:
        allowed_categories.update({"write_file", "code_exec"})

    policy = ToolPolicy(
        max_result_chars=4000,
        tool_timeout_seconds=settings.TOOL_TIMEOUT_SECONDS,
        allowed_categories=allowed_categories,
        tool_categories=tool_categories,
        approval_mode=settings.TOOL_APPROVAL_MODE,
        approval_store=get_approval_store(),
    )
    registry = ToolRegistry.from_callables(tools=tools_list, policy=policy)
    mcp_provider = _build_in_process_mcp_provider(
        retriever=retriever,
        rag_pipeline=rag_pipeline,
        kb_app=kb_app,
    )
    if mcp_provider is not None:
        registry.register_provider(mcp_provider, category="read_only")
    return registry


def _build_in_process_mcp_provider(
    *,
    retriever: Optional[HybridRetriever],
    rag_pipeline: Optional[RAGPipeline],
    kb_app: KnowledgeBaseApp,
) -> MCPToolClientAdapter:
    """Expose the governed read-only tools through the MCP port.

    DeepAssist can still run without a network MCP transport.  The same
    server/client contracts are used in-process, so the default Agent path
    exercises MCP discovery, schema validation, timeout handling, and
    structured error conversion just like an externally hosted MCP server.
    """
    kb_tool = KnowledgeBaseTool(
        retriever=retriever,
        rag_pipeline=rag_pipeline,
        collection_name="__all__",
    )
    server = DeepAssistMCPServer(
        kb_app=kb_app,
        kb_search_provider=(
            (
                lambda query, collection_name: kb_tool.search_knowledge_base(query)
            )
            if retriever is not None
            else None
        ),
    )
    return MCPToolClientAdapter(
        client=server,
        server_name="deepassist",
        tool_aliases={
            "search_kb": "search_knowledge_base",
            "list_kb_files": "list_knowledge_base_files",
            "list_kb_collections": "list_knowledge_base_collections",
            "read_file": "read_local_file",
            "list_files": "list_sandbox_files",
        },
    )


def get_runtime_diagnostics(refresh: bool = False) -> Dict[str, Any]:
    if refresh:
        clear_adapter_cache()

    embedding_model = get_embedding_model()
    vector_db = get_vector_db()
    keyword_db = get_keyword_db()
    memory_store = _get_memory_store_optional()
    retriever = get_retriever()
    rag_pipeline = get_rag_pipeline()
    try:
        agent_runtime = get_agent_runtime()
        agent_runtime_error = None
    except Exception as exc:
        agent_runtime = None
        agent_runtime_error = f"{type(exc).__name__}: {exc}"
    tool_registry = get_tool_registry()
    kb_app = get_kb_app()
    collections_payload = kb_app.list_collections()
    collections = collections_payload.get("data", [])
    if not isinstance(collections, list):
        collections = []

    adapters = {
        "embedding_model": {
            "available": embedding_model is not None,
            "implementation": type(embedding_model).__name__ if embedding_model else None,
            "error": _ADAPTER_ERRORS.get("embedding_model"),
        },
        "vector_db": {
            "available": vector_db is not None,
            "implementation": type(vector_db).__name__ if vector_db else None,
            "error": _ADAPTER_ERRORS.get("vector_db"),
        },
        "keyword_db": {
            "available": keyword_db is not None,
            "implementation": type(keyword_db).__name__ if keyword_db else None,
            "backend": settings.KEYWORD_BACKEND,
            "error": _adapter_error("keyword_db"),
        },
        "memory_store": {
            "available": memory_store is not None,
            "implementation": type(memory_store).__name__ if memory_store else None,
            "backend": settings.MEMORY_BACKEND,
            "error": None if memory_store is not None else _adapter_error("memory_store"),
        },
        "retriever": {
            "available": retriever is not None,
            "implementation": type(retriever).__name__ if retriever else None,
            "error": None if retriever is not None else _rag_unavailable_reason(),
        },
        "reranker": {
            "available": True,
            "implementation": type(get_reranker()).__name__,
            "backend": settings.RERANKER_BACKEND,
            "model": getattr(get_reranker(), "model_name", None),
            "last_backend": getattr(get_reranker(), "last_backend", None),
            "fallback_reason": getattr(get_reranker(), "last_fallback_reason", None),
        },
        "rag_pipeline": {
            "available": rag_pipeline is not None,
            "implementation": type(rag_pipeline).__name__ if rag_pipeline else None,
            "error": None if rag_pipeline is not None else _rag_unavailable_reason(),
        },
        "agent_runtime": {
            "available": agent_runtime is not None,
            "implementation": type(agent_runtime).__name__ if agent_runtime else None,
            "configured_backend": settings.AGENT_RUNTIME_BACKEND,
            "runtime_name": getattr(agent_runtime, "runtime_name", None)
            if agent_runtime
            else None,
            "native_available": getattr(agent_runtime, "native_available", None)
            if agent_runtime
            else None,
            "fallback_reason": getattr(agent_runtime, "fallback_reason", None)
            if agent_runtime
            else None,
            "error": agent_runtime_error,
        },
    }

    return {
        "status": "success",
        "data": {
            "python": {
                "executable": sys.executable,
                "version": sys.version.split()[0],
            },
            "config": {
                "embedding_model_path": settings.EMBEDDING_MODEL_PATH,
                "vector_db_path": settings.VECTOR_DB_PATH,
                "keyword_db_path": settings.KEYWORD_DB_PATH,
                "agent_sandbox_dir": settings.AGENT_SANDBOX_DIR,
                "memory_backend": settings.MEMORY_BACKEND,
                "keyword_backend": settings.KEYWORD_BACKEND,
                "reranker_backend": settings.RERANKER_BACKEND,
                "reranker_model_name": settings.RERANKER_MODEL_NAME,
                "agent_runtime_backend": settings.AGENT_RUNTIME_BACKEND,
                "enable_development_tools": settings.ENABLE_DEVELOPMENT_TOOLS,
                "tool_approval_mode": settings.TOOL_APPROVAL_MODE,
                "allowed_tool_categories": [
                    item.strip()
                    for item in settings.ALLOWED_TOOL_CATEGORIES.split(",")
                    if item.strip()
                ],
            },
            "rag": {
                "available": rag_pipeline is not None,
                "unavailable_reason": None
                if rag_pipeline is not None
                else _rag_unavailable_reason(),
                "adapters": adapters,
                "collections": collections,
            },
            "tools": {
                "allowed_categories": sorted(tool_registry.policy.allowed_categories)
                if tool_registry.policy.allowed_categories is not None
                else None,
                "items": tool_registry.list_tool_specs(),
            },
            "tracing": {
                "available": True,
                "implementation": type(get_trace_store()).__name__,
                "recent_event_count": len(get_trace_store().list_recent(limit=100)),
            },
            "hitl": {
                "available": True,
                "implementation": type(get_approval_store()).__name__,
                "pending_action_count": len(get_approval_store().list(status="pending")),
            },
        },
    }


def _rag_unavailable_reason() -> str:
    missing = []
    if get_embedding_model() is None:
        missing.append("embedding_model")
    if get_vector_db() is None:
        missing.append("vector_db")
    if get_keyword_db() is None:
        missing.append("keyword_db")
    if missing:
        details = {
            name: _adapter_error(name)
            for name in missing
            if _adapter_error(name)
        }
        detail_text = f"; details={details}" if details else ""
        return f"missing adapters: {', '.join(missing)}{detail_text}"
    return "retriever_or_pipeline_unavailable"


def _adapter_error(name: str) -> Optional[str]:
    direct = _ADAPTER_ERRORS.get(name)
    if direct:
        return direct
    prefix = f"{name}:"
    for key, value in _ADAPTER_ERRORS.items():
        if key.startswith(prefix):
            return value
    return None

def get_agent_engine() -> AgentEngine:
    return AgentEngine(
        llm=get_llm(),
        tool_registry=get_tool_registry(),
        run_config=AgentRunConfig(max_iterations=settings.MAX_AGENT_STEPS),
    )

def get_agent_supervisor() -> AgentSupervisor:
    llm = get_llm()
    context_engine = get_context_engine()
    rag_pipeline = get_rag_pipeline()
    chat_worker = ChatWorker(llm=llm, context_engine=context_engine)
    rag_worker = (
        RAGWorker(
            llm=llm,
            context_engine=context_engine,
            rag_pipeline=rag_pipeline,
            collection_name="__all__",
        )
        if rag_pipeline is not None
        else None
    )
    tool_worker = ToolAgentWorker(
        agent_engine=get_agent_engine(),
        context_engine=context_engine,
    )

    return AgentSupervisor(
        intent_router=get_intent_router(),
        chat_worker=chat_worker,
        rag_worker=rag_worker,
        tool_worker=tool_worker,
        orchestrator_worker=OrchestratorWorker(
            llm=llm,
            chat_worker=chat_worker,
            rag_worker=rag_worker,
            tool_worker=tool_worker,
            task_decomposer=get_task_decomposer(),
        ),
        task_decomposer=get_task_decomposer(),
    )


def get_agent_runtime() -> Optional[AgentRuntimePort]:
    backend = (settings.AGENT_RUNTIME_BACKEND or "custom").strip().lower()
    if backend in {"supervisor", "legacy"}:
        return None
    if backend in {"custom", "agent_engine"}:
        from backend.infrastructure.agent_runtime.custom_runtime import (
            CustomAgentRuntimeAdapter,
        )

        return CustomAgentRuntimeAdapter(get_agent_engine())
    if backend in {"langgraph", "lg"}:
        from backend.infrastructure.agent_runtime.langgraph_runtime import (
            LangGraphAgentRuntimeAdapter,
        )

        return LangGraphAgentRuntimeAdapter(get_agent_engine())
    raise ValueError(f"Unsupported AGENT_RUNTIME_BACKEND={settings.AGENT_RUNTIME_BACKEND}")

# -----------------------------------------------------------------------------
# Application workflows (Request Scoped)
# -----------------------------------------------------------------------------

def get_chat_app() -> ChatApplication:
    """Build a request-scoped chat application with current dependencies."""
    return ChatApplication(
        llm=get_llm(),
        session_manager=get_session_manager(),
        context_engine=get_context_engine(),
        intent_router=get_intent_router(),
        profile_extractor=get_profile_extractor(),
        retriever=get_retriever(),
        rag_pipeline=get_rag_pipeline(),
    )

def get_agent_app() -> AgentApplication:
    return AgentApplication(
        agent_engine=get_agent_engine(),
        session_manager=get_session_manager(),
        context_engine=get_context_engine(),
        profile_extractor=get_profile_extractor(),
        supervisor=get_agent_supervisor(),
        agent_runtime=get_agent_runtime(),
    )

def get_conversation_app() -> ConversationApplication:
    return ConversationApplication(
        conversation_router=get_conversation_router(),
        chat_app=get_chat_app(),
        agent_app=get_agent_app(),
        session_manager=get_session_manager(),
        trace_store=get_trace_store(),
    )

@lru_cache()
def get_kb_app() -> KnowledgeBaseApp:
    chunker = DocumentChunker()
    return KnowledgeBaseApp(
        chunker=chunker,
        embedding_model=get_embedding_model(),
        vector_db=get_vector_db(),
        keyword_db=get_keyword_db(),
        manifest_store=get_manifest_store(),
        langchain_ingestion=build_default_ingestion_adapter(chunker),
    )

def get_runtime_app() -> RuntimeApplication:
    return RuntimeApplication(
        kb_app=get_kb_app(),
        tool_registry=get_tool_registry(),
        diagnostics_provider=get_runtime_diagnostics,
        trace_store=get_trace_store(),
        approval_store=get_approval_store(),
        skill_registry=build_default_skill_registry(),
    )

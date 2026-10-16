"""Deterministic unit gates for the capabilities described in the resume."""

import time
from types import SimpleNamespace
from unittest.mock import patch

from backend.application.conversation_app import ConversationApplication
from backend.application.kb_app import KnowledgeBaseApp
from backend.domain.entities.document import DocumentChunk
from backend.domain.entities.retrieval import RetrievalResult
from backend.domain.entities.stream_event import StreamEvent
from backend.domain.entities.tooling import ToolCall
from backend.domain.entities.agent_worker import AgentWorkerType
from backend.domain.services.agent.engine import AgentEngine
from backend.domain.services.agent.workers import OrchestratorWorker
from backend.domain.services.agent.tooling import ToolPolicy, ToolRegistry
from backend.domain.services.rag.chunking import DocumentChunker
from backend.domain.services.rag.answer_guard import SourceAwareResponseGuard
from backend.domain.services.rag.context_packer import ContextPacker, ContextPackingConfig
from backend.domain.services.rag.fusion import HybridRetriever, RetrievalConfig
from backend.infrastructure.databases.chroma_store import ChromaStore
from backend.infrastructure.embeddings.bge_m3_local import BGEM3Local
from backend.infrastructure.tracing.local_trace_store import LocalTraceStore
from backend.infrastructure.tracing.langsmith_trace_store import LangSmithTraceStore
from backend.infrastructure.rag.langchain_adapters import (
    LangChainIngestionAdapter,
    LangChainRetrieverAdapter,
    LangChainRunnableAdapter,
    LangChainVectorStoreAdapter,
    build_default_ingestion_adapter,
)
from backend.infrastructure.agent_runtime.custom_runtime import engine_event_to_stream_events
from backend.infrastructure.mcp.server import DeepAssistMCPServer
from backend.common.config import settings


class _FakeAgentLLM:
    model_name = "deepseek-chat"

    def chat(self, messages, tools=None, **kwargs):
        return SimpleNamespace(content="done", reasoning_content=None, tool_calls=[])


    def chat_stream(self, messages, **kwargs):
        yield {"type": "content", "content": "done"}


def test_resume_defaults_use_cross_encoder_configuration():
    assert settings.RERANKER_BACKEND == "cross_encoder"
    assert settings.RERANKER_MODEL_NAME == "BAAI/bge-reranker-base"


def test_agent_engine_records_completed_model_call():
    events = list(
        AgentEngine(llm=_FakeAgentLLM()).stream_run(
            [{"role": "user", "content": "hello"}],
            model_options={"model_name": "deepseek-chat"},
        )
    )
    model_events = [event for event in events if event.get("type") == "model_call"]
    assert len(model_events) == 1
    assert model_events[0]["data"]["success"] is True
    assert model_events[0]["data"]["model_name"] == "deepseek-chat"


def test_agent_engine_emits_tool_result_and_state_snapshot():
    class ToolLLM(_FakeAgentLLM):
        def __init__(self):
            self.calls = 0

        def chat(self, messages, tools=None, **kwargs):
            self.calls += 1
            if self.calls == 1:
                return SimpleNamespace(
                    content="",
                    reasoning_content=None,
                    tool_calls=[
                        {
                            "id": "lookup-1",
                            "type": "function",
                            "function": {
                                "name": "lookup",
                                "arguments": '{"value":"ok"}',
                            },
                        }
                    ],
                )
            return SimpleNamespace(content="final", reasoning_content=None, tool_calls=[])

    def lookup(value: str) -> str:
        return value

    events = list(
        AgentEngine(
            llm=ToolLLM(),
            tool_registry=ToolRegistry.from_callables([lookup]),
        ).stream_run([{"role": "user", "content": "lookup"}])
    )
    tool_results = [event for event in events if event.get("type") == "tool_result"]
    snapshots = [event for event in events if event.get("type") == "task_snapshot"]
    assert tool_results and tool_results[0]["success"] is True
    assert snapshots and "lifecycle_state" in snapshots[-1]["data"]


def test_custom_runtime_maps_model_call_to_transport_event():
    event = next(
        engine_event_to_stream_events(
            {"type": "model_call", "data": {"phase": "completed", "success": True}}
        )
    )
    assert event.event == "model_call"
    assert event.data["phase"] == "completed"


def test_mcp_server_exposes_governed_knowledge_and_file_tools():
    server = DeepAssistMCPServer(
        file_read_provider=lambda name: f"read:{name}",
        file_list_provider=lambda path: f"list:{path}",
    )
    names = {item["name"] for item in server.list_tools()}
    assert {"search_kb", "list_kb_files", "read_file", "list_files"}.issubset(names)
    assert server.call_tool("read_file", {"file_name": "notes.md"})["structuredContent"]["data"] == "read:notes.md"
    assert server.call_tool("list_files", {"path": "."})["structuredContent"]["data"] == "list:."


def test_default_tool_registry_routes_read_only_kb_and_file_tools_through_mcp():
    from backend.api import dependencies as dependency_module

    class FakeRetriever:
        def retrieve_with_trace(self, collection_name, query):
            return SimpleNamespace(
                documents=[DocumentChunk(id="doc-1", content=f"evidence:{query}")]
            )

    class FakeKBApp:
        def list_collections(self):
            return {"status": "success", "data": []}

        def list_files(self, collection_name="tech_docs_kb"):
            return {"status": "success", "data": []}

    with patch.object(dependency_module, "get_retriever", return_value=FakeRetriever()):
        with patch.object(dependency_module, "get_rag_pipeline", return_value=None):
            with patch.object(dependency_module, "get_kb_app", return_value=FakeKBApp()):
                registry = dependency_module.get_tool_registry()

    specs = {item["name"]: item for item in registry.list_tool_specs()}
    assert specs["search_knowledge_base"]["provider"] == "mcp"
    assert specs["list_knowledge_base_files"]["provider"] == "mcp"
    assert specs["read_local_file"]["provider"] == "mcp"
    assert specs["list_sandbox_files"]["provider"] == "mcp"

    result = registry.execute(
        ToolCall(id="mcp-read-1", name="read_local_file", args={"file_name": "missing.md"})
    )
    assert result.success is True
    assert "does not exist" in result.content

    search = registry.execute(
        ToolCall(id="mcp-search-1", name="search_knowledge_base", args={"query": "DeepAssist"})
    )
    assert search.success is True
    assert "evidence:DeepAssist" in search.content


def test_tool_policy_enforces_execution_timeout():
    def slow_tool() -> str:
        time.sleep(0.08)
        return "late"

    registry = ToolRegistry.from_callables(
        [slow_tool],
        policy=ToolPolicy(tool_timeout_seconds=0.01),
    )
    result = registry.execute(ToolCall(id="timeout-1", name="slow_tool", args={}))
    assert result.success is False
    assert result.metadata["error_type"] == "tool_timeout"
    assert result.metadata["timeout_seconds"] == 0.01


def test_tool_policy_validates_arguments_and_permission_boundaries():
    def read_record(record_id: int) -> str:
        return str(record_id)

    registry = ToolRegistry.from_callables(
        [read_record],
        policy=ToolPolicy(
            allowed_categories={"read_only"},
            tool_categories={"read_record": "read_only"},
        ),
    )
    invalid = registry.execute(
        ToolCall(id="invalid-1", name="read_record", args={"record_id": "bad"})
    )
    assert invalid.metadata["error_type"] == "invalid_arguments"
    assert invalid.metadata["repair_strategy"] == "fix_arguments"

    blocked_registry = ToolRegistry.from_callables(
        [read_record],
        policy=ToolPolicy(
            allowed_categories={"write_file"},
            tool_categories={"read_record": "read_only"},
        ),
    )
    blocked = blocked_registry.execute(
        ToolCall(id="blocked-1", name="read_record", args={"record_id": 1})
    )
    assert blocked.metadata["error_type"] == "tool_blocked"
    assert blocked.metadata["retryable"] is False


def test_bge_m3_adapter_normalizes_single_and_batch_embeddings():
    class Encoded:
        def __init__(self, value):
            self.value = value

        def tolist(self):
            return self.value

    class FakeModel:
        def __init__(self):
            self.calls = []

        def encode(self, value, **kwargs):
            self.calls.append((value, kwargs))
            return Encoded([0.6, 0.8] if isinstance(value, str) else [[0.6, 0.8] for _ in value])

    adapter = BGEM3Local.__new__(BGEM3Local)
    adapter.model = FakeModel()
    adapter.batch_size = 4
    assert adapter.embed_text("query") == [0.6, 0.8]
    assert adapter.embed_documents(["a", "b"]) == [[0.6, 0.8], [0.6, 0.8]]
    assert adapter.model.calls[0][1]["normalize_embeddings"] is True
    assert adapter.model.calls[1][1]["batch_size"] == 4


def test_chroma_adapter_adds_and_reads_document_chunks():
    class Collection:
        def __init__(self):
            self.added = None

        def add(self, **kwargs):
            self.added = kwargs

        def query(self, **kwargs):
            return {
                "ids": [["c1"]],
                "documents": [["content"]],
                "metadatas": [[{"source_file": "a.md"}]],
                "distances": [[0.2]],
            }

    class Client:
        def __init__(self):
            self.collection = Collection()

        def get_or_create_collection(self, name):
            return self.collection

        def get_collection(self, name):
            return self.collection

    store = ChromaStore.__new__(ChromaStore)
    store.client = Client()
    chunk = DocumentChunk(id="c1", content="content", metadata={"source_file": "a.md"})
    assert store.add_chunks("kb", [chunk], [[0.1, 0.2]]) is True
    found = store.search("kb", [0.1, 0.2], top_k=1)
    assert found[0].id == "c1"
    assert found[0].score == 0.8


def test_hybrid_retrieval_records_weighted_rrf_contributions():
    class Store:
        def __init__(self, chunks):
            self.chunks = chunks

        def search(self, collection_name, query, top_k):
            return self.chunks[:top_k]

    class Embedding:
        def embed_text(self, text):
            return [1.0]

    vector = [
        DocumentChunk(id="a", content="alpha", score=0.9),
        DocumentChunk(id="b", content="beta", score=0.8),
    ]
    keyword = [
        DocumentChunk(id="b", content="beta", score=4.0),
        DocumentChunk(id="c", content="gamma", score=3.0),
    ]
    retriever = HybridRetriever(
        vector_db=Store(vector),
        keyword_db=Store(keyword),
        embedding_model=Embedding(),
        config=RetrievalConfig(top_k=3, candidate_multiplier=1, rrf_k=10, enable_rerank=False),
    )
    result = retriever.retrieve_with_trace("kb", "alpha beta")
    by_id = {item.id: item for item in result.documents}
    assert by_id["b"].metadata["retrieval"]["vector_rrf"] > 0
    assert by_id["b"].metadata["retrieval"]["keyword_rrf"] > 0
    assert by_id["b"].score > by_id["a"].score


def test_document_chunker_keeps_heading_source_and_overlap_metadata():
    chunker = DocumentChunker(max_tokens=10, overlap_tokens=3, min_chars=3)
    chunks = chunker.split_markdown(
        "# Architecture\n\nfirst paragraph with enough words.\n\nsecond paragraph with enough words.",
        source_name="README.md",
    )
    assert len(chunks) >= 2
    assert all(item.metadata["source_file"] == "README.md" for item in chunks)
    assert chunks[0].metadata["heading_path"] == "Architecture"
    assert all(item.metadata["token_count"] > 0 for item in chunks)


def test_kb_ingestion_separates_chunk_embedding_and_dual_index_writes():
    class Chunker:
        def split_markdown(self, text, source_name):
            return [
                DocumentChunk(
                    id="chunk-1",
                    content=text,
                    metadata={"source_file": source_name, "heading_path": "Root"},
                )
            ]

    class Embedding:
        def __init__(self):
            self.inputs = None

        def embed_documents(self, texts):
            self.inputs = texts
            return [[0.1, 0.2] for _ in texts]

    class Vector:
        def __init__(self):
            self.calls = []

        def add_chunks(self, collection, chunks, embeddings):
            self.calls.append((collection, chunks, embeddings))
            return True

        def delete_by_source(self, collection, source):
            return True

    class Keyword:
        def __init__(self):
            self.calls = []

        def build_index(self, collection, chunks):
            self.calls.append((collection, chunks))
            return True

    embedding, vector, keyword = Embedding(), Vector(), Keyword()
    app = KnowledgeBaseApp(Chunker(), embedding, vector, keyword)
    result = app.process_document("guide.md", "text", "kb")
    assert result["status"] == "success"
    assert embedding.inputs == ["text"]
    assert vector.calls[0][2] == [[0.1, 0.2]]
    assert keyword.calls[0][0] == "kb"


def test_context_packing_and_citation_guard_validate_grounding():
    retrieval = RetrievalResult(
        query="question",
        collection_name="kb",
        documents=[
            DocumentChunk(
                id="c1",
                content="A supported fact with enough context for the answer.",
                metadata={"source_file": "guide.md", "heading_path": "Root > Facts"},
                score=0.9,
            )
        ],
        top_k=1,
        candidate_k=1,
    )
    pack = ContextPacker(
        ContextPackingConfig(budget_chars=1000, per_chunk_max_chars=500, min_chunk_chars=10)
    ).pack("question", retrieval)
    report = SourceAwareResponseGuard().check("The fact is supported [C1].", pack)
    assert report.grounded is True
    assert report.used_citations == ["C1"]
    unknown = SourceAwareResponseGuard().check("Unsupported [C9].", pack)
    assert unknown.grounded is False
    assert "answer_has_unknown_citations" in unknown.warnings


def test_conversation_trace_preserves_model_call_payload():
    trace_store = LocalTraceStore()
    app = object.__new__(ConversationApplication)
    app.trace_store = trace_store
    event = StreamEvent.model_call(
        {"phase": "completed", "model_name": "deepseek-chat", "latency_ms": 12.5}
    )
    from backend.infrastructure.streaming.sse_manager import SSEManager

    app._trace_sse_chunk(
        SSEManager.format_event(event),
        trace_id="trace-model-call",
        session_id="session-model-call",
        route="agent",
    )
    traces = trace_store.list_by_trace_id("trace-model-call")
    model_trace = next(item for item in traces if item.event_type == "model_call")
    assert model_trace.metadata["event_data"]["model_name"] == "deepseek-chat"


def test_langsmith_trace_store_mirrors_local_event():
    class FakeLangSmithClient:
        def __init__(self):
            self.runs = []

        def create_run(self, **kwargs):
            self.runs.append(kwargs)

    client = FakeLangSmithClient()
    store = LangSmithTraceStore(client=client, project_name="deepassist-test")
    event = StreamEvent.model_call(
        {"phase": "completed", "model_name": "deepseek-chat", "latency_ms": 3.0}
    )
    from backend.domain.entities.trace_event import TraceEvent

    trace = TraceEvent(
        trace_id="trace-langsmith",
        route="agent",
        event_type="model_call",
        span_name="agent.model_call",
        latency_ms=3.0,
        metadata={"event_data": event.data},
    )
    store.append(trace)

    assert store.list_by_trace_id("trace-langsmith") == [trace]
    assert client.runs[0]["project_name"] == "deepassist-test"
    assert client.runs[0]["run_type"] == "llm"
    assert client.runs[0]["extra"]["metadata"]["deepassist_trace_id"] == "trace-langsmith"


def test_orchestrator_reflects_and_replans_at_most_once():
    class FakeWorker:
        def __init__(self, worker_type, answer):
            self.worker_type = worker_type
            self.answer = answer
            self.calls = 0

        def stream(self, **kwargs):
            self.calls += 1
            yield {"type": "status", "content": "worker"}
            if self.answer:
                yield {"type": "message_delta", "content": self.answer}
            yield {
                "type": "finish",
                "new_messages": [
                    {"role": "assistant", "content": self.answer}
                ],
            }

    class FakeLLM:
        model_name = "test-model"

        def chat_stream(self, messages, **kwargs):
            yield "synthesized"

    rag_worker = FakeWorker(AgentWorkerType.RAG, "")
    tool_worker = FakeWorker(AgentWorkerType.TOOL, "tool evidence")
    chat_worker = FakeWorker(AgentWorkerType.CHAT, "chat evidence")
    worker = OrchestratorWorker(
        llm=FakeLLM(),
        chat_worker=chat_worker,
        rag_worker=rag_worker,
        tool_worker=tool_worker,
    )

    events = list(
        worker.stream(
            query="Use rag to search docs, then use python to calculate.",
            history=[],
        )
    )

    reflections = [event for event in events if event.get("type") == "reflection"]
    plans = [event for event in events if event.get("type") == "multi_agent_plan"]
    replan_starts = [
        event
        for event in events
        if event.get("type") == "collaborator_trace"
        and event.get("data", {}).get("reason") == "reflection_replan"
        and event.get("data", {}).get("phase") == "start"
    ]

    assert reflections and reflections[0]["data"]["should_replan"] is True
    assert len(replan_starts) == 1
    assert plans[-1]["data"]["replan_count"] == 1
    assert plans[-1]["data"]["max_replans"] == 1
    assert rag_worker.calls == 2


def test_langchain_retriever_and_runnable_adapters_normalize_framework_values():
    class FakeDocument:
        id = "doc-1"
        page_content = "content"
        metadata = {"source_file": "guide.md", "score": 0.8}

    class FakeRetriever:
        def invoke(self, query, config=None):
            assert query == "question"
            return [FakeDocument()]

    class FakeRunnable:
        def invoke(self, value, config=None):
            return {"answer": value}

        def stream(self, value, config=None):
            yield "a"
            yield "b"

    documents = LangChainRetrieverAdapter(FakeRetriever()).invoke("question")
    runnable = LangChainRunnableAdapter(FakeRunnable())
    assert documents[0].id == "doc-1"
    assert documents[0].content == "content"
    assert documents[0].score == 0.8
    assert runnable.invoke("question")["answer"] == "question"
    assert list(runnable.stream("question")) == ["a", "b"]


def test_langchain_loader_splitter_and_kb_ingestion_boundary_are_connected():
    class FakeLoader:
        def __init__(self):
            self.sources = []

        def load(self, source):
            self.sources.append(source)
            return [{"page_content": source, "metadata": {"title": "Guide"}}]

    class FakeSplitter:
        def split_documents(self, documents):
            return [
                {
                    "id": "langchain-chunk-1",
                    "page_content": documents[0]["page_content"].upper(),
                    "metadata": documents[0]["metadata"],
                }
            ]

    class Embedding:
        def embed_documents(self, texts):
            return [[0.1, 0.2] for _ in texts]

    class Vector:
        def __init__(self):
            self.chunks = []

        def add_chunks(self, collection, chunks, embeddings):
            self.chunks.extend(chunks)
            return True

        def delete_by_source(self, collection, source):
            return True

    class Keyword:
        def build_index(self, collection, chunks):
            return True

    loader = FakeLoader()
    ingestion = LangChainIngestionAdapter(loader, FakeSplitter())
    app = KnowledgeBaseApp(
        chunker=None,
        embedding_model=Embedding(),
        vector_db=Vector(),
        keyword_db=Keyword(),
        langchain_ingestion=ingestion,
    )

    result = app.process_document("guide.md", "langchain input", "kb")

    assert result["status"] == "success"
    assert loader.sources == ["langchain input"]
    assert app.vector_db.chunks[0].content == "LANGCHAIN INPUT"
    assert app.vector_db.chunks[0].metadata["source_file"] == "guide.md"


def test_default_kb_upload_uses_langchain_ingestion_boundary():
    ingestion = build_default_ingestion_adapter(DocumentChunker())

    chunks = ingestion.load_and_split(
        "# Guide\n\nDeepAssist keeps document provenance for every chunk.",
        source_name="guide.md",
    )

    assert chunks
    assert chunks[0].metadata["source_file"] == "guide.md"
    assert chunks[0].metadata["heading_path"] == "Guide"


def test_langchain_vector_store_adapter_uses_project_document_contract():
    class FakeVectorStore:
        def __init__(self):
            self.documents = []
            self.deleted = None

        def add_documents(self, documents, ids=None):
            self.documents.extend(documents)
            assert ids == ["chunk-1"]

        def similarity_search_by_vector_with_relevance_scores(self, vector, k):
            return [(self.documents[0], 0.91)][:k]

        def delete(self, where=None):
            self.deleted = where

        def get(self, include=None):
            return {"metadatas": [document.metadata for document in self.documents]}

    store = FakeVectorStore()
    adapter = LangChainVectorStoreAdapter(store, collection_name="kb")
    chunk = DocumentChunk(
        id="chunk-1",
        content="content",
        metadata={"source_file": "guide.md"},
    )

    assert adapter.add_chunks("kb", [chunk], [[0.1, 0.2]]) is True
    found = adapter.search("kb", [0.1, 0.2], 1)
    assert found[0].id == "chunk-1"
    assert found[0].score == 0.91
    assert adapter.list_sources("kb")[0].source_file == "guide.md"
    assert adapter.delete_by_source("kb", "guide.md") is True
    assert store.deleted == {"source_file": "guide.md"}


def test_langchain_vector_store_adapter_preserves_project_embeddings():
    class FakeVectorStore:
        def __init__(self):
            self.embeddings = []
            self.metadatas = []
            self.ids = []

        def add_embeddings(self, text_embeddings, metadatas=None, ids=None):
            self.embeddings.extend(text_embeddings)
            self.metadatas.extend(metadatas or [])
            self.ids.extend(ids or [])

    store = FakeVectorStore()
    adapter = LangChainVectorStoreAdapter(store, collection_name="kb")
    chunk = DocumentChunk(
        id="chunk-embedding-1",
        content="content",
        metadata={"source_file": "guide.md"},
    )

    assert adapter.add_chunks("kb", [chunk], [[0.3, 0.4]]) is True
    assert store.embeddings == [("content", [0.3, 0.4])]
    assert store.metadatas == [{"source_file": "guide.md"}]
    assert store.ids == ["chunk-embedding-1"]

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

from backend.application.agent_app import AgentApplication
from backend.application.chat_app import ChatApplication
from backend.application.conversation_app import ConversationApplication
from backend.application.kb_app import KnowledgeBaseApp
from backend.application.runtime_app import RuntimeApplication
from backend.common.config import settings
from backend.common.exceptions import ConfigurationError
from backend.domain.entities.context_window import ContextSummary
from backend.domain.entities.agent_run import AgentRunConfig
from backend.domain.entities.agent_plan import MultiAgentPlan
from backend.domain.entities.agent_worker import AgentWorkerType
from backend.domain.entities.document import DocumentChunk
from backend.domain.entities.knowledge_base import KnowledgeBaseHealthReport
from backend.domain.entities.message import Message
from backend.domain.entities.rag_pipeline import RAGPipelineResult
from backend.domain.entities.retrieval import Citation, RAGContextPack, RerankTrace, RetrievalResult
from backend.domain.entities.route import ConversationRoute, RouteDecision, SourcePolicy
from backend.domain.entities.stream_event import StreamEvent
from backend.domain.entities.task_snapshot import TaskSnapshot
from backend.domain.entities.tooling import ToolCall
from backend.domain.entities.trace_event import TraceEvent
from backend.domain.interfaces.agent_runtime import AgentRuntimeRequest
from backend.domain.interfaces.memory_db import BaseMemoryStore
from backend.infrastructure.approvals.local_approval_store import LocalApprovalStore
from backend.infrastructure.tools.kb_catalog_tool import KnowledgeBaseCatalogTool
from backend.infrastructure.tools.rag_tool import KnowledgeBaseTool
from backend.infrastructure.tools.file_ops import (
    list_sandbox_files,
    read_local_file,
    write_local_file,
)
from backend.infrastructure.tools.python_sandbox import execute_python_sandbox
from backend.domain.services.agent.engine import AgentEngine
from backend.domain.services.agent.intent_router import IntentRouter
from backend.domain.services.agent.supervisor import AgentSupervisor
from backend.domain.services.agent.task_decomposer import TaskDecomposer
from backend.domain.services.agent.tooling import ToolPolicy, ToolRegistry
from backend.domain.services.agent.workers import OrchestratorWorker, ToolAgentWorker
from backend.domain.services.conversation.router import ConversationRouter
from backend.domain.services.session.context_engine import ContextEngine
from backend.domain.services.rag.answer_guard import SourceAwareResponseGuard
from backend.domain.services.rag.fusion import HybridRetriever
from backend.domain.services.rag.query_planner import QueryPlanner
from backend.domain.services.rag.query_rewriter import QueryRewriteService
from backend.domain.services.rag.reranker import LexicalOverlapReranker
from backend.infrastructure.rerankers.cross_encoder_reranker import CrossEncoderReranker
from backend.infrastructure.rerankers.factory import build_reranker
from backend.infrastructure.rerankers.noop_reranker import NoOpReranker
from backend.infrastructure.memory.mysql_memory import MySQLMemoryStoreAdapter
from backend.infrastructure.search.elasticsearch_store import ElasticsearchKeywordSearchAdapter
from backend.infrastructure.agent_runtime.custom_runtime import CustomAgentRuntimeAdapter
from backend.infrastructure.agent_runtime.custom_runtime import engine_event_to_stream_events
from backend.infrastructure.agent_runtime.langgraph_runtime import LangGraphAgentRuntimeAdapter
from backend.infrastructure.mcp.client_adapter import MCPToolClientAdapter
from backend.infrastructure.mcp.server import DeepAssistMCPServer
from backend.infrastructure.tracing.local_trace_store import LocalTraceStore
from backend.infrastructure.tool_adapters.langchain_adapter import LangChainToolAdapter
from backend.domain.services.session.manager import SessionManager
from backend.infrastructure.databases.kb_manifest_store import KnowledgeBaseManifestStore
from backend.infrastructure.llms.deepseek_client import DeepSeekClient
from evals.dataset_catalog import DATASET_SPECS
from evals.prepare_datasets import convert_agent_row, convert_rag_row, validate_case
from evals.report import generate_report
from evals.run_eval import build_eval_report, build_scaffold_report, run_benchmark_eval


def parse_sse_events(chunks: list[str]) -> list[dict[str, Any]]:
    payloads = []
    for chunk in chunks:
        if not chunk.startswith("data: "):
            continue
        raw = chunk[6:].strip()
        if raw and raw != "[DONE]":
            payloads.append(json.loads(raw))
    return payloads


class FakeMemoryStore(BaseMemoryStore):
    def __init__(self):
        self.messages: dict[str, list[Any]] = {}
        self.profiles: dict[str, str] = {}
        self.task_snapshots: dict[str, TaskSnapshot] = {}
        self.session_summaries: dict[str, ContextSummary] = {}
        self.routing_logs: list[dict[str, Any]] = []

    def get_history(self, session_id: str, limit: int = 10):
        return list(self.messages.get(session_id, []))[-limit:]

    def add_message(self, session_id: str, message):
        self.messages.setdefault(session_id, []).append(message)
        return True

    def clear_history(self, session_id: str) -> bool:
        self.messages.pop(session_id, None)
        return True

    def get_all_sessions(self):
        return []

    def get_profile(self, key: str):
        return self.profiles.get(key)

    def set_profile(self, key: str, value: str) -> bool:
        self.profiles[key] = value
        return True

    def get_all_profiles(self):
        return dict(self.profiles)

    def get_task_snapshot(self, session_id: str):
        return self.task_snapshots.get(session_id)

    def save_task_snapshot(self, snapshot: TaskSnapshot) -> bool:
        self.task_snapshots[snapshot.session_id] = snapshot
        return True

    def clear_task_snapshot(self, session_id: str) -> bool:
        self.task_snapshots.pop(session_id, None)
        return True

    def get_session_summary(self, session_id: str):
        return self.session_summaries.get(session_id)

    def save_session_summary(self, session_id: str, summary: ContextSummary) -> bool:
        self.session_summaries[session_id] = summary
        return True

    def clear_session_summary(self, session_id: str) -> bool:
        self.session_summaries.pop(session_id, None)
        return True

    def save_routing_log(self, log_data: dict[str, Any]) -> bool:
        self.routing_logs.append(log_data)
        return True


class FakeLLM:
    def __init__(self, chunks: list[str]):
        self.chunks = chunks
        self.calls: list[list[dict[str, Any]]] = []

    def chat_stream(self, messages, model_name=None, temperature=None, top_p=None):
        self.calls.append(messages)
        for chunk in self.chunks:
            yield chunk


class FakeRouter:
    def __init__(self, decision: RouteDecision):
        self.decision = decision
        self.calls: list[dict[str, Any]] = []

    def route(self, query: str, history=None, **kwargs):
        self.calls.append({"query": query, "history": history or [], **kwargs})
        return self.decision


class FakeAgentLLM:
    def __init__(self, responses: list[Any]):
        self.responses = responses
        self.calls: list[list[dict[str, Any]]] = []

    def chat(self, messages, tools=None, model_name=None, temperature=None, top_p=None):
        self.calls.append(messages)
        return self.responses[len(self.calls) - 1]


class FakeRouteLLM:
    def __init__(self, content: str):
        self.content = content
        self.calls: list[dict[str, Any]] = []

    def chat(self, messages, tools=None, model_name=None, temperature=None, top_p=None):
        self.calls.append(
            {
                "messages": messages,
                "model_name": model_name,
                "temperature": temperature,
            }
        )
        return SimpleNamespace(content=self.content)


class FakeProfileExtractor:
    def render_profile(self) -> str:
        return "- role: platform engineer"


class NoCallPipeline:
    def build_context(self, query: str, collection_name: str):
        raise AssertionError("RAG pipeline should not run for small-talk downgrade")


class FakeRAGPipeline:
    def __init__(self, score: float = 0.8):
        self.score = score
        self.calls: list[tuple[str, str]] = []

    def build_context(self, query: str, collection_name: str) -> RAGPipelineResult:
        self.calls.append((query, collection_name))
        retrieval = RetrievalResult(
            query=query,
            collection_name=collection_name,
            documents=[
                DocumentChunk(
                    id="chunk-1",
                    content="DeepSeek API Key guide",
                    metadata={"source_file": "kb.md"},
                    score=self.score,
                )
            ],
            top_k=5,
            candidate_k=10,
            metadata={
                "diagnostics": {
                    "reason_code": "ok" if self.score >= 0.35 else "low_relevance",
                    "reason_message": "retrieval_ready"
                    if self.score >= 0.35
                    else "top_document_score_below_threshold",
                    "suggested_action": "proceed_with_rag"
                    if self.score >= 0.35
                    else "fallback_to_chat",
                    "best_score": self.score,
                    "low_relevance_threshold": 0.35,
                }
            },
            rerank=RerankTrace(
                enabled=True,
                success=True,
                model="fake",
                input_count=1,
                returned=1,
            ),
        )
        context_pack = RAGContextPack(
            query=query,
            citations=[
                Citation(
                    ref_id="C1",
                    chunk_id="chunk-1",
                    content="DeepSeek API Key guide",
                    source_file="kb.md",
                    score=self.score,
                )
            ],
            rendered_context="[C1] DeepSeek API Key guide",
            budget_chars=1000,
            used_chars=28,
        )
        return RAGPipelineResult(
            query=query,
            collection_name=collection_name,
            retrieval_result=retrieval,
            context_pack=context_pack,
        )

    def check_answer(self, answer: str, pipeline_result: RAGPipelineResult):
        class Report:
            grounded = True
            recommended_action = "accept"
            reason = "answer_grounded"

            def to_stream_data(self):
                return {
                    "grounded": True,
                    "recommended_action": "accept",
                    "reason": "answer_grounded",
                    "warnings": [],
                }

        return Report()


class FakeSupervisor:
    def stream(self, query: str, history, user_profile=None, model_options=None):
        yield {"type": "message_delta", "content": "partial-agent"}


class FakeHealthInspector:
    def __init__(self):
        self.load_calls = 0
        self.inspect_calls: list[dict[str, Any]] = []
        self.cached_report = KnowledgeBaseHealthReport(
            vector_db_path="vector",
            keyword_db_path="keyword",
            checked_at="2026-05-04T16:00:00",
        )
        self.live_report = KnowledgeBaseHealthReport(
            vector_db_path="vector",
            keyword_db_path="keyword",
            checked_at="2026-05-04T16:05:00",
        )

    def load_report(self):
        self.load_calls += 1
        return self.cached_report

    def inspect(self, collections=None, repair=False, batch_size=256, persist=True):
        self.inspect_calls.append(
            {
                "collections": collections,
                "repair": repair,
                "batch_size": batch_size,
                "persist": persist,
            }
        )
        return self.live_report


class FakeKBApp:
    def list_collections(self):
        return {
            "status": "success",
            "data": [
                {
                    "collection_name": "medical_kb",
                    "file_count": 2,
                    "chunk_count": 18,
                    "stores": ["keyword", "vector"],
                },
                {
                    "collection_name": "tech_docs_kb",
                    "file_count": 4,
                    "chunk_count": 36,
                    "stores": ["keyword", "vector"],
                },
            ],
        }


class SlowListSourcesVectorDB:
    def list_collections(self):
        return ["medical_kb"]

    def list_sources(self, collection_name: str):
        raise AssertionError("list_sources should not be called when manifest summary exists")


class SlowListSourcesKeywordDB:
    def list_collections(self):
        return ["medical_kb"]

    def list_sources(self, collection_name: str):
        raise AssertionError("list_sources should not be called when manifest summary exists")


class FakeEmbeddingModel:
    def embed_text(self, query: str):
        return [0.1, 0.2, 0.3]


class FakeVectorDB:
    def __init__(self, docs):
        self.docs = docs

    def search(self, collection_name, query_vector, top_k):
        return list(self.docs)[:top_k]


class FakeKeywordDB:
    def __init__(self, docs):
        self.docs = docs

    def search(self, collection_name, query, top_k):
        return list(self.docs)[:top_k]

    def list_files(self, collection_name: str = "tech_docs_kb"):
        return {
            "status": "success",
            "data": [
                {
                    "source_file": "guide.md",
                    "chunk_count": 12,
                    "consistent": True,
                }
            ],
        }


class FakeCollaboratorWorker:
    def __init__(self, content: str, extra_events: list[dict[str, Any]] | None = None):
        self.content = content
        self.extra_events = extra_events or []
        self.calls: list[dict[str, Any]] = []

    def stream(self, query: str, history, user_profile=None, model_options=None, recovery_state=None):
        self.calls.append(
            {
                "query": query,
                "history": history,
                "user_profile": user_profile,
                "model_options": model_options,
                "recovery_state": recovery_state,
            }
        )
        for event in self.extra_events:
            yield event
        yield {
            "type": "finish",
            "new_messages": [{"role": "assistant", "content": self.content}],
            "worker": "fake_worker",
        }


class FakeSynthesisLLM:
    def __init__(self, chunks: list[str]):
        self.chunks = chunks
        self.calls: list[list[dict[str, Any]]] = []

    def chat_stream(self, messages, model_name=None, temperature=None, top_p=None):
        self.calls.append(messages)
        for chunk in self.chunks:
            yield chunk


class FakeRecoverySupervisor:
    def __init__(self):
        self.resume_calls: list[dict[str, Any]] = []

    def stream(self, query: str, history, user_profile=None, model_options=None):
        yield {"type": "message_delta", "content": "fresh-run"}
        yield {
            "type": "finish",
            "new_messages": [{"role": "assistant", "content": "fresh-run"}],
            "worker": "tool_agent_worker",
        }

    def stream_recovery(
        self,
        worker_kind: str,
        query: str,
        history,
        user_profile=None,
        model_options=None,
        recovery_state=None,
    ):
        self.resume_calls.append(
            {
                "worker_kind": worker_kind,
                "query": query,
                "history": history,
                "recovery_state": recovery_state,
            }
        )
        yield {"type": "message_delta", "content": "resumed-run"}
        yield {
            "type": "finish",
            "new_messages": [{"role": "assistant", "content": "resumed-run"}],
            "worker": worker_kind,
        }


class BackendRegressionTests(unittest.TestCase):
    def test_chat_disconnect_persists_partial_response(self):
        store = FakeMemoryStore()
        session_manager = SessionManager(store)
        app = ChatApplication(
            llm=FakeLLM(["partial-chat", " rest"]),
            session_manager=session_manager,
            context_engine=ContextEngine(),
            intent_router=IntentRouter(),
        )

        stream = app.stream_chat("session-chat", "remember me", "quick")
        for chunk in stream:
            if '"event": "message_delta"' in chunk:
                break
        stream.close()

        saved = [message.model_dump() for message in store.messages["session-chat"]]
        self.assertEqual(saved[0]["role"], "user")
        self.assertEqual(saved[0]["content"], "remember me")
        self.assertEqual(saved[1]["role"], "assistant")
        self.assertEqual(saved[1]["content"], "partial-chat")

    def test_agent_disconnect_persists_partial_response(self):
        store = FakeMemoryStore()
        session_manager = SessionManager(store)
        app = AgentApplication(
            agent_engine=None,
            session_manager=session_manager,
            context_engine=ContextEngine(),
            profile_extractor=FakeProfileExtractor(),
            supervisor=FakeSupervisor(),
        )

        stream = app.stream_agent_task("session-agent", "do task")
        for chunk in stream:
            if '"event": "message_delta"' in chunk:
                break
        stream.close()

        saved = [message.model_dump() for message in store.messages["session-agent"]]
        self.assertEqual(saved[0]["role"], "user")
        self.assertEqual(saved[0]["content"], "do task")
        self.assertEqual(saved[1]["role"], "assistant")
        self.assertEqual(saved[1]["content"], "partial-agent")

    def test_rag_small_talk_downgrades_without_pipeline_call(self):
        store = FakeMemoryStore()
        session_manager = SessionManager(store)
        llm = FakeLLM(["hello"])
        app = ChatApplication(
            llm=llm,
            session_manager=session_manager,
            context_engine=ContextEngine(),
            intent_router=IntentRouter(),
            profile_extractor=FakeProfileExtractor(),
            rag_pipeline=NoCallPipeline(),
        )

        list(app.stream_chat("session-rag-smalltalk", "hello", "rag", use_user_memory=True))

        saved = [message.model_dump() for message in store.messages["session-rag-smalltalk"]]
        self.assertEqual(saved[0]["content"], "hello")
        self.assertEqual(saved[1]["content"], "hello")
        self.assertEqual(len(llm.calls), 1)

    def test_rag_prompt_includes_user_profile(self):
        store = FakeMemoryStore()
        session_manager = SessionManager(store)
        llm = FakeLLM(["grounded"])
        app = ChatApplication(
            llm=llm,
            session_manager=session_manager,
            context_engine=ContextEngine(),
            intent_router=IntentRouter(),
            profile_extractor=FakeProfileExtractor(),
            rag_pipeline=FakeRAGPipeline(score=0.8),
        )

        list(
            app.stream_chat(
                "session-rag-profile",
                "API Key acquisition flow",
                "rag",
                collection_name="__all__",
                use_user_memory=True,
            )
        )

        self.assertTrue(llm.calls)
        self.assertIn("[User Profile]", llm.calls[0][0]["content"])

    def test_agent_knowledge_base_tool_uses_cross_collection_pipeline(self):
        pipeline = FakeRAGPipeline(score=0.8)
        tool = KnowledgeBaseTool(
            retriever=None,
            rag_pipeline=pipeline,
            collection_name="__all__",
        )

        result = tool.search("DeepSeek deployment guide")

        self.assertIn("Knowledge-base retrieval succeeded", result)
        self.assertEqual(pipeline.calls, [("DeepSeek deployment guide", "__all__")])

    def test_conversation_direct_chat_routes_without_retrieval_or_tools(self):
        store = FakeMemoryStore()
        session_manager = SessionManager(store)
        decision = RouteDecision(
            route=ConversationRoute.DIRECT_CHAT,
            confidence=0.91,
            reason="general concept question",
            rewritten_query="What is the difference between temperature and top_p?",
            source_policy=SourcePolicy.NOT_APPLICABLE,
        )
        app = ConversationApplication(
            conversation_router=FakeRouter(decision),
            chat_app=ChatApplication(
                llm=FakeLLM(["direct answer"]),
                session_manager=session_manager,
                context_engine=ContextEngine(),
                intent_router=IntentRouter(),
            ),
            agent_app=AgentApplication(
                agent_engine=None,
                session_manager=session_manager,
                context_engine=ContextEngine(),
                profile_extractor=FakeProfileExtractor(),
                supervisor=FakeSupervisor(),
            ),
            session_manager=session_manager,
        )

        payloads = parse_sse_events(
            list(app.stream_conversation("session-direct", "What is the difference between temperature and top_p?"))
        )

        self.assertEqual(payloads[0]["event"], "routing_decision")
        self.assertEqual(payloads[0]["data"]["route"], "direct_chat")
        self.assertFalse(any(item.get("event") == "retrieval_trace" for item in payloads))
        self.assertFalse(any(item.get("event") == "tool_call" for item in payloads))
        saved = [message.model_dump() for message in store.messages["session-direct"]]
        self.assertEqual(saved[0]["content"], "What is the difference between temperature and top_p?")

    def test_conversation_stream_records_local_trace_events(self):
        store = FakeMemoryStore()
        trace_store = LocalTraceStore()
        session_manager = SessionManager(store)
        decision = RouteDecision(
            route=ConversationRoute.DIRECT_CHAT,
            confidence=0.9,
            reason="general chat",
            source_policy=SourcePolicy.NOT_APPLICABLE,
        )
        app = ConversationApplication(
            conversation_router=FakeRouter(decision),
            chat_app=ChatApplication(
                llm=FakeLLM(["hello from trace"]),
                session_manager=session_manager,
                context_engine=ContextEngine(),
                intent_router=IntentRouter(),
            ),
            agent_app=AgentApplication(
                agent_engine=None,
                session_manager=session_manager,
                context_engine=ContextEngine(),
                profile_extractor=FakeProfileExtractor(),
                supervisor=FakeSupervisor(),
            ),
            session_manager=session_manager,
            trace_store=trace_store,
        )

        payloads = parse_sse_events(
            list(app.stream_conversation("session-trace-local", "hello"))
        )

        trace_id = payloads[0]["data"]["trace_id"]
        events = trace_store.list_by_trace_id(trace_id)
        event_types = {event.event_type for event in events}
        self.assertIn("routing_decision", event_types)
        self.assertIn("message_delta", event_types)
        self.assertTrue(trace_store.list_recent(limit=1))

    def test_conversation_rag_required_uses_rewritten_query_but_persists_original(self):
        store = FakeMemoryStore()
        session_manager = SessionManager(store)
        pipeline = FakeRAGPipeline(score=0.8)
        decision = RouteDecision(
            route=ConversationRoute.RAG,
            confidence=0.88,
            reason="source-grounded request",
            rewritten_query="rewritten KB query",
            source_policy=SourcePolicy.REQUIRED,
            required_capabilities=["knowledge_base"],
        )
        app = ConversationApplication(
            conversation_router=FakeRouter(decision),
            chat_app=ChatApplication(
                llm=FakeLLM(["grounded answer"]),
                session_manager=session_manager,
                context_engine=ContextEngine(),
                intent_router=IntentRouter(),
                rag_pipeline=pipeline,
            ),
            agent_app=AgentApplication(
                agent_engine=None,
                session_manager=session_manager,
                context_engine=ContextEngine(),
                profile_extractor=FakeProfileExtractor(),
                supervisor=FakeSupervisor(),
            ),
            session_manager=session_manager,
        )

        payloads = parse_sse_events(
            list(
                app.stream_conversation(
                    "session-rag-required",
                    "Please answer from the knowledge base about RAG evaluation and include citations.",
                    collection_name="__all__",
                )
            )
        )

        self.assertEqual(payloads[0]["data"]["source_policy"], "required")
        self.assertTrue(any(item.get("event") == "retrieval_trace" for item in payloads))
        self.assertTrue(any(item.get("event") == "citation_trace" for item in payloads))
        self.assertTrue(any(item.get("event") == "answer_guard" for item in payloads))
        self.assertEqual(pipeline.calls, [("rewritten KB query", "__all__")])
        saved = [message.model_dump() for message in store.messages["session-rag-required"]]
        self.assertEqual(saved[0]["content"], "Please answer from the knowledge base about RAG evaluation and include citations.")
        self.assertEqual(store.routing_logs[0]["rewritten_query"], "rewritten KB query")

    def test_conversation_rag_optional_falls_back_without_kb_only_refusal(self):
        store = FakeMemoryStore()
        session_manager = SessionManager(store)
        decision = RouteDecision(
            route=ConversationRoute.RAG,
            confidence=0.76,
            reason="continued KB topic",
            rewritten_query="plain follow-up",
            source_policy=SourcePolicy.OPTIONAL,
            required_capabilities=["knowledge_base"],
        )
        app = ConversationApplication(
            conversation_router=FakeRouter(decision),
            chat_app=ChatApplication(
                llm=FakeLLM(["plain explanation"]),
                session_manager=session_manager,
                context_engine=ContextEngine(),
                intent_router=IntentRouter(),
                rag_pipeline=FakeRAGPipeline(score=0.1),
            ),
            agent_app=AgentApplication(
                agent_engine=None,
                session_manager=session_manager,
                context_engine=ContextEngine(),
                profile_extractor=FakeProfileExtractor(),
                supervisor=FakeSupervisor(),
            ),
            session_manager=session_manager,
        )

        payloads = parse_sse_events(
            list(app.stream_conversation("session-rag-optional", "Can you explain that concept again in plain language?"))
        )
        answer = "".join(
            item.get("content", "")
            for item in payloads
            if item.get("event") == "message_delta"
        )

        self.assertIn("plain explanation", answer)
        self.assertFalse("insufficient evidence" in answer.lower())

    def test_conversation_agent_route_preserves_outer_and_inner_route_events(self):
        store = FakeMemoryStore()
        session_manager = SessionManager(store)
        decision = RouteDecision(
            route=ConversationRoute.AGENT,
            confidence=0.9,
            reason="agent workflow requested",
            rewritten_query="run an agent workflow",
            source_policy=SourcePolicy.NOT_APPLICABLE,
            required_capabilities=["multi_step_planning"],
        )
        app = ConversationApplication(
            conversation_router=FakeRouter(decision),
            chat_app=ChatApplication(
                llm=FakeLLM(["unused"]),
                session_manager=session_manager,
                context_engine=ContextEngine(),
                intent_router=IntentRouter(),
            ),
            agent_app=AgentApplication(
                agent_engine=None,
                session_manager=session_manager,
                context_engine=ContextEngine(),
                profile_extractor=FakeProfileExtractor(),
                supervisor=FakeSupervisor(),
            ),
            session_manager=session_manager,
        )

        payloads = parse_sse_events(
            list(app.stream_conversation("session-agent-route", "Run an agent workflow and explain the result"))
        )

        self.assertEqual(payloads[0]["event"], "routing_decision")
        self.assertEqual(payloads[0]["data"]["route"], "agent")
        self.assertTrue(any(item.get("event") == "message_delta" for item in payloads))
        saved = [message.model_dump() for message in store.messages["session-agent-route"]]
        self.assertEqual(saved[0]["content"], "Run an agent workflow and explain the result")

    def test_conversation_clarify_saves_only_clarifying_reply(self):
        store = FakeMemoryStore()
        session_manager = SessionManager(store)
        decision = RouteDecision(
            route=ConversationRoute.CLARIFY,
            confidence=0.52,
            reason="missing target",
            rewritten_query="Help me process this item",
            source_policy=SourcePolicy.NOT_APPLICABLE,
            clarifying_question="Which item do you want me to process?",
        )
        app = ConversationApplication(
            conversation_router=FakeRouter(decision),
            chat_app=ChatApplication(
                llm=FakeLLM(["unused"]),
                session_manager=session_manager,
                context_engine=ContextEngine(),
                intent_router=IntentRouter(),
            ),
            agent_app=AgentApplication(
                agent_engine=None,
                session_manager=session_manager,
                context_engine=ContextEngine(),
                profile_extractor=FakeProfileExtractor(),
                supervisor=FakeSupervisor(),
            ),
            session_manager=session_manager,
        )

        payloads = parse_sse_events(
            list(app.stream_conversation("session-clarify", "Help me process this item"))
        )

        self.assertEqual(payloads[0]["data"]["route"], "clarify")
        self.assertFalse(any(item.get("event") == "retrieval_trace" for item in payloads))
        self.assertFalse(any(item.get("event") == "tool_call" for item in payloads))
        saved = [message.model_dump() for message in store.messages["session-clarify"]]
        self.assertEqual(saved[0]["content"], "Help me process this item")
        self.assertEqual(saved[1]["content"], "Which item do you want me to process?")

    def test_explicit_chinese_kb_query_is_not_downgraded_as_small_talk(self):
        store = FakeMemoryStore()
        session_manager = SessionManager(store)
        pipeline = FakeRAGPipeline(score=0.8)
        app = ChatApplication(
            llm=FakeLLM(["kb answer"]),
            session_manager=session_manager,
            context_engine=ContextEngine(),
            intent_router=IntentRouter(),
            rag_pipeline=pipeline,
        )

        payloads = parse_sse_events(
            list(
                app.stream_chat(
                    "session-explicit-kb",
                    "Search the knowledge base: I have a headache, please give advice.",
                    "rag",
                    collection_name="medical_kb",
                )
            )
        )

        self.assertEqual(pipeline.calls, [("Search the knowledge base: I have a headache, please give advice.", "medical_kb")])
        self.assertTrue(any(item.get("event") == "retrieval_trace" for item in payloads))
        statuses = [
            item.get("content", "")
            for item in payloads
            if item.get("event") == "status"
        ]
        self.assertFalse(any("casual chat" in status for status in statuses))

    def test_rag_unavailable_emits_specific_event(self):
        store = FakeMemoryStore()
        session_manager = SessionManager(store)
        app = ChatApplication(
            llm=FakeLLM(["unused"]),
            session_manager=session_manager,
            context_engine=ContextEngine(),
            intent_router=IntentRouter(),
            rag_pipeline=None,
        )

        payloads = parse_sse_events(
            list(app.stream_rag_chat("session-rag-unavailable", "Search the knowledge base for headache advice."))
        )

        unavailable = [item for item in payloads if item.get("event") == "rag_unavailable"]
        self.assertTrue(unavailable)
        self.assertEqual(unavailable[0]["data"].get("component"), "rag_pipeline")

    def test_agent_no_longer_preflights_demo_code_execution(self):
        store = FakeMemoryStore()
        session_manager = SessionManager(store)

        def sample_high_risk_tool(code: str) -> str:
            return "should not run"

        registry = ToolRegistry.from_callables(
            [sample_high_risk_tool],
            policy=ToolPolicy(
                allowed_categories={"read_only", "network"},
                tool_categories={"sample_high_risk_tool": "code_exec"},
            ),
        )
        app = AgentApplication(
            agent_engine=AgentEngine(
                llm=FakeAgentLLM([]),
                tool_registry=registry,
            ),
            session_manager=session_manager,
            context_engine=ContextEngine(),
            profile_extractor=FakeProfileExtractor(),
            supervisor=FakeSupervisor(),
        )

        payloads = parse_sse_events(
            list(app.stream_agent_task("session-no-demo-preflight", "run python code"))
        )

        self.assertFalse(any(item.get("event") == "tool_policy_blocked" for item in payloads))
        self.assertTrue(
            any(
                item.get("event") == "message_delta" and item.get("content") == "partial-agent"
                for item in payloads
            )
        )

    def test_custom_agent_runtime_adapter_preserves_engine_events(self):
        class FakeEngine:
            def stream_run(self, messages, model_options=None, resume_state=None):
                yield {"type": "status", "content": "starting"}
                yield {
                    "type": "tool_call",
                    "name": "search_knowledge_base",
                    "args": {"query": "DeepAssist"},
                    "tool_call_id": "call-1",
                }
                yield {
                    "type": "tool_result",
                    "name": "search_knowledge_base",
                    "content": "result",
                    "tool_call_id": "call-1",
                    "success": True,
                    "metadata": {"source": "test"},
                }
                yield {"type": "final_answer", "content": "done", "state": {"status": "completed"}}
                yield {
                    "type": "finish",
                    "new_messages": [{"role": "assistant", "content": "done"}],
                    "status": "completed",
                }

        runtime = CustomAgentRuntimeAdapter(FakeEngine())
        events = list(
            runtime.stream_run(
                AgentRuntimeRequest(
                    session_id="session-runtime",
                    query="Search DeepAssist",
                    messages=[{"role": "user", "content": "Search DeepAssist"}],
                )
            )
        )

        event_names = [event.event for event in events]
        self.assertIn("agent_runtime_node", event_names)
        self.assertIn("tool_call", event_names)
        self.assertIn("tool_result", event_names)
        self.assertIn("final_answer", event_names)
        final_event = next(event for event in events if event.event == "final_answer")
        self.assertEqual(final_event.content, "done")

    def test_custom_agent_runtime_adapter_preserves_failed_finish_status(self):
        class FailingEngine:
            def stream_run(self, messages, model_options=None, resume_state=None):
                yield {"type": "error", "content": "boom"}
                yield {
                    "type": "finish",
                    "new_messages": [],
                    "status": "failed",
                    "termination_reason": "llm_error",
                }

        runtime = CustomAgentRuntimeAdapter(FailingEngine())
        events = list(
            runtime.stream_run(
                AgentRuntimeRequest(
                    session_id="session-runtime-failed",
                    query="fail",
                    messages=[{"role": "user", "content": "fail"}],
                )
            )
        )

        final_node = events[-1]
        self.assertEqual(final_node.event, "agent_runtime_node")
        self.assertEqual(final_node.data["status"], "failed")

    def test_langgraph_runtime_adapter_emits_node_trace_and_engine_events(self):
        class FakeEngine:
            def stream_run(self, messages, model_options=None, resume_state=None):
                yield {"type": "status", "content": "runtime started"}
                yield {"type": "final_answer", "content": "runtime answer", "state": {"status": "completed"}}
                yield {
                    "type": "finish",
                    "new_messages": [{"role": "assistant", "content": "runtime answer"}],
                    "status": "completed",
                }

        runtime = LangGraphAgentRuntimeAdapter(FakeEngine())
        events = list(
            runtime.stream_run(
                AgentRuntimeRequest(
                    session_id="session-langgraph",
                    query="Run runtime",
                    messages=[{"role": "user", "content": "Run runtime"}],
                )
            )
        )

        self.assertTrue(any(event.event == "agent_runtime_node" for event in events))
        self.assertTrue(
            any(
                event.event == "final_answer" and event.content == "runtime answer"
                for event in events
            )
        )

    def test_langgraph_runtime_fallback_preserves_failed_finish_status(self):
        class FailingEngine:
            def stream_run(self, messages, model_options=None, resume_state=None):
                yield {"type": "error", "content": "boom"}
                yield {
                    "type": "finish",
                    "new_messages": [],
                    "status": "failed",
                    "termination_reason": "llm_error",
                }

        runtime = LangGraphAgentRuntimeAdapter(FailingEngine())
        with patch.object(runtime, "_compile_graph", return_value=None):
            events = list(
                runtime.stream_run(
                    AgentRuntimeRequest(
                        session_id="session-langgraph-failed",
                        query="fail",
                        messages=[{"role": "user", "content": "fail"}],
                    )
                )
            )

        final_node = events[-1]
        self.assertEqual(final_node.event, "agent_runtime_node")
        self.assertEqual(final_node.name, "final_answer")
        self.assertEqual(final_node.data["status"], "failed")
        self.assertTrue(final_node.data["fallback_without_langgraph"])

    def test_agent_application_can_stream_with_runtime_adapter(self):
        class FakeRuntime:
            def stream_run(self, request):
                yield StreamEvent(
                    event="agent_runtime_node",
                    name="FakeRuntime",
                    data={"runtime": "test", "node": "tool_execution", "status": "started"},
                )
                yield StreamEvent.message_delta("runtime answer")
                yield StreamEvent(
                    event="finish",
                    data={
                        "new_messages": [{"role": "assistant", "content": "runtime answer"}],
                        "status": "completed",
                    },
                )

        store = FakeMemoryStore()
        session_manager = SessionManager(store)
        app = AgentApplication(
            agent_engine=None,
            session_manager=session_manager,
            context_engine=ContextEngine(),
            profile_extractor=FakeProfileExtractor(),
            supervisor=None,
            agent_runtime=FakeRuntime(),
        )

        payloads = parse_sse_events(
            list(app.stream_agent_task("session-runtime-app", "Use the runtime"))
        )

        self.assertTrue(any(item.get("event") == "agent_runtime_node" for item in payloads))
        self.assertTrue(any(item.get("event") == "message_delta" for item in payloads))
        saved = [message.model_dump() for message in store.messages["session-runtime-app"]]
        self.assertEqual(saved[0]["content"], "Use the runtime")
        self.assertEqual(saved[1]["content"], "runtime answer")

    def test_dependency_runtime_backend_custom_uses_runtime_adapter(self):
        from backend.api import dependencies as dependency_module

        class FakeEngine:
            pass

        with patch.object(dependency_module.settings, "AGENT_RUNTIME_BACKEND", "custom"):
            with patch.object(dependency_module, "get_agent_engine", return_value=FakeEngine()):
                runtime = dependency_module.get_agent_runtime()

        self.assertIsInstance(runtime, CustomAgentRuntimeAdapter)

        with patch.object(dependency_module.settings, "AGENT_RUNTIME_BACKEND", "supervisor"):
            self.assertIsNone(dependency_module.get_agent_runtime())

    def test_dependency_registry_exposes_development_python_tool_with_hitl(self):
        from backend.api import dependencies as dependency_module

        with patch.object(dependency_module.settings, "ENABLE_DEVELOPMENT_TOOLS", True):
            with patch.object(dependency_module.settings, "ALLOWED_TOOL_CATEGORIES", "read_only"):
                with patch.object(dependency_module.settings, "TOOL_APPROVAL_MODE", "hitl"):
                    with patch.object(dependency_module, "get_retriever", return_value=None):
                        with patch.object(dependency_module, "get_rag_pipeline", return_value=None):
                            with patch.object(dependency_module, "get_kb_app", return_value=FakeKBApp()):
                                registry = dependency_module.get_tool_registry()

        specs = {item["name"]: item for item in registry.list_tool_specs()}
        schema_names = {item["function"]["name"] for item in registry.openai_schemas()}

        self.assertTrue(specs["execute_python_sandbox"]["enabled"])
        self.assertTrue(specs["execute_python_sandbox"]["approval_required"])
        self.assertIn("execute_python_sandbox", schema_names)

    def test_tool_specs_expose_category_and_policy_state(self):
        def sample_high_risk_tool(code: str) -> str:
            return "ok"

        registry = ToolRegistry.from_callables(
            [sample_high_risk_tool],
            policy=ToolPolicy(
                allowed_categories={"read_only"},
                tool_categories={"sample_high_risk_tool": "code_exec"},
            ),
        )

        specs = registry.list_tool_specs()

        self.assertEqual(specs[0]["category"], "code_exec")
        self.assertFalse(specs[0]["enabled"])
        self.assertIn("category_not_allowed", specs[0]["blocked_reason"])

    def test_tool_policy_fails_closed_when_categories_are_missing(self):
        policy = ToolPolicy(allowed_categories={"read_only"}, tool_categories=None)

        self.assertFalse(policy.can_execute("unclassified_tool"))
        self.assertEqual(policy.blocked_reason("unclassified_tool"), "tool_category_unknown")

    def test_openai_schemas_hide_tools_blocked_by_policy(self):
        def read_local_file(file_name: str) -> str:
            return file_name

        def write_local_file(file_name: str, content: str) -> str:
            return content

        registry = ToolRegistry.from_callables(
            [read_local_file, write_local_file],
            policy=ToolPolicy(
                allowed_categories={"read_only"},
                tool_categories={
                    "read_local_file": "read_only",
                    "write_local_file": "write_file",
                },
            ),
        )

        schema_names = {
            item["function"]["name"]
            for item in registry.openai_schemas()
        }
        self.assertEqual(schema_names, {"read_local_file"})

    def test_high_risk_tool_requires_hitl_before_execution(self):
        approval_store = LocalApprovalStore()
        executed: list[str] = []

        def write_local_file(file_name: str, content: str) -> str:
            executed.append(file_name)
            return f"written:{file_name}:{content}"

        registry = ToolRegistry.from_callables(
            [write_local_file],
            policy=ToolPolicy(
                allowed_categories={"write_file"},
                tool_categories={"write_local_file": "write_file"},
                approval_store=approval_store,
            ),
        )

        result = registry.execute(
            ToolCall(
                id="call-write",
                name="write_local_file",
                args={"file_name": "note.txt", "content": "hello"},
            )
        )

        self.assertFalse(result.success)
        self.assertEqual(result.metadata["error_type"], "approval_required")
        self.assertFalse(executed)
        action_id = result.metadata["pending_action"]["action_id"]
        approval_store.approve(action_id, decided_by="tester", reason="safe sandbox write")

        approved_result = registry.execute_approved(action_id)

        self.assertTrue(approved_result.success)
        self.assertEqual(executed, ["note.txt"])
        action = approval_store.get(action_id)
        self.assertEqual(action.status.value, "completed")
        self.assertIn("written:note.txt", action.result["content"])

    def test_high_risk_tool_can_execute_directly_when_approval_is_disabled(self):
        executed: list[str] = []

        def write_local_file(file_name: str, content: str) -> str:
            executed.append(file_name)
            return f"written:{content}"

        registry = ToolRegistry.from_callables(
            [write_local_file],
            policy=ToolPolicy(
                allowed_categories={"write_file"},
                tool_categories={"write_local_file": "write_file"},
                approval_mode="disabled",
            ),
        )

        result = registry.execute(
            ToolCall(
                id="call-write-direct",
                name="write_local_file",
                args={"file_name": "note.txt", "content": "hello"},
            )
        )

        self.assertTrue(result.success)
        self.assertEqual(result.content, "written:hello")
        self.assertEqual(executed, ["note.txt"])

    def test_local_development_tools_are_callable_when_approval_disabled(self):
        registry = ToolRegistry.from_callables(
            [
                write_local_file,
                read_local_file,
                list_sandbox_files,
                execute_python_sandbox,
            ],
            policy=ToolPolicy(
                allowed_categories={"read_only", "write_file", "code_exec"},
                tool_categories={
                    "write_local_file": "write_file",
                    "read_local_file": "read_only",
                    "list_sandbox_files": "read_only",
                    "execute_python_sandbox": "code_exec",
                },
                approval_mode="disabled",
            ),
        )

        write_result = registry.execute(
            ToolCall(
                id="call-write-local",
                name="write_local_file",
                args={"file_name": "tool_smoke.txt", "content": "smoke-ok"},
            )
        )
        read_result = registry.execute(
            ToolCall(
                id="call-read-local",
                name="read_local_file",
                args={"file_name": "tool_smoke.txt"},
            )
        )
        list_result = registry.execute(
            ToolCall(
                id="call-list-local",
                name="list_sandbox_files",
                args={"path": "."},
            )
        )
        python_result = registry.execute(
            ToolCall(
                id="call-python-local",
                name="execute_python_sandbox",
                args={"code": "print(2 + 3)", "timeout_seconds": 5},
            )
        )

        self.assertTrue(write_result.success)
        self.assertTrue(read_result.success)
        self.assertIn("smoke-ok", read_result.content)
        self.assertTrue(list_result.success)
        self.assertIn("tool_smoke.txt", list_result.content)
        self.assertTrue(python_result.success)
        self.assertIn("stdout:\n5", python_result.content)

    def test_approval_required_tool_result_emits_pending_action_stream_event(self):
        events = list(
            engine_event_to_stream_events(
                {
                    "type": "tool_result",
                    "name": "write_local_file",
                    "content": "approval required",
                    "success": False,
                    "tool_call_id": "call-write",
                    "metadata": {
                        "error_type": "approval_required",
                        "category": "write_file",
                        "approval_store_available": True,
                        "pending_action": {"action_id": "pending-1"},
                    },
                }
            )
        )

        self.assertEqual(events[0].event, "pending_action")
        self.assertEqual(events[0].data["pending_action"]["action_id"], "pending-1")
        self.assertEqual(events[1].event, "tool_result")

    def test_agent_engine_stops_at_pending_approval_without_parametric_answer(self):
        approval_store = LocalApprovalStore()
        llm = FakeAgentLLM(
            [
                SimpleNamespace(
                    content="I will run Python.",
                    tool_calls=[
                        {
                            "id": "call-python-hitl",
                            "type": "function",
                            "function": {
                                "name": "execute_python_sandbox",
                                "arguments": json.dumps(
                                    {"code": "print(2 + 3)", "timeout_seconds": 5}
                                ),
                            },
                        }
                    ],
                    reasoning_content=None,
                )
            ]
        )
        registry = ToolRegistry.from_callables(
            [execute_python_sandbox],
            policy=ToolPolicy(
                allowed_categories={"code_exec"},
                tool_categories={"execute_python_sandbox": "code_exec"},
                approval_store=approval_store,
            ),
        )
        engine = AgentEngine(
            llm=llm,
            tool_registry=registry,
            run_config=AgentRunConfig(max_iterations=3),
        )

        events = list(
            engine.stream_run(
                [
                    {"role": "system", "content": "agent"},
                    {"role": "user", "content": "calculate with python"},
                ]
            )
        )

        finish = next(event for event in events if event.get("type") == "finish")
        snapshot = next(
            event
            for event in events
            if event.get("type") == "task_snapshot"
            and event.get("data", {}).get("status") == "pending_approval"
        )
        self.assertEqual(finish["status"], "pending_approval")
        self.assertEqual(finish["termination_reason"], "human_approval_required")
        self.assertEqual(len(llm.calls), 1)
        self.assertEqual(len(approval_store.list(status="pending")), 1)
        self.assertEqual(snapshot["data"]["lifecycle_state"]["status"], "pending_approval")

    def test_agent_application_approves_pending_python_action_and_resumes_answer(self):
        approval_store = LocalApprovalStore()
        llm = FakeAgentLLM(
            [
                SimpleNamespace(
                    content="I will run Python.",
                    tool_calls=[
                        {
                            "id": "call-python-hitl",
                            "type": "function",
                            "function": {
                                "name": "execute_python_sandbox",
                                "arguments": json.dumps(
                                    {
                                        "code": "print(round(7/3, 4))",
                                        "timeout_seconds": 5,
                                    }
                                ),
                            },
                        }
                    ],
                    reasoning_content=None,
                ),
                SimpleNamespace(
                    content="Python sandbox returned 2.3333.",
                    tool_calls=[],
                    reasoning_content=None,
                ),
            ]
        )
        registry = ToolRegistry.from_callables(
            [execute_python_sandbox],
            policy=ToolPolicy(
                allowed_categories={"code_exec"},
                tool_categories={"execute_python_sandbox": "code_exec"},
                approval_store=approval_store,
            ),
        )
        engine = AgentEngine(
            llm=llm,
            tool_registry=registry,
            run_config=AgentRunConfig(max_iterations=5),
        )
        store = FakeMemoryStore()
        session_manager = SessionManager(store)
        app = AgentApplication(
            agent_engine=engine,
            session_manager=session_manager,
            context_engine=ContextEngine(),
            profile_extractor=FakeProfileExtractor(),
            supervisor=None,
            agent_runtime=LangGraphAgentRuntimeAdapter(engine),
        )

        first_payloads = parse_sse_events(
            list(
                app.stream_agent_task(
                    "session-hitl-python",
                    "给我执行Python沙箱，为我计算7/3的结果，保留小数点后4位",
                )
            )
        )
        self.assertTrue(
            any(
                item.get("event") == "agent_runtime_node"
                and item.get("data", {}).get("fallback_without_langgraph") is False
                for item in first_payloads
            )
        )
        pending_event = next(
            item for item in first_payloads if item.get("event") == "pending_action"
        )
        action_id = pending_event["data"]["pending_action"]["action_id"]

        resumed_payloads = parse_sse_events(
            list(
                app.stream_pending_action_decision(
                    action_id=action_id,
                    approved=True,
                    session_id="session-hitl-python",
                    decided_by="tester",
                    reason="safe local calculation",
                )
            )
        )

        self.assertTrue(
            any(item.get("event") == "approval_result" for item in resumed_payloads)
        )
        tool_result = next(
            item
            for item in resumed_payloads
            if item.get("event") == "tool_result"
            and item.get("name") == "execute_python_sandbox"
        )
        self.assertTrue(tool_result["data"]["success"])
        self.assertIn("stdout:\n2.3333", tool_result["content"])
        final_answer = next(
            item for item in resumed_payloads if item.get("event") == "final_answer"
        )
        self.assertIn("2.3333", final_answer["content"])
        self.assertIsNone(store.get_task_snapshot("session-hitl-python"))

    def test_agent_application_rejects_pending_action_and_resumes_with_rejection_observation(self):
        approval_store = LocalApprovalStore()
        llm = FakeAgentLLM(
            [
                SimpleNamespace(
                    content="I will run Python.",
                    tool_calls=[
                        {
                            "id": "call-python-reject",
                            "type": "function",
                            "function": {
                                "name": "execute_python_sandbox",
                                "arguments": json.dumps(
                                    {"code": "print(1)", "timeout_seconds": 5}
                                ),
                            },
                        }
                    ],
                    reasoning_content=None,
                ),
                SimpleNamespace(
                    content="The Python execution was rejected, so I cannot run it.",
                    tool_calls=[],
                    reasoning_content=None,
                ),
            ]
        )
        registry = ToolRegistry.from_callables(
            [execute_python_sandbox],
            policy=ToolPolicy(
                allowed_categories={"code_exec"},
                tool_categories={"execute_python_sandbox": "code_exec"},
                approval_store=approval_store,
            ),
        )
        engine = AgentEngine(llm=llm, tool_registry=registry)
        store = FakeMemoryStore()
        app = AgentApplication(
            agent_engine=engine,
            session_manager=SessionManager(store),
            context_engine=ContextEngine(),
            profile_extractor=FakeProfileExtractor(),
            supervisor=None,
            agent_runtime=LangGraphAgentRuntimeAdapter(engine),
        )

        first_payloads = parse_sse_events(
            list(app.stream_agent_task("session-hitl-reject", "run python"))
        )
        action_id = next(
            item for item in first_payloads if item.get("event") == "pending_action"
        )["data"]["pending_action"]["action_id"]
        resumed_payloads = parse_sse_events(
            list(
                app.stream_pending_action_decision(
                    action_id=action_id,
                    approved=False,
                    session_id="session-hitl-reject",
                    decided_by="tester",
                    reason="do not run code",
                )
            )
        )

        approval = next(
            item for item in resumed_payloads if item.get("event") == "approval_result"
        )
        self.assertEqual(approval["data"]["status"], "rejected")
        tool_result = next(
            item
            for item in resumed_payloads
            if item.get("event") == "tool_result"
            and item.get("name") == "execute_python_sandbox"
        )
        self.assertFalse(tool_result["data"]["success"])
        self.assertEqual(
            tool_result["data"]["metadata"]["error_type"],
            "human_rejected",
        )
        final_answer = next(
            item for item in resumed_payloads if item.get("event") == "final_answer"
        )
        self.assertIn("rejected", final_answer["content"])

    def test_agent_application_injects_only_enabled_tool_context_for_runtime(self):
        class CaptureRuntime:
            def __init__(self):
                self.messages = None

            def stream_run(self, request):
                self.messages = request.messages
                yield StreamEvent(
                    event="finish",
                    data={
                        "new_messages": [{"role": "assistant", "content": "done"}],
                        "status": "completed",
                    },
                )

        registry = ToolRegistry.from_callables(
            [read_local_file, write_local_file],
            policy=ToolPolicy(
                allowed_categories={"read_only"},
                tool_categories={
                    "read_local_file": "read_only",
                    "write_local_file": "write_file",
                },
            ),
        )
        engine = AgentEngine(
            llm=FakeAgentLLM([]),
            tool_registry=registry,
            run_config=AgentRunConfig(max_iterations=1),
        )
        runtime = CaptureRuntime()
        app = AgentApplication(
            agent_engine=engine,
            session_manager=SessionManager(FakeMemoryStore()),
            context_engine=ContextEngine(),
            profile_extractor=FakeProfileExtractor(),
            supervisor=None,
            agent_runtime=runtime,
        )

        parse_sse_events(list(app.stream_agent_task("session-tool-context", "read file")))

        rendered_context = "\n".join(str(item.get("content") or "") for item in runtime.messages)
        self.assertIn("read_local_file", rendered_context)
        self.assertNotIn("write_local_file", rendered_context)

    def test_conversation_router_low_confidence_agent_downgrades_to_clarify(self):
        llm = FakeRouteLLM(
            json.dumps(
                {
                    "route": "agent",
                    "confidence": 0.4,
                    "reason": "maybe a tool task",
                    "rewritten_query": "run something",
                    "source_policy": "not_applicable",
                    "required_capabilities": ["multi_step_planning"],
                    "clarifying_question": "",
                    "risk_level": "medium",
                    "should_log_for_training": True,
                }
            )
        )
        router = ConversationRouter(llm=llm, min_confidence=0.65)

        decision = router.route("Help me process this item")

        self.assertEqual(decision.route, ConversationRoute.CLARIFY)
        self.assertIn("below minimum", decision.reason)

    def test_conversation_router_routes_agent_kb_composite_to_agent(self):
        router = ConversationRouter(llm=FakeRouteLLM("should not be called"))

        decision = router.route("Run an agent workflow and search the knowledge base")

        self.assertEqual(decision.route, ConversationRoute.AGENT)
        self.assertIn("knowledge_base", decision.required_capabilities)
        self.assertIn("multi_step_planning", decision.required_capabilities)

    def test_conversation_router_routes_explicit_agent_task_to_agent(self):
        router = ConversationRouter(llm=FakeRouteLLM("should not be called"))

        decision = router.route("Execute an agent task to handle this problem")

        self.assertEqual(decision.route, ConversationRoute.AGENT)
        self.assertIn("multi_step_planning", decision.required_capabilities)

    def test_conversation_router_high_risk_agent_downgrades_to_clarify(self):
        llm = FakeRouteLLM(
            json.dumps(
                {
                    "route": "agent",
                    "confidence": 0.9,
                    "reason": "external action",
                    "rewritten_query": "delete files",
                    "source_policy": "not_applicable",
                    "required_capabilities": ["file_write"],
                    "clarifying_question": "",
                    "risk_level": "high",
                    "should_log_for_training": True,
                }
            )
        )
        router = ConversationRouter(llm=llm, min_confidence=0.65)

        decision = router.route("Delete these files")

        self.assertEqual(decision.route, ConversationRoute.CLARIFY)
        self.assertEqual(decision.risk_level.value, "high")

    def test_conversation_router_invalid_json_downgrades_to_clarify(self):
        router = ConversationRouter(llm=FakeRouteLLM("not-json"), min_confidence=0.65)

        decision = router.route("What is the difference between temperature and top_p?")

        self.assertEqual(decision.route, ConversationRoute.CLARIFY)
        self.assertIn("invalid structured JSON", decision.reason)

    def test_kb_health_prefers_cached_report_until_refresh(self):
        inspector = FakeHealthInspector()
        app = KnowledgeBaseApp(
            chunker=object(),
            embedding_model=None,
            vector_db=None,
            keyword_db=None,
            health_inspector=inspector,
        )

        cached = app.get_health_report(refresh=False)
        refreshed = app.get_health_report(refresh=True, collections=["medical_kb"])

        self.assertEqual(cached["status"], "success")
        self.assertEqual(cached["source"], "cached")
        self.assertEqual(cached["data"]["checked_at"], "2026-05-04T16:00:00")
        self.assertEqual(refreshed["status"], "success")
        self.assertEqual(refreshed["source"], "live")
        self.assertEqual(refreshed["data"]["checked_at"], "2026-05-04T16:05:00")
        self.assertEqual(
            inspector.inspect_calls,
            [
                {
                    "collections": ["medical_kb"],
                    "repair": False,
                    "batch_size": 256,
                    "persist": True,
                }
            ],
        )

    def test_priority_context_keeps_profile_turn_under_budget(self):
        store = FakeMemoryStore()
        session_manager = SessionManager(store)
        session_id = "session-priority"

        session_manager.save_interaction(session_id, "hello", "hi")
        session_manager.save_interaction(session_id, "tell me a joke", "maybe later")
        session_manager.save_interaction(
            session_id,
            "I am a backend engineer, please answer concisely.",
            "Noted.",
        )
        session_manager.save_interaction(session_id, "random follow-up one", "ack one")
        session_manager.save_interaction(session_id, "random follow-up two", "ack two")
        session_manager.save_interaction(session_id, "latest production issue?", "latest answer")

        history = session_manager.get_chat_context(session_id, max_rounds=6)
        user_messages = [message["content"] for message in history if message["role"] == "user"]

        self.assertEqual(history[0]["role"], "system")
        self.assertIn("[Conversation Summary]", history[0]["content"])
        self.assertIn("I am a backend engineer, please answer concisely.", user_messages)
        self.assertIn("random follow-up two", user_messages)
        self.assertIn("latest production issue?", user_messages)
        self.assertNotIn("hello", user_messages)

    def test_session_history_remains_full_without_summary_injection(self):
        store = FakeMemoryStore()
        session_manager = SessionManager(store)
        session_id = "session-history"

        session_manager.save_interaction(session_id, "first", "one")
        session_manager.save_interaction(session_id, "second", "two")
        session_manager.save_interaction(session_id, "third", "three")

        history = session_manager.get_session_history(session_id, limit=10)

        self.assertEqual([message["role"] for message in history], ["user", "assistant"] * 3)
        self.assertNotIn("[Conversation Summary]", "\n".join(message["content"] for message in history))

    def test_long_term_memory_is_recalled_into_context_window(self):
        store = FakeMemoryStore()
        store.set_profile("user_facts", '["I am a backend engineer.", "I prefer concise answers."]')
        store.set_profile("topics", '["architecture", "rag"]')
        session_manager = SessionManager(store)
        session_id = "session-memory"

        session_manager.save_interaction(session_id, "old question", "old answer")
        session_manager.save_interaction(session_id, "new question", "new answer")

        plan = session_manager.plan_chat_context(
            session_id,
            max_rounds=6,
            query="Please answer like a backend engineer and keep it concise.",
            use_long_term_memory=True,
        )
        flattened = plan.flattened_messages()
        memory_messages = [
            message["content"]
            for message in flattened
            if message["role"] == "system" and "[Long-Term Memory]" in message["content"]
        ]

        self.assertTrue(memory_messages)
        self.assertTrue(any("backend engineer" in message for message in memory_messages))
        self.assertTrue(any("concise" in message.lower() for message in memory_messages))

    def test_long_term_memory_consumes_budget_but_keeps_recent_turns(self):
        store = FakeMemoryStore()
        store.set_profile("user_facts", '["I am a backend engineer."]')
        session_manager = SessionManager(store)
        session_id = "session-memory-budget"

        session_manager.save_interaction(session_id, "turn one", "one")
        session_manager.save_interaction(session_id, "turn two", "two")
        session_manager.save_interaction(session_id, "turn three", "three")
        session_manager.save_interaction(session_id, "turn four", "four")

        plan = session_manager.plan_chat_context(
            session_id,
            max_rounds=4,
            query="As a backend engineer, help me with turn four.",
            use_long_term_memory=True,
        )
        user_messages = [
            message["content"]
            for message in plan.flattened_messages()
            if message["role"] == "user"
        ]

        self.assertTrue(plan.recalled_memories)
        self.assertIn("turn four", user_messages)

    def test_context_window_trace_data_contains_summary_and_memory(self):
        store = FakeMemoryStore()
        store.set_profile("user_facts", '["I am a backend engineer."]')
        session_manager = SessionManager(store)
        session_id = "session-trace"

        session_manager.save_interaction(session_id, "turn one", "one")
        session_manager.save_interaction(session_id, "turn two", "two")
        session_manager.save_interaction(session_id, "turn three", "three")
        session_manager.save_interaction(session_id, "turn four", "four")

        plan = session_manager.plan_chat_context(
            session_id,
            max_rounds=4,
            query="backend engineer turn four",
            use_long_term_memory=True,
        )
        trace_data = plan.to_trace_data()

        self.assertIn("budget", trace_data)
        self.assertIn("selected_turns", trace_data)
        self.assertIn("recalled_memories", trace_data)
        self.assertEqual(trace_data["recalled_memory_count"], len(plan.recalled_memories))
        self.assertTrue(trace_data["summary_injected"] or trace_data["dropped_turn_count"] == 0)

    def test_rag_context_filters_agent_tool_protocol_history(self):
        pipeline_result = FakeRAGPipeline(score=0.8).build_context("What is inflammation?", "medical_kb")
        history = [
            {"role": "user", "content": "previous agent task"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call-1",
                        "type": "function",
                        "function": {
                            "name": "search_knowledge_base",
                            "arguments": "{}",
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call-1",
                "name": "search_knowledge_base",
                "content": "tool observation",
            },
            {"role": "assistant", "content": "agent final answer"},
        ]

        context = ContextEngine().build_rag_context_from_pipeline(
            history=history,
            pipeline_result=pipeline_result,
        )

        self.assertFalse(any(message.get("role") == "tool" for message in context.messages))
        self.assertFalse(any(message.get("tool_calls") for message in context.messages))
        self.assertTrue(any(message.get("content") == "agent final answer" for message in context.messages))

    def test_persisted_session_summary_is_injected_when_runtime_summary_is_absent(self):
        store = FakeMemoryStore()
        store.save_session_summary(
            "session-persisted-summary",
            ContextSummary(
                content="[Conversation Summary]\nPersisted earlier context.",
                source="persisted",
                source_turn_ids=["turn-0"],
                dropped_turn_count=1,
                dropped_message_count=2,
                reason_counts={"recent_turn": 1},
            ),
        )
        session_manager = SessionManager(store)
        session_manager.save_interaction("session-persisted-summary", "latest question", "latest answer")

        plan = session_manager.plan_chat_context(
            "session-persisted-summary",
            max_rounds=6,
            query="follow up",
            use_long_term_memory=False,
        )

        self.assertIsNotNone(plan.summary)
        self.assertEqual(plan.summary.source, "persisted")
        self.assertIn("Persisted earlier context", plan.summary.content)

    def test_long_term_memory_recall_matches_style_preference_synonyms(self):
        store = FakeMemoryStore()
        store.set_profile("user_facts", '["I prefer concise answers."]')
        store.set_profile("topics", '["architecture"]')
        session_manager = SessionManager(store)

        plan = session_manager.plan_chat_context(
            "session-style-memory",
            max_rounds=5,
            query="Please keep it short when you explain the architecture.",
            use_long_term_memory=True,
        )

        recalled = [memory.content for memory in plan.recalled_memories]
        self.assertTrue(recalled)
        self.assertTrue(any("concise answers" in item for item in recalled))

    def test_chat_stream_emits_context_window_trace_event(self):
        store = FakeMemoryStore()
        store.set_profile("user_facts", '["I am a backend engineer."]')
        session_manager = SessionManager(store)
        session_id = "session-chat-trace"
        session_manager.save_interaction(session_id, "older one", "one")
        session_manager.save_interaction(session_id, "older two", "two")
        session_manager.save_interaction(session_id, "older three", "three")

        app = ChatApplication(
            llm=FakeLLM(["ok"]),
            session_manager=session_manager,
            context_engine=ContextEngine(),
            intent_router=IntentRouter(),
        )

        events = list(
            app.stream_chat(
                session_id,
                "I am a backend engineer, help now",
                "quick",
                history_rounds=4,
                use_user_memory=True,
            )
        )
        payloads = []
        for chunk in events:
            if not chunk.startswith("data: "):
                continue
            raw = chunk[6:].strip()
            if not raw:
                continue
            payloads.append(json.loads(raw))

        trace_payloads = [payload for payload in payloads if payload.get("event") == "context_window_trace"]
        self.assertTrue(trace_payloads)
        self.assertIn("budget", trace_payloads[0]["data"])

    def test_tool_registry_returns_structured_argument_repair_metadata(self):
        def sample_tool(city: str, days: int) -> str:
            return f"{city}:{days}"

        registry = ToolRegistry.from_callables([sample_tool])
        call = ToolCall(id="tool-1", name="sample_tool", args={"city": "beijing"})

        result = registry.execute(call)

        self.assertFalse(result.success)
        self.assertEqual(result.metadata.get("error_type"), "invalid_arguments")
        self.assertEqual(result.metadata.get("repair_strategy"), "fix_arguments")
        self.assertIn("days", result.metadata.get("missing_required_args", []))
        self.assertTrue(result.is_retryable())

    def test_tool_registry_suggests_closest_tool_for_unknown_name(self):
        def catalog_lookup(collection: str) -> str:
            return collection

        registry = ToolRegistry.from_callables([catalog_lookup])
        call = ToolCall(id="tool-2", name="catalog_lookp", args={"collection": "docs"})

        result = registry.execute(call)

        self.assertFalse(result.success)
        self.assertEqual(result.metadata.get("error_type"), "unknown_tool")
        self.assertEqual(result.metadata.get("suggested_tool"), "catalog_lookup")
        self.assertEqual(result.metadata.get("repair_strategy"), "switch_tool")

    def test_agent_engine_injects_self_correction_instruction_after_tool_failure(self):
        def sample_tool(city: str, days: int) -> str:
            return f"{city}:{days}"

        first_response = SimpleNamespace(
            content="",
            tool_calls=[
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {
                        "name": "sample_tool",
                        "arguments": json.dumps({"city": "beijing"}),
                    },
                }
            ],
            reasoning_content=None,
        )
        second_response = SimpleNamespace(
            content="Recovered with fallback answer.",
            tool_calls=[],
            reasoning_content=None,
        )
        llm = FakeAgentLLM([first_response, second_response])
        engine = AgentEngine(
            llm=llm,
            tool_registry=ToolRegistry.from_callables([sample_tool]),
            run_config=AgentRunConfig(max_iterations=3, max_self_corrections=2),
        )

        events = list(
            engine.stream_run(
                [
                    {"role": "system", "content": "agent"},
                    {"role": "user", "content": "check lookup"},
                ]
            )
        )

        self.assertEqual(len(llm.calls), 2)
        second_call_messages = llm.calls[1]
        system_messages = [
            message["content"]
            for message in second_call_messages
            if message.get("role") == "system"
        ]
        self.assertTrue(
            any("[Self-Correction Instruction]" in content for content in system_messages)
        )
        self.assertTrue(any("Missing required args" in content for content in system_messages))

        self_corrections = [event for event in events if event.get("type") == "self_correction"]
        self.assertTrue(self_corrections)
        self.assertEqual(self_corrections[0]["data"].get("repair_strategy"), "fix_arguments")

    def test_agent_supervisor_routes_kb_catalog_queries_to_tool_worker(self):
        supervisor = AgentSupervisor(
            intent_router=IntentRouter(),
            chat_worker=object(),
            rag_worker=object(),
            tool_worker=object(),
        )

        decision = supervisor.decide("Which knowledge bases are available, and can you answer using them?")

        self.assertEqual(decision.worker, AgentWorkerType.TOOL)
        self.assertIn("kb_catalog", decision.signals)

    def test_tool_agent_worker_injects_registered_tool_inventory(self):
        def list_knowledge_base_collections() -> str:
            return "medical_kb, tech_docs_kb"

        llm = FakeAgentLLM(
            [
                SimpleNamespace(
                    content="I can inspect the connected knowledge bases.",
                    tool_calls=[],
                    reasoning_content=None,
                )
            ]
        )
        worker = ToolAgentWorker(
            agent_engine=AgentEngine(
                llm=llm,
                tool_registry=ToolRegistry.from_callables([list_knowledge_base_collections]),
                run_config=AgentRunConfig(max_iterations=2),
            ),
            context_engine=ContextEngine(),
        )

        list(worker.stream("Which tools can you call?", history=[]))

        self.assertTrue(llm.calls)
        system_messages = [
            message["content"]
            for message in llm.calls[0]
            if message.get("role") == "system"
        ]
        self.assertTrue(any("Registered tools are listed below" in item for item in system_messages))
        self.assertTrue(any("list_knowledge_base_collections" in item for item in system_messages))

    def test_kb_catalog_tool_lists_connected_collections(self):
        tool = KnowledgeBaseCatalogTool(FakeKBApp())

        result = tool.list_knowledge_base_collections()

        self.assertIn("medical_kb", result)
        self.assertIn("tech_docs_kb", result)
        self.assertIn("across all connected collections", result)

    def test_runtime_capabilities_expose_real_models_tools_and_kb_scopes(self):
        def read_local_file(file: str) -> str:
            return file

        app = RuntimeApplication(
            kb_app=FakeKBApp(),
            tool_registry=ToolRegistry.from_callables([read_local_file]),
        )

        payload = app.get_capabilities()
        data = payload.get("data", {})
        tools = data.get("tools", [])
        scopes = data.get("knowledge_base", {}).get("rag_scope_options", [])

        self.assertEqual(payload.get("status"), "success")
        self.assertTrue(any(item.get("id") == "deepseek-chat" for item in data.get("models", [])))
        self.assertTrue(any(item.get("name") == "read_local_file" for item in tools))
        self.assertTrue(any(item.get("id") == "__all__" for item in scopes))
        self.assertTrue(any(item.get("id") == "medical_kb" for item in scopes))
        self.assertIn("rag_available", data.get("knowledge_base", {}))
        self.assertIn("enabled", tools[0])

    def test_runtime_diagnostics_provider_is_exposed(self):
        app = RuntimeApplication(
            kb_app=FakeKBApp(),
            tool_registry=ToolRegistry.from_callables([]),
            diagnostics_provider=lambda refresh=False: {
                "status": "success",
                "data": {"rag": {"available": False, "unavailable_reason": "missing adapters"}},
            },
        )

        payload = app.get_diagnostics(refresh=True)

        self.assertEqual(payload.get("status"), "success")
        self.assertFalse(payload["data"]["rag"]["available"])
        self.assertEqual(payload["data"]["rag"]["unavailable_reason"], "missing adapters")

    def test_runtime_application_exposes_trace_queries(self):
        trace_store = LocalTraceStore()
        trace_store.append(
            TraceEvent(
                trace_id="trace-1",
                session_id="session-1",
                route="direct_chat",
                event_type="routing_decision",
                span_name="conversation.route",
            )
        )
        app = RuntimeApplication(
            kb_app=FakeKBApp(),
            tool_registry=ToolRegistry.from_callables([]),
            trace_store=trace_store,
        )

        by_id = app.list_trace_events("trace-1")
        recent = app.list_recent_traces(limit=10)

        self.assertEqual(by_id["status"], "success")
        self.assertEqual(by_id["data"][0]["trace_id"], "trace-1")
        self.assertEqual(recent["data"][0]["event_type"], "routing_decision")

    def test_runtime_application_approves_pending_action_and_executes_tool(self):
        approval_store = LocalApprovalStore()
        executed: list[str] = []

        def write_local_file(file_name: str, content: str) -> str:
            executed.append(file_name)
            return f"written:{content}"

        registry = ToolRegistry.from_callables(
            [write_local_file],
            policy=ToolPolicy(
                allowed_categories={"write_file"},
                tool_categories={"write_local_file": "write_file"},
                approval_store=approval_store,
            ),
        )
        pending = registry.execute(
            ToolCall(
                id="call-runtime-write",
                name="write_local_file",
                args={"file_name": "note.txt", "content": "ok"},
            )
        )
        action_id = pending.metadata["pending_action"]["action_id"]
        app = RuntimeApplication(
            kb_app=FakeKBApp(),
            tool_registry=registry,
            approval_store=approval_store,
        )

        listed = app.list_pending_actions()
        approved = app.approve_pending_action(
            action_id,
            decided_by="tester",
            reason="verified sandbox write",
        )

        self.assertEqual(listed["data"][0]["action_id"], action_id)
        self.assertEqual(approved["status"], "success")
        self.assertEqual(executed, ["note.txt"])
        self.assertEqual(approved["data"]["action"]["status"], "completed")

    def test_runtime_diagnostics_reports_mysql_memory_adapter_error(self):
        from backend.api import dependencies as dependency_module

        class FailingMySQLMemoryStore:
            def __init__(self):
                raise RuntimeError("pymysql is required")

        class FakeToolRegistry:
            policy = ToolPolicy(allowed_categories={"read_only"})

            def list_tool_specs(self):
                return []

        dependency_module.clear_adapter_cache()
        with patch.object(dependency_module.settings, "MEMORY_BACKEND", "mysql"):
            with patch(
                "backend.infrastructure.memory.mysql_memory.MySQLMemoryStoreAdapter",
                FailingMySQLMemoryStore,
            ):
                with patch.object(dependency_module, "get_embedding_model", return_value=None):
                    with patch.object(dependency_module, "get_vector_db", return_value=None):
                        with patch.object(dependency_module, "get_keyword_db", return_value=None):
                            with patch.object(dependency_module, "get_retriever", return_value=None):
                                with patch.object(dependency_module, "get_rag_pipeline", return_value=None):
                                    with patch.object(
                                        dependency_module,
                                        "get_tool_registry",
                                        return_value=FakeToolRegistry(),
                                    ):
                                        with patch.object(
                                            dependency_module,
                                            "get_kb_app",
                                            return_value=FakeKBApp(),
                                        ):
                                            payload = dependency_module.get_runtime_diagnostics()

        dependency_module.clear_adapter_cache()
        memory = payload["data"]["rag"]["adapters"]["memory_store"]
        self.assertFalse(memory["available"])
        self.assertEqual(memory["backend"], "mysql")
        self.assertIn("pymysql is required", memory["error"])

    def test_deepseek_client_initializes_without_api_key_but_fails_on_call(self):
        with patch.object(settings, "DEEPSEEK_API_KEY", ""):
            client = DeepSeekClient()

            self.assertIsNone(client.client)
            with self.assertRaises(ConfigurationError):
                client.chat([{"role": "user", "content": "hello"}])

    def test_query_rewriter_adds_domain_hints_for_technical_query(self):
        result = QueryRewriteService().expand(
            normalized_query="DeepSeek API error troubleshooting",
            keyword_query="DeepSeek API error troubleshooting",
            key_terms=["DeepSeek", "API", "error"],
            quoted_phrases=[],
        )

        self.assertTrue(any("domain_hint_expansion" == note for note in result.rewrite_notes))
        self.assertTrue(result.domain_hints)
        self.assertTrue(any("DeepSeek" in query for query in result.semantic_queries))

    def test_kb_app_uses_manifest_summary_for_fast_collection_listing(self):
        app = KnowledgeBaseApp(
            chunker=object(),
            embedding_model=None,
            vector_db=SlowListSourcesVectorDB(),
            keyword_db=SlowListSourcesKeywordDB(),
            health_inspector=None,
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            manifest_path = Path(temp_dir) / "kb_manifest.json"
            app.manifest_store = KnowledgeBaseManifestStore(str(manifest_path))
            app.manifest_store.upsert_file(
                collection_name="medical_kb",
                source_file="data/sources/medical/train.json",
                chunk_count=18,
                metadata={"domain": "medical", "source_format": "json"},
            )

            payload = app.list_collections()
            files_payload = app.list_files("medical_kb")

        self.assertEqual(payload.get("status"), "success")
        self.assertEqual(payload.get("source"), "manifest_summary")
        self.assertEqual(payload.get("data")[0].get("file_count"), 1)
        self.assertEqual(payload.get("data")[0].get("chunk_count"), 18)
        self.assertEqual(files_payload.get("source"), "manifest")
        self.assertEqual(files_payload.get("data")[0].get("source_file"), "data/sources/medical/train.json")

    def test_query_planner_emits_rewrite_metadata_for_short_technical_query(self):
        planner = QueryPlanner()

        plan = planner.plan("DeepSeek API error troubleshooting")

        self.assertTrue(plan.metadata.get("rewrite_applied"))
        self.assertIn("domain_hint_expansion", plan.metadata.get("rewrite_notes", []))
        self.assertTrue(plan.metadata.get("domain_hints"))
        self.assertGreaterEqual(len(plan.semantic_queries), 1)

    def test_hybrid_retriever_attaches_failure_attribution_metadata(self):
        docs = [
            DocumentChunk(
                id="chunk-1",
                content="Unrelated finance report and quarterly summary",
                metadata={"source_file": "guide.md"},
                score=0.12,
            )
        ]
        retriever = HybridRetriever(
            vector_db=FakeVectorDB(docs),
            keyword_db=FakeKeywordDB([]),
            embedding_model=FakeEmbeddingModel(),
        )

        result = retriever.retrieve_with_trace("tech_docs_kb", "DeepSeek deployment error troubleshooting")

        diagnostics = result.metadata.get("diagnostics", {})
        self.assertEqual(diagnostics.get("reason_code"), "low_relevance")
        self.assertEqual(diagnostics.get("suggested_action"), "fallback_to_chat")
        self.assertTrue(diagnostics.get("rewrite_applied"))

    def test_cross_encoder_reranker_orders_documents_by_model_score(self):
        class FakeCrossEncoder:
            def predict(self, pairs, batch_size=16, show_progress_bar=False):
                self.pairs = pairs
                self.batch_size = batch_size
                return [0.2, 0.95]

        fake_model = FakeCrossEncoder()
        docs = [
            DocumentChunk(id="chunk-a", content="generic deployment notes", metadata={}, score=0.6),
            DocumentChunk(id="chunk-b", content="DeepSeek API key setup flow", metadata={}, score=0.4),
        ]
        reranker = CrossEncoderReranker(
            model_name="fake-cross-encoder",
            batch_size=8,
            model_factory=lambda model_name, device: fake_model,
        )

        result = reranker.rerank("DeepSeek API key flow", docs, top_k=2)

        self.assertEqual([doc.id for doc in result], ["chunk-b", "chunk-a"])
        self.assertEqual(fake_model.batch_size, 8)
        self.assertEqual(result[0].metadata["rerank"]["backend"], "cross_encoder")
        self.assertEqual(result[0].metadata["rerank"]["model"], "fake-cross-encoder")
        self.assertEqual(result[0].metadata["rerank"]["rerank_rank"], 1)

    def test_cross_encoder_reranker_falls_back_to_lexical_when_model_load_fails(self):
        docs = [
            DocumentChunk(id="chunk-a", content="finance summary", metadata={}, score=0.9),
            DocumentChunk(id="chunk-b", content="DeepSeek API troubleshooting guide", metadata={}, score=0.2),
        ]
        reranker = CrossEncoderReranker(
            model_name="missing-model",
            model_factory=lambda model_name, device: (_ for _ in ()).throw(RuntimeError("model unavailable")),
        )

        result = reranker.rerank("DeepSeek API troubleshooting", docs, top_k=2)

        self.assertEqual(reranker.last_backend, "lexical_fallback")
        self.assertIn("model unavailable", reranker.last_fallback_reason or "")
        self.assertEqual(result[0].metadata["rerank"]["backend"], "lexical_fallback")
        self.assertEqual(result[0].metadata["rerank"]["requested_model"], "missing-model")

    def test_reranker_factory_supports_noop_backend(self):
        reranker = build_reranker(backend="noop")

        self.assertIsInstance(reranker, NoOpReranker)

    def test_lexical_reranker_uses_readable_chinese_stop_terms(self):
        terms = LexicalOverlapReranker._generic_query_terms

        self.assertIn("知识库", terms)
        self.assertIn("资料", terms)
        replacement_char = chr(0xFFFD)
        mojibake_marker = "".join(chr(codepoint) for codepoint in (0x9225, 0x2025))
        self.assertFalse(
            any(replacement_char in item or mojibake_marker in item for item in terms)
        )

    def test_mysql_memory_adapter_serializes_messages_and_snapshots(self):
        class FakeCursor:
            def __init__(self, connection):
                self.connection = connection

            def execute(self, sql, params=None):
                self.connection.executed.append((sql, params))

            def fetchall(self):
                return list(self.connection.rows)

        class FakeConnection:
            def __init__(self):
                self.executed = []
                self.rows = []
                self.commits = 0
                self.closed = False

            def cursor(self):
                return FakeCursor(self)

            def commit(self):
                self.commits += 1

            def rollback(self):
                pass

            def close(self):
                self.closed = True

        connection = FakeConnection()
        adapter = MySQLMemoryStoreAdapter(
            connection_factory=lambda: connection,
            initialize_schema=False,
        )
        snapshot = TaskSnapshot(
            session_id="session-1",
            query="continue task",
            route_worker="tool",
            status="running",
            payload={"step": 2},
        )

        self.assertTrue(adapter.add_message("session-1", Message(role="user", content="hello")))
        self.assertTrue(adapter.save_task_snapshot(snapshot))

        executed_sql = "\n".join(sql for sql, _params in connection.executed)
        self.assertIn("INSERT INTO messages", executed_sql)
        self.assertIn("INSERT INTO task_snapshots", executed_sql)
        self.assertEqual(connection.commits, 2)
        task_params = connection.executed[-1][1]
        self.assertIn('"step": 2', task_params[-1])

    def test_mysql_memory_adapter_reads_task_snapshot(self):
        class FakeCursor:
            def __init__(self, connection):
                self.connection = connection

            def execute(self, sql, params=None):
                self.connection.executed.append((sql, params))

            def fetchall(self):
                return [
                    (
                        "session-1",
                        "original query",
                        "rag",
                        "running",
                        '{"retrieved": true}',
                        "2026-06-16 10:00:00",
                    )
                ]

        class FakeConnection:
            def __init__(self):
                self.executed = []

            def cursor(self):
                return FakeCursor(self)

            def close(self):
                pass

        adapter = MySQLMemoryStoreAdapter(
            connection_factory=FakeConnection,
            initialize_schema=False,
        )

        snapshot = adapter.get_task_snapshot("session-1")

        self.assertIsNotNone(snapshot)
        self.assertEqual(snapshot.session_id, "session-1")
        self.assertEqual(snapshot.payload, {"retrieved": True})

    def test_elasticsearch_keyword_adapter_indexes_and_searches_chunks(self):
        class FakeIndices:
            def __init__(self):
                self.created = []

            def exists(self, index):
                return False

            def create(self, index, mappings):
                self.created.append((index, mappings))

            def refresh(self, index):
                self.refreshed = index

            def get(self, index, allow_no_indices=True):
                return {"deepassist-kb-medical": {}, "deepassist-kb-tech": {}}

        class FakeElasticsearch:
            def __init__(self):
                self.indices = FakeIndices()
                self.indexed = []
                self.deleted = []

            def index(self, index, id, document):
                self.indexed.append((index, id, document))

            def search(self, index, size=10, query=None, aggs=None):
                if aggs:
                    return {
                        "aggregations": {
                            "sources": {
                                "buckets": [{"key": "guide.md", "doc_count": 2}]
                            }
                        }
                    }
                return {
                    "hits": {
                        "hits": [
                            {
                                "_id": "chunk-1",
                                "_score": 3.5,
                                "_source": {
                                    "id": "chunk-1",
                                    "content": "DeepAssist RAG guide",
                                    "source_file": "guide.md",
                                    "metadata": {"title": "Guide"},
                                },
                            }
                        ]
                    }
                }

            def delete_by_query(self, index, query, conflicts, refresh):
                self.deleted.append((index, query, conflicts, refresh))

        client = FakeElasticsearch()
        adapter = ElasticsearchKeywordSearchAdapter(client=client)
        chunk = DocumentChunk(
            id="chunk-1",
            content="DeepAssist RAG guide",
            metadata={"source_file": "guide.md"},
            score=1.0,
        )

        self.assertTrue(adapter.build_index("medical", [chunk]))
        docs = adapter.search("medical", "rag guide", top_k=3)
        sources = adapter.list_sources("medical")
        collections = adapter.list_collections()
        self.assertTrue(adapter.delete_by_source("medical", "guide.md"))

        self.assertEqual(client.indexed[0][0], "deepassist-kb-medical")
        self.assertEqual(docs[0].id, "chunk-1")
        self.assertEqual(docs[0].metadata["keyword_backend"], "elasticsearch")
        self.assertEqual(sources[0].source_file, "guide.md")
        self.assertEqual(collections, ["medical", "tech"])
        self.assertEqual(client.deleted[0][1], {"term": {"source_file.keyword": "guide.md"}})

    def test_answer_guard_recommends_regeneration_when_citations_are_missing(self):
        guard = SourceAwareResponseGuard()
        context_pack = RAGContextPack(
            query="what is DeepSeek API Key flow",
            citations=[
                Citation(
                    ref_id="C1",
                    chunk_id="chunk-1",
                    content="DeepSeek API Key guide",
                    source_file="kb.md",
                )
            ],
            rendered_context="[C1] DeepSeek API Key guide",
            budget_chars=1000,
            used_chars=28,
        )

        report = guard.check("The API key is issued in the console.", context_pack)

        self.assertFalse(report.grounded)
        self.assertEqual(report.recommended_action, "regenerate_with_citations")
        self.assertIn("answer_missing_citations", report.warnings)

    def test_explicit_kb_query_with_low_relevance_returns_safe_kb_miss_answer(self):
        store = FakeMemoryStore()
        session_manager = SessionManager(store)
        llm = FakeLLM(["hallucinated answer"])
        app = ChatApplication(
            llm=llm,
            session_manager=session_manager,
            context_engine=ContextEngine(),
            intent_router=IntentRouter(),
            rag_pipeline=FakeRAGPipeline(score=0.1),
        )

        events = list(
            app.stream_chat(
                "session-rag-kb-miss",
                "Answer from the knowledge base about the DeepSeek API key flow and include citations.",
                "rag",
                collection_name="__all__",
            )
        )
        payloads = [
            json.loads(chunk[6:].strip())
            for chunk in events
            if chunk.startswith("data: ") and chunk[6:].strip()
        ]
        answer_text = "".join(
            payload.get("content", "")
            for payload in payloads
            if payload.get("event") == "message_delta"
        )

        self.assertEqual(len(llm.calls), 0)
        self.assertTrue(answer_text.strip())


    def test_required_rag_low_relevance_trace_uses_kb_miss_action(self):
        store = FakeMemoryStore()
        session_manager = SessionManager(store)
        app = ChatApplication(
            llm=FakeLLM(["hallucinated answer"]),
            session_manager=session_manager,
            context_engine=ContextEngine(),
            intent_router=IntentRouter(),
            rag_pipeline=FakeRAGPipeline(score=0.1),
        )

        payloads = parse_sse_events(
            list(
                app.stream_rag_chat(
                    "session-required-rag-kb-miss",
                    "Use the knowledge base: what should I do about a headache?",
                    collection_name="medical_kb",
                    source_policy=SourcePolicy.REQUIRED,
                )
            )
        )
        retrieval = next(item for item in payloads if item.get("event") == "retrieval_trace")
        diagnostics = retrieval.get("data", {}).get("metadata", {}).get("diagnostics", {})

        self.assertEqual(diagnostics.get("source_policy"), "required")
        self.assertEqual(diagnostics.get("final_action"), "direct_kb_miss")
        self.assertEqual(diagnostics.get("suggested_action"), "direct_kb_miss")

    def test_cross_collection_rag_metadata_recomputes_low_relevance_from_aggregate_score(self):
        from backend.domain.services.rag.pipeline import RAGPipeline

        pipeline = RAGPipeline.__new__(RAGPipeline)
        pipeline.low_relevance_threshold = 0.2
        doc = DocumentChunk(
            id="chunk-good",
            content="Headache relief evidence",
            metadata={"source_file": "medical.json"},
            score=0.93,
        )
        base = RetrievalResult(
            query="What should I do about a headache?",
            collection_name="enron_kb",
            documents=[doc],
            top_k=5,
            candidate_k=10,
            metadata={
                "diagnostics": {
                    "reason_code": "low_relevance",
                    "reason_message": "top_document_score_below_threshold",
                    "suggested_action": "fallback_to_chat",
                    "low_relevance_threshold": 0.2,
                }
            },
        )

        metadata = pipeline._aggregate_metadata(base, [base], [doc])
        diagnostics = metadata.get("diagnostics", {})

        self.assertEqual(diagnostics.get("reason_code"), "ok")
        self.assertEqual(diagnostics.get("suggested_action"), "proceed_with_rag")
        self.assertEqual(diagnostics.get("best_score"), 0.93)

    def test_agent_engine_assesses_and_dedupes_duplicate_tool_calls_in_one_iteration(self):
        execution_counter = {"count": 0}

        def sample_tool(city: str) -> str:
            execution_counter["count"] += 1
            return f"lookup:{city}"

        first_response = SimpleNamespace(
            content="",
            tool_calls=[
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {
                        "name": "sample_tool",
                        "arguments": json.dumps({"city": "beijing"}),
                    },
                },
                {
                    "id": "call-2",
                    "type": "function",
                    "function": {
                        "name": "sample_tool",
                        "arguments": json.dumps({"city": "beijing"}),
                    },
                },
            ],
            reasoning_content=None,
        )
        second_response = SimpleNamespace(
            content="Done.",
            tool_calls=[],
            reasoning_content=None,
        )
        engine = AgentEngine(
            llm=FakeAgentLLM([first_response, second_response]),
            tool_registry=ToolRegistry.from_callables([sample_tool]),
            run_config=AgentRunConfig(max_iterations=3, max_self_corrections=2),
        )

        events = list(
            engine.stream_run(
                [
                    {"role": "system", "content": "agent"},
                    {"role": "user", "content": "check lookup twice"},
                ]
            )
        )

        self.assertEqual(execution_counter["count"], 1)
        plan_events = [event for event in events if event.get("type") == "plan_assessment"]
        self.assertTrue(plan_events)
        self.assertEqual(plan_events[0]["data"].get("recommended_mode"), "simplify_plan")
        self.assertIn(
            "duplicate_tool_call_in_iteration",
            plan_events[0]["data"].get("warnings", []),
        )

    def test_agent_engine_finalizes_partial_answer_at_step_limit_after_successful_tools(self):
        def fetch_fact() -> str:
            return "observed fact"

        first_response = SimpleNamespace(
            content="",
            tool_calls=[
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {"name": "fetch_fact", "arguments": json.dumps({})},
                }
            ],
            reasoning_content=None,
        )
        second_response = SimpleNamespace(
            content="Need one more lookup.",
            tool_calls=[
                {
                    "id": "call-2",
                    "type": "function",
                    "function": {"name": "fetch_fact", "arguments": json.dumps({})},
                }
            ],
            reasoning_content=None,
        )
        engine = AgentEngine(
            llm=FakeAgentLLM([first_response, second_response]),
            tool_registry=ToolRegistry.from_callables([fetch_fact]),
            run_config=AgentRunConfig(max_iterations=2, max_self_corrections=0),
        )

        events = list(
            engine.stream_run(
                [
                    {"role": "system", "content": "agent"},
                    {"role": "user", "content": "collect facts"},
                ]
            )
        )
        final = next(event for event in events if event.get("type") == "final_answer")
        finish = next(event for event in events if event.get("type") == "finish")

        self.assertIn("observed fact", final.get("content", ""))
        self.assertFalse(any(event.get("type") == "error" for event in events))
        self.assertEqual(finish.get("new_messages", [])[-1].get("role"), "assistant")
        self.assertFalse(finish.get("new_messages", [])[-1].get("tool_calls"))

    def test_agent_engine_escalates_recovery_to_partial_result_finalize_after_repeated_failure(self):
        def fetch_fact() -> str:
            return "partial fact"

        def failing_tool(target: str) -> str:
            raise RuntimeError("temporary timeout while contacting upstream")

        first_response = SimpleNamespace(
            content="",
            tool_calls=[
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {"name": "fetch_fact", "arguments": json.dumps({})},
                },
                {
                    "id": "call-2",
                    "type": "function",
                    "function": {
                        "name": "failing_tool",
                        "arguments": json.dumps({"target": "alpha"}),
                    },
                },
            ],
            reasoning_content=None,
        )
        second_response = SimpleNamespace(
            content="",
            tool_calls=[
                {
                    "id": "call-3",
                    "type": "function",
                    "function": {
                        "name": "failing_tool",
                        "arguments": json.dumps({"target": "alpha"}),
                    },
                }
            ],
            reasoning_content=None,
        )
        third_response = SimpleNamespace(
            content="Recovered with partial results.",
            tool_calls=[],
            reasoning_content=None,
        )
        engine = AgentEngine(
            llm=FakeAgentLLM([first_response, second_response, third_response]),
            tool_registry=ToolRegistry.from_callables([fetch_fact, failing_tool]),
            run_config=AgentRunConfig(max_iterations=4, max_self_corrections=3),
        )

        events = list(
            engine.stream_run(
                [
                    {"role": "system", "content": "agent"},
                    {"role": "user", "content": "solve with tools"},
                ]
            )
        )

        recovery_events = [event for event in events if event.get("type") == "failure_recovery"]
        self.assertGreaterEqual(len(recovery_events), 2)
        self.assertEqual(
            recovery_events[-1]["data"].get("action"),
            "finalize_with_partial_results",
        )
        self.assertEqual(
            recovery_events[-1]["data"].get("successful_tool_calls"),
            1,
        )

    def test_agent_supervisor_routes_complex_mixed_task_to_orchestrator(self):
        supervisor = AgentSupervisor(
            intent_router=IntentRouter(),
            chat_worker=object(),
            rag_worker=object(),
            tool_worker=object(),
            orchestrator_worker=object(),
            task_decomposer=TaskDecomposer(),
        )

        decision = supervisor.decide(
            "Use rag to search DeepSeek docs, then use python to calculate, and summarize."
        )

        self.assertEqual(decision.worker, AgentWorkerType.ORCHESTRATOR)
        self.assertTrue(any(signal in decision.signals for signal in ["rag", "python"]))

    def test_orchestrator_worker_decomposes_and_synthesizes_collaborator_results(self):
        rag_worker = FakeCollaboratorWorker(
            "RAG evidence: API Key is created in console.",
            extra_events=[
                {
                    "type": "retrieval_trace",
                    "data": {"hit_count": 2, "candidate_k": 10, "fusion": "weighted_rrf"},
                }
            ],
        )
        tool_worker = FakeCollaboratorWorker("Tool result: computed delta is 4.4.")
        chat_worker = FakeCollaboratorWorker("Chat framing: answer concisely.")
        llm = FakeSynthesisLLM(["Final", " synthesized", " answer"])
        worker = OrchestratorWorker(
            llm=llm,
            chat_worker=chat_worker,
            rag_worker=rag_worker,
            tool_worker=tool_worker,
            task_decomposer=TaskDecomposer(),
        )

        events = list(
            worker.stream(
                query="Use rag to search DeepSeek docs, then use python to calculate, and summarize.",
                history=[],
            )
        )

        plan_events = [event for event in events if event.get("type") == "multi_agent_plan"]
        collaborator_events = [
            event for event in events if event.get("type") == "collaborator_trace"
        ]
        final_chunks = [
            event.get("content", "")
            for event in events
            if event.get("type") == "message_delta"
        ]

        self.assertTrue(plan_events)
        self.assertEqual(plan_events[0]["data"].get("task_count"), 2)
        self.assertGreaterEqual(len(collaborator_events), 4)
        self.assertEqual("".join(final_chunks), "Final synthesized answer")
        self.assertTrue(llm.calls)
        synthesis_prompt = llm.calls[0][1]["content"]
        self.assertIn("RAG evidence", synthesis_prompt)
        self.assertIn("Tool result", synthesis_prompt)

    def test_agent_application_resumes_interrupted_task_from_snapshot(self):
        store = FakeMemoryStore()
        store.save_task_snapshot(
            TaskSnapshot(
                session_id="session-recover",
                query="compute stats",
                route_worker="tool_agent_worker",
                status="interrupted",
                payload={
                    "messages": [
                        {"role": "system", "content": "agent"},
                        {"role": "user", "content": "compute stats"},
                    ],
                    "lifecycle_state": {"iterations": 1},
                },
            )
        )
        supervisor = FakeRecoverySupervisor()
        app = AgentApplication(
            agent_engine=None,
            session_manager=SessionManager(store),
            context_engine=ContextEngine(),
            profile_extractor=FakeProfileExtractor(),
            supervisor=supervisor,
        )

        events = list(app.stream_agent_task("session-recover", "continue"))
        payloads = [
            json.loads(chunk[6:].strip())
            for chunk in events
            if chunk.startswith("data: ") and chunk[6:].strip()
        ]

        recovery_events = [payload for payload in payloads if payload.get("event") == "task_recovery"]
        self.assertTrue(recovery_events)
        self.assertEqual(supervisor.resume_calls[0]["worker_kind"], "tool_agent_worker")
        self.assertEqual(supervisor.resume_calls[0]["query"], "compute stats")
        self.assertIsNone(store.get_task_snapshot("session-recover"))

    def test_orchestrator_worker_resumes_from_saved_progress(self):
        rag_worker = FakeCollaboratorWorker("RAG evidence: API Key is created in console.")
        tool_worker = FakeCollaboratorWorker("Tool result: computed delta is 4.4.")
        chat_worker = FakeCollaboratorWorker("Chat framing: answer concisely.")
        llm = FakeSynthesisLLM(["Recovered final answer"])
        worker = OrchestratorWorker(
            llm=llm,
            chat_worker=chat_worker,
            rag_worker=rag_worker,
            tool_worker=tool_worker,
            task_decomposer=TaskDecomposer(),
        )
        recovery_state = {
            "plan": MultiAgentPlan(
                tasks=[
                    {
                        "task_id": "task-rag-1",
                        "title": "Collect grounded knowledge-base evidence",
                        "worker": AgentWorkerType.RAG,
                        "query": "query",
                        "rationale": "rag",
                    },
                    {
                        "task_id": "task-tool-1",
                        "title": "Execute tools and gather operational results",
                        "worker": AgentWorkerType.TOOL,
                        "query": "query",
                        "rationale": "tool",
                    },
                ]
            ).model_dump(),
            "collaborator_results": [
                {
                    "task_id": "task-rag-1",
                    "title": "Collect grounded knowledge-base evidence",
                    "worker": AgentWorkerType.RAG.value,
                    "answer": "Recovered RAG evidence",
                }
            ],
            "next_task_index": 1,
        }

        events = list(
            worker.stream(
                query="complex task",
                history=[],
                recovery_state=recovery_state,
            )
        )

        self.assertEqual(len(rag_worker.calls), 0)
        self.assertEqual(len(tool_worker.calls), 1)
        self.assertTrue(any(event.get("type") == "task_snapshot" for event in events))
        final_chunks = [
            event.get("content", "")
            for event in events
            if event.get("type") == "message_delta"
        ]
        self.assertEqual("".join(final_chunks), "Recovered final answer")


class MCPAndCapabilityTests(unittest.TestCase):
    def test_deepassist_mcp_server_exposes_read_only_tools(self):
        class FakeKBApp:
            def list_collections(self):
                return {
                    "status": "success",
                    "data": [{"collection_name": "tech_docs_kb", "file_count": 1}],
                }

            def list_files(self, collection_name: str = "tech_docs_kb"):
                return {
                    "status": "success",
                    "data": [
                        {
                            "source_file": "guide.md",
                            "source_path": "C:\\secret\\guide.md",
                            "chunk_count": 3,
                        }
                    ],
                }

        server = DeepAssistMCPServer(kb_app=FakeKBApp())

        tool_names = {tool["name"] for tool in server.list_tools()}
        self.assertIn("search_kb", tool_names)
        self.assertIn("list_kb_files", tool_names)
        self.assertIn("get_runtime_capabilities", tool_names)

        result = server.call_tool("list_kb_files", {"collection_name": "tech_docs_kb"})
        payload = result["structuredContent"]

        self.assertFalse(result["isError"])
        self.assertEqual(payload["data"][0]["source_path"], "guide.md")

    def test_mcp_tool_client_adapter_normalizes_and_executes_tools(self):
        class FakeMCPClient:
            def list_tools(self):
                return [
                    {
                        "name": "lookup_docs",
                        "description": "Lookup docs",
                        "inputSchema": {
                            "type": "object",
                            "properties": {"query": {"type": "string"}},
                            "required": ["query"],
                        },
                    }
                ]

            def call_tool(self, name, arguments):
                return {"name": name, "arguments": arguments, "answer": "ok"}

        adapter = MCPToolClientAdapter(
            client=FakeMCPClient(),
            server_name="docs",
            tool_prefix="mcp_docs__",
        )

        tools = adapter.list_tools()
        self.assertEqual(tools[0]["name"], "mcp_docs__lookup_docs")
        self.assertEqual(tools[0]["parameters"]["required"], ["query"])

        result = adapter.execute(
            ToolCall(id="call-1", name="mcp_docs__lookup_docs", args={"query": "DeepAssist"})
        )

        self.assertTrue(result.success)
        self.assertIn('"answer": "ok"', result.content)
        self.assertEqual(result.metadata["remote_tool_name"], "lookup_docs")

    def test_tool_registry_executes_external_provider_under_policy(self):
        class FakeProvider:
            server_name = "docs"

            def list_tools(self):
                return [
                    {
                        "name": "mcp_docs__lookup_docs",
                        "description": "Lookup docs",
                        "parameters": {
                            "type": "object",
                            "properties": {"query": {"type": "string"}},
                            "required": ["query"],
                        },
                    }
                ]

            def execute(self, tool_call):
                from backend.domain.entities.tooling import ToolResult

                return ToolResult.ok(tool_call, f"found:{tool_call.args['query']}")

        registry = ToolRegistry(policy=ToolPolicy(allowed_categories={"read_only"}))
        registry.register_provider(FakeProvider(), category="read_only")

        specs = registry.list_tool_specs()
        self.assertEqual(specs[0]["provider"], "mcp")
        result = registry.execute(
            ToolCall(id="call-1", name="mcp_docs__lookup_docs", args={"query": "RAG"})
        )

        self.assertTrue(result.success)
        self.assertEqual(result.content, "found:RAG")

    def test_langchain_tool_adapter_normalizes_and_executes_tools(self):
        class FakeLangChainTool:
            name = "lookup_docs"
            description = "Lookup docs"
            args = {"query": {"type": "string", "description": "Search query"}}

            def invoke(self, args):
                return {"answer": f"found:{args['query']}"}

        adapter = LangChainToolAdapter([FakeLangChainTool()])
        tools = adapter.list_tools()
        self.assertEqual(tools[0]["name"], "lookup_docs")
        self.assertEqual(tools[0]["provider"], "langchain")

        registry = ToolRegistry(policy=ToolPolicy(allowed_categories={"read_only"}))
        registry.register_provider(adapter, category="read_only")
        result = registry.execute(
            ToolCall(id="call-langchain", name="lookup_docs", args={"query": "DeepAssist"})
        )

        self.assertTrue(result.success)
        self.assertIn("found:DeepAssist", result.content)

    def test_runtime_capabilities_include_tool_groups(self):
        class FakeKBApp:
            def list_collections(self):
                return {
                    "status": "success",
                    "data": [{"collection_name": "tech_docs_kb"}],
                }

        def search_knowledge_base(query: str) -> str:
            """Search KB."""
            return query

        def list_knowledge_base_collections() -> str:
            """List KB collections."""
            return "tech_docs_kb"

        registry = ToolRegistry.from_callables(
            [search_knowledge_base, list_knowledge_base_collections],
            policy=ToolPolicy(
                allowed_categories={"read_only"},
                tool_categories={
                    "search_knowledge_base": "read_only",
                    "list_knowledge_base_collections": "read_only",
                },
            ),
        )
        app = RuntimeApplication(kb_app=FakeKBApp(), tool_registry=registry)

        payload = app.get_capabilities()["data"]

        capability_names = {item["name"] for item in payload["capabilities"]}
        group_names = {item["name"] for item in payload["tool_groups"]}
        self.assertIn("knowledge_search", capability_names)
        self.assertIn("file_write", capability_names)
        self.assertIn("python_execution", capability_names)
        self.assertIn("external_mcp_tools", capability_names)
        self.assertIn("knowledge_search", group_names)
        self.assertIn("python_execution", group_names)
        file_write_group = next(
            item for item in payload["tool_groups"] if item["name"] == "file_write_legacy"
        )
        self.assertEqual(file_write_group["capability"], "file_write")
        self.assertFalse(file_write_group["enabled_by_default"])
        knowledge_group = next(
            item for item in payload["tool_groups"] if item["name"] == "knowledge_search"
        )
        self.assertIn("search_knowledge_base", knowledge_group["tools"])

    def test_eval_runner_scaffold_validates_case_files(self):
        report = build_scaffold_report(Path("evals"))

        self.assertEqual(report["status"], "success")
        self.assertTrue(report["full_eval_executed"])
        self.assertFalse(report["live_adapter_eval_executed"])
        self.assertEqual(report["case_counts"]["route"], 1)
        self.assertEqual(report["case_counts"]["rag"], 1)
        self.assertEqual(report["case_counts"]["agent_tool"], 1)
        self.assertEqual(report["metrics"]["route_accuracy"], 1.0)

    def test_eval_runner_build_eval_report_outputs_offline_metrics(self):
        report = build_eval_report(Path("evals"))

        self.assertEqual(report["mode"], "offline_static")
        self.assertEqual(report["metrics"]["rag_case_validity"], 1.0)
        self.assertEqual(report["metrics"]["agent_tool_case_validity"], 1.0)

    def test_public_eval_catalog_lists_expected_benchmark_sources(self):
        self.assertIn("squad", DATASET_SPECS)
        self.assertIn("gsm8k", DATASET_SPECS)
        self.assertIn("beir_nfcorpus", DATASET_SPECS)
        self.assertEqual(DATASET_SPECS["gsm8k"].suite, "agent")
        self.assertEqual(DATASET_SPECS["squad"].suite, "rag")

    def test_prepare_dataset_converts_rag_and_agent_rows_to_standard_schema(self):
        rag_case = convert_rag_row(
            "squad",
            {
                "id": "case-1",
                "question": "What reranker is used?",
                "context": "DeepAssist uses Cross-Encoder reranking.",
                "answers": {"text": ["Cross-Encoder"]},
            },
        )
        agent_case = convert_agent_row(
            "gsm8k",
            {
                "id": "math-1",
                "question": "Compute a value.",
                "answer": "The answer is 7.\n#### 7",
            },
        )

        validate_case(rag_case)
        validate_case(agent_case)
        self.assertEqual(rag_case["suite"], "rag")
        self.assertEqual(rag_case["expected_tools"], ["search_knowledge_base"])
        self.assertEqual(agent_case["suite"], "agent")
        self.assertEqual(agent_case["gold_answer"], "7")
        self.assertEqual(agent_case["expected_tools"], ["execute_python_sandbox"])

    def test_benchmark_eval_outputs_metrics_and_artifacts_from_processed_cases(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            processed = root / "processed"
            processed.mkdir()
            (processed / "retrieval_cases.jsonl").write_text(
                json.dumps(
                    {
                        "id": "ret-1",
                        "suite": "retrieval",
                        "question": "Find reranker doc",
                        "gold_doc_ids": ["doc-1"],
                        "ranked_doc_ids": ["doc-1", "doc-2"],
                        "pre_rerank_doc_ids": ["doc-2", "doc-1"],
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            (processed / "rag_cases.jsonl").write_text(
                json.dumps(
                    {
                        "id": "rag-1",
                        "suite": "rag",
                        "question": "What reranker?",
                        "gold_answer": "Cross-Encoder",
                        "predicted_answer": "Cross-Encoder [C1]",
                        "gold_doc_ids": ["doc-1"],
                        "citation_doc_ids": ["doc-1"],
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            (processed / "agent_cases.jsonl").write_text(
                json.dumps(
                    {
                        "id": "agent-1",
                        "suite": "agent",
                        "question": "Use Python",
                        "gold_answer": "2.3333",
                        "predicted_answer": "2.3333",
                        "expected_route": "agent",
                        "predicted_route": "agent",
                        "expected_tools": ["execute_python_sandbox"],
                        "called_tools": ["execute_python_sandbox"],
                        "arguments_valid": True,
                        "tool_execution_success": True,
                        "hitl_resume_success": True,
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )

            report = run_benchmark_eval(
                processed_dir=processed,
                suite="all",
                sample_size=20,
                live=False,
                results_dir=root / "results",
                run_id="test-run",
            )

            self.assertEqual(report["status"], "success")
            self.assertEqual(report["mode"], "offline_processed")
            self.assertEqual(report["metrics"]["retrieval_recall@10"], 1.0)
            self.assertGreater(report["metrics"]["rag_token_f1"], 0.6)
            self.assertEqual(report["metrics"]["rag_citation_presence"], 1.0)
            self.assertEqual(report["metrics"]["agent_final_answer_accuracy"], 1.0)
            self.assertTrue((root / "results" / "test-run" / "metrics.json").exists())

    def test_eval_report_generator_writes_markdown_summary(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            processed = root / "processed"
            results = root / "results"
            processed.mkdir()
            (processed / "retrieval_cases.jsonl").write_text(
                '{"id":"ret-1","suite":"retrieval","question":"q","gold_doc_ids":["d1"],"ranked_doc_ids":["d1"]}\n',
                encoding="utf-8",
            )
            (processed / "rag_cases.jsonl").write_text("", encoding="utf-8")
            (processed / "agent_cases.jsonl").write_text("", encoding="utf-8")
            run_benchmark_eval(
                processed_dir=processed,
                suite="retrieval",
                sample_size=20,
                live=False,
                results_dir=results,
                run_id="report-run",
            )

            output = root / "evaluation_report.md"
            payload = generate_report(
                results_dir=results,
                run_id="latest",
                output_path=output,
            )

            self.assertEqual(payload["status"], "success")
            text = output.read_text(encoding="utf-8")
            self.assertIn("DeepAssist Evaluation Report", text)
            self.assertIn("retrieval_recall@10", text)


if __name__ == "__main__":
    unittest.main()

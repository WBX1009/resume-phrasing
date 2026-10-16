import json
import time
from typing import Iterator, Optional
from uuid import uuid4

from backend.application.agent_app import AgentApplication
from backend.application.chat_app import ChatApplication
from backend.common.config import settings
from backend.common.logger import get_logger
from backend.domain.entities.route import ConversationRoute, RouteDecision
from backend.domain.entities.stream_event import StreamEvent
from backend.domain.entities.trace_event import TraceEvent
from backend.domain.interfaces.trace_store import TraceStorePort
from backend.domain.services.conversation.router import ConversationRouter
from backend.domain.services.session.manager import SessionManager
from backend.infrastructure.streaming.sse_manager import SSEManager

logger = get_logger(__name__)


class ConversationApplication:
    """Unified conversation entrypoint that routes each turn to a workflow."""

    def __init__(
        self,
        conversation_router: ConversationRouter,
        chat_app: ChatApplication,
        agent_app: AgentApplication,
        session_manager: SessionManager,
        trace_store: Optional[TraceStorePort] = None,
    ):
        self.router = conversation_router
        self.chat_app = chat_app
        self.agent_app = agent_app
        self.session_mgr = session_manager
        self.trace_store = trace_store

    def stream_conversation(
        self,
        session_id: str,
        query: str,
        collection_name: str = "__all__",
        model_name: Optional[str] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        history_rounds: int = 10,
        use_user_memory: bool = False,
    ) -> Iterator[str]:
        trace_id = uuid4().hex
        request_started = time.perf_counter()
        history_budget = max(1, history_rounds)
        history = self.session_mgr.get_session_history(session_id, limit=min(12, history_budget * 2))
        decision = self.router.route(
            query=query,
            history=history[-6:],
        )

        self._log_route(
            session_id=session_id,
            query=query,
            history=history[-6:],
            decision=decision,
            final_used_route=decision.route.value,
        )
        route_data = {
            **decision.to_stream_data(),
            "trace_id": trace_id,
            "session_id": session_id,
        }
        self._append_trace(
            TraceEvent(
                trace_id=trace_id,
                session_id=session_id,
                route=decision.route.value,
                event_type="routing_decision",
                span_name="conversation.route",
                latency_ms=self._elapsed_ms(request_started),
                input_summary=self._summarize(query),
                output_summary=decision.reason,
                metadata={
                    "confidence": decision.confidence,
                    "source_policy": decision.source_policy.value,
                    "risk_level": decision.risk_level.value,
                },
            )
        )
        yield SSEManager.format_event(StreamEvent.routing_decision(route_data))

        rewritten_query = decision.rewritten_query or query
        if decision.route == ConversationRoute.CLARIFY:
            answer = decision.clarifying_question or "Can you clarify what you want me to do next?"
            for chunk in self._chunk_text(answer):
                yield self._format_and_trace_event(
                    StreamEvent.message_delta(chunk),
                    trace_id,
                    session_id,
                    decision.route.value,
                )
            self.session_mgr.save_interaction(session_id, query, answer)
            yield self._format_and_trace_event(
                StreamEvent.done(),
                trace_id,
                session_id,
                decision.route.value,
            )
            return

        if decision.route == ConversationRoute.DIRECT_CHAT:
            for chunk in self.chat_app.stream_direct_chat(
                session_id=session_id,
                query=rewritten_query,
                model_name=model_name,
                temperature=temperature,
                top_p=top_p,
                history_rounds=history_rounds,
                use_user_memory=use_user_memory,
                original_query=query,
            ):
                yield self._trace_sse_chunk(
                    chunk,
                    trace_id,
                    session_id,
                    decision.route.value,
                )
            return

        if decision.route == ConversationRoute.RAG:
            for chunk in self.chat_app.stream_rag_chat(
                session_id=session_id,
                query=rewritten_query,
                collection_name=collection_name,
                model_name=model_name,
                temperature=temperature,
                top_p=top_p,
                history_rounds=history_rounds,
                use_user_memory=use_user_memory,
                source_policy=decision.source_policy,
                original_query=query,
            ):
                yield self._trace_sse_chunk(
                    chunk,
                    trace_id,
                    session_id,
                    decision.route.value,
                )
            return

        if decision.route == ConversationRoute.AGENT:
            for chunk in self.agent_app.stream_agent_task(
                session_id=session_id,
                query=rewritten_query,
                use_user_memory=use_user_memory,
                model_name=model_name,
                temperature=temperature,
                top_p=top_p,
                history_rounds=history_rounds,
                original_query=query,
            ):
                yield self._trace_sse_chunk(
                    chunk,
                    trace_id,
                    session_id,
                    decision.route.value,
                )
            return

        yield self._format_and_trace_event(
            StreamEvent.error(f"Unsupported conversation route: {decision.route.value}"),
            trace_id,
            session_id,
            decision.route.value,
        )
        yield self._format_and_trace_event(
            StreamEvent.done(),
            trace_id,
            session_id,
            decision.route.value,
        )

    def stream_pending_action_decision(
        self,
        action_id: str,
        approved: bool,
        session_id: Optional[str] = None,
        decided_by: Optional[str] = None,
        reason: Optional[str] = None,
        model_name: Optional[str] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
    ) -> Iterator[str]:
        trace_id = uuid4().hex
        route = "agent_approval_resume"
        resolved_session_id = session_id or ""
        self._append_trace(
            TraceEvent(
                trace_id=trace_id,
                session_id=resolved_session_id,
                route=route,
                event_type="approval_decision",
                span_name="conversation.approval_decision",
                input_summary=action_id,
                output_summary="approved" if approved else "rejected",
                metadata={"approved": approved},
            )
        )
        for chunk in self.agent_app.stream_pending_action_decision(
            action_id=action_id,
            approved=approved,
            session_id=session_id,
            decided_by=decided_by,
            reason=reason,
            model_name=model_name,
            temperature=temperature,
            top_p=top_p,
        ):
            yield self._trace_sse_chunk(
                chunk,
                trace_id,
                resolved_session_id,
                route,
            )

    def _log_route(
        self,
        session_id: str,
        query: str,
        history: list[dict],
        decision: RouteDecision,
        final_used_route: str,
        error: str = "",
    ) -> None:
        if not settings.ENABLE_ROUTING_LOG or not decision.should_log_for_training:
            return
        self.session_mgr.save_routing_log(
            {
                "session_id": session_id,
                "query": query,
                "history_preview": [
                    {
                        "role": str(item.get("role", "")),
                        "content": " ".join(str(item.get("content", "")).split())[:300],
                    }
                    for item in history
                    if item.get("content")
                ],
                "route": decision.route.value,
                "confidence": decision.confidence,
                "reason": decision.reason,
                "rewritten_query": decision.rewritten_query,
                "source_policy": decision.source_policy.value,
                "required_capabilities": decision.required_capabilities,
                "risk_level": decision.risk_level.value,
                "final_used_route": final_used_route,
                "error": error,
            }
        )

    def _chunk_text(self, text: str, chunk_size: int = 160):
        for start in range(0, len(text), chunk_size):
            yield text[start : start + chunk_size]

    def _format_and_trace_event(
        self,
        event: StreamEvent,
        trace_id: str,
        session_id: str,
        route: str,
    ) -> str:
        chunk = SSEManager.format_event(event)
        return self._trace_sse_chunk(chunk, trace_id, session_id, route)

    def _trace_sse_chunk(
        self,
        chunk: str,
        trace_id: str,
        session_id: str,
        route: str,
    ) -> str:
        payload = self._extract_sse_payload(chunk)
        if payload:
            event_type = str(payload.get("event") or "unknown")
            data = payload.get("data") if isinstance(payload.get("data"), dict) else {}
            metadata = data.get("metadata") if isinstance(data.get("metadata"), dict) else {}
            error_type = None
            if event_type == "error":
                error_type = "stream_error"
            elif metadata.get("error_type"):
                error_type = str(metadata["error_type"])
            elif data.get("success") is False:
                error_type = "tool_result_failed"
            self._append_trace(
                TraceEvent(
                    trace_id=trace_id,
                    session_id=session_id,
                    route=route,
                    event_type=event_type,
                    span_name=f"conversation.{event_type}",
                    input_summary=None,
                    output_summary=self._summarize(
                        str(payload.get("content") or payload.get("message") or "")
                    ),
                    error_type=error_type,
                    metadata={
                        "name": payload.get("name"),
                        "tool_call_id": data.get("tool_call_id"),
                        "success": data.get("success"),
                        "event_data": data,
                    },
                )
            )
        return chunk

    def _append_trace(self, event: TraceEvent) -> None:
        if self.trace_store is None:
            return
        try:
            self.trace_store.append(event)
        except Exception as exc:
            logger.warning("Failed to append trace event: %s", exc)

    def _extract_sse_payload(self, chunk: str) -> Optional[dict]:
        if not chunk.startswith("data: "):
            return None
        raw = chunk[6:].strip()
        if not raw:
            return None
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            return None
        return payload if isinstance(payload, dict) else None

    def _elapsed_ms(self, started_at: float) -> float:
        return round((time.perf_counter() - started_at) * 1000, 3)

    def _summarize(self, text: str, limit: int = 300) -> str:
        return " ".join(text.split())[:limit]

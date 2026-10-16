from typing import Any, Dict, Iterable, Iterator

from backend.domain.entities.stream_event import StreamEvent
from backend.domain.interfaces.agent_runtime import AgentRuntimePort, AgentRuntimeRequest
from backend.domain.services.agent.engine import AgentEngine


class CustomAgentRuntimeAdapter(AgentRuntimePort):
    """Adapter wrapper around the existing AgentEngine runtime."""

    runtime_name = "custom"

    def __init__(self, engine: AgentEngine):
        self.engine = engine

    def stream_run(self, request: AgentRuntimeRequest) -> Iterator[StreamEvent]:
        final_status = "completed"
        self._set_approval_context(request)
        yield StreamEvent(
            event="agent_runtime_node",
            name="CustomAgentRuntime",
            data={
                "runtime": self.runtime_name,
                "node": "custom_agent_engine",
                "status": "started",
            },
        )
        for event in self.engine.stream_run(
            request.messages,
            model_options=request.model_options,
            resume_state=request.resume_state,
        ):
            event_type = event.get("type")
            if event_type == "error":
                final_status = "failed"
            elif event_type == "finish":
                status = str(event.get("status") or "").strip()
                if status:
                    final_status = status
            yield from engine_event_to_stream_events(event)
        yield StreamEvent(
            event="agent_runtime_node",
            name="CustomAgentRuntime",
            data={
                "runtime": self.runtime_name,
                "node": "custom_agent_engine",
                "status": final_status,
            },
        )

    def _set_approval_context(self, request: AgentRuntimeRequest) -> None:
        registry = getattr(self.engine, "tool_registry", None)
        policy = getattr(registry, "policy", None)
        if policy is None or not hasattr(policy, "approval_context"):
            return
        policy.approval_context = {
            "session_id": request.session_id,
            "query": request.query,
            "runtime": self.runtime_name,
        }


def engine_event_to_stream_events(event: Dict[str, Any]) -> Iterable[StreamEvent]:
    """Convert AgentEngine internal events to DeepAssist StreamEvent objects."""

    event_type = event.get("type")

    if event_type == "status":
        yield StreamEvent.status(event.get("content", ""))
    elif event_type == "reasoning":
        yield StreamEvent.reasoning(event.get("content", ""))
    elif event_type == "model_call":
        yield StreamEvent.model_call(event.get("data", {}) or {})
    elif event_type == "message_delta":
        yield StreamEvent.message_delta(event.get("content", ""))
    elif event_type == "retrieval_trace":
        yield StreamEvent.retrieval_trace(event.get("data", {}))
    elif event_type == "multi_agent_plan":
        yield StreamEvent.multi_agent_plan(event.get("data", {}))
    elif event_type == "collaborator_trace":
        yield StreamEvent.collaborator_trace(event.get("data", {}))
    elif event_type == "citation_trace":
        yield StreamEvent.citation_trace(event.get("data", {}))
    elif event_type == "answer_guard":
        yield StreamEvent.answer_guard(event.get("data", {}))
    elif event_type == "plan_assessment":
        yield StreamEvent.plan_assessment(event.get("data", {}))
    elif event_type == "task_recovery":
        yield StreamEvent.task_recovery(event.get("data", {}))
    elif event_type == "task_snapshot":
        yield StreamEvent(event="task_snapshot", data=event.get("data", {}) or {})
    elif event_type == "tool_call":
        yield StreamEvent.tool_call(
            name=event.get("name", ""),
            args=event.get("args", {}),
            tool_call_id=event.get("tool_call_id"),
        )
    elif event_type == "tool_result":
        metadata = event.get("metadata", {}) or {}
        if metadata.get("error_type") == "tool_blocked":
            yield StreamEvent.tool_policy_blocked(
                {
                    "tool_name": event.get("name", ""),
                    "category": metadata.get("category"),
                    "allowed_categories": metadata.get("allowed_categories"),
                    "blocked_reason": metadata.get("blocked_reason"),
                    "how_to_enable_for_local_demo": metadata.get(
                        "how_to_enable_for_local_demo"
                    ),
                }
            )
        elif metadata.get("error_type") == "approval_required":
            pending_action = metadata.get("pending_action") or {}
            yield StreamEvent.pending_action(
                {
                    "tool_name": event.get("name", ""),
                    "category": metadata.get("category"),
                    "pending_action": pending_action,
                    "approval_store_available": metadata.get(
                        "approval_store_available",
                    ),
                    "reason": metadata.get("diagnosis"),
                }
            )
        yield StreamEvent.tool_result(
            name=event.get("name", ""),
            content=str(event.get("content", "")),
            tool_call_id=event.get("tool_call_id"),
            success=event.get("success"),
            error=event.get("error"),
            metadata=metadata,
        )
    elif event_type == "self_correction":
        yield StreamEvent.self_correction(
            message=event.get("content", ""),
            data={
                "tool_call_id": event.get("tool_call_id"),
                "name": event.get("name"),
                "error": event.get("error"),
                **(event.get("data", {}) or {}),
                "state": event.get("state", {}),
            },
        )
    elif event_type == "failure_recovery":
        yield StreamEvent.failure_recovery(
            message=event.get("content", ""),
            data={
                "tool_call_id": event.get("tool_call_id"),
                "name": event.get("name"),
                "error": event.get("error"),
                **(event.get("data", {}) or {}),
                "state": event.get("state", {}),
            },
        )
    elif event_type == "final_answer":
        yield StreamEvent(
            event="final_answer",
            content=event.get("content", ""),
            data=event.get("state", {}) or {},
        )
    elif event_type == "error":
        yield StreamEvent(
            event="error",
            message=event.get("content", ""),
            data=event.get("state", {}) or {},
        )
    elif event_type == "finish":
        yield StreamEvent(
            event="finish",
            data={
                "new_messages": event.get("new_messages", []),
                "status": event.get("status"),
                "termination_reason": event.get("termination_reason"),
                "state": event.get("state", {}),
            },
        )
    elif event_type:
        yield StreamEvent(
            event=str(event_type),
            content=event.get("content"),
            message=event.get("message"),
            name=event.get("name"),
            args=event.get("args"),
            data=event.get("data", {}) or {},
        )


def stream_event_to_dict(event: StreamEvent) -> Dict[str, Any]:
    payload = event.to_payload()
    data = payload.get("data", {}) or {}
    return {
        "type": payload.get("event"),
        "content": payload.get("content") or payload.get("message") or "",
        "name": payload.get("name"),
        "args": payload.get("args"),
        "data": data,
        "tool_call_id": data.get("tool_call_id"),
        "success": data.get("success"),
        "error": data.get("error"),
        "metadata": data.get("metadata", {}),
    }

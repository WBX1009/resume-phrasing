import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, Iterator, List

from backend.domain.entities.agent_state import AgentState
from backend.domain.entities.stream_event import StreamEvent
from backend.domain.entities.tooling import ToolCall, ToolResult
from backend.domain.interfaces.agent_runtime import AgentRuntimePort, AgentRuntimeRequest
from backend.domain.services.agent.engine import AgentEngine
from backend.infrastructure.agent_runtime.custom_runtime import (
    CustomAgentRuntimeAdapter,
)
from backend.common.config import settings


class LangGraphAgentRuntimeAdapter(AgentRuntimePort):
    """LangGraph state runtime that keeps DeepAssist tool governance semantics."""

    runtime_name = "langgraph"

    def __init__(self, engine: AgentEngine):
        self.engine = engine
        self.custom_runtime = CustomAgentRuntimeAdapter(engine)
        self._compiled_graph = None
        self._checkpointer = None
        self._checkpoint_connection = None
        self.native_available, self.fallback_reason = self._probe_native_runtime()

    def stream_run(self, request: AgentRuntimeRequest) -> Iterator[StreamEvent]:
        initial_state = AgentState(
            session_id=request.session_id,
            query=request.query,
            messages=request.messages,
            status="running",
        ).model_dump()
        self._set_approval_context(request)

        if not self.native_available:
            yield from self._stream_without_langgraph(
                request,
                initial_state,
                reason=self.fallback_reason,
            )
            return

        graph = self._compile_graph()
        if graph is None:
            yield from self._stream_without_langgraph(
                request,
                initial_state,
                reason=self.fallback_reason or "langgraph_compile_failed",
            )
            return

        state = {
            **initial_state,
            "initial_message_count": len(request.messages),
            "model_options": request.model_options,
            "resume_state": request.resume_state or {},
            "runtime_events": [],
            "fallback_without_langgraph": False,
        }
        # The LangGraph checkpoint and the application task snapshot are both
        # durable by default. The latter rehydrates an interrupted HTTP
        # request; the former keeps graph state for the stable session thread.
        thread_id = request.session_id
        if request.resume_state:
            thread_id = f"{request.session_id}:resume:{id(request)}"
        config = {"configurable": {"thread_id": thread_id}}
        stream_input: Any = state
        while True:
            interrupted = False
            for update in graph.stream(stream_input, config=config):
                if "__interrupt__" in update:
                    checkpoint = graph.get_state(config)
                    checkpoint_state = dict(checkpoint.values or {})
                    if self._checkpoint_requires_approval(checkpoint_state):
                        yield from self._stream_checkpointed_approval(
                            checkpoint_state,
                            checkpoint_id=self._checkpoint_id(checkpoint),
                        )
                        return
                    # ``interrupt_before`` is deliberately placed on the
                    # approval gate so high-risk calls can be surfaced before
                    # execution. Read-only calls pass through this checkpoint
                    # automatically and continue through the normal graph.
                    yield self._node_event(
                        "approval_gate",
                        checkpoint_state,
                        status="checkpointed",
                    )
                    stream_input = self._resume_command()
                    interrupted = True
                    break

                for node_name, node_state in update.items():
                    if not isinstance(node_state, dict):
                        continue
                    yield self._node_event(node_name, node_state)
                    for event_payload in node_state.get("runtime_events", []):
                        yield StreamEvent.model_validate(event_payload)
            if not interrupted:
                break

    def _probe_native_runtime(self) -> tuple[bool, str | None]:
        try:
            import langgraph  # noqa: F401
            from langgraph.checkpoint.memory import InMemorySaver  # noqa: F401
        except Exception as exc:
            return False, f"langgraph_unavailable:{type(exc).__name__}: {exc}"

        if not getattr(self.engine, "llm", None):
            return False, "engine_llm_missing"
        if not getattr(self.engine, "tool_registry", None):
            return False, "engine_tool_registry_missing"
        return True, None

    def _compile_graph(self):
        if self._compiled_graph is not None:
            return self._compiled_graph
        try:
            from langgraph.graph import END, START, StateGraph
            from langgraph.checkpoint.memory import InMemorySaver
        except Exception as exc:
            self.native_available = False
            self.fallback_reason = f"langgraph_unavailable:{type(exc).__name__}: {exc}"
            return None

        graph = StateGraph(dict)
        graph.add_node("plan", self._plan_node)
        graph.add_node("approval_gate", self._approval_gate_node)
        graph.add_node("tool_execute", self._tool_execute_node)
        graph.add_node("final_answer", self._final_answer_node)
        graph.add_edge(START, "plan")
        graph.add_conditional_edges(
            "plan",
            self._after_plan,
            {
                "approval_gate": "approval_gate",
                "final_answer": "final_answer",
            },
        )
        graph.add_conditional_edges(
            "approval_gate",
            self._after_gate,
            {
                "tool_execute": "tool_execute",
                "approval_gate": "approval_gate",
                "plan": "plan",
                "final_answer": "final_answer",
            },
        )
        graph.add_conditional_edges(
            "tool_execute",
            self._after_tool_execute,
            {
                "approval_gate": "approval_gate",
                "plan": "plan",
                "final_answer": "final_answer",
            },
        )
        graph.add_edge("final_answer", END)
        backend = (settings.LANGGRAPH_CHECKPOINT_BACKEND or "sqlite").strip().lower()
        if backend == "sqlite":
            try:
                from langgraph.checkpoint.sqlite import SqliteSaver

                checkpoint_path = Path(settings.LANGGRAPH_CHECKPOINT_DB_PATH)
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                self._checkpoint_connection = sqlite3.connect(
                    str(checkpoint_path),
                    check_same_thread=False,
                )
                self._checkpointer = SqliteSaver(self._checkpoint_connection)
            except Exception as exc:
                # Keep local development usable when the optional checkpoint
                # package is absent, but expose the downgrade in diagnostics.
                self.fallback_reason = f"sqlite_checkpoint_unavailable:{type(exc).__name__}: {exc}"
                self._checkpointer = InMemorySaver()
        elif backend == "memory":
            self._checkpointer = InMemorySaver()
        else:
            self.native_available = False
            self.fallback_reason = f"unsupported_checkpoint_backend:{backend}"
            return None
        self._compiled_graph = graph.compile(
            checkpointer=self._checkpointer,
            interrupt_before=["approval_gate"],
        )
        return self._compiled_graph

    def _resume_command(self):
        """Return the version-safe LangGraph command for an interrupt resume."""
        from langgraph.types import Command

        # ``Command(resume=None)`` is rejected by some LangGraph releases;
        # an empty resume payload means "continue the interrupted node".
        return Command(resume={})

    def _checkpoint_id(self, checkpoint: Any) -> str | None:
        config = getattr(checkpoint, "config", {}) or {}
        configurable = config.get("configurable", {}) if isinstance(config, dict) else {}
        value = configurable.get("checkpoint_id")
        return str(value) if value else None

    def _checkpoint_requires_approval(self, state: Dict[str, Any]) -> bool:
        tool_calls = list(state.get("tool_calls") or [])
        index = int(state.get("current_tool_index") or 0)
        if index >= len(tool_calls):
            return False
        try:
            tool_call = ToolCall.model_validate(tool_calls[index])
        except Exception:
            return False
        if tool_call.validation_error:
            return False
        policy = getattr(self.engine.tool_registry, "policy", None)
        return bool(policy and policy.requires_approval(tool_call.name))

    def _stream_checkpointed_approval(
        self,
        state: Dict[str, Any],
        checkpoint_id: str | None,
    ) -> Iterator[StreamEvent]:
        tool_calls = list(state.get("tool_calls") or [])
        index = int(state.get("current_tool_index") or 0)
        tool_call = ToolCall.model_validate(tool_calls[index])
        runtime_events: List[StreamEvent] = [
            StreamEvent.tool_call(
                name=tool_call.name,
                args=tool_call.args,
                tool_call_id=tool_call.id,
            )
        ]
        result = self.engine.tool_registry.execute(tool_call)
        messages = list(state.get("messages") or [])
        messages.append(result.to_tool_message())
        runtime_events.extend(self._events_for_tool_result(result))
        runtime_events.append(
            StreamEvent.status(
                "Waiting for human approval before executing the high-risk tool."
            )
        )
        checkpoint_state = {**state, "checkpoint_id": checkpoint_id}
        for payload in (
            self._dump_events(runtime_events)
            + self._task_snapshot_events(messages, checkpoint_state, "pending_approval")
            + self._finish_events(
                messages,
                checkpoint_state,
                "pending_approval",
                "human_approval_required",
            )
        ):
            yield StreamEvent.model_validate(payload)

    def _plan_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        state = {**state, "runtime_events": []}
        if int(state.get("step_count") or 0) >= self.engine.run_config.max_iterations:
            return self._finish_failed(
                state,
                "max_steps_exceeded",
                f"Reached max agent iterations ({self.engine.run_config.max_iterations}).",
            )

        messages = list(state.get("messages") or [])
        model_options = self._model_options(state.get("model_options") or {})
        model_started_at = time.perf_counter()
        try:
            response_msg = self.engine.llm.chat(
                messages=messages,
                tools=self.engine.tool_registry.openai_schemas(),
                **model_options,
            )
        except Exception as exc:
            failure = self._finish_failed(state, "llm_error", str(exc))
            failure["runtime_events"] = self._dump_events(
                [
                    StreamEvent.model_call(
                        self._model_call_data(
                            state=state,
                            model_options=model_options,
                            started_at=model_started_at,
                            phase="error",
                            error=exc,
                        )
                    )
                ]
            ) + list(failure.get("runtime_events") or [])
            return failure

        safe_content = getattr(response_msg, "content", None) or ""
        tool_calls = [
            ToolCall.from_llm_tool_call(tool_call)
            for tool_call in (getattr(response_msg, "tool_calls", None) or [])
        ]
        messages.append(self._assistant_message(safe_content, tool_calls))
        runtime_events: List[StreamEvent] = [
            StreamEvent.model_call(
                self._model_call_data(
                    state=state,
                    model_options=model_options,
                    started_at=model_started_at,
                    phase="completed",
                    response_tool_count=len(tool_calls),
                )
            )
        ]
        reasoning = getattr(response_msg, "reasoning_content", None)
        if reasoning:
            runtime_events.append(StreamEvent.reasoning(str(reasoning)))

        if not tool_calls:
            runtime_events.append(StreamEvent.final_answer(safe_content))
            return {
                **state,
                "messages": messages,
                "runtime_events": self._dump_events(runtime_events)
                + self._finish_events(messages, state, "completed", "final_answer"),
                "status": "completed",
                "final_answer": safe_content,
                "step_count": int(state.get("step_count") or 0) + 1,
                "tool_calls": [],
            }

        if safe_content.strip():
            runtime_events.append(StreamEvent.status(safe_content))
        runtime_events.append(
            StreamEvent.plan_assessment(
                {
                    "tool_call_count": len(tool_calls),
                    "tool_names": [tool_call.name for tool_call in tool_calls],
                    "recommended_mode": "execute",
                    "warnings": [],
                    "summary": "Tool plan looks executable.",
                }
            )
        )
        return {
            **state,
            "messages": messages,
            "runtime_events": self._dump_events(runtime_events),
            "status": "running",
            "step_count": int(state.get("step_count") or 0) + 1,
            "tool_calls": [tool_call.model_dump() for tool_call in tool_calls],
            "current_tool_index": 0,
        }

    def _model_call_data(
        self,
        *,
        state: Dict[str, Any],
        model_options: Dict[str, Any],
        started_at: float,
        phase: str,
        response_tool_count: int = 0,
        error: Exception | None = None,
    ) -> Dict[str, Any]:
        model_name = (
            model_options.get("model_name")
            or getattr(self.engine.llm, "model_name", None)
            or type(self.engine.llm).__name__
        )
        data: Dict[str, Any] = {
            "phase": phase,
            "model_name": str(model_name),
            "provider": type(self.engine.llm).__name__,
            "message_count": len(state.get("messages") or []),
            "tool_count": len(self.engine.tool_registry.openai_schemas()),
            "response_tool_count": response_tool_count,
            "latency_ms": round((time.perf_counter() - started_at) * 1000, 3),
            "success": phase == "completed",
        }
        if error is not None:
            data.update({"error_type": type(error).__name__, "error": str(error)})
        return data

    def _approval_gate_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        state = {**state, "runtime_events": []}
        tool_calls = list(state.get("tool_calls") or [])
        index = int(state.get("current_tool_index") or 0)
        if index >= len(tool_calls):
            return {**state, "status": "running"}

        tool_call = ToolCall.model_validate(tool_calls[index])
        runtime_events = [
            StreamEvent.tool_call(
                name=tool_call.name,
                args=tool_call.args,
                tool_call_id=tool_call.id,
            )
        ]

        if tool_call.validation_error:
            return self._append_tool_result_and_continue(
                state,
                ToolResult.invalid_arguments(tool_call, tool_call.validation_error),
                runtime_events,
            )

        policy = self.engine.tool_registry.policy
        if not policy.can_execute(tool_call.name):
            return self._append_tool_result_and_continue(
                state,
                self.engine.tool_registry.execute(tool_call),
                runtime_events,
            )

        if policy.requires_approval(tool_call.name):
            result = self.engine.tool_registry.execute(tool_call)
            messages = list(state.get("messages") or [])
            messages.append(result.to_tool_message())
            runtime_events.extend(self._events_for_tool_result(result))
            runtime_events.append(
                StreamEvent.status(
                    "Waiting for human approval before executing the high-risk tool."
                )
            )
            return {
                **state,
                "messages": messages,
                "runtime_events": self._dump_events(runtime_events)
                + self._task_snapshot_events(messages, state, "pending_approval")
                + self._finish_events(
                    messages,
                    state,
                    "pending_approval",
                    "human_approval_required",
                ),
                "status": "pending_approval",
                "pending_action": result.metadata.get("pending_action"),
                "step_count": int(state.get("step_count") or 0) + 1,
            }

        return {
            **state,
            "runtime_events": self._dump_events(runtime_events),
            "current_tool_call": tool_call.model_dump(),
            "status": "running",
            "step_count": int(state.get("step_count") or 0) + 1,
        }

    def _tool_execute_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        state = {**state, "runtime_events": []}
        tool_call_payload = state.get("current_tool_call")
        if not tool_call_payload:
            return {**state, "status": "running"}
        tool_call = ToolCall.model_validate(tool_call_payload)
        result = self.engine.tool_registry.execute(tool_call)
        return self._append_tool_result_and_continue(state, result, [])

    def _final_answer_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        status = state.get("status") or "completed"
        if status == "running":
            status = "completed"
        return {
            **state,
            "status": status,
            "runtime_events": [],
            "step_count": int(state.get("step_count") or 0) + 1,
        }

    def _append_tool_result_and_continue(
        self,
        state: Dict[str, Any],
        result: ToolResult,
        runtime_events: List[StreamEvent],
    ) -> Dict[str, Any]:
        messages = list(state.get("messages") or [])
        messages.append(result.to_tool_message())
        tool_calls = list(state.get("tool_calls") or [])
        next_index = int(state.get("current_tool_index") or 0) + 1
        runtime_events.extend(self._events_for_tool_result(result))
        return {
            **state,
            "messages": messages,
            "runtime_events": self._dump_events(runtime_events)
            + self._task_snapshot_events(messages, state, "running"),
            "current_tool_index": next_index,
            "current_tool_call": None,
            "status": "running",
            "step_count": int(state.get("step_count") or 0) + 1,
            "error_count": int(state.get("error_count") or 0)
            + (0 if result.success else 1),
            "observations": [
                *list(state.get("observations") or []),
                result.model_dump(),
            ],
            "tool_calls": tool_calls,
        }

    def _after_plan(self, state: Dict[str, Any]) -> str:
        if state.get("status") != "running":
            return "final_answer"
        return "approval_gate" if state.get("tool_calls") else "final_answer"

    def _after_gate(self, state: Dict[str, Any]) -> str:
        if state.get("status") != "running":
            return "final_answer"
        if state.get("current_tool_call"):
            return "tool_execute"
        tool_calls = list(state.get("tool_calls") or [])
        index = int(state.get("current_tool_index") or 0)
        return "approval_gate" if index < len(tool_calls) else "plan"

    def _after_tool_execute(self, state: Dict[str, Any]) -> str:
        if state.get("status") != "running":
            return "final_answer"
        tool_calls = list(state.get("tool_calls") or [])
        index = int(state.get("current_tool_index") or 0)
        return "approval_gate" if index < len(tool_calls) else "plan"

    def _finish_failed(
        self,
        state: Dict[str, Any],
        reason: str,
        message: str,
    ) -> Dict[str, Any]:
        messages = list(state.get("messages") or [])
        runtime_events = [
            StreamEvent(
                event="error",
                message=message,
                data={"status": "failed", "termination_reason": reason},
            )
        ]
        return {
            **state,
            "messages": messages,
            "status": "failed",
            "runtime_events": self._dump_events(runtime_events)
            + self._finish_events(messages, state, "failed", reason),
            "error_count": int(state.get("error_count") or 0) + 1,
        }

    def _events_for_tool_result(self, result: ToolResult) -> List[StreamEvent]:
        events: List[StreamEvent] = []
        if result.metadata.get("error_type") == "approval_required":
            events.append(
                StreamEvent.pending_action(
                    {
                        "tool_name": result.name,
                        "category": result.metadata.get("category"),
                        "pending_action": result.metadata.get("pending_action") or {},
                        "approval_store_available": result.metadata.get(
                            "approval_store_available"
                        ),
                        "reason": result.metadata.get("diagnosis"),
                    }
                )
            )
        elif result.metadata.get("error_type") == "tool_blocked":
            events.append(
                StreamEvent.tool_policy_blocked(
                    {
                        "tool_name": result.name,
                        "category": result.metadata.get("category"),
                        "allowed_categories": result.metadata.get("allowed_categories"),
                        "blocked_reason": result.metadata.get("blocked_reason"),
                        "how_to_enable_for_local_demo": result.metadata.get(
                            "how_to_enable_for_local_demo"
                        ),
                    }
                )
            )
        events.append(result.to_stream_event())
        return events

    def _task_snapshot_events(
        self,
        messages: List[Dict[str, Any]],
        state: Dict[str, Any],
        status: str,
    ) -> List[Dict[str, Any]]:
        return self._dump_events(
            [
                StreamEvent(
                    event="task_snapshot",
                    data={
                        "route_worker": "langgraph_agent_runtime",
                        "status": status,
                        "checkpoint_id": state.get("checkpoint_id"),
                        "messages": messages,
                        "lifecycle_state": {
                            "status": status,
                            "iterations": int(state.get("step_count") or 0),
                            "error": None
                            if status in {"running", "completed"}
                            else "Human approval is required.",
                            "termination_reason": "human_approval_required"
                            if status == "pending_approval"
                            else None,
                            "successful_tool_calls": len(
                                [
                                    item
                                    for item in state.get("observations", [])
                                    if item.get("success")
                                ]
                            ),
                            "tool_errors": int(state.get("error_count") or 0),
                        },
                    },
                )
            ]
        )

    def _finish_events(
        self,
        messages: List[Dict[str, Any]],
        state: Dict[str, Any],
        status: str,
        reason: str | None,
    ) -> List[Dict[str, Any]]:
        initial_count = int(state.get("initial_message_count") or 0)
        return self._dump_events(
            [
                StreamEvent(
                    event="finish",
                    data={
                        "new_messages": messages[initial_count:],
                        "status": status,
                        "termination_reason": reason,
                        "state": {
                            "status": status,
                            "termination_reason": reason,
                            "iterations": int(state.get("step_count") or 0),
                            "error": None if status == "completed" else reason,
                        },
                    },
                )
            ]
        )

    def _stream_without_langgraph(
        self,
        request: AgentRuntimeRequest,
        initial_state: Dict[str, Any],
        reason: str | None = None,
    ) -> Iterator[StreamEvent]:
        route_state = self._route_node(initial_state)
        yield self._node_event("plan", route_state, fallback=True, reason=reason)
        final_status = "completed"
        for event in self.custom_runtime.stream_run(request):
            if event.event == "error":
                final_status = "failed"
            elif event.event == "finish":
                finish_status = str((event.data or {}).get("status") or "").strip()
                if finish_status:
                    final_status = finish_status
            elif event.event == "agent_runtime_node":
                node_status = str((event.data or {}).get("status") or "").strip()
                if node_status in {"failed", "error"}:
                    final_status = "failed"
            yield event
        final_state = self._final_answer_node({**route_state, "status": final_status})
        yield self._node_event(
            "final_answer",
            final_state,
            fallback=True,
            reason=reason,
        )

    def _route_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        return {
            **state,
            "route": "tool_agent",
            "step_count": int(state.get("step_count") or 0) + 1,
        }

    def _node_event(
        self,
        node_name: str,
        state: Dict[str, Any],
        status: str | None = None,
        fallback: bool | None = None,
        reason: str | None = None,
    ) -> StreamEvent:
        is_fallback = bool(
            fallback
            if fallback is not None
            else state.get("fallback_without_langgraph", False)
        )
        data = {
            "runtime": self.runtime_name,
            "node": node_name,
            "status": status or state.get("status") or "running",
            "route": state.get("route"),
            "step_count": state.get("step_count", 0),
            "fallback_without_langgraph": is_fallback,
        }
        fallback_reason = reason or self.fallback_reason
        if is_fallback and fallback_reason:
            data["fallback_reason"] = fallback_reason
        return StreamEvent(event="agent_runtime_node", name=node_name, data=data)

    def _assistant_message(
        self,
        safe_content: str,
        tool_calls: List[ToolCall],
    ) -> Dict[str, Any]:
        msg_dict: Dict[str, Any] = {"role": "assistant", "content": safe_content}
        if tool_calls:
            msg_dict["tool_calls"] = [
                tool_call.to_openai_tool_call() for tool_call in tool_calls
            ]
        return msg_dict

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

    def _model_options(self, model_options: Dict[str, Any]) -> Dict[str, Any]:
        return {
            key: value
            for key, value in model_options.items()
            if value is not None
        }

    def _dump_events(self, events: List[StreamEvent]) -> List[Dict[str, Any]]:
        return [event.model_dump(exclude_none=True) for event in events]

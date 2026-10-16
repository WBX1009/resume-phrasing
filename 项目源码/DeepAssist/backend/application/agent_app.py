import asyncio
import re
from typing import Any, Dict, Iterator, Optional

from backend.common.logger import get_logger
from backend.domain.interfaces.agent_runtime import AgentRuntimePort, AgentRuntimeRequest
from backend.domain.entities.stream_event import StreamEvent
from backend.domain.entities.task_snapshot import TaskSnapshot
from backend.domain.entities.tooling import ToolResult
from backend.domain.services.agent.engine import AgentEngine
from backend.domain.services.agent.supervisor import AgentSupervisor
from backend.domain.services.session.context_engine import ContextEngine
from backend.domain.services.session.profile_extractor import ProfileExtractor
from backend.domain.services.session.manager import SessionManager
from backend.infrastructure.streaming.sse_manager import SSEManager

logger = get_logger(__name__)

RECOVERY_QUERY_PATTERN = re.compile(
    r"^(继续|接着|恢复|继续刚才|接着刚才|resume|continue)",
    flags=re.IGNORECASE,
)

class AgentApplication:
    """Agent workflow orchestrator."""

    def __init__(
        self,
        agent_engine: AgentEngine,
        session_manager: SessionManager,
        context_engine: ContextEngine,
        profile_extractor: ProfileExtractor,
        supervisor: AgentSupervisor | None = None,
        agent_runtime: AgentRuntimePort | None = None,
    ):
        self.engine = agent_engine
        self.session_mgr = session_manager
        self.context_engine = context_engine
        self.profile_extractor = profile_extractor
        self.supervisor = supervisor
        self.agent_runtime = agent_runtime

    def stream_agent_task(
        self,
        session_id: str,
        query: str,
        use_user_memory: bool = False,
        model_name: Optional[str] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        history_rounds: int = 10,
        original_query: Optional[str] = None,
    ) -> Iterator[str]:
        final_answer_accumulated = ""
        persist_messages: list[dict] | None = None
        should_persist_partial = False
        history_budget = max(1, history_rounds)
        latest_snapshot: TaskSnapshot | None = None
        persisted_query = original_query if original_query is not None else query
        try:
            logger.info("Entering agent mode for session %s", session_id)
            context_plan = self.session_mgr.plan_chat_context(
                session_id,
                max_rounds=history_budget,
                query=query,
                use_long_term_memory=use_user_memory,
            )
            history = context_plan.flattened_messages()
            yield SSEManager.format_event(
                StreamEvent.context_window_trace(context_plan.to_trace_data())
            )
            if context_plan.recalled_memories:
                yield SSEManager.format_event(
                    StreamEvent.status(
                        f"Recalled {len(context_plan.recalled_memories)} long-term memory item(s)."
                    )
                )
            if context_plan.summary is not None:
                yield SSEManager.format_event(
                    StreamEvent.status(
                        f"Compressed {context_plan.summary.dropped_turn_count} earlier turns into a continuity summary."
                    )
                )
            user_profile = (
                self.profile_extractor.render_profile() if use_user_memory else None
            )
            model_options: Dict[str, Any] = {
                "model_name": model_name,
                "temperature": temperature,
                "top_p": top_p,
            }
            recovery_snapshot = self.session_mgr.get_task_snapshot(session_id)
            should_resume = self._should_resume_task(query, recovery_snapshot)
            if should_resume and recovery_snapshot is not None:
                latest_snapshot = recovery_snapshot
                persisted_query = recovery_snapshot.query
                yield SSEManager.format_event(
                    StreamEvent.task_recovery(recovery_snapshot.to_trace_data())
                )

            if self.agent_runtime is not None:
                runtime_query = recovery_snapshot.query if should_resume and recovery_snapshot else query
                context = self.context_engine.build_agent_context(
                    query=runtime_query,
                    history=history,
                    use_user_memory=use_user_memory,
                    user_profile=user_profile,
                    available_tool_names=self._available_tool_names(),
                )
                runtime_messages = (
                    list(recovery_snapshot.payload.get("messages", []))
                    if should_resume and recovery_snapshot
                    else self._with_tool_context(context.messages)
                )
                runtime_request = AgentRuntimeRequest(
                    session_id=session_id,
                    query=runtime_query,
                    messages=runtime_messages,
                    model_options=model_options,
                    resume_state=recovery_snapshot.payload
                    if should_resume and recovery_snapshot
                    else None,
                )
                for runtime_event in self.agent_runtime.stream_run(runtime_request):
                    if runtime_event.event == "task_snapshot":
                        latest_snapshot = self._build_task_snapshot(
                            session_id=session_id,
                            query=runtime_query,
                            event_data=runtime_event.data or {},
                        )
                        self.session_mgr.save_task_snapshot(latest_snapshot)
                        # Keep the persisted state visible on the SSE stream so
                        # the unified conversation tracer can correlate state
                        # changes with model and tool events.
                        yield SSEManager.format_event(runtime_event)
                        continue

                    if runtime_event.event == "message_delta":
                        delta = runtime_event.content or ""
                        final_answer_accumulated += delta
                        should_persist_partial = True
                        yield SSEManager.format_event(runtime_event)
                        continue

                    if runtime_event.event == "final_answer":
                        final_answer_accumulated = runtime_event.content or ""
                        should_persist_partial = True
                        yield SSEManager.format_event(runtime_event)
                        continue

                    if runtime_event.event == "finish":
                        new_messages = runtime_event.data.get("new_messages", [])
                        if runtime_event.data.get("status") == "pending_approval":
                            persist_messages = None
                        else:
                            persist_messages = [{"role": "user", "content": persisted_query}] + new_messages
                            self.session_mgr.clear_task_snapshot(session_id)
                        yield SSEManager.format_end()
                        return

                    yield SSEManager.format_event(runtime_event)

                yield SSEManager.format_end()
                return

            if should_resume and recovery_snapshot is not None:
                event_stream = self.supervisor.stream_recovery(
                    worker_kind=recovery_snapshot.route_worker,
                    query=recovery_snapshot.query,
                    history=history,
                    user_profile=user_profile,
                    model_options=model_options,
                    recovery_state=recovery_snapshot.payload,
                )
            else:
                event_stream = (
                    self.supervisor.stream(
                        query,
                        history,
                        user_profile=user_profile,
                        model_options=model_options,
                    )
                    if self.supervisor
                    else self._stream_legacy_agent(
                        query,
                        history,
                        use_user_memory,
                        user_profile,
                        model_options,
                    )
                )

            for event in event_stream:
                event_type = event.get("type")

                if event_type == "supervisor_route":
                    yield SSEManager.format_event(
                        StreamEvent(
                            event="supervisor_route",
                            data={
                                "worker": event.get("worker"),
                                "worker_kind": event.get("worker_kind"),
                                "intent": event.get("intent"),
                                "confidence": event.get("confidence"),
                                "reason": event.get("reason"),
                                "signals": event.get("signals",[]),
                            },
                        )
                    )

                elif event_type == "status":
                    yield SSEManager.format_event(StreamEvent.status(event.get("content", "")))

                elif event_type == "reasoning":
                    yield SSEManager.format_event(StreamEvent.reasoning(event.get("content", "")))

                elif event_type == "model_call":
                    yield SSEManager.format_event(
                        StreamEvent.model_call(event.get("data", {}) or {})
                    )

                elif event_type == "task_snapshot":
                    latest_snapshot = self._build_task_snapshot(
                        session_id=session_id,
                        query=recovery_snapshot.query if should_resume and recovery_snapshot else query,
                        event_data=event.get("data", {}) or {},
                    )
                    self.session_mgr.save_task_snapshot(latest_snapshot)

                elif event_type == "message_delta":
                    delta = event.get("content", "")
                    final_answer_accumulated += delta
                    should_persist_partial = True
                    yield SSEManager.format_event(StreamEvent.message_delta(delta))

                elif event_type == "retrieval_trace":
                    yield SSEManager.format_event(
                        StreamEvent.retrieval_trace(event.get("data", {}))
                    )

                elif event_type == "multi_agent_plan":
                    yield SSEManager.format_event(
                        StreamEvent.multi_agent_plan(event.get("data", {}))
                    )

                elif event_type == "collaborator_trace":
                    yield SSEManager.format_event(
                        StreamEvent.collaborator_trace(event.get("data", {}))
                    )

                elif event_type == "citation_trace":
                    yield SSEManager.format_event(
                        StreamEvent.citation_trace(event.get("data", {}))
                    )

                elif event_type == "answer_guard":
                    yield SSEManager.format_event(
                        StreamEvent.answer_guard(event.get("data", {}))
                    )

                elif event_type == "self_correction":
                    data = event.get("data", {}) or {}
                    yield SSEManager.format_event(
                        StreamEvent.self_correction(
                            message=event.get("content", ""),
                            data={
                                "tool_call_id": event.get("tool_call_id"),
                                "name": event.get("name"),
                                "error": event.get("error"),
                                **data,
                                "state": event.get("state", {}),
                            },
                        )
                    )

                elif event_type == "failure_recovery":
                    yield SSEManager.format_event(
                        StreamEvent.failure_recovery(
                            message=event.get("content", ""),
                            data={
                                "tool_call_id": event.get("tool_call_id"),
                                "name": event.get("name"),
                                "error": event.get("error"),
                                **(event.get("data", {}) or {}),
                                "state": event.get("state", {}),
                            },
                        )
                    )

                elif event_type == "plan_assessment":
                    yield SSEManager.format_event(
                        StreamEvent.plan_assessment(event.get("data", {}))
                    )

                elif event_type == "task_recovery":
                    yield SSEManager.format_event(
                        StreamEvent.task_recovery(event.get("data", {}))
                    )

                elif event_type == "tool_call":
                    yield SSEManager.format_event(
                        StreamEvent.tool_call(
                            name=event.get("name", ""),
                            args=event.get("args", {}),
                            tool_call_id=event.get("tool_call_id"),
                        )
                    )

                elif event_type == "tool_result":
                    metadata = event.get("metadata", {}) or {}
                    if metadata.get("error_type") == "tool_blocked":
                        yield SSEManager.format_event(
                            StreamEvent.tool_policy_blocked(
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
                        )
                    yield SSEManager.format_event(
                        StreamEvent.tool_result(
                            name=event.get("name", ""),
                            content=str(event.get("content", "")),
                            tool_call_id=event.get("tool_call_id"),
                            success=event.get("success"),
                            error=event.get("error"),
                            metadata=event.get("metadata", {}),
                        )
                    )

                elif event_type == "final_answer":
                    final_answer_accumulated = event.get("content", "")
                    should_persist_partial = True
                    yield SSEManager.format_event(
                        StreamEvent(
                            event="final_answer",
                            content=final_answer_accumulated,
                            data=event.get("state", {}),
                        )
                    )

                elif event_type == "error":
                    yield SSEManager.format_event(
                        StreamEvent(
                            event="error",
                            message=event.get("content", ""),
                            data=event.get("state", {}),
                        )
                    )

                elif event_type == "finish":
                    new_messages = event.get("new_messages", [])
                    if event.get("status") == "pending_approval":
                        persist_messages = None
                    else:
                        persist_messages = [{"role": "user", "content": persisted_query}] + new_messages
                        self.session_mgr.clear_task_snapshot(session_id)
                    yield SSEManager.format_end()
                    return

        except (GeneratorExit, asyncio.CancelledError):
            logger.warning("Agent stream closed early for session %s", session_id)
            if latest_snapshot is not None:
                self.session_mgr.save_task_snapshot(
                    latest_snapshot.model_copy(update={"status": "interrupted"})
                )
            raise
        except Exception as exc:
            logger.error("Agent workflow failed: %s", exc)
            if latest_snapshot is not None:
                self.session_mgr.save_task_snapshot(
                    latest_snapshot.model_copy(update={"status": "failed"})
                )
            yield SSEManager.format_error(str(exc))
            yield SSEManager.format_end()
        finally:
            if persist_messages:
                self.session_mgr.add_messages(session_id, persist_messages)
                logger.info("Persisted %s agent trace messages", len(persist_messages))
            elif should_persist_partial and (query.strip() or final_answer_accumulated.strip()):
                self.session_mgr.save_interaction(
                    session_id,
                    persisted_query,
                    final_answer_accumulated,
                )

    def _emit_buffered_text(
        self,
        text: str,
        chunk_size: int = 160,
    ) -> Iterator[str]:
        for start in range(0, len(text), chunk_size):
            yield SSEManager.format_event(
                StreamEvent.message_delta(text[start : start + chunk_size])
            )

    def _stream_legacy_agent(
        self,
        query: str,
        history,
        use_user_memory: bool,
        user_profile: str | None,
        model_options: Optional[Dict[str, Any]] = None,
    ):
        context = self.context_engine.build_agent_context(
            query=query,
            history=history,
            use_user_memory=use_user_memory,
            user_profile=user_profile,
            available_tool_names=self._available_tool_names(),
        )
        yield from self.engine.stream_run(context.messages, model_options=model_options)

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
        approval_store = self.engine.tool_registry.policy.approval_store
        if approval_store is None:
            yield SSEManager.format_error("Approval store is not configured.")
            yield SSEManager.format_end()
            return

        action = approval_store.get(action_id)
        if action is None:
            yield SSEManager.format_error(f"Unknown pending action: {action_id}")
            yield SSEManager.format_end()
            return

        resolved_session_id = (
            session_id
            or str(action.metadata.get("session_id") or "").strip()
        )
        if not resolved_session_id:
            yield SSEManager.format_error(
                "Pending action does not include a session_id; provide session_id in the request."
            )
            yield SSEManager.format_end()
            return

        snapshot = self.session_mgr.get_task_snapshot(resolved_session_id)
        if snapshot is None:
            yield SSEManager.format_error(
                f"No resumable task snapshot found for session: {resolved_session_id}"
            )
            yield SSEManager.format_end()
            return

        try:
            if approved:
                approval_store.approve(action_id, decided_by=decided_by, reason=reason)
                tool_result = self.engine.tool_registry.execute_approved(action_id)
                action = approval_store.get(action_id) or action
                decision_status = "approved"
            else:
                action = approval_store.reject(
                    action_id,
                    decided_by=decided_by,
                    reason=reason,
                )
                tool_result = ToolResult.execution_failed(
                    action.tool_call,
                    "Human rejected this high-risk tool call.",
                    metadata={
                        "error_type": "human_rejected",
                        "retryable": False,
                        "repair_strategy": "fallback_answer",
                        "diagnosis": reason or "The user rejected the pending action.",
                    },
                )
                decision_status = "rejected"
        except Exception as exc:
            yield SSEManager.format_error(str(exc))
            yield SSEManager.format_end()
            return

        yield SSEManager.format_event(
            StreamEvent(
                event="approval_result",
                name=tool_result.name,
                data={
                    "action_id": action_id,
                    "status": decision_status,
                    "session_id": resolved_session_id,
                    "tool_result": tool_result.model_dump(),
                    "action": action.to_public_dict() if action else None,
                },
            )
        )
        yield SSEManager.format_event(tool_result.to_stream_event())

        messages = self._replace_pending_observation(
            list(snapshot.payload.get("messages", [])),
            tool_result,
        )
        resume_state = self._resume_state_after_approval(
            snapshot.payload.get("lifecycle_state", {}),
            tool_result,
        )
        model_options: Dict[str, Any] = {
            "model_name": model_name,
            "temperature": temperature,
            "top_p": top_p,
        }
        runtime_request = AgentRuntimeRequest(
            session_id=resolved_session_id,
            query=snapshot.query,
            messages=messages,
            model_options=model_options,
            resume_state=resume_state,
        )
        runtime = self.agent_runtime
        if runtime is None:
            from backend.infrastructure.agent_runtime.custom_runtime import (
                CustomAgentRuntimeAdapter,
            )

            runtime = CustomAgentRuntimeAdapter(self.engine)

        final_answer_accumulated = ""
        persist_messages: list[dict] | None = None
        for runtime_event in runtime.stream_run(runtime_request):
            if runtime_event.event == "task_snapshot":
                latest_snapshot = self._build_task_snapshot(
                    session_id=resolved_session_id,
                    query=snapshot.query,
                    event_data=runtime_event.data or {},
                )
                self.session_mgr.save_task_snapshot(latest_snapshot)
                continue

            if runtime_event.event == "message_delta":
                final_answer_accumulated += runtime_event.content or ""
                yield SSEManager.format_event(runtime_event)
                continue

            if runtime_event.event == "final_answer":
                final_answer_accumulated = runtime_event.content or final_answer_accumulated
                yield SSEManager.format_event(runtime_event)
                continue

            if runtime_event.event == "finish":
                new_messages = runtime_event.data.get("new_messages", [])
                if runtime_event.data.get("status") == "pending_approval":
                    persist_messages = None
                else:
                    persist_messages = [
                        {"role": "user", "content": snapshot.query},
                        *new_messages,
                    ]
                    self.session_mgr.clear_task_snapshot(resolved_session_id)
                yield SSEManager.format_end()
                break

            yield SSEManager.format_event(runtime_event)
        else:
            yield SSEManager.format_end()

        if persist_messages:
            self.session_mgr.add_messages(resolved_session_id, persist_messages)
        elif final_answer_accumulated.strip():
            self.session_mgr.save_interaction(
                resolved_session_id,
                snapshot.query,
                final_answer_accumulated,
            )

    def _should_resume_task(
        self,
        query: str,
        snapshot: TaskSnapshot | None,
    ) -> bool:
        if snapshot is None or snapshot.status not in {
            "running",
            "interrupted",
            "failed",
            "pending_approval",
        }:
            return False
        normalized_query = " ".join((query or "").split()).strip()
        if not normalized_query:
            return False
        if normalized_query == snapshot.query:
            return True
        return bool(RECOVERY_QUERY_PATTERN.search(normalized_query))

    def _build_task_snapshot(
        self,
        session_id: str,
        query: str,
        event_data: Dict[str, Any],
    ) -> TaskSnapshot:
        route_worker = str(event_data.get("route_worker") or "tool_agent_worker")
        status = str(event_data.get("status") or "running")
        payload = {key: value for key, value in event_data.items() if key not in {"route_worker", "status"}}
        return TaskSnapshot(
            session_id=session_id,
            query=query,
            route_worker=route_worker,
            status=status,
            payload=payload,
        )

    def _with_tool_context(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if self.engine is None or not getattr(self.engine, "tool_registry", None):
            return list(messages)
        tool_inventory = self.engine.tool_registry.describe_tools(include_disabled=False)
        policy = self.engine.tool_registry.policy
        approval_categories = ", ".join(sorted(policy.approval_required_categories))
        inventory_message = {
            "role": "system",
            "content": (
                "Current DeepAssist tool context:\n"
                "- Only call tools listed below; do not invent tool names.\n"
                "- Tools marked as high-risk may pause with a pending approval event.\n"
                "- If approval is required, wait for the approval result before finalizing.\n"
                f"- Approval mode: {policy.approval_mode}; high-risk categories: {approval_categories}.\n\n"
                f"{tool_inventory}"
            ),
        }
        result = list(messages)
        if result and result[0].get("role") == "system":
            result.insert(1, inventory_message)
        else:
            result.insert(0, inventory_message)
        return result

    def _available_tool_names(self) -> list[str]:
        registry = getattr(self.engine, "tool_registry", None)
        if registry is None:
            return []
        return [
            str(item.get("name"))
            for item in registry.list_tool_specs()
            if item.get("name") and item.get("enabled", True)
        ]

    def _replace_pending_observation(
        self,
        messages: list[dict[str, Any]],
        tool_result: ToolResult,
    ) -> list[dict[str, Any]]:
        replacement = tool_result.to_tool_message()
        for index in range(len(messages) - 1, -1, -1):
            message = messages[index]
            if (
                message.get("role") == "tool"
                and message.get("tool_call_id") == tool_result.tool_call_id
            ):
                messages[index] = replacement
                return messages
        messages.append(replacement)
        return messages

    def _resume_state_after_approval(
        self,
        lifecycle_state: Dict[str, Any],
        tool_result: ToolResult,
    ) -> Dict[str, Any]:
        resume_state = dict(lifecycle_state or {})
        resume_state["status"] = "running"
        resume_state["termination_reason"] = None
        resume_state["error"] = None
        if tool_result.success:
            resume_state["successful_tool_calls"] = int(
                resume_state.get("successful_tool_calls") or 0
            ) + 1
            resume_state["consecutive_tool_failures"] = 0
        else:
            resume_state["tool_errors"] = int(resume_state.get("tool_errors") or 0) + 1
            resume_state["consecutive_tool_failures"] = int(
                resume_state.get("consecutive_tool_failures") or 0
            ) + 1
            failed_names = dict(resume_state.get("failed_tool_names") or {})
            failed_names[tool_result.name] = int(failed_names.get(tool_result.name) or 0) + 1
            resume_state["failed_tool_names"] = failed_names
            resume_state["last_failed_tool_name"] = tool_result.name
        return resume_state

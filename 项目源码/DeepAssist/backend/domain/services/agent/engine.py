import json
import time
from typing import Any, Callable, Dict, Iterator, List, Optional

from backend.common.logger import get_logger
from backend.domain.entities.agent_run import (
    AgentRunConfig,
    AgentRunState,
    AgentTerminationReason,
)
from backend.domain.entities.tooling import ToolCall, ToolResult
from backend.domain.interfaces.llm import BaseLLM
from backend.domain.services.agent.lifecycle import AgentLifecycle
from backend.domain.services.agent.tooling import ToolRegistry

logger = get_logger(__name__)


class AgentEngine:
    """ReAct-style agent loop with canonical ToolCall/ToolResult flow."""

    def __init__(
        self,
        llm: BaseLLM,
        tools: Optional[List[Callable[..., Any]]] = None,
        tool_registry: Optional[ToolRegistry] = None,
        run_config: Optional[AgentRunConfig] = None,
    ):
        self.llm = llm
        self.tool_registry = tool_registry or ToolRegistry.from_callables(tools or [])
        self.openai_tools = self.tool_registry.openai_schemas()
        self.run_config = run_config or AgentRunConfig(
            max_iterations=10,
            max_tool_errors=3,
            repeated_tool_call_limit=3,
        )

    def stream_run(
        self,
        messages: List[Dict[str, Any]],
        model_options: Optional[Dict[str, Any]] = None,
        resume_state: Optional[Dict[str, Any]] = None,
    ) -> Iterator[Dict[str, Any]]:
        current_messages = messages.copy()
        restored_state = (
            AgentRunState.model_validate(resume_state or {})
            if resume_state
            else None
        )
        lifecycle = AgentLifecycle(
            self.run_config,
            state=restored_state,
        )
        model_kwargs = {
            key: value for key, value in (model_options or {}).items() if value is not None
        }

        while lifecycle.start_iteration():
            yield {
                "type": "status",
                "content": (
                    f"Starting reasoning step {lifecycle.state.iterations}/"
                    f"{self.run_config.max_iterations}"
                ),
                "state": lifecycle.state.terminal_payload(),
            }
            yield self._task_snapshot_event(current_messages, lifecycle)

            model_started_at = time.perf_counter()
            try:
                response_msg = self.llm.chat(
                    messages=current_messages,
                    tools=self.openai_tools,
                    **model_kwargs,
                )
            except Exception as exc:
                logger.error("LLM call failed during agent run: %s", exc)
                yield self._model_call_event(
                    phase="error",
                    started_at=model_started_at,
                    message_count=len(current_messages),
                    tool_count=len(self.openai_tools),
                    model_kwargs=model_kwargs,
                    error=exc,
                )
                lifecycle.fail(AgentTerminationReason.LLM_ERROR, str(exc))
                yield self._terminal_error_event(lifecycle)
                yield self._finish_event(messages, current_messages, lifecycle)
                return

            safe_content = response_msg.content or ""
            if getattr(response_msg, "reasoning_content", None):
                yield {"type": "reasoning", "content": response_msg.reasoning_content}

            tool_calls = [
                ToolCall.from_llm_tool_call(tool_call)
                for tool_call in (getattr(response_msg, "tool_calls", None) or [])
            ]
            yield self._model_call_event(
                phase="completed",
                started_at=model_started_at,
                message_count=len(current_messages),
                tool_count=len(self.openai_tools),
                model_kwargs=model_kwargs,
                response_tool_count=len(tool_calls),
            )
            current_messages.append(self._assistant_message(safe_content, tool_calls))

            if tool_calls:
                plan_assessment = self._assess_plan(tool_calls, lifecycle)
                yield {
                    "type": "plan_assessment",
                    "content": plan_assessment["summary"],
                    "data": plan_assessment,
                }

            if not tool_calls:
                lifecycle.complete(safe_content)
                yield {
                    "type": "final_answer",
                    "content": safe_content,
                    "state": lifecycle.state.terminal_payload(),
                }
                yield self._task_snapshot_event(current_messages, lifecycle, status="completed")
                yield self._finish_event(messages, current_messages, lifecycle)
                return

            if safe_content.strip():
                yield {"type": "status", "content": safe_content}

            if lifecycle_error := lifecycle.guard_tool_phase():
                if lifecycle.state.successful_tool_calls > 0:
                    final_answer = self._partial_answer_from_tool_history(
                        safe_content=safe_content,
                        current_messages=current_messages,
                        lifecycle_error=lifecycle_error,
                    )
                    if current_messages and current_messages[-1].get("tool_calls"):
                        current_messages.pop()
                    current_messages.append({"role": "assistant", "content": final_answer})
                    lifecycle.complete(final_answer)
                    yield {
                        "type": "failure_recovery",
                        "content": "Agent reached the step limit after collecting tool observations; returning a bounded partial answer.",
                        "data": {
                            "action": "finalize_partial_answer",
                            "reason": "max_steps_after_successful_tools",
                            "instruction": "No more tools will be called in this run.",
                        },
                        "state": lifecycle.state.terminal_payload(),
                    }
                    yield {
                        "type": "final_answer",
                        "content": final_answer,
                        "state": lifecycle.state.terminal_payload(),
                    }
                    yield self._task_snapshot_event(current_messages, lifecycle, status="completed")
                    yield self._finish_event(messages, current_messages, lifecycle)
                    return

                for pending_call in tool_calls:
                    result = ToolResult.lifecycle_error(
                        pending_call,
                        lifecycle_error,
                        metadata=lifecycle.state.terminal_payload(),
                    )
                    yield self._tool_result_event(result)
                    current_messages.append(result.to_tool_message())

                yield self._terminal_error_event(lifecycle)
                yield self._task_snapshot_event(current_messages, lifecycle, status="failed")
                yield self._finish_event(messages, current_messages, lifecycle)
                return

            iteration_seen_signatures: set[str] = set()
            for tool_call in tool_calls:
                yield {
                    "type": "tool_call",
                    "name": tool_call.name,
                    "args": tool_call.args,
                    "tool_call_id": tool_call.id,
                    "data": tool_call.to_stream_data(),
                }

                duplicate_plan_result = self._dedupe_iteration_tool_call(
                    tool_call,
                    iteration_seen_signatures,
                )
                if duplicate_plan_result is not None:
                    yield self._tool_result_event(duplicate_plan_result)
                    current_messages.append(duplicate_plan_result.to_tool_message())
                    yield self._task_snapshot_event(current_messages, lifecycle)
                    if lifecycle.record_tool_result(duplicate_plan_result):
                        yield self._terminal_error_event(lifecycle)
                        yield self._task_snapshot_event(current_messages, lifecycle, status="failed")
                        yield self._finish_event(messages, current_messages, lifecycle)
                        return
                    recovery_decision = self._build_recovery_decision(
                        duplicate_plan_result,
                        lifecycle,
                    )
                    current_messages.append(
                        self._self_correction_message(
                            duplicate_plan_result,
                            lifecycle,
                            recovery_decision,
                        )
                    )
                    yield self._failure_recovery_event(
                        duplicate_plan_result,
                        lifecycle,
                        recovery_decision,
                    )
                    yield self._self_correction_event(
                        duplicate_plan_result,
                        lifecycle,
                        recovery_decision,
                    )
                    continue

                tool_result = self._execute_tool_call(tool_call, lifecycle)
                yield self._tool_result_event(tool_result)
                current_messages.append(tool_result.to_tool_message())

                if tool_result.metadata.get("error_type") == "approval_required":
                    lifecycle.pending_approval(
                        "Human approval is required before executing the requested tool."
                    )
                    yield self._task_snapshot_event(
                        current_messages,
                        lifecycle,
                        status="pending_approval",
                    )
                    yield self._finish_event(messages, current_messages, lifecycle)
                    return

                yield self._task_snapshot_event(current_messages, lifecycle)

                if lifecycle.state.is_terminal:
                    yield self._terminal_error_event(lifecycle)
                    yield self._task_snapshot_event(current_messages, lifecycle, status="failed")
                    yield self._finish_event(messages, current_messages, lifecycle)
                    return

                if lifecycle.record_tool_result(tool_result):
                    yield self._terminal_error_event(lifecycle)
                    yield self._task_snapshot_event(current_messages, lifecycle, status="failed")
                    yield self._finish_event(messages, current_messages, lifecycle)
                    return

                if not tool_result.success:
                    recovery_decision = self._build_recovery_decision(tool_result, lifecycle)
                    current_messages.append(
                        self._self_correction_message(
                            tool_result,
                            lifecycle,
                            recovery_decision,
                        )
                    )
                    yield self._failure_recovery_event(
                        tool_result,
                        lifecycle,
                        recovery_decision,
                    )
                    yield self._self_correction_event(
                        tool_result,
                        lifecycle,
                        recovery_decision,
                    )

        yield self._terminal_error_event(lifecycle)
        yield self._task_snapshot_event(current_messages, lifecycle, status="failed")
        yield self._finish_event(messages, current_messages, lifecycle)

    def _model_call_event(
        self,
        *,
        phase: str,
        started_at: float,
        message_count: int,
        tool_count: int,
        model_kwargs: Dict[str, Any],
        response_tool_count: int = 0,
        error: Optional[Exception] = None,
    ) -> Dict[str, Any]:
        model_name = (
            model_kwargs.get("model_name")
            or getattr(self.llm, "model_name", None)
            or type(self.llm).__name__
        )
        data: Dict[str, Any] = {
            "phase": phase,
            "model_name": str(model_name),
            "provider": type(self.llm).__name__,
            "message_count": message_count,
            "tool_count": tool_count,
            "response_tool_count": response_tool_count,
            "latency_ms": round((time.perf_counter() - started_at) * 1000, 3),
            "success": phase == "completed",
        }
        if error is not None:
            data.update({"error_type": type(error).__name__, "error": str(error)})
        return {"type": "model_call", "data": data}

    def _partial_answer_from_tool_history(
        self,
        safe_content: str,
        current_messages: List[Dict[str, Any]],
        lifecycle_error: str,
    ) -> str:
        observations = [
            str(message.get("content") or "").strip()
            for message in current_messages
            if message.get("role") == "tool" and str(message.get("content") or "").strip()
        ]
        recent_observations = "\n\n".join(observations[-3:])
        parts = [
            "The agent reached the maximum reasoning steps for this run, so it stopped calling tools.",
            "Below is a bounded summary of the tool observations collected so far.",
        ]
        if safe_content.strip():
            parts.append(safe_content.strip())
        if recent_observations:
            parts.append(recent_observations[:3000])
        parts.append(f"Stop reason: {lifecycle_error}")
        return "\n\n".join(parts)

    def _execute_tool_call(
        self,
        tool_call: ToolCall,
        lifecycle: AgentLifecycle,
    ) -> ToolResult:
        if tool_call.validation_error:
            return ToolResult.invalid_arguments(tool_call, tool_call.validation_error)

        repeated_call_error = lifecycle.record_tool_call(tool_call)
        if repeated_call_error:
            return repeated_call_error

        return self.tool_registry.execute(tool_call)

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

    def _tool_result_event(self, tool_result: ToolResult) -> Dict[str, Any]:
        payload = tool_result.to_stream_event().to_payload()
        return {
            "type": payload["event"],
            "name": payload.get("name", tool_result.name),
            "content": payload.get("content", tool_result.to_observation()),
            "success": tool_result.success,
            "error": tool_result.error,
            "tool_call_id": tool_result.tool_call_id,
            "metadata": tool_result.metadata,
            "data": payload.get("data", {}),
        }

    def _terminal_error_event(self, lifecycle: AgentLifecycle) -> Dict[str, Any]:
        payload = lifecycle.state.terminal_payload()
        return {
            "type": "error",
            "content": lifecycle.state.error or "Agent run stopped.",
            "status": payload["status"],
            "termination_reason": payload["termination_reason"],
            "state": payload,
        }

    def _self_correction_event(
        self,
        tool_result: ToolResult,
        lifecycle: AgentLifecycle,
        recovery_decision: Dict[str, Any],
    ) -> Dict[str, Any]:
        remaining_budget = max(
            self.run_config.max_self_corrections - lifecycle.state.self_corrections,
            0,
        )
        return {
            "type": "self_correction",
            "content": (
                "Tool call failed; the error observation was returned to the model "
                "for bounded self-correction."
            ),
            "tool_call_id": tool_result.tool_call_id,
            "name": tool_result.name,
            "error": tool_result.error,
            "retryable": tool_result.is_retryable(),
            "repair_strategy": tool_result.repair_strategy(),
            "state": lifecycle.state.terminal_payload(),
            "data": {
                "retryable": tool_result.is_retryable(),
                "repair_strategy": tool_result.repair_strategy(),
                "diagnosis": tool_result.metadata.get("diagnosis"),
                "suggested_tool": tool_result.metadata.get("suggested_tool"),
                "remaining_self_corrections": remaining_budget,
                "recovery_action": recovery_decision.get("action"),
                "recovery_reason": recovery_decision.get("reason"),
                "tool_metadata": tool_result.metadata,
            },
        }

    def _self_correction_message(
        self,
        tool_result: ToolResult,
        lifecycle: AgentLifecycle,
        recovery_decision: Dict[str, Any],
    ) -> Dict[str, Any]:
        remaining_budget = max(
            self.run_config.max_self_corrections - lifecycle.state.self_corrections,
            0,
        )
        lines = [
            "[Self-Correction Instruction]",
            f"Tool failure for `{tool_result.name}`.",
            f"Remaining self-correction budget: {remaining_budget}.",
        ]

        diagnosis = tool_result.metadata.get("diagnosis")
        if diagnosis:
            lines.append(f"Diagnosis: {diagnosis}")

        repair_strategy = tool_result.repair_strategy()
        if repair_strategy:
            lines.append(f"Repair strategy: {repair_strategy}")

        suggested_tool = tool_result.metadata.get("suggested_tool")
        if suggested_tool:
            lines.append(f"Suggested tool: {suggested_tool}")

        missing_args = tool_result.metadata.get("missing_required_args") or []
        if missing_args:
            lines.append(
                "Missing required args: " + ", ".join(str(item) for item in missing_args)
            )

        type_errors = tool_result.metadata.get("argument_type_errors") or []
        if type_errors:
            lines.append(
                "Argument type issues: " + ", ".join(str(item) for item in type_errors)
            )

        lines.append(
            f"Recovery action: {recovery_decision.get('action', 'fallback_answer')}"
        )
        lines.append(
            f"Recovery reason: {recovery_decision.get('reason', 'tool_failure')}"
        )
        lines.append(recovery_decision.get("instruction", "Continue safely."))

        return {
            "role": "system",
            "content": "\n".join(lines),
        }

    def _failure_recovery_event(
        self,
        tool_result: ToolResult,
        lifecycle: AgentLifecycle,
        recovery_decision: Dict[str, Any],
    ) -> Dict[str, Any]:
        return {
            "type": "failure_recovery",
            "content": recovery_decision.get(
                "summary",
                "Agent recovery strategy updated after tool failure.",
            ),
            "tool_call_id": tool_result.tool_call_id,
            "name": tool_result.name,
            "error": tool_result.error,
            "state": lifecycle.state.terminal_payload(),
            "data": {
                "action": recovery_decision.get("action"),
                "reason": recovery_decision.get("reason"),
                "instruction": recovery_decision.get("instruction"),
                "same_tool_failures": lifecycle.state.failed_tool_names.get(
                    tool_result.name,
                    0,
                ),
                "successful_tool_calls": lifecycle.state.successful_tool_calls,
                "consecutive_tool_failures": lifecycle.state.consecutive_tool_failures,
                "remaining_self_corrections": max(
                    self.run_config.max_self_corrections
                    - lifecycle.state.self_corrections,
                    0,
                ),
            },
        }

    def _assess_plan(
        self,
        tool_calls: List[ToolCall],
        lifecycle: AgentLifecycle,
    ) -> Dict[str, Any]:
        signatures: Dict[str, int] = {}
        duplicate_signatures: List[str] = []
        validation_error_count = 0
        tool_names: List[str] = []
        warnings: List[str] = []
        recommended_mode = "execute"
        summary = "Tool plan looks executable."

        for tool_call in tool_calls:
            signature = self._tool_signature(tool_call)
            signatures[signature] = signatures.get(signature, 0) + 1
            if signatures[signature] == 2:
                duplicate_signatures.append(signature)
            tool_names.append(tool_call.name)
            if tool_call.validation_error:
                validation_error_count += 1

        if duplicate_signatures:
            warnings.append("duplicate_tool_call_in_iteration")
            recommended_mode = "simplify_plan"
            summary = "Detected duplicate tool calls in the same reasoning step."
        elif len(tool_calls) > self.run_config.max_tools_per_iteration:
            warnings.append("plan_too_wide_for_single_iteration")
            recommended_mode = "narrow_plan"
            summary = "Too many tool calls were planned for one iteration."
        elif validation_error_count:
            warnings.append("invalid_tool_arguments_emitted")
            recommended_mode = "repair_before_retry"
            summary = "The model emitted tool calls with invalid arguments."
        elif lifecycle.state.consecutive_tool_failures >= 1 and len(tool_calls) > 1:
            warnings.append("multi_tool_plan_under_failure_streak")
            recommended_mode = "reduce_parallelism"
            summary = "The plan widened while the agent is already in a failure streak."

        return {
            "iteration": lifecycle.state.iterations,
            "tool_call_count": len(tool_calls),
            "tool_names": tool_names,
            "duplicate_signature_count": len(duplicate_signatures),
            "duplicate_signatures": duplicate_signatures,
            "validation_error_count": validation_error_count,
            "consecutive_tool_failures": lifecycle.state.consecutive_tool_failures,
            "successful_tool_calls": lifecycle.state.successful_tool_calls,
            "recommended_mode": recommended_mode,
            "warnings": warnings,
            "summary": summary,
        }

    def _dedupe_iteration_tool_call(
        self,
        tool_call: ToolCall,
        iteration_seen_signatures: set[str],
    ) -> Optional[ToolResult]:
        signature = self._tool_signature(tool_call)
        if signature not in iteration_seen_signatures:
            iteration_seen_signatures.add(signature)
            return None

        return ToolResult.lifecycle_error(
            tool_call,
            (
                f"Skipped duplicate tool call in the same iteration for {tool_call.name}. "
                "Reuse the previous observation or choose a different action."
            ),
            metadata={
                "error_type": "duplicate_tool_call_in_iteration",
                "retryable": False,
                "repair_strategy": "reuse_previous_observation",
                "diagnosis": "The same tool call signature was emitted twice in one reasoning step.",
            },
        )

    def _build_recovery_decision(
        self,
        tool_result: ToolResult,
        lifecycle: AgentLifecycle,
    ) -> Dict[str, Any]:
        same_tool_failures = lifecycle.state.failed_tool_names.get(tool_result.name, 0)
        remaining_self_corrections = max(
            self.run_config.max_self_corrections - lifecycle.state.self_corrections,
            0,
        )

        action = tool_result.repair_strategy() or "fallback_answer"
        reason = "tool_failure"
        instruction = (
            "Retry only if you can materially improve the tool name or arguments; "
            "otherwise continue with a fallback answer."
            if tool_result.is_retryable()
            else "Do not retry the same tool path. Choose a different tool or answer without it."
        )
        summary = "Agent will attempt bounded self-correction."

        if not tool_result.is_retryable():
            action = "fallback_answer"
            reason = "non_retryable_failure"
            instruction = (
                "Do not retry this tool path. Either switch to a different tool or answer "
                "with the information already available."
            )
            summary = "Recovery escalated to fallback because the tool failure is non-retryable."
        elif same_tool_failures >= 2 and lifecycle.state.successful_tool_calls > 0:
            action = "finalize_with_partial_results"
            reason = "same_tool_failed_repeatedly_after_progress"
            instruction = (
                "Stop retrying the same tool. Use the successful observations already collected "
                "to produce the best possible final answer, and clearly note any uncertainty."
            )
            summary = "Recovery escalated to finalize with partial results after repeated tool failures."
        elif same_tool_failures >= 2:
            action = "switch_tool_or_fallback"
            reason = "same_tool_failed_repeatedly_without_progress"
            instruction = (
                "Do not call the same failing tool again. Switch to a different tool if it can "
                "advance the task; otherwise provide a bounded fallback answer."
            )
            summary = "Recovery escalated to switching strategy after repeated failures."
        elif remaining_self_corrections == 0 and lifecycle.state.successful_tool_calls > 0:
            action = "finalize_with_partial_results"
            reason = "self_correction_budget_exhausted_with_partial_progress"
            instruction = (
                "Self-correction budget is exhausted. Stop trying more tools and answer from the "
                "useful observations already obtained."
            )
            summary = "Recovery switched to partial-result finalization because correction budget is exhausted."
        elif remaining_self_corrections == 0:
            action = "fallback_answer"
            reason = "self_correction_budget_exhausted"
            instruction = (
                "Self-correction budget is exhausted. Do not attempt another retry; provide a bounded "
                "fallback answer or explain the limitation concisely."
            )
            summary = "Recovery switched to fallback because correction budget is exhausted."

        return {
            "action": action,
            "reason": reason,
            "instruction": instruction,
            "summary": summary,
        }

    def _tool_signature(self, tool_call: ToolCall) -> str:
        try:
            args_text = tool_call.raw_arguments or json.dumps(
                tool_call.args,
                ensure_ascii=False,
                sort_keys=True,
            )
        except TypeError:
            args_text = repr(tool_call.args)
        return f"{tool_call.name}:{args_text}"

    def _finish_event(
        self,
        initial_messages: List[Dict[str, Any]],
        current_messages: List[Dict[str, Any]],
        lifecycle: AgentLifecycle,
    ) -> Dict[str, Any]:
        return {
            "type": "finish",
            "new_messages": current_messages[len(initial_messages) :],
            "status": lifecycle.state.status.value,
            "termination_reason": lifecycle.state.termination_reason.value
            if lifecycle.state.termination_reason
            else None,
            "state": lifecycle.state.terminal_payload(),
        }

    def _task_snapshot_event(
        self,
        current_messages: List[Dict[str, Any]],
        lifecycle: AgentLifecycle,
        status: str = "running",
    ) -> Dict[str, Any]:
        return {
            "type": "task_snapshot",
            "data": {
                "route_worker": "tool_agent_worker",
                "status": status,
                "messages": current_messages,
                "lifecycle_state": lifecycle.state.model_dump(exclude_none=True),
            },
        }

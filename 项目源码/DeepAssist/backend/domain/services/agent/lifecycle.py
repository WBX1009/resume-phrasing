import json
from typing import Any, Dict, Optional

from backend.domain.entities.agent_run import (
    AgentRunConfig,
    AgentRunState,
    AgentRunStatus,
    AgentTerminationReason,
)
from backend.domain.entities.tooling import ToolCall, ToolResult


class AgentLifecycle:
    """Guards a single ReAct run against runaway loops and broken tool states."""

    def __init__(self, config: AgentRunConfig, state: Optional[AgentRunState] = None):
        self.config = config
        self.state = state or AgentRunState()

    @property
    def remaining_steps(self) -> int:
        return max(self.config.max_iterations - self.state.iterations, 0)

    def start_iteration(self) -> bool:
        if self.state.iterations >= self.config.max_iterations:
            self.stop(
                AgentRunStatus.MAX_STEPS_EXCEEDED,
                AgentTerminationReason.MAX_STEPS_EXCEEDED,
                f"Reached max agent iterations ({self.config.max_iterations}).",
            )
            return False

        self.state.iterations += 1
        return True

    def guard_tool_phase(self) -> Optional[str]:
        if self.remaining_steps < 1:
            self.stop(
                AgentRunStatus.MAX_STEPS_EXCEEDED,
                AgentTerminationReason.MAX_STEPS_EXCEEDED,
                "The model requested tools but no remaining iteration is available for observation.",
            )
            return self.state.error
        return None

    def record_tool_call(self, tool_call: ToolCall) -> Optional[ToolResult]:
        signature = self._tool_signature(tool_call.name, tool_call.args)
        count = self.state.repeated_tool_calls.get(signature, 0) + 1
        self.state.repeated_tool_calls[signature] = count

        if count <= self.config.repeated_tool_call_limit:
            return None

        self.stop(
            AgentRunStatus.FAILED,
            AgentTerminationReason.REPEATED_TOOL_CALL_EXCEEDED,
            (
                f"Repeated tool call limit exceeded for {tool_call.name}; "
                f"same call was requested {count} times."
            ),
        )
        return ToolResult.lifecycle_error(
            tool_call,
            self.state.error or "Repeated tool call limit exceeded.",
            metadata=self.state.terminal_payload(),
        )

    def record_tool_result(self, result: ToolResult) -> Optional[str]:
        if result.success:
            self.state.successful_tool_calls += 1
            self.state.consecutive_tool_failures = 0
            return None

        self.state.tool_errors += 1
        self.state.consecutive_tool_failures += 1
        self.state.last_failed_tool_name = result.name
        self.state.failed_tool_names[result.name] = (
            self.state.failed_tool_names.get(result.name, 0) + 1
        )
        if self.state.tool_errors < self.config.max_tool_errors:
            if self.state.self_corrections < self.config.max_self_corrections:
                self.state.self_corrections += 1
                return None

            self.stop(
                AgentRunStatus.TOOL_ERROR_EXCEEDED,
                AgentTerminationReason.TOOL_ERROR_EXCEEDED,
                (
                    f"Self-correction budget exceeded "
                    f"({self.state.self_corrections}/{self.config.max_self_corrections})."
                ),
            )
            return self.state.error

        self.stop(
            AgentRunStatus.TOOL_ERROR_EXCEEDED,
            AgentTerminationReason.TOOL_ERROR_EXCEEDED,
            (
                f"Tool error budget exceeded "
                f"({self.state.tool_errors}/{self.config.max_tool_errors})."
            ),
        )
        return self.state.error

    def complete(self, final_answer: str) -> None:
        self.state.final_answer = final_answer
        self.stop(
            AgentRunStatus.COMPLETED,
            AgentTerminationReason.FINAL_ANSWER,
            error=None,
        )

    def pending_approval(self, message: str = "Human approval is required.") -> None:
        self.stop(
            AgentRunStatus.PENDING_APPROVAL,
            AgentTerminationReason.HUMAN_APPROVAL_REQUIRED,
            message,
        )

    def fail(self, reason: AgentTerminationReason, error: str) -> None:
        self.stop(AgentRunStatus.FAILED, reason, error)

    def stop(
        self,
        status: AgentRunStatus,
        reason: AgentTerminationReason,
        error: Optional[str],
    ) -> None:
        self.state.status = status
        self.state.termination_reason = reason
        self.state.error = error

    def _tool_signature(self, tool_name: str, args: Dict[str, Any]) -> str:
        try:
            args_text = json.dumps(args, ensure_ascii=False, sort_keys=True)
        except TypeError:
            args_text = repr(args)
        return f"{tool_name}:{args_text}"

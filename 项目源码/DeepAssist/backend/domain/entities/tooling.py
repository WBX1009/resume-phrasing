import json
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field, model_validator


class ToolCall(BaseModel):
    """Canonical tool call passed through the agent runtime."""

    id: str
    name: str
    args: Dict[str, Any] = Field(default_factory=dict)
    type: str = "function"
    raw_arguments: Optional[str] = None
    validation_error: Optional[str] = None

    @model_validator(mode="before")
    @classmethod
    def accept_openai_shape(cls, value: Any) -> Any:
        if not isinstance(value, dict) or "function" not in value:
            return value

        function = value.get("function") or {}
        raw_arguments = function.get("arguments", "{}")
        args: Dict[str, Any] = {}
        validation_error = None

        if isinstance(raw_arguments, str):
            try:
                parsed = json.loads(raw_arguments or "{}")
                if isinstance(parsed, dict):
                    args = parsed
                else:
                    validation_error = "arguments must decode to a JSON object"
            except json.JSONDecodeError as exc:
                validation_error = str(exc)
        elif isinstance(raw_arguments, dict):
            args = raw_arguments
            raw_arguments = json.dumps(raw_arguments, ensure_ascii=False)
        else:
            validation_error = "arguments must be a JSON object or JSON string"

        return {
            "id": str(value.get("id", "")),
            "name": function.get("name", ""),
            "args": args,
            "type": value.get("type", "function"),
            "raw_arguments": raw_arguments if isinstance(raw_arguments, str) else None,
            "validation_error": validation_error,
        }

    @classmethod
    def from_llm_tool_call(cls, tool_call: Any) -> "ToolCall":
        if isinstance(tool_call, dict):
            return cls.model_validate(tool_call)

        return cls.model_validate(
            {
                "id": tool_call.id,
                "type": "function",
                "function": {
                    "name": tool_call.function.name,
                    "arguments": tool_call.function.arguments,
                },
            }
        )

    def to_openai_tool_call(self) -> Dict[str, Any]:
        arguments = self.raw_arguments
        if arguments is None:
            arguments = json.dumps(self.args, ensure_ascii=False)
        return {
            "id": self.id,
            "type": self.type,
            "function": {
                "name": self.name,
                "arguments": arguments,
            },
        }

    def to_stream_data(self) -> Dict[str, Any]:
        return {
            "tool_call_id": self.id,
            "name": self.name,
            "args": self.args,
            "validation_error": self.validation_error,
        }


class ToolResult(BaseModel):
    """Canonical tool result returned by every tool execution path."""

    tool_call_id: str
    name: str
    success: bool
    content: str = ""
    error: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def ok(
        cls,
        call: ToolCall,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "ToolResult":
        return cls(
            tool_call_id=call.id,
            name=call.name,
            success=True,
            content=content,
            metadata=metadata or {},
        )

    @classmethod
    def unknown_tool(
        cls,
        call: ToolCall,
        available_tools: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "ToolResult":
        return cls(
            tool_call_id=call.id,
            name=call.name,
            success=False,
            error=f"Unknown tool '{call.name}'. Available tools: {available_tools}",
            metadata=metadata or {"error_type": "unknown_tool"},
        )

    @classmethod
    def invalid_arguments(
        cls,
        call: ToolCall,
        error: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "ToolResult":
        return cls(
            tool_call_id=call.id,
            name=call.name,
            success=False,
            error=f"Invalid tool arguments for '{call.name}': {error}",
            metadata=metadata or {"error_type": "invalid_arguments"},
        )

    @classmethod
    def execution_failed(
        cls,
        call: ToolCall,
        error: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "ToolResult":
        return cls(
            tool_call_id=call.id,
            name=call.name,
            success=False,
            error=f"Tool execution failed for '{call.name}': {error}",
            metadata=metadata or {"error_type": "tool_execution_failed"},
        )

    @classmethod
    def lifecycle_error(
        cls,
        call: ToolCall,
        error: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "ToolResult":
        return cls(
            tool_call_id=call.id,
            name=call.name,
            success=False,
            error=error,
            metadata=metadata or {"error_type": "agent_lifecycle"},
        )

    def to_observation(self) -> str:
        if self.success:
            return self.content
        error = self.error or self.content
        metadata_lines = self._metadata_hint_lines()
        hint_block = "\n".join(metadata_lines)
        if hint_block:
            hint_block = f"{hint_block}\n"
        return (
            f"{error}\n"
            f"{hint_block}"
            "Self-correction hint: inspect the tool name and arguments, then either "
            "retry with corrected JSON arguments, choose a better tool, or answer "
            "without the tool if it is not needed."
        )

    def repair_strategy(self) -> str:
        return str(self.metadata.get("repair_strategy") or "")

    def is_retryable(self) -> bool:
        return bool(self.metadata.get("retryable"))

    def _metadata_hint_lines(self) -> list[str]:
        if not self.metadata:
            return []

        lines: list[str] = []
        diagnosis = self.metadata.get("diagnosis")
        if diagnosis:
            lines.append(f"Diagnosis: {diagnosis}")

        suggested_tool = self.metadata.get("suggested_tool")
        if suggested_tool:
            lines.append(f"Suggested tool: {suggested_tool}")

        missing = self.metadata.get("missing_required_args") or []
        if missing:
            lines.append(f"Missing required args: {', '.join(str(item) for item in missing)}")

        unexpected = self.metadata.get("unexpected_args") or []
        if unexpected:
            lines.append(f"Unexpected args: {', '.join(str(item) for item in unexpected)}")

        type_errors = self.metadata.get("argument_type_errors") or []
        if type_errors:
            rendered = ", ".join(str(item) for item in type_errors)
            lines.append(f"Argument type issues: {rendered}")

        allowed = self.metadata.get("allowed_tools") or self.metadata.get("available_tools") or []
        if isinstance(allowed, list) and allowed:
            lines.append(f"Available tools: {', '.join(str(item) for item in allowed)}")

        strategy = self.metadata.get("repair_strategy")
        if strategy:
            lines.append(f"Repair strategy: {strategy}")

        return lines

    def to_tool_message(self) -> Dict[str, Any]:
        return {
            "role": "tool",
            "tool_call_id": self.tool_call_id,
            "name": self.name,
            "content": self.to_observation(),
        }

    def to_stream_event(self):
        from backend.domain.entities.stream_event import StreamEvent

        return StreamEvent.tool_result(
            name=self.name,
            content=self.to_observation(),
            tool_call_id=self.tool_call_id,
            success=self.success,
            error=self.error,
            metadata=self.metadata,
        )

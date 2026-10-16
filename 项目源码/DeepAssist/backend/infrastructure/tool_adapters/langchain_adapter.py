import json
from typing import Any, Dict, List

from backend.domain.entities.tooling import ToolCall, ToolResult
from backend.domain.interfaces.tool_provider import ToolProviderPort


class LangChainToolAdapter(ToolProviderPort):
    """Adapt LangChain-style tools into DeepAssist governed tool calls."""

    def __init__(self, tools: List[object] | None = None, provider_name: str = "langchain"):
        self.tools = tools or []
        self.provider_name = provider_name

    def list_tools(self) -> List[Dict[str, Any]]:
        return [self._tool_spec(tool) for tool in self.tools if self._tool_name(tool)]

    def execute(self, tool_call: ToolCall) -> ToolResult:
        tool = self._find_tool(tool_call.name)
        if tool is None:
            return ToolResult.unknown_tool(
                tool_call,
                ", ".join(sorted(self._tool_name(item) for item in self.tools if self._tool_name(item))),
                metadata={
                    "error_type": "unknown_tool",
                    "provider": self.provider_name,
                    "retryable": True,
                },
            )

        try:
            raw_result = self._invoke_tool(tool, tool_call.args)
            return ToolResult.ok(
                tool_call,
                self._stringify(raw_result),
                metadata={"provider": self.provider_name},
            )
        except Exception as exc:
            return ToolResult.execution_failed(
                tool_call,
                str(exc),
                metadata={
                    "error_type": "langchain_tool_execution_failed",
                    "provider": self.provider_name,
                    "retryable": True,
                    "exception_type": type(exc).__name__,
                },
            )

    def _find_tool(self, name: str) -> object | None:
        for tool in self.tools:
            if self._tool_name(tool) == name:
                return tool
        return None

    def _tool_spec(self, tool: object) -> Dict[str, Any]:
        name = self._tool_name(tool)
        return {
            "name": name,
            "description": str(getattr(tool, "description", "") or f"LangChain tool {name}"),
            "parameters": self._parameters_schema(tool),
            "provider": self.provider_name,
        }

    def _tool_name(self, tool: object) -> str:
        return str(getattr(tool, "name", "") or "").strip()

    def _parameters_schema(self, tool: object) -> Dict[str, Any]:
        args_schema = getattr(tool, "args_schema", None)
        if args_schema is not None:
            if hasattr(args_schema, "model_json_schema"):
                return self._normalize_schema(args_schema.model_json_schema())
            if hasattr(args_schema, "schema"):
                return self._normalize_schema(args_schema.schema())

        args = getattr(tool, "args", None)
        if isinstance(args, dict):
            return self._normalize_schema(
                {
                    "type": "object",
                    "properties": args,
                    "required": [
                        name
                        for name, schema in args.items()
                        if isinstance(schema, dict) and not schema.get("default")
                    ],
                }
            )

        return {"type": "object", "properties": {}, "required": []}

    def _normalize_schema(self, schema: Dict[str, Any]) -> Dict[str, Any]:
        if schema.get("type") == "object":
            schema.setdefault("properties", {})
            schema.setdefault("required", [])
            return schema
        return {
            "type": "object",
            "properties": schema.get("properties", {}),
            "required": schema.get("required", []),
        }

    def _invoke_tool(self, tool: object, args: Dict[str, Any]) -> Any:
        invoke = getattr(tool, "invoke", None)
        if callable(invoke):
            return invoke(args)

        run = getattr(tool, "run", None)
        if callable(run):
            try:
                return run(args)
            except TypeError:
                return run(**args)

        call = getattr(tool, "__call__", None)
        if callable(call):
            return call(**args)

        raise RuntimeError(f"LangChain tool {self._tool_name(tool)} is not callable.")

    def _stringify(self, raw_result: Any) -> str:
        if raw_result is None:
            return ""
        if isinstance(raw_result, str):
            return raw_result
        if isinstance(raw_result, (dict, list)):
            return json.dumps(raw_result, ensure_ascii=False, default=str)
        if hasattr(raw_result, "model_dump"):
            return json.dumps(raw_result.model_dump(), ensure_ascii=False, default=str)
        return str(raw_result)

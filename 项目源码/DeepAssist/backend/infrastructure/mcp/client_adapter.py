import json
import inspect
from typing import Any, Dict, List

from backend.domain.entities.tooling import ToolCall, ToolResult
from backend.domain.interfaces.tool_provider import ToolProviderPort


class MCPToolClientAdapter(ToolProviderPort):
    """Import tools from an external MCP client behind DeepAssist's tool port."""

    def __init__(
        self,
        client: Any | None = None,
        server_name: str = "",
        tool_prefix: str = "",
        tool_aliases: Dict[str, str] | None = None,
    ):
        self.client = client
        self.server_name = server_name or "external_mcp"
        self.tool_prefix = tool_prefix
        # The MCP server owns transport-facing names while the application
        # may already expose a stable tool vocabulary to the model.  Aliases
        # keep that public vocabulary unchanged without bypassing MCP.
        self.tool_aliases = dict(tool_aliases or {})
        self._reverse_tool_aliases = {
            local_name: remote_name
            for remote_name, local_name in self.tool_aliases.items()
        }

    def list_tools(self) -> List[Dict[str, Any]]:
        if self.client is None:
            return []

        raw_tools = self._call_client_list_tools()
        return [self._normalize_tool(item) for item in raw_tools if self._tool_name(item)]

    def execute(self, tool_call: ToolCall) -> ToolResult:
        if self.client is None:
            return ToolResult.execution_failed(
                tool_call,
                "MCP client is not configured.",
                metadata={"error_type": "tool_provider_unavailable", "retryable": False},
            )

        remote_name = self._remote_tool_name(tool_call.name)
        try:
            raw_result = self._call_client_tool(remote_name, tool_call.args)
            return ToolResult.ok(
                tool_call,
                self._stringify_result(raw_result),
                metadata={
                    "provider": "mcp",
                    "server_name": self.server_name,
                    "remote_tool_name": remote_name,
                },
            )
        except Exception as exc:
            return ToolResult.execution_failed(
                tool_call,
                str(exc),
                metadata={
                    "error_type": "mcp_tool_execution_failed",
                    "retryable": True,
                    "provider": "mcp",
                    "server_name": self.server_name,
                    "remote_tool_name": remote_name,
                    "exception_type": type(exc).__name__,
                },
            )

    async def alist_tools(self) -> List[Dict[str, Any]]:
        """Async MCP discovery for clients exposing awaitable methods."""
        if self.client is None:
            return []
        raw_tools = await self._call_client_list_tools_async()
        return [self._normalize_tool(item) for item in raw_tools if self._tool_name(item)]

    async def aexecute(self, tool_call: ToolCall) -> ToolResult:
        """Async MCP execution while preserving the canonical ToolResult port."""
        if self.client is None:
            return ToolResult.execution_failed(
                tool_call,
                "MCP client is not configured.",
                metadata={"error_type": "tool_provider_unavailable", "retryable": False},
            )
        remote_name = self._remote_tool_name(tool_call.name)
        try:
            raw_result = await self._call_client_tool_async(remote_name, tool_call.args)
            return ToolResult.ok(
                tool_call,
                self._stringify_result(raw_result),
                metadata={
                    "provider": "mcp",
                    "server_name": self.server_name,
                    "remote_tool_name": remote_name,
                },
            )
        except Exception as exc:
            return ToolResult.execution_failed(
                tool_call,
                str(exc),
                metadata={
                    "error_type": "mcp_tool_execution_failed",
                    "retryable": True,
                    "provider": "mcp",
                    "server_name": self.server_name,
                    "remote_tool_name": remote_name,
                    "exception_type": type(exc).__name__,
                },
            )

    def _call_client_list_tools(self) -> List[Any]:
        list_tools = getattr(self.client, "list_tools", None)
        if not callable(list_tools):
            raise RuntimeError("MCP client does not expose list_tools().")
        response = list_tools()
        tools = getattr(response, "tools", response)
        return list(tools or [])

    def _call_client_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        for method_name in ("call_tool", "execute_tool", "call"):
            method = getattr(self.client, method_name, None)
            if callable(method):
                return method(tool_name, arguments)
        raise RuntimeError("MCP client does not expose a supported tool call method.")

    async def _call_client_list_tools_async(self) -> List[Any]:
        method = getattr(self.client, "list_tools", None)
        if not callable(method):
            raise RuntimeError("MCP client does not expose list_tools().")
        response = method()
        if inspect.isawaitable(response):
            response = await response
        tools = getattr(response, "tools", response)
        return list(tools or [])

    async def _call_client_tool_async(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        for method_name in ("call_tool", "execute_tool", "call"):
            method = getattr(self.client, method_name, None)
            if callable(method):
                response = method(tool_name, arguments)
                if inspect.isawaitable(response):
                    response = await response
                return response
        raise RuntimeError("MCP client does not expose a supported tool call method.")

    def _normalize_tool(self, raw_tool: Any) -> Dict[str, Any]:
        data = self._tool_to_dict(raw_tool)
        remote_name = str(data.get("name") or "").strip()
        schema = (
            data.get("inputSchema")
            or data.get("input_schema")
            or data.get("parameters")
            or {"type": "object", "properties": {}, "required": []}
        )
        return {
            "name": f"{self.tool_prefix}{self.tool_aliases.get(remote_name, remote_name)}",
            "description": str(data.get("description") or f"MCP tool {remote_name}"),
            "parameters": schema,
            "provider": "mcp",
            "server_name": self.server_name,
            "remote_name": remote_name,
        }

    def _tool_name(self, raw_tool: Any) -> str:
        return str(self._tool_to_dict(raw_tool).get("name") or "").strip()

    def _remote_tool_name(self, local_name: str) -> str:
        if self.tool_prefix and local_name.startswith(self.tool_prefix):
            local_name = local_name[len(self.tool_prefix) :]
        return self._reverse_tool_aliases.get(local_name, local_name)

    def _tool_to_dict(self, raw_tool: Any) -> Dict[str, Any]:
        if isinstance(raw_tool, dict):
            return raw_tool
        if hasattr(raw_tool, "model_dump"):
            return raw_tool.model_dump()
        return {
            "name": getattr(raw_tool, "name", ""),
            "description": getattr(raw_tool, "description", ""),
            "inputSchema": getattr(raw_tool, "inputSchema", None)
            or getattr(raw_tool, "input_schema", None)
            or getattr(raw_tool, "parameters", None),
        }

    def _stringify_result(self, raw_result: Any) -> str:
        if raw_result is None:
            return ""
        if isinstance(raw_result, str):
            return raw_result
        if hasattr(raw_result, "content"):
            content = getattr(raw_result, "content")
            if isinstance(content, list):
                rendered = []
                for item in content:
                    if isinstance(item, dict):
                        rendered.append(str(item.get("text") or item.get("content") or item))
                    else:
                        rendered.append(str(getattr(item, "text", item)))
                return "\n".join(rendered)
            return str(content)
        if isinstance(raw_result, (dict, list)):
            return json.dumps(raw_result, ensure_ascii=False, default=str)
        if hasattr(raw_result, "model_dump"):
            return json.dumps(raw_result.model_dump(), ensure_ascii=False, default=str)
        return str(raw_result)

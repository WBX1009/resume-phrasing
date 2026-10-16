from typing import Dict, List, Protocol

from backend.domain.entities.tooling import ToolCall, ToolResult


class ToolProviderPort(Protocol):
    """Port for external tool ecosystems such as MCP or LangChain tools."""

    def list_tools(self) -> List[Dict]:
        ...

    def execute(self, tool_call: ToolCall) -> ToolResult:
        ...

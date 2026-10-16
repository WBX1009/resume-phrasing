import json
import inspect
from pathlib import Path
from typing import Any, Callable, Dict, List

from backend.infrastructure.tools.file_ops import list_sandbox_files, read_local_file


class DeepAssistMCPServer:
    """Expose DeepAssist read-only capabilities through an MCP-shaped adapter."""

    def __init__(
        self,
        kb_app: Any | None = None,
        runtime_app: Any | None = None,
        session_summary_provider: Callable[[str], Any] | None = None,
        kb_search_provider: Callable[[str, str], Any] | None = None,
        file_read_provider: Callable[[str], Any] | None = None,
        file_list_provider: Callable[[str], Any] | None = None,
    ):
        self.kb_app = kb_app
        self.runtime_app = runtime_app
        self.session_summary_provider = session_summary_provider
        self.kb_search_provider = kb_search_provider
        self.file_read_provider = file_read_provider or read_local_file
        self.file_list_provider = file_list_provider or list_sandbox_files

    def list_tools(self) -> List[Dict[str, Any]]:
        return [
            self._tool_schema(
                name="search_kb",
                description="Search DeepAssist knowledge bases for read-only evidence.",
                properties={
                    "query": {"type": "string", "description": "Search query."},
                    "collection_name": {
                        "type": "string",
                        "description": "Knowledge-base collection, or __all__.",
                    },
                },
                required=["query"],
            ),
            self._tool_schema(
                name="list_kb_files",
                description="List indexed files in a DeepAssist knowledge-base collection.",
                properties={
                    "collection_name": {
                        "type": "string",
                        "description": "Knowledge-base collection name.",
                    }
                },
            ),
            self._tool_schema(
                name="list_kb_collections",
                description="List DeepAssist knowledge-base collections.",
            ),
            self._tool_schema(
                name="get_session_summary",
                description="Return a saved session summary without exposing raw message history.",
                properties={
                    "session_id": {"type": "string", "description": "DeepAssist session id."}
                },
                required=["session_id"],
            ),
            self._tool_schema(
                name="get_runtime_capabilities",
                description="Return DeepAssist runtime capabilities and tool groups.",
            ),
            self._tool_schema(
                name="read_file",
                description="Read a UTF-8 text file inside the governed DeepAssist sandbox.",
                properties={
                    "file_name": {
                        "type": "string",
                        "description": "Relative path inside the sandbox.",
                    }
                },
                required=["file_name"],
            ),
            self._tool_schema(
                name="list_files",
                description="List files inside the governed DeepAssist sandbox.",
                properties={
                    "path": {
                        "type": "string",
                        "description": "Relative directory path inside the sandbox.",
                    }
                },
            ),
        ]

    def call_tool(self, name: str, arguments: Dict[str, Any] | None = None) -> Dict[str, Any]:
        args = arguments or {}
        if name == "search_kb":
            payload = self._search_kb(
                query=str(args.get("query") or ""),
                collection_name=str(args.get("collection_name") or "__all__"),
            )
        elif name == "list_kb_files":
            payload = self._list_kb_files(str(args.get("collection_name") or "tech_docs_kb"))
        elif name == "list_kb_collections":
            payload = self._list_kb_collections()
        elif name == "get_session_summary":
            payload = self._get_session_summary(str(args.get("session_id") or ""))
        elif name == "get_runtime_capabilities":
            payload = self._get_runtime_capabilities()
        elif name == "read_file":
            payload = self._read_file(str(args.get("file_name") or ""))
        elif name == "list_files":
            payload = self._list_files(str(args.get("path") or "."))
        else:
            payload = {
                "status": "error",
                "message": f"Unknown MCP tool: {name}",
                "available_tools": [tool["name"] for tool in self.list_tools()],
            }
        return self._mcp_result(self._sanitize_payload(payload))

    async def acall_tool(
        self, name: str, arguments: Dict[str, Any] | None = None
    ) -> Dict[str, Any]:
        """Async facade used by MCP transports with awaitable providers."""
        args = arguments or {}
        if name == "search_kb" and self.kb_search_provider is not None:
            value = self.kb_search_provider(
                str(args.get("query") or ""),
                str(args.get("collection_name") or "__all__"),
            )
            if inspect.isawaitable(value):
                value = await value
            payload = self._coerce_payload(value)
            return self._mcp_result(self._sanitize_payload(payload))
        return self.call_tool(name, args)

    def run(self) -> None:
        try:
            from mcp.server.fastmcp import FastMCP
        except Exception as exc:
            raise RuntimeError(
                "MCP SDK is not installed. Install the optional MCP dependency before starting "
                "DeepAssistMCPServer as a network service."
            ) from exc

        server = FastMCP("deepassist")
        for tool in self.list_tools():
            tool_name = tool["name"]

            def _handler(tool_name: str = tool_name, **kwargs: Any) -> Dict[str, Any]:
                return self.call_tool(tool_name, kwargs)

            server.tool(name=tool_name, description=tool["description"])(_handler)
        server.run()

    def _search_kb(self, query: str, collection_name: str) -> Dict[str, Any]:
        if not query.strip():
            return {"status": "error", "message": "query is required"}
        if self.kb_search_provider is None:
            return {"status": "error", "message": "KB search provider is not configured"}
        return self._coerce_payload(self.kb_search_provider(query, collection_name))

    def _list_kb_files(self, collection_name: str) -> Dict[str, Any]:
        if self.kb_app is None:
            return {"status": "error", "message": "KnowledgeBaseApp is not configured"}
        return self._coerce_payload(self.kb_app.list_files(collection_name=collection_name))

    def _list_kb_collections(self) -> Dict[str, Any]:
        if self.kb_app is None:
            return {"status": "error", "message": "KnowledgeBaseApp is not configured"}
        return self._coerce_payload(self.kb_app.list_collections())

    def _get_session_summary(self, session_id: str) -> Dict[str, Any]:
        if not session_id.strip():
            return {"status": "error", "message": "session_id is required"}
        if self.session_summary_provider is None:
            return {"status": "error", "message": "Session summary provider is not configured"}
        summary = self.session_summary_provider(session_id)
        if summary is None:
            return {"status": "success", "data": None}
        if hasattr(summary, "model_dump"):
            return {"status": "success", "data": summary.model_dump()}
        return {"status": "success", "data": summary}

    def _get_runtime_capabilities(self) -> Dict[str, Any]:
        if self.runtime_app is None:
            return {"status": "error", "message": "RuntimeApplication is not configured"}
        return self._coerce_payload(self.runtime_app.get_capabilities())

    def _read_file(self, file_name: str) -> Dict[str, Any]:
        if not file_name.strip():
            return {"status": "error", "message": "file_name is required"}
        try:
            return {"status": "success", "data": self.file_read_provider(file_name)}
        except Exception as exc:
            return {"status": "error", "message": f"File read failed: {exc}"}

    def _list_files(self, path: str) -> Dict[str, Any]:
        try:
            return {"status": "success", "data": self.file_list_provider(path)}
        except Exception as exc:
            return {"status": "error", "message": f"File listing failed: {exc}"}

    def _tool_schema(
        self,
        name: str,
        description: str,
        properties: Dict[str, Any] | None = None,
        required: List[str] | None = None,
    ) -> Dict[str, Any]:
        return {
            "name": name,
            "description": description,
            "inputSchema": {
                "type": "object",
                "properties": properties or {},
                "required": required or [],
            },
            "provider": "deepassist",
            "read_only": True,
        }

    def _mcp_result(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "content": [
                {
                    "type": "text",
                    "text": json.dumps(payload, ensure_ascii=False, default=str),
                }
            ],
            "structuredContent": payload,
            "isError": payload.get("status") == "error",
        }

    def _coerce_payload(self, value: Any) -> Dict[str, Any]:
        if isinstance(value, dict):
            return value
        if hasattr(value, "model_dump"):
            return value.model_dump()
        return {"status": "success", "data": value}

    def _sanitize_payload(self, value: Any) -> Any:
        if isinstance(value, dict):
            return {key: self._sanitize_payload_item(key, item) for key, item in value.items()}
        if isinstance(value, list):
            return [self._sanitize_payload(item) for item in value]
        return value

    def _sanitize_payload_item(self, key: str, value: Any) -> Any:
        if isinstance(value, str) and key in {"path", "file_path", "source_path", "absolute_path"}:
            if Path(value).is_absolute():
                return Path(value).name
        return self._sanitize_payload(value)

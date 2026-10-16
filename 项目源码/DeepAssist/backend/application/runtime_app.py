from typing import Any, Callable

from backend.common.config import settings
from backend.application.kb_app import KnowledgeBaseApp
from backend.domain.entities.capability import CapabilitySpec, ToolGroupSpec
from backend.domain.interfaces.approval_store import ApprovalStorePort
from backend.domain.interfaces.trace_store import TraceStorePort
from backend.domain.services.agent.tooling import ToolRegistry
from backend.domain.services.agent.skill_registry import SkillRegistry


class RuntimeApplication:
    """Expose the effective runtime contract consumed by the UI layer."""

    def __init__(
        self,
        kb_app: KnowledgeBaseApp,
        tool_registry: ToolRegistry,
        diagnostics_provider: Callable[[bool], dict[str, Any]] | None = None,
        trace_store: TraceStorePort | None = None,
        approval_store: ApprovalStorePort | None = None,
        skill_registry: SkillRegistry | None = None,
    ):
        self.kb_app = kb_app
        self.tool_registry = tool_registry
        self.diagnostics_provider = diagnostics_provider
        self.trace_store = trace_store
        self.approval_store = approval_store
        self.skill_registry = skill_registry

    def get_capabilities(self) -> dict[str, Any]:
        collections_payload = self.kb_app.list_collections()
        collections = collections_payload.get("data", [])
        if not isinstance(collections, list):
            collections = []

        provider = (settings.LLM_PROVIDER or "deepseek").strip().lower().replace("-", "_")
        provider_label = (
            "OpenAI-compatible endpoint"
            if provider in {"openai_compatible", "openai"}
            else "DeepSeek"
        )
        model_options = [
            {
                "id": settings.LLM_CHAT_MODEL,
                "label": f"{provider_label} chat",
                "supported": True,
                "supports_streaming": True,
                "supports_tool_calls": True,
                "notes": "Default model for quick chat, RAG answering, and tool-enabled agent turns.",
            },
        ]
        if provider not in {"openai_compatible", "openai"}:
            model_options.append(
                {
                    "id": settings.LLM_REASONER_MODEL,
                    "label": f"{provider_label} reasoner",
                    "supported": True,
                    "supports_streaming": True,
                    "supports_tool_calls": False,
                    "notes": "Used for reasoning-heavy turns. Tool-enabled flows automatically fall back to the chat model when tool calls are needed.",
                }
            )

        rag_scope_options = [
            {
                "id": "__all__",
                "label": "All connected collections",
                "kind": "all",
            }
        ]
        rag_available = any(
            tool.get("name") == "search_knowledge_base" and tool.get("enabled", False)
            for tool in self.tool_registry.list_tool_specs()
        )
        rag_unavailable_reason = None
        if not rag_available:
            rag_unavailable_reason = "search_knowledge_base tool is not available; check RAG adapters."
        tool_specs = self.tool_registry.list_tool_specs()
        for item in collections:
            if not isinstance(item, dict):
                continue
            collection_name = str(item.get("collection_name") or "").strip()
            if not collection_name:
                continue
            rag_scope_options.append(
                {
                    "id": collection_name,
                    "label": collection_name,
                    "kind": "collection",
                }
            )

        return {
            "status": "success",
            "data": {
                "models": model_options,
                "settings_schema": {
                    "temperature": {"min": 0.0, "max": 2.0, "default": 0.7, "step": 0.1},
                    "top_p": {"min": 0.0, "max": 1.0, "default": 1.0, "step": 0.05},
                    "history_rounds": {"min": 1, "max": 30, "default": 10, "step": 1},
                    "use_user_memory": {"default": False},
                },
                "modes": [
                    {
                        "id": "conversation",
                        "title": "Unified Conversation",
                        "description": "Single entrypoint with structured routing to direct chat, knowledge-grounded RAG, agent execution, or clarification.",
                        "endpoint": "/api/conversation/stream",
                        "request_fields": [
                            "session_id",
                            "query",
                            "collection_name",
                            "model_name",
                            "temperature",
                            "top_p",
                            "history_rounds",
                            "use_user_memory",
                        ],
                        "recommended": True,
                    },
                ],
                "knowledge_base": {
                    "management_default_collection": "tech_docs_kb",
                    "rag_default_scope": "__all__",
                    "agent_default_scope": "__all__",
                    "rag_available": rag_available,
                    "rag_unavailable_reason": rag_unavailable_reason,
                    "rag_scope_options": rag_scope_options,
                    "collections": collections,
                    "notes": [
                        "Knowledge Q&A mode can target all connected collections or one selected collection.",
                        "Agent retrieval always searches across all connected collections by default.",
                    ],
                },
                "capabilities": [
                    item.model_dump()
                    for item in self._build_capabilities(tool_specs, rag_available)
                ],
                "tool_groups": [
                    item.model_dump()
                    for item in self._build_tool_groups(tool_specs, rag_available)
                ],
                "tools": tool_specs,
                "skills": self.skill_registry.manifest() if self.skill_registry else [],
            },
        }

    def get_diagnostics(self, refresh: bool = False) -> dict[str, Any]:
        if self.diagnostics_provider is None:
            return {
                "status": "error",
                "message": "Runtime diagnostics provider is not configured.",
                "data": {},
            }
        return self.diagnostics_provider(refresh)

    def list_recent_traces(self, limit: int = 50) -> dict[str, Any]:
        if self.trace_store is None:
            return {
                "status": "error",
                "message": "Trace store is not configured.",
                "data": [],
            }
        events = self.trace_store.list_recent(limit=limit)
        return {
            "status": "success",
            "data": [event.model_dump() for event in events],
        }

    def list_trace_events(self, trace_id: str) -> dict[str, Any]:
        if self.trace_store is None:
            return {
                "status": "error",
                "message": "Trace store is not configured.",
                "data": [],
            }
        events = self.trace_store.list_by_trace_id(trace_id)
        return {
            "status": "success",
            "data": [event.model_dump() for event in events],
        }

    def list_pending_actions(self, status: str | None = "pending") -> dict[str, Any]:
        if self.approval_store is None:
            return {
                "status": "error",
                "message": "Approval store is not configured.",
                "data": [],
            }
        actions = self.approval_store.list(status=status or None)
        return {
            "status": "success",
            "data": [action.to_public_dict() for action in actions],
        }

    def approve_pending_action(
        self,
        action_id: str,
        decided_by: str | None = None,
        reason: str | None = None,
    ) -> dict[str, Any]:
        if self.approval_store is None:
            return {
                "status": "error",
                "message": "Approval store is not configured.",
                "data": {},
            }
        try:
            self.approval_store.approve(
                action_id,
                decided_by=decided_by,
                reason=reason,
            )
            result = self.tool_registry.execute_approved(action_id)
            action = self.approval_store.get(action_id)
        except Exception as exc:
            return {
                "status": "error",
                "message": str(exc),
                "data": {"action_id": action_id},
            }
        return {
            "status": "success" if result.success else "error",
            "message": result.error,
            "data": {
                "action": action.to_public_dict() if action else None,
                "tool_result": result.model_dump(),
            },
        }

    def reject_pending_action(
        self,
        action_id: str,
        decided_by: str | None = None,
        reason: str | None = None,
    ) -> dict[str, Any]:
        if self.approval_store is None:
            return {
                "status": "error",
                "message": "Approval store is not configured.",
                "data": {},
            }
        try:
            action = self.approval_store.reject(
                action_id,
                decided_by=decided_by,
                reason=reason,
            )
        except Exception as exc:
            return {
                "status": "error",
                "message": str(exc),
                "data": {"action_id": action_id},
            }
        return {
            "status": "success",
            "data": action.to_public_dict(),
        }

    def _build_capabilities(
        self,
        tool_specs: list[dict[str, Any]],
        rag_available: bool,
    ) -> list[CapabilitySpec]:
        tool_names = {str(item.get("name")) for item in tool_specs}
        return [
            CapabilitySpec(
                name="knowledge_search",
                description="Knowledge-base search and catalog inspection.",
                risk_level="low",
                enabled_by_default=rag_available
                or bool({"list_knowledge_base_collections", "list_knowledge_base_files"} & tool_names),
                metadata={"source": "deepassist"},
            ),
            CapabilitySpec(
                name="file_read",
                description="Read-only access to the configured local sandbox.",
                risk_level="low",
                enabled_by_default=bool({"read_local_file", "list_sandbox_files"} & tool_names),
                metadata={"scope": "sandbox"},
            ),
            CapabilitySpec(
                name="file_write",
                description="Local sandbox write capability reserved for policy or HITL approval.",
                risk_level="high",
                enabled_by_default=any(
                    item.get("name") == "write_local_file"
                    and item.get("enabled")
                    for item in tool_specs
                ),
                metadata={"scope": "sandbox", "governance": "policy_or_hitl"},
            ),
            CapabilitySpec(
                name="python_execution",
                description="Execute Python code inside the configured local agent sandbox.",
                risk_level="high",
                enabled_by_default=any(
                    item.get("name") == "execute_python_sandbox"
                    and item.get("enabled")
                    for item in tool_specs
                ),
                metadata={"scope": "sandbox", "governance": "hitl_or_dev_override"},
            ),
            CapabilitySpec(
                name="runtime_diagnostics",
                description="Runtime capability and adapter diagnostics.",
                risk_level="low",
                enabled_by_default=True,
                metadata={"source": "runtime_application"},
            ),
            CapabilitySpec(
                name="external_mcp_tools",
                description="Externally supplied MCP tools imported through ToolProviderPort.",
                risk_level="policy_managed",
                enabled_by_default=any(item.get("provider") == "mcp" for item in tool_specs),
                metadata={"source": "mcp_tool_client_adapter"},
            ),
            CapabilitySpec(
                name="hitl_approval",
                description="Human approval workflow for high-risk tool calls.",
                risk_level="governance",
                enabled_by_default=self.approval_store is not None,
                metadata={"source": "local_approval_store"},
            ),
        ]

    def _build_tool_groups(
        self,
        tool_specs: list[dict[str, Any]],
        rag_available: bool,
    ) -> list[ToolGroupSpec]:
        tool_names = {str(item.get("name")) for item in tool_specs}

        def existing(names: list[str]) -> list[str]:
            return [name for name in names if name in tool_names]

        groups = [
            ToolGroupSpec(
                name="knowledge_search",
                description="Search and inspect connected knowledge bases.",
                tools=existing(
                    [
                        "search_knowledge_base",
                        "list_knowledge_base_collections",
                        "list_knowledge_base_files",
                    ]
                ),
                policy="read_only",
                capability="knowledge_search",
                enabled_by_default=rag_available,
            ),
            ToolGroupSpec(
                name="file_read",
                description="Inspect files inside the configured sandbox.",
                tools=existing(["read_local_file", "list_sandbox_files"]),
                policy="read_only",
                capability="file_read",
                enabled_by_default=bool({"read_local_file", "list_sandbox_files"} & tool_names),
            ),
            ToolGroupSpec(
                name="file_write_legacy",
                description="Legacy local write tool kept behind policy for future HITL governance.",
                tools=existing(["write_local_file"]),
                policy="high_risk",
                capability="file_write",
                enabled_by_default="write_local_file" in tool_names,
            ),
            ToolGroupSpec(
                name="python_execution",
                description="Run local Python snippets inside the configured sandbox.",
                tools=existing(["execute_python_sandbox"]),
                policy="high_risk_hitl",
                capability="python_execution",
                enabled_by_default="execute_python_sandbox" in tool_names,
            ),
        ]

        mcp_tools = [
            str(item.get("name"))
            for item in tool_specs
            if item.get("provider") == "mcp" and item.get("name")
        ]
        if mcp_tools:
            groups.append(
                ToolGroupSpec(
                    name="external_mcp_tools",
                    description="Tools imported from external MCP servers.",
                    tools=sorted(mcp_tools),
                    policy="policy_managed",
                    capability="external_mcp_tools",
                    enabled_by_default=True,
                )
            )
        return groups

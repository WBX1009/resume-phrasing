import inspect
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from difflib import get_close_matches
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, get_origin

from backend.domain.entities.tooling import ToolCall, ToolResult
from backend.domain.interfaces.approval_store import ApprovalStorePort
from backend.domain.interfaces.tool_provider import ToolProviderPort


@dataclass(frozen=True)
class ToolSpec:
    """Registered tool contract exposed to the LLM."""

    name: str
    description: str
    parameters: Dict[str, Any]
    handler: Callable[..., Any]
    signature: inspect.Signature
    provider: Optional[ToolProviderPort] = None

    def to_openai_schema(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


@dataclass
class ToolPolicy:
    """Small governance layer for tool execution."""

    allowed_tools: Optional[Set[str]] = None
    allowed_categories: Optional[Set[str]] = None
    tool_categories: Optional[Dict[str, str]] = None
    max_result_chars: int = 4000
    # A positive value bounds synchronous local/MCP adapter calls. ``None`` or
    # a non-positive value disables the wrapper for deployments with their own
    # deadline mechanism.
    tool_timeout_seconds: Optional[float] = 30.0
    approval_required_categories: Set[str] = field(
        default_factory=lambda: {
            "write_file",
            "code_exec",
            "db_write",
            "external_post",
            "permission_change",
        }
    )
    approval_mode: str = "hitl"
    approval_store: Optional[ApprovalStorePort] = None
    approval_context: Dict[str, Any] = field(default_factory=dict)

    def can_execute(self, tool_name: str) -> bool:
        # Check specific tool whitelist first
        if self.allowed_tools is not None and tool_name not in self.allowed_tools:
            return False
        # If category-based policy is active, verify category
        if self.allowed_categories is not None:
            category = self.category_for(tool_name)
            if category is None or category not in self.allowed_categories:
                return False
        return True

    def category_for(self, tool_name: str) -> Optional[str]:
        if not self.tool_categories:
            return None
        return self.tool_categories.get(tool_name)

    def blocked_reason(self, tool_name: str) -> Optional[str]:
        if self.can_execute(tool_name):
            return None
        if self.allowed_tools is not None and tool_name not in self.allowed_tools:
            return "tool_not_in_allowed_tools"
        category = self.category_for(tool_name)
        if self.allowed_categories is not None:
            if category is None:
                return "tool_category_unknown"
            if category not in self.allowed_categories:
                return f"category_not_allowed:{category}"
        return "tool_blocked_by_policy"

    def requires_approval(self, tool_name: str) -> bool:
        if self.approval_mode.strip().lower() in {"disabled", "off", "none"}:
            return False
        category = self.category_for(tool_name)
        return bool(category and category in self.approval_required_categories)

    def create_pending_action(self, tool_call: ToolCall) -> Optional[Dict[str, Any]]:
        if self.approval_store is None:
            return None

        category = self.category_for(tool_call.name) or "unknown"
        action = self.approval_store.create_pending_action(
            tool_call=tool_call,
            category=category,
            reason=(
                f"Tool `{tool_call.name}` belongs to high-risk category `{category}` "
                "and requires explicit approval before execution."
            ),
            metadata={"source": "tool_policy", **self.approval_context},
        )
        return action.to_public_dict()

    def enable_hint(self) -> str:
        return (
            "Set ALLOWED_TOOL_CATEGORIES in .env to the minimum required "
            "categories, then restart the backend. Stage-1 core tools are "
            "read-only by default; high-risk tools should be supplied by "
            "governed adapters in later phases."
        )

    def apply_result_limits(self, result: ToolResult) -> ToolResult:
        if len(result.content) <= self.max_result_chars:
            return result

        truncated = result.content[: self.max_result_chars]
        metadata = {
            **result.metadata,
            "truncated": True,
            "original_length": len(result.content),
        }
        return result.model_copy(
            update={
                "content": f"{truncated}\n...[truncated]",
                "metadata": metadata,
            }
        )


class ToolRegistry:
    """Registry that turns Python callables into governed agent tools."""

    def __init__(self, policy: Optional[ToolPolicy] = None):
        self.policy = policy or ToolPolicy()
        self._tools: Dict[str, ToolSpec] = {}

    @classmethod
    def from_callables(
        cls,
        tools: Iterable[Callable[..., Any]],
        policy: Optional[ToolPolicy] = None,
    ) -> "ToolRegistry":
        registry = cls(policy=policy)
        for tool in tools:
            registry.register_callable(tool)
        return registry

    def register_callable(
        self,
        handler: Callable[..., Any],
        name: Optional[str] = None,
        description: Optional[str] = None,
    ) -> None:
        tool_name = name or handler.__name__
        signature = inspect.signature(handler)
        self._tools[tool_name] = ToolSpec(
            name=tool_name,
            description=description or inspect.getdoc(handler) or f"Execute {tool_name}",
            parameters=self._build_parameters_schema(signature),
            handler=handler,
            signature=signature,
        )

    def register_provider(
        self,
        provider: ToolProviderPort,
        category: str = "read_only",
    ) -> None:
        """Register externally provided tools behind the same policy layer."""

        for external_tool in provider.list_tools():
            tool_name = str(external_tool.get("name") or "").strip()
            if not tool_name:
                continue
            self._tools[tool_name] = ToolSpec(
                name=tool_name,
                description=str(external_tool.get("description") or f"Execute {tool_name}"),
                parameters=self._normalize_external_parameters(external_tool),
                handler=self._external_provider_handler,
                signature=inspect.signature(self._external_provider_handler),
                provider=provider,
            )
            if self.policy.tool_categories is None:
                self.policy.tool_categories = {}
            self.policy.tool_categories.setdefault(tool_name, category)

    def openai_schemas(self) -> List[Dict[str, Any]]:
        return [
            tool.to_openai_schema()
            for tool in self._tools.values()
            if self.policy.can_execute(tool.name)
        ]

    def describe_tools(self, include_disabled: bool = True) -> str:
        if not self._tools:
            return "No tools are currently registered."

        lines: List[str] = []
        for spec in sorted(self._tools.values(), key=lambda item: item.name):
            if not include_disabled and not self.policy.can_execute(spec.name):
                continue
            properties = spec.parameters.get("properties", {})
            required_args = set(spec.parameters.get("required", []))
            rendered_args: List[str] = []
            for name, schema in properties.items():
                json_type = schema.get("type", "string")
                marker = "*" if name in required_args else "?"
                rendered_args.append(f"{name}:{json_type}{marker}")

            args_text = ", ".join(rendered_args) if rendered_args else "no arguments"
            category = self.policy.category_for(spec.name) or "uncategorized"
            enabled = self.policy.can_execute(spec.name)
            state = "enabled" if enabled else f"disabled ({self.policy.blocked_reason(spec.name)})"
            lines.append(
                f"- {spec.name}({args_text}) [{category}, {state}]: {spec.description}"
            )

        return "\n".join(lines) or "No tools are currently enabled by policy."

    def list_tool_specs(self) -> List[Dict[str, Any]]:
        specs: List[Dict[str, Any]] = []
        for spec in sorted(self._tools.values(), key=lambda item: item.name):
            properties = spec.parameters.get("properties", {})
            required_args = list(spec.parameters.get("required", []))
            specs.append(
                {
                    "name": spec.name,
                    "description": spec.description,
                    "category": self.policy.category_for(spec.name),
                    "enabled": self.policy.can_execute(spec.name),
                    "blocked_reason": self.policy.blocked_reason(spec.name),
                    "approval_required": self.policy.requires_approval(spec.name),
                    "provider": self._provider_name(spec),
                    "allowed_categories": sorted(self.policy.allowed_categories)
                    if self.policy.allowed_categories is not None
                    else None,
                    "required_args": required_args,
                    "parameters": [
                        {
                            "name": name,
                            "type": schema.get("type", "string"),
                            "required": name in required_args,
                            "description": schema.get("description", ""),
                        }
                        for name, schema in properties.items()
                    ],
                }
            )
        return specs

    def execute(self, tool_call: ToolCall) -> ToolResult:
        spec = self._tools.get(tool_call.name)
        if not spec:
            return ToolResult.unknown_tool(
                tool_call,
                self._available_tool_names(),
                metadata=self._unknown_tool_metadata(tool_call),
            )

        if not self.policy.can_execute(tool_call.name):
            category = self.policy.category_for(tool_call.name)
            return ToolResult.execution_failed(
                tool_call,
                f"Tool is blocked by policy: {tool_call.name}",
                metadata={
                    "error_type": "tool_blocked",
                    "retryable": False,
                    "repair_strategy": "fallback_answer",
                    "diagnosis": "The requested tool is blocked by execution policy.",
                    "tool_name": tool_call.name,
                    "category": category,
                    "allowed_categories": sorted(self.policy.allowed_categories)
                    if self.policy.allowed_categories is not None
                    else None,
                    "blocked_reason": self.policy.blocked_reason(tool_call.name),
                    "how_to_enable_for_local_demo": self.policy.enable_hint(),
                },
            )

        validation_error = self._validate_arguments(spec, tool_call.args)
        if validation_error:
            return ToolResult.invalid_arguments(
                tool_call,
                validation_error,
                metadata=self._invalid_argument_metadata(spec, tool_call, validation_error),
            )

        if self.policy.requires_approval(tool_call.name):
            pending_action = self.policy.create_pending_action(tool_call)
            category = self.policy.category_for(tool_call.name)
            return ToolResult.execution_failed(
                tool_call,
                "approval required before executing this high-risk tool",
                metadata={
                    "error_type": "approval_required",
                    "retryable": False,
                    "repair_strategy": "wait_for_human_approval",
                    "diagnosis": "The tool is allowed by policy but requires HITL approval.",
                    "tool_name": tool_call.name,
                    "category": category,
                    "pending_action": pending_action,
                    "approval_store_available": pending_action is not None,
                },
            )

        return self.policy.apply_result_limits(self._execute_spec(spec, tool_call))

    def execute_approved(self, action_id: str) -> ToolResult:
        if self.policy.approval_store is None:
            raise RuntimeError("Approval store is not configured.")

        action = self.policy.approval_store.get(action_id)
        if action is None:
            raise KeyError(f"Unknown pending action: {action_id}")
        if action.status.value != "approved":
            raise ValueError(f"Pending action {action_id} must be approved before execution.")

        spec = self._tools.get(action.tool_call.name)
        if not spec:
            result = ToolResult.unknown_tool(
                action.tool_call,
                self._available_tool_names(),
                metadata=self._unknown_tool_metadata(action.tool_call),
            )
            self.policy.approval_store.complete(action_id, result.model_dump(), failed=True)
            return result

        validation_error = self._validate_arguments(spec, action.tool_call.args)
        if validation_error:
            result = ToolResult.invalid_arguments(
                action.tool_call,
                validation_error,
                metadata=self._invalid_argument_metadata(
                    spec,
                    action.tool_call,
                    validation_error,
                ),
            )
            self.policy.approval_store.complete(action_id, result.model_dump(), failed=True)
            return result

        result = self.policy.apply_result_limits(
            self._execute_spec(spec, action.tool_call)
        )
        self.policy.approval_store.complete(
            action_id,
            result.model_dump(),
            failed=not result.success,
        )
        return result

    def _execute_spec(self, spec: ToolSpec, tool_call: ToolCall) -> ToolResult:
        timeout = self.policy.tool_timeout_seconds
        if timeout is None or float(timeout) <= 0:
            return self._execute_spec_once(spec, tool_call)

        executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="deepassist-tool")
        future = executor.submit(self._execute_spec_once, spec, tool_call)
        try:
            return future.result(timeout=float(timeout))
        except FutureTimeoutError:
            future.cancel()
            return ToolResult.execution_failed(
                tool_call,
                f"Tool execution exceeded the {float(timeout):g}s timeout.",
                metadata={
                    "error_type": "tool_timeout",
                    "retryable": True,
                    "repair_strategy": "retry_or_fallback",
                    "diagnosis": "The tool exceeded the configured execution deadline.",
                    "timeout_seconds": float(timeout),
                    "tool_name": spec.name,
                    "allowed_tools": sorted(self._tools),
                },
            )
        finally:
            # A running Python function cannot be force-killed safely. Cancel
            # pending work and avoid making the request wait for a timed-out
            # adapter; production adapters should enforce their own deadline.
            executor.shutdown(wait=False, cancel_futures=True)

    def _execute_spec_once(self, spec: ToolSpec, tool_call: ToolCall) -> ToolResult:
        try:
            if spec.provider is not None:
                result = spec.provider.execute(tool_call)
            else:
                raw_result = spec.handler(**tool_call.args)
                if isinstance(raw_result, ToolResult):
                    result = raw_result
                else:
                    result = ToolResult.ok(tool_call, str(raw_result))
        except Exception as exc:
            result = ToolResult.execution_failed(
                tool_call,
                str(exc),
                metadata=self._execution_failure_metadata(spec, tool_call, exc),
            )

        return result

    def _available_tool_names(self) -> str:
        return ", ".join(sorted(self._tools)) or "<none>"

    def _provider_name(self, spec: ToolSpec) -> Optional[str]:
        if spec.provider is None:
            return "local"
        name = getattr(spec.provider, "server_name", None)
        if name:
            return "mcp"
        return type(spec.provider).__name__

    def _validate_arguments(self, spec: ToolSpec, args: Dict[str, Any]) -> Optional[str]:
        if spec.provider is not None:
            return self._validate_provider_arguments(spec, args)

        try:
            spec.signature.bind(**args)
        except TypeError as exc:
            return str(exc)

        errors: List[str] = []
        for name, value in args.items():
            parameter = spec.signature.parameters.get(name)
            if parameter is None:
                continue

            expected = parameter.annotation
            if expected == inspect.Parameter.empty:
                continue

            if not self._matches_annotation(value, expected):
                expected_name = getattr(expected, "__name__", str(expected))
                errors.append(
                    f"{name} expected {expected_name}, got {type(value).__name__}"
                )

        return "; ".join(errors) if errors else None

    def _build_parameters_schema(self, signature: inspect.Signature) -> Dict[str, Any]:
        properties: Dict[str, Any] = {}
        required: List[str] = []

        for name, parameter in signature.parameters.items():
            properties[name] = self._parameter_to_schema(name, parameter)
            if parameter.default == inspect.Parameter.empty:
                required.append(name)

        return {"type": "object", "properties": properties, "required": required}

    def _parameter_to_schema(self, name: str, parameter: inspect.Parameter) -> Dict[str, Any]:
        return {
            "type": self._json_type_for_annotation(parameter.annotation),
            "description": f"Parameter {name}",
        }

    def _json_type_for_annotation(self, annotation: Any) -> str:
        origin = get_origin(annotation)
        if annotation == inspect.Parameter.empty:
            return "string"
        if annotation is bool:
            return "boolean"
        if annotation is int:
            return "integer"
        if annotation is float:
            return "number"
        if annotation is str:
            return "string"
        if annotation is list or origin is list:
            return "array"
        if annotation is dict or origin is dict:
            return "object"
        return "string"

    def _normalize_external_parameters(self, external_tool: Dict[str, Any]) -> Dict[str, Any]:
        parameters = (
            external_tool.get("parameters")
            or external_tool.get("inputSchema")
            or external_tool.get("input_schema")
            or {}
        )
        if not isinstance(parameters, dict):
            parameters = {}
        if parameters.get("type") != "object":
            parameters = {
                "type": "object",
                "properties": parameters.get("properties", {})
                if isinstance(parameters.get("properties"), dict)
                else {},
                "required": parameters.get("required", [])
                if isinstance(parameters.get("required"), list)
                else [],
            }
        parameters.setdefault("properties", {})
        parameters.setdefault("required", [])
        return parameters

    def _validate_provider_arguments(self, spec: ToolSpec, args: Dict[str, Any]) -> Optional[str]:
        properties = spec.parameters.get("properties", {})
        required = spec.parameters.get("required", [])
        errors: List[str] = []

        for name in required:
            if name not in args:
                errors.append(f"missing required argument: {name}")

        for name, value in args.items():
            schema = properties.get(name)
            if not isinstance(schema, dict):
                continue
            expected = schema.get("type")
            if expected and not self._matches_json_type(value, expected):
                errors.append(f"{name} expected {expected}, got {type(value).__name__}")

        return "; ".join(errors) if errors else None

    def _matches_json_type(self, value: Any, json_type: str) -> bool:
        if json_type == "string":
            return isinstance(value, str)
        if json_type == "boolean":
            return isinstance(value, bool)
        if json_type == "integer":
            return isinstance(value, int) and not isinstance(value, bool)
        if json_type == "number":
            return isinstance(value, (int, float)) and not isinstance(value, bool)
        if json_type == "array":
            return isinstance(value, list)
        if json_type == "object":
            return isinstance(value, dict)
        return True

    def _external_provider_handler(self, **kwargs: Any) -> Any:
        raise RuntimeError("External provider tools must be executed through ToolProviderPort.")

    def _matches_annotation(self, value: Any, annotation: Any) -> bool:
        origin = get_origin(annotation)
        if annotation is bool:
            return isinstance(value, bool)
        if annotation is int:
            return isinstance(value, int) and not isinstance(value, bool)
        if annotation is float:
            return isinstance(value, (int, float)) and not isinstance(value, bool)
        if annotation is str:
            return isinstance(value, str)
        if annotation is list or origin is list:
            return isinstance(value, list)
        if annotation is dict or origin is dict:
            return isinstance(value, dict)
        return True

    def _unknown_tool_metadata(self, tool_call: ToolCall) -> Dict[str, Any]:
        tool_names = sorted(self._tools)
        matches = get_close_matches(tool_call.name, tool_names, n=1, cutoff=0.5)
        metadata: Dict[str, Any] = {
            "error_type": "unknown_tool",
            "retryable": bool(matches),
            "repair_strategy": "switch_tool" if matches else "fallback_answer",
            "available_tools": tool_names,
            "diagnosis": "The tool name does not match any registered tool.",
        }
        if matches:
            metadata["suggested_tool"] = matches[0]
        return metadata

    def _invalid_argument_metadata(
        self,
        spec: ToolSpec,
        tool_call: ToolCall,
        validation_error: str,
    ) -> Dict[str, Any]:
        if spec.provider is not None:
            required_args = list(spec.parameters.get("required", []))
            expected_args = set(spec.parameters.get("properties", {}))
        else:
            required_args = [
                name
                for name, parameter in spec.signature.parameters.items()
                if parameter.default == inspect.Parameter.empty
            ]
            expected_args = set(spec.signature.parameters)
        provided_args = set(tool_call.args)
        missing_required_args = [name for name in required_args if name not in provided_args]
        unexpected_args = sorted(provided_args - expected_args)
        argument_type_errors = self._argument_type_errors(spec, tool_call.args)
        return {
            "error_type": "invalid_arguments",
            "retryable": True,
            "repair_strategy": "fix_arguments",
            "diagnosis": validation_error,
            "required_args": required_args,
            "missing_required_args": missing_required_args,
            "unexpected_args": unexpected_args,
            "argument_type_errors": argument_type_errors,
            "allowed_tools": sorted(self._tools),
        }

    def _execution_failure_metadata(
        self,
        spec: ToolSpec,
        tool_call: ToolCall,
        exc: Exception,
    ) -> Dict[str, Any]:
        error_text = str(exc).lower()
        diagnosis = "The tool raised an exception during execution."
        repair_strategy = "adjust_arguments_or_fallback"
        retryable = True

        if "missing" in error_text or "not found" in error_text:
            diagnosis = "The tool could not find the requested target or input."
            repair_strategy = "adjust_arguments"
        elif "permission" in error_text or "denied" in error_text:
            diagnosis = "The tool lacks permission to access the requested resource."
            repair_strategy = "fallback_answer"
            retryable = False
        elif "timeout" in error_text:
            diagnosis = "The tool timed out before producing a result."
            repair_strategy = "retry_or_fallback"
        elif "json" in error_text or "decode" in error_text:
            diagnosis = "The tool received malformed structured input."
            repair_strategy = "fix_arguments"

        return {
            "error_type": "tool_execution_failed",
            "retryable": retryable,
            "repair_strategy": repair_strategy,
            "diagnosis": diagnosis,
            "exception_type": type(exc).__name__,
            "tool_name": spec.name,
            "allowed_tools": sorted(self._tools),
            "provided_args": {key: type(value).__name__ for key, value in tool_call.args.items()},
        }

    def _argument_type_errors(self, spec: ToolSpec, args: Dict[str, Any]) -> List[str]:
        issues: List[str] = []
        if spec.provider is not None:
            properties = spec.parameters.get("properties", {})
            for name, value in args.items():
                schema = properties.get(name)
                if not isinstance(schema, dict):
                    continue
                expected = schema.get("type")
                if expected and not self._matches_json_type(value, expected):
                    issues.append(f"{name} expected {expected}, got {type(value).__name__}")
            return issues

        for name, value in args.items():
            parameter = spec.signature.parameters.get(name)
            if parameter is None:
                continue
            expected = parameter.annotation
            if expected == inspect.Parameter.empty:
                continue
            if self._matches_annotation(value, expected):
                continue
            expected_name = getattr(expected, "__name__", str(expected))
            issues.append(f"{name} expected {expected_name}, got {type(value).__name__}")
        return issues

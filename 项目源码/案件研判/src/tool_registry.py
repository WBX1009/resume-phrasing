from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ParamSpec:
    name: str
    kind: str
    required: bool = True
    max_length: int = 128
    max_items: int = 8


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    params: tuple[ParamSpec, ...]
    model_kinds: tuple[str, ...] = ()

    @property
    def required_params(self) -> tuple[str, ...]:
        return tuple(item.name for item in self.params if item.required)

    @property
    def optional_params(self) -> tuple[str, ...]:
        return tuple(item.name for item in self.params if not item.required)


MODEL_ID = ParamSpec("model_id", "int")
ENTITY_NAME = ParamSpec("entity_name", "str")
SOURCE = ParamSpec("source", "str")
TARGET = ParamSpec("target", "str", required=False)
MODEL_IDS = ParamSpec("model_ids", "int_list", required=False, max_items=8)


TOOL_REGISTRY: dict[str, ToolSpec] = {
    "entity_model_slice": ToolSpec(
        name="entity_model_slice",
        description="Recompute a focused profile for one entity within one model.",
        params=(MODEL_ID, ENTITY_NAME),
        model_kinds=("entity", "relation"),
    ),
    "relation_neighbor_subgraph": ToolSpec(
        name="relation_neighbor_subgraph",
        description="Summarize incoming/outgoing relation edges around one entity.",
        params=(MODEL_ID, ENTITY_NAME),
        model_kinds=("relation",),
    ),
    "fund_flow_path_probe": ToolSpec(
        name="fund_flow_path_probe",
        description="Search bounded two-hop relation paths between source and target candidates.",
        params=(MODEL_ID, SOURCE, TARGET),
        model_kinds=("relation",),
    ),
    "same_amount_window_probe": ToolSpec(
        name="same_amount_window_probe",
        description="Summarize same-day same-amount transfer patterns for one entity.",
        params=(MODEL_ID, ENTITY_NAME),
        model_kinds=("entity",),
    ),
    "counterparty_overlap_probe": ToolSpec(
        name="counterparty_overlap_probe",
        description="Compare counterparties for one entity across available model profiles.",
        params=(ENTITY_NAME, MODEL_IDS),
        model_kinds=("entity", "relation"),
    ),
    "transfer_structure_analyzer": ToolSpec(
        name="transfer_structure_analyzer",
        description="Analyze model 86 transfer structure with self-transfer, concentration, time and cycle metrics.",
        params=(MODEL_ID, ENTITY_NAME),
        model_kinds=("relation",),
    ),
}


def allowed_tools_for_strategy(strategy: dict[str, Any]) -> list[str]:
    configured = strategy.get("allowed_tools")
    if configured:
        return [name for name in configured if name in TOOL_REGISTRY]
    model_kind = strategy.get("model_kind")
    result = []
    for name, spec in TOOL_REGISTRY.items():
        if not spec.model_kinds or model_kind in spec.model_kinds:
            result.append(name)
    return result


def normalize_tool_params(tool_name: str, params: dict[str, Any]) -> dict[str, Any]:
    spec = TOOL_REGISTRY.get(tool_name)
    if spec is None:
        raise ValueError(f"unknown tool: {tool_name}")
    normalized: dict[str, Any] = {}
    for param in spec.params:
        if param.name not in params:
            continue
        value = params[param.name]
        if param.kind == "int":
            if isinstance(value, bool):
                raise ValueError(f"{param.name} must be an integer")
            try:
                normalized[param.name] = int(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{param.name} must be an integer") from exc
        elif param.kind == "str":
            if not isinstance(value, str):
                raise ValueError(f"{param.name} must be a string")
            normalized[param.name] = value.strip()
        elif param.kind == "int_list":
            if not isinstance(value, (list, tuple)):
                raise ValueError(f"{param.name} must be an integer list")
            values: list[int] = []
            for item in value:
                if isinstance(item, bool):
                    raise ValueError(f"{param.name} must be an integer list")
                try:
                    parsed = int(item)
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"{param.name} must be an integer list") from exc
                if parsed not in values:
                    values.append(parsed)
            normalized[param.name] = values
    return normalized


def validate_tool_request(
    request: dict[str, Any],
    strategies_by_model: dict[int, dict[str, Any]],
    allowed_entities: set[str],
) -> tuple[bool, str]:
    tool_name = str(request.get("tool_name") or "")
    spec = TOOL_REGISTRY.get(tool_name)
    if spec is None:
        return False, f"unknown tool: {tool_name}"
    params = request.get("params")
    if not isinstance(params, dict):
        return False, "params must be an object"
    allowed_param_names = {item.name for item in spec.params}
    unknown = sorted(set(params) - allowed_param_names)
    if unknown:
        return False, f"unknown params: {', '.join(unknown)}"
    try:
        normalized = normalize_tool_params(tool_name, params)
    except ValueError as exc:
        return False, str(exc)
    by_name = {item.name: item for item in spec.params}
    for name, param in by_name.items():
        value = normalized.get(name)
        if param.required and value in [None, "", []]:
            return False, f"missing required param: {name}"
        if param.kind == "str" and value is not None and len(value) > param.max_length:
            return False, f"{name} exceeds maximum length {param.max_length}"
        if param.kind == "int_list" and value is not None and len(value) > param.max_items:
            return False, f"{name} exceeds maximum item count {param.max_items}"

    model_ids = []
    if normalized.get("model_id") is not None:
        model_ids.append(int(normalized["model_id"]))
    model_ids.extend(int(value) for value in normalized.get("model_ids", []))
    for model_id in model_ids:
        strategy = strategies_by_model.get(model_id)
        if strategy is None:
            return False, f"unknown model_id: {model_id}"
        if tool_name != "counterparty_overlap_probe" and tool_name not in allowed_tools_for_strategy(strategy):
            return False, f"tool {tool_name} is not allowed for model {model_id}"

    for name_key in ("entity_name", "source", "target"):
        value = normalized.get(name_key)
        if value and str(value) not in allowed_entities:
            return False, f"{name_key} is not in known FactPack entities: {value}"
    request["params"] = normalized
    return True, ""

"""Snapshot-backed model catalog and strategy profile registry."""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from .quality_models import ModelProfile, ReasoningMode


DEFAULT_DATA_DIR = Path(__file__).resolve().parents[1] / "表格数据抽样_ josnl 版"

# Every sampled model has a deliberate family assignment.  The mapping is
# intentionally explicit: a future model cannot silently fall into a generic
# prompt just because its name resembles an existing model.
MODEL_FAMILIES: dict[int, str] = {
    5: "account_profile",
    98: "account_profile",
    10: "communication_temporal",
    15: "communication_temporal",
    16: "communication_temporal",
    23: "communication_temporal",
    24: "communication_temporal",
    25: "communication_temporal",
    14: "contact_device_relation",
    18: "contact_device_relation",
    19: "contact_device_relation",
    26: "contact_device_relation",
    27: "distribution_keyword",
    28: "distribution_keyword",
    30: "distribution_keyword",
    31: "distribution_keyword",
    32: "distribution_keyword",
    33: "distribution_keyword",
    34: "distribution_keyword",
    35: "distribution_keyword",
    36: "distribution_keyword",
    37: "distribution_keyword",
    38: "distribution_keyword",
    39: "distribution_keyword",
    40: "distribution_keyword",
    41: "distribution_keyword",
    45: "distribution_keyword",
    61: "distribution_keyword",
    65: "distribution_keyword",
    29: "threshold_event",
    42: "threshold_event",
    43: "threshold_event",
    46: "threshold_event",
    49: "threshold_event",
    50: "threshold_event",
    51: "threshold_event",
    52: "threshold_event",
    57: "threshold_event",
    68: "threshold_event",
    75: "threshold_event",
    76: "threshold_event",
    44: "temporal_sequence",
    54: "temporal_sequence",
    55: "temporal_sequence",
    56: "temporal_sequence",
    58: "temporal_sequence",
    59: "temporal_sequence",
    62: "temporal_sequence",
    69: "temporal_sequence",
    70: "temporal_sequence",
    47: "graph_cross_domain",
    48: "graph_cross_domain",
    53: "graph_cross_domain",
    63: "graph_cross_domain",
    64: "graph_cross_domain",
    66: "graph_cross_domain",
    67: "graph_cross_domain",
    71: "graph_cross_domain",
    86: "graph_cross_domain",
    87: "graph_cross_domain",
    88: "graph_cross_domain",
}

FAMILY_OPERATORS: dict[str, tuple[str, ...]] = {
    "account_profile": ("row_count", "entity_cardinality", "numeric_summary"),
    "communication_temporal": (
        "row_count",
        "entity_cardinality",
        "temporal_summary",
        "category_summary",
    ),
    "contact_device_relation": (
        "row_count",
        "entity_cardinality",
        "set_overlap",
        "relation_summary",
    ),
    "distribution_keyword": (
        "row_count",
        "category_summary",
        "numeric_summary",
        "concentration_summary",
    ),
    "threshold_event": (
        "row_count",
        "numeric_summary",
        "temporal_summary",
        "event_summary",
    ),
    "temporal_sequence": (
        "row_count",
        "temporal_summary",
        "sequence_summary",
        "pair_match_summary",
    ),
    "graph_cross_domain": (
        "row_count",
        "entity_cardinality",
        "relation_summary",
        "graph_summary",
    ),
}

FAMILY_TO_TOOLS: dict[str, tuple[str, ...]] = {
    "account_profile": (),
    "communication_temporal": ("temporal_window_summary", "top_entities"),
    "contact_device_relation": ("overlap_summary", "top_entities"),
    "distribution_keyword": (),
    "threshold_event": ("threshold_event_summary",),
    "temporal_sequence": (
        "temporal_window_summary",
        "same_amount_window_summary",
    ),
    "graph_cross_domain": ("edge_summary", "graph_neighbor_summary"),
}

FAMILY_REASONING: dict[str, ReasoningMode] = {
    "account_profile": "deterministic",
    "communication_temporal": "react",
    "contact_device_relation": "react",
    "distribution_keyword": "deterministic",
    "threshold_event": "react",
    "temporal_sequence": "plan_solve",
    "graph_cross_domain": "plan_react_reflect",
}

FAMILY_REPORT_SECTIONS: dict[str, tuple[str, ...]] = {
    "account_profile": ("entity_profile", "numeric_facts"),
    "communication_temporal": ("communication_pattern", "time_distribution", "evidence_table"),
    "contact_device_relation": ("relation_overlap", "entity_ranking", "evidence_table"),
    "distribution_keyword": ("distribution", "category_breakdown", "evidence_table"),
    "threshold_event": ("threshold_hits", "time_distribution", "evidence_table"),
    "temporal_sequence": ("sequence_pattern", "matched_events", "evidence_table"),
    "graph_cross_domain": ("relation_graph", "cross_model_links", "evidence_table"),
}

MODEL_DEPENDENCIES: dict[int, tuple[int, ...]] = {
    5: (98,),
    10: (98,),
    14: (88,),
    18: (88,),
    19: (88,),
    26: (88,),
    27: (98,),
    28: (98,),
    42: (5, 98),
    43: (5, 98),
    46: (5, 98),
    47: (5, 48, 86),
    48: (5, 86),
    53: (98,),
    54: (5,),
    55: (5, 46),
    56: (5, 43),
    58: (5, 46),
    59: (5, 46),
    61: (5,),
    62: (5, 46),
    63: (5,),
    64: (5,),
    65: (5,),
    66: (47, 48, 86),
    67: (5, 46),
    68: (5, 98),
    69: (5, 98),
    70: (5, 46),
    71: (46, 88),
    75: (5, 43),
    76: (5, 43),
    86: (5, 46),
    87: (5, 46),
    88: (10, 14),
}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                decoded = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc
            if not isinstance(decoded, dict):
                raise ValueError(f"Expected JSON object at {path}:{line_number}")
            rows.append(decoded)
    return rows


def load_catalog(data_dir: Path | str = DEFAULT_DATA_DIR) -> list[dict[str, Any]]:
    path = Path(data_dir) / "ch_model_info.jsonl"
    return _read_jsonl(path)


def _decode_json_data(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if value is None:
        return {}
    if isinstance(value, str):
        try:
            decoded = json.loads(value)
        except json.JSONDecodeError:
            return {}
        return decoded if isinstance(decoded, dict) else {}
    return {}


def load_sample_rows(data_dir: Path | str = DEFAULT_DATA_DIR) -> list[dict[str, Any]]:
    path = Path(data_dir) / "ch_model_info_data_sample.jsonl"
    rows = _read_jsonl(path)
    for row_number, row in enumerate(rows, start=1):
        try:
            row["row_id"] = int(row.get("id", 0))
            row["model_id"] = int(row.get("model_id", 0))
            row["case_id"] = int(row.get("case_id", 0))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid integer key at {path}:{row_number}") from exc
        row["data"] = _decode_json_data(row.get("json_data"))
    return rows


def _required_field_hints(name: str, description: str) -> tuple[str, ...]:
    # These are semantic hints, not hard requirements.  Runtime completeness
    # is evaluated against the actual snapshot and never guessed by an LLM.
    text = f"{name} {description}"
    hints: list[str] = []
    for token in ("姓名", "账号", "卡号", "手机号", "交易时间", "交易日期", "交易金额", "金额", "时间"):
        if token in text:
            hints.append(token)
    return tuple(dict.fromkeys(hints))


def _slug(text: str) -> str:
    text = re.sub(r"[^0-9A-Za-z\u4e00-\u9fff]+", "_", text).strip("_")
    return text[:28] or "model"


def build_profiles(
    catalog: Iterable[dict[str, Any]],
    sample_rows: Iterable[dict[str, Any]],
) -> list[ModelProfile]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in sample_rows:
        grouped[int(row["model_id"])].append(row)

    profiles: list[ModelProfile] = []
    for item in catalog:
        model_id = int(item["id"])
        rows = grouped.get(model_id, [])
        family = MODEL_FAMILIES.get(model_id)
        if rows and family is None:
            raise ValueError(f"Sampled model {model_id} has no explicit family mapping")
        if family is None:
            family = "unsupported"
        fields = sorted({str(key) for row in rows for key in row.get("data", {})})
        case_ids = sorted({int(row["case_id"]) for row in rows})
        coverage = "sampled" if rows else "unsupported_no_sample"
        profiles.append(
            ModelProfile(
                model_id=model_id,
                name=str(item.get("model_name") or f"model-{model_id}"),
                description=str(item.get("model_desc") or ""),
                family=family,
                coverage=coverage,
                reasoning_mode=(FAMILY_REASONING.get(family, "deterministic") if rows else "deterministic"),
                observed_fields=tuple(fields),
                observed_row_count=len(rows),
                observed_case_ids=tuple(case_ids),
                operators=FAMILY_OPERATORS.get(family, ()),
                specialized_tools=FAMILY_TO_TOOLS.get(family, ()),
                required_field_hints=_required_field_hints(
                    str(item.get("model_name") or ""), str(item.get("model_desc") or "")
                ),
                strategy_key=f"m{model_id}_{_slug(str(item.get('model_name') or 'model'))}",
                report_sections=FAMILY_REPORT_SECTIONS.get(family, ("coverage_status",)),
                cross_model_dependencies=MODEL_DEPENDENCIES.get(model_id, ()),
                unsupported_reason=(
                    "当前抽样快照未提供该模型结果，禁止推测性研判。" if not rows else None
                ),
            )
        )
    return sorted(profiles, key=lambda profile: profile.model_id)


def build_registry(data_dir: Path | str = DEFAULT_DATA_DIR) -> tuple[list[ModelProfile], list[dict[str, Any]]]:
    samples = load_sample_rows(data_dir)
    return build_profiles(load_catalog(data_dir), samples), samples


def validate_registry(profiles: Iterable[ModelProfile]) -> dict[str, Any]:
    profiles = list(profiles)
    ids = [profile.model_id for profile in profiles]
    duplicates = sorted({model_id for model_id in ids if ids.count(model_id) > 1})
    sampled = [profile for profile in profiles if profile.coverage == "sampled"]
    unsupported = [profile for profile in profiles if profile.coverage == "unsupported_no_sample"]
    errors: list[str] = []
    if duplicates:
        errors.append(f"duplicate model ids: {duplicates}")
    errors.extend(
        f"sampled model {profile.model_id} is missing a strategy family"
        for profile in sampled
        if profile.family == "unsupported"
    )
    return {
        "valid": not errors,
        "errors": errors,
        "total_models": len(profiles),
        "sampled_models": len(sampled),
        "unsupported_no_sample": len(unsupported),
        "sampled_model_ids": [profile.model_id for profile in sampled],
        "unsupported_model_ids": [profile.model_id for profile in unsupported],
        "families": {
            family: sum(profile.family == family for profile in profiles)
            for family in sorted({profile.family for profile in profiles})
        },
    }

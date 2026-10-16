from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from .case_fact_pack import compact_fact_pack


PROFILE_OPERATOR_TYPES = [
    "field_presence_profile",
    "type_profile",
    "numeric_range_profile",
    "datetime_range_profile",
    "topk_profile",
    "role_inference_profile",
]


def estimate_tokens(text: str) -> int:
    # A rough, provider-agnostic planning estimate for Chinese/JSON mixed prompts.
    return max(1, round(len(text) / 2))


def _compact_field_for_planner(field: dict[str, Any]) -> dict[str, Any]:
    item: dict[str, Any] = {
        "field_name": field.get("field_name"),
        "coverage_rate": field.get("coverage_rate"),
        "parsed_type": field.get("parsed_type"),
        "roles": field.get("roles", []),
    }
    if field.get("topk"):
        item["topk"] = field["topk"][:3]
    if field.get("numeric_range"):
        item["numeric_range"] = field["numeric_range"]
    if field.get("time_range"):
        item["time_range"] = field["time_range"]
    return item


def build_planner_context(model: dict[str, Any], profile: dict[str, Any]) -> dict[str, Any]:
    """Build the future LLM-planner input without raw rows.

    The current pipeline uses deterministic rule planning. This structure is only
    used to estimate how large a future model-level tool-selection prompt would be.
    """
    fields = sorted(
        profile.get("fields", []),
        key=lambda item: (
            -len(item.get("roles", [])),
            -(item.get("coverage_rate") or 0),
            item.get("field_name") or "",
        ),
    )
    return {
        "planner_task": "select_allowed_analysis_operators_and_parameters",
        "allowed_operator_types": [
            "amount_topk",
            "amount_stats",
            "category_topk",
            "entity_frequency",
            "time_range",
            "time_bucket",
            "relation_edges",
            "flow_summary",
            "duration_stats",
            "evidence_samples",
        ],
        "model_context": {
            "case_id": model.get("case_id"),
            "model_id": model.get("model_id"),
            "model_name": model.get("model_name"),
            "category": model.get("category"),
            "result_count": model.get("result_count"),
        },
        "field_profile_summary": [_compact_field_for_planner(field) for field in fields[:40]],
        "selection_constraints": [
            "Only choose operators from allowed_operator_types.",
            "Do not request SQL, Python, or raw full data.",
            "Use field_name values from field_profile_summary as parameters.",
            "Return skipped reason when required field roles are absent.",
        ],
    }


def build_node_metrics(
    fact_pack: dict[str, Any],
    field_profiles: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    compact = compact_fact_pack(fact_pack)
    compact_text = json.dumps(compact, ensure_ascii=False, separators=(",", ":"))

    model_metrics = []
    planner_context_total_chars = 0
    planner_context_total_tokens = 0
    planner_context_max_tokens = 0
    for model in fact_pack["models"]:
        model_id = int(model["model_id"])
        profile = field_profiles.get(model_id, {})
        fields = profile.get("fields", [])
        roles = sorted({role for field in fields for role in field.get("roles", [])})
        operator_results = model.get("operator_results", [])
        successful_ops = [result for result in operator_results if result.get("status") == "success"]
        planner_context = build_planner_context(model, profile)
        planner_context_text = json.dumps(planner_context, ensure_ascii=False, separators=(",", ":"))
        planner_context_chars = len(planner_context_text)
        planner_context_tokens = estimate_tokens(planner_context_text)
        planner_context_total_chars += planner_context_chars
        planner_context_total_tokens += planner_context_tokens
        planner_context_max_tokens = max(planner_context_max_tokens, planner_context_tokens)
        model_metrics.append(
            {
                "model_id": model_id,
                "model_name": model["model_name"],
                "category": model["category"],
                "input_rows": model["result_count"],
                "loaded_rows": model.get("loaded_result_count", model["result_count"]),
                "field_count": len(fields),
                "profile_operator_type_count": len(PROFILE_OPERATOR_TYPES),
                "profile_operator_types": PROFILE_OPERATOR_TYPES,
                "detected_role_count": len(roles),
                "detected_roles": roles,
                "analysis_operator_count": len(operator_results),
                "successful_analysis_operator_count": len(successful_ops),
                "analysis_operators": [result.get("operator") for result in operator_results],
                "fact_count": len(model.get("facts", [])),
                "json_parse_errors": model.get("json_parse_errors", 0),
                "planner_context_chars": planner_context_chars,
                "planner_context_estimated_tokens": planner_context_tokens,
            }
        )

    model_count = len(fact_pack["models"])
    return {
        "case_id": fact_pack["case_id"],
        "data_source": fact_pack["data_source"],
        "profile_operator_types": PROFILE_OPERATOR_TYPES,
        "profile_operator_type_count": len(PROFILE_OPERATOR_TYPES),
        "analysis_operator_types": sorted(
            {
                result.get("operator")
                for model in fact_pack["models"]
                for result in model.get("operator_results", [])
                if result.get("operator")
            }
        ),
        "analysis_operator_type_count": len(
            {
                result.get("operator")
                for model in fact_pack["models"]
                for result in model.get("operator_results", [])
                if result.get("operator")
            }
        ),
        "input_rows": sum(model["result_count"] for model in fact_pack["models"]),
        "model_count": len(fact_pack["models"]),
        "fact_count": sum(len(model.get("facts", [])) for model in fact_pack["models"]),
        "compact_llm_context_chars": len(compact_text),
        "compact_llm_context_estimated_tokens": estimate_tokens(compact_text),
        "compact_llm_fact_count": sum(len(group["facts"]) for group in compact.get("category_fact_groups", [])),
        "planner_context_total_chars": planner_context_total_chars,
        "planner_context_estimated_tokens_total": planner_context_total_tokens,
        "planner_context_estimated_tokens_avg": round(planner_context_total_tokens / model_count, 2) if model_count else 0,
        "planner_context_estimated_tokens_max": planner_context_max_tokens,
        "model_metrics": model_metrics,
    }


def write_node_metrics_csv(path: Path, metrics: dict[str, Any]) -> None:
    rows = metrics["model_metrics"]
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "model_id",
                "model_name",
                "category",
                "input_rows",
                "loaded_rows",
                "field_count",
                "profile_operator_type_count",
                "detected_role_count",
                "detected_roles",
                "analysis_operator_count",
                "successful_analysis_operator_count",
                "analysis_operators",
                "fact_count",
                "json_parse_errors",
                "planner_context_chars",
                "planner_context_estimated_tokens",
            ],
        )
        writer.writeheader()
        for row in rows:
            serialized = {key: row.get(key, "") for key in writer.fieldnames}
            serialized["detected_roles"] = "|".join(row["detected_roles"])
            serialized["analysis_operators"] = "|".join(row["analysis_operators"])
            writer.writerow(serialized)

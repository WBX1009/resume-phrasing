from __future__ import annotations

from collections import defaultdict
from typing import Any

from .operators import run_model_operators


def _field_profile_summary(profile: dict[str, Any]) -> dict[str, list[str]]:
    summary: dict[str, list[str]] = defaultdict(list)
    for field in profile.get("fields", []):
        for role in field.get("roles", []):
            summary[f"{role}_fields"].append(field["field_name"])
    return dict(summary)


def _facts_from_operator_result(model_id: int, result: dict[str, Any]) -> list[dict[str, Any]]:
    if result.get("status") != "success":
        return []
    operator = result["operator"]
    if operator == "numeric_topk":
        return [
            {
                "fact_type": "numeric_topk",
                "description": f"模型 {model_id} 在字段 {result['field']} 上识别到数值 TopK。",
                "metrics": {"field": result["field"], "count": result["count"], "max": result["max"], "min": result["min"]},
                "evidence": result["topk"][:5],
            }
        ]
    if operator in {"category_topk", "entity_frequency"}:
        return [
            {
                "fact_type": operator,
                "description": f"模型 {model_id} 在字段 {result['field']} 上识别到高频取值。",
                "metrics": {"field": result["field"], "topk": result["topk"][:10]},
                "evidence": result["topk"][:5],
            }
        ]
    if operator == "time_bucket":
        return [
            {
                "fact_type": "time_bucket",
                "description": f"模型 {model_id} 在字段 {result['field']} 上完成时间分桶。",
                "metrics": {
                    "field": result["field"],
                    "time_range": result["time_range"],
                    "buckets": result["buckets"],
                },
                "evidence": result["buckets"][:5],
            }
        ]
    if operator == "relation_edges":
        return [
            {
                "fact_type": "relation_edges",
                "description": f"模型 {model_id} 聚合出关系边。",
                "metrics": {
                    "left_field": result["left_field"],
                    "right_field": result["right_field"],
                    "amount_field": result.get("amount_field"),
                    "top_edges": result["top_edges"][:10],
                },
                "evidence": result["top_edges"][:5],
            }
        ]
    if operator == "evidence_samples":
        return [
            {
                "fact_type": "evidence_samples",
                "description": f"模型 {model_id} 抽取了结果样本。",
                "metrics": {"sample_count": len(result["samples"])},
                "evidence": result["samples"],
            }
        ]
    return []


def build_fact_pack(
    case_id: int,
    contexts: list[dict[str, Any]],
    records_by_model: dict[int, list[dict[str, Any]]],
    field_profiles: dict[int, dict[str, Any]],
    data_source: str,
    warnings: list[str],
) -> dict[str, Any]:
    models = []
    for context in contexts:
        model_id = int(context["model_id"])
        profile = field_profiles.get(model_id, {"fields": []})
        operator_run = run_model_operators(model_id, records_by_model.get(model_id, []), profile)
        facts = []
        for result in operator_run["operator_results"]:
            facts.extend(_facts_from_operator_result(model_id, result))
        models.append(
            {
                "case_id": case_id,
                "model_id": model_id,
                "model_name": context["model_name"],
                "category": context["model_category"],
                "result_count": context["result_count"],
                "field_profile_summary": _field_profile_summary(profile),
                "operator_results": operator_run["operator_results"],
                "facts": facts,
                "json_parse_errors": operator_run["json_parse_errors"],
            }
        )

    return {
        "case_id": case_id,
        "data_source": data_source,
        "warnings": warnings,
        "models": models,
    }

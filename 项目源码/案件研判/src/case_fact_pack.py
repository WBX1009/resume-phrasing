from __future__ import annotations

from collections import defaultdict
import csv
from pathlib import Path
from typing import Any

from .case_operators import run_case_operators


def field_profile_summary(profile: dict[str, Any]) -> dict[str, list[str]]:
    summary: dict[str, list[str]] = defaultdict(list)
    for field in profile.get("fields", []):
        for role in field.get("roles", []):
            summary[role].append(field["field_name"])
    return {role: fields[:20] for role, fields in summary.items()}


def fact_description(model: dict[str, Any], result: dict[str, Any]) -> str:
    operator = result["operator"]
    metrics = result.get("metrics", {})
    name = model["model_name"]
    if operator == "amount_topk":
        return f"{name} 在 {metrics.get('field')} 字段上形成金额TopK事实。"
    if operator == "amount_stats":
        return f"{name} 在 {metrics.get('field')} 字段上形成金额统计事实。"
    if operator == "category_topk":
        return f"{name} 在 {metrics.get('field')} 字段上形成类别高频事实。"
    if operator == "entity_frequency":
        return f"{name} 在 {metrics.get('field')} 字段上形成主体高频事实。"
    if operator == "relation_edges":
        return f"{name} 基于 {metrics.get('source_field')} 到 {metrics.get('target_field')} 形成关系边事实。"
    if operator == "flow_summary":
        return f"{name} 基于 {metrics.get('amount_field')} 形成资金流汇总事实。"
    if operator == "time_range":
        return f"{name} 在 {metrics.get('field')} 字段上形成时间范围事实。"
    if operator == "time_bucket":
        return f"{name} 在 {metrics.get('field')} 字段上形成时间分桶事实。"
    if operator == "duration_stats":
        return f"{name} 在 {metrics.get('field')} 字段上形成时间间隔/时长统计事实。"
    if operator == "evidence_samples":
        return f"{name} 抽取了代表性证据样本。"
    return f"{name} 形成 {operator} 事实。"


def build_case_fact_pack(
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
        operator_run = run_case_operators(model_id, records_by_model.get(model_id, []), profile, context)
        facts = []
        model_stub = {"model_name": context["model_name"]}
        for result in operator_run["operator_results"]:
            if result.get("status") != "success":
                continue
            facts.append(
                {
                    "fact_id": result["fact_id"],
                    "fact_type": result["operator"],
                    "description": fact_description(model_stub, result),
                    "metrics": result.get("metrics", {}),
                    "evidence_refs": result.get("evidence_refs", []),
                    "evidence": result.get("evidence", []),
                }
            )
        models.append(
            {
                "case_id": case_id,
                "model_id": model_id,
                "model_name": context["model_name"],
                "category": context["model_category"],
                "result_count": context["result_count"],
                "loaded_result_count": context.get("loaded_result_count", context["result_count"]),
                "field_profile_summary": field_profile_summary(profile),
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


def write_operator_audit(path: Path, fact_pack: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "case_id",
                "model_id",
                "model_name",
                "category",
                "result_count",
                "operator",
                "status",
                "fact_id",
                "reason",
            ],
        )
        writer.writeheader()
        for model in fact_pack["models"]:
            for result in model["operator_results"]:
                writer.writerow(
                    {
                        "case_id": fact_pack["case_id"],
                        "model_id": model["model_id"],
                        "model_name": model["model_name"],
                        "category": model["category"],
                        "result_count": model["result_count"],
                        "operator": result.get("operator"),
                        "status": result.get("status"),
                        "fact_id": result.get("fact_id", ""),
                        "reason": result.get("reason", ""),
                    }
                )


def _brief_metrics(fact: dict[str, Any]) -> dict[str, Any]:
    metrics = fact.get("metrics", {})
    fact_type = fact.get("fact_type")
    if fact_type in {"amount_stats", "duration_stats"}:
        return {key: metrics.get(key) for key in ["field", "count", "sum", "min", "max", "avg"] if key in metrics}
    if fact_type == "amount_topk":
        return {"field": metrics.get("field"), "topk": metrics.get("topk", [])[:3]}
    if fact_type in {"category_topk", "entity_frequency"}:
        return {"field": metrics.get("field"), "topk": metrics.get("topk", [])[:5]}
    if fact_type == "relation_edges":
        return {
            "source_field": metrics.get("source_field"),
            "target_field": metrics.get("target_field"),
            "amount_field": metrics.get("amount_field"),
            "top_edges": metrics.get("top_edges", [])[:5],
        }
    if fact_type == "flow_summary":
        return {
            "amount_field": metrics.get("amount_field"),
            "direction_field": metrics.get("direction_field"),
            "by_direction": metrics.get("by_direction", [])[:5],
        }
    if fact_type == "time_range":
        return metrics
    if fact_type == "time_bucket":
        buckets = metrics.get("buckets", [])
        return {"field": metrics.get("field"), "bucket_count": len(buckets), "first_buckets": buckets[:3], "last_buckets": buckets[-3:]}
    return {key: metrics.get(key) for key in list(metrics.keys())[:5]}


def compact_fact_pack(fact_pack: dict[str, Any], max_facts_per_category: int = 2) -> dict[str, Any]:
    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for model in fact_pack["models"]:
        by_category[model["category"]].append(model)

    category_groups = []
    preferred_fact_types = [
        "relation_edges",
        "flow_summary",
        "amount_stats",
        "amount_topk",
        "entity_frequency",
        "category_topk",
        "time_range",
        "time_bucket",
        "duration_stats",
    ]
    rank = {name: index for index, name in enumerate(preferred_fact_types)}
    for category, models in by_category.items():
        category_facts = []
        sorted_models = sorted(models, key=lambda item: item["result_count"], reverse=True)
        for model in sorted_models:
            facts = [fact for fact in model["facts"] if fact.get("fact_type") != "evidence_samples"]
            facts.sort(key=lambda fact: rank.get(fact.get("fact_type"), 99))
            for fact in facts:
                category_facts.append(
                    {
                        "model_id": model["model_id"],
                        "model_name": model["model_name"],
                        "result_count": model["result_count"],
                        "fact_id": fact["fact_id"],
                        "fact_type": fact["fact_type"],
                        "description": fact["description"],
                        "metrics": _brief_metrics(fact),
                    }
                )
                if len(category_facts) >= max_facts_per_category:
                    break
            if len(category_facts) >= max_facts_per_category:
                break
        category_groups.append(
            {
                "category": category,
                "model_count": len(models),
                "row_count": sum(model["result_count"] for model in models),
                "facts": category_facts,
            }
        )
    return {
        "case_id": fact_pack["case_id"],
        "data_source": fact_pack["data_source"],
        "coverage_summary": {
            "model_count": len(fact_pack["models"]),
            "row_count": sum(model["result_count"] for model in fact_pack["models"]),
            "fact_count": sum(len(model["facts"]) for model in fact_pack["models"]),
        },
        "category_fact_groups": category_groups,
    }

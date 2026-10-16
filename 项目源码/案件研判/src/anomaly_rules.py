from __future__ import annotations

from typing import Any


def _flag(tag: str, rule: str, expression: str, metrics: dict[str, Any], severity: str = "medium") -> dict[str, Any]:
    return {
        "risk_tag": tag,
        "trigger_rule": rule,
        "report_expression": expression,
        "metrics": metrics,
        "severity": severity,
    }


def detect_entity_anomalies(profile: dict[str, Any], strategy: dict[str, Any], thresholds: dict[str, Any]) -> list[dict[str, Any]]:
    metrics = profile.get("metrics", {})
    rules = set(strategy.get("anomaly_rules") or [])
    flags: list[dict[str, Any]] = []

    near_count = int(metrics.get("near_threshold_count") or 0)
    if "near_threshold_amount" in rules and near_count > 0:
        flags.append(
            _flag(
                "near_threshold_amount",
                "amount in configured near-threshold interval",
                f"存在 {near_count} 笔接近大额阈值区间的交易。",
                {"near_threshold_count": near_count},
            )
        )

    night_ratio = float(metrics.get("night_ratio") or 0)
    night_count = int(metrics.get("night_count") or 0)
    if (
        "night_activity" in rules
        and night_count >= int(thresholds.get("min_night_count", 10))
        and night_ratio >= float(thresholds.get("night_ratio", 0.3))
    ):
        flags.append(
            _flag(
                "night_activity",
                "night_count >= min_night_count and night_ratio >= threshold",
                f"夜间交易 {night_count} 笔，占比 {night_ratio:.1%}。",
                {"night_count": night_count, "night_ratio": night_ratio},
            )
        )

    max_hourly = int(metrics.get("max_hourly_count") or 0)
    if "high_frequency_window" in rules and max_hourly >= int(thresholds.get("high_frequency_hour", 5)):
        flags.append(
            _flag(
                "high_frequency_window",
                "max_hourly_count >= threshold",
                f"单小时最高交易 {max_hourly} 笔，存在短时间集中交易特征。",
                {"max_hourly_count": max_hourly},
            )
        )

    counterparty_count = int(metrics.get("distinct_counterparty_count") or 0)
    if "many_counterparties" in rules and counterparty_count >= int(thresholds.get("many_counterparties", 20)):
        flags.append(
            _flag(
                "many_counterparties",
                "distinct_counterparty_count >= threshold",
                f"涉及对手数量 {counterparty_count} 个，关系分散度较高。",
                {"distinct_counterparty_count": counterparty_count},
            )
        )

    same_day_max = float(metrics.get("same_day_same_amount_max") or 0)
    if "same_day_same_amount" in rules and same_day_max >= 2:
        flags.append(
            _flag(
                "same_day_same_amount",
                "same_day_same_amount counter >= 2",
                f"同日相同金额转不同对端数量最高为 {same_day_max:g}。",
                {"same_day_same_amount_max": same_day_max},
            )
        )
    return flags


def detect_relation_anomalies(
    relation: dict[str, Any],
    strategy: dict[str, Any],
    relation_summary: dict[str, Any],
    thresholds: dict[str, Any],
) -> list[dict[str, Any]]:
    rules = set(strategy.get("anomaly_rules") or [])
    flags: list[dict[str, Any]] = []
    if "self_transfer_high_density" in rules and relation.get("self_transfer"):
        total_amount = float(relation_summary.get("self_edge_amount_sum") or 0)
        edge_amount = float(relation.get("amount_sum") or 0)
        ratio = edge_amount / total_amount if total_amount else 0
        if ratio >= float(thresholds.get("self_transfer_ratio", 0.5)):
            flags.append(
                _flag(
                    "self_transfer_high_density",
                    "source == target and self transfer is high-ranked",
                    f"该关系边为同名主体自转账，金额 {edge_amount:.2f}，在自转账边中占比较高。",
                    {"edge_amount_sum": edge_amount, "self_edge_amount_ratio": round(ratio, 4)},
                    "high",
                )
            )
    if (
        "structured_in_scattered_out" in rules
        and strategy.get("model_kind") == "relation"
        and "资金流向" in str(strategy.get("demo_role", ""))
        and relation.get("amount_sum", 0) > 0
    ):
        flags.append(
            _flag(
                "structured_in_scattered_out",
                "fund-flow model relation edge ranked by amount/count",
                "资金流向模型中该关系边金额或次数靠前，可作为整进散出线索候选。",
                {"amount_sum": relation.get("amount_sum"), "count": relation.get("count")},
            )
        )
    return flags


def attach_anomalies(factpack: dict[str, Any], strategy: dict[str, Any], thresholds: dict[str, Any]) -> None:
    model_flags: list[dict[str, Any]] = []
    for entity in factpack.get("focus_entities", []):
        flags = detect_entity_anomalies(entity, strategy, thresholds)
        entity["anomaly_flags"] = flags
        model_flags.extend(
            {
                **flag,
                "scope": "entity",
                "entity_name": entity.get("entity_name"),
                "fact_id": entity.get("fact_id"),
            }
            for flag in flags
        )
    relation_summary = factpack.get("relation_summary") or {}
    for relation in factpack.get("relation_profiles", []):
        flags = detect_relation_anomalies(relation, strategy, relation_summary, thresholds)
        relation["anomaly_flags"] = flags
        model_flags.extend(
            {
                **flag,
                "scope": "relation",
                "source": relation.get("source"),
                "target": relation.get("target"),
                "fact_id": relation.get("fact_id"),
            }
            for flag in flags
        )
    factpack["anomaly_flags"] = model_flags


def build_followup_hints(factpack: dict[str, Any], strategy: dict[str, Any]) -> list[dict[str, Any]]:
    hints = []
    context = factpack.get("model_context", {})
    for entity in factpack.get("focus_entities", [])[:3]:
        if not entity.get("anomaly_flags") and entity.get("rank", 99) > 1:
            continue
        hints.append(
            {
                "hint_type": "entity_followup",
                "source_model_id": context.get("model_id"),
                "entity_name": entity.get("entity_name"),
                "suggested_scope": {
                    "person": entity.get("entity_name") if entity.get("entity_type") == "person" else None,
                    "accounts": entity.get("accounts", [])[:5],
                },
                "suggested_next_models": strategy.get("allowed_followup_models", []),
                "reason": "重点主体在当前模型中排名靠前或触发异常规则，建议在相关模型中按主体/账户缩小范围复核。",
                "fact_id": entity.get("fact_id"),
            }
        )
    for relation in factpack.get("relation_profiles", [])[:3]:
        if not relation.get("anomaly_flags") and relation.get("rank", 99) > 1:
            continue
        hints.append(
            {
                "hint_type": "relation_followup",
                "source_model_id": context.get("model_id"),
                "source": relation.get("source"),
                "target": relation.get("target"),
                "suggested_scope": {"source": relation.get("source"), "target": relation.get("target")},
                "suggested_next_models": strategy.get("allowed_followup_models", []),
                "reason": "重点关系边金额或次数靠前，建议围绕 source/target 做关系链补强。",
                "fact_id": relation.get("fact_id"),
            }
        )
    return hints[:5]

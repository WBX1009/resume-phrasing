from __future__ import annotations

from collections import defaultdict
from typing import Any

from .entity_resolver import collect_entity_mentions


def build_cross_model_pack(factpacks: list[dict[str, Any]]) -> dict[str, Any]:
    mentions = collect_entity_mentions(factpacks)
    cross_entities = []
    for entity_name, rows in mentions.items():
        model_ids = {row["model_id"] for row in rows if row.get("model_id") is not None}
        total_amount = sum(float(row.get("metrics", {}).get("amount_sum") or 0) for row in rows)
        risk_tags = sorted({tag for row in rows for tag in row.get("risk_tags", [])})
        if len(model_ids) >= 2 or risk_tags:
            cross_entities.append(
                {
                    "entity_name": entity_name,
                    "model_count": len(model_ids),
                    "models": rows,
                    "total_amount_sum": round(total_amount, 2),
                    "risk_tags": risk_tags,
                    "support_level": "enhanced" if len(model_ids) >= 2 else "single_model_strong",
                }
            )
    cross_entities.sort(
        key=lambda item: (item["model_count"], len(item["risk_tags"]), item["total_amount_sum"]),
        reverse=True,
    )
    return {
        "cross_model_entities": cross_entities[:10],
        "hypothesis_candidates": build_hypothesis_candidates(cross_entities[:10]),
    }


def build_hypothesis_candidates(cross_entities: list[dict[str, Any]]) -> list[dict[str, Any]]:
    candidates = []
    for index, entity in enumerate(cross_entities[:5], start=1):
        candidates.append(
            {
                "hypothesis_id": f"hyp_{index:02d}",
                "entity_name": entity["entity_name"],
                "status": "candidate",
                "support_level": entity["support_level"],
                "basis": {
                    "model_count": entity["model_count"],
                    "risk_tags": entity["risk_tags"],
                    "total_amount_sum": entity["total_amount_sum"],
                },
                "next_step": "在相关模型中按主体/账户/关系边进行白名单补强查询。",
            }
        )
    return candidates


def build_core_findings(factpacks: list[dict[str, Any]], cross_pack: dict[str, Any]) -> list[dict[str, Any]]:
    findings = []
    for entity in cross_pack.get("cross_model_entities", [])[:3]:
        findings.append(
            {
                "finding_type": "cross_model_entity",
                "title": f"{entity['entity_name']} 在多个模型中形成交叉出现",
                "summary": (
                    f"该主体出现在 {entity['model_count']} 个模型的重点对象中，"
                    f"累计相关金额 {entity['total_amount_sum']:.2f}。"
                ),
                "fact_refs": [row.get("fact_id") for row in entity.get("models", []) if row.get("fact_id")],
                "risk_tags": entity.get("risk_tags", []),
            }
        )
    relation_candidates = []
    for factpack in factpacks:
        for relation in factpack.get("relation_profiles", [])[:2]:
            if relation.get("anomaly_flags"):
                relation_candidates.append((factpack, relation))
    for factpack, relation in relation_candidates[:2]:
        context = factpack.get("model_context", {})
        findings.append(
            {
                "finding_type": "relation_pattern",
                "title": f"{context.get('model_name')} 中存在高关注关系边",
                "summary": (
                    f"{relation.get('source')} -> {relation.get('target')} "
                    f"累计金额 {relation.get('amount_sum')}，次数 {relation.get('count')}。"
                ),
                "fact_refs": [relation.get("fact_id")],
                "risk_tags": [flag["risk_tag"] for flag in relation.get("anomaly_flags", [])],
            }
        )
    return findings[:5]


def build_root_summaries(factpacks: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for factpack in factpacks:
        context = factpack.get("model_context", {})
        result[context.get("root_category") or "未分类"].append(
            {
                "model_id": context.get("model_id"),
                "model_name": context.get("model_name"),
                "demo_level": context.get("demo_level"),
                "result_count": context.get("result_count"),
                "focus_entity_count": len(factpack.get("focus_entities", [])),
                "relation_profile_count": len(factpack.get("relation_profiles", [])),
                "anomaly_count": len(factpack.get("anomaly_flags", [])),
            }
        )
    return dict(result)

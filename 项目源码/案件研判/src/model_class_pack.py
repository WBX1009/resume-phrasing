from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from typing import Any

from .report_vocabulary import cn_risk_tags


NON_REASONING_MODEL_KINDS = {"entity_dictionary", "no_data", "unknown"}


def build_model_class_packs(
    model_factpacks: list[dict[str, Any]],
    bundle: Any | None = None,
) -> list[dict[str, Any]]:
    """Aggregate model-level FactPacks into model-class context packs.

    The output is deterministic and does not call the database or LLM. It is
    intended as the middle context layer between ModelFactPack and Planner.
    """

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for factpack in model_factpacks:
        groups[_class_name_for(factpack)].append(factpack)

    packs = []
    for class_name, factpacks in groups.items():
        models = [_model_summary(fp, bundle) for fp in factpacks]
        active_models = [
            model
            for model in models
            if int(model.get("result_count") or 0) > 0
            and model.get("model_kind") != "entity_dictionary"
        ]
        reasoning_models = [
            model
            for model in active_models
            if str(model.get("model_kind") or "") not in NON_REASONING_MODEL_KINDS
            and model.get("model_kind") != "category"
        ]
        shared_focus_entities, class_specific_entities = _entity_groups(factpacks)
        shared_relation_entities = _relation_entity_groups(factpacks)
        class_risk_patterns = _class_risk_patterns(
            class_name,
            shared_focus_entities,
            shared_relation_entities,
            class_specific_entities,
            factpacks,
        )
        pack = {
            "class_name": class_name,
            "model_count": len(models),
            "active_model_count": len(active_models),
            "active_model_ids": [model["model_id"] for model in active_models],
            "reasoning_model_count": len(reasoning_models),
            "reasoning_model_ids": [model["model_id"] for model in reasoning_models],
            "total_result_count": sum(int(model.get("result_count") or 0) for model in models),
            "models": models,
            "shared_focus_entities": shared_focus_entities,
            "shared_relation_entities": shared_relation_entities,
            "class_specific_entities": class_specific_entities,
            "class_risk_patterns": class_risk_patterns,
            "available_tools": _available_tools(factpacks, bundle),
            "candidate_hypotheses": _candidate_hypotheses(class_name, shared_focus_entities, shared_relation_entities),
        }
        pack["token_budget_estimate"] = len(json.dumps(pack, ensure_ascii=False, default=str))
        packs.append(pack)

    packs.sort(key=lambda item: (item["reasoning_model_count"], item["total_result_count"]), reverse=True)
    return packs


def _class_name_for(factpack: dict[str, Any]) -> str:
    context = factpack.get("model_context", {})
    root = context.get("root_category")
    if root:
        return str(root)
    path = context.get("model_tree_path") or factpack.get("model_tree_path") or []
    if isinstance(path, list):
        meaningful = [str(item) for item in path if item]
        if len(meaningful) >= 2:
            return meaningful[1]
        if meaningful:
            return meaningful[0]
    path_text = context.get("model_tree_path_text")
    if path_text:
        parts = [part.strip() for part in str(path_text).split("/") if part.strip()]
        if len(parts) >= 2:
            return parts[1]
        if parts:
            return parts[0]
    return "未分类"


def _model_summary(factpack: dict[str, Any], bundle: Any | None) -> dict[str, Any]:
    context = factpack.get("model_context", {})
    model_id = int(context.get("model_id"))
    strategy = getattr(bundle, "by_model_id", {}).get(model_id, {}) if bundle is not None else {}
    allowed_tools = factpack.get("allowed_tools") or strategy.get("allowed_tools") or []
    return {
        "model_id": model_id,
        "model_name": context.get("model_name"),
        "model_kind": context.get("model_kind"),
        "demo_level": context.get("demo_level"),
        "demo_role": context.get("demo_role"),
        "result_count": int(context.get("result_count") or 0),
        "focus_entity_count": len(factpack.get("focus_entities", [])),
        "relation_profile_count": len(factpack.get("relation_profiles", [])),
        "anomaly_count": len(factpack.get("anomaly_flags", [])),
        "allowed_tools": list(allowed_tools),
        "shared_entity_types": factpack.get("shared_entity_types") or strategy.get("shared_entity_types", []),
        "model_tree_path_text": context.get("model_tree_path_text"),
    }


def _entity_groups(factpacks: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    appearances_by_name: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for factpack in factpacks:
        context = factpack.get("model_context", {})
        model_id = int(context.get("model_id"))
        model_name = context.get("model_name")
        for entity in factpack.get("focus_entities", []):
            entity_name = entity.get("entity_name")
            if not entity_name:
                continue
            metrics = entity.get("metrics", {}) or {}
            risk_tags = [flag.get("risk_tag") for flag in entity.get("anomaly_flags", []) if flag.get("risk_tag")]
            appearances_by_name[str(entity_name)].append(
                {
                    "model_id": model_id,
                    "model_name": model_name,
                    "rank": entity.get("rank"),
                    "fact_id": entity.get("fact_id"),
                    "record_count": metrics.get("record_count"),
                    "amount_sum": metrics.get("amount_sum"),
                    "amount_max": metrics.get("amount_max"),
                    "distinct_counterparty_count": metrics.get("distinct_counterparty_count"),
                    "risk_tags": risk_tags,
                    "risk_tags_cn": cn_risk_tags(risk_tags),
                    "evidence_refs": entity.get("evidence_refs", []),
                }
            )

    shared = []
    specific = []
    for entity_name, appearances in appearances_by_name.items():
        model_ids = {item["model_id"] for item in appearances}
        total_amount = _sum_numeric(item.get("amount_sum") for item in appearances)
        risk_tags = sorted({tag for item in appearances for tag in item.get("risk_tags", [])})
        payload = {
            "entity_name": entity_name,
            "cross_model_count": len(model_ids),
            "appearance_count": len(appearances),
            "total_amount_sum": round(total_amount, 2),
            "risk_tags": risk_tags,
            "risk_tags_cn": cn_risk_tags(risk_tags),
            "appearances": sorted(appearances, key=lambda item: (int(item.get("rank") or 999), item["model_id"])),
        }
        if len(model_ids) >= 2:
            shared.append(payload)
        elif risk_tags:
            first = appearances[0]
            specific.append(
                {
                    "entity_name": entity_name,
                    "model_id": first["model_id"],
                    "model_name": first["model_name"],
                    "rank": first.get("rank"),
                    "fact_id": first.get("fact_id"),
                    "amount_sum": first.get("amount_sum"),
                    "risk_tags": risk_tags,
                    "risk_tags_cn": cn_risk_tags(risk_tags),
                }
            )

    shared.sort(key=lambda item: (item["cross_model_count"], len(item["risk_tags"]), item["total_amount_sum"]), reverse=True)
    specific.sort(key=lambda item: (len(item["risk_tags"]), float(item.get("amount_sum") or 0)), reverse=True)
    return shared[:10], specific[:10]


def _relation_entity_groups(factpacks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    appearances_by_name: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for factpack in factpacks:
        context = factpack.get("model_context", {})
        model_id = int(context.get("model_id"))
        model_name = context.get("model_name")
        for relation in factpack.get("relation_profiles", []):
            risk_tags = [flag.get("risk_tag") for flag in relation.get("anomaly_flags", []) if flag.get("risk_tag")]
            for role in ("source", "target"):
                name = relation.get(role)
                if not name:
                    continue
                appearances_by_name[str(name)].append(
                    {
                        "model_id": model_id,
                        "model_name": model_name,
                        "role": role,
                        "rank": relation.get("rank"),
                        "fact_id": relation.get("fact_id"),
                        "source": relation.get("source"),
                        "target": relation.get("target"),
                        "count": relation.get("count"),
                        "amount_sum": relation.get("amount_sum"),
                        "risk_tags": risk_tags,
                        "risk_tags_cn": cn_risk_tags(risk_tags),
                    }
                )

    result = []
    for entity_name, appearances in appearances_by_name.items():
        model_ids = {item["model_id"] for item in appearances}
        risk_tags = sorted({tag for item in appearances for tag in item.get("risk_tags", [])})
        if len(model_ids) >= 2 or risk_tags:
            result.append(
                {
                    "entity_name": entity_name,
                    "cross_model_count": len(model_ids),
                    "appearance_count": len(appearances),
                    "total_amount_sum": round(_sum_numeric(item.get("amount_sum") for item in appearances), 2),
                    "risk_tags": risk_tags,
                    "risk_tags_cn": cn_risk_tags(risk_tags),
                    "appearances": sorted(appearances, key=lambda item: (int(item.get("rank") or 999), item["model_id"]))[:8],
                }
            )
    result.sort(key=lambda item: (item["cross_model_count"], len(item["risk_tags"]), item["total_amount_sum"]), reverse=True)
    return result[:10]


def _class_risk_patterns(
    class_name: str,
    shared_focus_entities: list[dict[str, Any]],
    shared_relation_entities: list[dict[str, Any]],
    class_specific_entities: list[dict[str, Any]],
    factpacks: list[dict[str, Any]],
) -> list[str]:
    patterns = []
    if shared_focus_entities:
        names = "、".join(item["entity_name"] for item in shared_focus_entities[:3])
        patterns.append(f"{names} 在{class_name}类多个模型中交叉出现，可作为模型类级重点核查对象。")
    if shared_relation_entities:
        names = "、".join(item["entity_name"] for item in shared_relation_entities[:3])
        patterns.append(f"{names} 在{class_name}类关系边中反复出现，建议结合邻域、路径和对手方重合情况核查。")
    tag_counter: Counter[str] = Counter()
    for factpack in factpacks:
        for flag in factpack.get("anomaly_flags", []):
            tag = flag.get("risk_tag")
            if tag:
                tag_counter[str(tag)] += 1
        for entity in factpack.get("focus_entities", []):
            for flag in entity.get("anomaly_flags", []):
                tag = flag.get("risk_tag")
                if tag:
                    tag_counter[str(tag)] += 1
        for relation in factpack.get("relation_profiles", []):
            for flag in relation.get("anomaly_flags", []):
                tag = flag.get("risk_tag")
                if tag:
                    tag_counter[str(tag)] += 1
    for tag, count in tag_counter.most_common(3):
        patterns.append(f"{class_name}类中“{cn_risk_tags([tag])[0]}”出现 {count} 次，说明该类模型存在可聚合的共同风险模式。")
    if not patterns and class_specific_entities:
        names = "、".join(item["entity_name"] for item in class_specific_entities[:3])
        patterns.append(f"{names} 虽只在单模型中出现，但已触发高关注规则，可作为后续补证候选。")
    return patterns[:6]


def _available_tools(factpacks: list[dict[str, Any]], bundle: Any | None) -> list[str]:
    tools = set()
    by_model = getattr(bundle, "by_model_id", {}) if bundle is not None else {}
    for factpack in factpacks:
        context = factpack.get("model_context", {})
        model_id = int(context.get("model_id"))
        strategy = by_model.get(model_id, {})
        for tool in factpack.get("allowed_tools") or strategy.get("allowed_tools", []):
            tools.add(str(tool))
    return sorted(tools)


def _candidate_hypotheses(
    class_name: str,
    shared_focus_entities: list[dict[str, Any]],
    shared_relation_entities: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    candidates = []
    for index, entity in enumerate(shared_focus_entities[:3], start=1):
        fact_ids = [
            item.get("fact_id")
            for item in entity.get("appearances", [])
            if item.get("fact_id")
        ]
        candidates.append(
            {
                "hypothesis_id": f"class_{_slug(class_name)}_h{index:02d}",
                "entity_name": entity["entity_name"],
                "statement": (
                    f"{entity['entity_name']} 在{class_name}类 {entity['cross_model_count']} 个模型中交叉出现，"
                    "需要验证其金额、频次、关系或时间窗口特征是否形成同向支撑。"
                ),
                "basis_fact_ids": fact_ids[:6],
                "source": "model_class_pack",
            }
        )
    offset = len(candidates)
    for index, entity in enumerate(shared_relation_entities[:2], start=1):
        fact_ids = [
            item.get("fact_id")
            for item in entity.get("appearances", [])
            if item.get("fact_id")
        ]
        candidates.append(
            {
                "hypothesis_id": f"class_{_slug(class_name)}_r{index + offset:02d}",
                "entity_name": entity["entity_name"],
                "statement": f"{entity['entity_name']} 在{class_name}类关系边中出现，需要验证邻域和路径是否增强该线索。",
                "basis_fact_ids": fact_ids[:6],
                "source": "model_class_pack_relation",
            }
        )
    return candidates[:5]


def _sum_numeric(values: Any) -> float:
    total = 0.0
    for value in values:
        try:
            total += float(value or 0)
        except Exception:
            continue
    return total


def _slug(value: str) -> str:
    ascii_part = "".join(ch.lower() if ch.isascii() and ch.isalnum() else "_" for ch in value)
    ascii_part = ascii_part[:16].strip("_")
    digest = hashlib.sha1(value.encode("utf-8")).hexdigest()[:6]
    return f"{ascii_part or 'c'}_{digest}"

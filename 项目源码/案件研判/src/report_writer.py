from __future__ import annotations

from collections import defaultdict
from typing import Any


def _line_for_operator(result: dict[str, Any]) -> str:
    operator = result.get("operator", "unknown")
    if result.get("status") == "skipped":
        return f"- `{operator}`：跳过，原因：{result.get('reason', '未说明')}"
    if operator == "numeric_topk":
        return f"- `numeric_topk`：字段 `{result['field']}`，最大值 `{result['max']}`，可解析记录 `{result['count']}` 条。"
    if operator in {"category_topk", "entity_frequency"}:
        top = result.get("topk", [{}])[0]
        return f"- `{operator}`：字段 `{result['field']}`，最高频取值 `{top.get('value')}`，次数 `{top.get('count')}`。"
    if operator == "time_bucket":
        return f"- `time_bucket`：字段 `{result['field']}`，时间范围 `{result['time_range']['min']}` 至 `{result['time_range']['max']}`。"
    if operator == "relation_edges":
        edge = result.get("top_edges", [{}])[0]
        return f"- `relation_edges`：字段 `{result['left_field']}` → `{result['right_field']}`，首要关系 `{edge.get('source')}` → `{edge.get('target')}`，次数 `{edge.get('count')}`。"
    if operator == "evidence_samples":
        return f"- `evidence_samples`：抽取 `{len(result.get('samples', []))}` 条样本。"
    return f"- `{operator}`：已执行。"


def render_markdown_report(
    case_id: int,
    contexts: list[dict[str, Any]],
    fact_pack: dict[str, Any],
) -> str:
    lines: list[str] = [
        f"# case {case_id} 综合分析报告 Demo",
        "",
        "> 本报告由确定性模板基于 FactPack 生成，未调用 LLM；报告中的统计数字来自后端算子结果。",
        "",
        "## 1. Demo 模型覆盖情况",
        "",
        "| model_id | 模型名称 | 模型类别 | 数据库计数 | 载入条数 |",
        "|---:|---|---|---:|---:|",
    ]
    context_by_model = {item["model_id"]: item for item in contexts}
    for context in contexts:
        lines.append(
            f"| {context['model_id']} | {context['model_name']} | {context['model_category']} | "
            f"{context['result_count']} | {context.get('loaded_result_count', context['result_count'])} |"
        )

    lines.extend(["", "## 2. 模型级事实摘要", ""])
    for model in fact_pack["models"]:
        context = context_by_model.get(model["model_id"], {})
        lines.extend(
            [
                f"### {model['model_id']} {model['model_name']}",
                "",
                f"- 模型类别：{model['category']}",
                f"- 结果条数：{model['result_count']}",
                f"- 业务目标：{context.get('business_goal') or '未配置'}",
            ]
        )
        if model["json_parse_errors"]:
            lines.append(f"- JSON 解析失败：{model['json_parse_errors']} 条")
        for result in model["operator_results"]:
            lines.append(_line_for_operator(result))
        lines.append("")

    lines.extend(["## 3. 主题级综合", ""])
    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for model in fact_pack["models"]:
        by_category[model["category"]].append(model)
    for category, models in by_category.items():
        success_count = sum(
            1
            for model in models
            for result in model["operator_results"]
            if result.get("status") == "success"
        )
        skipped_count = sum(
            1
            for model in models
            for result in model["operator_results"]
            if result.get("status") == "skipped"
        )
        lines.append(f"- {category}：覆盖 {len(models)} 个模型，成功算子 {success_count} 个，跳过算子 {skipped_count} 个。")

    total_models = len(fact_pack["models"])
    total_facts = sum(len(model["facts"]) for model in fact_pack["models"])
    total_records = sum(model["result_count"] for model in fact_pack["models"])
    lines.extend(
        [
            "",
            "## 4. Case 级综合研判",
            "",
            f"- 本次 demo 针对 case `{case_id}` 覆盖 `{total_models}` 个代表性模型，共读取 `{total_records}` 条模型结果。",
            f"- 系统生成 `{total_facts}` 条结构化事实，后续可作为 LLM 润色和正式报告生成的唯一事实来源。",
            "- 当前版本以规则映射选择算子，已验证字段画像、受控分析、FactPack 和模板报告链路。",
            "",
            "## 5. 后续核查建议",
            "",
            "- 优先核查 FactPack 中金额 TopK、关系边 TopK 和证据样本对应的主体、账户、时间段。",
            "- 对 skipped 算子对应模型，回看字段画像，确认是否需要补充字段语义规则或模型专属算子。",
            "- 链路稳定后，再将规则映射替换为 LLM 受控工具选择。",
            "",
            "## 6. 数据与算子执行备注",
            "",
            f"- 数据来源：`{fact_pack['data_source']}`",
        ]
    )
    for warning in fact_pack.get("warnings", []):
        lines.append(f"- 备注：{warning}")
    return "\n".join(lines) + "\n"

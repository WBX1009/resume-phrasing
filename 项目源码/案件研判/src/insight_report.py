from __future__ import annotations

from typing import Any
import json
import re
import time

from .llm_utils import cached_chat_json, model_candidates, retry_delay_seconds, update_audit_from_result
from .report_vocabulary import (
    cn_aggregation_basis,
    cn_demo_level,
    cn_risk_tag,
    cn_risk_tags,
    cn_status,
    cn_tool_name,
)


def _fmt_amount(value: Any) -> str:
    try:
        number = float(value or 0)
    except Exception:
        return str(value)
    if abs(number) >= 100000000:
        return f"{number / 100000000:.2f}亿元"
    if abs(number) >= 10000:
        return f"{number / 10000:.2f}万元"
    return f"{number:.2f}元"


def _risk_text(flags: list[dict[str, Any]]) -> str:
    if not flags:
        return "未触发已配置的高关注规则。"
    by_tag: dict[str, dict[str, Any]] = {}
    for flag in flags:
        tag = str(flag.get("risk_tag") or "")
        if tag and tag not in by_tag:
            by_tag[tag] = flag
    return "；".join(
        _risk_expression(flag)
        for flag in list(by_tag.values())[:4]
    )


def _brief_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "record_count",
        "amount_sum",
        "amount_max",
        "amount_avg",
        "amount_p95",
        "near_threshold_count",
        "night_count",
        "night_ratio",
        "max_hourly_count",
        "distinct_counterparty_count",
        "same_day_same_amount_max",
    ]
    return {key: metrics.get(key) for key in keys if key in metrics}


def _brief_entity(entity: dict[str, Any]) -> dict[str, Any]:
    buckets = entity.get("top_time_buckets", {}) or {}
    return {
        "rank": entity.get("rank"),
        "fact_id": entity.get("fact_id"),
        "entity_name": entity.get("entity_name"),
        "entity_type": entity.get("entity_type"),
        "metrics": _brief_metrics(entity.get("metrics", {})),
        "top_time_buckets": {
            "days": (buckets.get("days") or [])[:2],
            "hours": (buckets.get("hours") or [])[:2],
        },
        "counterparty_topk": entity.get("counterparty_topk", [])[:2],
        "direction_counts": entity.get("direction_counts", [])[:2],
        "risk_tags": [flag.get("risk_tag") for flag in entity.get("anomaly_flags", [])],
    }


def _brief_relation(relation: dict[str, Any]) -> dict[str, Any]:
    return {
        "rank": relation.get("rank"),
        "fact_id": relation.get("fact_id"),
        "source": relation.get("source"),
        "target": relation.get("target"),
        "count": relation.get("count"),
        "amount_sum": relation.get("amount_sum"),
        "amount_max": relation.get("amount_max"),
        "self_transfer": relation.get("self_transfer"),
        "time_range": relation.get("time_range"),
        "risk_tags": [flag.get("risk_tag") for flag in relation.get("anomaly_flags", [])],
    }


def build_llm_brief_pack(
    model_factpacks: list[dict[str, Any]],
    case_pack: dict[str, Any],
    max_entities_per_model: int = 2,
    max_relations_per_model: int = 2,
) -> dict[str, Any]:
    """Build a small, evidence-addressable pack for LLM reasoning.

    The brief intentionally excludes evidence_records, top_amount_records, and raw rows.
    """

    models = []
    for fp in model_factpacks:
        context = fp.get("model_context", {})
        models.append(
            {
                "model_id": context.get("model_id"),
                "model_name": context.get("model_name"),
                "model_tree_path_text": context.get("model_tree_path_text"),
                "root_category": context.get("root_category"),
                "demo_level": context.get("demo_level"),
                "demo_role": context.get("demo_role"),
                "model_kind": context.get("model_kind"),
                "result_count": context.get("result_count"),
                "key_metrics": {
                    "aggregation_basis": fp.get("key_metrics", {}).get("aggregation_basis"),
                    "numeric_summary": _brief_numeric_summary(fp.get("key_metrics", {}).get("numeric_summary", {})),
                    "no_data_reason": fp.get("key_metrics", {}).get("no_data_reason"),
                },
                "focus_entities": [
                    _brief_entity(entity)
                    for entity in fp.get("focus_entities", [])[:max_entities_per_model]
                ],
                "relation_profiles": [
                    _brief_relation(relation)
                    for relation in fp.get("relation_profiles", [])[:max_relations_per_model]
                ],
                "anomaly_flag_count": len(fp.get("anomaly_flags", [])),
                "followup_hints": [
                    {
                        "hint_type": hint.get("hint_type"),
                        "source_model_id": hint.get("source_model_id"),
                        "entity_name": hint.get("entity_name"),
                        "source": hint.get("source"),
                        "target": hint.get("target"),
                        "fact_id": hint.get("fact_id"),
                        "suggested_next_models": hint.get("suggested_next_models", []),
                    }
                    for hint in fp.get("followup_hints", [])[:3]
                ],
            }
        )

    return {
        "case_id": case_pack.get("case_id"),
        "aggregation_basis": case_pack.get("aggregation_basis"),
        "core_findings": _brief_core_findings(case_pack.get("core_findings", [])[:5]),
        "cross_model_entities": _brief_cross_entities(case_pack.get("cross_model_entities", [])[:5]),
        "hypothesis_candidates": _brief_hypotheses(case_pack.get("hypothesis_candidates", [])[:5]),
        "models": models,
        "llm_constraints": {
            "raw_rows_available": False,
            "may_generate_sql": False,
            "must_reference_fact_ids": True,
            "facts_are_deterministic": True,
        },
    }


def _brief_numeric_summary(summary: dict[str, Any]) -> dict[str, Any]:
    result = {}
    for field, metrics in list(summary.items())[:3]:
        if not isinstance(metrics, dict):
            continue
        result[field] = {
            "count": metrics.get("count"),
            "sum": metrics.get("sum"),
            "max": metrics.get("max"),
            "p95": metrics.get("p95"),
        }
    return result


def _brief_core_findings(findings: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "finding_type": item.get("finding_type"),
            "title": _short_text(item.get("title"), 80),
            "summary": _short_text(item.get("summary"), 120),
            "fact_refs": item.get("fact_refs", [])[:5],
            "risk_tags": item.get("risk_tags", [])[:5],
        }
        for item in findings
    ]


def _brief_cross_entities(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for item in items:
        models = item.get("models", []) or []
        result.append(
            {
                "entity_name": item.get("entity_name"),
                "model_count": item.get("model_count"),
                "total_amount_sum": item.get("total_amount_sum"),
                "risk_tags": item.get("risk_tags", [])[:5],
                "support_level": item.get("support_level"),
                "models": [
                    {
                        "model_id": row.get("model_id"),
                        "rank": row.get("rank"),
                        "fact_id": row.get("fact_id"),
                    }
                    for row in models[:5]
                ],
            }
        )
    return result


def _brief_hypotheses(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "hypothesis_id": item.get("hypothesis_id"),
            "entity_name": item.get("entity_name"),
            "status": item.get("status"),
            "support_level": item.get("support_level"),
            "basis": item.get("basis"),
        }
        for item in items
    ]


def _short_text(value: Any, limit: int) -> str:
    text = str(value or "")
    return text if len(text) <= limit else text[: limit - 1] + "…"


def render_insight_report(
    case_id: int,
    model_factpacks: list[dict[str, Any]],
    case_pack: dict[str, Any],
    mode: str = "template",
    planner_result: dict[str, Any] | None = None,
) -> str:
    if planner_result:
        phase_name = "第 2 阶段规划器证据链"
        scope_note = "本报告展示基础事实、候选假设、补证工具信号和大模型 Judge 证据链结论。"
    elif mode == "llm":
        phase_name = "局部大模型叙事"
        scope_note = "本报告为模板骨架叠加模型级大模型叙事润色，不等同于完整大模型报告。"
    else:
        phase_name = "第 1.5 阶段模板事实"
        scope_note = "本报告只展示确定性基础事实，不包含规划器补证链路或大模型叙事润色。"
    lines = [
        f"# 案例 {case_id} {phase_name}报告",
        "",
        f"> {scope_note}",
        "> 本报告围绕模型树逐模型呈现。确定性统计来自完整缓存结果集。",
        "",
        "## 1. 案例概览",
        "",
        f"- 覆盖代表模型：{case_pack['model_count']} 个；结果总量：{case_pack['total_result_count']} 条。",
        f"- 业务洞察口径：{cn_aggregation_basis(case_pack['aggregation_basis'])}。",
        "- 运行审计信息仅写入结构化审计文件，不进入正文。",
        "",
        "## 2. 核心发现",
        "",
    ]

    findings = case_pack.get("core_findings") or []
    if not findings:
        lines.append("- 当前事实包未形成跨模型核心发现，建议扩展模型或调低重点对象数量阈值。")
    for index, finding in enumerate(findings, start=1):
        refs = ", ".join(ref for ref in finding.get("fact_refs", []) if ref)
        lines.append(f"{index}. {finding['title']}：{finding['summary']} 证据：{refs or '见模型片段'}")

    if planner_result:
        lines.extend(["", "## 3. 动态补证结论", ""])
        audit = planner_result.get("planner_audit", {})
        if audit.get("mode") == "agent-loop-demo":
            lines.append(
                "- 本节来自 Agent Loop：每条线索按“规划器决策 → 工具补证 → 大模型 Judge → 再规划”的证据链记录。"
            )
        for chain in planner_result.get("evidence_chains", [])[:5]:
            hyp = chain.get("hypothesis", {})
            lines.append(
                f"- {chain.get('chain_id')}：{hyp.get('statement')}；"
                f"判断结果：{cn_status(chain.get('final_status'))}；步骤数：{len(chain.get('steps', []))}。"
            )
            if audit.get("mode") == "agent-loop-demo":
                for step in chain.get("steps", [])[:6]:
                    decision = step.get("decision", {})
                    lines.append(
                        f"  - 第 {step.get('round')} 步：{cn_tool_name(decision.get('tool_name'))}；"
                        f"状态：{cn_status(step.get('status'))}；{step.get('assessment')}"
                    )

    section_index = 4 if planner_result else 3
    lines.extend(["", f"## {section_index}. 逐模型分析", ""])
    supplements_by_model = _supplements_by_model(planner_result or {})
    for factpack in model_factpacks:
        context = factpack["model_context"]
        model_id = int(context["model_id"])
        lines.extend(
            [
                f"### {context['model_id']} {context['model_name']}",
                "",
                f"- 模型路径：{context['model_tree_path_text']}",
                f"- 模型定位：{cn_demo_level(context.get('demo_level'))}，{context.get('demo_role')}",
                f"- 结果条数：{context['result_count']}；聚合口径：{cn_aggregation_basis(factpack['key_metrics']['aggregation_basis'])}",
            ]
        )
        if context["result_count"] == 0:
            lines.append(f"- 数据说明：{factpack['key_metrics'].get('no_data_reason')}")
            lines.append("")
            continue
        if factpack.get("entity_dictionary"):
            people_count = len(factpack["entity_dictionary"])
            lines.append(f"- 实体字典：识别 {people_count} 名人员，用于后续模型的账户/手机号归一化。")
            lines.append("")
            continue

        if factpack.get("focus_entities"):
            lines.append("**重点主体**")
            for entity in factpack["focus_entities"][:5]:
                lines.append(_entity_line(entity))
        if factpack.get("relation_profiles"):
            lines.append("")
            lines.append("**重点关系**")
            for relation in factpack["relation_profiles"][:5]:
                lines.append(_relation_line(relation))

        lines.append("")
        lines.append("**规律发现**")
        narrative = factpack.get("llm_narrative", {}).get("model_narrative") if mode == "llm" else None
        if narrative:
            lines.append(f"- {narrative}")
        else:
            lines.extend(_template_insight_lines(factpack))

        lines.append("")
        lines.append("**风险提示**")
        lines.append(f"- {_risk_text(factpack.get('anomaly_flags', []))}")

        model_supplements = supplements_by_model.get(model_id, [])
        if model_supplements:
            lines.append("")
            lines.append("**动态补证**")
            for supplement in model_supplements[:4]:
                output = supplement.get("output", {})
                lines.append(
                    f"- {supplement.get('supplement_fact_id')} / {cn_tool_name(supplement.get('tool_name'))}："
                    f"{output.get('summary')} 工具信号：{cn_status(output.get('status_signal'))}。"
                )

        hints = factpack.get("followup_hints", [])
        if hints:
            lines.append("")
            lines.append("**后续关注**")
            for hint in hints[:3]:
                if hint.get("hint_type") == "relation_followup":
                    lines.append(
                        f"- 围绕关系 `{hint.get('source')}` -> `{hint.get('target')}` 做白名单补强；依据 {hint.get('fact_id')}。"
                    )
                else:
                    lines.append(f"- 围绕主体 `{hint.get('entity_name')}` 做同类/关联模型补强；依据 {hint.get('fact_id')}。")
        lines.append("")

    section_index += 1
    lines.extend([f"## {section_index}. 跨模型印证", ""])
    cross_entities = case_pack.get("cross_model_entities") or []
    if not cross_entities:
        lines.append("- 当前重点主体尚未形成多模型交叉印证。")
    for item in cross_entities[:8]:
        model_text = "、".join(f"{row['model_id']}({row['model_name']})" for row in item.get("models", [])[:4])
        lines.append(
            f"- {item['entity_name']}：出现在 {item['model_count']} 个模型重点对象中，"
            f"累计相关金额 {_fmt_amount(item['total_amount_sum'])}；涉及模型：{model_text}。"
        )

    section_index += 1
    lines.extend(["", f"## {section_index}. 研判提示", ""])
    for candidate in case_pack.get("hypothesis_candidates", [])[:5]:
        lines.append(
            f"- {candidate['hypothesis_id']}：{candidate['entity_name']}；"
            f"{candidate['next_step']} 当前状态：{candidate['status']}。"
        )
    if planner_result:
        audit = planner_result.get("planner_audit", {})
        lines.append("")
        lines.append(f"- 规划器审计：{audit.get('status')}；{audit.get('reason', '')}")
    return "\n".join(lines) + "\n"


def generate_llm_narratives(config: Any, model_factpacks: list[dict[str, Any]], case_pack: dict[str, Any]) -> dict[str, Any]:
    audit = {"mode": "llm", "status": "skipped", "reason": "", "brief_chars": 0}
    if not config.llm.is_complete:
        audit["reason"] = "LLM configuration is incomplete"
        return audit
    try:
        brief = build_llm_brief_pack(model_factpacks, case_pack, max_entities_per_model=2, max_relations_per_model=2)
        brief_text = json.dumps(brief, ensure_ascii=False, default=str)
        audit["brief_chars"] = len(brief_text)
        allowed_terms = _allowed_fact_terms_from_brief(brief)
        prompt = (
            "你是模型结果研判助手。只能使用给定 BriefPack 中的事实，不要创造主体、金额、时间、关系。"
            "不要生成 SQL 或代码。输出 JSON，结构为："
            "{\"model_insights\":[{\"model_id\":43,\"model_narrative\":\"...\",\"fact_refs\":[\"m43_entity_01\"]}]}"
            "每条 narrative 必须引用至少一个 fact_refs 中的事实，并说明业务含义，不要复述 Top1。"
        )
        payload = None
        last_error = ""
        candidates = _llm_model_candidates(config.llm.model)
        audit["model_candidates"] = candidates
        for attempt, model_name in enumerate(candidates, start=1):
            try:
                result = cached_chat_json(
                    config,
                    model_name=model_name,
                    role="narrative",
                    messages=[
                        {"role": "system", "content": prompt},
                        {"role": "user", "content": brief_text},
                    ],
                    max_tokens=min(config.llm.max_tokens, 4096),
                )
                update_audit_from_result(audit, result, model_name, attempt)
                if audit.get("last_llm_error"):
                    audit["previous_llm_error"] = audit.pop("last_llm_error")
                audit["used_model"] = model_name
                payload = result.payload
                break
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {str(exc)[:200]}"
                audit["last_llm_error"] = last_error
                audit["llm_attempts"] = attempt
                delay = retry_delay_seconds(exc, attempt)
                audit["last_retry_delay_seconds"] = delay
                time.sleep(delay)
        if payload is None:
            audit.update({"status": "fallback", "reason": last_error or "LLM did not return JSON"})
            return audit
        insights = payload.get("model_insights") if isinstance(payload, dict) else None
        if not isinstance(insights, list):
            audit.update({"status": "fallback", "reason": "LLM did not return model_insights JSON"})
            return audit
        by_model = {}
        for item in insights:
            if not isinstance(item, dict):
                continue
            model_id = item.get("model_id")
            narrative = str(item.get("model_narrative") or "").strip()
            if not model_id or not narrative:
                continue
            if _mentions_unknown_fact_terms(narrative, allowed_terms):
                audit.update({"status": "fallback", "reason": f"LLM output referenced unknown fact term for model {model_id}"})
                return audit
            by_model[str(model_id)] = item
        for fp in model_factpacks:
            model_id = str(fp["model_context"]["model_id"])
            item = by_model.get(model_id)
            if item:
                fp["llm_narrative"] = {
                    "model_narrative": item.get("model_narrative"),
                    "fact_refs": item.get("fact_refs", []),
                }
        audit.update({"status": "success", "reason": ""})
        return audit
    except Exception as exc:
        audit.update({"status": "fallback", "reason": type(exc).__name__})
        return audit


def _entity_line(entity: dict[str, Any]) -> str:
    metrics = entity.get("metrics", {})
    tags = cn_risk_tags([flag.get("risk_tag") for flag in entity.get("anomaly_flags", [])])
    tag_text = f"，触发规则：{'、'.join(tags)}" if tags else ""
    return (
        f"- 第{entity.get('rank')}位 {entity.get('entity_name')}："
        f"{metrics.get('record_count')} 条，累计金额 {_fmt_amount(metrics.get('amount_sum'))}，"
        f"最大单笔 {_fmt_amount(metrics.get('amount_max'))}，"
        f"夜间占比 {float(metrics.get('night_ratio') or 0):.1%}，"
        f"对手数 {metrics.get('distinct_counterparty_count')}{tag_text}。"
    )


def _relation_line(relation: dict[str, Any]) -> str:
    tags = cn_risk_tags([flag.get("risk_tag") for flag in relation.get("anomaly_flags", [])])
    tag_text = f"，触发规则：{'、'.join(tags)}" if tags else ""
    return (
        f"- 第{relation.get('rank')}位 {relation.get('source')} -> {relation.get('target')}："
        f"{relation.get('count')} 笔，累计金额 {_fmt_amount(relation.get('amount_sum'))}，"
        f"最大单笔 {_fmt_amount(relation.get('amount_max'))}{tag_text}。"
    )


def _template_insight_lines(factpack: dict[str, Any]) -> list[str]:
    lines = []
    entity = (factpack.get("focus_entities") or [{}])[0]
    relation = (factpack.get("relation_profiles") or [{}])[0]
    if relation:
        lines.append(
            f"- `{relation.get('source')}` 与 `{relation.get('target')}` 的关系边在金额或次数上靠前，"
            f"但仍需结合邻域、路径和跨模型补证判断其业务含义。"
        )
    elif entity:
        metrics = entity.get("metrics", {})
        tags = cn_risk_tags([flag.get("risk_tag") for flag in entity.get("anomaly_flags", [])])
        signal = f"已触发 {'、'.join(tags)}" if tags else "未触发强异常标签"
        lines.append(
            f"- `{entity.get('entity_name')}` 在该模型内形成可聚焦主体画像："
            f"记录 {metrics.get('record_count')} 条、金额 {_fmt_amount(metrics.get('amount_sum'))}、"
            f"对手 {metrics.get('distinct_counterparty_count')} 个，{signal}。"
        )
    else:
        lines.append("- 当前模型主要输出类别/统计分布，未形成可聚焦到个人的主体画像。")
    return lines


def _supplements_by_model(planner_result: dict[str, Any]) -> dict[int, list[dict[str, Any]]]:
    result: dict[int, list[dict[str, Any]]] = {}
    for supplement in planner_result.get("supplement_facts", []):
        params = supplement.get("params", {})
        model_id = params.get("model_id")
        if model_id is None:
            continue
        result.setdefault(int(model_id), []).append(supplement)
    return result


def _risk_expression(flag: dict[str, Any]) -> str:
    expression = str(flag.get("report_expression") or "")
    risk_tag = str(flag.get("risk_tag") or "")
    if not expression:
        return cn_risk_tag(risk_tag)
    return expression.replace(risk_tag, cn_risk_tag(risk_tag))


def _allowed_fact_terms_from_brief(brief: dict[str, Any]) -> set[str]:
    terms: set[str] = set()
    for model in brief.get("models", []):
        for key in ("model_name", "model_tree_path_text", "root_category"):
            if model.get(key):
                terms.add(str(model[key]))
        for entity in model.get("focus_entities", []):
            for key in ("fact_id", "entity_name"):
                if entity.get(key):
                    terms.add(str(entity[key]))
        for relation in model.get("relation_profiles", []):
            for key in ("fact_id", "source", "target"):
                if relation.get(key):
                    terms.add(str(relation[key]))
    for finding in brief.get("core_findings", []):
        for ref in finding.get("fact_refs", []):
            if ref:
                terms.add(str(ref))
    return {term for term in terms if term}


def _mentions_unknown_fact_terms(text: str, allowed_terms: set[str]) -> bool:
    if not text.strip() or not allowed_terms:
        return False
    known_hits = [term for term in allowed_terms if term and term in text]
    return len(known_hits) == 0


def _llm_model_candidates(default_model: str) -> list[str]:
    return model_candidates(default_model, "narrative")

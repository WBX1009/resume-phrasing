from __future__ import annotations

from collections import defaultdict
import json
import re
import time
from typing import Any

from .case_fact_pack import compact_fact_pack


RECOMMENDATION_SECTION = "\u540e\u7eed\u6838\u67e5\u5efa\u8bae"


def render_template_case_report(case_id: int, fact_pack: dict[str, Any], reason: str | None = None) -> str:
    lines = [
        f"# case {case_id} 综合分析报告",
        "",
        "> 本报告由模板基于 FactPack 生成；报告中的数字来自后端算子结果。",
        "",
        "## 一、模型覆盖情况",
        "",
        "| 模型ID | 模型名称 | 模型类别 | 结果条数 | 事实数 |",
        "|---:|---|---|---:|---:|",
    ]
    for model in fact_pack["models"]:
        lines.append(
            f"| {model['model_id']} | {model['model_name']} | {model['category']} | "
            f"{model['result_count']} | {len(model['facts'])} |"
        )
    lines.extend(["", "## 二、模型级事实摘要", ""])
    for model in fact_pack["models"]:
        lines.append(f"### {model['model_id']} {model['model_name']}")
        for fact in model["facts"][:4]:
            lines.append(f"- [{fact['fact_id']}] {fact['description']}")
        if not model["facts"]:
            lines.append("- 未形成成功事实，请查看 operator_audit.csv。")
        lines.append("")
    lines.extend(["## 三、模型类级综合", ""])
    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for model in fact_pack["models"]:
        by_category[model["category"]].append(model)
    for category, models in by_category.items():
        fact_count = sum(len(model["facts"]) for model in models)
        row_count = sum(model["result_count"] for model in models)
        lines.append(f"- {category}：覆盖 {len(models)} 个模型、{row_count} 条结果、{fact_count} 条事实。")
    total_rows = sum(model["result_count"] for model in fact_pack["models"])
    total_facts = sum(len(model["facts"]) for model in fact_pack["models"])
    lines.extend(
        [
            "",
            "## 四、Case 级综合研判",
            "",
            f"- 本次覆盖 {len(fact_pack['models'])} 个模型、{total_rows} 条模型结果，生成 {total_facts} 条结构化事实。",
            "- 需要优先核查金额TopK、关系边、时间集中区间和高频主体对应证据。",
            "",
            "## 五、生成备注",
            "",
            f"- 数据来源：{fact_pack['data_source']}",
        ]
    )
    if reason:
        lines.append(f"- LLM 未生成或校验未通过，已降级模板报告：{reason}")
    return "\n".join(lines) + "\n"


def _prompt(case_id: int, compact: dict[str, Any]) -> list[dict[str, str]]:
    payload = json.dumps(compact, ensure_ascii=False, separators=(",", ":"))
    system = (
        "你是案件模型结果分析报告生成助手。你只能使用用户提供的FactPack事实，"
        "不得编造FactPack以外的金额、次数、时间、主体、关系或证据。"
        "关键结论必须引用真实fact_id，引用格式为[fact_id]。"
        "不要输出数据库连接、API Key或系统实现细节。"
    )
    user = f"""
请基于以下压缩FactPack，为case {case_id}生成中文Markdown短报告。

报告必须包含：
1. 模型覆盖概览；
2. 重点模型事实摘要；
3. 模型类级综合；
4. case级综合研判；
5. 后续核查建议。

要求：
- 每个关键结论至少引用一个真实fact_id；
- 不要使用FactPack外的事实；
- 如果事实不足，就说明“当前FactPack不足以支持更强结论”。
- 总长度控制在1200字以内。
- 使用“重点核查对象/重点核查线索”等审慎表述，不直接定性犯罪或嫌疑。

FactPack:
{payload}
""".strip()
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def referenced_fact_ids(report: str) -> set[str]:
    return set(re.findall(r"\[(m\d+_[A-Za-z0-9_]+_[a-f0-9]{10})\]", report))


def all_fact_ids(fact_pack: dict[str, Any]) -> set[str]:
    return {fact["fact_id"] for model in fact_pack["models"] for fact in model["facts"]}


def validate_report_references(report: str, fact_pack: dict[str, Any]) -> tuple[bool, list[str]]:
    known = all_fact_ids(fact_pack)
    refs = referenced_fact_ids(report)
    missing = sorted(refs - known)
    if missing:
        return False, missing
    if not refs and known:
        return False, ["report did not cite any fact_id"]
    if RECOMMENDATION_SECTION not in report:
        return False, ["report appears incomplete: missing final recommendation section"]
    if re.search(r"\[m\d+_[^\]\s]*$", report.strip()):
        return False, ["report appears truncated: unfinished fact_id reference"]
    return True, []


def generate_llm_report(config: Any, case_id: int, fact_pack: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    if not config.llm.is_complete:
        report = render_template_case_report(case_id, fact_pack, "LLM configuration is incomplete")
        return report, {"mode": "template", "status": "fallback", "reason": "LLM configuration is incomplete"}
    try:
        from openai import OpenAI

        compact = compact_fact_pack(fact_pack)
        client = OpenAI(api_key=config.llm.api_key, base_url=config.llm.base_url)
        response = None
        last_error: Exception | None = None
        for attempt in range(3):
            try:
                response = client.chat.completions.create(
                    model=config.llm.model,
                    messages=_prompt(case_id, compact),
                    temperature=config.llm.temperature,
                    max_tokens=min(config.llm.max_tokens, 4096),
                )
                last_error = None
                break
            except Exception as exc:
                last_error = exc
                if "RateLimit" not in type(exc).__name__ and "rate" not in str(exc).lower():
                    raise
                if attempt < 2:
                    time.sleep(10 * (attempt + 1))
        if response is None:
            assert last_error is not None
            raise last_error
        content = response.choices[0].message.content or ""
        valid, missing = validate_report_references(content, fact_pack)
        if not content.strip():
            raise RuntimeError("LLM returned empty content")
        if not valid:
            reason = f"invalid fact_id references: {missing[:5]}"
            report = render_template_case_report(case_id, fact_pack, reason)
            return report, {"mode": "template", "status": "fallback", "reason": reason}
        return content, {"mode": "llm", "status": "success", "model": config.llm.model}
    except Exception as exc:
        detail = str(exc)
        reason = f"{type(exc).__name__}: {detail}" if detail else type(exc).__name__
        report = render_template_case_report(case_id, fact_pack, type(exc).__name__)
        return report, {"mode": "template", "status": "fallback", "reason": reason}

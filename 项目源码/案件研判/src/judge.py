from __future__ import annotations

from typing import Any
import json
import re
import time

from .evidence_chain import is_incomplete_tool_output, make_chain_step, make_planner_decision
from .execution_policy import FallbackPolicy, normalize_fallback_policy
from .llm_utils import cached_chat_json, model_candidates, retry_delay_seconds, update_audit_from_result


def build_judge_steps(
    config: Any,
    hypotheses: list[dict[str, Any]],
    supplement_facts: list[dict[str, Any]],
    chain_steps: list[dict[str, Any]],
    judge_round: int,
    fallback_policy: str | FallbackPolicy = FallbackPolicy.REVIEW_SAFE,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    policy = normalize_fallback_policy(fallback_policy)
    audit = {
        "mode": "llm-judge",
        "status": "not_started",
        "llm_requested": bool(config.llm.is_complete),
        "judge_call_count": 0,
        "judge_success_count": 0,
        "judge_fallback_count": 0,
        "judge_results": [],
        "fallback_policy": policy.value,
        "truncated_downgrade_count": 0,
        "incomplete_downgrade_count": 0,
    }
    steps = []
    facts_by_hyp: dict[str, list[dict[str, Any]]] = {}
    steps_by_hyp: dict[str, list[dict[str, Any]]] = {}
    for fact in supplement_facts:
        facts_by_hyp.setdefault(str(fact.get("hypothesis_id")), []).append(fact)
    for step in chain_steps:
        steps_by_hyp.setdefault(str(step.get("hypothesis_id")), []).append(step)

    for hypothesis in hypotheses:
        hyp_id = str(hypothesis.get("hypothesis_id"))
        supplements = facts_by_hyp.get(hyp_id, [])
        prior_steps = steps_by_hyp.get(hyp_id, [])
        if not supplements:
            result = {
                "status": "unresolved",
                "assessment": "该假设没有形成可用于判断的补证事实，暂未获得进一步支撑。",
                "source": "deterministic_judge_fallback",
                "supporting_supplement_fact_ids": [],
            }
            audit["judge_fallback_count"] += 1
        else:
            result = _judge_single_hypothesis(config, hypothesis, supplements, prior_steps, audit, policy)
        decision = make_planner_decision(
            hyp_id,
            judge_round,
            "llm_judge",
            {
                "supplement_fact_ids": [
                    fact.get("supplement_fact_id") for fact in supplements if fact.get("supplement_fact_id")
                ]
            },
            "综合补证事实判断该假设的证据状态。",
            str(result.get("source") or "llm_judge"),
        )
        steps.append(
            make_chain_step(
                hyp_id,
                judge_round,
                decision,
                None,
                str(result.get("status") or "unresolved"),
                str(result.get("assessment") or ""),
            )
        )
        audit["judge_results"].append(
            {
                "hypothesis_id": hyp_id,
                "status": result.get("status"),
                "original_status": result.get("original_status", result.get("status")),
                "source": result.get("source"),
                "supporting_supplement_fact_ids": result.get("supporting_supplement_fact_ids", []),
            }
        )

    audit["status"] = "success" if audit["judge_success_count"] else "fallback"
    if not hypotheses:
        audit["status"] = "skipped"
    return steps, audit


def _judge_single_hypothesis(
    config: Any,
    hypothesis: dict[str, Any],
    supplements: list[dict[str, Any]],
    prior_steps: list[dict[str, Any]],
    audit: dict[str, Any],
    fallback_policy: FallbackPolicy = FallbackPolicy.REVIEW_SAFE,
) -> dict[str, Any]:
    fallback = _deterministic_judge(
        hypothesis,
        supplements,
        "deterministic_judge_fallback",
        fallback_policy,
    )
    if not config.llm.is_complete:
        audit["judge_fallback_count"] += 1
        return fallback

    brief = _build_judge_brief(hypothesis, supplements, prior_steps)
    audit.setdefault("context_chars", []).append(len(json.dumps(brief, ensure_ascii=False, default=str)))
    payload = _ask_llm_for_judge(config, brief, audit)
    if payload is None:
        audit["judge_fallback_count"] += 1
        return fallback
    status = str(payload.get("status") or "").strip()
    if status not in {"enhanced", "weakened", "unresolved"}:
        audit["judge_fallback_count"] += 1
        return fallback
    assessment = str(payload.get("assessment") or "").strip()
    if assessment and not _is_clear_chinese_assessment(assessment):
        audit["judge_chinese_retry_count"] = int(audit.get("judge_chinese_retry_count") or 0) + 1
        repair_brief = {
            **brief,
            "repair_instruction": "上一次 assessment 不是中文。保持事实和状态不变，只用中文重新输出完整 JSON。",
        }
        repaired = _ask_llm_for_judge(config, repair_brief, audit)
        if isinstance(repaired, dict):
            repaired_status = str(repaired.get("status") or "").strip()
            repaired_assessment = str(repaired.get("assessment") or "").strip()
            if repaired_status in {"enhanced", "weakened", "unresolved"} and _is_clear_chinese_assessment(
                repaired_assessment
            ):
                payload = repaired
                status = repaired_status
                assessment = repaired_assessment
    if not assessment:
        assessment = fallback["assessment"]
    localized_assessment = _localize_internal_terms(assessment)
    if localized_assessment != assessment and _contains_chinese(localized_assessment):
        assessment = localized_assessment
        audit["judge_localization_repair_count"] = int(audit.get("judge_localization_repair_count") or 0) + 1
    if not _is_clear_chinese_assessment(assessment):
        audit["judge_fallback_count"] += 1
        return fallback
    supporting_ids = [
        str(item)
        for item in payload.get("supporting_supplement_fact_ids", [])
        if str(item) in {str(fact.get("supplement_fact_id")) for fact in supplements}
    ]
    incomplete = any(is_incomplete_tool_output(fact.get("output")) for fact in supplements)
    original_status = status
    if incomplete and status != "unresolved":
        status = "unresolved"
        assessment = "补证工具结果被截断或聚合未完成，不能据此增强或削弱该假设，当前需继续核查。"
        audit["truncated_downgrade_count"] = int(audit.get("truncated_downgrade_count") or 0) + 1
        audit["incomplete_downgrade_count"] = int(audit.get("incomplete_downgrade_count") or 0) + 1
        audit.setdefault("incomplete_result_downgrades", []).append(
            {
                "hypothesis_id": hypothesis.get("hypothesis_id"),
                "original_status": original_status,
                "final_status": status,
                "reason": "tool_result_incomplete",
            }
        )
    audit["judge_success_count"] += 1
    return {
        "status": status,
        "assessment": assessment,
        "source": "llm_judge",
        "supporting_supplement_fact_ids": supporting_ids,
        "original_status": original_status if incomplete else status,
    }


def _build_judge_brief(
    hypothesis: dict[str, Any],
    supplements: list[dict[str, Any]],
    prior_steps: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "hypothesis": {
            "hypothesis_id": hypothesis.get("hypothesis_id"),
            "entity_name": hypothesis.get("entity_name"),
            "statement": hypothesis.get("statement"),
            "basis_fact_ids": hypothesis.get("basis_fact_ids", []),
            "target_model_ids": hypothesis.get("target_model_ids", []),
        },
        "supplement_facts": [
            {
                "supplement_fact_id": fact.get("supplement_fact_id"),
                "tool_name": fact.get("tool_name"),
                "params": fact.get("params", {}),
                "summary": fact.get("output", {}).get("summary"),
                "row_count": fact.get("output", {}).get("row_count"),
                "status_signal": fact.get("output", {}).get("status_signal"),
                "structured": fact.get("output", {}).get("structured", {}),
                "anomaly_flags": fact.get("output", {}).get("anomaly_flags", []),
                "rows_read": fact.get("output", {}).get("rows_read"),
                "truncated": bool(fact.get("output", {}).get("truncated")),
                "limit_reason": fact.get("output", {}).get("limit_reason"),
                "aggregation_complete": fact.get("output", {}).get("aggregation_complete", True),
                "fact_refs": fact.get("fact_refs", [])[:8],
                "evidence_refs": fact.get("evidence_refs", [])[:8],
                "key_outputs": _brief_supplement_output(fact.get("output", {})),
            }
            for fact in supplements
        ],
        "prior_tool_assessments": [
            {
                "round": step.get("round"),
                "tool_name": step.get("decision", {}).get("tool_name"),
                "status": step.get("status"),
                "assessment": step.get("assessment"),
            }
            for step in prior_steps
        ],
        "judge_policy": {
            "allowed_statuses": ["enhanced", "weakened", "unresolved"],
            "enhanced": "补证事实与假设方向一致，且能提供主体、关系、金额、时间窗口或对手方层面的实质支撑。",
            "weakened": "补证事实明显削弱或排除该假设。",
            "unresolved": "补证事实仅说明命中或信息不足，不能形成明确增强或削弱。",
            "structured_evidence_first": True,
            "anomaly_flag_rule": "至少2个独立 anomaly_flags，或 self_transfer_dominant、fund_cycle_detected、suspected_split_transfer、structured_in_scattered_out 任一关键标记，可作为 enhanced 的重要依据。",
            "do_not_create_new_facts": True,
            "incomplete_results_must_be_unresolved": True,
        },
    }


def _brief_supplement_output(output: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "top_outgoing",
        "top_incoming",
        "top_same_amount_groups",
        "overlaps",
        "direct_outgoing_top",
        "two_hop_paths",
        "focus_entities",
        "relation_profiles",
        "structured",
        "anomaly_flags",
    ]
    result = {}
    for key in keys:
        value = output.get(key)
        if isinstance(value, list):
            result[key] = value[:3]
        elif value:
            result[key] = value
    return result


def _ask_llm_for_judge(config: Any, brief: dict[str, Any], audit: dict[str, Any]) -> dict[str, Any] | None:
    last_error = ""
    candidates = _judge_model_candidates(config.llm.model)
    audit["model_candidates"] = candidates
    attempt = 0
    for model_name in candidates:
        per_model_attempts = 2 if model_name == config.llm.model else 1
        for _ in range(per_model_attempts):
            attempt += 1
            audit["judge_call_count"] += 1
            payload = _try_llm_judge_once(config, brief, audit, model_name, attempt)
            if payload is not None:
                audit["used_model"] = model_name
                return payload
            last_error = str(audit.get("last_judge_error") or "")
            if attempt < 5:
                time.sleep(int(audit.get("last_retry_delay_seconds") or 4 * min(attempt, 3)))
    audit["last_judge_error"] = last_error
    return None


def _try_llm_judge_once(
    config: Any,
    brief: dict[str, Any],
    audit: dict[str, Any],
    model_name: str,
    attempt: int,
) -> dict[str, Any] | None:
    started = time.perf_counter()
    try:
        prompt = (
            "你是证据链 Judge。只能根据给定假设和 SupplementFact 摘要判断证据状态。"
            "不要创造新主体、金额、时间或关系。不要生成 SQL 或代码。"
            "assessment 必须使用自然中文，不得出现 top_outgoing、top_incoming、row_count、truncated、aggregation_complete 等内部字段名。"
            "如存在 structured 和 anomaly_flags，优先依据结构化字段与异常标记判断；"
            "至少2个独立异常标记，或 self_transfer_dominant、fund_cycle_detected、suspected_split_transfer、structured_in_scattered_out 任一关键标记，可判断为证据增强。"
            "只能输出 JSON：{\"status\":\"enhanced|weakened|unresolved\","
            "\"assessment\":\"中文判断依据\","
            "\"supporting_supplement_fact_ids\":[\"sf_001\"]}。"
            "如果补证仅说明有记录命中但不能证明假设方向，选择 unresolved。"
            "如果任一补证结果 truncated=true 或 aggregation_complete=false，必须选择 unresolved。"
        )
        result = cached_chat_json(
            config,
            model_name=model_name,
            role="judge",
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": json.dumps(brief, ensure_ascii=False, default=str)},
            ],
            max_tokens=min(config.llm.max_tokens, 2048),
        )
        update_audit_from_result(audit, result, model_name, attempt)
        audit["last_latency_seconds"] = round(time.perf_counter() - started, 3)
        if audit.get("last_judge_error"):
            audit["previous_judge_error"] = audit.pop("last_judge_error")
        return result.payload
    except Exception as exc:
        audit["last_attempt_model"] = model_name
        audit["last_judge_error"] = f"{type(exc).__name__}: {str(exc)[:200]}"
        audit["llm_attempts"] = attempt
        audit["last_retry_delay_seconds"] = retry_delay_seconds(exc, attempt)
        audit["last_latency_seconds"] = round(time.perf_counter() - started, 3)
        return None


def _deterministic_judge(
    hypothesis: dict[str, Any],
    supplements: list[dict[str, Any]],
    source: str,
    fallback_policy: FallbackPolicy = FallbackPolicy.REVIEW_SAFE,
) -> dict[str, Any]:
    signals = [str(fact.get("output", {}).get("status_signal") or "") for fact in supplements]
    supporting_ids = [str(fact.get("supplement_fact_id")) for fact in supplements if fact.get("supplement_fact_id")]
    status = "unresolved"
    if any(is_incomplete_tool_output(fact.get("output")) for fact in supplements):
        assessment = "补证结果达到执行预算上限，机械回退不能使用不完整聚合形成增强或削弱结论。"
    elif any(signal == "enhanced" for signal in signals):
        assessment = "补证工具返回了相关命中，但大模型 Judge 未完成，保守回退为待进一步核查。"
    elif any(signal == "weakened" for signal in signals):
        assessment = "补证工具未形成支持信号，但大模型 Judge 未完成，保守回退为待进一步核查。"
    else:
        assessment = "补证结果不足以增强或削弱该假设，机械回退为待进一步核查。"
    return {
        "status": status,
        "assessment": assessment,
        "source": source,
        "supporting_supplement_fact_ids": supporting_ids,
    }


def _judge_model_candidates(default_model: str) -> list[str]:
    return model_candidates(default_model, "judge")


def _contains_chinese(text: str) -> bool:
    return bool(re.search(r"[\u4e00-\u9fff]", text or ""))


def _is_clear_chinese_assessment(text: str) -> bool:
    lowered = (text or "").lower()
    forbidden = [
        "top outgoing",
        "top incoming",
        "top_outgoing",
        "top_incoming",
        "row_count",
        "truncated",
        "aggregation_complete",
        "enhanced",
        "weakened",
        "unresolved",
    ]
    return _contains_chinese(text) and not any(term in lowered for term in forbidden)


def _localize_internal_terms(text: str) -> str:
    replacements = {
        "top outgoing": "主要流出关系",
        "top incoming": "主要流入关系",
        "top_outgoing": "主要流出关系",
        "top_incoming": "主要流入关系",
        "row_count": "记录条数",
        "truncated": "结果截断",
        "aggregation_complete": "聚合完整",
        "enhanced": "证据增强",
        "weakened": "证据削弱",
        "unresolved": "待进一步核查",
    }
    result = text
    for source, target in replacements.items():
        result = re.sub(re.escape(source), target, result, flags=re.I)
    return result

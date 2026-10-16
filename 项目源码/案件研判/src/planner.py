from __future__ import annotations

from typing import Any
import json
import time

from .evidence_chain import (
    build_evidence_chains,
    make_chain_step,
    make_hypothesis,
    make_planner_decision,
    make_supplement_fact,
)
from .judge import build_judge_steps
from .llm_utils import cached_chat_json, model_candidates, retry_delay_seconds, update_audit_from_result
from .supplement_tools import execute_tool, normalize_model_ids
from .tool_registry import TOOL_REGISTRY, allowed_tools_for_strategy, validate_tool_request


def run_planner_demo(
    config: Any,
    case_id: int,
    model_factpacks: list[dict[str, Any]],
    case_pack: dict[str, Any],
    bundle: Any,
    dictionary: Any,
    max_hypotheses: int = 3,
    max_rounds: int = 2,
    max_calls_per_hypothesis: int = 3,
    row_source: Any | None = None,
) -> dict[str, Any]:
    brief = _build_planner_brief(model_factpacks, case_pack, bundle)
    strategies_by_model = bundle.by_model_id
    allowed_entities = _known_entities(model_factpacks)
    allowed_fact_ids = _known_fact_ids(model_factpacks)

    planner_audit = {
        "mode": "planner-demo",
        "status": "not_started",
        "reason": "",
        "brief_chars": len(json.dumps(brief, ensure_ascii=False, default=str)),
        "llm_requested": bool(config.llm.is_complete),
        "llm_valid_request_count": 0,
        "fallback_used": False,
        "deterministic_completion_used": False,
    }

    llm_payload = _ask_llm_for_plan(config, brief, planner_audit) if config.llm.is_complete else None
    hypotheses, requests = _parse_llm_plan(llm_payload, allowed_fact_ids) if llm_payload else ([], [])
    valid_requests = []
    for request in requests:
        ok, reason = validate_tool_request(request, strategies_by_model, allowed_entities)
        if ok:
            valid_requests.append(request)
        else:
            request["validation_error"] = reason

    planner_audit["llm_valid_request_count"] = len(valid_requests)
    if not hypotheses or not valid_requests:
        planner_audit["fallback_used"] = True
        hypotheses, valid_requests = _deterministic_plan(model_factpacks, case_pack, bundle, max_hypotheses)

    hypotheses = hypotheses[:max_hypotheses]
    hypothesis_ids = {item["hypothesis_id"] for item in hypotheses}
    requests = [request for request in valid_requests if request.get("hypothesis_id") in hypothesis_ids]
    covered_hypothesis_ids = {str(request.get("hypothesis_id")) for request in requests}
    missing_hypotheses = [
        hypothesis for hypothesis in hypotheses if str(hypothesis.get("hypothesis_id")) not in covered_hypothesis_ids
    ]
    if missing_hypotheses:
        requests.extend(_requests_for_hypotheses(missing_hypotheses, model_factpacks, bundle))
        planner_audit["deterministic_completion_used"] = True
    if not requests:
        requests = _requests_for_hypotheses(hypotheses, model_factpacks, bundle)

    planner_decisions: list[dict[str, Any]] = []
    supplement_facts: list[dict[str, Any]] = []
    chain_steps: list[dict[str, Any]] = []
    calls_by_hypothesis: dict[str, int] = {}
    round_by_hypothesis: dict[str, int] = {}

    for request in requests:
        hyp_id = str(request.get("hypothesis_id"))
        if calls_by_hypothesis.get(hyp_id, 0) >= max_calls_per_hypothesis:
            continue
        round_index = min(max_rounds, int(request.get("round") or (round_by_hypothesis.get(hyp_id, 0) + 1)))
        round_by_hypothesis[hyp_id] = round_index
        decision = make_planner_decision(
            hyp_id,
            round_index,
            str(request["tool_name"]),
            dict(request["params"]),
            str(request.get("reason") or ""),
            str(request.get("source") or "llm_or_fallback"),
        )
        planner_decisions.append(decision)
        try:
            output = execute_tool(
                decision["tool_name"],
                config,
                case_id,
                strategies_by_model,
                dictionary,
                model_factpacks,
                decision["params"],
                bundle.thresholds,
                row_source=row_source,
            )
            supplement = make_supplement_fact(len(supplement_facts) + 1, hyp_id, decision, output)
            supplement_facts.append(supplement)
            status = _status_from_output(output)
            chain_steps.append(
                make_chain_step(
                    hyp_id,
                    round_index,
                    decision,
                    supplement,
                    status,
                    _assessment_text(output),
                )
            )
            calls_by_hypothesis[hyp_id] = calls_by_hypothesis.get(hyp_id, 0) + 1
            if round_index < max_rounds and calls_by_hypothesis[hyp_id] < max_calls_per_hypothesis:
                followup = _deterministic_round_two(hyp_id, decision, output, model_factpacks)
                if followup:
                    ok, _ = validate_tool_request(followup, strategies_by_model, allowed_entities)
                    if ok:
                        requests.append(followup)
        except Exception as exc:
            chain_steps.append(
                make_chain_step(
                    hyp_id,
                    round_index,
                    decision,
                    None,
                    "unresolved",
                    f"工具执行失败：{type(exc).__name__}",
                )
            )

    evidence_chains = build_evidence_chains(hypotheses, chain_steps)
    judge_steps, judge_audit = build_judge_steps(
        config,
        hypotheses,
        supplement_facts,
        chain_steps,
        max_rounds + 1,
    )
    chain_steps.extend(judge_steps)
    evidence_chains = build_evidence_chains(hypotheses, chain_steps)
    planner_audit["judge_audit"] = judge_audit
    planner_audit["status"] = "success" if supplement_facts else "fallback"
    if not supplement_facts and not planner_audit["reason"]:
        planner_audit["reason"] = "no supplement facts were generated"

    return {
        "llm_brief_pack": brief,
        "planner_audit": planner_audit,
        "hypotheses": hypotheses,
        "planner_decisions": planner_decisions,
        "supplement_facts": supplement_facts,
        "evidence_chain_steps": chain_steps,
        "evidence_chains": evidence_chains,
        "model_report_packs": _build_model_report_packs(model_factpacks, supplement_facts, evidence_chains),
    }


def run_model_class_planner_demo(
    config: Any,
    case_id: int,
    model_factpacks: list[dict[str, Any]],
    model_class_packs: list[dict[str, Any]],
    case_pack: dict[str, Any],
    bundle: Any,
    dictionary: Any,
    max_hypotheses: int = 6,
    max_hypotheses_per_class: int = 2,
    max_rounds: int = 2,
    max_calls_per_hypothesis: int = 3,
    row_source: Any | None = None,
) -> dict[str, Any]:
    reasoning_class_packs = [
        pack for pack in model_class_packs if int(pack.get("reasoning_model_count") or 0) > 0
    ]
    strategies_by_model = bundle.by_model_id
    all_allowed_entities = _known_entities(model_factpacks)
    all_allowed_fact_ids = _known_fact_ids(model_factpacks)
    fact_model_ids = _fact_model_ids(model_factpacks)
    class_briefs: list[dict[str, Any]] = []

    planner_audit = {
        "mode": "class-planner-demo",
        "status": "not_started",
        "reason": "",
        "brief_chars": 0,
        "llm_requested": bool(config.llm.is_complete),
        "llm_valid_request_count": 0,
        "fallback_used": False,
        "deterministic_completion_used": False,
        "planner_context": "model_class_pack_per_class",
        "model_class_count": len(model_class_packs),
        "reasoning_model_class_count": len(reasoning_class_packs),
        "llm_call_count": 0,
        "class_runs": [],
    }

    hypotheses: list[dict[str, Any]] = []
    valid_requests: list[dict[str, Any]] = []
    used_hypothesis_ids: set[str] = set()

    for class_index, class_pack in enumerate(reasoning_class_packs, start=1):
        class_name = str(class_pack.get("class_name") or f"class_{class_index}")
        class_key = f"class{class_index:02d}"
        class_brief = _build_model_class_planner_brief([class_pack], case_pack)
        class_briefs.append(class_brief)
        class_allowed_fact_ids = _class_fact_ids(class_pack, all_allowed_fact_ids)
        class_allowed_entities = _class_entities(class_pack) or all_allowed_entities
        class_audit = {
            "mode": "class-planner-demo",
            "status": "not_started",
            "reason": "",
            "brief_chars": len(json.dumps(class_brief, ensure_ascii=False, default=str)),
            "llm_requested": bool(config.llm.is_complete),
            "llm_valid_request_count": 0,
            "fallback_used": False,
            "deterministic_completion_used": False,
            "planner_context": "single_model_class_pack",
            "class_name": class_name,
        }
        planner_audit["brief_chars"] += class_audit["brief_chars"]

        llm_payload = _ask_llm_for_plan(config, class_brief, class_audit) if config.llm.is_complete else None
        class_hypotheses, class_requests = (
            _parse_llm_plan(llm_payload, class_allowed_fact_ids) if llm_payload else ([], [])
        )
        if llm_payload is not None:
            planner_audit["llm_call_count"] += 1
        for hypothesis in class_hypotheses:
            if not hypothesis.get("target_model_ids"):
                hypothesis["target_model_ids"] = _target_model_ids_from_facts(
                    hypothesis.get("basis_fact_ids", []),
                    fact_model_ids,
                )

        class_valid_requests = []
        for request in class_requests:
            _normalize_request_params(request)
            ok, reason = validate_tool_request(request, strategies_by_model, class_allowed_entities)
            if ok:
                class_valid_requests.append(request)
            else:
                request["validation_error"] = reason

        class_audit["llm_valid_request_count"] = len(class_valid_requests)
        if not class_hypotheses or not class_valid_requests:
            class_audit["fallback_used"] = True
            class_hypotheses, class_valid_requests = _deterministic_class_plan(
                [class_pack],
                fact_model_ids,
                max_hypotheses_per_class,
            )

        class_hypotheses, class_valid_requests = _namespace_hypothesis_ids(
            class_hypotheses[:max_hypotheses_per_class],
            class_valid_requests,
            class_key,
            used_hypothesis_ids,
        )
        class_hypothesis_ids = {item["hypothesis_id"] for item in class_hypotheses}
        class_valid_requests = [
            request for request in class_valid_requests if request.get("hypothesis_id") in class_hypothesis_ids
        ]
        covered_hypothesis_ids = {str(request.get("hypothesis_id")) for request in class_valid_requests}
        missing_hypotheses = [
            hypothesis
            for hypothesis in class_hypotheses
            if str(hypothesis.get("hypothesis_id")) not in covered_hypothesis_ids
        ]
        if missing_hypotheses:
            class_valid_requests.extend(_requests_for_hypotheses(missing_hypotheses, model_factpacks, bundle))
            class_audit["deterministic_completion_used"] = True

        planner_audit["llm_valid_request_count"] += len(class_valid_requests)
        planner_audit["fallback_used"] = planner_audit["fallback_used"] or class_audit["fallback_used"]
        planner_audit["deterministic_completion_used"] = (
            planner_audit["deterministic_completion_used"] or class_audit["deterministic_completion_used"]
        )
        planner_audit["class_runs"].append(class_audit)
        hypotheses.extend(class_hypotheses)
        valid_requests.extend(class_valid_requests)

    hypotheses = hypotheses[:max_hypotheses]
    hypothesis_ids = {item["hypothesis_id"] for item in hypotheses}
    requests = [request for request in valid_requests if request.get("hypothesis_id") in hypothesis_ids]
    if not requests:
        requests = _requests_for_hypotheses(hypotheses, model_factpacks, bundle)

    planner_decisions: list[dict[str, Any]] = []
    supplement_facts: list[dict[str, Any]] = []
    chain_steps: list[dict[str, Any]] = []
    calls_by_hypothesis: dict[str, int] = {}
    round_by_hypothesis: dict[str, int] = {}

    for request in requests:
        hyp_id = str(request.get("hypothesis_id"))
        if calls_by_hypothesis.get(hyp_id, 0) >= max_calls_per_hypothesis:
            continue
        round_index = min(max_rounds, int(request.get("round") or (round_by_hypothesis.get(hyp_id, 0) + 1)))
        round_by_hypothesis[hyp_id] = round_index
        decision = make_planner_decision(
            hyp_id,
            round_index,
            str(request["tool_name"]),
            dict(request["params"]),
            str(request.get("reason") or ""),
            str(request.get("source") or "llm_or_fallback"),
        )
        planner_decisions.append(decision)
        try:
            output = execute_tool(
                decision["tool_name"],
                config,
                case_id,
                strategies_by_model,
                dictionary,
                model_factpacks,
                decision["params"],
                bundle.thresholds,
                row_source=row_source,
            )
            supplement = make_supplement_fact(len(supplement_facts) + 1, hyp_id, decision, output)
            supplement_facts.append(supplement)
            status = _status_from_output(output)
            chain_steps.append(
                make_chain_step(
                    hyp_id,
                    round_index,
                    decision,
                    supplement,
                    status,
                    _assessment_text(output),
                )
            )
            calls_by_hypothesis[hyp_id] = calls_by_hypothesis.get(hyp_id, 0) + 1
            if round_index < max_rounds and calls_by_hypothesis[hyp_id] < max_calls_per_hypothesis:
                followup = _deterministic_round_two(hyp_id, decision, output, model_factpacks)
                if followup:
                    ok, _ = validate_tool_request(followup, strategies_by_model, all_allowed_entities)
                    if ok:
                        requests.append(followup)
        except Exception as exc:
            chain_steps.append(
                make_chain_step(
                    hyp_id,
                    round_index,
                    decision,
                    None,
                    "unresolved",
                    f"工具执行失败：{type(exc).__name__}",
                )
            )

    evidence_chains = build_evidence_chains(hypotheses, chain_steps)
    judge_steps, judge_audit = build_judge_steps(
        config,
        hypotheses,
        supplement_facts,
        chain_steps,
        max_rounds + 1,
    )
    chain_steps.extend(judge_steps)
    evidence_chains = build_evidence_chains(hypotheses, chain_steps)
    planner_audit["judge_audit"] = judge_audit
    planner_audit["status"] = "success" if supplement_facts else "fallback"
    if not supplement_facts and not planner_audit["reason"]:
        planner_audit["reason"] = "no supplement facts were generated"

    return {
        "llm_brief_pack": _build_per_class_planner_brief_pack(case_pack, class_briefs, planner_audit),
        "planner_audit": planner_audit,
        "hypotheses": hypotheses,
        "planner_decisions": planner_decisions,
        "supplement_facts": supplement_facts,
        "evidence_chain_steps": chain_steps,
        "evidence_chains": evidence_chains,
        "model_report_packs": _build_model_report_packs(model_factpacks, supplement_facts, evidence_chains),
    }


run_model_class_planner = run_model_class_planner_demo


def _ask_llm_for_plan(config: Any, brief: dict[str, Any], audit: dict[str, Any]) -> dict[str, Any] | None:
    last_error = ""
    model_candidates = _planner_model_candidates(config.llm.model)
    audit["model_candidates"] = model_candidates
    attempt = 0
    for model_name in model_candidates:
        per_model_attempts = 2 if model_name == config.llm.model else 1
        for _ in range(per_model_attempts):
            attempt += 1
            payload = _try_llm_plan_once(config, brief, audit, model_name, attempt)
            if payload is not None:
                audit["used_model"] = model_name
                return payload
            last_error = str(audit.get("last_llm_error") or "")
            if attempt < 5:
                time.sleep(int(audit.get("last_retry_delay_seconds") or 4 * min(attempt, 3)))
    audit["status"] = "llm_plan_fallback"
    audit["reason"] = last_error
    return None


def _try_llm_plan_once(
    config: Any,
    brief: dict[str, Any],
    audit: dict[str, Any],
    model_name: str,
    attempt: int,
) -> dict[str, Any] | None:
    try:
        allowed_tools = sorted(TOOL_REGISTRY)
        prompt = (
            "你是案件模型结果研判助手。只能根据 BriefPack 选择关注对象和下一步核查动作。"
            "不要生成 SQL，不要编造 BriefPack 外的主体、金额、模型或 fact_id。"
            f"可用工具：{', '.join(allowed_tools)}。"
            "输出 JSON：{\"hypothesis_candidates\":[{\"hypothesis_id\":\"hyp_01\",\"entity_name\":\"...\","
            "\"statement\":\"...\",\"basis_fact_ids\":[\"...\"],\"target_model_ids\":[86]}],"
            "\"followup_requests\":[{\"hypothesis_id\":\"hyp_01\",\"round\":1,\"tool_name\":\"entity_model_slice\","
            "\"params\":{\"model_id\":43,\"entity_name\":\"...\"},\"reason\":\"...\"}]}"
            "最多 3 个假设、每个假设最多 2 个工具请求。只输出 JSON，不要输出推理过程。"
        )
        result = cached_chat_json(
            config,
            model_name=model_name,
            role="planner",
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": json.dumps(brief, ensure_ascii=False, default=str)},
            ],
            max_tokens=min(config.llm.max_tokens, 4096),
        )
        update_audit_from_result(audit, result, model_name, attempt)
        if audit.get("last_llm_error"):
            audit["previous_llm_error"] = audit.pop("last_llm_error")
        audit["status"] = "llm_plan_received"
        return result.payload
    except Exception as exc:
        audit["status"] = "llm_plan_retrying"
        audit["last_attempt_model"] = model_name
        audit["last_llm_error"] = f"{type(exc).__name__}: {str(exc)[:200]}"
        audit["llm_attempts"] = attempt
        audit["last_retry_delay_seconds"] = retry_delay_seconds(exc, attempt)
        return None


def _planner_model_candidates(default_model: str) -> list[str]:
    return model_candidates(default_model, "planner")


def _build_planner_brief(model_factpacks: list[dict[str, Any]], case_pack: dict[str, Any], bundle: Any) -> dict[str, Any]:
    models = []
    for fp in model_factpacks:
        context = fp.get("model_context", {})
        model_id = int(context.get("model_id"))
        strategy = bundle.by_model_id.get(model_id, {})
        models.append(
            {
                "model_id": model_id,
                "model_name": context.get("model_name"),
                "model_kind": context.get("model_kind"),
                "demo_role": context.get("demo_role"),
                "result_count": context.get("result_count"),
                "allowed_tools": allowed_tools_for_strategy(strategy),
                "focus_entities": [
                    {
                        "fact_id": entity.get("fact_id"),
                        "rank": entity.get("rank"),
                        "entity_name": entity.get("entity_name"),
                        "record_count": entity.get("metrics", {}).get("record_count"),
                        "amount_sum": entity.get("metrics", {}).get("amount_sum"),
                        "risk_tags": [flag.get("risk_tag") for flag in entity.get("anomaly_flags", [])],
                    }
                    for entity in fp.get("focus_entities", [])[:2]
                ],
                "relations": [
                    {
                        "fact_id": relation.get("fact_id"),
                        "rank": relation.get("rank"),
                        "source": relation.get("source"),
                        "target": relation.get("target"),
                        "count": relation.get("count"),
                        "amount_sum": relation.get("amount_sum"),
                        "risk_tags": [flag.get("risk_tag") for flag in relation.get("anomaly_flags", [])],
                    }
                    for relation in fp.get("relation_profiles", [])[:1]
                ],
            }
        )
    return {
        "case_id": case_pack.get("case_id"),
        "models": models,
        "cross_model_entities": [
            {
                "entity_name": item.get("entity_name"),
                "model_count": item.get("model_count"),
                "total_amount_sum": item.get("total_amount_sum"),
                "risk_tags": item.get("risk_tags", [])[:3],
                "fact_refs": [row.get("fact_id") for row in item.get("models", [])[:5] if row.get("fact_id")],
                "model_ids": [row.get("model_id") for row in item.get("models", [])[:5] if row.get("model_id")],
            }
            for item in case_pack.get("cross_model_entities", [])[:3]
        ],
        "tool_policy": {
            "allowed_tool_names": sorted(TOOL_REGISTRY),
            "sql_generation_allowed": False,
            "params_must_use_known_entities": True,
            "max_hypotheses": 3,
            "max_requests_per_hypothesis": 2,
        },
    }


def _build_model_class_planner_brief(
    model_class_packs: list[dict[str, Any]],
    case_pack: dict[str, Any],
) -> dict[str, Any]:
    classes = []
    for pack in model_class_packs:
        if int(pack.get("reasoning_model_count") or 0) <= 0:
            continue
        classes.append(
            {
                "class_name": pack.get("class_name"),
                "model_count": pack.get("model_count"),
                "active_model_count": pack.get("active_model_count"),
                "reasoning_model_ids": pack.get("reasoning_model_ids", []),
                "total_result_count": pack.get("total_result_count"),
                "available_tools": pack.get("available_tools", []),
                "models": [
                    {
                        "model_id": model.get("model_id"),
                        "model_name": model.get("model_name"),
                        "model_kind": model.get("model_kind"),
                        "demo_level": model.get("demo_level"),
                        "result_count": model.get("result_count"),
                        "allowed_tools": model.get("allowed_tools", []),
                        "shared_entity_types": model.get("shared_entity_types", []),
                    }
                    for model in pack.get("models", [])
                    if int(model.get("result_count") or 0) > 0
                ][:6],
                "shared_focus_entities": [
                    {
                        "entity_name": entity.get("entity_name"),
                        "cross_model_count": entity.get("cross_model_count"),
                        "total_amount_sum": entity.get("total_amount_sum"),
                        "risk_tags": entity.get("risk_tags", [])[:5],
                        "risk_tags_cn": entity.get("risk_tags_cn", [])[:5],
                        "appearances": [
                            {
                                "model_id": item.get("model_id"),
                                "model_name": item.get("model_name"),
                                "rank": item.get("rank"),
                                "fact_id": item.get("fact_id"),
                                "record_count": item.get("record_count"),
                                "amount_sum": item.get("amount_sum"),
                                "risk_tags": item.get("risk_tags", [])[:5],
                            }
                            for item in entity.get("appearances", [])[:5]
                        ],
                    }
                    for entity in pack.get("shared_focus_entities", [])[:5]
                ],
                "shared_relation_entities": [
                    {
                        "entity_name": entity.get("entity_name"),
                        "cross_model_count": entity.get("cross_model_count"),
                        "appearance_count": entity.get("appearance_count"),
                        "risk_tags": entity.get("risk_tags", [])[:5],
                        "risk_tags_cn": entity.get("risk_tags_cn", [])[:5],
                        "appearances": [
                            {
                                "model_id": item.get("model_id"),
                                "model_name": item.get("model_name"),
                                "role": item.get("role"),
                                "rank": item.get("rank"),
                                "fact_id": item.get("fact_id"),
                                "source": item.get("source"),
                                "target": item.get("target"),
                                "count": item.get("count"),
                                "amount_sum": item.get("amount_sum"),
                            }
                            for item in entity.get("appearances", [])[:5]
                        ],
                    }
                    for entity in pack.get("shared_relation_entities", [])[:5]
                ],
                "class_risk_patterns": pack.get("class_risk_patterns", [])[:5],
                "candidate_hypotheses": pack.get("candidate_hypotheses", [])[:5],
            }
        )
    return {
        "case_id": case_pack.get("case_id"),
        "planner_context": "model_class_pack",
        "classes": classes[:4],
        "tool_policy": {
            "allowed_tool_names": sorted(TOOL_REGISTRY),
            "sql_generation_allowed": False,
            "params_must_use_known_entities": True,
            "params_must_use_fact_ids_from_model_class_pack": True,
            "max_hypotheses": 3,
            "max_requests_per_hypothesis": 2,
        },
    }


def _build_per_class_planner_brief_pack(
    case_pack: dict[str, Any],
    class_briefs: list[dict[str, Any]],
    planner_audit: dict[str, Any],
) -> dict[str, Any]:
    return {
        "case_id": case_pack.get("case_id"),
        "planner_context": "model_class_pack_per_class",
        "llm_call_count": planner_audit.get("llm_call_count", 0),
        "class_brief_count": len(class_briefs),
        "class_briefs": class_briefs,
        "tool_policy": {
            "single_llm_call_scope": "one_model_class_pack",
            "sql_generation_allowed": False,
            "params_must_use_known_entities": True,
        },
    }


def _parse_llm_plan(payload: dict[str, Any], allowed_fact_ids: set[str]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    hypotheses = []
    requests = []
    if not isinstance(payload, dict):
        return hypotheses, requests
    for index, item in enumerate(payload.get("hypothesis_candidates", [])[:3], start=1):
        if not isinstance(item, dict):
            continue
        fact_ids = [str(fid) for fid in item.get("basis_fact_ids", []) if str(fid) in allowed_fact_ids]
        if not fact_ids:
            continue
        hyp_id = str(item.get("hypothesis_id") or f"hyp_{index:02d}")
        hypotheses.append(
            {
                "hypothesis_id": hyp_id,
                "entity_name": item.get("entity_name"),
                "statement": str(item.get("statement") or ""),
                "basis_fact_ids": fact_ids,
                "target_model_ids": [int(mid) for mid in item.get("target_model_ids", []) if str(mid).isdigit()],
                "status": "candidate",
                "source": "llm",
            }
        )
    hyp_ids = {item["hypothesis_id"] for item in hypotheses}
    for item in payload.get("followup_requests", []):
        if not isinstance(item, dict) or item.get("hypothesis_id") not in hyp_ids:
            continue
        requests.append(
            {
                "hypothesis_id": item.get("hypothesis_id"),
                "round": int(item.get("round") or 1),
                "tool_name": item.get("tool_name"),
                "params": _normalize_params(item.get("params") or {}),
                "reason": item.get("reason") or "",
                "source": "llm",
            }
        )
    return hypotheses, requests


def _deterministic_class_plan(
    model_class_packs: list[dict[str, Any]],
    fact_model_ids: dict[str, int],
    max_hypotheses: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    hypotheses = []
    for pack in model_class_packs:
        if int(pack.get("reasoning_model_count") or 0) <= 0:
            continue
        for item in pack.get("candidate_hypotheses", []):
            fact_ids = [str(fid) for fid in item.get("basis_fact_ids", []) if str(fid) in fact_model_ids]
            if not fact_ids:
                continue
            hypotheses.append(
                {
                    "hypothesis_id": str(item.get("hypothesis_id") or f"class_hyp_{len(hypotheses) + 1:02d}"),
                    "entity_name": item.get("entity_name"),
                    "statement": str(item.get("statement") or ""),
                    "basis_fact_ids": fact_ids,
                    "target_model_ids": _target_model_ids_from_facts(fact_ids, fact_model_ids),
                    "status": "candidate",
                    "source": str(item.get("source") or "model_class_pack"),
                }
            )
            if len(hypotheses) >= max_hypotheses:
                break
        if len(hypotheses) >= max_hypotheses:
            break

    requests = []
    for hypothesis in hypotheses:
        hyp_id = hypothesis["hypothesis_id"]
        entity_name = hypothesis.get("entity_name")
        target_model_ids = hypothesis.get("target_model_ids", [])
        if not entity_name:
            continue
        for model_id in target_model_ids[:2]:
            requests.append(
                _request(
                    hyp_id,
                    1,
                    "entity_model_slice",
                    {"model_id": model_id, "entity_name": entity_name},
                    "基于模型类级候选假设，核查该主体在相关模型中的定向切片。",
                )
            )
        if target_model_ids:
            requests.append(
                _request(
                    hyp_id,
                    2,
                    "counterparty_overlap_probe",
                    {"entity_name": entity_name, "model_ids": target_model_ids},
                    "比较该主体在模型类内多个模型中的对手方重合情况。",
                )
            )
    return hypotheses, requests


def _namespace_hypothesis_ids(
    hypotheses: list[dict[str, Any]],
    requests: list[dict[str, Any]],
    class_key: str,
    used_hypothesis_ids: set[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    id_map = {}
    for index, hypothesis in enumerate(hypotheses, start=1):
        old_id = str(hypothesis.get("hypothesis_id") or f"hyp_{index:02d}")
        if old_id.startswith("class_") and old_id not in used_hypothesis_ids:
            new_id = old_id
        else:
            new_id = f"{class_key}_{old_id}"
            suffix = 2
            while new_id in used_hypothesis_ids:
                new_id = f"{class_key}_{old_id}_{suffix}"
                suffix += 1
        hypothesis["hypothesis_id"] = new_id
        used_hypothesis_ids.add(new_id)
        id_map[old_id] = new_id

    for request in requests:
        old_id = str(request.get("hypothesis_id") or "")
        if old_id in id_map:
            request["hypothesis_id"] = id_map[old_id]
    return hypotheses, requests


def _deterministic_plan(
    model_factpacks: list[dict[str, Any]],
    case_pack: dict[str, Any],
    bundle: Any,
    max_hypotheses: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    hypotheses = []
    requests = []
    cross_entities = case_pack.get("cross_model_entities", [])
    for index, entity in enumerate(cross_entities[:max_hypotheses], start=1):
        entity_name = entity.get("entity_name")
        fact_ids = [row.get("fact_id") for row in entity.get("models", []) if row.get("fact_id")]
        model_ids = [int(row["model_id"]) for row in entity.get("models", []) if row.get("model_id")]
        hypotheses.append(
            make_hypothesis(
                index,
                entity_name,
                f"{entity_name} 在多个模型中重复出现，需要验证其高金额/高频/关系模式是否互相印证。",
                fact_ids,
                "deterministic_fallback",
                model_ids,
            )
        )
    if not hypotheses:
        hypotheses = _relation_hypotheses(model_factpacks, max_hypotheses)
    requests = _requests_for_hypotheses(hypotheses, model_factpacks, bundle)
    return hypotheses, requests


def _relation_hypotheses(model_factpacks: list[dict[str, Any]], max_hypotheses: int) -> list[dict[str, Any]]:
    result = []
    for factpack in model_factpacks:
        context = factpack.get("model_context", {})
        for relation in factpack.get("relation_profiles", [])[:2]:
            result.append(
                make_hypothesis(
                    len(result) + 1,
                    relation.get("source"),
                    f"{relation.get('source')} -> {relation.get('target')} 是高金额关系边，需要验证邻域和两跳路径。",
                    [relation.get("fact_id")],
                    "deterministic_relation",
                    [int(context.get("model_id"))],
                )
            )
            if len(result) >= max_hypotheses:
                return result
    return result


def _requests_for_hypotheses(
    hypotheses: list[dict[str, Any]],
    model_factpacks: list[dict[str, Any]],
    bundle: Any,
) -> list[dict[str, Any]]:
    factpacks_by_model = {int(fp["model_context"]["model_id"]): fp for fp in model_factpacks}
    requests = []
    for hypothesis in hypotheses:
        hyp_id = hypothesis["hypothesis_id"]
        entity_name = hypothesis.get("entity_name")
        target_model_ids = hypothesis.get("target_model_ids") or []
        if not target_model_ids:
            target_model_ids = [
                int(fp["model_context"]["model_id"])
                for fp in model_factpacks
                if any(entity.get("entity_name") == entity_name for entity in fp.get("focus_entities", []))
            ]
        for model_id in target_model_ids[:3]:
            strategy = bundle.by_model_id.get(int(model_id), {})
            factpack = factpacks_by_model.get(int(model_id), {})
            allowed = allowed_tools_for_strategy(strategy)
            model_kind = strategy.get("model_kind")
            if model_kind == "relation" and "relation_neighbor_subgraph" in allowed:
                requests.append(
                    _request(hyp_id, 1, "relation_neighbor_subgraph", {"model_id": model_id, "entity_name": entity_name}, "验证该主体在关系模型中的上下游邻域。")
                )
                if "fund_flow_path_probe" in allowed:
                    relation = (factpack.get("relation_profiles") or [{}])[0]
                    source = relation.get("source") or entity_name
                    target = relation.get("target")
                    requests.append(
                        _request(hyp_id, 1, "fund_flow_path_probe", {"model_id": model_id, "source": source, "target": target}, "验证高金额关系边是否存在两跳资金路径。")
                    )
            elif model_id == 70 and "same_amount_window_probe" in allowed:
                requests.append(
                    _request(hyp_id, 1, "same_amount_window_probe", {"model_id": model_id, "entity_name": entity_name}, "验证同日同额转账窗口。")
                )
            elif "entity_model_slice" in allowed:
                requests.append(
                    _request(hyp_id, 1, "entity_model_slice", {"model_id": model_id, "entity_name": entity_name}, "验证该主体在模型内的定向切片画像。")
                )
        requests.append(
            _request(hyp_id, 2, "counterparty_overlap_probe", {"entity_name": entity_name, "model_ids": target_model_ids}, "比较该主体跨模型 Top 对手是否重合。")
        )
    return requests


def _deterministic_round_two(
    hyp_id: str,
    decision: dict[str, Any],
    output: dict[str, Any],
    model_factpacks: list[dict[str, Any]],
) -> dict[str, Any] | None:
    params = decision.get("params", {})
    entity_name = params.get("entity_name") or params.get("source")
    if not entity_name or decision.get("tool_name") == "counterparty_overlap_probe":
        return None
    model_ids = [
        int(fp["model_context"]["model_id"])
        for fp in model_factpacks
        if any(entity.get("entity_name") == entity_name for entity in fp.get("focus_entities", []))
    ]
    if len(model_ids) < 2:
        return None
    return _request(
        hyp_id,
        2,
        "counterparty_overlap_probe",
        {"entity_name": entity_name, "model_ids": model_ids},
        "第一轮补证命中后，比较跨模型对手方重合情况。",
    )


def _request(hypothesis_id: str, round_index: int, tool_name: str, params: dict[str, Any], reason: str) -> dict[str, Any]:
    return {
        "hypothesis_id": hypothesis_id,
        "round": round_index,
        "tool_name": tool_name,
        "params": _normalize_params(params),
        "reason": reason,
        "source": "deterministic_fallback",
    }


def _normalize_request_params(request: dict[str, Any]) -> None:
    request["params"] = _normalize_params(request.get("params") or {})


def _normalize_params(params: dict[str, Any]) -> dict[str, Any]:
    result = dict(params)
    if "model_ids" in result:
        result["model_ids"] = normalize_model_ids(result.get("model_ids"))
    if "model_id" in result and result.get("model_id") not in [None, ""]:
        try:
            result["model_id"] = int(result["model_id"])
        except Exception:
            pass
    return result


def _known_entities(model_factpacks: list[dict[str, Any]]) -> set[str]:
    values: set[str] = set()
    for fp in model_factpacks:
        for entity in fp.get("focus_entities", []):
            if entity.get("entity_name"):
                values.add(str(entity["entity_name"]))
        for relation in fp.get("relation_profiles", []):
            if relation.get("source"):
                values.add(str(relation["source"]))
            if relation.get("target"):
                values.add(str(relation["target"]))
    return values


def _known_fact_ids(model_factpacks: list[dict[str, Any]]) -> set[str]:
    values: set[str] = set()
    for fp in model_factpacks:
        for entity in fp.get("focus_entities", []):
            if entity.get("fact_id"):
                values.add(str(entity["fact_id"]))
        for relation in fp.get("relation_profiles", []):
            if relation.get("fact_id"):
                values.add(str(relation["fact_id"]))
    return values


def _class_fact_ids(class_pack: dict[str, Any], known_fact_ids: set[str]) -> set[str]:
    values: set[str] = set()
    for entity in class_pack.get("shared_focus_entities", []):
        for appearance in entity.get("appearances", []):
            if appearance.get("fact_id"):
                values.add(str(appearance["fact_id"]))
    for entity in class_pack.get("shared_relation_entities", []):
        for appearance in entity.get("appearances", []):
            if appearance.get("fact_id"):
                values.add(str(appearance["fact_id"]))
    for entity in class_pack.get("class_specific_entities", []):
        if entity.get("fact_id"):
            values.add(str(entity["fact_id"]))
    for hypothesis in class_pack.get("candidate_hypotheses", []):
        values.update(str(fact_id) for fact_id in hypothesis.get("basis_fact_ids", []) if fact_id)
    return values & known_fact_ids


def _class_entities(class_pack: dict[str, Any]) -> set[str]:
    values: set[str] = set()
    for entity in class_pack.get("shared_focus_entities", []):
        if entity.get("entity_name"):
            values.add(str(entity["entity_name"]))
        for appearance in entity.get("appearances", []):
            for key in ("source", "target"):
                if appearance.get(key):
                    values.add(str(appearance[key]))
    for entity in class_pack.get("shared_relation_entities", []):
        if entity.get("entity_name"):
            values.add(str(entity["entity_name"]))
        for appearance in entity.get("appearances", []):
            for key in ("source", "target"):
                if appearance.get(key):
                    values.add(str(appearance[key]))
    for entity in class_pack.get("class_specific_entities", []):
        if entity.get("entity_name"):
            values.add(str(entity["entity_name"]))
    for hypothesis in class_pack.get("candidate_hypotheses", []):
        if hypothesis.get("entity_name"):
            values.add(str(hypothesis["entity_name"]))
    return values


def _fact_model_ids(model_factpacks: list[dict[str, Any]]) -> dict[str, int]:
    result: dict[str, int] = {}
    for fp in model_factpacks:
        model_id = int(fp["model_context"]["model_id"])
        for entity in fp.get("focus_entities", []):
            if entity.get("fact_id"):
                result[str(entity["fact_id"])] = model_id
        for relation in fp.get("relation_profiles", []):
            if relation.get("fact_id"):
                result[str(relation["fact_id"])] = model_id
    return result


def _target_model_ids_from_facts(fact_ids: list[str], fact_model_ids: dict[str, int]) -> list[int]:
    result = []
    for fact_id in fact_ids:
        model_id = fact_model_ids.get(str(fact_id))
        if model_id is not None and model_id not in result:
            result.append(model_id)
    return result


def _status_from_output(output: dict[str, Any]) -> str:
    signal = str(output.get("status_signal") or "")
    if signal in {"enhanced", "weakened", "unresolved", "skipped"}:
        return signal
    row_count = int(output.get("row_count") or 0)
    return "enhanced" if row_count else "unresolved"


def _assessment_text(output: dict[str, Any]) -> str:
    summary = str(output.get("summary") or "")
    signal = str(output.get("status_signal") or "unresolved")
    signal_cn = {
        "enhanced": "证据增强",
        "weakened": "证据削弱",
        "unresolved": "待进一步核查",
        "skipped": "未执行补证",
    }.get(signal, signal)
    return f"{summary} 补证信号：{signal_cn}。"


def _build_model_report_packs(
    model_factpacks: list[dict[str, Any]],
    supplement_facts: list[dict[str, Any]],
    evidence_chains: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    chains_by_hyp = {chain.get("hypothesis", {}).get("hypothesis_id"): chain for chain in evidence_chains}
    result = []
    for factpack in model_factpacks:
        context = factpack.get("model_context", {})
        model_id = int(context.get("model_id"))
        supplements = [
            item
            for item in supplement_facts
            if item.get("params", {}).get("model_id") is not None
            and int(item.get("params", {}).get("model_id")) == model_id
        ]
        chain_ids = sorted({f"chain_{item.get('hypothesis_id')}" for item in supplements})
        result.append(
            {
                "model_id": model_id,
                "model_name": context.get("model_name"),
                "base_fact_ids": [
                    *[item.get("fact_id") for item in factpack.get("focus_entities", []) if item.get("fact_id")],
                    *[item.get("fact_id") for item in factpack.get("relation_profiles", []) if item.get("fact_id")],
                ],
                "supplement_fact_ids": [item.get("supplement_fact_id") for item in supplements],
                "evidence_chain_ids": chain_ids,
                "final_statuses": [
                    chains_by_hyp.get(chain_id.replace("chain_", ""), {}).get("final_status")
                    for chain_id in chain_ids
                ],
            }
        )
    return result

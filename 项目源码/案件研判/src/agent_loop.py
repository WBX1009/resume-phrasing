from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import json
import re
import time

from .evidence_chain import build_evidence_chains, make_chain_step, make_planner_decision, make_supplement_fact
from .execution_policy import FallbackPolicy, normalize_fallback_policy
from .judge import build_judge_steps
from .llm_utils import cached_chat_json, model_candidates, retry_delay_seconds, update_audit_from_result
from .planner import (
    _assessment_text,
    _build_model_class_planner_brief,
    _build_model_report_packs,
    _class_entities,
    _deterministic_class_plan,
    _fact_model_ids,
    _namespace_hypothesis_ids,
    _status_from_output,
)
from .supplement_tools import ToolExecutionBudget, execute_tool, normalize_model_ids
from .tool_registry import TOOL_REGISTRY, allowed_tools_for_strategy, validate_tool_request


@dataclass(frozen=True)
class AgentLoopConfig:
    max_model_classes: int = 3
    max_hypotheses_per_class: int = 1
    max_agent_rounds: int = 3
    max_tool_calls_per_hypothesis: int = 3
    max_invalid_requests: int = 2
    max_tool_rows_per_call: int = 50_000
    max_tool_unique_groups: int = 20_000
    max_tool_evidence_records: int = 5
    fallback_policy: str = FallbackPolicy.REVIEW_SAFE.value


def run_agent_loop_demo(
    config: Any,
    case_id: int,
    model_factpacks: list[dict[str, Any]],
    model_class_packs: list[dict[str, Any]],
    case_pack: dict[str, Any],
    bundle: Any,
    dictionary: Any,
    row_source: Any,
    loop_config: AgentLoopConfig | None = None,
    model_claims: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    loop_config = loop_config or AgentLoopConfig()
    fallback_policy = normalize_fallback_policy(loop_config.fallback_policy)
    tool_budget = ToolExecutionBudget(
        max_rows_per_call=loop_config.max_tool_rows_per_call,
        max_unique_groups=loop_config.max_tool_unique_groups,
        max_evidence_records=loop_config.max_tool_evidence_records,
    )
    reasoning_class_packs = [
        pack for pack in model_class_packs if int(pack.get("reasoning_model_count") or 0) > 0
    ][: loop_config.max_model_classes]
    strategies_by_model = bundle.by_model_id
    fact_model_ids = _fact_model_ids(model_factpacks)
    used_hypothesis_ids: set[str] = set()

    planner_audit = {
        "mode": "agent-loop-demo",
        "status": "not_started",
        "reason": "",
        "llm_requested": bool(config.llm.is_complete),
        "planner_context": "single_model_class_pack_loop_context",
        "model_class_count": len(model_class_packs),
        "reasoning_model_class_count": len(reasoning_class_packs),
        "max_model_classes": loop_config.max_model_classes,
        "max_hypotheses_per_class": loop_config.max_hypotheses_per_class,
        "max_agent_rounds": loop_config.max_agent_rounds,
        "max_tool_calls_per_hypothesis": loop_config.max_tool_calls_per_hypothesis,
        "max_invalid_requests": loop_config.max_invalid_requests,
        "fallback_policy": fallback_policy.value,
        "tool_budget": tool_budget.__dict__,
        "tool_truncation_count": 0,
        "rejected_request_count": 0,
        "llm_call_count": 0,
        "llm_valid_request_count": 0,
        "fallback_used": False,
        "class_runs": [],
        "loop_contexts": [],
    }

    hypotheses: list[dict[str, Any]] = []
    planner_decisions: list[dict[str, Any]] = []
    supplement_facts: list[dict[str, Any]] = []
    chain_steps: list[dict[str, Any]] = []
    all_class_briefs: list[dict[str, Any]] = []

    tool_cache: dict[str, dict[str, Any]] = {}

    for class_index, class_pack in enumerate(reasoning_class_packs, start=1):
        class_key = f"class{class_index:02d}"
        class_name = str(class_pack.get("class_name") or class_key)
        class_brief = _build_model_class_planner_brief([class_pack], case_pack)
        class_brief["existing_model_claims"] = _claim_summaries_for_class(model_claims or [], class_pack)
        all_class_briefs.append(class_brief)
        class_allowed_model_ids = _class_model_ids(class_pack)
        class_allowed_entities = _class_entities(class_pack)
        class_entity_fact_ids = _class_entity_fact_ids(
            class_pack,
            fact_model_ids,
            model_factpacks,
        )
        class_allowed_entities.update(class_entity_fact_ids)
        class_hypotheses, hypothesis_audit = _ask_llm_for_initial_hypotheses(
            config,
            class_brief,
            class_allowed_entities,
            class_entity_fact_ids,
            class_allowed_model_ids,
            loop_config.max_hypotheses_per_class,
        )
        planner_audit["llm_call_count"] += hypothesis_audit.get("llm_call_count", 0)
        planner_audit["fallback_used"] = planner_audit["fallback_used"] or bool(
            hypothesis_audit.get("fallback_used")
        )
        if not class_hypotheses:
            class_hypotheses, _ = _deterministic_class_plan(
                [class_pack],
                fact_model_ids,
                loop_config.max_hypotheses_per_class,
            )
        if not class_hypotheses:
            class_hypotheses = _seed_class_hypotheses(
                class_pack,
                model_factpacks,
                loop_config.max_hypotheses_per_class,
            )
        class_hypotheses, _ = _namespace_hypothesis_ids(
            class_hypotheses,
            [],
            class_key,
            used_hypothesis_ids,
        )
        class_audit = {
            "class_name": class_name,
            "hypothesis_count": len(class_hypotheses),
            "reason": "" if class_hypotheses else "no_valid_hypothesis",
            "hypothesis_audit": hypothesis_audit,
            "loop_runs": [],
        }
        class_allowed_tools = _class_allowed_tools(class_pack, strategies_by_model, class_allowed_model_ids)

        for hypothesis in class_hypotheses:
            hypotheses.append(hypothesis)
            loop_result = _run_hypothesis_loop(
                config,
                case_id,
                hypothesis,
                class_pack,
                class_brief,
                model_factpacks,
                strategies_by_model,
                dictionary,
                bundle.thresholds,
                row_source,
                class_allowed_model_ids,
                class_allowed_entities,
                class_allowed_tools,
                loop_config.max_agent_rounds,
                loop_config.max_tool_calls_per_hypothesis,
                loop_config.max_invalid_requests,
                supplement_facts,
                chain_steps,
                planner_decisions,
                planner_audit,
                tool_cache,
                tool_budget,
                fallback_policy,
            )
            class_audit["loop_runs"].append(loop_result)
        planner_audit["class_runs"].append(class_audit)

    evidence_chains = build_evidence_chains(hypotheses, chain_steps)
    planner_audit["status"] = "success" if supplement_facts else "fallback"
    if not supplement_facts:
        planner_audit["reason"] = planner_audit["reason"] or "no supplement facts were generated"

    return {
        "llm_brief_pack": {
            "mode": "agent-loop-demo",
            "case_id": case_pack.get("case_id"),
            "data_source": case_pack.get("data_source"),
            "class_briefs": all_class_briefs,
            "loop_contexts": planner_audit["loop_contexts"],
        },
        "planner_audit": planner_audit,
        "hypotheses": hypotheses,
        "planner_decisions": planner_decisions,
        "supplement_facts": supplement_facts,
        "evidence_chain_steps": chain_steps,
        "evidence_chains": evidence_chains,
        "model_report_packs": _build_model_report_packs(model_factpacks, supplement_facts, evidence_chains),
    }


def _ask_llm_for_initial_hypotheses(
    config: Any,
    class_brief: dict[str, Any],
    allowed_entities: set[str],
    entity_fact_ids: dict[str, list[str]],
    allowed_model_ids: list[int],
    limit: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    audit = {
        "status": "not_started",
        "llm_call_count": 0,
        "fallback_used": False,
        "brief_chars": len(json.dumps(class_brief, ensure_ascii=False, default=str)),
    }
    allowed_fact_ids = {fact_id for fact_ids in entity_fact_ids.values() for fact_id in fact_ids}
    if not config.llm.is_complete or not allowed_fact_ids or not allowed_entities:
        audit["status"] = "fallback"
        audit["fallback_used"] = True
        audit["reason"] = "LLM configuration or class facts/entities are unavailable"
        return [], audit

    prompt = (
        "你是模型类级研判规划器。只根据给定单个 ModelClassPack 选择最多一条值得补证的线索。"
        "主体、fact_id 和 model_id 必须来自 allowed_entities、entity_fact_ids、allowed_model_ids。"
        "不要创造金额、时间、主体或关系。只输出 JSON："
        "{\"hypotheses\":[{\"hypothesis_id\":\"hyp_01\",\"entity_name\":\"...\","
        "\"statement\":\"中文待补证陈述\",\"basis_fact_ids\":[\"...\"],"
        "\"target_model_ids\":[70]}]}。"
    )
    payload = {
        "class_context": class_brief,
        "allowed_entities": sorted(allowed_entities)[:80],
        "entity_fact_ids": entity_fact_ids,
        "allowed_model_ids": allowed_model_ids,
        "max_hypotheses": limit,
    }
    candidates = model_candidates(config.llm.model, "planner")[:2]
    audit["model_candidates"] = candidates
    for attempt, model_name in enumerate(candidates, start=1):
        started = time.perf_counter()
        audit["llm_call_count"] += 1
        try:
            result = cached_chat_json(
                config,
                model_name=model_name,
                role="planner-initial",
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": json.dumps(payload, ensure_ascii=False, default=str)},
                ],
                max_tokens=min(config.llm.max_tokens, 2048),
            )
            update_audit_from_result(audit, result, model_name, attempt)
            audit["last_latency_seconds"] = round(time.perf_counter() - started, 3)
            hypotheses = _parse_initial_hypotheses(
                result.payload,
                allowed_entities,
                entity_fact_ids,
                allowed_model_ids,
                limit,
            )
            if hypotheses:
                audit["status"] = "success"
                audit["used_model"] = model_name
                return hypotheses, audit
            audit["last_error"] = "initial hypothesis schema or references were invalid"
        except Exception as exc:
            audit["last_error"] = f"{type(exc).__name__}: {str(exc)[:200]}"
            audit["last_latency_seconds"] = round(time.perf_counter() - started, 3)
    audit["status"] = "fallback"
    audit["fallback_used"] = True
    return [], audit


def _parse_initial_hypotheses(
    payload: dict[str, Any] | None,
    allowed_entities: set[str],
    entity_fact_ids: dict[str, list[str]],
    allowed_model_ids: list[int],
    limit: int,
) -> list[dict[str, Any]]:
    if not isinstance(payload, dict):
        return []
    raw_items = payload.get("hypotheses") or payload.get("hypothesis_candidates") or []
    result = []
    for index, item in enumerate(raw_items[:limit], start=1):
        if not isinstance(item, dict):
            continue
        entity_name = str(item.get("entity_name") or "")
        statement = str(item.get("statement") or "").strip()
        entity_allowed_fact_ids = set(entity_fact_ids.get(entity_name, []))
        fact_ids = [
            str(value)
            for value in item.get("basis_fact_ids", [])
            if str(value) in entity_allowed_fact_ids
        ]
        model_ids = [
            int(value)
            for value in item.get("target_model_ids", [])
            if str(value).isdigit() and int(value) in allowed_model_ids
        ]
        for fact_id in fact_ids:
            match = re.match(r"m(\d+)_", fact_id)
            if match and int(match.group(1)) in allowed_model_ids:
                model_ids.append(int(match.group(1)))
        if entity_name not in allowed_entities or not statement or not fact_ids or not model_ids:
            continue
        result.append(
            {
                "hypothesis_id": str(item.get("hypothesis_id") or f"hyp_{index:02d}"),
                "entity_name": entity_name,
                "statement": statement,
                "basis_fact_ids": fact_ids,
                "target_model_ids": list(dict.fromkeys(model_ids)),
                "status": "candidate",
                "source": "llm_model_class_planner",
            }
        )
    return result


def _class_entity_fact_ids(
    class_pack: dict[str, Any],
    fact_model_ids: dict[str, int],
    model_factpacks: list[dict[str, Any]],
) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    for key in ("shared_focus_entities", "shared_relation_entities"):
        for entity in class_pack.get(key, []):
            entity_name = str(entity.get("entity_name") or "")
            if not entity_name:
                continue
            for appearance in entity.get("appearances", []):
                fact_id = str(appearance.get("fact_id") or "")
                if fact_id in fact_model_ids:
                    result.setdefault(entity_name, []).append(fact_id)
    for entity in class_pack.get("class_specific_entities", []):
        entity_name = str(entity.get("entity_name") or "")
        fact_id = str(entity.get("fact_id") or "")
        if entity_name and fact_id in fact_model_ids:
            result.setdefault(entity_name, []).append(fact_id)
    class_name = str(class_pack.get("class_name") or "")
    for factpack in model_factpacks:
        context = factpack.get("model_context", {})
        if str(context.get("root_category") or "") != class_name:
            continue
        for entity in factpack.get("focus_entities", []):
            entity_name = str(entity.get("entity_name") or "")
            fact_id = str(entity.get("fact_id") or "")
            if entity_name and fact_id in fact_model_ids:
                result.setdefault(entity_name, []).append(fact_id)
        for relation in factpack.get("relation_profiles", []):
            fact_id = str(relation.get("fact_id") or "")
            if fact_id not in fact_model_ids:
                continue
            for key in ("source", "target"):
                entity_name = str(relation.get(key) or "")
                if entity_name:
                    result.setdefault(entity_name, []).append(fact_id)
    return {name: list(dict.fromkeys(values)) for name, values in result.items()}


def _run_hypothesis_loop(
    config: Any,
    case_id: int,
    hypothesis: dict[str, Any],
    class_pack: dict[str, Any],
    class_brief: dict[str, Any],
    model_factpacks: list[dict[str, Any]],
    strategies_by_model: dict[int, dict[str, Any]],
    dictionary: Any,
    thresholds: dict[str, Any],
    row_source: Any,
    allowed_model_ids: list[int],
    allowed_entities: set[str],
    allowed_tools: list[str],
    max_agent_rounds: int,
    max_tool_calls: int,
    max_invalid_requests: int,
    supplement_facts: list[dict[str, Any]],
    chain_steps: list[dict[str, Any]],
    planner_decisions: list[dict[str, Any]],
    planner_audit: dict[str, Any],
    tool_cache: dict[str, dict[str, Any]],
    tool_budget: ToolExecutionBudget,
    fallback_policy: FallbackPolicy,
) -> dict[str, Any]:
    hyp_id = str(hypothesis.get("hypothesis_id"))
    previous_rounds: list[dict[str, Any]] = []
    called_keys: set[str] = set()
    invalid_count = 0
    tool_call_count = 0
    final_status = "unresolved"
    stop_reason = ""

    agent_round = 1
    while agent_round <= max_agent_rounds:
        if tool_call_count >= max_tool_calls:
            stop_reason = "budget_exhausted"
            break
        validation_feedback: list[str] = []
        request: dict[str, Any] | None = None
        round_invalid_count = 0
        while round_invalid_count < max_invalid_requests:
            loop_context = _build_loop_context(
                class_brief,
                hypothesis,
                allowed_entities,
                allowed_model_ids,
                allowed_tools,
                previous_rounds,
                called_keys,
                max_agent_rounds - agent_round + 1,
                max_tool_calls - tool_call_count,
            )
            planner_audit.setdefault("loop_context_chars", []).append(
                len(json.dumps(loop_context, ensure_ascii=False, default=str))
            )
            if validation_feedback:
                loop_context["validation_feedback"] = validation_feedback[-2:]
            planner_audit["loop_contexts"].append(loop_context)
            request, request_audit = _ask_llm_for_next_action(config, loop_context, agent_round)
            planner_audit["llm_call_count"] += request_audit.get("llm_call_count", 0)
            planner_audit["fallback_used"] = planner_audit["fallback_used"] or bool(
                request_audit.get("fallback_used")
            )
            planner_audit.setdefault("planner_calls", []).append(request_audit)

            if request is None:
                request = _fallback_next_action(hypothesis, model_factpacks, agent_round, previous_rounds)
                planner_audit["fallback_used"] = True
            if request is None:
                stop_reason = "no_available_action"
                break
            if request.get("action") == "stop":
                stop_reason = str(request.get("stop_reason") or "llm_stop")
                chain_steps.append(
                    _make_stop_step(
                        hyp_id,
                        agent_round * 2 - 1,
                        request,
                        stop_reason,
                        str(request.get("reason") or stop_reason),
                        final_status,
                    )
                )
                break

            request["hypothesis_id"] = hyp_id
            ok, reason = _validate_agent_request(
                request,
                strategies_by_model,
                allowed_entities,
                allowed_model_ids,
                allowed_tools,
                called_keys,
            )
            if ok:
                break
            invalid_count += 1
            round_invalid_count += 1
            validation_feedback.append(f"工具请求被拒绝：{reason}")
            planner_audit["rejected_request_count"] = int(planner_audit.get("rejected_request_count") or 0) + 1
            chain_steps.append(
                _make_stop_step(
                    hyp_id,
                    agent_round * 2 - 1,
                    request,
                    "invalid_request",
                    validation_feedback[-1],
                    final_status,
                )
            )
            request = None

        if stop_reason or request is None:
            if not stop_reason:
                stop_reason = "invalid_request_limit"
            break

        planner_audit["llm_valid_request_count"] += 1
        call_key = _call_key(request)
        called_keys.add(call_key)
        decision = make_planner_decision(
            hyp_id,
            agent_round * 2 - 1,
            str(request["tool_name"]),
            dict(request.get("params") or {}),
            str(request.get("reason") or request.get("expected_evidence") or ""),
            str(request.get("source") or "llm_agent_planner"),
        )
        decision["agent_round"] = agent_round
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
                thresholds,
                row_source=row_source,
                tool_cache=tool_cache,
                budget=tool_budget,
            )
        except Exception as exc:
            output = {
                "tool_name": decision["tool_name"],
                "summary": f"工具执行失败：{type(exc).__name__}: {str(exc)[:120]}",
                "row_count": 0,
                "status_signal": "unresolved",
                "fact_refs": [],
                "evidence_refs": [],
            }

        supplement = make_supplement_fact(len(supplement_facts) + 1, hyp_id, decision, output)
        supplement_facts.append(supplement)
        if output.get("truncated"):
            planner_audit["tool_truncation_count"] = int(planner_audit.get("tool_truncation_count") or 0) + 1
        tool_status = _status_from_output(output)
        chain_steps.append(
            make_chain_step(
                hyp_id,
                agent_round * 2 - 1,
                decision,
                supplement,
                tool_status,
                _assessment_text(output),
            )
        )
        tool_call_count += 1

        judge_steps, judge_audit = build_judge_steps(
            config,
            [hypothesis],
            [fact for fact in supplement_facts if fact.get("hypothesis_id") == hyp_id],
            [step for step in chain_steps if step.get("hypothesis_id") == hyp_id],
            agent_round * 2,
            fallback_policy=fallback_policy,
        )
        for step in judge_steps:
            step["agent_round"] = agent_round
        chain_steps.extend(judge_steps)
        planner_audit.setdefault("judge_audits", []).append(judge_audit)

        judge_step = judge_steps[-1] if judge_steps else {}
        judge_status = str(judge_step.get("status") or "unresolved")
        judge_assessment = str(judge_step.get("assessment") or "")
        summary = _compact_tool_summary(output)
        previous_rounds.append(
            {
                "round": agent_round,
                "tool_called": decision["tool_name"],
                "params": decision["params"],
                "tool_summary": summary,
                "judge_status": judge_status,
                "judge_assessment": _clip(judge_assessment, 220),
            }
        )
        final_status = judge_status

        if judge_status in {"enhanced", "weakened"}:
            stop_reason = f"judge_{judge_status}"
            break
        if agent_round >= max_agent_rounds:
            stop_reason = "budget_exhausted"
            break
        if _no_progress(previous_rounds):
            stop_reason = "no_progress"
            break
        agent_round += 1

    return {
        "hypothesis_id": hyp_id,
        "final_status": final_status,
        "stop_reason": stop_reason or "loop_completed",
        "tool_call_count": tool_call_count,
        "invalid_request_count": invalid_count,
        "previous_rounds": previous_rounds,
    }


def _ask_llm_for_next_action(
    config: Any,
    loop_context: dict[str, Any],
    agent_round: int,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    audit = {
        "status": "not_started",
        "round": agent_round,
        "llm_call_count": 0,
        "fallback_used": False,
    }
    if not config.llm.is_complete:
        audit["status"] = "llm_not_configured"
        audit["fallback_used"] = True
        return None, audit

    candidates = model_candidates(config.llm.model, "planner")[:2]
    audit["model_candidates"] = candidates
    attempt = 0
    last_error = ""
    for model_name in candidates:
        for _ in range(1):
            attempt += 1
            audit["llm_call_count"] += 1
            payload = _try_agent_plan_once(config, loop_context, audit, model_name, attempt)
            if payload is not None:
                action = _parse_agent_action(payload)
                if action is not None:
                    action["source"] = "llm_agent_planner"
                    audit["status"] = "success"
                    audit["used_model"] = model_name
                    return action, audit
                last_error = "payload did not match agent action schema"
            else:
                last_error = str(audit.get("last_llm_error") or "")
            if attempt < 2:
                time.sleep(min(int(audit.get("last_retry_delay_seconds") or 1), 2))
    audit["status"] = "fallback"
    audit["fallback_used"] = True
    audit["reason"] = last_error
    return None, audit


def _try_agent_plan_once(
    config: Any,
    loop_context: dict[str, Any],
    audit: dict[str, Any],
    model_name: str,
    attempt: int,
) -> dict[str, Any] | None:
    started = time.perf_counter()
    try:
        prompt = (
            "你是案件模型结果的受控 Agent Planner。只能根据 LoopContext 决定下一步工具调用或停止。"
            "不要生成 SQL，不要创造 LoopContext 之外的主体、金额、模型或关系。"
            "只输出 JSON，格式为："
            "{\"action\":\"call_tool|stop\",\"hypothesis_id\":\"...\",\"tool_name\":\"...\","
            "\"params\":{},\"reason\":\"...\",\"expected_evidence\":\"...\",\"stop_reason\":\"...\"}。"
            "如果上一轮 Judge 已经增强或削弱，应停止。"
            "如果仍 unresolved 且预算允许，选择一个未重复且最有信息增量的工具。"
        )
        result = cached_chat_json(
            config,
            model_name=model_name,
            role="planner",
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": json.dumps(loop_context, ensure_ascii=False, default=str)},
            ],
            max_tokens=min(config.llm.max_tokens, 2048),
        )
        update_audit_from_result(audit, result, model_name, attempt)
        audit["last_latency_seconds"] = round(time.perf_counter() - started, 3)
        return result.payload
    except Exception as exc:
        audit["last_attempt_model"] = model_name
        audit["last_llm_error"] = f"{type(exc).__name__}: {str(exc)[:200]}"
        audit["llm_attempts"] = attempt
        audit["last_retry_delay_seconds"] = retry_delay_seconds(exc, attempt)
        audit["last_latency_seconds"] = round(time.perf_counter() - started, 3)
        return None


def _parse_agent_action(payload: dict[str, Any]) -> dict[str, Any] | None:
    if not isinstance(payload, dict):
        return None
    if isinstance(payload.get("next_action"), dict):
        payload = payload["next_action"]
    action = str(payload.get("action") or "").strip()
    if action not in {"call_tool", "stop"}:
        return None
    if action == "stop":
        return {
            "action": "stop",
            "hypothesis_id": payload.get("hypothesis_id"),
            "tool_name": "stop",
            "params": {},
            "reason": str(payload.get("reason") or ""),
            "stop_reason": str(payload.get("stop_reason") or "llm_stop"),
        }
    return {
        "action": "call_tool",
        "hypothesis_id": payload.get("hypothesis_id"),
        "tool_name": payload.get("tool_name"),
        "params": payload.get("params") if isinstance(payload.get("params"), dict) else {},
        "reason": str(payload.get("reason") or ""),
        "expected_evidence": str(payload.get("expected_evidence") or ""),
    }


def _validate_agent_request(
    request: dict[str, Any],
    strategies_by_model: dict[int, dict[str, Any]],
    allowed_entities: set[str],
    allowed_model_ids: list[int],
    allowed_tools: list[str],
    called_keys: set[str],
) -> tuple[bool, str]:
    if request.get("action") == "stop":
        return False, str(request.get("stop_reason") or "llm_stop")
    tool_name = str(request.get("tool_name") or "")
    if tool_name not in allowed_tools:
        return False, f"tool {tool_name} is not allowed in this model class"
    params = request.get("params") or {}
    if params.get("model_id") is not None:
        try:
            model_id = int(params["model_id"])
        except Exception:
            return False, "model_id must be an integer"
        if model_id not in allowed_model_ids:
            return False, f"model_id {model_id} is outside current model class"
    requested_model_ids = normalize_model_ids(params.get("model_ids"))
    outside_model_ids = [model_id for model_id in requested_model_ids if model_id not in allowed_model_ids]
    if outside_model_ids:
        return False, f"model_ids {outside_model_ids} are outside current model class"
    ok, reason = validate_tool_request(request, strategies_by_model, allowed_entities)
    if not ok:
        return False, reason
    if _call_key(request) in called_keys:
        return False, "duplicate tool call"
    return True, ""


def _fallback_next_action(
    hypothesis: dict[str, Any],
    model_factpacks: list[dict[str, Any]],
    agent_round: int,
    previous_rounds: list[dict[str, Any]],
) -> dict[str, Any] | None:
    requests: list[dict[str, Any]] = []
    entity_name = hypothesis.get("entity_name")
    target_model_ids = [int(mid) for mid in hypothesis.get("target_model_ids", []) if str(mid).isdigit()]
    if entity_name:
        for model_id in target_model_ids[:3]:
            requests.append(
                {
                    "hypothesis_id": hypothesis.get("hypothesis_id"),
                    "round": agent_round,
                    "tool_name": "entity_model_slice",
                    "params": {"model_id": model_id, "entity_name": entity_name},
                    "reason": "LLM 规划失败后，使用确定性回退核查该主体在目标模型中的切片。",
                    "source": "deterministic_agent_fallback",
                    "action": "call_tool",
                }
            )
        if target_model_ids:
            requests.append(
                {
                    "hypothesis_id": hypothesis.get("hypothesis_id"),
                    "round": agent_round,
                    "tool_name": "counterparty_overlap_probe",
                    "params": {"entity_name": entity_name, "model_ids": target_model_ids},
                    "reason": "LLM 规划失败后，使用确定性回退比较该主体跨模型对手方重合情况。",
                    "source": "deterministic_agent_fallback",
                    "action": "call_tool",
                }
            )
    used = {(item.get("tool_called"), json.dumps(item.get("params", {}), sort_keys=True, ensure_ascii=False)) for item in previous_rounds}
    for request in requests:
        key = (request.get("tool_name"), json.dumps(request.get("params", {}), sort_keys=True, ensure_ascii=False))
        if key not in used:
            request["round"] = agent_round
            request["action"] = "call_tool"
            request["source"] = "deterministic_agent_fallback"
            return request
    return None


def _build_loop_context(
    class_brief: dict[str, Any],
    hypothesis: dict[str, Any],
    allowed_entities: set[str],
    allowed_model_ids: list[int],
    allowed_tools: list[str],
    previous_rounds: list[dict[str, Any]],
    called_keys: set[str] | None,
    rounds_remaining: int,
    calls_remaining: int,
) -> dict[str, Any]:
    return {
        "class_context": class_brief,
        "hypothesis": {
            "hypothesis_id": hypothesis.get("hypothesis_id"),
            "entity_name": hypothesis.get("entity_name"),
            "statement": hypothesis.get("statement"),
            "basis_fact_ids": hypothesis.get("basis_fact_ids", []),
            "target_model_ids": hypothesis.get("target_model_ids", []),
        },
        "known_entities": sorted(allowed_entities)[:80],
        "allowed_model_ids": allowed_model_ids,
        "allowed_tools": allowed_tools,
        "tool_schemas": {
            name: {
                "required_params": list(TOOL_REGISTRY[name].required_params),
                "optional_params": [
                    value for value in TOOL_REGISTRY[name].optional_params if value != "time_range"
                ],
            }
            for name in allowed_tools
            if name in TOOL_REGISTRY
        },
        "previous_rounds": previous_rounds[-3:],
        "already_called_fingerprints": sorted(called_keys or set())[-8:],
        "budget_remaining": {
            "rounds": rounds_remaining,
            "tool_calls": calls_remaining,
        },
    }


def _claim_summaries_for_class(
    model_claims: list[dict[str, Any]],
    class_pack: dict[str, Any],
) -> list[dict[str, Any]]:
    model_ids = {int(model.get("model_id")) for model in class_pack.get("models", []) if model.get("model_id") is not None}
    result = []
    for claim in model_claims:
        try:
            claim_model_id = int(claim.get("model_id"))
        except (TypeError, ValueError):
            continue
        if claim_model_id not in model_ids:
            continue
        result.append(
            {
                "claim_id": claim.get("claim_id"),
                "model_id": claim_model_id,
                "entity_name": claim.get("entity_name"),
                "status": claim.get("status"),
                "severity": claim.get("severity"),
                "statement": _clip(str(claim.get("statement") or ""), 220),
                "basis_fact_ids": claim.get("basis_fact_ids", [])[:5],
                "supplement_fact_ids": claim.get("supplement_fact_ids", [])[:5],
            }
        )
    return result[:5]


def _class_model_ids(class_pack: dict[str, Any]) -> list[int]:
    ids = []
    for model in class_pack.get("models", []):
        model_id = model.get("model_id")
        if model_id is not None and int(model.get("result_count") or 0) > 0:
            ids.append(int(model_id))
    if not ids:
        ids = [int(mid) for mid in class_pack.get("reasoning_model_ids", []) if str(mid).isdigit()]
    return sorted(dict.fromkeys(ids))


def _class_allowed_tools(
    class_pack: dict[str, Any],
    strategies_by_model: dict[int, dict[str, Any]],
    allowed_model_ids: list[int],
) -> list[str]:
    tools = set(class_pack.get("available_tools") or [])
    for model_id in allowed_model_ids:
        tools.update(allowed_tools_for_strategy(strategies_by_model.get(model_id, {})))
    return sorted(tool for tool in tools if tool in TOOL_REGISTRY)


def _seed_class_hypotheses(
    class_pack: dict[str, Any],
    model_factpacks: list[dict[str, Any]],
    limit: int,
) -> list[dict[str, Any]]:
    class_name = str(class_pack.get("class_name") or "")
    result: list[dict[str, Any]] = []
    for entity in class_pack.get("class_specific_entities", []):
        fact_id = entity.get("fact_id")
        model_id = entity.get("model_id")
        entity_name = entity.get("entity_name")
        if not fact_id or not model_id or not entity_name:
            continue
        result.append(
            {
                "hypothesis_id": f"seed_hyp_{len(result) + 1:02d}",
                "entity_name": entity_name,
                "statement": (
                    f"{entity_name} 在模型 {model_id} 中形成重点主体，需要通过工具补证其模型内记录、"
                    "对手方或关系特征是否能支撑异常线索。"
                ),
                "basis_fact_ids": [fact_id],
                "target_model_ids": [int(model_id)],
                "status": "candidate",
                "source": "agent_seed_from_class_specific_entity",
            }
        )
        if len(result) >= limit:
            return result

    for fp in model_factpacks:
        context = fp.get("model_context", {})
        if str(context.get("root_category") or "") != class_name:
            continue
        model_id = int(context.get("model_id"))
        for entity in fp.get("focus_entities", []):
            fact_id = entity.get("fact_id")
            entity_name = entity.get("entity_name")
            if not fact_id or not entity_name:
                continue
            result.append(
                {
                    "hypothesis_id": f"seed_hyp_{len(result) + 1:02d}",
                    "entity_name": entity_name,
                    "statement": (
                        f"{entity_name} 是模型 {model_id} 的重点主体，需要补证该主体在模型内是否存在"
                        "高频、高金额或对手方集中等可报告线索。"
                    ),
                    "basis_fact_ids": [fact_id],
                    "target_model_ids": [model_id],
                    "status": "candidate",
                    "source": "agent_seed_from_model_factpack",
                }
            )
            if len(result) >= limit:
                return result
        for relation in fp.get("relation_profiles", []):
            fact_id = relation.get("fact_id")
            source = relation.get("source")
            if not fact_id or not source:
                continue
            result.append(
                {
                    "hypothesis_id": f"seed_hyp_{len(result) + 1:02d}",
                    "entity_name": source,
                    "statement": (
                        f"{source} 在模型 {model_id} 中存在靠前关系边，需要补证其邻域或路径是否形成"
                        "可解释的异常关系。"
                    ),
                    "basis_fact_ids": [fact_id],
                    "target_model_ids": [model_id],
                    "status": "candidate",
                    "source": "agent_seed_from_relation_profile",
                }
            )
            if len(result) >= limit:
                return result
    return result


def _call_key(request: dict[str, Any]) -> str:
    params = request.get("params") or {}
    if "model_ids" in params:
        params = {**params, "model_ids": normalize_model_ids(params.get("model_ids"))}
    return json.dumps(
        {
            "tool_name": request.get("tool_name"),
            "params": params,
        },
        ensure_ascii=False,
        sort_keys=True,
        default=str,
    )


def _compact_tool_summary(output: dict[str, Any]) -> str:
    summary = str(output.get("summary") or "")
    parts = [summary]
    if output.get("row_count") is not None:
        parts.append(f"命中 {output.get('row_count')} 条")
    if output.get("status_signal"):
        parts.append(f"工具信号 {output.get('status_signal')}")
    return _clip("；".join(part for part in parts if part), 200)


def _no_progress(previous_rounds: list[dict[str, Any]]) -> bool:
    if len(previous_rounds) < 2:
        return False
    last = previous_rounds[-1]
    prev = previous_rounds[-2]
    return (
        last.get("judge_status") == prev.get("judge_status") == "unresolved"
        and last.get("tool_summary") == prev.get("tool_summary")
    )


def _make_stop_step(
    hypothesis_id: str,
    round_index: int,
    request: dict[str, Any],
    stop_reason: str,
    assessment: str,
    status: str = "unresolved",
) -> dict[str, Any]:
    decision = make_planner_decision(
        hypothesis_id,
        round_index,
        str(request.get("tool_name") or "stop"),
        dict(request.get("params") or {}),
        assessment,
        str(request.get("source") or "agent_guardrail"),
    )
    decision["stop_reason"] = stop_reason
    return make_chain_step(hypothesis_id, round_index, decision, None, status, assessment)


def _clip(text: str, limit: int) -> str:
    text = text.strip()
    return text if len(text) <= limit else text[: limit - 1] + "…"

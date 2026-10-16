from __future__ import annotations

from dataclasses import asdict, dataclass
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
from .execution_policy import FallbackPolicy, normalize_fallback_policy
from .judge import build_judge_steps
from .llm_utils import cached_chat_json, model_candidates, update_audit_from_result
from .supplement_tools import ToolExecutionBudget, execute_tool
from .tool_registry import TOOL_REGISTRY, allowed_tools_for_strategy, validate_tool_request


@dataclass(frozen=True)
class ModelInvestigationConfig:
    model_id: int = 86
    max_rounds: int = 4
    max_tool_calls: int = 6
    max_invalid_requests: int = 2
    max_tool_rows_per_call: int = 50_000
    max_tool_unique_groups: int = 20_000
    max_tool_evidence_records: int = 5
    fallback_policy: str = FallbackPolicy.REVIEW_SAFE.value

    @property
    def tool_budget(self) -> ToolExecutionBudget:
        return ToolExecutionBudget(
            max_rows_per_call=self.max_tool_rows_per_call,
            max_unique_groups=self.max_tool_unique_groups,
            max_evidence_records=self.max_tool_evidence_records,
        )


@dataclass(frozen=True)
class ModelClaim:
    claim_id: str
    model_id: int
    entity_name: str | None
    hypothesis_id: str
    status: str
    statement: str
    basis_fact_ids: list[str]
    supplement_fact_ids: list[str]
    evidence_chain_id: str
    limitation: str
    severity: str = "一般关注"
    version: int = 1
    superseded_by: str | None = None


@dataclass(frozen=True)
class InvestigationRound:
    round_index: int
    tool_name: str
    params: dict[str, Any]
    tool_output_summary: str
    tool_status_signal: str
    judge_status: str
    judge_assessment: str
    rows_read: int = 0
    truncated: bool = False


@dataclass(frozen=True)
class ModelInvestigationResult:
    model_id: int
    hypotheses: list[dict[str, Any]]
    supplement_facts: list[dict[str, Any]]
    chain_steps: list[dict[str, Any]]
    evidence_chains: list[dict[str, Any]]
    claims: list[ModelClaim]
    audit: dict[str, Any]
    tool_cache: dict[str, dict[str, Any]]

    @property
    def enhanced_claims(self) -> list[ModelClaim]:
        return [claim for claim in self.claims if claim.status == "enhanced"]

    @property
    def unresolved_claims(self) -> list[ModelClaim]:
        return [claim for claim in self.claims if claim.status == "unresolved"]


def run_model_investigation(
    config: Any,
    case_id: int,
    model_id: int,
    model_baseline: dict[str, Any],
    model_strategy: dict[str, Any],
    bundle: Any,
    dictionary: Any,
    row_source: Any,
    investigation_config: ModelInvestigationConfig | None = None,
    external_tool_cache: dict[str, dict[str, Any]] | None = None,
) -> ModelInvestigationResult:
    inv_config = investigation_config or ModelInvestigationConfig(model_id=model_id)
    policy = normalize_fallback_policy(inv_config.fallback_policy)
    tool_cache = external_tool_cache if external_tool_cache is not None else {}
    allowed_tools = _allowed_tools(model_strategy)
    allowed_entities = _known_entities(model_baseline)
    hypotheses = _seed_hypotheses_from_baseline(model_id, model_baseline, max_hypotheses=1)
    audit: dict[str, Any] = {
        "mode": "model-investigation",
        "model_id": model_id,
        "status": "not_started",
        "fallback_policy": policy.value,
        "llm_requested": bool(config.llm.is_complete),
        "allowed_tools": allowed_tools,
        "hypothesis_count": len(hypotheses),
        "llm_call_count": 0,
        "llm_valid_request_count": 0,
        "fallback_used": False,
        "rejected_request_count": 0,
        "tool_truncation_count": 0,
        "context_chars": [],
        "loop_runs": [],
        "judge_audits": [],
        "tool_budget": inv_config.tool_budget.__dict__,
    }
    supplement_facts: list[dict[str, Any]] = []
    chain_steps: list[dict[str, Any]] = []
    claims: list[ModelClaim] = []

    for hypothesis in hypotheses:
        loop_result = _run_model_hypothesis_loop(
            config,
            case_id,
            model_id,
            model_baseline,
            model_strategy,
            bundle,
            dictionary,
            row_source,
            hypothesis,
            allowed_tools,
            allowed_entities,
            inv_config,
            policy,
            tool_cache,
            supplement_facts,
            chain_steps,
            claims,
            audit,
        )
        audit["loop_runs"].append(loop_result)

    evidence_chains = build_evidence_chains(hypotheses, chain_steps)
    _bind_claim_chain_ids(claims, evidence_chains)
    audit["status"] = "success" if claims else "fallback"
    if not claims:
        audit["reason"] = "no model claim generated"
    return ModelInvestigationResult(
        model_id=model_id,
        hypotheses=hypotheses,
        supplement_facts=supplement_facts,
        chain_steps=chain_steps,
        evidence_chains=evidence_chains,
        claims=claims,
        audit=audit,
        tool_cache=tool_cache,
    )


def result_to_planner_result(
    model_results: list[ModelInvestigationResult],
    class_result: dict[str, Any] | None = None,
) -> dict[str, Any]:
    class_result = class_result or {}
    hypotheses = []
    supplement_facts = []
    chain_steps = []
    evidence_chains = []
    model_claims = []
    for result in model_results:
        hypotheses.extend(result.hypotheses)
        supplement_facts.extend(result.supplement_facts)
        chain_steps.extend(result.chain_steps)
        evidence_chains.extend(result.evidence_chains)
        model_claims.extend(asdict(claim) for claim in result.claims)
    hypotheses.extend(class_result.get("hypotheses", []))
    supplement_facts.extend(class_result.get("supplement_facts", []))
    chain_steps.extend(class_result.get("evidence_chain_steps", []))
    evidence_chains.extend(class_result.get("evidence_chains", []))
    planner_audit = {
        "mode": "full-agent",
        "status": "success" if supplement_facts else "fallback",
        "model_investigations": [result.audit for result in model_results],
        "class_agent": class_result.get("planner_audit", {}),
        "class_runs": class_result.get("planner_audit", {}).get("class_runs", []),
        "judge_audits": [
            audit
            for result in model_results
            for audit in result.audit.get("judge_audits", [])
        ]
        + list(class_result.get("planner_audit", {}).get("judge_audits", [])),
        "model_claim_count": len(model_claims),
        "fallback_used": any(result.audit.get("fallback_used") for result in model_results)
        or bool(class_result.get("planner_audit", {}).get("fallback_used")),
        "tool_budget": model_results[0].audit.get("tool_budget", {}) if model_results else {},
        "tool_truncation_count": sum(int(result.audit.get("tool_truncation_count") or 0) for result in model_results)
        + int(class_result.get("planner_audit", {}).get("tool_truncation_count") or 0),
        "rejected_request_count": sum(int(result.audit.get("rejected_request_count") or 0) for result in model_results)
        + int(class_result.get("planner_audit", {}).get("rejected_request_count") or 0),
    }
    return {
        "llm_brief_pack": {
            "mode": "full-agent",
            "model_investigations": [result.audit for result in model_results],
            "class_brief_pack": class_result.get("llm_brief_pack", {}),
        },
        "planner_audit": planner_audit,
        "hypotheses": hypotheses,
        "planner_decisions": class_result.get("planner_decisions", []),
        "supplement_facts": supplement_facts,
        "evidence_chain_steps": chain_steps,
        "evidence_chains": evidence_chains,
        "model_report_packs": class_result.get("model_report_packs", []),
        "model_claims": model_claims,
    }


def _run_model_hypothesis_loop(
    config: Any,
    case_id: int,
    model_id: int,
    model_baseline: dict[str, Any],
    model_strategy: dict[str, Any],
    bundle: Any,
    dictionary: Any,
    row_source: Any,
    hypothesis: dict[str, Any],
    allowed_tools: list[str],
    allowed_entities: set[str],
    inv_config: ModelInvestigationConfig,
    policy: FallbackPolicy,
    tool_cache: dict[str, dict[str, Any]],
    supplement_facts: list[dict[str, Any]],
    chain_steps: list[dict[str, Any]],
    claims: list[ModelClaim],
    audit: dict[str, Any],
) -> dict[str, Any]:
    previous_rounds: list[InvestigationRound] = []
    called_keys: set[str] = set()
    invalid_count = 0
    stop_reason = "budget_exhausted"
    final_status = "unresolved"
    final_assessment = "达到调查预算上限，暂未形成明确增强或削弱。"
    hyp_supplement_start = len(supplement_facts)
    hyp_id = str(hypothesis.get("hypothesis_id"))

    for round_index in range(1, inv_config.max_rounds + 1):
        if len(previous_rounds) >= inv_config.max_tool_calls:
            stop_reason = "max_tool_calls"
            break
        context = _build_model_planner_context(
            model_id,
            model_baseline,
            hypothesis,
            previous_rounds,
            allowed_tools,
            sorted(called_keys),
            inv_config.max_rounds - round_index + 1,
            inv_config.max_tool_calls - len(previous_rounds),
        )
        audit["context_chars"].append(len(json.dumps(context, ensure_ascii=False, default=str)))
        decision, decision_audit = _ask_model_planner(config, context, policy)
        audit["llm_call_count"] += decision_audit.get("llm_call_count", 0)
        audit["fallback_used"] = audit["fallback_used"] or bool(decision_audit.get("fallback_used"))
        if not decision:
            decision = _deterministic_model_request(model_id, hypothesis, allowed_tools, called_keys)
            audit["fallback_used"] = True
        if not decision or decision.get("action") == "stop":
            stop_reason = "planner_stop"
            break
        decision.setdefault("round", round_index)
        ok, reason = _validate_model_request(decision, model_id, model_strategy, allowed_entities, called_keys)
        if not ok:
            invalid_count += 1
            audit["rejected_request_count"] += 1
            if invalid_count >= inv_config.max_invalid_requests:
                stop_reason = "too_many_invalid_requests"
                break
            continue
        audit["llm_valid_request_count"] += 1
        tool_name = str(decision["tool_name"])
        params = dict(decision.get("params") or {})
        call_key = _call_key(tool_name, params)
        called_keys.add(call_key)
        planner_decision = make_planner_decision(
            hyp_id,
            round_index,
            tool_name,
            params,
            str(decision.get("reason") or "模型级 Planner 选择工具进行补证。"),
            str(decision.get("source") or "model_investigation_planner"),
        )
        output = execute_tool(
            tool_name,
            config,
            case_id,
            {model_id: model_strategy},
            dictionary,
            [model_baseline],
            params,
            bundle.thresholds,
            row_source=row_source,
            tool_cache=tool_cache,
            budget=inv_config.tool_budget,
        )
        if output.get("truncated"):
            audit["tool_truncation_count"] += 1
        supplement = make_supplement_fact(
            len(supplement_facts) + 1,
            hyp_id,
            planner_decision,
            output,
        )
        supplement_facts.append(supplement)
        judge_steps, judge_audit = build_judge_steps(
            config,
            [hypothesis],
            supplement_facts[hyp_supplement_start:],
            chain_steps,
            round_index,
            fallback_policy=policy,
        )
        audit["judge_audits"].append(judge_audit)
        judge_step = judge_steps[-1] if judge_steps else make_chain_step(
            hyp_id, round_index, planner_decision, supplement, "unresolved", "Judge 未返回结果。"
        )
        if judge_step.get("supplement_fact_id") is None:
            judge_step["supplement_fact_id"] = supplement.get("supplement_fact_id")
        chain_steps.append(judge_step)
        final_status = str(judge_step.get("status") or "unresolved")
        final_assessment = str(judge_step.get("assessment") or "")
        previous_rounds.append(
            InvestigationRound(
                round_index=round_index,
                tool_name=tool_name,
                params=params,
                tool_output_summary=str(output.get("summary") or ""),
                tool_status_signal=str(output.get("status_signal") or ""),
                judge_status=final_status,
                judge_assessment=final_assessment,
                rows_read=int(output.get("rows_read") or output.get("row_count") or 0),
                truncated=bool(output.get("truncated")),
            )
        )
        if final_status in {"enhanced", "weakened"}:
            stop_reason = f"judge_{final_status}"
            break
        if output.get("truncated"):
            stop_reason = "tool_truncated"
            break
        if _no_progress(previous_rounds):
            stop_reason = "no_progress"
            break

    if not chain_steps or str(chain_steps[-1].get("hypothesis_id")) != hyp_id:
        decision = make_planner_decision(hyp_id, len(previous_rounds) + 1, "stop", {}, stop_reason, "model_investigation")
        chain_steps.append(make_chain_step(hyp_id, len(previous_rounds) + 1, decision, None, final_status, final_assessment))
    hypothesis["status"] = final_status
    hyp_supplements = [
        fact for fact in supplement_facts[hyp_supplement_start:] if str(fact.get("hypothesis_id")) == hyp_id
    ]
    claims.append(
        _synthesize_claim(
            model_id,
            hypothesis,
            final_status,
            final_assessment,
            hyp_supplements,
        )
    )
    return {
        "hypothesis_id": hyp_id,
        "round_count": len(previous_rounds),
        "final_status": final_status,
        "stop_reason": stop_reason,
        "invalid_request_count": invalid_count,
        "called_fingerprints": sorted(called_keys),
    }


def _seed_hypotheses_from_baseline(
    model_id: int,
    model_baseline: dict[str, Any],
    max_hypotheses: int = 1,
) -> list[dict[str, Any]]:
    hypotheses = []
    for entity in model_baseline.get("focus_entities", [])[: max_hypotheses or 1]:
        entity_name = entity.get("entity_name")
        fact_id = entity.get("fact_id")
        if not entity_name or not fact_id:
            continue
        metrics = entity.get("metrics", {}) or {}
        statement = (
            f"{entity_name}在银行转账关系分析（模型 86）中位于重点主体，"
            f"基础事实记录{int(metrics.get('record_count') or 0):,}条，"
            "需要验证其转账结构是否存在自转账主导、对手方集中、夜间高频或资金循环特征。"
        )
        hyp = make_hypothesis(
            len(hypotheses) + 1,
            str(entity_name),
            statement,
            [str(fact_id)],
            "model_baseline_focus_entity",
            [model_id],
        )
        hyp["hypothesis_id"] = f"m{model_id}_{hyp['hypothesis_id']}"
        hypotheses.append(hyp)
    return hypotheses


def _build_model_planner_context(
    model_id: int,
    model_baseline: dict[str, Any],
    hypothesis: dict[str, Any],
    previous_rounds: list[InvestigationRound],
    allowed_tools: list[str],
    already_called_fingerprints: list[str],
    rounds_remaining: int,
    calls_remaining: int,
) -> dict[str, Any]:
    context = model_baseline.get("model_context", {})
    return {
        "scope": {
            "model_id": model_id,
            "model_name": context.get("model_name"),
            "model_kind": context.get("model_kind"),
            "result_count": context.get("result_count"),
        },
        "hypothesis": {
            "hypothesis_id": hypothesis.get("hypothesis_id"),
            "entity_name": hypothesis.get("entity_name"),
            "statement": hypothesis.get("statement"),
            "basis_fact_ids": hypothesis.get("basis_fact_ids", []),
            "target_model_ids": hypothesis.get("target_model_ids", []),
        },
        "baseline_snapshot": _compress_model_baseline(model_baseline),
        "previous_rounds": [_round_summary(item) for item in previous_rounds[-3:]],
        "available_tools": [
            {
                "name": name,
                "description": TOOL_REGISTRY[name].description,
                "required_params": list(TOOL_REGISTRY[name].required_params),
                "optional_params": list(TOOL_REGISTRY[name].optional_params),
            }
            for name in allowed_tools
            if name in TOOL_REGISTRY
        ],
        "already_called_fingerprints": already_called_fingerprints[-8:],
        "budget_remaining": {"rounds": rounds_remaining, "tool_calls": calls_remaining},
        "rules": [
            "只能调用 available_tools 中的工具。",
            "参数必须来自 baseline_snapshot、hypothesis 或 previous_rounds。",
            "优先使用 transfer_structure_analyzer 验证模型86转账结构。",
            "不要重复调用 already_called_fingerprints 中完全相同的工具和参数。",
        ],
    }


def _ask_model_planner(
    config: Any,
    context: dict[str, Any],
    policy: FallbackPolicy,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    audit = {"llm_call_count": 0, "fallback_used": False}
    if not config.llm.is_complete:
        audit["fallback_used"] = True
        return None, audit
    prompt = (
        "你是模型级 Investigation Planner。围绕单个模型和单个假设选择下一步白名单工具。"
        "不要创造主体、金额、时间、模型或参数。优先使用专属工具获取结构化证据。"
        "只输出 JSON：{\"action\":\"call_tool|stop\",\"tool_name\":\"transfer_structure_analyzer\","
        "\"params\":{\"model_id\":86,\"entity_name\":\"...\"},\"reason\":\"中文理由\"}。"
    )
    candidates = model_candidates(config.llm.model, "planner")[:2]
    audit["model_candidates"] = candidates
    for attempt, model_name in enumerate(candidates, start=1):
        started = time.perf_counter()
        audit["llm_call_count"] += 1
        try:
            result = cached_chat_json(
                config,
                model_name=model_name,
                role="model-investigation-planner",
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": json.dumps(context, ensure_ascii=False, default=str)},
                ],
                max_tokens=min(config.llm.max_tokens, 1024),
            )
            update_audit_from_result(audit, result, model_name, attempt)
            audit["last_latency_seconds"] = round(time.perf_counter() - started, 3)
            payload = result.payload if isinstance(result.payload, dict) else {}
            payload.setdefault("source", "llm_model_investigation_planner")
            return payload, audit
        except Exception as exc:
            audit["last_latency_seconds"] = round(time.perf_counter() - started, 3)
            audit["last_planner_error"] = f"{type(exc).__name__}: {str(exc)[:200]}"
    audit["fallback_used"] = True
    return None, audit


def _deterministic_model_request(
    model_id: int,
    hypothesis: dict[str, Any],
    allowed_tools: list[str],
    called_keys: set[str],
) -> dict[str, Any] | None:
    entity_name = hypothesis.get("entity_name")
    preferred = "transfer_structure_analyzer" if "transfer_structure_analyzer" in allowed_tools else None
    if not preferred or not entity_name:
        return None
    params = {"model_id": model_id, "entity_name": str(entity_name)}
    if _call_key(preferred, params) in called_keys:
        return {"action": "stop", "reason": "专属工具已调用，避免重复补证。", "source": "deterministic_model_planner"}
    return {
        "action": "call_tool",
        "tool_name": preferred,
        "params": params,
        "reason": "基于模型86重点主体，先使用专属转账结构工具验证自转账、集中度、时间和循环特征。",
        "source": "deterministic_model_planner",
    }


def _validate_model_request(
    request: dict[str, Any],
    model_id: int,
    model_strategy: dict[str, Any],
    allowed_entities: set[str],
    called_keys: set[str],
) -> tuple[bool, str]:
    tool_name = str(request.get("tool_name") or "")
    params = request.get("params")
    if not isinstance(params, dict):
        return False, "params must be an object"
    params["model_id"] = int(params.get("model_id") or model_id)
    ok, reason = validate_tool_request(request, {model_id: model_strategy}, allowed_entities)
    if not ok:
        return False, reason
    if int(request["params"].get("model_id")) != model_id:
        return False, "model-level agent can only call its own model"
    key = _call_key(tool_name, request["params"])
    if key in called_keys:
        return False, "duplicate tool call"
    return True, ""


def _allowed_tools(strategy: dict[str, Any]) -> list[str]:
    tools = allowed_tools_for_strategy(strategy)
    if int(strategy.get("model_id") or 0) == 86 and "transfer_structure_analyzer" not in tools:
        tools = [*tools, "transfer_structure_analyzer"]
    return [tool for tool in tools if tool in TOOL_REGISTRY]


def _known_entities(model_baseline: dict[str, Any]) -> set[str]:
    values: set[str] = set()
    for entity in model_baseline.get("focus_entities", []):
        if entity.get("entity_name"):
            values.add(str(entity["entity_name"]))
    for relation in model_baseline.get("relation_profiles", []):
        for key in ("source", "target"):
            if relation.get(key):
                values.add(str(relation[key]))
    return values


def _compress_model_baseline(model_baseline: dict[str, Any]) -> dict[str, Any]:
    return {
        "focus_entities": [
            {
                "entity_name": entity.get("entity_name"),
                "fact_id": entity.get("fact_id"),
                "rank": entity.get("rank"),
                "record_count": (entity.get("metrics") or {}).get("record_count"),
                "amount_sum": (entity.get("metrics") or {}).get("amount_sum"),
                "amount_max": (entity.get("metrics") or {}).get("amount_max"),
                "distinct_counterparty_count": (entity.get("metrics") or {}).get("distinct_counterparty_count"),
                "risk_signals": [
                    flag.get("risk_tag") for flag in entity.get("anomaly_flags", []) if flag.get("risk_tag")
                ],
            }
            for entity in model_baseline.get("focus_entities", [])[:3]
        ],
        "key_relations": [
            {
                "fact_id": relation.get("fact_id"),
                "source": relation.get("source"),
                "target": relation.get("target"),
                "count": relation.get("count"),
                "amount_sum": relation.get("amount_sum"),
                "risk_signals": [
                    flag.get("risk_tag") for flag in relation.get("anomaly_flags", []) if flag.get("risk_tag")
                ],
            }
            for relation in model_baseline.get("relation_profiles", [])[:2]
        ],
        "key_metrics": {
            "result_count": (model_baseline.get("model_context") or {}).get("result_count"),
            "numeric_summary": (model_baseline.get("key_metrics") or {}).get("numeric_summary", {}),
        },
    }


def _round_summary(item: InvestigationRound) -> dict[str, Any]:
    return {
        "round": item.round_index,
        "tool_used": item.tool_name,
        "params": item.params,
        "result": item.tool_output_summary[:180],
        "tool_status_signal": item.tool_status_signal,
        "judge_status": item.judge_status,
        "judge_key_point": item.judge_assessment[:120],
        "rows_read": item.rows_read,
        "truncated": item.truncated,
    }


def _synthesize_claim(
    model_id: int,
    hypothesis: dict[str, Any],
    status: str,
    assessment: str,
    supplements: list[dict[str, Any]],
) -> ModelClaim:
    entity_name = str(hypothesis.get("entity_name") or "") or None
    supplement_ids = [
        str(fact.get("supplement_fact_id")) for fact in supplements if fact.get("supplement_fact_id")
    ]
    summaries = [str(fact.get("output", {}).get("summary") or "") for fact in supplements]
    statement = assessment or "模型级调查未形成明确判断。"
    if summaries and status == "enhanced":
        statement = f"{summaries[-1]}{assessment}"
    severity = "重点关注" if status == "enhanced" else "待核查" if status == "unresolved" else "一般关注"
    return ModelClaim(
        claim_id=f"m{model_id}_claim_01",
        model_id=model_id,
        entity_name=entity_name,
        hypothesis_id=str(hypothesis.get("hypothesis_id")),
        status=status,
        statement=statement,
        basis_fact_ids=[str(value) for value in hypothesis.get("basis_fact_ids", [])],
        supplement_fact_ids=supplement_ids,
        evidence_chain_id=f"chain_{hypothesis.get('hypothesis_id')}",
        limitation="模型级样板当前仅覆盖模型86专属转账结构分析，跨模型依赖图尚未启用。",
        severity=severity,
    )


def _bind_claim_chain_ids(claims: list[ModelClaim], evidence_chains: list[dict[str, Any]]) -> None:
    # Claims already use the stable chain id convention. Kept as a hook for later
    # version projection without mutating the baseline or evidence facts.
    return None


def _call_key(tool_name: str, params: dict[str, Any]) -> str:
    return f"{tool_name}:{json.dumps(params, ensure_ascii=False, sort_keys=True, default=str)}"


def _no_progress(rounds: list[InvestigationRound]) -> bool:
    if len(rounds) < 2:
        return False
    return rounds[-1].judge_status == rounds[-2].judge_status == "unresolved"

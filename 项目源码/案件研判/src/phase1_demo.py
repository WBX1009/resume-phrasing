from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import psycopg2.extras

from .anomaly_rules import attach_anomalies, build_followup_hints
from .config import AppConfig
from .cross_model import build_core_findings, build_cross_model_pack, build_root_summaries
from .data_loader import _connect, _placeholders, count_case_model_rows
from .entity_profile import build_model_profiles, safe_json
from .entity_resolver import EntityDictionary, build_entity_dictionary
from .execution_policy import FallbackPolicy, normalize_fallback_policy
from .field_profile import build_field_profiles
from .insight_report import build_llm_brief_pack, generate_llm_narratives, render_insight_report
from .model_class_pack import build_model_class_packs
from .model_strategy import DEFAULT_STRATEGY_PATH, StrategyBundle, load_strategy, strategy_for
from .planner import run_model_class_planner_demo, run_planner_demo
from .row_source import RowSource, create_row_source


CASE_ID = 13
DEMO_MODEL_IDS = load_strategy().model_ids
OBSERVATION_MODEL_IDS = load_strategy().observation_model_ids


@dataclass(frozen=True)
class DemoLoadedData:
    source: str
    sample_records: list[dict[str, Any]]
    model_info: dict[int, dict[str, Any]]
    orchestration: dict[int, list[dict[str, Any]]]
    row_counts: dict[int, int]
    model_tree_paths: dict[int, list[str]]
    echarts: dict[int, list[dict[str, Any]]]
    observation_counts: dict[int, int]
    case_info: dict[str, Any]
    warnings: list[str]


def _load_model_tree_paths(cur: Any, model_ids: list[int]) -> dict[int, list[str]]:
    model_ph = _placeholders(model_ids)
    cur.execute(
        f"""
        SELECT mi.id AS model_id, mi.tree_id, mi.model_name
        FROM ch_model_info mi
        WHERE mi.id IN ({model_ph})
        """,
        model_ids,
    )
    model_tree_rows = [dict(row) for row in cur.fetchall()]

    cur.execute(
        """
        SELECT id, parent_id, ancestors, node_name, model_category
        FROM ch_model_tree
        WHERE COALESCE(delete_flag, '0') <> '1'
        """,
    )
    tree_by_id = {int(row["id"]): dict(row) for row in cur.fetchall()}

    def path_for(tree_id: Any, model_name: str) -> list[str]:
        if tree_id is None:
            return [model_name]
        path: list[str] = []
        seen: set[int] = set()
        current = int(tree_id)
        while current and current not in seen and current in tree_by_id:
            seen.add(current)
            row = tree_by_id[current]
            name = row.get("node_name") or row.get("model_category")
            if name:
                path.append(str(name))
            parent = row.get("parent_id")
            current = int(parent) if parent not in [None, 0, "0", ""] else 0
        path.reverse()
        path.append(model_name)
        return path

    result: dict[int, list[str]] = {}
    for row in model_tree_rows:
        model_id = int(row["model_id"])
        result[model_id] = path_for(row.get("tree_id"), row.get("model_name") or f"model_{model_id}")
    return result


def _parse_json_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str) and value.strip():
        import json

        try:
            payload = json.loads(value)
            return payload if isinstance(payload, dict) else {}
        except Exception:
            return {}
    return {}


def _load_echarts(cur: Any, model_ids: list[int]) -> dict[int, list[dict[str, Any]]]:
    model_ph = _placeholders(model_ids)
    try:
        cur.execute(
            f"""
            SELECT model_id, chart_name, chart_type, config_mode, dimension_mapping,
                   style_config, is_active, delete_flag
            FROM ch_model_echarts
            WHERE model_id IN ({model_ph})
              AND COALESCE(delete_flag, '') <> 'DELETE'
              AND COALESCE(is_active, '0') = '1'
            ORDER BY model_id, id
            """,
            model_ids,
        )
    except Exception:
        return {}
    result: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in cur.fetchall():
        item = dict(row)
        item["dimension_mapping"] = _parse_json_mapping(item.get("dimension_mapping"))
        item["style_config"] = _parse_json_mapping(item.get("style_config"))
        result[int(item["model_id"])].append(item)
    return dict(result)


def _model_context(
    case_id: int,
    model_id: int,
    strategy: dict[str, Any],
    loaded: DemoLoadedData,
    sample_count: int,
    sample_limit: int,
) -> dict[str, Any]:
    info = loaded.model_info.get(model_id, {})
    path = loaded.model_tree_paths.get(model_id) or [info.get("model_name") or f"model_{model_id}"]
    result_count = loaded.row_counts.get(model_id, 0)
    return {
        "case_id": case_id,
        "model_id": model_id,
        "model_name": info.get("model_name") or f"model_{model_id}",
        "model_tree_path": path,
        "model_tree_path_text": " / ".join(path),
        "root_category": strategy.get("root_category"),
        "demo_level": strategy.get("demo_level"),
        "demo_role": strategy.get("demo_role"),
        "model_kind": strategy.get("model_kind"),
        "model_desc": info.get("model_desc") or "",
        "required_data": info.get("required_data") or "",
        "result_count": result_count,
        "sample_limit": sample_limit,
        "loaded_sample_count": sample_count,
        "aggregation_scope": "full_result_set" if result_count else "no_data",
    }


def load_phase1_demo_data(
    config: AppConfig,
    case_id: int,
    model_ids: list[int],
    bundle: StrategyBundle,
    row_source: RowSource | None = None,
) -> DemoLoadedData:
    row_source = row_source or create_row_source(config, "db")
    observation_counts_raw = row_source.count_model_rows(case_id, bundle.observation_model_ids)
    observation_counts = {
        model_id: observation_counts_raw.get(model_id, 0)
        for model_id in bundle.observation_model_ids
    }
    row_counts = row_source.count_model_rows(case_id, model_ids)
    model_info = row_source.load_model_info(model_ids)
    orchestration = row_source.load_orchestration(model_ids)
    model_tree_paths = row_source.load_model_tree_paths(model_ids)
    echarts = row_source.load_echarts([*model_ids, *bundle.observation_model_ids])
    case_info = row_source.load_case_info(case_id)
    sample_records: list[dict[str, Any]] = []
    for model_id in model_ids:
        for index, (row_id, payload) in enumerate(row_source.iter_model_rows(case_id, model_id)):
            if index >= bundle.sample_limit:
                break
            sample_records.append(
                {
                    "id": row_id,
                    "model_id": model_id,
                    "json_data": payload,
                    "case_id": case_id,
                    "sql_text": None,
                    "select_fields": None,
                    "create_date": None,
                }
            )

    warnings = list(row_source.warnings)
    if not case_info.get("event_name") or not case_info.get("event_code"):
        warnings.append("case metadata is incomplete; report overview will use conservative placeholders.")
    if observation_counts.get(100, 0) != 0:
        warnings.append("model_id=100 is no longer empty; confirm whether it should replace model_id=101.")
    if observation_counts.get(119, 0) > 0:
        warnings.append("model_id=119 has rows but is excluded from the report body because it was not displayed in the frontend selected list.")
    return DemoLoadedData(
        source=row_source.source_name,
        sample_records=sample_records,
        model_info=model_info,
        orchestration=dict(orchestration),
        row_counts=row_counts,
        model_tree_paths=model_tree_paths,
        echarts=echarts,
        observation_counts=observation_counts,
        case_info=case_info,
        warnings=warnings,
    )


def iter_model_rows(config: AppConfig, case_id: int, model_id: int, batch_size: int = 5000) -> Iterable[tuple[Any, dict[str, Any]]]:
    with _connect(config) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, json_data
                FROM ch_model_info_data
                WHERE case_id = %s AND model_id = %s
                ORDER BY id
                """,
                [case_id, model_id],
            )
            while True:
                rows = cur.fetchmany(batch_size)
                if not rows:
                    break
                for row_id, raw_payload in rows:
                    try:
                        yield row_id, safe_json(raw_payload)
                    except Exception:
                        continue


def _field_profile_summary(profile: dict[str, Any]) -> dict[str, Any]:
    roles: dict[str, list[str]] = defaultdict(list)
    for field in profile.get("fields", []):
        for role in field.get("roles", []):
            roles[role].append(field["field_name"])
    return {
        "profile_row_count": profile.get("row_count", 0),
        "parsed_row_count": profile.get("parsed_row_count", 0),
        "json_parse_errors": profile.get("json_parse_errors", 0),
        "field_count": len(profile.get("fields", [])),
        "fields_by_role": dict(roles),
        "top_fields": [
            {
                "field_name": field["field_name"],
                "coverage_rate": field.get("coverage_rate"),
                "parsed_type": field.get("parsed_type"),
                "roles": field.get("roles", []),
            }
            for field in profile.get("fields", [])[:12]
        ],
    }


def _evidence_refs_from_profiles(model_id: int, focus_entities: list[dict[str, Any]], relation_profiles: list[dict[str, Any]]) -> list[dict[str, Any]]:
    refs = []
    for item in focus_entities:
        records = item.get("evidence_records", [])
        if not records:
            records = [record["evidence"] for record in item.get("top_amount_records", []) if record.get("evidence")]
        if item.get("evidence_refs"):
            refs.append(
                {
                    "fact_id": item["fact_id"],
                    "model_id": model_id,
                    "evidence_count": len(records),
                    "records": records[:3],
                }
            )
    for item in relation_profiles:
        refs.append(
            {
                "fact_id": item["fact_id"],
                "model_id": model_id,
                "evidence_count": len(item.get("evidence_records", [])),
                "records": item.get("evidence_records", [])[:3],
            }
        )
    return refs


def _build_entity_dictionary_factpack(
    context: dict[str, Any],
    dictionary: EntityDictionary,
    profile: dict[str, Any],
    strategy: dict[str, Any],
) -> dict[str, Any]:
    return {
        "model_context": context,
        "model_tree_path": context["model_tree_path"],
        "field_profile_summary": _field_profile_summary(profile),
        "key_metrics": {
            "result_count": context["result_count"],
            "aggregation_basis": "entity_dictionary",
            "person_count": len(dictionary.people),
        },
        "entity_dictionary": dictionary.to_jsonable(),
        "focus_entities": [],
        "entity_profiles": [],
        "relation_profiles": [],
        "relation_summary": {},
        "category_topk": [],
        "anomaly_flags": [],
        "followup_hints": [],
        "evidence_refs": [],
        "frontend_query_params": {"caseId": context["case_id"], "modelId": context["model_id"], "conditions": []},
        "llm_narrative": {},
        "shared_entity_types": ["Person", "Account", "Phone"],
        "evidence_chain": [],
        "allowed_followup_models": strategy.get("allowed_followup_models", []),
        "allowed_tools": strategy.get("allowed_tools", []),
    }


def build_model_factpacks(
    config: AppConfig,
    case_id: int,
    model_ids: list[int],
    loaded: DemoLoadedData,
    bundle: StrategyBundle,
    field_profiles: dict[int, dict[str, Any]],
    row_source: RowSource | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    row_source = row_source or create_row_source(config, "db")
    sample_counts = Counter(int(row["model_id"]) for row in loaded.sample_records)
    contexts = {
        model_id: _model_context(
            case_id,
            model_id,
            strategy_for(bundle, model_id),
            loaded,
            sample_counts.get(model_id, 0),
            bundle.sample_limit,
        )
        for model_id in model_ids
    }

    dictionary_strategy = strategy_for(bundle, bundle.entity_dictionary_model_id)
    dictionary_rows = [
        payload for _, payload in row_source.iter_model_rows(case_id, bundle.entity_dictionary_model_id)
    ]
    dictionary = build_entity_dictionary(dictionary_rows, dictionary_strategy)

    factpacks = []
    for model_id in model_ids:
        strategy = strategy_for(bundle, model_id)
        context = contexts[model_id]
        profile = field_profiles.get(model_id, {"fields": []})
        if strategy.get("model_kind") == "entity_dictionary":
            factpacks.append(_build_entity_dictionary_factpack(context, dictionary, profile, strategy))
            continue
        factpack = {
            "model_context": context,
            "model_tree_path": context["model_tree_path"],
            "field_profile_summary": _field_profile_summary(profile),
            "key_metrics": {
                "result_count": context["result_count"],
                "sample_limit": bundle.sample_limit,
                "loaded_sample_count": context["loaded_sample_count"],
                "aggregation_basis": "full_result_set" if context["result_count"] else "no_data",
            },
            "focus_entities": [],
            "entity_profiles": [],
            "relation_profiles": [],
            "relation_summary": {},
            "category_topk": [],
            "anomaly_flags": [],
            "followup_hints": [],
            "evidence_refs": [],
            "frontend_query_params": {"caseId": case_id, "modelId": model_id, "conditions": []},
            "llm_narrative": {},
            "shared_entity_types": strategy.get("shared_entity_types", ["Person", "Account"]),
            "evidence_chain": [],
            "allowed_followup_models": strategy.get("allowed_followup_models", []),
            "allowed_tools": strategy.get("allowed_tools", []),
        }
        if not context["result_count"]:
            factpack["key_metrics"]["no_data_reason"] = strategy.get("no_data_reason", "当前案例下该模型没有缓存结果。")
            factpacks.append(factpack)
            continue

        profiles = build_model_profiles(
            model_id,
            row_source.iter_model_rows(case_id, model_id),
            strategy,
            dictionary,
            {
                "entities": bundle.top_entity_limit,
                "relations": bundle.top_relation_limit,
                "evidence": bundle.evidence_per_item,
            },
            bundle.thresholds,
        )
        factpack["key_metrics"]["numeric_summary"] = profiles["numeric_summary"]
        factpack["key_metrics"]["json_parse_errors_full_scan"] = profiles["parse_errors"]
        factpack["focus_entities"] = profiles["focus_entities"]
        factpack["entity_profiles"] = profiles["focus_entities"]
        factpack["relation_profiles"] = profiles["relation_profiles"]
        factpack["relation_summary"] = profiles["relation_summary"]
        factpack["category_topk"] = profiles["category_topk"]
        attach_anomalies(factpack, strategy, bundle.thresholds)
        factpack["followup_hints"] = build_followup_hints(factpack, strategy)
        factpack["evidence_refs"] = _evidence_refs_from_profiles(
            model_id,
            factpack["focus_entities"],
            factpack["relation_profiles"],
        )
        factpacks.append(factpack)

    cross_pack = build_cross_model_pack(factpacks)
    case_pack = {
        "case_id": case_id,
        "case_info": loaded.case_info,
        "data_source": loaded.source,
        "sample_limit_per_model": bundle.sample_limit,
        "aggregation_basis": "full cached result set; sample rows are only for field profile preview and evidence snippets",
        "model_count": len(factpacks),
        "total_result_count": sum(fp["model_context"]["result_count"] for fp in factpacks),
        "root_summaries": build_root_summaries(factpacks),
        "cross_model_entities": cross_pack["cross_model_entities"],
        "hypothesis_candidates": cross_pack["hypothesis_candidates"],
        "core_findings": build_core_findings(factpacks, cross_pack),
        "observation_counts": loaded.observation_counts,
        "warnings": loaded.warnings,
    }
    return factpacks, case_pack


def build_phase1_demo(
    config: AppConfig,
    case_id: int = CASE_ID,
    model_ids: list[int] | None = None,
    strategy_path: Path | None = None,
    report_mode: str = "both",
    analysis_mode: str = "template",
    data_source: str = "db",
    agent_loop_config: Any | None = None,
    full_agent_model_ids: list[int] | None = None,
) -> dict[str, Any]:
    fallback_policy = normalize_fallback_policy(
        getattr(agent_loop_config, "fallback_policy", FallbackPolicy.REVIEW_SAFE.value)
    )
    bundle = load_strategy(strategy_path or DEFAULT_STRATEGY_PATH)
    model_ids = model_ids or bundle.model_ids
    row_source = create_row_source(config, data_source)
    loaded = load_phase1_demo_data(config, case_id, model_ids, bundle, row_source=row_source)
    records_by_model: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for record in loaded.sample_records:
        payload = safe_json(record.get("json_data"))
        sampled = dict(record)
        sampled["json_data"] = payload
        records_by_model[int(record["model_id"])].append(sampled)
    field_profiles = build_field_profiles(records_by_model)
    model_factpacks, case_pack = build_model_factpacks(
        config,
        case_id,
        model_ids,
        loaded,
        bundle,
        field_profiles,
        row_source=row_source,
    )
    model_class_packs = build_model_class_packs(model_factpacks, bundle)
    llm_brief_pack = build_llm_brief_pack(model_factpacks, case_pack)
    planner_result = None
    if analysis_mode in {"planner-demo", "class-planner-demo", "agent-loop-demo", "full-agent"} and fallback_policy != FallbackPolicy.FACTS_ONLY:
        dictionary_strategy = strategy_for(bundle, bundle.entity_dictionary_model_id)
        dictionary_rows = [
            payload for _, payload in row_source.iter_model_rows(case_id, bundle.entity_dictionary_model_id)
        ]
        dictionary = build_entity_dictionary(dictionary_rows, dictionary_strategy)
        if analysis_mode == "class-planner-demo":
            planner_result = run_model_class_planner_demo(
                config,
                case_id,
                model_factpacks,
                model_class_packs,
                case_pack,
                bundle,
                dictionary,
                row_source=row_source,
            )
        elif analysis_mode == "agent-loop-demo":
            from .agent_loop import run_agent_loop_demo

            planner_result = run_agent_loop_demo(
                config,
                case_id,
                model_factpacks,
                model_class_packs,
                case_pack,
                bundle,
                dictionary,
                row_source,
                loop_config=agent_loop_config,
            )
        elif analysis_mode == "full-agent":
            from .agent_loop import run_agent_loop_demo
            from .model_investigation import (
                ModelInvestigationConfig,
                result_to_planner_result,
                run_model_investigation,
            )

            model_results = []
            for target_model_id in full_agent_model_ids or [86]:
                baseline = next(
                    (item for item in model_factpacks if int(item.get("model_context", {}).get("model_id") or 0) == int(target_model_id)),
                    None,
                )
                if baseline is None:
                    continue
                inv_config = ModelInvestigationConfig(
                    model_id=int(target_model_id),
                    max_rounds=getattr(agent_loop_config, "max_agent_rounds", 3),
                    max_tool_calls=getattr(agent_loop_config, "max_tool_calls_per_hypothesis", 3),
                    max_invalid_requests=getattr(agent_loop_config, "max_invalid_requests", 2),
                    max_tool_rows_per_call=getattr(agent_loop_config, "max_tool_rows_per_call", 50_000),
                    max_tool_unique_groups=getattr(agent_loop_config, "max_tool_unique_groups", 20_000),
                    max_tool_evidence_records=getattr(agent_loop_config, "max_tool_evidence_records", 5),
                    fallback_policy=fallback_policy.value,
                )
                model_results.append(
                    run_model_investigation(
                        config,
                        case_id,
                        int(target_model_id),
                        baseline,
                        strategy_for(bundle, int(target_model_id)),
                        bundle,
                        dictionary,
                        row_source,
                        investigation_config=inv_config,
                    )
                )
            model_claims = [
                claim
                for result in model_results
                for claim in result.claims
            ]
            class_result = run_agent_loop_demo(
                config,
                case_id,
                model_factpacks,
                model_class_packs,
                case_pack,
                bundle,
                dictionary,
                row_source,
                loop_config=agent_loop_config,
                model_claims=[claim.__dict__ for claim in model_claims],
            )
            planner_result = result_to_planner_result(model_results, class_result)
        else:
            planner_result = run_planner_demo(
                config,
                case_id,
                model_factpacks,
                case_pack,
                bundle,
                dictionary,
                row_source=row_source,
            )
        llm_brief_pack = planner_result.get("llm_brief_pack", llm_brief_pack)
    llm_audit = {"mode": report_mode, "status": "not_requested"}
    if report_mode in {"llm", "both"}:
        llm_audit = generate_llm_narratives(config, model_factpacks, case_pack)
    contexts = [factpack["model_context"] for factpack in model_factpacks]
    template_report = render_insight_report(
        case_id,
        model_factpacks,
        case_pack,
        mode="template",
        planner_result=None,
    )
    planner_report = (
        render_insight_report(
            case_id,
            model_factpacks,
            case_pack,
            mode="template",
            planner_result=planner_result,
        )
        if planner_result
        else None
    )
    llm_narrative_report = (
        render_insight_report(case_id, model_factpacks, case_pack, mode="llm", planner_result=None)
        if report_mode in {"llm", "both"}
        else None
    )
    final_report = None
    report_assertions: list[dict[str, Any]] = []
    final_report_audit = {"mode": "structured-final-writer", "status": "not_requested"}
    if report_mode == "final" and analysis_mode in {"agent-loop-demo", "full-agent"} and fallback_policy == FallbackPolicy.FACTS_ONLY:
        final_report = template_report
        final_report_audit = {
            "mode": "facts-only",
            "status": "degraded",
            "fallback_used": True,
            "fallback_policy": fallback_policy.value,
            "assertion_count": 0,
            "reason": "facts_only policy skipped Agent conclusions and Final Writer",
        }
    elif report_mode == "final" and analysis_mode in {"agent-loop-demo", "full-agent"} and planner_result:
        from .final_agent_report import generate_final_agent_report

        final_report, report_assertions, final_report_audit = generate_final_agent_report(
            config,
            case_id,
            model_factpacks,
            model_class_packs,
            case_pack,
            planner_result,
            fallback_policy=fallback_policy,
        )
    llm_report = final_report
    planner_audit = planner_result.get("planner_audit", {}) if planner_result else {}
    judge_audits = planner_audit.get("judge_audits", [])
    judge_enabled = any(audit.get("judge_call_count", 0) > 0 for audit in judge_audits)
    report_manifest = {
        "execution_policy": {
            "fallback_policy": fallback_policy.value,
            "tool_budget": planner_audit.get("tool_budget", {}),
            "tool_truncation_count": int(planner_audit.get("tool_truncation_count") or 0),
            "rejected_request_count": int(planner_audit.get("rejected_request_count") or 0),
        },
        "template_report": {
            "generated": True,
            "description": "纯模板报告：只展示确定性基础事实。",
        },
        "llm_narrative_report": {
            "generated": llm_narrative_report is not None,
            "description": "局部大模型叙事报告：模板骨架 + 模型级 narrative 润色；不等同于完整 LLM 报告。",
            "audit": llm_audit,
        },
        "planner_report": {
            "generated": planner_report is not None,
            "description": "调试报告：展示基础事实、规划、补证事实和 Judge 证据链。",
            "planner_mode": planner_result.get("planner_audit", {}).get("mode") if planner_result else None,
            "judge_enabled": judge_enabled,
            "judge_call_count": sum(int(audit.get("judge_call_count") or 0) for audit in judge_audits),
            "judge_success_count": sum(int(audit.get("judge_success_count") or 0) for audit in judge_audits),
        },
        "final_report": {
            "generated": final_report is not None,
            "description": (
                "确定性事实报告：降级策略禁止生成 Agent 研判断言。"
                if fallback_policy == FallbackPolicy.FACTS_ONLY
                else "最终模型研判报告：结构化断言经证据引用校验后确定性渲染。"
            ),
            "audit": final_report_audit,
        },
    }
    return {
        "contexts": contexts,
        "field_profiles": field_profiles,
        "model_factpacks": model_factpacks,
        "model_class_packs": model_class_packs,
        "case_pack": {**case_pack, "llm_audit": llm_audit},
        "llm_brief_pack": llm_brief_pack,
        "planner_result": planner_result,
        "template_report": template_report,
        "planner_report": planner_report,
        "llm_narrative_report": llm_narrative_report,
        "llm_report": llm_report,
        "final_report": final_report,
        "report_assertions": report_assertions,
        "final_report_audit": final_report_audit,
        "report_manifest": report_manifest,
        "loaded": loaded,
        "strategy": bundle,
    }


def render_phase1_demo_report(
    case_id: int,
    contexts: list[dict[str, Any]],
    model_factpacks: list[dict[str, Any]],
    case_pack: dict[str, Any],
) -> str:
    return render_insight_report(case_id, model_factpacks, case_pack, mode="template")

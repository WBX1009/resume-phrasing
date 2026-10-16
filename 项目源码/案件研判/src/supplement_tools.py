from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from datetime import datetime
from typing import Any, Iterable
import copy
import json

from .entity_profile import build_model_profiles, preview_row, safe_float, safe_json
from .entity_resolver import EntityDictionary, clean_value
from .evidence_chain import is_incomplete_tool_output
from .field_profile import parse_datetime
from .model_tools import transfer_structure_analyzer
from .row_source import RowQuery, RowSource
from .stable_ids import stable_id


@dataclass(frozen=True)
class ToolExecutionBudget:
    max_rows_per_call: int = 50_000
    max_unique_groups: int = 20_000
    max_evidence_records: int = 5

    def __post_init__(self) -> None:
        for name, value in self.__dict__.items():
            if int(value) <= 0:
                raise ValueError(f"{name} must be positive")


def _iter_rows(
    case_id: int,
    model_id: int,
    row_source: RowSource | None,
    query: RowQuery,
) -> Iterable[tuple[Any, dict[str, Any]]]:
    if row_source is None:
        raise RuntimeError("row_source is required for row-scanning supplement tools")
    yield from row_source.iter_model_rows_query(case_id, model_id, query)


def _read_bounded_rows(
    case_id: int,
    model_id: int,
    row_source: RowSource | None,
    query: RowQuery,
    max_rows: int,
) -> tuple[list[tuple[Any, dict[str, Any]]], bool]:
    if max_rows <= 0:
        return [], True
    rows = list(_iter_rows(case_id, model_id, row_source, replace(query, limit=max_rows + 1)))
    return rows[:max_rows], len(rows) > max_rows


def _budget_fields(rows_read: int, truncated: bool, limit_reason: str | None = None) -> dict[str, Any]:
    return {
        "rows_read": rows_read,
        "truncated": truncated,
        "limit_reason": limit_reason if truncated else None,
        "aggregation_complete": not truncated,
    }


def _finalize_tool_result(
    case_id: int,
    tool_name: str,
    params: dict[str, Any],
    output: dict[str, Any],
    extra_fact_refs: list[str] | None = None,
) -> dict[str, Any]:
    request_fingerprint = stable_id(
        "req",
        {"case_id": case_id, "tool_name": tool_name, "params": params},
    )
    fact_payload = {
        "case_id": case_id,
        "tool_name": tool_name,
        "params": params,
        "output": output,
    }
    fact_id = stable_id("tool_fact", fact_payload)
    output["tool_name"] = tool_name
    output["request_fingerprint"] = request_fingerprint
    output["fact_id"] = fact_id
    output["fact_refs"] = [fact_id, *(extra_fact_refs or [])]
    if is_incomplete_tool_output(output):
        output["status_signal"] = "unresolved"
    return output


def _candidate_fields(strategy: dict[str, Any]) -> list[str]:
    relation = strategy.get("relation") or {}
    fields = [
        *list(strategy.get("person_fields") or []),
        *list(strategy.get("account_fields") or []),
        *list(strategy.get("counterparty_fields") or []),
        strategy.get("category_field"),
        relation.get("source"),
        relation.get("target"),
    ]
    return [field for field in dict.fromkeys(fields) if field]


def _amount_fields(strategy: dict[str, Any]) -> list[str]:
    fields = [strategy.get("primary_amount_field"), *list(strategy.get("amount_fields") or [])]
    return [field for field in dict.fromkeys(fields) if field]


def _time_fields(strategy: dict[str, Any]) -> list[str]:
    return list(strategy.get("time_fields") or [])


def _row_has_value(payload: dict[str, Any], fields: list[str], value: str | None) -> bool:
    if not value:
        return True
    wanted = str(value)
    for field in fields:
        raw = clean_value(payload.get(field))
        if raw and (raw == wanted or wanted in raw or raw in wanted):
            return True
    return False


def _first_amount(payload: dict[str, Any], strategy: dict[str, Any]) -> float | None:
    for field in _amount_fields(strategy):
        value = safe_float(payload.get(field))
        if value is not None:
            return value
    return None


def _first_datetime(payload: dict[str, Any], strategy: dict[str, Any]) -> datetime | None:
    for field in _time_fields(strategy):
        value = parse_datetime(payload.get(field))
        if value:
            return value
    return None


def _date_key(dt: datetime | None) -> str:
    return dt.strftime("%Y-%m-%d") if dt else "unknown_date"


def _preferred_fields(strategy: dict[str, Any]) -> list[str]:
    relation = strategy.get("relation") or {}
    fields = [
        *list(strategy.get("person_fields") or []),
        *list(strategy.get("account_fields") or []),
        *list(strategy.get("counterparty_fields") or []),
        *_amount_fields(strategy),
        *_time_fields(strategy),
        relation.get("source"),
        relation.get("target"),
    ]
    return [field for field in dict.fromkeys(fields) if field]


def entity_model_slice(
    config: Any,
    case_id: int,
    strategy: dict[str, Any],
    dictionary: EntityDictionary,
    params: dict[str, Any],
    thresholds: dict[str, Any],
    row_source: RowSource | None = None,
    budget: ToolExecutionBudget | None = None,
) -> dict[str, Any]:
    budget = budget or ToolExecutionBudget()
    model_id = int(params["model_id"])
    entity_name = str(params["entity_name"])
    fields = _candidate_fields(strategy)
    rows, truncated = _read_bounded_rows(
        case_id,
        model_id,
        row_source,
        RowQuery(any_field_equals=tuple((field, entity_name) for field in fields)),
        budget.max_rows_per_call,
    )
    filtered = [(row_id, payload) for row_id, payload in rows if _row_has_value(payload, fields, entity_name)]
    profiles = build_model_profiles(
        model_id,
        filtered,
        strategy,
        dictionary,
        {"entities": 3, "relations": 3, "evidence": budget.max_evidence_records},
        thresholds,
    )
    output = {
        "summary": f"{entity_name} 在模型 {model_id} 的定向切片读取并命中 {len(filtered)} 条记录。",
        "row_count": len(filtered),
        "focus_entities": profiles.get("focus_entities", [])[:3],
        "relation_profiles": profiles.get("relation_profiles", [])[:3],
        "numeric_summary": profiles.get("numeric_summary", {}),
        "evidence_refs": _collect_evidence_refs(profiles),
        "status_signal": "enhanced" if filtered else "weakened",
        **_budget_fields(len(rows), truncated, "max_rows_per_call"),
    }
    return _finalize_tool_result(case_id, "entity_model_slice", params, output)


def relation_neighbor_subgraph(
    config: Any,
    case_id: int,
    strategy: dict[str, Any],
    params: dict[str, Any],
    row_source: RowSource | None = None,
    budget: ToolExecutionBudget | None = None,
) -> dict[str, Any]:
    budget = budget or ToolExecutionBudget()
    model_id = int(params["model_id"])
    entity_name = str(params.get("entity_name") or params.get("source") or params.get("target"))
    relation = strategy.get("relation") or {}
    source_field = relation.get("source")
    target_field = relation.get("target")
    preferred = _preferred_fields(strategy)
    outgoing: dict[tuple[str, str], dict[str, Any]] = {}
    incoming: dict[tuple[str, str], dict[str, Any]] = {}
    row_count = 0
    evidence: list[dict[str, Any]] = []

    rows, rows_truncated = _read_bounded_rows(
        case_id,
        model_id,
        row_source,
        RowQuery(any_field_equals=((source_field, entity_name), (target_field, entity_name))),
        budget.max_rows_per_call,
    )
    groups_truncated = False
    for row_id, payload in rows:
        source = clean_value(payload.get(source_field))
        target = clean_value(payload.get(target_field))
        if not source or not target or entity_name not in {source, target}:
            continue
        row_count += 1
        amount = _first_amount(payload, strategy) or 0.0
        bucket = outgoing if source == entity_name else incoming
        key = (source, target)
        if key not in bucket and len(outgoing) + len(incoming) >= budget.max_unique_groups:
            groups_truncated = True
            continue
        item = bucket.setdefault(key, {"source": source, "target": target, "count": 0, "amount_sum": 0.0})
        item["count"] += 1
        item["amount_sum"] += amount
        if len(evidence) < budget.max_evidence_records:
            evidence.append(preview_row(row_id, model_id, payload, preferred))

    top_outgoing = _top_edges(outgoing)
    top_incoming = _top_edges(incoming)
    truncated = rows_truncated or groups_truncated
    output = {
        "summary": f"{entity_name} 在模型 {model_id} 的关系邻域命中 {row_count} 条记录。",
        "row_count": row_count,
        "top_outgoing": top_outgoing,
        "top_incoming": top_incoming,
        "evidence_records": evidence,
        "evidence_refs": [],
        "status_signal": "enhanced" if row_count else "weakened",
        **_budget_fields(
            len(rows),
            truncated,
            "max_rows_per_call" if rows_truncated else "max_unique_groups" if groups_truncated else None,
        ),
    }
    result = _finalize_tool_result(case_id, "relation_neighbor_subgraph", params, output)
    result["evidence_refs"] = [result["fact_id"]] if evidence else []
    return result


def fund_flow_path_probe(
    config: Any,
    case_id: int,
    strategy: dict[str, Any],
    params: dict[str, Any],
    row_source: RowSource | None = None,
    budget: ToolExecutionBudget | None = None,
) -> dict[str, Any]:
    budget = budget or ToolExecutionBudget()
    model_id = int(params["model_id"])
    source_entity = str(params["source"])
    target_entity = str(params.get("target") or "")
    relation = strategy.get("relation") or {}
    source_field = relation.get("source")
    target_field = relation.get("target")
    edge_map: dict[tuple[str, str], dict[str, Any]] = {}
    preferred = _preferred_fields(strategy)
    evidence_by_edge: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)

    first_rows, first_truncated = _read_bounded_rows(
        case_id,
        model_id,
        row_source,
        RowQuery(all_field_equals=((source_field, source_entity),)),
        budget.max_rows_per_call,
    )
    all_mids = {
        clean_value(payload.get(target_field))
        for _, payload in first_rows
        if clean_value(payload.get(target_field))
    }
    mids_truncated = len(all_mids) > budget.max_unique_groups
    mids = set(sorted(all_mids)[: budget.max_unique_groups])
    remaining = max(0, budget.max_rows_per_call - len(first_rows))
    if mids and remaining:
        second_rows, second_truncated = _read_bounded_rows(
            case_id,
            model_id,
            row_source,
            RowQuery(field_in=((source_field, tuple(sorted(mids))),)),
            remaining,
        )
    else:
        second_rows, second_truncated = [], bool(mids and not remaining)
    seen_row_ids: set[Any] = set()
    groups_truncated = mids_truncated
    for row_id, payload in [*first_rows, *second_rows]:
        if row_id in seen_row_ids:
            continue
        seen_row_ids.add(row_id)
        source = clean_value(payload.get(source_field))
        target = clean_value(payload.get(target_field))
        if not source or not target:
            continue
        amount = _first_amount(payload, strategy) or 0.0
        key = (source, target)
        if key not in edge_map and len(edge_map) >= budget.max_unique_groups:
            groups_truncated = True
            continue
        item = edge_map.setdefault(key, {"source": source, "target": target, "count": 0, "amount_sum": 0.0})
        item["count"] += 1
        item["amount_sum"] += amount
        if len(evidence_by_edge[key]) < min(2, budget.max_evidence_records):
            evidence_by_edge[key].append(preview_row(row_id, model_id, payload, preferred))

    outgoing = [edge for key, edge in edge_map.items() if key[0] == source_entity]
    outgoing.sort(key=lambda item: (item["amount_sum"], item["count"]), reverse=True)
    two_hop_paths = []
    for first in outgoing[:20]:
        mid = first["target"]
        second_edges = [
            edge
            for key, edge in edge_map.items()
            if key[0] == mid and (not target_entity or key[1] == target_entity)
        ]
        second_edges.sort(key=lambda item: (item["amount_sum"], item["count"]), reverse=True)
        for second in second_edges[:5]:
            first_key = (first["source"], first["target"])
            second_key = (second["source"], second["target"])
            two_hop_paths.append(
                {
                    "path": [first["source"], first["target"], second["target"]],
                    "first_edge_amount_sum": round(first["amount_sum"], 2),
                    "second_edge_amount_sum": round(second["amount_sum"], 2),
                    "first_edge_count": first["count"],
                    "second_edge_count": second["count"],
                    "amount_gap": round(abs(first["amount_sum"] - second["amount_sum"]), 2),
                    "evidence_records": [
                        *evidence_by_edge.get(first_key, [])[:1],
                        *evidence_by_edge.get(second_key, [])[:1],
                    ],
                }
            )
    two_hop_paths.sort(key=lambda item: (-(item["amount_gap"]), item["first_edge_amount_sum"] + item["second_edge_amount_sum"]))
    two_hop_paths = sorted(
        two_hop_paths,
        key=lambda item: (item["first_edge_amount_sum"] + item["second_edge_amount_sum"], -item["amount_gap"]),
        reverse=True,
    )[:5]
    truncated = first_truncated or second_truncated or groups_truncated
    output = {
        "summary": f"{source_entity} 在模型 {model_id} 中形成 {len(outgoing)} 条直接下游边，探查到 {len(two_hop_paths)} 条两跳候选路径。",
        "source": source_entity,
        "target": target_entity or None,
        "direct_outgoing_top": _rank_edges(outgoing[:5]),
        "two_hop_paths": two_hop_paths,
        "evidence_refs": [],
        "status_signal": "enhanced" if two_hop_paths or outgoing else "weakened",
        **_budget_fields(
            len(first_rows) + len(second_rows),
            truncated,
            "max_rows_per_call" if first_truncated or second_truncated else "max_unique_groups" if groups_truncated else None,
        ),
    }
    result = _finalize_tool_result(case_id, "fund_flow_path_probe", params, output)
    result["evidence_refs"] = [result["fact_id"]] if two_hop_paths else []
    return result


def same_amount_window_probe(
    config: Any,
    case_id: int,
    strategy: dict[str, Any],
    params: dict[str, Any],
    row_source: RowSource | None = None,
    budget: ToolExecutionBudget | None = None,
) -> dict[str, Any]:
    budget = budget or ToolExecutionBudget()
    model_id = int(params["model_id"])
    entity_name = str(params["entity_name"])
    fields = _candidate_fields(strategy)
    preferred = _preferred_fields(strategy)
    groups: dict[tuple[str, float], dict[str, Any]] = {}
    evidence: list[dict[str, Any]] = []
    row_count = 0
    rows, rows_truncated = _read_bounded_rows(
        case_id,
        model_id,
        row_source,
        RowQuery(any_field_equals=tuple((field, entity_name) for field in fields)),
        budget.max_rows_per_call,
    )
    groups_truncated = False
    for row_id, payload in rows:
        if not _row_has_value(payload, fields, entity_name):
            continue
        amount = _first_amount(payload, strategy)
        if amount is None:
            continue
        row_count += 1
        dt = _first_datetime(payload, strategy)
        key = (_date_key(dt), round(amount, 2))
        if key not in groups and len(groups) >= budget.max_unique_groups:
            groups_truncated = True
            continue
        item = groups.setdefault(key, {"date": key[0], "amount": key[1], "count": 0, "counterparties": Counter()})
        item["count"] += 1
        for field in list(strategy.get("counterparty_fields") or []):
            counterparty = clean_value(payload.get(field))
            if counterparty:
                item["counterparties"][counterparty] += 1
        if len(evidence) < budget.max_evidence_records:
            evidence.append(preview_row(row_id, model_id, payload, preferred))
    top_groups = sorted(groups.values(), key=lambda item: item["count"], reverse=True)[:5]
    for item in top_groups:
        item["counterparties"] = [
            {"value": value, "count": count}
            for value, count in item["counterparties"].most_common(5)
        ]
    repeated_groups = [item for item in top_groups if int(item.get("count") or 0) >= 2]
    truncated = rows_truncated or groups_truncated
    output = {
        "summary": f"{entity_name} 在模型 {model_id} 的同日同额切片命中 {row_count} 条记录。",
        "row_count": row_count,
        "top_same_amount_groups": top_groups,
        "evidence_records": evidence,
        "evidence_refs": [],
        "status_signal": "enhanced" if repeated_groups else "unresolved",
        **_budget_fields(
            len(rows),
            truncated,
            "max_rows_per_call" if rows_truncated else "max_unique_groups" if groups_truncated else None,
        ),
    }
    result = _finalize_tool_result(case_id, "same_amount_window_probe", params, output)
    result["evidence_refs"] = [result["fact_id"]] if evidence else []
    return result


def counterparty_overlap_probe(
    case_id: int,
    factpacks: list[dict[str, Any]],
    params: dict[str, Any],
) -> dict[str, Any]:
    entity_name = str(params["entity_name"])
    wanted_models = _normalize_model_ids(params.get("model_ids"))
    wanted_model_set = set(wanted_models) if wanted_models else None
    rows = []
    counter_sets: dict[int, set[str]] = {}
    for factpack in factpacks:
        context = factpack.get("model_context", {})
        model_id = int(context.get("model_id"))
        if wanted_model_set and model_id not in wanted_model_set:
            continue
        for entity in factpack.get("focus_entities", []):
            if entity.get("entity_name") != entity_name:
                continue
            counterparties = {str(item.get("value")) for item in entity.get("counterparty_topk", []) if item.get("value")}
            counter_sets[model_id] = counterparties
            rows.append(
                {
                    "model_id": model_id,
                    "model_name": context.get("model_name"),
                    "fact_id": entity.get("fact_id"),
                    "counterparty_topk": entity.get("counterparty_topk", [])[:5],
                }
            )
    overlaps = []
    model_ids = sorted(counter_sets)
    for index, left in enumerate(model_ids):
        for right in model_ids[index + 1 :]:
            common = sorted(counter_sets[left] & counter_sets[right])
            if common:
                overlaps.append({"left_model_id": left, "right_model_id": right, "common_counterparties": common[:10]})
    base_fact_refs = [row.get("fact_id") for row in rows if row.get("fact_id")]
    output = {
        "summary": f"{entity_name} 在 {len(rows)} 个模型中存在可比较的对手方画像，发现 {len(overlaps)} 组 Top 对手交集。",
        "entity_name": entity_name,
        "model_counterparties": rows,
        "overlaps": overlaps,
        "evidence_refs": [],
        "status_signal": "enhanced" if overlaps else "unresolved",
        **_budget_fields(0, False),
    }
    result = _finalize_tool_result(case_id, "counterparty_overlap_probe", params, output, base_fact_refs)
    result["evidence_refs"] = [result["fact_id"]]
    return result


def normalize_model_ids(value: Any) -> list[int]:
    if value in [None, ""]:
        return []
    if isinstance(value, int):
        return [value]
    if isinstance(value, str):
        import re

        return [int(item) for item in re.findall(r"\d+", value)]
    if isinstance(value, (list, tuple, set)):
        result = []
        for item in value:
            try:
                result.append(int(item))
            except Exception:
                continue
        return result
    return []


_normalize_model_ids = normalize_model_ids


def execute_tool(
    tool_name: str,
    config: Any,
    case_id: int,
    strategies_by_model: dict[int, dict[str, Any]],
    dictionary: EntityDictionary,
    factpacks: list[dict[str, Any]],
    params: dict[str, Any],
    thresholds: dict[str, Any],
    row_source: RowSource | None = None,
    tool_cache: dict[str, dict[str, Any]] | None = None,
    budget: ToolExecutionBudget | None = None,
) -> dict[str, Any]:
    budget = budget or ToolExecutionBudget()
    cache_key = json.dumps(
        {"case_id": case_id, "tool_name": tool_name, "params": params, "budget": budget.__dict__},
        ensure_ascii=False,
        sort_keys=True,
        default=str,
    )
    if tool_cache is not None and cache_key in tool_cache:
        cached = copy.deepcopy(tool_cache[cache_key])
        cached["cache_hit"] = True
        return cached
    if tool_name == "counterparty_overlap_probe":
        result = counterparty_overlap_probe(case_id, factpacks, params)
    else:
        if row_source is None:
            raise RuntimeError("row_source is required for row-scanning supplement tools")
        model_id = int(params["model_id"])
        strategy = strategies_by_model[model_id]
        if tool_name == "entity_model_slice":
            result = entity_model_slice(
                config, case_id, strategy, dictionary, params, thresholds, row_source=row_source, budget=budget
            )
        elif tool_name == "relation_neighbor_subgraph":
            result = relation_neighbor_subgraph(config, case_id, strategy, params, row_source=row_source, budget=budget)
        elif tool_name == "fund_flow_path_probe":
            result = fund_flow_path_probe(config, case_id, strategy, params, row_source=row_source, budget=budget)
        elif tool_name == "same_amount_window_probe":
            result = same_amount_window_probe(config, case_id, strategy, params, row_source=row_source, budget=budget)
        elif tool_name == "transfer_structure_analyzer":
            result = transfer_structure_analyzer(config, case_id, strategy, params, row_source=row_source, budget=budget)
        else:
            raise ValueError(f"unsupported tool: {tool_name}")
    result["cache_hit"] = False
    if tool_cache is not None:
        tool_cache[cache_key] = copy.deepcopy(result)
    return result


def _top_edges(edges: dict[tuple[str, str], dict[str, Any]]) -> list[dict[str, Any]]:
    return _rank_edges(sorted(edges.values(), key=lambda item: (item["amount_sum"], item["count"]), reverse=True)[:5])


def _rank_edges(edges: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for index, edge in enumerate(edges, start=1):
        result.append(
            {
                "rank": index,
                "source": edge["source"],
                "target": edge["target"],
                "count": edge["count"],
                "amount_sum": round(edge["amount_sum"], 2),
            }
        )
    return result


def _collect_evidence_refs(profiles: dict[str, Any]) -> list[str]:
    refs = []
    for entity in profiles.get("focus_entities", []):
        refs.extend(entity.get("evidence_refs", []))
    for relation in profiles.get("relation_profiles", []):
        refs.extend(relation.get("evidence_refs", []))
    return refs[:10]

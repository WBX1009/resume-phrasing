from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
from typing import Any, Callable

from .field_profile import is_empty, parse_datetime, parse_json_data, parse_number


@dataclass(frozen=True)
class OperatorSpec:
    name: str
    description: str
    run: Callable[[int, list[dict[str, Any]], dict[str, Any], dict[str, Any]], dict[str, Any]]


def fact_id(model_id: int, operator: str, payload: Any) -> str:
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:10]
    return f"m{model_id}_{operator}_{digest}"


def _field_names(profile: dict[str, Any]) -> list[str]:
    return [field["field_name"] for field in profile.get("fields", [])]


def _fields_with_role(profile: dict[str, Any], role: str) -> list[str]:
    return [field["field_name"] for field in profile.get("fields", []) if role in field.get("roles", [])]


def _first_existing(profile: dict[str, Any], preferred: list[str], fallback_roles: list[str]) -> str | None:
    names = _field_names(profile)
    for preferred_name in preferred:
        if preferred_name in names:
            return preferred_name
    for role in fallback_roles:
        fields = _fields_with_role(profile, role)
        if fields:
            return fields[0]
    return None


def _amount_field(profile: dict[str, Any]) -> str | None:
    return _first_existing(
        profile,
        [
            "交易金额",
            "金额",
            "交易金额(元)",
            "支出金额",
            "退回金额",
            "收入金额",
            "流入金额",
            "流出金额",
            "A转B总额",
            "B转A总额",
            "总流水",
            "净流水",
            "累计交易金额",
            "最大交易金额",
        ],
        ["amount", "balance"],
    )


def _count_field(profile: dict[str, Any]) -> str | None:
    return _first_existing(profile, ["次数", "交易次数", "通联总次数", "关联交易笔数", "消费次数"], ["count"])


def _duration_field(profile: dict[str, Any]) -> str | None:
    fields = [field for field in profile.get("fields", []) if "duration" in field.get("roles", [])]
    numeric_fields = [field["field_name"] for field in fields if field.get("parsed_type") == "number"]
    for preferred in ["交易通联时间间隔(秒)", "时间差_秒", "通话时长", "时长"]:
        if preferred in numeric_fields:
            return preferred
    return numeric_fields[0] if numeric_fields else None


def _time_field(profile: dict[str, Any]) -> str | None:
    return _first_existing(
        profile,
        ["交易时间", "交易日期", "最近通联时间", "首次交易时间", "末次交易时间", "通话时间", "开始时间"],
        ["time", "date"],
    )


def _entity_field(profile: dict[str, Any]) -> str | None:
    return _first_existing(
        profile,
        ["姓名", "对象", "交易对手", "对方姓名", "联系人", "共同联系人号码", "银行卡户主姓名", "对方单位"],
        ["entity", "account", "phone", "org"],
    )


def _category_field(profile: dict[str, Any]) -> str | None:
    return _first_existing(
        profile,
        ["地点名称", "交易地域", "交易类型", "交易方向", "交易渠道", "接听状态", "对方单位", "对方职位"],
        ["location", "org", "entity", "account", "phone", "text"],
    )


def _relation_pair(profile: dict[str, Any]) -> tuple[str | None, str | None]:
    names = _field_names(profile)
    preferred_pairs = [
        ("流出对象（起点）", "流入对象（终点）"),
        ("交易发起方（起点）", "交易接收方（终点）"),
        ("交易发起方", "交易接收方"),
        ("对象A", "对象B"),
        ("主体A", "主体B"),
        ("人员A", "人员B"),
        ("账户1", "账户2"),
        ("银行卡户主姓名", "交易对手"),
        ("交易对手", "关联手机号码"),
        ("手机号", "姓名"),
        ("主叫号码", "被叫号码"),
        ("本端号码", "对端号码"),
        ("分析对象手机号码", "对方手机号码"),
    ]
    for left, right in preferred_pairs:
        if left in names and right in names:
            return left, right
    sources = _fields_with_role(profile, "relation_source")
    targets = _fields_with_role(profile, "relation_target")
    if sources and targets:
        return sources[0], targets[0]
    endpoints = _fields_with_role(profile, "entity") + _fields_with_role(profile, "account") + _fields_with_role(profile, "phone")
    endpoints = list(dict.fromkeys(endpoints))
    if len(endpoints) >= 2:
        return endpoints[0], endpoints[1]
    return None, None


def evidence_ref(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "row_id": row.get("_source_row_id"),
        "row_index": row.get("_source_index"),
    }


def row_preview(row: dict[str, Any], limit: int = 8) -> dict[str, Any]:
    preview: dict[str, Any] = {}
    for key, value in row.items():
        if key.startswith("_"):
            continue
        if len(preview) >= limit:
            break
        preview[key] = value
    preview.update(evidence_ref(row))
    return preview


def success(model_id: int, operator: str, metrics: dict[str, Any], evidence: list[dict[str, Any]]) -> dict[str, Any]:
    fid = fact_id(model_id, operator, {"metrics": metrics, "evidence": evidence[:3]})
    return {
        "operator": operator,
        "status": "success",
        "fact_id": fid,
        "metrics": metrics,
        "evidence_refs": [evidence_ref(row) for row in evidence[:10]],
        "evidence": [row_preview(row) for row in evidence[:5]],
    }


def skipped(operator: str, reason: str) -> dict[str, Any]:
    return {"operator": operator, "status": "skipped", "reason": reason}


def amount_topk(model_id: int, rows: list[dict[str, Any]], profile: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    field = _amount_field(profile)
    if not field:
        return skipped("amount_topk", "no amount-like field detected")
    items = []
    for row in rows:
        value = parse_number(row.get(field))
        if value is not None:
            items.append((value, row))
    if not items:
        return skipped("amount_topk", f"field {field} has no parseable numbers")
    items.sort(key=lambda item: item[0], reverse=True)
    metrics = {
        "field": field,
        "count": len(items),
        "topk": [{"rank": i + 1, "value": value, **evidence_ref(row)} for i, (value, row) in enumerate(items[:10])],
    }
    return success(model_id, "amount_topk", metrics, [row for _, row in items[:10]])


def amount_stats(model_id: int, rows: list[dict[str, Any]], profile: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    field = _amount_field(profile)
    if not field:
        return skipped("amount_stats", "no amount-like field detected")
    values = [parse_number(row.get(field)) for row in rows]
    values = [value for value in values if value is not None]
    if not values:
        return skipped("amount_stats", f"field {field} has no parseable numbers")
    metrics = {
        "field": field,
        "count": len(values),
        "sum": round(sum(values), 2),
        "min": min(values),
        "max": max(values),
        "avg": round(sum(values) / len(values), 2),
    }
    evidence = sorted(rows, key=lambda row: parse_number(row.get(field)) or float("-inf"), reverse=True)[:5]
    return success(model_id, "amount_stats", metrics, evidence)


def category_topk(model_id: int, rows: list[dict[str, Any]], profile: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    field = _category_field(profile)
    if not field:
        return skipped("category_topk", "no category-like field detected")
    counts = Counter(str(row.get(field)) for row in rows if not is_empty(row.get(field)))
    if not counts:
        return skipped("category_topk", f"field {field} has no values")
    metrics = {"field": field, "topk": [{"value": value, "count": count} for value, count in counts.most_common(10)]}
    lookup = {str(row.get(field)): row for row in rows if not is_empty(row.get(field))}
    evidence = [lookup[value] for value, _ in counts.most_common(5) if value in lookup]
    return success(model_id, "category_topk", metrics, evidence)


def entity_frequency(model_id: int, rows: list[dict[str, Any]], profile: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    field = _entity_field(profile)
    if not field:
        return skipped("entity_frequency", "no entity/account/phone field detected")
    counts = Counter(str(row.get(field)) for row in rows if not is_empty(row.get(field)))
    if not counts:
        return skipped("entity_frequency", f"field {field} has no values")
    metrics = {"field": field, "topk": [{"value": value, "count": count} for value, count in counts.most_common(10)]}
    lookup = {str(row.get(field)): row for row in rows if not is_empty(row.get(field))}
    evidence = [lookup[value] for value, _ in counts.most_common(5) if value in lookup]
    return success(model_id, "entity_frequency", metrics, evidence)


def time_range(model_id: int, rows: list[dict[str, Any]], profile: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    field = _time_field(profile)
    if not field:
        return skipped("time_range", "no time/date field detected")
    values: list[tuple[datetime, dict[str, Any]]] = []
    for row in rows:
        value = parse_datetime(row.get(field))
        if value:
            values.append((value, row))
    if not values:
        return skipped("time_range", f"field {field} has no parseable datetime values")
    values.sort(key=lambda item: item[0])
    metrics = {
        "field": field,
        "count": len(values),
        "min": values[0][0].isoformat(sep=" "),
        "max": values[-1][0].isoformat(sep=" "),
    }
    return success(model_id, "time_range", metrics, [values[0][1], values[-1][1]])


def time_bucket(model_id: int, rows: list[dict[str, Any]], profile: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    field = _time_field(profile)
    if not field:
        return skipped("time_bucket", "no time/date field detected")
    buckets: Counter[str] = Counter()
    evidence_by_bucket: dict[str, dict[str, Any]] = {}
    for row in rows:
        value = parse_datetime(row.get(field))
        if value:
            key = value.strftime("%Y-%m")
            buckets[key] += 1
            evidence_by_bucket.setdefault(key, row)
    if not buckets:
        return skipped("time_bucket", f"field {field} has no parseable datetime values")
    metrics = {"field": field, "buckets": [{"bucket": key, "count": buckets[key]} for key in sorted(buckets)]}
    top_keys = [key for key, _ in buckets.most_common(5)]
    return success(model_id, "time_bucket", metrics, [evidence_by_bucket[key] for key in top_keys])


def relation_edges(model_id: int, rows: list[dict[str, Any]], profile: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    left, right = _relation_pair(profile)
    if not left or not right:
        return skipped("relation_edges", "no relation source/target fields detected")
    amount_field = _amount_field(profile)
    edges: dict[tuple[str, str], dict[str, Any]] = defaultdict(lambda: {"count": 0, "amount_sum": 0.0, "evidence": []})
    for row in rows:
        source = row.get(left)
        target = row.get(right)
        if is_empty(source) or is_empty(target):
            continue
        key = (str(source), str(target))
        edges[key]["count"] += 1
        if amount_field:
            edges[key]["amount_sum"] += parse_number(row.get(amount_field)) or 0.0
        if len(edges[key]["evidence"]) < 5:
            edges[key]["evidence"].append(row)
    if not edges:
        return skipped("relation_edges", f"fields {left}/{right} produced no edges")
    ranked = sorted(edges.items(), key=lambda item: (item[1]["amount_sum"], item[1]["count"]), reverse=True)
    metrics = {
        "source_field": left,
        "target_field": right,
        "amount_field": amount_field,
        "top_edges": [
            {
                "source": source,
                "target": target,
                "count": data["count"],
                "amount_sum": round(data["amount_sum"], 2),
            }
            for (source, target), data in ranked[:10]
        ],
    }
    evidence = []
    for _, data in ranked[:5]:
        evidence.extend(data["evidence"][:1])
    return success(model_id, "relation_edges", metrics, evidence)


def flow_summary(model_id: int, rows: list[dict[str, Any]], profile: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    amount_field = _amount_field(profile)
    direction_field = _first_existing(profile, ["交易方向", "方向", "收支标志"], ["text"])
    if not amount_field:
        return skipped("flow_summary", "no amount-like field detected")
    by_direction: dict[str, dict[str, float]] = defaultdict(lambda: {"count": 0, "amount_sum": 0.0})
    evidence = []
    for row in rows:
        amount = parse_number(row.get(amount_field))
        if amount is None:
            continue
        direction = str(row.get(direction_field) or "未区分方向")
        by_direction[direction]["count"] += 1
        by_direction[direction]["amount_sum"] += amount
        if len(evidence) < 5:
            evidence.append(row)
    if not by_direction:
        return skipped("flow_summary", "no parseable amount values for flow summary")
    metrics = {
        "amount_field": amount_field,
        "direction_field": direction_field,
        "by_direction": [
            {"direction": key, "count": int(value["count"]), "amount_sum": round(value["amount_sum"], 2)}
            for key, value in by_direction.items()
        ],
    }
    return success(model_id, "flow_summary", metrics, evidence)


def duration_stats(model_id: int, rows: list[dict[str, Any]], profile: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    field = _duration_field(profile)
    if not field:
        return skipped("duration_stats", "no duration/interval field detected")
    values = [parse_number(row.get(field)) for row in rows]
    values = [value for value in values if value is not None]
    if not values:
        return skipped("duration_stats", f"field {field} has no parseable numbers")
    metrics = {
        "field": field,
        "count": len(values),
        "min": min(values),
        "max": max(values),
        "avg": round(sum(values) / len(values), 2),
    }
    evidence = sorted(rows, key=lambda row: parse_number(row.get(field)) or float("-inf"), reverse=True)[:5]
    return success(model_id, "duration_stats", metrics, evidence)


def evidence_samples(model_id: int, rows: list[dict[str, Any]], profile: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    if not rows:
        return skipped("evidence_samples", "no rows to sample")
    metrics = {"sample_count": min(5, len(rows)), "row_count": len(rows)}
    return success(model_id, "evidence_samples", metrics, rows[:5])


OPERATORS: dict[str, OperatorSpec] = {
    "amount_topk": OperatorSpec("amount_topk", "Top rows by amount-like field", amount_topk),
    "amount_stats": OperatorSpec("amount_stats", "Summary stats for amount-like field", amount_stats),
    "category_topk": OperatorSpec("category_topk", "Top categories", category_topk),
    "entity_frequency": OperatorSpec("entity_frequency", "Top entities/accounts/phones", entity_frequency),
    "time_range": OperatorSpec("time_range", "Minimum and maximum timestamp", time_range),
    "time_bucket": OperatorSpec("time_bucket", "Monthly buckets", time_bucket),
    "relation_edges": OperatorSpec("relation_edges", "Source-target relation edges", relation_edges),
    "flow_summary": OperatorSpec("flow_summary", "Amount summary by direction", flow_summary),
    "duration_stats": OperatorSpec("duration_stats", "Duration/interval stats", duration_stats),
    "evidence_samples": OperatorSpec("evidence_samples", "Representative rows", evidence_samples),
}


def planned_operators(context: dict[str, Any], profile: dict[str, Any]) -> list[str]:
    category = context.get("model_category") or ""
    name = context.get("model_name") or ""
    roles = {role for field in profile.get("fields", []) for role in field.get("roles", [])}
    ops: list[str] = ["evidence_samples"]
    if roles & {"amount", "balance"}:
        ops.extend(["amount_stats", "amount_topk"])
    if roles & {"time", "date"}:
        ops.extend(["time_range", "time_bucket"])
    if roles & {"entity", "account", "phone", "org"}:
        ops.append("entity_frequency")
    if roles & {"location", "text"}:
        ops.append("category_topk")
    if (
        "转账" in category
        or "资金" in category
        or "通联" in category
        or "关系" in category
        or "流向" in name
        or "关系" in name
        or len(_fields_with_role(profile, "relation_source") + _fields_with_role(profile, "relation_target")) >= 2
    ):
        ops.append("relation_edges")
    if "资金" in category or "交易" in category or "现金" in category or "转账" in category:
        ops.append("flow_summary")
    if _duration_field(profile):
        ops.append("duration_stats")
    return list(dict.fromkeys(ops))


def run_case_operators(
    model_id: int,
    records: list[dict[str, Any]],
    profile: dict[str, Any],
    context: dict[str, Any],
) -> dict[str, Any]:
    rows, parse_errors = parse_json_data(records)
    results = []
    for operator_name in planned_operators(context, profile):
        spec = OPERATORS[operator_name]
        try:
            results.append(spec.run(model_id, rows, profile, context))
        except Exception as exc:
            results.append({"operator": operator_name, "status": "error", "reason": type(exc).__name__})
    return {"json_parse_errors": parse_errors, "operator_results": results}

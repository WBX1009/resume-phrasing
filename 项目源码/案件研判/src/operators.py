from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime
import json
from typing import Any

from .field_profile import parse_datetime, parse_json_data, parse_number


OPERATOR_RULES = {
    5: ["numeric_topk", "entity_frequency", "evidence_samples"],
    29: ["category_topk", "entity_frequency", "evidence_samples"],
    43: ["numeric_topk", "time_bucket", "evidence_samples"],
    48: ["relation_edges", "numeric_topk", "evidence_samples"],
    71: ["entity_frequency", "relation_edges", "evidence_samples"],
}


def _fields_with_role(profile: dict[str, Any], role: str) -> list[str]:
    return [field["field_name"] for field in profile.get("fields", []) if role in field.get("roles", [])]


def _fields_with_type(profile: dict[str, Any], parsed_type: str) -> list[str]:
    return [field["field_name"] for field in profile.get("fields", []) if field.get("parsed_type") == parsed_type]


def _best_amount_field(profile: dict[str, Any]) -> str | None:
    candidates = _fields_with_role(profile, "amount") or _fields_with_type(profile, "number")
    if not candidates:
        return None
    priority = ["交易金额", "金额", "总额", "净额", "余额"]
    return sorted(candidates, key=lambda name: next((i for i, key in enumerate(priority) if key in name), 99))[0]


def _best_time_field(profile: dict[str, Any]) -> str | None:
    return (_fields_with_role(profile, "time") or [None])[0]


def _best_entity_field(profile: dict[str, Any]) -> str | None:
    return (_fields_with_role(profile, "entity") or _fields_with_role(profile, "account") or [None])[0]


def _category_field(profile: dict[str, Any]) -> str | None:
    for role in ["location", "entity", "account", "text"]:
        fields = _fields_with_role(profile, role)
        if fields:
            return fields[0]
    return None


def _relation_fields(profile: dict[str, Any]) -> tuple[str | None, str | None]:
    field_names = [field["field_name"] for field in profile.get("fields", [])]
    pairs = [
        ("对象A", "对象B"),
        ("主体A", "主体B"),
        ("人员A", "人员B"),
        ("账户1", "账户2"),
        ("银行卡户主姓名", "交易对手"),
        ("银行卡户主姓名", "关联手机号码"),
        ("交易对手", "关联手机号码"),
        ("交易发起方（起点）", "交易接收方（终点）"),
        ("交易发起方", "交易接收方"),
        ("付款方", "收款方"),
    ]
    for left, right in pairs:
        if left in field_names and right in field_names:
            return left, right
    fields = _fields_with_role(profile, "relation_endpoint") + _fields_with_role(profile, "entity")
    fields = list(dict.fromkeys(fields))
    if len(fields) >= 2:
        return fields[0], fields[1]
    return None, None


def _success(operator: str, result: dict[str, Any]) -> dict[str, Any]:
    return {"operator": operator, "status": "success", **result}


def _skipped(operator: str, reason: str) -> dict[str, Any]:
    return {"operator": operator, "status": "skipped", "reason": reason}


def numeric_topk(rows: list[dict[str, Any]], profile: dict[str, Any], limit: int = 10) -> dict[str, Any]:
    field = _best_amount_field(profile)
    if not field:
        return _skipped("numeric_topk", "未识别到可用于数值 TopK 的金额或数值字段")
    values = []
    for row in rows:
        number = parse_number(row.get(field))
        if number is not None:
            values.append({"value": number, "row": row})
    if not values:
        return _skipped("numeric_topk", f"字段 {field} 没有可解析数值")
    values.sort(key=lambda item: item["value"], reverse=True)
    top_items = [
        {
            "rank": index + 1,
            "field": field,
            "value": item["value"],
            "evidence": _row_summary(item["row"]),
        }
        for index, item in enumerate(values[:limit])
    ]
    return _success(
        "numeric_topk",
        {
            "field": field,
            "count": len(values),
            "max": values[0]["value"],
            "min": values[-1]["value"],
            "topk": top_items,
        },
    )


def category_topk(rows: list[dict[str, Any]], profile: dict[str, Any], limit: int = 10) -> dict[str, Any]:
    field = _category_field(profile)
    if not field:
        return _skipped("category_topk", "未识别到可用于类别 TopK 的文本字段")
    counts = Counter(str(row.get(field)) for row in rows if row.get(field) not in [None, ""])
    if not counts:
        return _skipped("category_topk", f"字段 {field} 没有可统计取值")
    return _success(
        "category_topk",
        {
            "field": field,
            "topk": [{"value": value, "count": count} for value, count in counts.most_common(limit)],
        },
    )


def entity_frequency(rows: list[dict[str, Any]], profile: dict[str, Any], limit: int = 10) -> dict[str, Any]:
    field = _best_entity_field(profile)
    if not field:
        return _skipped("entity_frequency", "未识别到主体、姓名、账号或号码字段")
    counts = Counter(str(row.get(field)) for row in rows if row.get(field) not in [None, ""])
    if not counts:
        return _skipped("entity_frequency", f"字段 {field} 没有可统计取值")
    return _success(
        "entity_frequency",
        {
            "field": field,
            "topk": [{"value": value, "count": count} for value, count in counts.most_common(limit)],
        },
    )


def time_bucket(rows: list[dict[str, Any]], profile: dict[str, Any]) -> dict[str, Any]:
    field = _best_time_field(profile)
    if not field:
        return _skipped("time_bucket", "未识别到时间字段")
    buckets: Counter[str] = Counter()
    parsed: list[datetime] = []
    for row in rows:
        dt = parse_datetime(row.get(field))
        if dt:
            parsed.append(dt)
            buckets[dt.strftime("%Y-%m")] += 1
    if not parsed:
        return _skipped("time_bucket", f"字段 {field} 没有可解析时间")
    return _success(
        "time_bucket",
        {
            "field": field,
            "time_range": {
                "min": min(parsed).isoformat(sep=" "),
                "max": max(parsed).isoformat(sep=" "),
            },
            "buckets": [{"bucket": key, "count": buckets[key]} for key in sorted(buckets)],
        },
    )


def relation_edges(rows: list[dict[str, Any]], profile: dict[str, Any], limit: int = 10) -> dict[str, Any]:
    left, right = _relation_fields(profile)
    if not left or not right:
        return _skipped("relation_edges", "未识别到可用于关系分析的起点字段和终点字段")
    amount_field = _best_amount_field(profile)
    edges: dict[tuple[str, str], dict[str, Any]] = defaultdict(lambda: {"count": 0, "amount_sum": 0.0, "evidence": []})
    for row in rows:
        left_value = row.get(left)
        right_value = row.get(right)
        if left_value in [None, ""] or right_value in [None, ""]:
            continue
        key = (str(left_value), str(right_value))
        edges[key]["count"] += 1
        if amount_field:
            edges[key]["amount_sum"] += parse_number(row.get(amount_field)) or 0.0
        if len(edges[key]["evidence"]) < 3:
            edges[key]["evidence"].append(_row_summary(row))
    if not edges:
        return _skipped("relation_edges", f"字段 {left} 与 {right} 没有可聚合关系")
    items = sorted(edges.items(), key=lambda item: (item[1]["amount_sum"], item[1]["count"]), reverse=True)
    return _success(
        "relation_edges",
        {
            "left_field": left,
            "right_field": right,
            "amount_field": amount_field,
            "top_edges": [
                {
                    "source": left_value,
                    "target": right_value,
                    "count": data["count"],
                    "amount_sum": round(data["amount_sum"], 2),
                    "evidence": data["evidence"],
                }
                for (left_value, right_value), data in items[:limit]
            ],
        },
    )


def evidence_samples(rows: list[dict[str, Any]], limit: int = 5) -> dict[str, Any]:
    if not rows:
        return _skipped("evidence_samples", "没有可抽取的结果记录")
    return _success(
        "evidence_samples",
        {"samples": [_row_summary(row) for row in rows[:limit]]},
    )


def _row_summary(row: dict[str, Any], limit: int = 8) -> dict[str, Any]:
    summary = {}
    for key, value in row.items():
        if key.startswith("_"):
            continue
        if len(summary) >= limit:
            break
        try:
            json.dumps(value, ensure_ascii=False)
            summary[key] = value
        except TypeError:
            summary[key] = str(value)
    if "_source_row_id" in row:
        summary["_source_row_id"] = row["_source_row_id"]
    if "_source_index" in row:
        summary["_source_index"] = row["_source_index"]
    return summary


def run_model_operators(
    model_id: int,
    records: list[dict[str, Any]],
    profile: dict[str, Any],
) -> dict[str, Any]:
    rows, parse_errors = parse_json_data(records)
    operator_names = OPERATOR_RULES.get(model_id, ["numeric_topk", "category_topk", "evidence_samples"])
    results = []
    for operator_name in operator_names:
        if operator_name == "numeric_topk":
            results.append(numeric_topk(rows, profile))
        elif operator_name == "category_topk":
            results.append(category_topk(rows, profile))
        elif operator_name == "entity_frequency":
            results.append(entity_frequency(rows, profile))
        elif operator_name == "time_bucket":
            results.append(time_bucket(rows, profile))
        elif operator_name == "relation_edges":
            results.append(relation_edges(rows, profile))
        elif operator_name == "evidence_samples":
            results.append(evidence_samples(rows))
        else:
            results.append(_skipped(operator_name, "未知算子"))
    return {"json_parse_errors": parse_errors, "operator_results": results}

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime
import json
import math
import re
from typing import Any

from dateutil import parser as date_parser


EMPTY_VALUES = {"", "null", "none", "nan", "未知", "未知时间", "-"}


def parse_json_data(records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    parsed: list[dict[str, Any]] = []
    errors = 0
    for index, row in enumerate(records):
        raw = row.get("json_data")
        try:
            payload = json.loads(raw) if isinstance(raw, str) else raw
            if not isinstance(payload, dict):
                raise ValueError("json_data is not an object")
            payload = dict(payload)
            payload["_source_row_id"] = row.get("id")
            payload["_source_index"] = index
            parsed.append(payload)
        except Exception:
            errors += 1
    return parsed, errors


def is_empty(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    return isinstance(value, str) and value.strip().lower() in EMPTY_VALUES


def parse_number(value: Any) -> float | None:
    if isinstance(value, bool) or is_empty(value):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        text = value.strip().replace(",", "")
        text = re.sub(r"[￥¥元%]", "", text)
        if re.fullmatch(r"[-+]?\d+(\.\d+)?", text):
            return float(text)
    return None


def parse_datetime(value: Any) -> datetime | None:
    if not isinstance(value, str) or is_empty(value):
        return None
    text = value.strip()
    if not re.search(r"\d", text):
        return None
    try:
        return date_parser.parse(text, fuzzy=False)
    except Exception:
        return None


def infer_role(field_name: str, parsed_type: str, top_values: list[Any]) -> list[str]:
    name = field_name.lower()
    roles: list[str] = []
    is_note_like = any(key in field_name for key in ["摘要", "备注", "原因", "方向", "类型", "状态", "标记"])
    is_duration_like = any(key in field_name for key in ["时间差", "间隔", "时长", "耗时"]) or (
        parsed_type == "number" and any(key in field_name for key in ["秒", "分钟", "小时"])
    )
    if any(key in field_name for key in ["金额", "余额", "总额", "净额", "保费"]) or any(
        key in name for key in ["amount", "balance", "money"]
    ):
        if not is_note_like:
            roles.append("amount")
    if any(key in field_name for key in ["占比", "比例", "百分比"]) or "%" in field_name:
        roles.append("percent")
    if any(key in field_name for key in ["次数", "笔数", "频次", "数量", "个数"]):
        roles.append("count")
    if is_duration_like:
        roles.append("duration")
    if not is_duration_like and (parsed_type in {"datetime", "date", "time"} or any(key in field_name for key in ["时间", "日期"])):
        roles.append("time")
    if any(key in field_name for key in ["姓名", "主体", "人员", "对象", "对方", "对手", "联系人", "客户", "单位"]):
        roles.append("entity")
    if any(key in field_name for key in ["卡号", "账号", "账户", "银行卡", "手机号", "电话", "号码"]):
        roles.append("account")
    if any(key in field_name for key in ["地点", "地址", "地区", "地域", "位置", "网点", "支行", "POI"]):
        roles.append("location")
    if any(key in field_name for key in ["单位", "机构", "公司", "组织"]):
        roles.append("org")
    if any(key in field_name for key in ["手机号", "电话", "号码"]):
        roles.append("phone")
    if any(key in field_name for key in ["转出", "转入", "起点", "终点", "发起", "接收", "流向", "关系"]):
        roles.append("relation_endpoint")
    if not roles and parsed_type == "text":
        roles.append("text")
    return roles


def _parsed_type(non_empty_values: list[Any]) -> str:
    if not non_empty_values:
        return "empty"
    number_count = sum(parse_number(value) is not None for value in non_empty_values)
    dt_values = [parse_datetime(value) for value in non_empty_values]
    datetime_count = sum(value is not None for value in dt_values)
    total = len(non_empty_values)
    if number_count / total >= 0.8:
        return "number"
    if datetime_count / total >= 0.8:
        has_time = any(value and (value.hour or value.minute or value.second) for value in dt_values)
        return "datetime" if has_time else "date"
    return "text"


def build_field_profiles(records_by_model: dict[int, list[dict[str, Any]]]) -> dict[int, dict[str, Any]]:
    result: dict[int, dict[str, Any]] = {}
    for model_id, rows in records_by_model.items():
        parsed_rows, parse_errors = parse_json_data(rows)
        total = len(parsed_rows)
        field_values: dict[str, list[Any]] = defaultdict(list)
        raw_types: dict[str, Counter[str]] = defaultdict(Counter)

        for row in parsed_rows:
            for key, value in row.items():
                if key.startswith("_"):
                    continue
                field_values[key].append(value)
                raw_types[key][type(value).__name__] += 1

        fields: list[dict[str, Any]] = []
        for field_name, values in field_values.items():
            non_empty = [value for value in values if not is_empty(value)]
            parsed_type = _parsed_type(non_empty)
            top_counts = Counter(str(value) for value in non_empty).most_common(10)
            top_values = [{"value": value, "count": count} for value, count in top_counts]

            numbers = [parse_number(value) for value in non_empty]
            numbers = [value for value in numbers if value is not None]
            datetimes = [parse_datetime(value) for value in non_empty]
            datetimes = [value for value in datetimes if value is not None]

            field_profile = {
                "field_name": field_name,
                "appear_count": len(values),
                "coverage_rate": round(len(values) / total, 4) if total else 0,
                "empty_count": len(values) - len(non_empty),
                "empty_rate": round((len(values) - len(non_empty)) / len(values), 4) if values else 0,
                "raw_types": dict(raw_types[field_name]),
                "parsed_type": parsed_type,
                "topk": top_values,
                "roles": infer_role(field_name, parsed_type, [item["value"] for item in top_values]),
            }
            if numbers:
                field_profile["numeric_range"] = {
                    "min": min(numbers),
                    "max": max(numbers),
                    "avg": round(sum(numbers) / len(numbers), 4),
                }
            if datetimes:
                field_profile["time_range"] = {
                    "min": min(datetimes).isoformat(sep=" "),
                    "max": max(datetimes).isoformat(sep=" "),
                }
            fields.append(field_profile)

        result[model_id] = {
            "model_id": model_id,
            "row_count": len(rows),
            "parsed_row_count": total,
            "json_parse_errors": parse_errors,
            "fields": sorted(fields, key=lambda item: (-item["coverage_rate"], item["field_name"])),
        }
    return result

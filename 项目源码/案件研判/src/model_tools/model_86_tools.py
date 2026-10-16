from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime
from typing import Any

from ..entity_profile import preview_row, safe_float
from ..entity_resolver import clean_value
from ..evidence_chain import is_incomplete_tool_output
from ..field_profile import parse_datetime
from ..row_source import RowQuery, RowSource
from ..stable_ids import stable_id


MODEL_ID = 86


def transfer_structure_analyzer(
    config: Any,
    case_id: int,
    strategy: dict[str, Any],
    params: dict[str, Any],
    row_source: RowSource | None = None,
    budget: Any | None = None,
) -> dict[str, Any]:
    """Analyze transfer-network structure for model 86 around one entity."""

    if row_source is None:
        raise RuntimeError("row_source is required for transfer_structure_analyzer")
    if budget is None:
        raise RuntimeError("budget is required for transfer_structure_analyzer")
    model_id = int(params.get("model_id") or MODEL_ID)
    if model_id != MODEL_ID:
        raise ValueError("transfer_structure_analyzer only supports model_id=86")
    entity_name = str(params.get("entity_name") or "").strip()
    if not entity_name:
        raise ValueError("entity_name is required")

    relation = strategy.get("relation") or {}
    source_field = relation.get("source")
    target_field = relation.get("target")
    amount_field = relation.get("value") or strategy.get("primary_amount_field")
    account_fields = list(strategy.get("account_fields") or [])
    source_account_field = account_fields[0] if account_fields else None
    target_account_field = account_fields[1] if len(account_fields) > 1 else None
    time_fields = list(strategy.get("time_fields") or [])
    preferred = [
        source_field,
        target_field,
        source_account_field,
        target_account_field,
        amount_field,
        *time_fields,
    ]
    preferred = [field for field in dict.fromkeys(preferred) if field]

    rows = list(
        row_source.iter_model_rows_query(
            case_id,
            model_id,
            RowQuery(
                any_field_equals=((source_field, entity_name), (target_field, entity_name)),
                limit=budget.max_rows_per_call + 1,
            ),
        )
    )
    rows_truncated = len(rows) > budget.max_rows_per_call
    rows = rows[: budget.max_rows_per_call]

    total_rows = 0
    total_amount = 0.0
    self_transfer_count = 0
    self_transfer_amount = 0.0
    incoming: dict[str, dict[str, Any]] = {}
    outgoing: dict[str, dict[str, Any]] = {}
    counterparty_amounts: Counter[str] = Counter()
    counterparty_counts: Counter[str] = Counter()
    hour_buckets: Counter[int] = Counter()
    day_buckets: Counter[str] = Counter()
    weekend_count = 0
    night_count = 0
    time_values: list[datetime] = []
    accounts: set[str] = set()
    account_pairs: dict[tuple[str, str], dict[str, Any]] = {}
    evidence_records: list[dict[str, Any]] = []
    groups_truncated = False

    for row_id, payload in rows:
        source = clean_value(payload.get(source_field))
        target = clean_value(payload.get(target_field))
        if not source or not target or entity_name not in {source, target}:
            continue
        total_rows += 1
        amount = safe_float(payload.get(amount_field)) or 0.0
        total_amount += amount

        counterparty = target if source == entity_name else source
        counterparty_counts[counterparty] += 1
        counterparty_amounts[counterparty] += amount
        bucket = outgoing if source == entity_name else incoming
        if counterparty not in bucket and len(incoming) + len(outgoing) >= budget.max_unique_groups:
            groups_truncated = True
        else:
            item = bucket.setdefault(counterparty, {"name": counterparty, "count": 0, "amount": 0.0})
            item["count"] += 1
            item["amount"] += amount

        if source == target == entity_name:
            self_transfer_count += 1
            self_transfer_amount += amount

        source_account = clean_value(payload.get(source_account_field))
        target_account = clean_value(payload.get(target_account_field))
        if source_account:
            accounts.add(source_account)
        if target_account:
            accounts.add(target_account)
        if source_account or target_account:
            key = (source_account or "", target_account or "")
            pair = account_pairs.setdefault(
                key,
                {"source_account": _mask_account(source_account), "target_account": _mask_account(target_account), "count": 0, "amount": 0.0},
            )
            pair["count"] += 1
            pair["amount"] += amount

        dt = _first_datetime(payload, time_fields)
        if dt:
            time_values.append(dt)
            hour_buckets[dt.hour] += 1
            day_buckets[dt.strftime("%Y-%m-%d")] += 1
            if dt.hour >= 22 or dt.hour < 6:
                night_count += 1
            if dt.weekday() >= 5:
                weekend_count += 1

        if len(evidence_records) < budget.max_evidence_records:
            evidence_records.append(preview_row(row_id, model_id, payload, preferred))

    top_outgoing = _top_counterparties(outgoing, total_amount)
    top_incoming = _top_counterparties(incoming, total_amount)
    top_account_pairs = sorted(account_pairs.values(), key=lambda item: (item["amount"], item["count"]), reverse=True)[:5]
    for item in top_account_pairs:
        item["amount"] = round(float(item.get("amount") or 0), 2)

    structured = {
        "entity_name": entity_name,
        "model_id": model_id,
        "total_rows": total_rows,
        "total_amount": round(total_amount, 2),
        "self_transfer_count": self_transfer_count,
        "self_transfer_amount": round(self_transfer_amount, 2),
        "self_transfer_ratio": round(self_transfer_count / total_rows, 4) if total_rows else 0,
        "self_transfer_amount_ratio": round(self_transfer_amount / total_amount, 4) if total_amount else 0,
        "distinct_counterparty_count": len(counterparty_counts),
        "counterparty_hhi": _hhi(counterparty_amounts),
        "top_outgoing": top_outgoing,
        "top_incoming": top_incoming,
        "night_ratio": round(night_count / total_rows, 4) if total_rows else 0,
        "hour_concentration": _top_n_ratio(hour_buckets, total_rows, 3),
        "day_concentration": _top_n_ratio(day_buckets, total_rows, 1),
        "weekend_ratio": round(weekend_count / total_rows, 4) if total_rows else 0,
        "time_span_days": (max(time_values) - min(time_values)).days if len(time_values) >= 2 else 0,
        "active_years": len({dt.year for dt in time_values}),
        "distinct_account_count": len(accounts),
        "account_pairs": top_account_pairs,
        "fund_cycle_patterns": _fund_cycle_patterns(entity_name, incoming, outgoing, self_transfer_count, self_transfer_amount),
    }
    anomaly_flags = _detect_anomalies(structured)
    structured["anomaly_flags"] = anomaly_flags
    status_signal = _status_signal(total_rows, anomaly_flags)
    truncated = rows_truncated or groups_truncated

    summary = _summary(entity_name, structured, anomaly_flags)
    output = {
        "summary": summary,
        "row_count": total_rows,
        "status_signal": status_signal,
        "structured": structured,
        "anomaly_flags": anomaly_flags,
        "evidence_records": evidence_records,
        "evidence_refs": [],
        "rows_read": len(rows),
        "truncated": truncated,
        "limit_reason": "max_rows_per_call" if rows_truncated else "max_unique_groups" if groups_truncated else None,
        "aggregation_complete": not truncated,
    }
    return _finalize(case_id, "transfer_structure_analyzer", params, output)


def _first_datetime(payload: dict[str, Any], fields: list[str]) -> datetime | None:
    for field in fields:
        value = parse_datetime(payload.get(field))
        if value:
            return value
    return None


def _mask_account(value: str | None) -> str:
    text = str(value or "")
    if len(text) <= 8:
        return text
    return f"{text[:4]}...{text[-4:]}"


def _top_counterparties(items: dict[str, dict[str, Any]], total_amount: float) -> list[dict[str, Any]]:
    result = []
    ordered = sorted(items.values(), key=lambda item: (item["amount"], item["count"]), reverse=True)[:5]
    for index, item in enumerate(ordered, start=1):
        amount = float(item.get("amount") or 0)
        result.append(
            {
                "rank": index,
                "name": item.get("name"),
                "count": int(item.get("count") or 0),
                "amount": round(amount, 2),
                "amount_ratio": round(amount / total_amount, 4) if total_amount else 0,
            }
        )
    return result


def _hhi(counter: Counter[str]) -> float:
    total = sum(float(value) for value in counter.values())
    if total <= 0:
        return 0.0
    return round(sum((float(value) / total * 100) ** 2 for value in counter.values()), 2)


def _top_n_ratio(counter: Counter[Any], total: int, limit: int) -> float:
    if not total:
        return 0.0
    count = sum(value for _, value in counter.most_common(limit))
    return round(count / total, 4)


def _fund_cycle_patterns(
    entity_name: str,
    incoming: dict[str, dict[str, Any]],
    outgoing: dict[str, dict[str, Any]],
    self_transfer_count: int,
    self_transfer_amount: float,
) -> list[dict[str, Any]]:
    patterns: list[dict[str, Any]] = []
    if self_transfer_count:
        patterns.append(
            {
                "pattern_type": "self_loop",
                "entities": [entity_name],
                "total_flow": round(self_transfer_amount, 2),
                "round_trips": self_transfer_count,
            }
        )
    for name in sorted(set(incoming).intersection(outgoing)):
        in_item = incoming[name]
        out_item = outgoing[name]
        patterns.append(
            {
                "pattern_type": "a_b_a",
                "entities": [entity_name, name],
                "total_flow": round(float(in_item.get("amount") or 0) + float(out_item.get("amount") or 0), 2),
                "round_trips": min(int(in_item.get("count") or 0), int(out_item.get("count") or 0)),
            }
        )
    patterns.sort(key=lambda item: (item["total_flow"], item["round_trips"]), reverse=True)
    return patterns[:5]


def _detect_anomalies(structured: dict[str, Any]) -> list[str]:
    flags = []
    if structured["self_transfer_ratio"] > 0.5 or structured["self_transfer_amount_ratio"] > 0.7:
        flags.append("self_transfer_dominant")
    if structured["counterparty_hhi"] > 2500:
        flags.append("high_concentration")
    if structured["night_ratio"] > 0.3 and structured["total_rows"] > 100:
        flags.append("night_intensive")
    if structured["weekend_ratio"] > 0.25:
        flags.append("weekend_active")
    if structured["distinct_account_count"] >= 5 and structured["total_rows"] > 500:
        flags.append("multi_account_rotation")
    if structured["fund_cycle_patterns"]:
        flags.append("fund_cycle_detected")
    return flags


def _status_signal(total_rows: int, anomaly_flags: list[str]) -> str:
    if total_rows == 0:
        return "weakened"
    critical = {"self_transfer_dominant", "fund_cycle_detected"}
    if len(anomaly_flags) >= 2 or critical.intersection(anomaly_flags):
        return "enhanced"
    if total_rows > 100 and anomaly_flags:
        return "enhanced"
    return "unresolved"


def _summary(entity_name: str, structured: dict[str, Any], anomaly_flags: list[str]) -> str:
    return (
        f"{entity_name}在银行转账关系分析（模型 86）中命中{structured['total_rows']:,}条邻域记录，"
        f"自转账占比{structured['self_transfer_ratio']:.2%}，"
        f"对手方{structured['distinct_counterparty_count']}个，"
        f"对手方HHI为{structured['counterparty_hhi']:.2f}，"
        f"夜间占比{structured['night_ratio']:.2%}，"
        f"触发{len(anomaly_flags)}个结构化异常标记。"
    )


def _finalize(case_id: int, tool_name: str, params: dict[str, Any], output: dict[str, Any]) -> dict[str, Any]:
    request_fingerprint = stable_id("req", {"case_id": case_id, "tool_name": tool_name, "params": params})
    fact_id = stable_id("tool_fact", {"case_id": case_id, "tool_name": tool_name, "params": params, "output": output})
    output["tool_name"] = tool_name
    output["request_fingerprint"] = request_fingerprint
    output["fact_id"] = fact_id
    output["fact_refs"] = [fact_id]
    if is_incomplete_tool_output(output):
        output["status_signal"] = "unresolved"
    return output

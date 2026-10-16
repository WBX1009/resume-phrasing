from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
import json
import math
import re

from .entity_resolver import EntityDictionary, clean_value, merge_sets, resolve_entity
from .field_profile import parse_datetime, parse_number

TIME_PATTERN = re.compile(r"\d{1,2}:\d{2}")
DATE_PATTERN = re.compile(r"\d{4}[-/.年]\d{1,2}")


def safe_json(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, str):
        payload = json.loads(value)
        if isinstance(payload, dict):
            return payload
    raise ValueError("json_data is not an object")


def safe_float(value: Any) -> float | None:
    number = parse_number(value)
    if number is None or math.isnan(number):
        return None
    return float(number)


def top_counter(counter: Counter[str], limit: int = 5) -> list[dict[str, Any]]:
    return [
        {"rank": index + 1, "value": value, "count": count}
        for index, (value, count) in enumerate(counter.most_common(limit))
    ]


def amount_bins(values: list[float]) -> dict[str, int]:
    bins = {"lt_10k": 0, "10k_50k": 0, "50k_100k": 0, "100k_1m": 0, "gte_1m": 0}
    for value in values:
        if value < 10000:
            bins["lt_10k"] += 1
        elif value < 50000:
            bins["10k_50k"] += 1
        elif value < 100000:
            bins["50k_100k"] += 1
        elif value < 1000000:
            bins["100k_1m"] += 1
        else:
            bins["gte_1m"] += 1
    return bins


def percentile(values: list[float], ratio: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * ratio))))
    return round(ordered[index], 2)


def preview_row(row_id: Any, model_id: int, payload: dict[str, Any], preferred: list[str]) -> dict[str, Any]:
    result: dict[str, Any] = {"source_row_id": row_id, "model_id": model_id}
    keys = [key for key in preferred if key and key in payload]
    keys.extend(key for key in payload.keys() if key not in keys and not str(key).startswith("_"))
    for key in keys[:10]:
        result[key] = payload.get(key)
    return result


@dataclass
class EntityAccumulator:
    entity_key: str
    entity_name: str
    entity_type: str
    source_field: str
    record_count: int = 0
    amount_sum: float = 0.0
    amount_max: float = 0.0
    amount_values: list[float] = field(default_factory=list)
    top_amounts: list[dict[str, Any]] = field(default_factory=list)
    time_values: list[datetime] = field(default_factory=list)
    month_buckets: Counter[str] = field(default_factory=Counter)
    day_buckets: Counter[str] = field(default_factory=Counter)
    hour_buckets: Counter[str] = field(default_factory=Counter)
    night_count: int = 0
    direction_counts: Counter[str] = field(default_factory=Counter)
    counterparty_counts: Counter[str] = field(default_factory=Counter)
    category_counts: Counter[str] = field(default_factory=Counter)
    accounts: dict[str, set[str]] = field(default_factory=lambda: {"accounts": set(), "phones": set()})
    near_threshold_count: int = 0
    same_day_same_amount_max: float = 0.0
    evidence: list[dict[str, Any]] = field(default_factory=list)

    def add_amount(
        self,
        amount: float | None,
        row_id: Any,
        model_id: int,
        payload: dict[str, Any],
        preferred: list[str],
        thresholds: dict[str, Any],
    ) -> None:
        if amount is None:
            return
        self.amount_sum += amount
        self.amount_max = max(self.amount_max, amount)
        self.amount_values.append(amount)
        floor = float(thresholds.get("near_amount_floor", 45000))
        ceiling = float(thresholds.get("near_amount_ceiling", 50000))
        if floor <= amount < ceiling:
            self.near_threshold_count += 1
        item = {
            "amount": round(amount, 2),
            "evidence": preview_row(row_id, model_id, payload, preferred),
        }
        self.top_amounts.append(item)
        self.top_amounts.sort(key=lambda value: value["amount"], reverse=True)
        del self.top_amounts[5:]

    def add_time(self, dt: datetime | None, has_clock_time: bool, has_calendar_date: bool) -> None:
        if not dt:
            return
        self.time_values.append(dt)
        if has_calendar_date:
            self.month_buckets[dt.strftime("%Y-%m")] += 1
            self.day_buckets[dt.strftime("%Y-%m-%d")] += 1
        if has_clock_time:
            hour_key = dt.strftime("%Y-%m-%d %H") if has_calendar_date else dt.strftime("%H")
            self.hour_buckets[hour_key] += 1
        if has_clock_time and (dt.hour >= 22 or dt.hour < 6):
            self.night_count += 1

    def add_evidence(self, row_id: Any, model_id: int, payload: dict[str, Any], preferred: list[str], limit: int) -> None:
        if len(self.evidence) < limit:
            self.evidence.append(preview_row(row_id, model_id, payload, preferred))

    def to_profile(self, model_id: int, rank: int) -> dict[str, Any]:
        amount_count = len(self.amount_values)
        time_count = len(self.time_values)
        fact_id = f"m{model_id}_entity_{rank:02d}"
        return {
            "rank": rank,
            "fact_id": fact_id,
            "entity_key": self.entity_key,
            "entity_name": self.entity_name,
            "entity_type": self.entity_type,
            "source_field": self.source_field,
            "metrics": {
                "record_count": self.record_count,
                "amount_sum": round(self.amount_sum, 2),
                "amount_max": round(self.amount_max, 2),
                "amount_avg": round(self.amount_sum / amount_count, 2) if amount_count else 0,
                "amount_p50": percentile(self.amount_values, 0.5),
                "amount_p95": percentile(self.amount_values, 0.95),
                "amount_bins": amount_bins(self.amount_values),
                "near_threshold_count": self.near_threshold_count,
                "night_count": self.night_count,
                "night_ratio": round(self.night_count / time_count, 4) if time_count else 0,
                "max_hourly_count": max(self.hour_buckets.values()) if self.hour_buckets else 0,
                "distinct_counterparty_count": len(self.counterparty_counts),
                "same_day_same_amount_max": self.same_day_same_amount_max,
            },
            "top_amount_records": self.top_amounts,
            "top_time_buckets": {
                "months": top_counter(self.month_buckets),
                "days": top_counter(self.day_buckets),
                "hours": top_counter(self.hour_buckets),
            },
            "direction_counts": top_counter(self.direction_counts),
            "counterparty_topk": top_counter(self.counterparty_counts),
            "category_topk": top_counter(self.category_counts),
            "accounts": sorted(self.accounts["accounts"]),
            "phones": sorted(self.accounts["phones"]),
            "evidence_refs": [fact_id] if self.evidence else [],
            "evidence_records": self.evidence,
            "anomaly_flags": [],
            "followup_hints": [],
        }


@dataclass
class RelationAccumulator:
    source: str
    target: str
    count: int = 0
    amount_sum: float = 0.0
    amount_max: float = 0.0
    time_values: list[datetime] = field(default_factory=list)
    evidence: list[dict[str, Any]] = field(default_factory=list)

    def to_profile(self, model_id: int, rank: int) -> dict[str, Any]:
        fact_id = f"m{model_id}_relation_{rank:02d}"
        return {
            "rank": rank,
            "fact_id": fact_id,
            "source": self.source,
            "target": self.target,
            "count": self.count,
            "amount_sum": round(self.amount_sum, 2),
            "amount_max": round(self.amount_max, 2),
            "self_transfer": self.source == self.target,
            "time_range": {
                "min": min(self.time_values).isoformat(sep=" ") if self.time_values else None,
                "max": max(self.time_values).isoformat(sep=" ") if self.time_values else None,
            },
            "evidence_refs": [fact_id] if self.evidence else [],
            "evidence_records": self.evidence,
            "anomaly_flags": [],
        }


def _first_number(payload: dict[str, Any], fields: list[str]) -> float | None:
    for field in fields:
        value = safe_float(payload.get(field))
        if value is not None:
            return value
    return None


def _first_datetime(payload: dict[str, Any], fields: list[str]) -> tuple[datetime | None, bool, bool]:
    for field in fields:
        raw_value = payload.get(field)
        dt = parse_datetime(raw_value)
        if dt:
            text = str(raw_value)
            return dt, bool(TIME_PATTERN.search(text)), bool(DATE_PATTERN.search(text))
    return None, False, False


def _field_values(payload: dict[str, Any], fields: list[str]) -> list[str]:
    result = []
    for field in fields:
        value = clean_value(payload.get(field))
        if value:
            result.append(value)
    return result


def build_model_profiles(
    model_id: int,
    rows: list[tuple[Any, dict[str, Any]]],
    strategy: dict[str, Any],
    dictionary: EntityDictionary,
    limits: dict[str, int],
    thresholds: dict[str, Any] | None = None,
) -> dict[str, Any]:
    thresholds = thresholds or {}
    entity_limit = int(limits.get("entities", 5))
    relation_limit = int(limits.get("relations", 5))
    evidence_limit = int(limits.get("evidence", 3))

    amount_fields = [strategy.get("primary_amount_field"), *list(strategy.get("amount_fields") or [])]
    amount_fields = [field for field in dict.fromkeys(amount_fields) if field]
    time_fields = list(strategy.get("time_fields") or [])
    counterparty_fields = list(strategy.get("counterparty_fields") or [])
    direction_field = strategy.get("direction_field")
    category_field = strategy.get("category_field")
    count_fields = list(strategy.get("count_fields") or [])
    relation = strategy.get("relation") or {}
    preferred = [
        *list(strategy.get("person_fields") or []),
        *list(strategy.get("account_fields") or []),
        *counterparty_fields,
        *amount_fields,
        *time_fields,
        direction_field,
        category_field,
        relation.get("source"),
        relation.get("target"),
    ]
    preferred = [field for field in dict.fromkeys(preferred) if field]

    entities: dict[str, EntityAccumulator] = {}
    relations: dict[tuple[str, str], RelationAccumulator] = {}
    category_counts: Counter[str] = Counter()
    numeric_values: dict[str, list[float]] = defaultdict(list)
    parse_errors = 0

    for row_id, payload in rows:
        try:
            amount = _first_number(payload, amount_fields)
            dt, has_clock_time, has_calendar_date = _first_datetime(payload, time_fields)
        except Exception:
            parse_errors += 1
            continue

        for field in [*amount_fields, *count_fields]:
            value = safe_float(payload.get(field))
            if value is not None:
                numeric_values[field].append(value)

        if category_field:
            value = clean_value(payload.get(category_field))
            if value:
                category_counts[value] += 1

        resolved = resolve_entity(payload, strategy, dictionary)
        if resolved:
            key = resolved["entity_key"]
            item = entities.setdefault(
                key,
                EntityAccumulator(
                    entity_key=key,
                    entity_name=resolved["entity_name"],
                    entity_type=resolved["entity_type"],
                    source_field=resolved.get("source_field") or "",
                ),
            )
            item.record_count += 1
            merge_sets(item.accounts, resolved.get("accounts", []), resolved.get("phones", []))
            item.add_amount(amount, row_id, model_id, payload, preferred, thresholds)
            item.add_time(dt, has_clock_time, has_calendar_date)
            if direction_field:
                direction = clean_value(payload.get(direction_field))
                if direction:
                    item.direction_counts[direction] += 1
            for value in _field_values(payload, counterparty_fields):
                item.counterparty_counts[value] += 1
            if category_field:
                value = clean_value(payload.get(category_field))
                if value:
                    item.category_counts[value] += 1
            for count_field in count_fields:
                count_value = safe_float(payload.get(count_field))
                if count_value is not None:
                    item.same_day_same_amount_max = max(item.same_day_same_amount_max, count_value)
            item.add_evidence(row_id, model_id, payload, preferred, evidence_limit)

        if relation:
            source = clean_value(payload.get(relation.get("source")))
            target = clean_value(payload.get(relation.get("target")))
            if source and target:
                edge_key = (source, target)
                edge = relations.setdefault(edge_key, RelationAccumulator(source=source, target=target))
                edge.count += 1
                if amount is not None:
                    edge.amount_sum += amount
                    edge.amount_max = max(edge.amount_max, amount)
                if dt:
                    edge.time_values.append(dt)
                if len(edge.evidence) < evidence_limit:
                    edge.evidence.append(preview_row(row_id, model_id, payload, preferred))

    entity_items = sorted(
        entities.values(),
        key=lambda item: (item.amount_sum, item.record_count, len(item.counterparty_counts)),
        reverse=True,
    )
    relation_items = sorted(
        relations.values(),
        key=lambda item: (item.amount_sum, item.count),
        reverse=True,
    )
    numeric_summary = {}
    for field, values in numeric_values.items():
        if values:
            numeric_summary[field] = {
                "count": len(values),
                "sum": round(sum(values), 2),
                "max": round(max(values), 2),
                "min": round(min(values), 2),
                "avg": round(sum(values) / len(values), 2),
                "p95": percentile(values, 0.95),
            }
    return {
        "parse_errors": parse_errors,
        "numeric_summary": numeric_summary,
        "category_topk": top_counter(category_counts, 10),
        "focus_entities": [
            item.to_profile(model_id, index + 1)
            for index, item in enumerate(entity_items[:entity_limit])
        ],
        "relation_profiles": [
            item.to_profile(model_id, index + 1)
            for index, item in enumerate(relation_items[:relation_limit])
        ],
        "relation_summary": {
            "edge_count": len(relations),
            "relation_row_count": sum(item.count for item in relations.values()),
            "self_edge_count": sum(1 for item in relations.values() if item.source == item.target),
            "self_edge_amount_sum": round(
                sum(item.amount_sum for item in relations.values() if item.source == item.target),
                2,
            ),
        },
    }

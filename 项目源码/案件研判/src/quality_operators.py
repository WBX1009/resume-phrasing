"""Deterministic operators used by every quality-pipeline profile."""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter, defaultdict
from datetime import datetime
from statistics import mean
from typing import Any, Iterable

from .quality_models import Fact, ModelProfile


_NUMBER_RE = re.compile(r"[-+]?\d+(?:\.\d+)?")
_IDENTIFIER_FIELD_TOKENS = (
    "手机号",
    "手机号码",
    "电话号码",
    "电话号",
    "身份证",
    "证件号",
    "账号",
    "账户",
    "卡号",
    "银行卡",
    "号码",
    "编号",
)
_TEMPORAL_FIELD_TOKENS = (
    "时间",
    "日期",
    "年月",
    "年份",
    "timestamp",
    "datetime",
    "date",
)
_DURATION_FIELD_TOKENS = ("时长", "间隔", "小时", "分钟", "秒数", "天数")


def _row_data(row: dict[str, Any]) -> dict[str, Any]:
    value = row.get("data", {})
    return value if isinstance(value, dict) else {}


def stable_id(kind: str, *parts: object) -> str:
    payload = "|".join([kind, *(str(part) for part in parts)])
    return f"{kind}_{hashlib.sha256(payload.encode('utf-8')).hexdigest()[:16]}"


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value) if math.isfinite(float(value)) else None
    if isinstance(value, str):
        match = _NUMBER_RE.search(value.replace(",", ""))
        if match:
            try:
                parsed = float(match.group())
                return parsed if math.isfinite(parsed) else None
            except ValueError:
                return None
    return None


def _datetime(value: Any) -> datetime | None:
    if not isinstance(value, str):
        return None
    text = value.strip().replace("/", "-")
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.strptime(text[:19], fmt)
        except ValueError:
            continue
    return None


def _source_ids(rows: list[dict[str, Any]], limit: int = 50) -> list[int]:
    source_ids: list[int] = []
    for row in rows[:limit]:
        try:
            source_ids.append(int(row.get("row_id", 0)))
        except (AttributeError, TypeError, ValueError):
            # A malformed source row must not break the evidence ledger.  The
            # data-quality Fact emitted by ``compute_facts`` records the loss
            # of source precision explicitly.
            continue
    return source_ids


def _fact(
    profile: ModelProfile,
    case_id: int | str,
    kind: str,
    subject: str,
    value: Any,
    rows: list[dict[str, Any]],
    fields: Iterable[str] = (),
    complete: bool = True,
) -> Fact:
    return Fact(
        fact_id=stable_id(kind, profile.model_id, case_id, subject, value),
        model_id=profile.model_id,
        case_id=case_id,
        kind=kind,
        subject=subject,
        value=value,
        source_row_ids=_source_ids(rows),
        source_fields=sorted(set(fields)),
        complete=complete,
    )


def _candidate_fields(rows: list[dict[str, Any]]) -> list[str]:
    counts: Counter[str] = Counter()
    for row in rows:
        for key, value in _row_data(row).items():
            if value not in (None, "", [], {}):
                counts[str(key)] += 1
    return [field for field, _ in counts.most_common()]


def _is_identifier_field(field: str) -> bool:
    """Prevent phone/account/card identifiers from becoming numeric metrics."""
    text = str(field).strip().lower()
    return any(token.lower() in text for token in _IDENTIFIER_FIELD_TOKENS) or text.endswith("_id")


def _is_temporal_field(field: str) -> bool:
    """Prevent date/time encodings from becoming amount-like metrics."""
    text = str(field).strip().lower()
    if not any(token.lower() in text for token in _TEMPORAL_FIELD_TOKENS):
        return False
    return not any(token.lower() in text for token in _DURATION_FIELD_TOKENS)


def _numeric_summary(profile: ModelProfile, case_id: int | str, rows: list[dict[str, Any]]) -> list[Fact]:
    facts: list[Fact] = []
    fields = _candidate_fields(rows)
    for field in fields:
        if _is_identifier_field(field) or _is_temporal_field(field):
            continue
        values = [value for row in rows for value in [_number(_row_data(row).get(field))] if value is not None]
        if not values or len(values) < max(2, min(3, len(rows))):
            continue
        facts.append(
            _fact(
                profile,
                case_id,
                "numeric_summary",
                field,
                {
                    "count": len(values),
                    "sum": round(sum(values), 4),
                    "min": round(min(values), 4),
                    "max": round(max(values), 4),
                    "mean": round(mean(values), 4),
                },
                rows,
                [field],
            )
        )
    return facts


def _category_summary(profile: ModelProfile, case_id: int | str, rows: list[dict[str, Any]]) -> list[Fact]:
    facts: list[Fact] = []
    fields = _candidate_fields(rows)
    for field in fields:
        values = [str(_row_data(row).get(field)).strip() for row in rows]
        values = [value for value in values if value and value.lower() not in {"none", "null"}]
        if len(values) < 3:
            continue
        counts = Counter(values)
        if len(counts) > 1 and len(counts) <= max(50, len(values)):
            top = [{"value": key, "count": count} for key, count in counts.most_common(5)]
            facts.append(
                _fact(
                    profile,
                    case_id,
                    "category_summary",
                    field,
                    {"distinct": len(counts), "top": top},
                    rows,
                    [field],
                )
            )
    return facts


def _temporal_summary(profile: ModelProfile, case_id: int | str, rows: list[dict[str, Any]]) -> list[Fact]:
    facts: list[Fact] = []
    for field in _candidate_fields(rows):
        parsed = [value for row in rows for value in [_datetime(_row_data(row).get(field))] if value]
        if len(parsed) < 2:
            continue
        parsed.sort()
        intervals = [(right - left).total_seconds() for left, right in zip(parsed, parsed[1:])]
        value = {
            "count": len(parsed),
            "first": parsed[0].isoformat(sep=" "),
            "last": parsed[-1].isoformat(sep=" "),
            "span_days": round((parsed[-1] - parsed[0]).total_seconds() / 86400, 4),
            "median_interval_hours": round(sorted(intervals)[len(intervals) // 2] / 3600, 4) if intervals else None,
        }
        facts.append(_fact(profile, case_id, "temporal_summary", field, value, rows, [field]))
        break  # one representative time field keeps the fact pack compact
    return facts


def _relation_summary(profile: ModelProfile, case_id: int | str, rows: list[dict[str, Any]]) -> list[Fact]:
    facts: list[Fact] = []
    data_keys = _candidate_fields(rows)
    source_candidates = [key for key in data_keys if any(token in key for token in ("流出", "起点", "本方", "对象A", "号码A"))]
    target_candidates = [key for key in data_keys if any(token in key for token in ("流入", "终点", "对端", "对象B", "号码B"))]
    if source_candidates and target_candidates:
        source_key, target_key = source_candidates[0], target_candidates[0]
        edges: Counter[tuple[str, str]] = Counter()
        for row in rows:
            data = _row_data(row)
            source, target = str(data.get(source_key, "")).strip(), str(data.get(target_key, "")).strip()
            if source and target and source.lower() != "none" and target.lower() != "none":
                edges[(source, target)] += 1
        if edges:
            top_edges = [
                {"source": source, "target": target, "count": count}
                for (source, target), count in edges.most_common(10)
            ]
            facts.append(
                _fact(
                    profile,
                    case_id,
                    "relation_summary",
                    f"{source_key}->{target_key}",
                    {"edge_count": len(edges), "top_edges": top_edges},
                    rows,
                    [source_key, target_key],
                )
            )
    return facts


def _sequence_summary(profile: ModelProfile, case_id: int | str, rows: list[dict[str, Any]]) -> list[Fact]:
    facts: list[Fact] = []
    time_fields = [field for field in _candidate_fields(rows) if "时间" in field or "日期" in field]
    amount_fields = [field for field in _candidate_fields(rows) if "金额" in field or "余额" in field]
    if not time_fields:
        return facts
    time_key = time_fields[0]
    times = sorted(value for row in rows for value in [_datetime(_row_data(row).get(time_key))] if value)
    if len(times) < 2:
        return facts
    intervals = [(right - left).total_seconds() for left, right in zip(times, times[1:])]
    value: dict[str, Any] = {
        "time_field": time_key,
        "events": len(times),
        "short_interval_count": sum(interval <= 86400 for interval in intervals),
    }
    if amount_fields:
        amount_key = amount_fields[0]
        amounts = [value for row in rows for value in [_number(_row_data(row).get(amount_key))] if value is not None]
        if amounts:
            value["amount_field"] = amount_key
            value["distinct_amounts"] = len({round(amount, 2) for amount in amounts})
    facts.append(_fact(profile, case_id, "sequence_summary", time_key, value, rows, [time_key, *amount_fields[:1]]))
    return facts


def _entity_cardinality(profile: ModelProfile, case_id: int | str, rows: list[dict[str, Any]]) -> list[Fact]:
    facts: list[Fact] = []
    for field in _candidate_fields(rows):
        if not any(token in field for token in ("姓名", "账号", "卡号", "手机号", "对象", "对端", "联系人")):
            continue
        values = {str(_row_data(row).get(field)).strip() for row in rows}
        values.discard("")
        values.discard("None")
        if values:
            facts.append(
                _fact(
                    profile,
                    case_id,
                    "entity_cardinality",
                    field,
                    {"distinct": len(values), "examples": sorted(values)[:5]},
                    rows,
                    [field],
                )
            )
    return facts[:8]


def compute_facts(profile: ModelProfile, rows: list[dict[str, Any]], case_id: int | str) -> list[Fact]:
    """Compute a compact, deterministic fact pack for one model slice."""
    raw_rows = list(rows)
    if not raw_rows:
        return [
            _fact(profile, case_id, "empty_result", "rows", {"count": 0}, [], complete=False)
        ]

    valid_rows = [row for row in raw_rows if isinstance(row, dict)]
    invalid_count = len(raw_rows) - len(valid_rows)
    if not valid_rows:
        return [
            _fact(
                profile,
                case_id,
                "row_count",
                "rows",
                {"count": len(raw_rows), "model_id": profile.model_id},
                [],
                complete=False,
            ),
            _fact(
                profile,
                case_id,
                "data_quality",
                "invalid_rows",
                {"invalid_count": invalid_count, "valid_count": 0},
                [],
                complete=False,
            ),
        ]

    facts: list[Fact] = [
        _fact(
            profile,
            case_id,
            "row_count",
            "rows",
            {"count": len(raw_rows), "model_id": profile.model_id},
            valid_rows,
            complete=invalid_count == 0,
        )
    ]
    facts.extend(_entity_cardinality(profile, case_id, valid_rows))
    facts.extend(_numeric_summary(profile, case_id, valid_rows))
    facts.extend(_category_summary(profile, case_id, valid_rows))
    facts.extend(_temporal_summary(profile, case_id, valid_rows))
    facts.extend(_relation_summary(profile, case_id, valid_rows))
    facts.extend(_sequence_summary(profile, case_id, valid_rows))
    if invalid_count:
        facts.append(
            _fact(
                profile,
                case_id,
                "data_quality",
                "invalid_rows",
                {"invalid_count": invalid_count, "valid_count": len(valid_rows)},
                valid_rows,
                complete=False,
            )
        )
    # De-duplicate by ID while preserving order.  This makes replay stable if
    # a field is both a family operator input and a semantic hint.
    unique: dict[str, Fact] = {}
    for fact in facts:
        unique.setdefault(fact.fact_id, fact)
    return list(unique.values())


def fact_index(facts: Iterable[Fact]) -> dict[str, Fact]:
    return {fact.fact_id: fact for fact in facts}

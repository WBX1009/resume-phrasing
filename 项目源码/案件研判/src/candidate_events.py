"""Build traceable candidate events from deterministic facts and model clues.

The public project does not contain the production model or business database.
This module therefore accepts structured model clues as an injected boundary and
builds the deterministic part of candidate generation from fact-pack anomaly
flags. It never interprets free-form evidence text as a confirmed event.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
from typing import Any


class CandidateEventError(ValueError):
    """Raised when a candidate event or model clue is not structured enough."""


@dataclass(frozen=True)
class CandidateEventBuilder:
    """Convert rule flags and structured model clues into candidate events."""

    max_events: int = 100

    def __post_init__(self) -> None:
        if not isinstance(self.max_events, int) or isinstance(self.max_events, bool) or self.max_events <= 0:
            raise ValueError("max_events must be a positive integer")

    def build(
        self,
        fact_pack: Mapping[str, Any],
        *,
        model_clues: Sequence[Mapping[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        if not isinstance(fact_pack, Mapping):
            raise CandidateEventError("fact_pack must be a mapping")

        candidates: dict[str, dict[str, Any]] = {}
        for flag, owner in _iter_rule_flags(fact_pack):
            _merge_event(candidates, _event_from_rule_flag(flag, owner))
        for clue in model_clues or ():
            _merge_event(candidates, _event_from_model_clue(clue))

        events = sorted(candidates.values(), key=lambda item: item["event_id"])
        if len(events) > self.max_events:
            raise CandidateEventError(
                f"candidate event limit exceeded ({len(events)}/{self.max_events})"
            )
        return events


def _iter_rule_flags(fact_pack: Mapping[str, Any]):
    """Yield each anomaly flag with the fact/entity/relation that produced it."""

    for flag in fact_pack.get("anomaly_flags", []) or []:
        if isinstance(flag, Mapping):
            yield flag, fact_pack
    for entity in fact_pack.get("focus_entities", []) or []:
        if not isinstance(entity, Mapping):
            continue
        for flag in entity.get("anomaly_flags", []) or []:
            if isinstance(flag, Mapping):
                yield flag, entity
    for relation in fact_pack.get("relation_profiles", []) or []:
        if not isinstance(relation, Mapping):
            continue
        for flag in relation.get("anomaly_flags", []) or []:
            if isinstance(flag, Mapping):
                yield flag, relation


def _event_from_rule_flag(flag: Mapping[str, Any], owner: Mapping[str, Any]) -> dict[str, Any]:
    risk_tag = _required_text(flag.get("risk_tag"), "rule risk_tag")
    subjects = _subject_ids(owner, flag) or ["case"]
    fact_ids = _unique_strings([owner.get("fact_id"), flag.get("fact_id")])
    evidence_refs = _unique_strings(
        _sequence_or_empty(owner.get("evidence_refs"))
        + _sequence_or_empty(flag.get("evidence_refs"))
    )
    event_type = f"risk_{risk_tag}"
    event = {
        "event_id": _event_id(event_type, subjects, fact_ids, flag.get("time_range")),
        "event_type": event_type,
        "subject_ids": subjects,
        "time_range": _time_range(flag.get("time_range") or owner.get("time_range")),
        "risk_tags": [risk_tag],
        "fact_ids": fact_ids,
        "evidence_refs": evidence_refs,
        "source": ["rule"],
        "status": "candidate",
        "rationale": str(flag.get("report_expression") or flag.get("trigger_rule") or risk_tag),
        "metrics": dict(flag.get("metrics") or {}) if isinstance(flag.get("metrics"), Mapping) else {},
    }
    if len(subjects) == 1:
        event["subject_id"] = subjects[0]
    return event


def _event_from_model_clue(clue: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(clue, Mapping):
        raise CandidateEventError("model clues must contain mappings")
    event_type = _required_text(clue.get("event_type"), "model clue event_type")
    subjects = _unique_strings(
        _sequence_or_empty(clue.get("subject_ids")) + [clue.get("subject_id")]
    )
    if not subjects:
        raise CandidateEventError("model clue must contain subject_ids or subject_id")
    fact_ids = _unique_strings(clue.get("fact_ids") or [])
    evidence_refs = _unique_strings(clue.get("evidence_refs") or [])
    risk_tags = _unique_strings(clue.get("risk_tags") or [])
    event = {
        "event_id": str(clue.get("event_id") or _event_id(event_type, subjects, fact_ids, clue.get("time_range"))),
        "event_type": event_type,
        "subject_ids": subjects,
        "time_range": _time_range(clue.get("time_range")),
        "risk_tags": risk_tags,
        "fact_ids": fact_ids,
        "evidence_refs": evidence_refs,
        "source": ["model"],
        "status": "candidate",
        "rationale": str(clue.get("rationale") or "structured model clue"),
    }
    if len(subjects) == 1:
        event["subject_id"] = subjects[0]
    if "confidence" in clue:
        confidence = clue["confidence"]
        if not isinstance(confidence, (int, float)) or isinstance(confidence, bool) or not 0 <= confidence <= 1:
            raise CandidateEventError("model clue confidence must be between 0 and 1")
        event["confidence"] = float(confidence)
    return event


def _merge_event(target: dict[str, dict[str, Any]], event: dict[str, Any]) -> None:
    existing = target.get(event["event_id"])
    if existing is None:
        target[event["event_id"]] = event
        return
    for field in ("subject_ids", "risk_tags", "fact_ids", "evidence_refs", "source"):
        existing[field] = _unique_strings([*(existing.get(field) or []), *(event.get(field) or [])])
    if event.get("rationale") and event["rationale"] not in existing.get("rationale", ""):
        existing["rationale"] = f"{existing.get('rationale', '')}; {event['rationale']}".strip("; ")
    if event.get("metrics"):
        existing["metrics"] = {**existing.get("metrics", {}), **event["metrics"]}
    if event.get("confidence") is not None:
        existing["confidence"] = max(existing.get("confidence", 0.0), event["confidence"])


def _subject_ids(owner: Mapping[str, Any], flag: Mapping[str, Any]) -> list[str]:
    return _unique_strings(
        [
            owner.get("entity_name"),
            owner.get("source"),
            owner.get("target"),
            flag.get("entity_name"),
            flag.get("source"),
            flag.get("target"),
        ]
    )


def _sequence_or_empty(value: Any) -> list[Any]:
    if value is None or isinstance(value, (str, bytes)):
        return []
    return list(value) if isinstance(value, Sequence) else []


def _unique_strings(values: Any) -> list[str]:
    if values is None or isinstance(values, (str, bytes)):
        values = [values]
    result: list[str] = []
    for value in values:
        if value is None:
            continue
        text = str(value).strip()
        if text and text not in result:
            result.append(text)
    return result


def _time_range(value: Any) -> list[str] | None:
    if value is None:
        return None
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or len(value) != 2:
        raise CandidateEventError("time_range must contain two bounds")
    return [str(value[0]), str(value[1])]


def _required_text(value: Any, name: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise CandidateEventError(f"{name} must be non-empty")
    return text


def _event_id(event_type: str, subjects: Sequence[str], fact_ids: Sequence[str], time_range: Any) -> str:
    payload = {
        "event_type": event_type,
        "subjects": sorted(subjects),
        "facts": sorted(fact_ids),
        "time_range": list(time_range) if isinstance(time_range, Sequence) and not isinstance(time_range, (str, bytes)) else time_range,
    }
    digest = hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]
    return f"candidate-{digest}"

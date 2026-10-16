"""Case-level event matching and micro-aggregated evaluation.

The evaluator deliberately requires callers to choose the event identity
fields. This public project has no authoritative event taxonomy or adjudicated
time-window tolerance, so the library must not invent either one.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import json
import unicodedata
from typing import Any


Event = Mapping[str, Any]
EventPredicate = Callable[[Event, Event], bool]


@dataclass(frozen=True)
class EventMatchPolicy:
    """A frozen, explicit event equivalence rule.

    ``key_fields`` must name canonical fields shared by expert annotations
    and system outputs. Add a normalized time-window or business discriminator
    here when the annotation protocol requires it. ``predicate`` may impose
    extra conditions (for example, an adjudicated time-overlap rule), but it
    cannot relax equality of the configured identity fields.
    """

    key_fields: tuple[str, ...]
    predicate: EventPredicate | None = None
    case_sensitive: bool = True
    version: str = "event-match-v1"

    def __post_init__(self) -> None:
        fields = tuple(self.key_fields)
        if not fields or any(not isinstance(field, str) or not field.strip() for field in fields):
            raise ValueError("key_fields must contain at least one non-empty field name")
        if len(fields) != len(set(fields)):
            raise ValueError("key_fields must be unique")
        if not isinstance(self.version, str) or not self.version.strip():
            raise ValueError("version must be a non-empty string")
        object.__setattr__(self, "key_fields", fields)

    def to_dict(self) -> dict[str, Any]:
        if self.predicate is not None:
            raise ValueError("policies with a callable predicate cannot be serialized")
        return {
            "version": self.version,
            "key_fields": list(self.key_fields),
            "case_sensitive": self.case_sensitive,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "EventMatchPolicy":
        if not isinstance(value, Mapping):
            raise TypeError("matching policy must be a mapping")
        return cls(
            key_fields=tuple(value.get("key_fields") or ()),
            case_sensitive=bool(value.get("case_sensitive", True)),
            version=str(value.get("version") or "event-match-v1"),
        )

    def matches(self, predicted: Event, gold: Event) -> bool:
        for field in self.key_fields:
            if field not in predicted or field not in gold:
                return False
            left = _canonical_value(predicted[field], self.case_sensitive)
            right = _canonical_value(gold[field], self.case_sensitive)
            if left is None or right is None or left != right:
                return False
        return self.predicate(predicted, gold) if self.predicate is not None else True


@dataclass(frozen=True)
class EventEvaluation:
    tp: int
    fp: int
    fn: int
    precision: float | None
    recall: float | None
    matched_pairs: tuple[tuple[int, int], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "tp": self.tp,
            "fp": self.fp,
            "fn": self.fn,
            "precision": self.precision,
            "recall": self.recall,
            "matched_pairs": [list(pair) for pair in self.matched_pairs],
        }


def evaluate_events(
    predicted_events: Sequence[Event],
    gold_events: Sequence[Event],
    policy: EventMatchPolicy,
) -> EventEvaluation:
    """Match events one-to-one, maximizing TP, then calculate case metrics.

    The augmenting-path matcher prevents an early broad prediction from
    consuming the only Gold event available to a later, more specific one.
    Duplicate predictions can match a Gold at most once and count as FP.
    """

    predictions = _validated_events(predicted_events, "predicted_events")
    gold = _validated_events(gold_events, "gold_events")

    edges = [
        [gold_index for gold_index, gold_event in enumerate(gold) if policy.matches(prediction, gold_event)]
        for prediction in predictions
    ]
    gold_to_prediction: dict[int, int] = {}

    def assign(prediction_index: int, visited_gold: set[int]) -> bool:
        for gold_index in edges[prediction_index]:
            if gold_index in visited_gold:
                continue
            visited_gold.add(gold_index)
            previous = gold_to_prediction.get(gold_index)
            if previous is None or assign(previous, visited_gold):
                gold_to_prediction[gold_index] = prediction_index
                return True
        return False

    for prediction_index in range(len(predictions)):
        assign(prediction_index, set())

    pairs = tuple(sorted((prediction_index, gold_index) for gold_index, prediction_index in gold_to_prediction.items()))
    tp = len(pairs)
    fp = len(predictions) - tp
    fn = len(gold) - tp
    precision = tp / len(predictions) if predictions else None
    recall = tp / len(gold) if gold else None
    return EventEvaluation(tp, fp, fn, precision, recall, pairs)


def aggregate_event_evaluations(results: Sequence[EventEvaluation]) -> dict[str, Any]:
    """Micro-aggregate run/case counts; never union predictions across runs."""

    tp = sum(result.tp for result in results)
    fp = sum(result.fp for result in results)
    fn = sum(result.fn for result in results)
    predicted = tp + fp
    gold = tp + fn
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": tp / predicted if predicted else None,
        "recall": tp / gold if gold else None,
        "runs": len(results),
        "aggregation": "micro_event_counts",
    }


def validate_case_splits(
    split_by_case: Mapping[str, str],
    group_by_case: Mapping[str, str] | None = None,
) -> None:
    """Reject missing/unknown splits and related-case leakage across splits."""

    if not split_by_case:
        raise ValueError("split_by_case must not be empty")
    invalid = sorted(case_id for case_id, split in split_by_case.items() if split not in {"dev", "test"})
    if invalid:
        raise ValueError(f"cases have an invalid split: {invalid}")
    if not {"dev", "test"}.issubset(set(split_by_case.values())):
        raise ValueError("both dev and test cases are required")
    if group_by_case is None:
        return
    unknown_cases = sorted(set(split_by_case) - set(group_by_case))
    if unknown_cases:
        raise ValueError(f"case groups are missing for: {unknown_cases}")
    split_by_group: dict[str, str] = {}
    for case_id, split in split_by_case.items():
        group_id = group_by_case[case_id]
        previous = split_by_group.setdefault(group_id, split)
        if previous != split:
            raise ValueError(f"related case group {group_id!r} crosses dev/test split")


def _validated_events(events: Sequence[Event], name: str) -> list[Event]:
    if isinstance(events, (str, bytes)) or not isinstance(events, Sequence):
        raise TypeError(f"{name} must be a sequence of event objects")
    result: list[Event] = []
    for index, event in enumerate(events):
        if not isinstance(event, Mapping):
            raise TypeError(f"{name}[{index}] must be a mapping")
        result.append(event)
    return result


def _canonical_value(value: Any, case_sensitive: bool) -> str | None:
    if value is None or isinstance(value, (dict, list, tuple, set)):
        return None
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return json.dumps(value, allow_nan=False, separators=(",", ":"))
    normalized = unicodedata.normalize("NFC", str(value)).strip()
    if not case_sensitive:
        normalized = normalized.casefold()
    return normalized or None

import pytest

from src.event_evaluation import (
    EventMatchPolicy,
    aggregate_event_evaluations,
    evaluate_events,
    validate_case_splits,
)


def test_event_match_uses_explicit_identity_fields():
    policy = EventMatchPolicy(("event_type", "subject_id", "object_id", "time_scope"))
    gold = [{"event_type": "资金往来", "subject_id": "person-1", "object_id": "company-2", "time_scope": "2026-05-01/2026-05-10"}]
    predicted = [{"event_type": "资金往来", "subject_id": "person-1", "object_id": "company-2", "time_scope": "2026-05-01/2026-05-10"}]

    result = evaluate_events(predicted, gold, policy)

    assert (result.tp, result.fp, result.fn) == (1, 0, 0)
    assert result.precision == result.recall == 1.0


def test_one_gold_is_never_counted_twice_and_duplicate_prediction_is_false_positive():
    policy = EventMatchPolicy(("event_type", "subject_id", "object_id"))
    event = {"event_type": "关联", "subject_id": "p1", "object_id": "p2"}

    result = evaluate_events([event, dict(event)], [event], policy)

    assert (result.tp, result.fp, result.fn) == (1, 1, 0)
    assert result.matched_pairs == ((0, 0),)


def test_missing_identity_field_does_not_match():
    policy = EventMatchPolicy(("event_type", "subject_id", "object_id"))

    result = evaluate_events(
        [{"event_type": "关联", "subject_id": "p1"}],
        [{"event_type": "关联", "subject_id": "p1", "object_id": "p2"}],
        policy,
    )

    assert (result.tp, result.fp, result.fn) == (0, 1, 1)


def test_adjudicated_predicate_can_apply_extra_time_rule():
    policy = EventMatchPolicy(
        ("event_type", "subject_id", "object_id"),
        predicate=lambda predicted, gold: predicted["time_scope"] == gold["time_scope"],
    )
    result = evaluate_events(
        [{"event_type": "资金", "subject_id": "p1", "object_id": "p2", "time_scope": "May"}],
        [{"event_type": "资金", "subject_id": "p1", "object_id": "p2", "time_scope": "June"}],
        policy,
    )

    assert (result.tp, result.fp, result.fn) == (0, 1, 1)


def test_empty_gold_reports_undefined_recall_not_zero():
    result = evaluate_events([], [], EventMatchPolicy(("event_type",)))

    assert result.precision is None
    assert result.recall is None


def test_aggregate_sums_counts_before_calculating_micro_metrics():
    policy = EventMatchPolicy(("event_type", "subject_id", "object_id"))
    small = {"event_type": "A", "subject_id": "p1", "object_id": "p2"}
    large = {"event_type": "B", "subject_id": "p1", "object_id": "p3"}
    first = evaluate_events([small], [small], policy)
    second = evaluate_events([large] * 4, [large] * 5, policy)

    summary = aggregate_event_evaluations([first, second])

    assert summary == {
        "tp": 5,
        "fp": 0,
        "fn": 1,
        "precision": 1.0,
        "recall": 5 / 6,
        "runs": 2,
        "aggregation": "micro_event_counts",
    }


def test_split_validation_rejects_related_case_leakage():
    validate_case_splits(
        {"case-a": "dev", "case-b": "dev", "case-c": "test"},
        {"case-a": "group-1", "case-b": "group-1", "case-c": "group-2"},
    )

    with pytest.raises(ValueError, match="crosses dev/test"):
        validate_case_splits(
            {"case-a": "dev", "case-b": "test"},
            {"case-a": "same-group", "case-b": "same-group"},
        )


def test_split_validation_rejects_unlabeled_cases_and_invalid_role():
    with pytest.raises(ValueError, match="both dev and test"):
        validate_case_splits({"case-a": "dev"})
    with pytest.raises(ValueError, match="invalid split"):
        validate_case_splits({"case-a": "train", "case-b": "test"})

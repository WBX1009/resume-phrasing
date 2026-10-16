import pytest

from src.event_evaluation import EventMatchPolicy
from src.evaluation_protocol import EvaluationProtocol, EvaluationVariant


def _output(events):
    return {
        "limits": {
            "candidate_limit": 10,
            "llm_call_budget": 20,
            "tool_call_budget": 20,
            "token_budget": 1000,
        },
        "candidate_events": events,
        "final_events": events,
        "usage": {"llm_calls": 0, "tool_calls": 0, "reserved_tokens": 0},
    }


def test_policy_is_versioned_and_serializable():
    policy = EventMatchPolicy(
        key_fields=("event_type", "subject_id"),
        version="demo-v2",
    )
    assert EventMatchPolicy.from_dict(policy.to_dict()) == policy


def test_protocol_keeps_variant_controls_equal_and_scores_outputs():
    policy = EventMatchPolicy(key_fields=("event_type", "subject_id"))
    protocol = EvaluationProtocol(
        dataset_version="synthetic-v1",
        policy=policy,
        candidate_limit=10,
        llm_call_budget=20,
        tool_call_budget=20,
        token_budget=1000,
    )
    variants = [
        EvaluationVariant("baseline", "baseline-v1"),
        EvaluationVariant("full", "full-v1"),
        EvaluationVariant("ablation_no_recheck", "full-v1-no-recheck", ("directed_supplement",)),
    ]
    event = {"event_type": "risk_a", "subject_id": "主体A"}
    result = protocol.evaluate_outputs(
        variants,
        {variant.name: _output([event]) for variant in variants},
        [event],
    )

    assert result["matching_policy_version"] == "event-match-v1"
    assert result["variants"]["full"]["final"]["precision"] == 1.0


def test_protocol_rejects_changed_limits():
    protocol = EvaluationProtocol(
        dataset_version="synthetic-v1",
        policy=EventMatchPolicy(key_fields=("event_type",)),
        candidate_limit=10,
        llm_call_budget=20,
        tool_call_budget=20,
        token_budget=1000,
    )
    with pytest.raises(ValueError, match="frozen limits"):
        protocol.evaluate_outputs(
            [EvaluationVariant("full", "full-v1")],
            {"full": {**_output([]), "limits": {"candidate_limit": 9}}},
            [],
        )

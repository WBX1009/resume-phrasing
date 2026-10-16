import pytest
from uuid import uuid4

pytest.importorskip("langsmith")
from langsmith import Client
from langsmith.schemas import Example

from src.event_evaluation import EventMatchPolicy
from src.langsmith_event_evaluation import (
    build_event_evaluators,
    compare_experiment_metadata,
    run_langsmith_event_experiment,
)


POLICY = EventMatchPolicy(("event_type", "subject_id", "object_id"))
GOLD = {"event_type": "funds", "subject_id": "person-1", "object_id": "company-1"}
CONTROLS = {
    "candidate_limit": 10,
    "llm_call_budget": 20,
    "tool_call_budget": 30,
    "token_budget": 40_000,
}


def metadata(strategy="fixture-v1"):
    return {
        "protocol_version": "test-protocol-v1",
        "dataset_version": "fixture-v1",
        "matching_policy_version": "event-key-v1",
        **CONTROLS,
        "code_revision": "test-code-v1",
        "model_revision": "test-model-v1",
        "prompt_revision": "test-prompt-v1",
        "strategy_version": strategy,
    }


def test_langsmith_row_and_summary_evaluators_apply_case_matching_and_micro_counts():
    candidate, final, summary = build_event_evaluators(POLICY, **CONTROLS)
    inputs = {"case_id": "case-1"}
    outputs = {
        "case_id": "case-1",
        "candidate_events": [GOLD, GOLD],
        "final_events": [GOLD],
        "limits": CONTROLS,
        "usage": {"llm_calls": 2, "tool_calls": 3, "reserved_tokens": 4000},
    }
    reference = {"gold_events": [GOLD]}

    candidate_scores = candidate(inputs=inputs, outputs=outputs, reference_outputs=reference)
    final_scores = final(inputs=inputs, outputs=outputs, reference_outputs=reference)

    assert {item["key"]: item["score"] for item in candidate_scores} == {
        "candidate_tp": 1,
        "candidate_fp": 1,
        "candidate_fn": 0,
        "candidate_precision": 0.5,
        "candidate_recall": 1.0,
    }
    assert {item["key"]: item["score"] for item in final_scores} == {
        "final_tp": 1,
        "final_fp": 0,
        "final_fn": 0,
        "final_precision": 1.0,
        "final_recall": 1.0,
    }


def test_langsmith_sdk_runs_local_repeated_experiment_without_uploading_case_data():
    def target(inputs):
        return {
            "case_id": inputs["case_id"],
            "candidate_events": [GOLD],
            "final_events": [GOLD],
            "limits": CONTROLS,
            "usage": {"llm_calls": 1, "tool_calls": 1, "reserved_tokens": 1000},
        }

    results = run_langsmith_event_experiment(
        Client(),
        target,
        data=[Example(id=uuid4(), inputs={"case_id": "case-1"}, outputs={"gold_events": [GOLD]})],
        policy=POLICY,
        experiment_prefix="offline-contract-test",
        metadata=metadata(),
        upload_results=False,
        num_repetitions=2,
    )
    rows = list(results)
    assert len(rows) == 2
    assert all(row["evaluation_results"] for row in rows)
    summary_scores = {
        item.key: item.score
        for item in results._summary_results["results"]
    }
    assert summary_scores["candidate_tp"] == 2
    assert summary_scores["candidate_recall"] == 1.0
    assert summary_scores["final_precision"] == 1.0
    assert summary_scores["final_recall"] == 1.0
    assert summary_scores["evaluation_run_count"] == 2


def test_langsmith_target_must_report_the_same_case_as_the_example():
    candidate, _, _ = build_event_evaluators(POLICY, **CONTROLS)
    with pytest.raises(ValueError, match="case_id must match"):
        candidate(
            inputs={"case_id": "case-1"},
            outputs={
                "case_id": "case-2", "candidate_events": [], "limits": CONTROLS,
                "usage": {"llm_calls": 0, "tool_calls": 0, "reserved_tokens": 0},
            },
            reference_outputs={"gold_events": []},
        )


def test_evaluator_rejects_candidate_or_call_budget_overrun():
    candidate, _, _ = build_event_evaluators(POLICY, **CONTROLS)
    with pytest.raises(ValueError, match="candidate_limit"):
        candidate(
            inputs={"case_id": "case-1"},
            outputs={
                "case_id": "case-1", "candidate_events": [GOLD] * 11, "final_events": [],
                "limits": CONTROLS,
                "usage": {"llm_calls": 0, "tool_calls": 0, "reserved_tokens": 0},
            },
            reference_outputs={"gold_events": [GOLD]},
        )
    with pytest.raises(ValueError, match="tool_call_budget"):
        candidate(
            inputs={"case_id": "case-1"},
            outputs={
                "case_id": "case-1", "candidate_events": [], "final_events": [],
                "limits": CONTROLS,
                "usage": {"llm_calls": 0, "tool_calls": 31, "reserved_tokens": 0},
            },
            reference_outputs={"gold_events": []},
        )


def test_version_comparison_refuses_mismatched_frozen_controls():
    baseline = metadata("baseline-v1")
    candidate = metadata("candidate-v2")
    compare_experiment_metadata(baseline, candidate)
    candidate["candidate_limit"] = 11
    with pytest.raises(ValueError, match="controls differ"):
        compare_experiment_metadata(baseline, candidate)

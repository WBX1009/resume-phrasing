"""LangSmith adapters for deterministic case-level event evaluation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Callable

from .event_evaluation import EventMatchPolicy, aggregate_event_evaluations, evaluate_events


_FROZEN_METADATA_KEYS = (
    "protocol_version",
    "dataset_version",
    "matching_policy_version",
    "candidate_limit",
    "llm_call_budget",
    "tool_call_budget",
    "token_budget",
    "code_revision",
    "model_revision",
    "prompt_revision",
)


def build_event_evaluators(
    policy: EventMatchPolicy,
    *,
    candidate_limit: int,
    llm_call_budget: int,
    tool_call_budget: int,
    token_budget: int,
) -> tuple[Callable[..., list[dict[str, Any]]], Callable[..., list[dict[str, Any]]], Callable[..., list[dict[str, Any]]]]:
    """Return candidate, final, and experiment-summary evaluators for LangSmith.

    The matching policy is injected and stays application-owned. Gold events
    must be stored in Dataset Example outputs; target output must include the
    case_id and both candidate/final event arrays.
    """

    controls = {
        "candidate_limit": candidate_limit,
        "llm_call_budget": llm_call_budget,
        "tool_call_budget": tool_call_budget,
        "token_budget": token_budget,
    }
    for name, value in controls.items():
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")

    def candidate_event_evaluator(
        inputs: Mapping[str, Any],
        outputs: Mapping[str, Any],
        reference_outputs: Mapping[str, Any],
    ) -> list[dict[str, Any]]:
        result = _score_one("candidate_events", inputs, outputs, reference_outputs, policy, controls)
        return _scores("candidate", result)

    def final_event_evaluator(
        inputs: Mapping[str, Any],
        outputs: Mapping[str, Any],
        reference_outputs: Mapping[str, Any],
    ) -> list[dict[str, Any]]:
        result = _score_one("final_events", inputs, outputs, reference_outputs, policy, controls)
        return _scores("final", result)

    def event_count_summary(runs: Sequence[Any], examples: Sequence[Any]) -> dict[str, list[dict[str, Any]]]:
        if len(runs) != len(examples):
            raise ValueError("LangSmith summary requires one reference Example per run")
        candidate_results = []
        final_results = []
        for run, example in zip(runs, examples, strict=True):
            outputs = getattr(run, "outputs", None) or {}
            inputs = getattr(example, "inputs", None) or {}
            reference = getattr(example, "outputs", None) or {}
            candidate_results.append(_score_one("candidate_events", inputs, outputs, reference, policy, controls))
            final_results.append(_score_one("final_events", inputs, outputs, reference, policy, controls))
        candidate = aggregate_event_evaluations(candidate_results)
        final = aggregate_event_evaluations(final_results)
        return {
            "results": [
                *_scores("candidate", candidate),
                *_scores("final", final),
                {"key": "evaluation_run_count", "score": len(runs)},
            ]
        }

    candidate_event_evaluator.__name__ = "candidate_event_evaluator"
    final_event_evaluator.__name__ = "final_event_evaluator"
    event_count_summary.__name__ = "event_count_summary"
    return candidate_event_evaluator, final_event_evaluator, event_count_summary


def run_langsmith_event_experiment(
    client: Any,
    target: Callable[[dict[str, Any]], dict[str, Any]],
    *,
    data: Any,
    policy: EventMatchPolicy,
    experiment_prefix: str,
    metadata: Mapping[str, Any],
    upload_results: bool,
    num_repetitions: int = 1,
) -> Any:
    """Run an event experiment without silently widening concurrency or policy."""
    if not isinstance(experiment_prefix, str) or not experiment_prefix.strip():
        raise ValueError("experiment_prefix must be non-empty")
    if not isinstance(num_repetitions, int) or isinstance(num_repetitions, bool) or num_repetitions <= 0:
        raise ValueError("num_repetitions must be a positive integer")
    required_metadata = {*_FROZEN_METADATA_KEYS, "strategy_version"}
    missing = sorted(required_metadata - set(metadata))
    if missing:
        raise ValueError(f"experiment metadata is missing frozen controls: {missing}")
    dataset_name = _dataset_version(data)
    if not isinstance(metadata["dataset_version"], str) or not metadata["dataset_version"].strip():
        raise ValueError("dataset_version must be a non-empty frozen identifier")
    if dataset_name is not None and metadata.get("dataset_version") != dataset_name:
        raise ValueError("dataset_version metadata must match the Dataset name or explicit version")
    candidate, final, summary = build_event_evaluators(
        policy,
        candidate_limit=metadata["candidate_limit"],
        llm_call_budget=metadata["llm_call_budget"],
        tool_call_budget=metadata["tool_call_budget"],
        token_budget=metadata["token_budget"],
    )
    return client.evaluate(
        target,
        data=data,
        evaluators=[candidate, final],
        summary_evaluators=[summary],
        experiment_prefix=experiment_prefix,
        metadata=dict(metadata),
        num_repetitions=num_repetitions,
        max_concurrency=1,
        upload_results=upload_results,
    )


def _score_one(
    event_key: str,
    inputs: Mapping[str, Any],
    outputs: Mapping[str, Any],
    reference_outputs: Mapping[str, Any],
    policy: EventMatchPolicy,
    controls: Mapping[str, int],
) -> Any:
    case_id = inputs.get("case_id")
    if not isinstance(case_id, str) or not case_id:
        raise ValueError("Dataset inputs must contain case_id")
    if outputs.get("case_id") != case_id:
        raise ValueError("Target output case_id must match the Dataset input case_id")
    predicted = outputs.get(event_key)
    gold = reference_outputs.get("gold_events")
    if not isinstance(predicted, Sequence) or isinstance(predicted, (str, bytes)):
        raise ValueError(f"Target output must contain {event_key} as an event sequence")
    if not isinstance(gold, Sequence) or isinstance(gold, (str, bytes)):
        raise ValueError("Dataset reference outputs must contain gold_events as an event sequence")
    limits = outputs.get("limits")
    expected_limits = dict(controls)
    if limits != expected_limits:
        raise ValueError("Target output limits must exactly match the frozen experiment budgets")
    usage = outputs.get("usage")
    if not isinstance(usage, Mapping):
        raise ValueError("Target output must include actual usage counts")
    for name in ("llm_call_budget", "tool_call_budget", "token_budget"):
        usage_key = {"llm_call_budget": "llm_calls", "tool_call_budget": "tool_calls", "token_budget": "reserved_tokens"}[name]
        amount = usage.get(usage_key)
        if not isinstance(amount, int) or isinstance(amount, bool) or amount < 0:
            raise ValueError(f"Target usage {usage_key} must be a non-negative integer")
        if amount > controls[name]:
            raise ValueError(f"Target exceeded the frozen {name}")
    if event_key == "candidate_events" and len(predicted) > controls["candidate_limit"]:
        raise ValueError("Target exceeded the frozen candidate_limit")
    return evaluate_events(predicted, gold, policy)


def compare_experiment_metadata(left: Mapping[str, Any], right: Mapping[str, Any]) -> None:
    """Reject a version comparison when any frozen dataset or budget differs."""
    for label, metadata in (("left", left), ("right", right)):
        missing = sorted(set(_FROZEN_METADATA_KEYS) - set(metadata))
        if missing:
            raise ValueError(f"{label} experiment metadata is missing frozen controls: {missing}")
    differences = {
        key: (left.get(key), right.get(key))
        for key in _FROZEN_METADATA_KEYS
        if left.get(key) != right.get(key)
    }
    if differences:
        raise ValueError(f"frozen experiment controls differ: {differences}")


def _dataset_version(data: Any) -> str | None:
    return data if isinstance(data, str) else None


def _scores(prefix: str, result: Any) -> list[dict[str, Any]]:
    get_value = result.get if isinstance(result, Mapping) else lambda key: getattr(result, key)
    values = [
        {"key": f"{prefix}_tp", "score": get_value("tp")},
        {"key": f"{prefix}_fp", "score": get_value("fp")},
        {"key": f"{prefix}_fn", "score": get_value("fn")},
    ]
    precision = get_value("precision")
    recall = get_value("recall")
    if precision is not None:
        values.append({"key": f"{prefix}_precision", "score": precision})
    if recall is not None:
        values.append({"key": f"{prefix}_recall", "score": recall})
    return values

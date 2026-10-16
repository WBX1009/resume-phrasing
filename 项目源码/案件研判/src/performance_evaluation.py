"""Strict, small-sample reporting for paired case-level end-to-end runs."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
import math
from statistics import median


_STATUSES = {"success", "timeout", "failed", "cancelled", "incomplete"}


@dataclass(frozen=True)
class PerformanceRun:
    run_id: str
    case_id: str
    repetition: int
    configuration: str
    status: str
    duration_seconds: float
    control_fingerprint: str

    def __post_init__(self) -> None:
        for name in ("run_id", "case_id", "configuration", "control_fingerprint"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")
        if not isinstance(self.repetition, int) or isinstance(self.repetition, bool) or self.repetition <= 0:
            raise ValueError("repetition must be a positive integer")
        if self.status not in _STATUSES:
            raise ValueError(f"status must be one of {sorted(_STATUSES)}")
        if not isinstance(self.duration_seconds, (int, float)) or isinstance(self.duration_seconds, bool):
            raise ValueError("duration_seconds must be numeric")
        if not math.isfinite(self.duration_seconds) or self.duration_seconds <= 0:
            raise ValueError("duration_seconds must be finite and positive")


def nearest_rank(values: Sequence[float], quantile: float) -> float | None:
    """Return the nearest-rank empirical quantile (rank = ceil(q*n))."""
    if not 0 < quantile <= 1:
        raise ValueError("quantile must be in (0, 1]")
    if not values:
        return None
    ordered = sorted(values)
    rank = math.ceil(quantile * len(ordered))
    return float(ordered[rank - 1])


def summarize_performance(
    runs: Sequence[PerformanceRun],
    *,
    configuration: str,
    expected_case_ids: Sequence[str],
    repetitions_per_case: int,
) -> dict[str, object]:
    """Report raw-run tail and per-case medians without hiding failures."""
    _validate_expected_cases(expected_case_ids, repetitions_per_case)
    selected = [run for run in runs if run.configuration == configuration]
    _validate_unique_runs(selected)
    grouped: dict[str, list[PerformanceRun]] = defaultdict(list)
    for run in selected:
        grouped[run.case_id].append(run)
    unknown = sorted(set(grouped) - set(expected_case_ids))
    missing = sorted(set(expected_case_ids) - set(grouped))
    if unknown or missing:
        raise ValueError(f"case coverage mismatch; missing={missing}, unknown={unknown}")
    for case_id in expected_case_ids:
        case_runs = grouped[case_id]
        repetitions = sorted(run.repetition for run in case_runs)
        expected_repetitions = list(range(1, repetitions_per_case + 1))
        if repetitions != expected_repetitions:
            raise ValueError(f"case {case_id} must have exactly one record for every planned repetition")

    successful = [run.duration_seconds for run in selected if run.status == "success"]
    per_case: dict[str, dict[str, object]] = {}
    for case_id in expected_case_ids:
        case_runs = grouped[case_id]
        case_success = [run.duration_seconds for run in case_runs if run.status == "success"]
        value = median(case_success) if case_success else None
        per_case[case_id] = {
            "planned_runs": repetitions_per_case,
            "successful_runs": len(case_success),
            "failed_runs": len(case_runs) - len(case_success),
            "median_success_seconds": float(value) if value is not None else None,
        }
    statuses = {status: sum(run.status == status for run in selected) for status in sorted(_STATUSES)}
    return {
        "configuration": configuration,
        "cases": len(expected_case_ids),
        "planned_runs": len(expected_case_ids) * repetitions_per_case,
        "successful_runs": len(successful),
        "success_rate": len(successful) / (len(expected_case_ids) * repetitions_per_case),
        "status_counts": statuses,
        "run_p50_seconds": nearest_rank(successful, 0.50),
        "run_p95_seconds": nearest_rank(successful, 0.95),
        "p95_method": "nearest_rank",
        "per_case": per_case,
        "all_runs_successful": len(successful) == len(selected),
    }


def compare_paired_configurations(
    runs: Sequence[PerformanceRun],
    *,
    baseline: str,
    treatment: str,
    expected_case_ids: Sequence[str],
    repetitions_per_case: int,
) -> dict[str, object]:
    """Compare only matched case/repetition runs under the same frozen controls."""
    baseline_runs = [run for run in runs if run.configuration == baseline]
    treatment_runs = [run for run in runs if run.configuration == treatment]
    left = summarize_performance(
        baseline_runs,
        configuration=baseline,
        expected_case_ids=expected_case_ids,
        repetitions_per_case=repetitions_per_case,
    )
    right = summarize_performance(
        treatment_runs,
        configuration=treatment,
        expected_case_ids=expected_case_ids,
        repetitions_per_case=repetitions_per_case,
    )
    by_key_left = {(run.case_id, run.repetition): run for run in baseline_runs}
    by_key_right = {(run.case_id, run.repetition): run for run in treatment_runs}
    if by_key_left.keys() != by_key_right.keys():
        raise ValueError("baseline and treatment must contain the same case/repetition pairs")
    for key in by_key_left:
        if by_key_left[key].control_fingerprint != by_key_right[key].control_fingerprint:
            raise ValueError(f"frozen controls differ for case/repetition {key}")

    case_changes: dict[str, dict[str, object]] = {}
    reductions: list[float] = []
    for case_id in expected_case_ids:
        base_values = [
            by_key_left[(case_id, repetition)].duration_seconds
            for repetition in range(1, repetitions_per_case + 1)
            if by_key_left[(case_id, repetition)].status == "success"
        ]
        treatment_values = [
            by_key_right[(case_id, repetition)].duration_seconds
            for repetition in range(1, repetitions_per_case + 1)
            if by_key_right[(case_id, repetition)].status == "success"
        ]
        if len(base_values) != repetitions_per_case or len(treatment_values) != repetitions_per_case:
            case_changes[case_id] = {"reduction": None, "reason": "incomplete_successful_pairs"}
            continue
        base_median = float(median(base_values))
        treatment_median = float(median(treatment_values))
        reduction = 1 - treatment_median / base_median
        reductions.append(reduction)
        case_changes[case_id] = {
            "baseline_median_seconds": base_median,
            "treatment_median_seconds": treatment_median,
            "reduction": reduction,
        }
    complete = bool(left["all_runs_successful"] and right["all_runs_successful"])
    return {
        "baseline": left,
        "treatment": right,
        "per_case_paired_change": case_changes,
        "median_case_reduction": float(median(reductions)) if reductions else None,
        "min_case_reduction": min(reductions) if reductions else None,
        "max_case_reduction": max(reductions) if reductions else None,
        "cases_faster": sum(value > 0 for value in reductions),
        "cases_with_complete_pairs": len(reductions),
        "performance_improvement_claimable": (
            complete
            and len(reductions) == len(expected_case_ids)
            and median(reductions) > 0
        ),
    }


def _validate_expected_cases(case_ids: Sequence[str], repetitions: int) -> None:
    if not case_ids or any(not isinstance(case_id, str) or not case_id.strip() for case_id in case_ids):
        raise ValueError("expected_case_ids must be a non-empty sequence of non-empty strings")
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("expected_case_ids must be unique")
    if not isinstance(repetitions, int) or isinstance(repetitions, bool) or repetitions <= 0:
        raise ValueError("repetitions_per_case must be a positive integer")


def _validate_unique_runs(runs: Sequence[PerformanceRun]) -> None:
    run_ids = [run.run_id for run in runs]
    if len(run_ids) != len(set(run_ids)):
        raise ValueError("run_id must be unique within a configuration")
    keys = [(run.case_id, run.repetition) for run in runs]
    if len(keys) != len(set(keys)):
        raise ValueError("each case/repetition pair must have exactly one immutable run record")

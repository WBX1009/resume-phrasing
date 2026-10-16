import pytest

from src.performance_evaluation import (
    PerformanceRun,
    compare_paired_configurations,
    nearest_rank,
    summarize_performance,
)


def run(run_id, case, repetition, configuration, seconds, *, status="success", fingerprint="frozen"):
    return PerformanceRun(
        run_id=run_id,
        case_id=case,
        repetition=repetition,
        configuration=configuration,
        status=status,
        duration_seconds=seconds,
        control_fingerprint=fingerprint,
    )


def test_nearest_rank_quantiles_and_empty_values():
    assert nearest_rank([], 0.95) is None
    assert nearest_rank([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 0.95) == 10
    assert nearest_rank(list(range(1, 51)), 0.95) == 48
    with pytest.raises(ValueError, match="quantile"):
        nearest_rank([1], 0)


def test_performance_report_keeps_planned_failure_and_reports_success_p95():
    records = [
        run(f"r-{index}", f"case-{index}", 1, "serial", index, status="failed" if index == 10 else "success")
        for index in range(1, 11)
    ]
    result = summarize_performance(
        records,
        configuration="serial",
        expected_case_ids=[f"case-{index}" for index in range(1, 11)],
        repetitions_per_case=1,
    )
    assert result["planned_runs"] == 10
    assert result["successful_runs"] == 9
    assert result["run_p95_seconds"] == 9
    assert result["all_runs_successful"] is False


def test_paired_comparison_requires_identical_cases_repetitions_and_controls():
    records = []
    for case_number in range(1, 3):
        for repetition in range(1, 3):
            records.append(run(f"s-{case_number}-{repetition}", f"case-{case_number}", repetition, "serial", 100))
            records.append(run(f"p-{case_number}-{repetition}", f"case-{case_number}", repetition, "bounded", 80))
    report = compare_paired_configurations(
        records,
        baseline="serial",
        treatment="bounded",
        expected_case_ids=["case-1", "case-2"],
        repetitions_per_case=2,
    )
    assert report["median_case_reduction"] == pytest.approx(0.2)
    assert report["cases_faster"] == 2
    assert report["performance_improvement_claimable"] is True

    records[-1] = run("p-2-2", "case-2", 2, "bounded", 80, fingerprint="changed-config")
    with pytest.raises(ValueError, match="frozen controls differ"):
        compare_paired_configurations(
            records,
            baseline="serial",
            treatment="bounded",
            expected_case_ids=["case-1", "case-2"],
            repetitions_per_case=2,
        )


def test_incomplete_runs_and_duplicate_measurements_are_not_silently_accepted():
    with pytest.raises(ValueError, match="exactly one record"):
        summarize_performance(
            [run("r1", "case-1", 1, "serial", 10)],
            configuration="serial",
            expected_case_ids=["case-1"],
            repetitions_per_case=2,
        )
    with pytest.raises(ValueError, match="exactly one immutable"):
        summarize_performance(
            [run("r1", "case-1", 1, "serial", 10), run("r2", "case-1", 1, "serial", 11)],
            configuration="serial",
            expected_case_ids=["case-1"],
            repetitions_per_case=1,
        )


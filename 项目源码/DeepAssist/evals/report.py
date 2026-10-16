from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


def generate_report(
    results_dir: Path = Path("evals/results"),
    run_id: str = "latest",
    output_path: Path = Path("docs/evaluation_report.md"),
) -> Dict[str, Any]:
    resolved_run_id = resolve_run_id(results_dir, run_id)
    metrics_path = results_dir / resolved_run_id / "metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing metrics file: {metrics_path}")

    report = json.loads(metrics_path.read_text(encoding="utf-8"))
    markdown = render_markdown(report)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(markdown, encoding="utf-8")
    return {
        "status": "success",
        "run_id": resolved_run_id,
        "metrics_path": str(metrics_path.as_posix()),
        "output_path": str(output_path.as_posix()),
    }


def resolve_run_id(results_dir: Path, run_id: str) -> str:
    if run_id != "latest":
        return run_id
    latest_path = results_dir / "latest_run.txt"
    if not latest_path.exists():
        raise FileNotFoundError(
            f"{latest_path} does not exist. Run evals/run_eval.py before report generation."
        )
    return latest_path.read_text(encoding="utf-8").strip()


def render_markdown(report: Dict[str, Any]) -> str:
    metrics = report.get("metrics", {}) or {}
    suites = report.get("suites", {}) or {}
    config = report.get("config_snapshot", {}) or {}
    processed_dir = str(report.get("processed_dir", "unknown"))
    smoke_fixture_run = "evals/fixtures" in processed_dir.replace("\\", "/")
    lines: List[str] = [
        "# DeepAssist Evaluation Report",
        "",
        "> This report is generated from local eval artifacts. Only numbers in this file should be copied into README or resume bullets.",
        "",
        "## Run Summary",
        "",
        f"- Run ID: `{report.get('run_id', 'unknown')}`",
        f"- Mode: `{report.get('mode', 'unknown')}`",
        f"- Status: `{report.get('status', 'unknown')}`",
        f"- Processed dir: `{processed_dir}`",
        f"- Generated at: `{datetime.now(timezone.utc).isoformat()}`",
        f"- Live backend eval: `{report.get('live_adapter_eval_executed', False)}`",
        "",
    ]
    if smoke_fixture_run:
        lines.extend(
            [
                "## Dataset Warning",
                "",
                "This run used the checked-in smoke fixture, not downloaded public benchmark data. Treat the metrics below as pipeline validation only.",
                "",
            ]
        )
    lines.extend(
        [
        "## Aggregate Metrics",
        "",
        "| Metric | Value |",
        "|---|---:|",
        ]
    )
    for name, value in sorted(metrics.items()):
        lines.append(f"| `{name}` | {value} |")

    lines.extend(["", "## Suite Details", ""])
    for suite_name, suite in suites.items():
        lines.extend(
            [
                f"### {suite_name}",
                "",
                f"- Evaluated: `{suite.get('evaluated', 0)}`",
                f"- Passed: `{suite.get('passed', 0)}`",
                f"- Pass rate: `{suite.get('pass_rate', 0.0)}`",
                "",
                "| Metric | Value |",
                "|---|---:|",
            ]
        )
        for metric_name, value in sorted((suite.get("metrics") or {}).items()):
            lines.append(f"| `{metric_name}` | {value} |")
        lines.append("")

    failures = report.get("failures") or []
    lines.extend(["## Failure Sample", ""])
    if not failures:
        lines.append("No failed cases in this run.")
    else:
        lines.extend(["| Suite | Case ID | Reason |", "|---|---|---|"])
        for item in failures[:20]:
            reason = item.get("reason") or "metric_threshold_not_met"
            lines.append(f"| {item.get('suite', '')} | `{item.get('id', '')}` | {reason} |")

    lines.extend(
        [
            "",
            "## Config Snapshot",
            "",
            "```json",
            json.dumps(config, ensure_ascii=False, indent=2),
            "```",
            "",
            "## Resume-safe Summary",
            "",
            "Use this section only after running public benchmark data, not just smoke fixtures.",
            "",
            "- Retrieval: report Recall@k, MRR@10, nDCG@10, and reranker delta.",
            "- RAG: report EM/F1, citation presence, and source hit rate.",
            "- Agent: report route accuracy, tool choice accuracy, tool execution success, final answer accuracy, and HITL resume success.",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate DeepAssist evaluation Markdown report.")
    parser.add_argument("--results-dir", default="evals/results")
    parser.add_argument("--run-id", default="latest")
    parser.add_argument("--output", default="docs/evaluation_report.md")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = generate_report(
        results_dir=Path(args.results_dir),
        run_id=args.run_id,
        output_path=Path(args.output),
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

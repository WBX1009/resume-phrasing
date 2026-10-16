"""Repeat release checks until the executable wall-clock gate opens.

The watcher is intentionally side-effect-limited: it only creates ignored
quality-run/watch artifacts and never edits goal state or source files. It is
useful when the desktop task remains open overnight; ``--once`` is the local
smoke-test mode.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any


ROOT_DIR = Path(__file__).resolve().parents[1]
STATE_PATH = ROOT_DIR / "docs" / "execution" / "goal_state.json"
DEFAULT_LOG = ROOT_DIR / "output" / "goal_watch" / "watch.jsonl"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Repeat release checks until the goal gate opens")
    parser.add_argument("--interval-minutes", type=float, default=30.0)
    parser.add_argument("--log", type=Path, default=DEFAULT_LOG)
    parser.add_argument("--once", action="store_true", help="run one check cycle and exit")
    return parser


def _run(label: str, args: list[str]) -> dict[str, Any]:
    started = time.perf_counter()
    completed = subprocess.run(
        [sys.executable, *args],
        cwd=ROOT_DIR,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    output = completed.stdout.strip()
    parsed: Any = None
    if output:
        try:
            parsed = json.loads(output)
        except json.JSONDecodeError:
            parsed = None
    return {
        "label": label,
        "returncode": completed.returncode,
        "elapsed_seconds": round(time.perf_counter() - started, 4),
        "json": parsed,
        "tail": output[-1200:],
    }


def _cycle() -> dict[str, Any]:
    checks = [
        _run("pytest", ["-m", "pytest", "-q"]),
        _run("catalog", ["-m", "src.quality_cli", "catalog", "validate"]),
        _run("evaluate", ["-m", "src.quality_cli", "evaluate", "--output-root", "output/quality_runs"]),
        _run("benchmark", ["scripts/benchmark_quality_pipeline.py", "--runs", "10", "--threshold", "1.3"]),
        _run("fuzz", ["scripts/fuzz_quality_pipeline.py", "--rounds", "100", "--seed", "20260907"]),
        _run("goal_gate", ["scripts/check_goal_gate.py", "--state", str(STATE_PATH)]),
    ]
    gate = next((item["json"] for item in checks if item["label"] == "goal_gate" and isinstance(item["json"], dict)), {})
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "checks": checks,
        "can_complete": bool(gate.get("can_complete")),
        "blocked_reasons": gate.get("blocked_reasons", []),
    }


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.interval_minutes <= 0:
        raise SystemExit("--interval-minutes must be positive")
    args.log.parent.mkdir(parents=True, exist_ok=True)
    while True:
        result = _cycle()
        with args.log.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")
        if args.once or result["can_complete"]:
            return 0 if all(item["returncode"] == 0 for item in result["checks"][:-1]) else 1
        time.sleep(args.interval_minutes * 60)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

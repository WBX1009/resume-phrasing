"""Repeat the offline pipeline and enforce a wall-clock performance budget."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys
import tempfile
import time


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.quality_engine import QualityEngine
from src.quality_eval import replay_fingerprint
from src.quality_registry import DEFAULT_DATA_DIR


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark the offline evidence pipeline")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--threshold", type=float, default=1.3, help="maximum allowed run time in seconds")
    parser.add_argument("--output-root", type=Path, default=None, help="keep per-run artifacts under this directory")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.runs < 1:
        raise SystemExit("--runs must be positive")
    if args.threshold <= 0:
        raise SystemExit("--threshold must be positive")

    temporary_root: tempfile.TemporaryDirectory[str] | None = None
    if args.output_root is None:
        temporary_root = tempfile.TemporaryDirectory(prefix="model_insight_benchmark_")
        root = Path(temporary_root.name)
    else:
        root = args.output_root
        root.mkdir(parents=True, exist_ok=True)

    elapsed: list[float] = []
    run_dirs: list[str] = []
    fingerprints: list[str] = []
    try:
        for index in range(args.runs):
            run_root = root / f"run_{index:02d}"
            started = time.perf_counter()
            report, run_dir = QualityEngine(data_dir=args.data_dir, output_root=run_root).run()
            seconds = time.perf_counter() - started
            if report["manifest"]["status"] != "success":
                print(json.dumps({"valid": False, "reason": "pipeline_failed", "run_dir": str(run_dir)}, ensure_ascii=False))
                return 1
            elapsed.append(seconds)
            run_dirs.append(str(run_dir))
            fingerprints.append(replay_fingerprint(report))
    finally:
        if temporary_root is not None:
            temporary_root.cleanup()

    ordered = sorted(elapsed)
    p95_index = min(len(ordered) - 1, max(0, int(len(ordered) * 0.95) - 1))
    result = {
        "valid": max(elapsed) <= args.threshold and len(set(fingerprints)) == 1,
        "runs": args.runs,
        "threshold_seconds": args.threshold,
        "mean_seconds": round(statistics.mean(elapsed), 4),
        "p95_seconds": round(ordered[p95_index], 4),
        "max_seconds": round(max(elapsed), 4),
        "samples_seconds": [round(value, 4) for value in elapsed],
        "fingerprint_stable": len(set(fingerprints)) == 1,
        "replay_fingerprint": fingerprints[0] if fingerprints else None,
        "run_dirs": run_dirs if args.output_root is not None else [],
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["valid"] else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

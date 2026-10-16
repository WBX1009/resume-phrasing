"""Run deterministic malformed-input fuzzing across every model profile."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.quality_operators import compute_facts
from src.quality_reasoning import ToolRegistry, build_claims, run_policy, validate_claims
from src.quality_registry import DEFAULT_DATA_DIR, build_registry


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fuzz the evidence pipeline with malformed rows")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--rounds", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260907)
    return parser


def _make_rows(rng: random.Random) -> list[Any]:
    values: list[Any] = [None, "", "bad", 0, 1, -3.5, "2024-01-01", [], {}, {"nested": ["x", None]}]
    rows: list[Any] = []
    for row_id in range(rng.randint(0, 12)):
        shape = rng.randrange(5)
        if shape == 0:
            rows.append(None)
        elif shape == 1:
            rows.append(rng.choice(["text", 17, [], {}]))
        elif shape == 2:
            rows.append({"row_id": row_id, "data": rng.choice(values)})
        elif shape == 3:
            rows.append(
                {
                    "row_id": row_id,
                    "data": {rng.choice(["金额", "时间", "手机号", "联系人", "状态"]): rng.choice(values)},
                }
            )
        else:
            fields = ["金额", "时间", "手机号", "联系人"]
            rows.append(
                {
                    "row_id": row_id,
                    "data": {key: rng.choice(values) for key in rng.sample(fields, rng.randint(0, len(fields)))},
                }
            )
    return rows


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.rounds < 1:
        raise SystemExit("--rounds must be positive")
    profiles, _ = build_registry(args.data_dir)
    rng = random.Random(args.seed)
    checked = 0
    try:
        for round_id in range(args.rounds):
            rows = _make_rows(rng)
            for profile in profiles:
                facts = compute_facts(profile, rows, f"fuzz-{round_id}")
                observations, _ = run_policy(profile, rows, facts, f"fuzz-{round_id}", ToolRegistry(), 3)
                result = validate_claims(build_claims(profile, facts, observations), facts)
                if not result["valid"]:
                    print(
                        json.dumps(
                            {"valid": False, "round": round_id, "model_id": profile.model_id, "errors": result["errors"]},
                            ensure_ascii=False,
                            indent=2,
                        )
                    )
                    return 1
                checked += 1
    except Exception as exc:  # fuzz harness must report the first crashing input
        print(
            json.dumps(
                {"valid": False, "rounds": args.rounds, "seed": args.seed, "error": f"{type(exc).__name__}: {exc}"},
                ensure_ascii=False,
                indent=2,
            )
        )
        return 1

    print(
        json.dumps(
            {"valid": True, "rounds": args.rounds, "seed": args.seed, "profiles": len(profiles), "profile_runs": checked, "status": "passed"},
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

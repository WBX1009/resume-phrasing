"""Executable release gate for the long-running Codex goal."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path


REQUIRED_ACCEPTANCE = (
    "tests",
    "model_coverage",
    "evidence_integrity",
    "fuzz_harness",
    "report_formats",
    "performance",
    "context_packing",
    "resume",
    "resume_content",
    "report_visual_qa",
    "repository_hygiene",
    "claim_identity",
    "llm_cache",
    "artifact_consistency",
    "report_contract",
    "projection_integrity",
    "model_set_contract",
    "docx_integrity",
    "benchmark_replay",
)
DEFAULT_MAX_CALLS = 200
DEFAULT_MAX_TOKENS = 2_000_000
MIN_WALL_CLOCK_HOURS = 24


def check_gate(state_path: Path) -> dict[str, object]:
    state = json.loads(state_path.read_text(encoding="utf-8"))
    now = datetime.now(timezone.utc)
    try:
        started = datetime.fromisoformat(state["started_at"]).astimezone(timezone.utc)
        minimum = datetime.fromisoformat(state["minimum_completion_at"]).astimezone(timezone.utc)
    except (KeyError, TypeError, ValueError):
        started = now
        minimum = now
        time_gate_error = True
    else:
        time_gate_error = minimum < started + timedelta(hours=MIN_WALL_CLOCK_HOURS)
    pending = [
        key for key in REQUIRED_ACCEPTANCE
        if not str(state.get("acceptance", {}).get(key, "pending")).startswith("passed")
    ]
    blocked_reasons: list[str] = []
    if time_gate_error:
        blocked_reasons.append("invalid_time_gate")
    elif now < minimum:
        blocked_reasons.append("blocked_time_gate")
    if pending:
        blocked_reasons.append("pending_acceptance:" + ",".join(pending))
    if state.get("status") != "active":
        blocked_reasons.append("goal_not_active")
    budget = state.get("online_budget")
    if not isinstance(budget, dict):
        blocked_reasons.append("invalid_online_budget")
    else:
        max_calls = budget.get("max_calls", DEFAULT_MAX_CALLS)
        max_tokens = budget.get("max_tokens", DEFAULT_MAX_TOKENS)
        calls_used = budget.get("calls_used", 0)
        tokens_used = budget.get("tokens_used", 0)
        integers = {
            "max_calls": max_calls,
            "max_tokens": max_tokens,
            "calls_used": calls_used,
            "tokens_used": tokens_used,
        }
        invalid_fields = [
            name for name, value in integers.items()
            if not isinstance(value, int) or isinstance(value, bool) or value < 0
        ]
        if invalid_fields:
            blocked_reasons.append("invalid_online_budget:" + ",".join(invalid_fields))
        else:
            if max_calls > DEFAULT_MAX_CALLS or calls_used > DEFAULT_MAX_CALLS or calls_used > max_calls:
                blocked_reasons.append("online_call_budget_exceeded")
            if max_tokens > DEFAULT_MAX_TOKENS or tokens_used > DEFAULT_MAX_TOKENS or tokens_used > max_tokens:
                blocked_reasons.append("online_token_budget_exceeded")
    return {
        "can_complete": not blocked_reasons,
        "now": now.isoformat(),
        "minimum_completion_at": minimum.isoformat(),
        "elapsed_hours": round((now - started).total_seconds() / 3600, 4),
        "blocked_reasons": blocked_reasons,
        "acceptance": state.get("acceptance", {}),
        "online_budget": state.get("online_budget", {}),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Check the 24-hour Codex goal release gate")
    parser.add_argument("--state", type=Path, default=Path("docs/execution/goal_state.json"))
    args = parser.parse_args()
    result = check_gate(args.state)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["can_complete"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

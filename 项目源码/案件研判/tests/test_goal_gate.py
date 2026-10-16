import json
from datetime import datetime, timedelta, timezone

from scripts.check_goal_gate import check_gate


def test_goal_gate_blocks_before_minimum_time(tmp_path):
    started = datetime.now(timezone.utc)
    state = {
        "status": "active",
        "started_at": started.isoformat(),
        "minimum_completion_at": (started + timedelta(hours=24)).isoformat(),
        "acceptance": {key: "passed" for key in (
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
        )},
        "online_budget": {"calls_used": 1, "tokens_used": 3043},
    }
    path = tmp_path / "goal_state.json"
    path.write_text(json.dumps(state), encoding="utf-8")
    result = check_gate(path)
    assert result["can_complete"] is False
    assert "blocked_time_gate" in result["blocked_reasons"]


def test_goal_gate_blocks_missing_extended_acceptance(tmp_path):
    started = datetime.now(timezone.utc) - timedelta(hours=25)
    state = {
        "status": "active",
        "started_at": started.isoformat(),
        "minimum_completion_at": (started + timedelta(hours=24)).isoformat(),
        "acceptance": {"tests": "passed"},
        "online_budget": {"calls_used": 0, "tokens_used": 0},
    }
    path = tmp_path / "goal_state.json"
    path.write_text(json.dumps(state), encoding="utf-8")
    result = check_gate(path)
    assert result["can_complete"] is False
    assert any(reason.startswith("pending_acceptance:") for reason in result["blocked_reasons"])


def test_goal_gate_blocks_failed_acceptance_value(tmp_path):
    started = datetime.now(timezone.utc) - timedelta(hours=25)
    state = {
        "status": "active",
        "started_at": started.isoformat(),
        "minimum_completion_at": (started + timedelta(hours=24)).isoformat(),
        "acceptance": {key: "passed" for key in (
            "tests", "model_coverage", "evidence_integrity", "fuzz_harness", "report_formats", "performance",
            "context_packing", "resume", "resume_content", "report_visual_qa", "repository_hygiene", "claim_identity",
            "llm_cache", "artifact_consistency", "report_contract", "projection_integrity",
            "model_set_contract", "docx_integrity", "benchmark_replay",
        )},
        "online_budget": {"calls_used": 0, "tokens_used": 0},
    }
    state["acceptance"]["report_contract"] = "failed-contract"
    path = tmp_path / "goal_state.json"
    path.write_text(json.dumps(state), encoding="utf-8")
    result = check_gate(path)
    assert result["can_complete"] is False
    assert "pending_acceptance:report_contract" in result["blocked_reasons"]

    state["acceptance"]["report_contract"] = "passed-contract"
    state["minimum_completion_at"] = (started + timedelta(hours=1)).isoformat()
    path.write_text(json.dumps(state), encoding="utf-8")
    early_time_result = check_gate(path)
    assert early_time_result["can_complete"] is False
    assert "invalid_time_gate" in early_time_result["blocked_reasons"]

    state["minimum_completion_at"] = (started + timedelta(hours=24)).isoformat()
    state["online_budget"] = {"max_calls": 1000, "calls_used": 201, "tokens_used": 0}
    path.write_text(json.dumps(state), encoding="utf-8")
    call_budget_result = check_gate(path)
    assert call_budget_result["can_complete"] is False
    assert "online_call_budget_exceeded" in call_budget_result["blocked_reasons"]

    state["online_budget"] = {"max_tokens": 3_000_000, "calls_used": 0, "tokens_used": 2_000_001}
    path.write_text(json.dumps(state), encoding="utf-8")
    token_budget_result = check_gate(path)
    assert token_budget_result["can_complete"] is False
    assert "online_token_budget_exceeded" in token_budget_result["blocked_reasons"]

    state["online_budget"] = {"max_calls": 200, "max_tokens": 2_000_000, "calls_used": 0, "tokens_used": 0}
    path.write_text(json.dumps(state), encoding="utf-8")
    valid_result = check_gate(path)
    assert valid_result["can_complete"] is True

    state["started_at"] = "not-a-timestamp"
    path.write_text(json.dumps(state), encoding="utf-8")
    malformed_start_result = check_gate(path)
    assert malformed_start_result["can_complete"] is False
    assert "invalid_time_gate" in malformed_start_result["blocked_reasons"]

    state["started_at"] = started.isoformat()
    state["minimum_completion_at"] = "not-a-timestamp"
    path.write_text(json.dumps(state), encoding="utf-8")
    malformed_minimum_result = check_gate(path)
    assert malformed_minimum_result["can_complete"] is False
    assert "invalid_time_gate" in malformed_minimum_result["blocked_reasons"]

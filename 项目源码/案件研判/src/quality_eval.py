"""Deterministic acceptance and fault-injection checks."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from typing import Any

from .quality_models import Claim
from .quality_operators import compute_facts
from .quality_reasoning import ToolRegistry, build_claims, validate_claims
from .quality_registry import build_registry, validate_registry


def validate_report_contract(report: dict[str, Any]) -> dict[str, Any]:
    """Validate the release JSON shape without requiring a third-party schema package."""
    errors: list[str] = []
    is_int = lambda value: isinstance(value, int) and not isinstance(value, bool)
    required = {"schema_version", "manifest", "coverage", "models", "facts", "claims", "validation"}
    missing = sorted(required - set(report))
    errors.extend(f"missing top-level key: {key}" for key in missing)
    if report.get("schema_version") != "1.0":
        errors.append("schema_version must be 1.0")
    manifest = report.get("manifest")
    if not isinstance(manifest, dict):
        errors.append("manifest must be an object")
    else:
        for key in ("run_id", "snapshot_sha256", "profile_count", "sampled_profile_count", "unsupported_profile_count", "online", "status"):
            if key not in manifest:
                errors.append(f"manifest missing key: {key}")
        if not isinstance(manifest.get("run_id"), str) or not manifest.get("run_id"):
            errors.append("manifest.run_id must be a non-empty string")
        snapshot = manifest.get("snapshot_sha256")
        if not isinstance(snapshot, str) or len(snapshot) != 64 or any(char not in "0123456789abcdef" for char in snapshot):
            errors.append("manifest.snapshot_sha256 must be a lowercase sha256")
        for key in ("profile_count", "sampled_profile_count", "unsupported_profile_count"):
            if not is_int(manifest.get(key)) or manifest.get(key) < 0:
                errors.append(f"manifest.{key} must be a non-negative integer")
        if not isinstance(manifest.get("online"), bool):
            errors.append("manifest.online must be boolean")
        if manifest.get("status") not in {"running", "success", "failed"}:
            errors.append("manifest.status is invalid")
    coverage = report.get("coverage")
    if not isinstance(coverage, dict):
        errors.append("coverage must be an object")
    else:
        for key in ("total_models", "sampled_models", "unsupported_no_sample", "valid"):
            if key not in coverage:
                errors.append(f"coverage missing key: {key}")
        for key in ("total_models", "sampled_models", "unsupported_no_sample"):
            if not is_int(coverage.get(key)) or coverage.get(key) < 0:
                errors.append(f"coverage.{key} must be a non-negative integer")
        if not isinstance(coverage.get("valid"), bool):
            errors.append("coverage.valid must be boolean")
    for key in ("models", "facts", "claims"):
        if not isinstance(report.get(key), list):
            errors.append(f"{key} must be an array")
    facts = report.get("facts") if isinstance(report.get("facts"), list) else []
    fact_ids: list[Any] = []
    for index, item in enumerate(facts):
        if not isinstance(item, dict):
            errors.append(f"facts[{index}] must be an object")
            continue
        for key in ("fact_id", "model_id", "case_id", "kind", "subject", "value", "complete"):
            if key not in item:
                errors.append(f"facts[{index}] missing key: {key}")
        fact_id = item.get("fact_id")
        fact_ids.append(fact_id)
        if not isinstance(fact_id, str) or not fact_id:
            errors.append(f"facts[{index}].fact_id must be a non-empty string")
        if not is_int(item.get("model_id")):
            errors.append(f"facts[{index}].model_id must be an integer")
        if not isinstance(item.get("kind"), str) or not item.get("kind"):
            errors.append(f"facts[{index}].kind must be a non-empty string")
        if not isinstance(item.get("subject"), str):
            errors.append(f"facts[{index}].subject must be a string")
        if not isinstance(item.get("complete"), bool):
            errors.append(f"facts[{index}].complete must be boolean")
    valid_fact_ids = [fact_id for fact_id in fact_ids if isinstance(fact_id, str)]
    if len(valid_fact_ids) != len(set(valid_fact_ids)):
        errors.append("fact_id values must be unique")
    known_fact_ids = {fact_id for fact_id in fact_ids if isinstance(fact_id, str) and fact_id}
    for index, item in enumerate(report.get("models") if isinstance(report.get("models"), list) else []):
        if not isinstance(item, dict):
            errors.append(f"models[{index}] must be an object")
            continue
        for key in ("model_id", "name", "strategy_key", "coverage", "reasoning_mode", "status"):
            if key not in item:
                errors.append(f"models[{index}] missing key: {key}")
        if not is_int(item.get("model_id")):
            errors.append(f"models[{index}].model_id must be an integer")
        for key in ("name", "strategy_key", "status"):
            if not isinstance(item.get(key), str):
                errors.append(f"models[{index}].{key} must be a string")
        if item.get("coverage") not in {"sampled", "unsupported_no_sample"}:
            errors.append(f"models[{index}].coverage is invalid")
        if item.get("reasoning_mode") not in {"deterministic", "react", "plan_solve", "plan_react_reflect"}:
            errors.append(f"models[{index}].reasoning_mode is invalid")
    model_rows = report.get("models") if isinstance(report.get("models"), list) else []
    model_ids = [item.get("model_id") for item in model_rows if isinstance(item, dict) and is_int(item.get("model_id"))]
    if len(model_ids) != len(set(model_ids)):
        errors.append("model_id values must be unique")
    known_model_ids = set(model_ids)
    if isinstance(coverage, dict):
        sampled_count = sum(item.get("coverage") == "sampled" for item in model_rows if isinstance(item, dict))
        unsupported_count = sum(item.get("coverage") == "unsupported_no_sample" for item in model_rows if isinstance(item, dict))
        if is_int(coverage.get("total_models")) and coverage["total_models"] != len(model_rows):
            errors.append("coverage.total_models does not match models length")
        if is_int(coverage.get("sampled_models")) and coverage["sampled_models"] != sampled_count:
            errors.append("coverage.sampled_models does not match model rows")
        if is_int(coverage.get("unsupported_no_sample")) and coverage["unsupported_no_sample"] != unsupported_count:
            errors.append("coverage.unsupported_no_sample does not match model rows")
        if isinstance(manifest, dict):
            expected_counts = {
                "profile_count": coverage.get("total_models"),
                "sampled_profile_count": coverage.get("sampled_models"),
                "unsupported_profile_count": coverage.get("unsupported_no_sample"),
            }
            for key, expected in expected_counts.items():
                if is_int(manifest.get(key)) and is_int(expected) and manifest[key] != expected:
                    errors.append(f"manifest.{key} does not match coverage")
    for index, item in enumerate(report.get("claims") if isinstance(report.get("claims"), list) else []):
        if not isinstance(item, dict):
            errors.append(f"claims[{index}] must be an object")
            continue
        for key in ("claim_id", "level", "text", "model_ids", "fact_ids", "status"):
            if key not in item:
                errors.append(f"claims[{index}] missing key: {key}")
        if not isinstance(item.get("claim_id"), str) or not item.get("claim_id"):
            errors.append(f"claims[{index}].claim_id must be a non-empty string")
        if not isinstance(item.get("model_ids"), list) or not all(is_int(model_id) for model_id in item.get("model_ids", [])):
            errors.append(f"claims[{index}].model_ids must be an integer array")
        elif missing_models := sorted(set(item.get("model_ids", [])) - known_model_ids):
            errors.append(f"claims[{index}] references missing model ids {missing_models}")
        if not isinstance(item.get("fact_ids"), list) or not all(isinstance(fact_id, str) for fact_id in item.get("fact_ids", [])):
            errors.append(f"claims[{index}].fact_ids must be a string array")
        elif missing_facts := sorted(set(item.get("fact_ids", [])) - known_fact_ids):
            errors.append(f"claims[{index}] references missing fact ids {missing_facts}")
        if not isinstance(item.get("text"), str):
            errors.append(f"claims[{index}].text must be a string")
        if item.get("level") not in {"fact", "rule_hit", "inference", "unresolved"}:
            errors.append(f"claims[{index}].level is invalid")
        if item.get("status") not in {"accepted", "downgraded", "rejected"}:
            errors.append(f"claims[{index}].status is invalid")
    claim_ids: list[Any] = []
    for item in report.get("claims", []) if isinstance(report.get("claims"), list) else []:
        if not isinstance(item, dict):
            continue
        claim_id = item.get("claim_id")
        if any(claim_id == previous for previous in claim_ids):
            errors.append("claim_id values must be unique")
        claim_ids.append(claim_id)
    validation = report.get("validation")
    if not isinstance(validation, dict) or "claims" not in validation or "registry" not in validation:
        errors.append("validation must contain claims and registry results")
    return {"valid": not errors, "errors": errors}


def replay_fingerprint(report: dict[str, Any]) -> str:
    stable_payload = {
        "coverage": report.get("coverage"),
        "models": report.get("models"),
        "facts": report.get("facts"),
        "observations": report.get("observations"),
        "claims": report.get("claims"),
        "cross_model_links": report.get("cross_model_links"),
        "trace": [
            {"event_type": event.get("event_type"), "step": event.get("step"), "payload": event.get("payload")}
            for event in report.get("trace", [])
        ],
    }
    encoded = json.dumps(stable_payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def run_fault_injection_suite(data_dir) -> dict[str, Any]:
    profiles, rows = build_registry(data_dir)
    sampled = next(profile for profile in profiles if profile.coverage == "sampled")
    sample_rows = [row for row in rows if row["model_id"] == sampled.model_id][:8]
    results: dict[str, dict[str, Any]] = {}

    # Empty, malformed, and incomplete rows must never crash the fact layer.
    for name, mutated in {
        "empty": [],
        "malformed_data": [{**sample_rows[0], "data": None}] if sample_rows else [],
        "wrong_type_data": [{**sample_rows[0], "data": ["not", "a", "mapping"]}] if sample_rows else [],
        "duplicate_rows": sample_rows + deepcopy(sample_rows),
    }.items():
        try:
            facts = compute_facts(sampled, mutated, "mutation")
            claims = build_claims(sampled, facts, [])
            results[name] = {
                "passed": bool(facts) and validate_claims(claims, facts)["valid"],
                "fact_count": len(facts),
                "claim_count": len(claims),
            }
        except Exception as exc:  # the suite should report, not hide, regressions
            results[name] = {"passed": False, "error": f"{type(exc).__name__}: {exc}"}

    unknown = ToolRegistry().execute("unknown_tool", sampled, sample_rows, [], "mutation")
    results["unknown_tool"] = {"passed": unknown.status == "blocked", "status": unknown.status, "reason": unknown.reason}

    orphan = Claim("orphan", "fact", "bad", [sampled.model_id], ["missing_fact"])
    orphan_validation = validate_claims([orphan], [])
    results["orphan_reference"] = {"passed": not orphan_validation["valid"], "errors": orphan_validation["errors"]}

    return {"valid": all(item.get("passed") for item in results.values()), "cases": results}


def acceptance_summary(data_dir) -> dict[str, Any]:
    profiles, _ = build_registry(data_dir)
    registry = validate_registry(profiles)
    faults = run_fault_injection_suite(data_dir)
    sampled = [profile for profile in profiles if profile.coverage == "sampled"]
    strategy_keys = [profile.strategy_key for profile in sampled]
    contract = {
        "valid": (
            len(strategy_keys) == len(set(strategy_keys)) == 61
            and all(profile.operators and profile.report_sections for profile in sampled)
        ),
        "missing_strategy_keys": [profile.model_id for profile in sampled if not profile.strategy_key],
        "duplicate_strategy_keys": sorted({key for key in strategy_keys if strategy_keys.count(key) > 1}),
        "profiles_with_operators": sum(bool(profile.operators) for profile in sampled),
        "profiles_with_report_sections": sum(bool(profile.report_sections) for profile in sampled),
    }
    return {
        "valid": registry["valid"] and registry["sampled_models"] == 61 and registry["unsupported_no_sample"] == 43 and faults["valid"] and contract["valid"],
        "registry": registry,
        "strategy_contract": contract,
        "fault_injection": faults,
    }

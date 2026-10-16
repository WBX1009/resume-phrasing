"""Command line entrypoint for the evidence-first quality pipeline."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Any

from .quality_engine import QualityEngine, _docx, _html, _markdown, load_report
from .quality_eval import acceptance_summary, replay_fingerprint, validate_report_contract
from .quality_registry import DEFAULT_DATA_DIR, build_registry, validate_registry


def _json_dump(payload: Any) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def _docx_document_xml(path: Path) -> bytes:
    with zipfile.ZipFile(path) as archive:
        return archive.read("word/document.xml")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="model-insight", description="Evidence-first model investigation CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    catalog = sub.add_parser("catalog", help="validate the model catalog")
    catalog_sub = catalog.add_subparsers(dest="catalog_command", required=True)
    validate = catalog_sub.add_parser("validate")
    validate.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)

    run = sub.add_parser("run", help="run the offline evidence pipeline")
    run.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    run.add_argument("--output-root", type=Path, default=None)
    run.add_argument("--case-id", type=int, default=None)
    run.add_argument("--max-steps", type=int, default=3)
    run.add_argument("--online", action="store_true", help="reserved for the budgeted LLM adapter")
    run.add_argument("--resume", type=Path, default=None, help="reuse and verify a completed run directory")
    run.add_argument("--model-ids", default=None, help="comma-separated model ids for a focused smoke run")

    evaluate = sub.add_parser("evaluate", help="run coverage and artifact acceptance checks")
    evaluate.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    evaluate.add_argument("--output-root", type=Path, default=None)

    replay = sub.add_parser("replay", help="verify a previous run artifact")
    replay.add_argument("run_dir", type=Path)
    return parser


def _validate_artifacts(report: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    # Replay is a public trust boundary: malformed reports must produce a
    # structured invalid result rather than an uncaught KeyError/TypeError.
    if not isinstance(report, dict):
        report = {}
    required = [
        "report.json",
        "report.md",
        "report.html",
        "report.docx",
        "trace.jsonl",
        "run_manifest.json",
        "checkpoint.json",
        "checkpoint_models.jsonl",
    ]
    missing = [name for name in required if not (run_dir / name).exists()]
    report_models = report.get("models", []) if isinstance(report.get("models", []), list) else []
    sampled_items = [item for item in report_models if isinstance(item, dict) and item.get("coverage") == "sampled"]
    unsupported_items = [item for item in report_models if isinstance(item, dict) and item.get("coverage") == "unsupported_no_sample"]
    processed_sampled = [item for item in sampled_items if item.get("status") == "processed"]
    fact_rows = report.get("facts", []) if isinstance(report.get("facts", []), list) else []
    fact_ids = {fact.get("fact_id") for fact in fact_rows if isinstance(fact, dict)}
    report_trace = report.get("trace", []) if isinstance(report.get("trace", []), list) else []
    context_events = [event for event in report_trace if isinstance(event, dict) and event.get("event_type") == "context_packed"]
    context_ratios: list[float] = []
    context_errors: list[str] = []
    for event in context_events:
        try:
            context_ratios.append(float(event.get("payload", {}).get("compression_ratio", 1.0)))
        except (AttributeError, TypeError, ValueError):
            context_errors.append("context_packed compression_ratio must be numeric")
    context_budget_exceeded = [event for event in context_events if event.get("payload", {}).get("budget_exceeded")]
    context_missing = sorted({fact_id for event in context_events for fact_id in event.get("payload", {}).get("fact_ids", []) if fact_id not in fact_ids})
    context_result = {
        "valid": bool(context_events) and not context_errors and not context_missing and not context_budget_exceeded and len(context_ratios) == len(context_events) and (sum(context_ratios) / len(context_ratios) <= 0.6),
        "events": len(context_events),
        "average_compression_ratio": round(sum(context_ratios) / len(context_ratios), 6) if context_ratios else None,
        "missing_fact_ids": context_missing,
        "budget_exceeded_events": len(context_budget_exceeded),
        "errors": context_errors,
    }
    schema_result = validate_report_contract(report)
    consistency_errors: list[str] = []
    selected_model_ids: set[int] | None = None
    sampled_model_ids: set[int] | None = None
    unsupported_model_ids: set[int] | None = None
    coverage = report.get("coverage") if isinstance(report.get("coverage"), dict) else {}
    raw_sampled_ids = coverage.get("sampled_model_ids") if isinstance(coverage, dict) else None
    raw_unsupported_ids = coverage.get("unsupported_model_ids") if isinstance(coverage, dict) else None
    if isinstance(raw_sampled_ids, list) and all(isinstance(item, int) and not isinstance(item, bool) for item in raw_sampled_ids):
        sampled_model_ids = set(raw_sampled_ids)
        if len(sampled_model_ids) != len(raw_sampled_ids):
            consistency_errors.append("coverage.sampled_model_ids must be unique")
    else:
        consistency_errors.append("coverage.sampled_model_ids must be an integer array")
    if isinstance(raw_unsupported_ids, list) and all(isinstance(item, int) and not isinstance(item, bool) for item in raw_unsupported_ids):
        unsupported_model_ids = set(raw_unsupported_ids)
        if len(unsupported_model_ids) != len(raw_unsupported_ids):
            consistency_errors.append("coverage.unsupported_model_ids must be unique")
    else:
        consistency_errors.append("coverage.unsupported_model_ids must be an integer array")
    if sampled_model_ids is not None and unsupported_model_ids is not None:
        overlap = sorted(sampled_model_ids & unsupported_model_ids)
        if overlap:
            consistency_errors.append(
                "coverage sampled/unsupported model sets overlap: "
                + ", ".join(str(item) for item in overlap)
            )
    report_manifest = report.get("manifest") if isinstance(report.get("manifest"), dict) else {}
    try:
        manifest_file = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
        if not isinstance(manifest_file, dict):
            raise TypeError("manifest must be an object")
        if manifest_file != report_manifest:
            consistency_errors.append("run_manifest.json does not exactly match report manifest")
        if manifest_file.get("run_id") != report_manifest.get("run_id"):
            consistency_errors.append("run_manifest.run_id does not match report manifest")
        if manifest_file.get("snapshot_sha256") != report_manifest.get("snapshot_sha256"):
            consistency_errors.append("run_manifest.snapshot_sha256 does not match report manifest")
        if manifest_file.get("status") != report_manifest.get("status"):
            consistency_errors.append("run_manifest.status does not match report manifest")
    except (OSError, json.JSONDecodeError, AttributeError, TypeError):
        consistency_errors.append("run_manifest.json is not valid JSON")
    try:
        checkpoint = json.loads((run_dir / "checkpoint.json").read_text(encoding="utf-8"))
        if not isinstance(checkpoint, dict):
            raise TypeError("checkpoint must be an object")
        if checkpoint.get("run_id") != report_manifest.get("run_id"):
            consistency_errors.append("checkpoint.run_id does not match report manifest")
        if checkpoint.get("status") != report_manifest.get("status"):
            consistency_errors.append("checkpoint.status does not match report manifest")
        if checkpoint.get("processed_models") != len(report_models):
            consistency_errors.append("checkpoint.processed_models does not match report models")
        checkpoint_manifest = checkpoint.get("manifest")
        if not isinstance(checkpoint_manifest, dict):
            consistency_errors.append("checkpoint.manifest must be an object")
        else:
            # The final checkpoint intentionally omits the derived
            # elapsed_seconds field, but all identity and scope fields must
            # agree with the report's source-of-truth manifest.
            for key in (
                "run_id",
                "snapshot_sha256",
                "profile_count",
                "sampled_profile_count",
                "unsupported_profile_count",
                "case_ids",
                "online",
                "command",
            ):
                if checkpoint_manifest.get(key) != report_manifest.get(key):
                    consistency_errors.append(f"checkpoint.manifest.{key} does not match report manifest")
        raw_processed_ids = checkpoint.get("processed_model_ids")
        if not isinstance(raw_processed_ids, list) or any(
            not isinstance(item, int) or isinstance(item, bool) for item in raw_processed_ids
        ):
            consistency_errors.append("checkpoint.processed_model_ids must be an integer array")
        elif len(raw_processed_ids) != len(set(raw_processed_ids)):
            consistency_errors.append("checkpoint.processed_model_ids must be unique")
        elif set(raw_processed_ids) != {
            item.get("model_id")
            for item in report_models
            if isinstance(item, dict) and isinstance(item.get("model_id"), int) and not isinstance(item.get("model_id"), bool)
        }:
            consistency_errors.append("checkpoint.processed_model_ids set does not match report models")
        raw_selected_ids = checkpoint.get("selected_model_ids")
        if raw_selected_ids is not None:
            if not isinstance(raw_selected_ids, list) or any(
                not isinstance(item, int) or isinstance(item, bool) for item in raw_selected_ids
            ) or len(raw_selected_ids) != len(set(raw_selected_ids)):
                consistency_errors.append("checkpoint.selected_model_ids must be a unique integer array or null")
            else:
                selected_model_ids = set(raw_selected_ids)
                known_model_ids = (sampled_model_ids or set()) | (unsupported_model_ids or set())
                unknown_selected = sorted(selected_model_ids - known_model_ids)
                if unknown_selected:
                    consistency_errors.append(
                        "checkpoint.selected_model_ids contains unknown models: "
                        + ", ".join(str(item) for item in unknown_selected)
                    )
                if sampled_model_ids is not None:
                    selected_sampled = selected_model_ids & sampled_model_ids
                    model_by_id = {
                        item.get("model_id"): item
                        for item in sampled_items
                        if isinstance(item.get("model_id"), int) and not isinstance(item.get("model_id"), bool)
                    }
                    for model_id in sampled_model_ids:
                        item = model_by_id.get(model_id)
                        expected_status = "processed" if model_id in selected_sampled else "not_selected"
                        if item is None or item.get("status") != expected_status:
                            consistency_errors.append(
                                f"sampled model {model_id} status does not match selected_model_ids"
                            )
    except (OSError, json.JSONDecodeError, AttributeError, TypeError):
        consistency_errors.append("checkpoint.json is not valid JSON")
    try:
        trace_lines = [line for line in (run_dir / "trace.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
        trace_events = [json.loads(line) for line in trace_lines]
        trace_ids = [event.get("event_id") for event in trace_events if isinstance(event, dict)]
        report_trace_ids = [event.get("event_id") for event in report_trace if isinstance(event, dict)]
        if len(trace_events) != len(report_trace):
            consistency_errors.append("trace.jsonl line count does not match report trace")
        if len(trace_ids) != len(trace_events) or any(not isinstance(event_id, str) or not event_id for event_id in trace_ids):
            consistency_errors.append("trace.jsonl events must have non-empty event_id values")
        if len(trace_ids) != len(set(trace_ids)):
            consistency_errors.append("trace.jsonl event_id values must be unique")
        if trace_ids != report_trace_ids:
            consistency_errors.append("trace.jsonl event_id sequence does not match report trace")
        if trace_events != report_trace:
            consistency_errors.append("trace.jsonl event payloads do not match report trace")
    except (OSError, json.JSONDecodeError, TypeError):
        consistency_errors.append("trace.jsonl is not valid JSONL")
    for projection in ("report.md", "report.html"):
        try:
            actual_projection = (run_dir / projection).read_text(encoding="utf-8")
            if not all(isinstance(report.get(key), list) for key in ("models", "claims")):
                consistency_errors.append(f"{projection} cannot be regenerated from malformed report")
                continue
            expected_markdown = _markdown(report)
            expected_projection = expected_markdown if projection == "report.md" else _html(report, expected_markdown)
            if actual_projection != expected_projection:
                consistency_errors.append(f"{projection} does not match report source of truth")
        except (OSError, KeyError, TypeError, AttributeError):
            consistency_errors.append(f"{projection} cannot be read")
    try:
        with tempfile.TemporaryDirectory(prefix="model-insight-replay-") as temp_dir:
            expected_docx = Path(temp_dir) / "expected.docx"
            _docx(report, expected_docx)
            if _docx_document_xml(run_dir / "report.docx") != _docx_document_xml(expected_docx):
                consistency_errors.append("report.docx does not match report source of truth")
    except (OSError, KeyError, TypeError, zipfile.BadZipFile, RuntimeError):
        consistency_errors.append("report.docx cannot be regenerated or read")
    try:
        state_lines = [line for line in (run_dir / "checkpoint_models.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
        state_records = [json.loads(line) for line in state_lines]
        state_ids = [int(record["model_id"]) for record in state_records]
        report_model_ids = [item.get("model_id") for item in report_models]
        if len(state_ids) != len(state_records) or any(not isinstance(record, dict) for record in state_records):
            raise TypeError("checkpoint model records must be objects")
        if len(state_ids) != len(set(state_ids)):
            consistency_errors.append("checkpoint_models.jsonl model_id values must be unique")
        if len(state_ids) != len(report_models):
            consistency_errors.append("checkpoint_models.jsonl line count does not match report models")
        if set(state_ids) != set(report_model_ids):
            consistency_errors.append("checkpoint_models.jsonl model_id set does not match report models")
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
        consistency_errors.append("checkpoint_models.jsonl is not valid model JSONL")
    consistency = {"valid": not consistency_errors, "errors": consistency_errors}
    expected_processed_sampled = (
        len(sampled_model_ids)
        if selected_model_ids is None and sampled_model_ids is not None
        else len((selected_model_ids or set()) & (sampled_model_ids or set()))
    )
    return {
        "valid": (
            not missing
            and schema_result["valid"]
            and consistency["valid"]
            and isinstance(report.get("validation"), dict)
            and isinstance(report.get("validation", {}).get("claims"), dict)
            and report.get("validation", {}).get("claims", {}).get("valid") is True
            and len(processed_sampled) == expected_processed_sampled
            and len(unsupported_items) == (report.get("coverage", {}).get("unsupported_no_sample") if isinstance(report.get("coverage"), dict) else None)
            and context_result["valid"]
        ),
        "missing_artifacts": missing,
        "processed_sampled_models": len(processed_sampled),
        "unsupported_models": len(unsupported_items),
        "claim_validation": report.get("validation", {}).get("claims", {}) if isinstance(report.get("validation"), dict) else {},
        "schema_contract": schema_result,
        "artifact_consistency": consistency,
        "context_packing": context_result,
    }


def _structured_report_load_failure(run_dir: Path, exc: Exception) -> dict[str, Any]:
    """Return the same diagnostic shape for an unreadable report as replay validation."""
    result = _validate_artifacts({}, run_dir)
    result["valid"] = False
    result.setdefault("artifact_consistency", {}).setdefault("errors", []).append(
        f"report.json is not valid JSON: {exc}"
    )
    return result


def _command_validate(args: argparse.Namespace) -> int:
    try:
        profiles, _ = build_registry(args.data_dir)
        result = validate_registry(profiles)
    except (OSError, ValueError, RuntimeError, TypeError) as exc:
        _json_dump(
            {
                "valid": False,
                "error": f"{type(exc).__name__}: {exc}",
                "data_dir": str(args.data_dir),
            }
        )
        return 2
    _json_dump(result)
    return 0 if result["valid"] and result["sampled_models"] == 61 and result["unsupported_no_sample"] == 43 else 1


def _command_run(args: argparse.Namespace) -> int:
    model_ids = None
    if args.model_ids:
        try:
            model_ids = {int(item.strip()) for item in args.model_ids.split(",") if item.strip()}
        except ValueError:
            print("--model-ids must be comma-separated integers", file=sys.stderr)
            return 2
    if args.resume is not None:
        checkpoint_path = args.resume / "checkpoint.json"
        try:
            checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            print(f"cannot read resume checkpoint: {checkpoint_path}: {exc}", file=sys.stderr)
            return 2
        checkpoint_manifest = checkpoint.get("manifest") if isinstance(checkpoint, dict) else None
        hint_cases = list(checkpoint_manifest.get("case_ids") or []) if isinstance(checkpoint_manifest, dict) else []
        if args.case_id is not None and hint_cases:
            try:
                case_mismatch = sorted(int(item) for item in hint_cases) != [int(args.case_id)]
            except (TypeError, ValueError):
                case_mismatch = True
            if case_mismatch:
                _json_dump(
                    {
                        "valid": False,
                        "run_dir": str(args.resume),
                        "resumed": False,
                        "error": "resume case filter does not match the checkpoint",
                    }
                )
                return 2
        hint_models = checkpoint.get("selected_model_ids") if isinstance(checkpoint, dict) else None
        if model_ids is not None:
            try:
                hint_model_set = {int(item) for item in hint_models} if hint_models is not None else None
            except (TypeError, ValueError):
                hint_model_set = None
            if hint_model_set != model_ids:
                _json_dump(
                    {
                        "valid": False,
                        "run_dir": str(args.resume),
                        "resumed": False,
                        "error": "resume model filter does not match the checkpoint",
                    }
                )
                return 2
        if checkpoint.get("status") == "success":
            try:
                report = load_report(args.resume)
            except (OSError, json.JSONDecodeError, TypeError, UnicodeError) as exc:
                result = _structured_report_load_failure(args.resume, exc)
                _json_dump({"run_dir": str(args.resume), "manifest": None, "replay": result})
                return 1
            result = _validate_artifacts(report, args.resume)
            manifest = report.get("manifest") if isinstance(report, dict) else None
            _json_dump({"run_dir": str(args.resume), "manifest": manifest, "replay": result})
            return 0 if result["valid"] else 1
        if checkpoint.get("status") != "running":
            print("--resume requires a running or completed checkpoint", file=sys.stderr)
            return 2
        try:
            report, run_dir = QualityEngine(
                data_dir=args.data_dir,
                output_root=args.output_root,
                online=args.online,
                max_steps=max(1, min(args.max_steps, 5)),
            ).run(case_id=args.case_id, model_ids=model_ids, resume_dir=args.resume)
        except Exception as exc:
            _json_dump(
                {
                    "valid": False,
                    "run_dir": str(args.resume),
                    "resumed": False,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            return 2
        _json_dump({"run_dir": str(run_dir), "resumed": True, "manifest": report["manifest"], "validation": report["validation"]})
        return 0 if report["manifest"]["status"] == "success" else 1
    try:
        report, run_dir = QualityEngine(
            data_dir=args.data_dir,
            output_root=args.output_root,
            online=args.online,
            max_steps=max(1, min(args.max_steps, 5)),
        ).run(case_id=args.case_id, model_ids=model_ids)
    except (ValueError, OSError, RuntimeError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    _json_dump({"run_dir": str(run_dir), "manifest": report["manifest"], "validation": report["validation"]})
    return 0 if report["manifest"]["status"] == "success" else 1


def _command_evaluate(args: argparse.Namespace) -> int:
    try:
        report, run_dir = QualityEngine(data_dir=args.data_dir, output_root=args.output_root).run()
        artifact_result = _validate_artifacts(report, run_dir)
        fault_result = acceptance_summary(args.data_dir)
        replay_report, replay_dir = QualityEngine(data_dir=args.data_dir, output_root=args.output_root).run()
        replay_result = {
            "valid": replay_fingerprint(report) == replay_fingerprint(replay_report),
            "fingerprint": replay_fingerprint(report),
            "replay_fingerprint": replay_fingerprint(replay_report),
            "replay_dir": str(replay_dir),
        }
        result = {
            "valid": artifact_result["valid"] and report["coverage"]["total_models"] == 104 and fault_result["valid"] and replay_result["valid"],
            "run_dir": str(run_dir),
            "coverage": report["coverage"],
            "artifacts": artifact_result,
            "fault_injection": fault_result["fault_injection"],
            "replay": replay_result,
            "elapsed_seconds": report["manifest"].get("elapsed_seconds", 0),
        }
    except (OSError, ValueError, RuntimeError, KeyError, TypeError) as exc:
        _json_dump(
            {
                "valid": False,
                "run_dir": None,
                "error": f"{type(exc).__name__}: {exc}",
                "data_dir": str(args.data_dir),
            }
        )
        return 2
    _json_dump(result)
    return 0 if result["valid"] else 1


def _command_replay(args: argparse.Namespace) -> int:
    try:
        report = load_report(args.run_dir)
    except (OSError, json.JSONDecodeError, TypeError, UnicodeError) as exc:
        result = _structured_report_load_failure(args.run_dir, exc)
        result["run_id"] = None
        _json_dump(result)
        return 1
    result = _validate_artifacts(report, args.run_dir)
    manifest = report.get("manifest") if isinstance(report, dict) else None
    result["run_id"] = manifest.get("run_id") if isinstance(manifest, dict) else None
    _json_dump(result)
    return 0 if result["valid"] else 1


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "catalog":
        return _command_validate(args)
    if args.command == "run":
        return _command_run(args)
    if args.command == "evaluate":
        return _command_evaluate(args)
    if args.command == "replay":
        return _command_replay(args)
    raise AssertionError(f"unknown command: {args.command}")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

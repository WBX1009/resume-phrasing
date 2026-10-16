from pathlib import Path
import json
from copy import deepcopy
import subprocess
from docx import Document

import pytest

from src.quality_engine import QualityEngine
from src import quality_engine as quality_engine_module
from src.quality_cli import main as quality_cli_main
from src.quality_eval import acceptance_summary, replay_fingerprint, run_fault_injection_suite, validate_report_contract
from src.quality_cross_model import build_cross_model_links
from src.quality_models import Fact
from src.quality_operators import compute_facts
from src.quality_reasoning import ToolRegistry, build_claims, run_policy, validate_claims
from src.quality_registry import build_registry, load_sample_rows, validate_registry
from scripts.benchmark_quality_pipeline import main as benchmark_main
from scripts.fuzz_quality_pipeline import main as fuzz_main


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "表格数据抽样_ josnl 版"

pytestmark = pytest.mark.skipif(
    not DATA_DIR.exists(),
    reason="Public clean copy excludes the private snapshot; provide an authorized fixture to run quality tests.",
)


def test_registry_has_explicit_61_43_coverage():
    profiles, _ = build_registry(DATA_DIR)
    result = validate_registry(profiles)
    assert result["valid"] is True
    assert result["total_models"] == 104
    assert result["sampled_models"] == 61
    assert result["unsupported_no_sample"] == 43
    assert len(set(result["sampled_model_ids"])) == 61
    assert all(profile.strategy_key for profile in profiles if profile.coverage == "sampled")
    assert len({profile.strategy_key for profile in profiles if profile.coverage == "sampled"}) == 61


def test_case13_pipeline_emits_auditable_artifacts(tmp_path, monkeypatch):
    monkeypatch.setattr(quality_engine_module.time, "strftime", lambda *_args: "20260907_053500")
    report, run_dir = QualityEngine(data_dir=DATA_DIR, output_root=tmp_path).run(case_id=13)
    second_report, second_run_dir = QualityEngine(data_dir=DATA_DIR, output_root=tmp_path).run(case_id=13)
    assert report["manifest"]["status"] == "success"
    assert second_report["manifest"]["status"] == "success"
    assert second_run_dir != run_dir
    assert second_run_dir.name == "quality_20260907_053500_001"
    assert report["manifest"]["case_ids"] == [13]
    assert report["validation"]["claims"]["valid"] is True
    assert (run_dir / "report.json").exists()
    assert (run_dir / "report.md").exists()
    assert (run_dir / "report.html").exists()
    assert (run_dir / "report.docx").exists()
    assert (run_dir / "trace.jsonl").exists()
    assert list(run_dir.glob("*.tmp")) == []
    assert sum(item["coverage"] == "unsupported_no_sample" for item in report["models"]) == 43
    html = (run_dir / "report.html").read_text(encoding="utf-8")
    assert "<table>" in html
    assert "<h2>证据结论</h2>" in html


def test_all_sampled_models_process_without_llm(tmp_path):
    report, _ = QualityEngine(data_dir=DATA_DIR, output_root=tmp_path).run()
    sampled = [item for item in report["models"] if item["coverage"] == "sampled"]
    assert len(sampled) == 61
    assert all(item["status"] == "processed" for item in sampled)
    assert report["manifest"]["online"] is False


def test_policy_router_covers_all_reasoning_modes_and_trace_events(tmp_path):
    report, _ = QualityEngine(data_dir=DATA_DIR, output_root=tmp_path).run()
    sampled = [item for item in report["models"] if item["coverage"] == "sampled"]
    modes = {item["reasoning_mode"] for item in sampled}
    assert modes == {"deterministic", "react", "plan_solve", "plan_react_reflect"}
    event_types = {event["event_type"] for event in report["trace"]}
    assert {"policy_selected", "plan_created", "reflection", "context_packed"} <= event_types
    selected_policies = {
        event["payload"]["policy"]
        for event in report["trace"]
        if event["event_type"] == "policy_selected" and isinstance(event.get("payload"), dict)
    }
    assert selected_policies == modes


def test_checkpoint_and_replay_contract(tmp_path):
    report, run_dir = QualityEngine(data_dir=DATA_DIR, output_root=tmp_path).run(case_id=13)
    assert report["manifest"]["status"] == "success"
    assert (run_dir / "checkpoint.json").read_text(encoding="utf-8").find('"status": "success"') >= 0


def test_running_checkpoint_can_resume_without_fingerprint_drift(tmp_path):
    interrupted_root = tmp_path / "interrupted"
    with pytest.raises(RuntimeError, match="intentional checkpoint interruption"):
        QualityEngine(data_dir=DATA_DIR, output_root=interrupted_root).run(
            case_id=13,
            stop_after_models=5,
        )
    run_dir = next(interrupted_root.glob("quality_*/"))
    checkpoint = json.loads((run_dir / "checkpoint.json").read_text(encoding="utf-8"))
    assert checkpoint["status"] == "running"
    assert checkpoint["processed_models"] == 5
    state_path = run_dir / checkpoint["state_file"]
    assert state_path.exists()
    # Simulate a torn append from a process dying between the state write and
    # the next metadata flush; recovery must discard it rather than duplicate
    # or deserialize partial evidence.
    with state_path.open("a", encoding="utf-8") as handle:
        handle.write('{"model_id":999}\n{"torn"')

    resumed, resumed_dir = QualityEngine(data_dir=DATA_DIR, output_root=interrupted_root).run(
        resume_dir=run_dir
    )
    clean, _ = QualityEngine(data_dir=DATA_DIR, output_root=tmp_path / "clean").run(case_id=13)
    assert resumed_dir == run_dir
    assert resumed["manifest"]["status"] == "success"
    assert replay_fingerprint(resumed) == replay_fingerprint(clean)
    state_ids = [
        json.loads(line)["model_id"]
        for line in state_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(state_ids) == len(set(state_ids))
    final_checkpoint = json.loads((run_dir / "checkpoint.json").read_text(encoding="utf-8"))
    assert final_checkpoint["status"] == "success"
    assert list(run_dir.glob("*.tmp")) == []


def test_selected_checkpoint_resume_preserves_focus_scope_and_replay(tmp_path, capsys):
    interrupted_root = tmp_path / "selected-resume"
    selected_ids = {5, 10, 14}
    with pytest.raises(RuntimeError, match="intentional checkpoint interruption"):
        QualityEngine(data_dir=DATA_DIR, output_root=interrupted_root).run(
            model_ids=selected_ids,
            stop_after_models=2,
        )
    run_dir = next(interrupted_root.glob("quality_*/"))
    checkpoint = json.loads((run_dir / "checkpoint.json").read_text(encoding="utf-8"))
    assert checkpoint["selected_model_ids"] == sorted(selected_ids)
    resumed, resumed_dir = QualityEngine(data_dir=DATA_DIR, output_root=interrupted_root).run(
        resume_dir=run_dir
    )
    assert resumed_dir == run_dir
    assert resumed["manifest"]["status"] == "success"
    assert sum(item["status"] == "processed" and item["coverage"] == "sampled" for item in resumed["models"]) == len(selected_ids)
    replay_code = quality_cli_main(["replay", str(run_dir)])
    replay_payload = json.loads(capsys.readouterr().out)
    assert replay_code == 0
    assert replay_payload["valid"] is True
    assert replay_payload["processed_sampled_models"] == len(selected_ids)


def test_resume_rejects_missing_committed_state_record(tmp_path):
    interrupted_root = tmp_path / "missing-state"
    with pytest.raises(RuntimeError, match="intentional checkpoint interruption"):
        QualityEngine(data_dir=DATA_DIR, output_root=interrupted_root).run(
            case_id=13,
            stop_after_models=5,
        )
    run_dir = next(interrupted_root.glob("quality_*/"))
    checkpoint = json.loads((run_dir / "checkpoint.json").read_text(encoding="utf-8"))
    state_path = run_dir / checkpoint["state_file"]
    state_lines = [line for line in state_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(state_lines) == checkpoint["processed_models"]
    state_path.write_text("\n".join(state_lines[:-1]) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing committed models"):
        QualityEngine(data_dir=DATA_DIR, output_root=interrupted_root).run(resume_dir=run_dir)


def test_resume_rejects_case_filter_mismatch(tmp_path):
    interrupted_root = tmp_path / "case-mismatch"
    with pytest.raises(RuntimeError, match="intentional checkpoint interruption"):
        QualityEngine(data_dir=DATA_DIR, output_root=interrupted_root).run(
            case_id=13,
            stop_after_models=5,
        )
    run_dir = next(interrupted_root.glob("quality_*/"))
    with pytest.raises(ValueError, match="case filter does not match"):
        QualityEngine(data_dir=DATA_DIR, output_root=interrupted_root).run(
            case_id=2,
            resume_dir=run_dir,
        )


def test_resume_rejects_model_filter_mismatch_with_full_checkpoint(tmp_path):
    interrupted_root = tmp_path / "model-mismatch"
    with pytest.raises(RuntimeError, match="intentional checkpoint interruption"):
        QualityEngine(data_dir=DATA_DIR, output_root=interrupted_root).run(
            case_id=13,
            stop_after_models=5,
        )
    run_dir = next(interrupted_root.glob("quality_*/"))
    with pytest.raises(ValueError, match="model filter does not match"):
        QualityEngine(data_dir=DATA_DIR, output_root=interrupted_root).run(
            model_ids={86},
            resume_dir=run_dir,
        )


def test_cli_resume_and_replay_commands_validate_artifacts(tmp_path, capsys):
    interrupted_root = tmp_path / "cli-resume"
    with pytest.raises(RuntimeError, match="intentional checkpoint interruption"):
        QualityEngine(data_dir=DATA_DIR, output_root=interrupted_root).run(
            case_id=13,
            stop_after_models=5,
        )
    run_dir = next(interrupted_root.glob("quality_*/"))
    resume_code = quality_cli_main(
        [
            "run",
            "--data-dir",
            str(DATA_DIR),
            "--output-root",
            str(interrupted_root),
            "--resume",
            str(run_dir),
        ]
    )
    resume_payload = json.loads(capsys.readouterr().out)
    assert resume_code == 0
    assert resume_payload["resumed"] is True
    assert resume_payload["manifest"]["status"] == "success"

    replay_code = quality_cli_main(["replay", str(run_dir)])
    replay_payload = json.loads(capsys.readouterr().out)
    assert replay_code == 0
    assert replay_payload["valid"] is True


def test_cli_running_resume_failure_returns_structured_json(tmp_path, capsys):
    interrupted_root = tmp_path / "cli-resume-error"
    with pytest.raises(RuntimeError, match="intentional checkpoint interruption"):
        QualityEngine(data_dir=DATA_DIR, output_root=interrupted_root).run(
            case_id=13,
            stop_after_models=5,
        )
    run_dir = next(interrupted_root.glob("quality_*/"))
    code = quality_cli_main(
        [
            "run",
            "--data-dir",
            str(DATA_DIR),
            "--output-root",
            str(interrupted_root),
            "--resume",
            str(run_dir),
            "--case-id",
            "2",
        ]
    )
    payload = json.loads(capsys.readouterr().out)
    assert code == 2
    assert payload["valid"] is False
    assert payload["resumed"] is False
    assert payload["run_dir"] == str(run_dir)
    assert "case filter" in payload["error"]


def test_cli_focused_run_replay_accepts_selected_model_subset(tmp_path, capsys):
    output_root = tmp_path / "focused"
    run_code = quality_cli_main(
        [
            "run",
            "--data-dir",
            str(DATA_DIR),
            "--output-root",
            str(output_root),
            "--model-ids",
            "5,10,14",
        ]
    )
    run_payload = json.loads(capsys.readouterr().out)
    assert run_code == 0
    assert run_payload["manifest"]["command"].endswith("--model-ids 5,10,14")
    run_dir = Path(run_payload["run_dir"])

    replay_code = quality_cli_main(["replay", str(run_dir)])
    replay_payload = json.loads(capsys.readouterr().out)
    assert replay_code == 0
    assert replay_payload["valid"] is True
    assert replay_payload["processed_sampled_models"] == 3


def test_cli_catalog_validate_reports_locked_coverage(tmp_path, capsys):
    code = quality_cli_main(["catalog", "validate", "--data-dir", str(DATA_DIR)])
    payload = json.loads(capsys.readouterr().out)
    assert code == 0
    assert payload["valid"] is True
    assert payload["total_models"] == 104
    assert payload["sampled_models"] == 61
    assert payload["unsupported_no_sample"] == 43

    missing_data_dir = tmp_path / "missing-data"
    missing_catalog_code = quality_cli_main(["catalog", "validate", "--data-dir", str(missing_data_dir)])
    missing_catalog_payload = json.loads(capsys.readouterr().out)
    assert missing_catalog_code == 2
    assert missing_catalog_payload["valid"] is False
    assert "FileNotFoundError" in missing_catalog_payload["error"]

    missing_evaluate_code = quality_cli_main(
        ["evaluate", "--data-dir", str(missing_data_dir), "--output-root", str(tmp_path / "evaluate-output")]
    )
    missing_evaluate_payload = json.loads(capsys.readouterr().out)
    assert missing_evaluate_code == 2
    assert missing_evaluate_payload["valid"] is False
    assert missing_evaluate_payload["run_dir"] is None
    assert "FileNotFoundError" in missing_evaluate_payload["error"]


def test_cli_replay_rejects_manifest_drift(tmp_path, capsys):
    report, run_dir = QualityEngine(data_dir=DATA_DIR, output_root=tmp_path).run(case_id=13)
    manifest_path = run_dir / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["run_id"] = "tampered-run"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    code = quality_cli_main(["replay", str(run_dir)])
    payload = json.loads(capsys.readouterr().out)
    assert code == 1
    assert payload["valid"] is False
    assert payload["artifact_consistency"]["valid"] is False
    assert any("run_manifest.run_id" in error for error in payload["artifact_consistency"]["errors"])


def test_cli_replay_rejects_checkpoint_manifest_and_processed_id_drift(tmp_path, capsys):
    report, run_dir = QualityEngine(data_dir=DATA_DIR, output_root=tmp_path).run(case_id=13)
    checkpoint_path = run_dir / "checkpoint.json"
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    checkpoint["manifest"]["snapshot_sha256"] = "0" * 64
    checkpoint["processed_model_ids"] = checkpoint["processed_model_ids"][:-1]
    checkpoint_path.write_text(json.dumps(checkpoint), encoding="utf-8")
    code = quality_cli_main(["replay", str(run_dir)])
    payload = json.loads(capsys.readouterr().out)
    assert code == 1
    assert payload["valid"] is False
    errors = payload["artifact_consistency"]["errors"]
    assert any("checkpoint.manifest.snapshot_sha256" in error for error in errors)
    assert any("checkpoint.processed_model_ids set" in error for error in errors)


def test_cli_replay_fails_closed_when_report_manifest_is_missing(tmp_path, capsys):
    report, run_dir = QualityEngine(data_dir=DATA_DIR, output_root=tmp_path).run(case_id=13)
    corrupted = json.loads((run_dir / "report.json").read_text(encoding="utf-8"))
    corrupted.pop("manifest")
    (run_dir / "report.json").write_text(json.dumps(corrupted), encoding="utf-8")
    code = quality_cli_main(["replay", str(run_dir)])
    payload = json.loads(capsys.readouterr().out)
    assert code == 1
    assert payload["valid"] is False
    assert payload["run_id"] is None
    assert payload["schema_contract"]["valid"] is False


def test_cli_replay_returns_structured_error_when_report_file_is_missing(tmp_path, capsys):
    report, run_dir = QualityEngine(data_dir=DATA_DIR, output_root=tmp_path).run(case_id=13)
    (run_dir / "report.json").unlink()
    code = quality_cli_main(["replay", str(run_dir)])
    payload = json.loads(capsys.readouterr().out)
    assert code == 1
    assert payload["valid"] is False
    assert payload["run_id"] is None
    assert "report.json" in payload["missing_artifacts"]


def test_cli_run_rejects_unknown_or_empty_model_filters(capsys):
    unknown_code = quality_cli_main(["run", "--model-ids", "999999"])
    unknown_error = capsys.readouterr().err
    assert unknown_code == 2
    assert "unknown models" in unknown_error

    empty_code = quality_cli_main(["run", "--model-ids", ","])
    empty_error = capsys.readouterr().err
    assert empty_code == 2
    assert "at least one model id" in empty_error


def test_cli_completed_resume_returns_structured_error_when_report_file_is_missing(tmp_path, capsys):
    report, run_dir = QualityEngine(data_dir=DATA_DIR, output_root=tmp_path).run(case_id=13)
    (run_dir / "report.json").unlink()
    code = quality_cli_main(["run", "--resume", str(run_dir)])
    payload = json.loads(capsys.readouterr().out)
    assert code == 1
    assert payload["manifest"] is None
    assert payload["replay"]["valid"] is False
    assert payload["replay"]["schema_contract"]["valid"] is False


def test_cli_completed_resume_rejects_model_filter_scope_mismatch(tmp_path, capsys):
    report, run_dir = QualityEngine(data_dir=DATA_DIR, output_root=tmp_path).run(case_id=13)
    code = quality_cli_main(["run", "--resume", str(run_dir), "--model-ids", "86"])
    payload = json.loads(capsys.readouterr().out)
    assert code == 2
    assert payload["valid"] is False
    assert payload["resumed"] is False
    assert "model filter" in payload["error"]


def test_cli_replay_fails_closed_on_structural_and_projection_drift(tmp_path, capsys):
    report, run_dir = QualityEngine(data_dir=DATA_DIR, output_root=tmp_path).run(case_id=13)
    report_path = run_dir / "report.json"
    corrupted = json.loads(report_path.read_text(encoding="utf-8"))
    corrupted.pop("models")
    report_path.write_text(json.dumps(corrupted), encoding="utf-8")
    code = quality_cli_main(["replay", str(run_dir)])
    payload = json.loads(capsys.readouterr().out)
    assert code == 1
    assert payload["valid"] is False
    assert payload["schema_contract"]["valid"] is False
    assert "processed_sampled_models" in payload


def test_cli_replay_rejects_trace_and_checkpoint_id_drift(tmp_path, capsys):
    report, run_dir = QualityEngine(data_dir=DATA_DIR, output_root=tmp_path).run(case_id=13)
    trace_path = run_dir / "trace.jsonl"
    trace_lines = [line for line in trace_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    first_trace = json.loads(trace_lines[0])
    first_trace.pop("event_id", None)
    trace_lines[0] = json.dumps(first_trace)
    trace_path.write_text("\n".join(trace_lines) + "\n", encoding="utf-8")
    state_path = run_dir / "checkpoint_models.jsonl"
    state_lines = [line for line in state_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    state = json.loads(state_lines[0])
    state["model_id"] = 999999
    state_lines[0] = json.dumps(state)
    state_path.write_text("\n".join(state_lines) + "\n", encoding="utf-8")
    code = quality_cli_main(["replay", str(run_dir)])
    payload = json.loads(capsys.readouterr().out)
    assert code == 1
    assert payload["artifact_consistency"]["valid"] is False
    assert any("event_id" in error for error in payload["artifact_consistency"]["errors"])
    assert any("model_id set" in error for error in payload["artifact_consistency"]["errors"])


def test_cli_replay_rejects_projection_content_drift_with_same_run_id(tmp_path, capsys):
    report, run_dir = QualityEngine(data_dir=DATA_DIR, output_root=tmp_path).run(case_id=13)
    markdown_path = run_dir / "report.md"
    markdown = markdown_path.read_text(encoding="utf-8")
    markdown_path.write_text(markdown.replace("限制与待核验", "被篡改的限制与待核验", 1), encoding="utf-8")
    code = quality_cli_main(["replay", str(run_dir)])
    payload = json.loads(capsys.readouterr().out)
    assert code == 1
    assert payload["artifact_consistency"]["valid"] is False
    assert any("report.md does not match" in error for error in payload["artifact_consistency"]["errors"])


def test_cli_replay_rejects_html_content_drift_with_same_run_id(tmp_path, capsys):
    report, run_dir = QualityEngine(data_dir=DATA_DIR, output_root=tmp_path).run(case_id=13)
    html_path = run_dir / "report.html"
    html = html_path.read_text(encoding="utf-8")
    html_path.write_text(html.replace("<h2>证据结论</h2>", "<h2>被篡改的证据结论</h2>", 1), encoding="utf-8")
    code = quality_cli_main(["replay", str(run_dir)])
    payload = json.loads(capsys.readouterr().out)
    assert code == 1
    assert payload["artifact_consistency"]["valid"] is False
    assert any("report.html does not match" in error for error in payload["artifact_consistency"]["errors"])


def test_cli_replay_rejects_trace_payload_drift_with_same_event_id(tmp_path, capsys):
    report, run_dir = QualityEngine(data_dir=DATA_DIR, output_root=tmp_path).run(case_id=13)
    trace_path = run_dir / "trace.jsonl"
    trace_lines = [line for line in trace_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    first_trace = json.loads(trace_lines[0])
    first_trace.setdefault("payload", {})["tampered"] = True
    trace_lines[0] = json.dumps(first_trace, ensure_ascii=False)
    trace_path.write_text("\n".join(trace_lines) + "\n", encoding="utf-8")
    code = quality_cli_main(["replay", str(run_dir)])
    payload = json.loads(capsys.readouterr().out)
    assert code == 1
    assert payload["artifact_consistency"]["valid"] is False
    assert any("trace.jsonl event payloads" in error for error in payload["artifact_consistency"]["errors"])


def test_cli_replay_rejects_docx_content_drift_with_same_run_id(tmp_path, capsys):
    report, run_dir = QualityEngine(data_dir=DATA_DIR, output_root=tmp_path).run(case_id=13)
    docx_path = run_dir / "report.docx"
    document = Document(docx_path)
    document.add_paragraph("tampered content")
    document.save(docx_path)
    code = quality_cli_main(["replay", str(run_dir)])
    payload = json.loads(capsys.readouterr().out)
    assert code == 1
    assert payload["artifact_consistency"]["valid"] is False
    assert any("report.docx does not match" in error for error in payload["artifact_consistency"]["errors"])


def test_repository_hygiene_excludes_local_credentials():
    tracked = subprocess.run(
        ["git", "ls-files"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    ).stdout.splitlines()
    forbidden_names = (".env", ".pem", ".key", "secret", "token", "credential", "password")
    leaked = [
        path
        for path in tracked
        if Path(path).name != ".env.example"
        and any(token in Path(path).name.lower() for token in forbidden_names)
    ]
    assert leaked == []
    for local_name in (".env", "DeepSeek秘钥.txt"):
        ignored = subprocess.run(
            ["git", "check-ignore", "--quiet", local_name],
            cwd=PROJECT_ROOT,
        )
        assert ignored.returncode == 0, local_name


def test_cross_model_links_are_downgraded_and_multi_source():
    facts = [
        Fact("f1", 5, 13, "entity_cardinality", "姓名", {"distinct": 1, "examples": ["同一对象"]}),
        Fact("f2", 86, 13, "entity_cardinality", "流出对象", {"distinct": 1, "examples": ["同一对象"]}),
    ]
    links, claims = build_cross_model_links(facts)
    assert len(links) == 1
    assert claims[0].level == "inference"
    assert claims[0].status == "downgraded"
    assert len(claims[0].model_ids) == 2


def test_focused_model_filter_is_explicit(tmp_path):
    report, _ = QualityEngine(data_dir=DATA_DIR, output_root=tmp_path).run(case_id=13, model_ids={86})
    selected = next(item for item in report["models"] if item["model_id"] == 86)
    skipped = next(item for item in report["models"] if item["model_id"] == 70)
    assert selected["status"] == "processed"
    assert skipped["status"] == "not_selected"


def test_offline_replay_fingerprint_is_stable(tmp_path):
    first, _ = QualityEngine(data_dir=DATA_DIR, output_root=tmp_path / "one").run(case_id=13)
    second, _ = QualityEngine(data_dir=DATA_DIR, output_root=tmp_path / "two").run(case_id=13)
    assert replay_fingerprint(first) == replay_fingerprint(second)


def test_every_sample_case_runs_end_to_end(tmp_path):
    case_ids = sorted({int(row["case_id"]) for row in load_sample_rows(DATA_DIR)})
    assert case_ids == [2, 5, 8, 9, 10, 11, 13]
    for case_id in case_ids:
        report, _ = QualityEngine(data_dir=DATA_DIR, output_root=tmp_path / str(case_id)).run(case_id=case_id)
        assert report["manifest"]["status"] == "success"
        assert report["manifest"]["case_ids"] == [case_id]
        assert report["validation"]["claims"]["valid"] is True


def test_fault_injection_suite_is_safe():
    result = run_fault_injection_suite(DATA_DIR)
    assert result["valid"] is True
    assert all(case["passed"] for case in result["cases"].values())


def test_non_mapping_rows_are_downgraded_without_crashing():
    profiles, _ = build_registry(DATA_DIR)
    profile = next(item for item in profiles if item.model_id == 14)
    rows = ["not-a-row", None, {"row_id": 7, "data": {"联系人": "甲"}}]
    facts = compute_facts(profile, rows, "fuzz")
    assert any(fact.kind == "data_quality" and fact.complete is False for fact in facts)
    assert any(fact.kind == "row_count" and fact.value["count"] == 3 for fact in facts)


def test_identifier_columns_are_not_treated_as_numeric_metrics():
    profiles, _ = build_registry(DATA_DIR)
    profile = next(item for item in profiles if item.model_id == 14)
    rows = [
        {"row_id": 1, "data": {"手机号": "phone-A", "交易金额": "120.5"}},
        {"row_id": 2, "data": {"手机号": "phone-B", "交易金额": "80.5"}},
    ]
    facts = compute_facts(profile, rows, "identifier-fuzz")
    numeric_subjects = {fact.subject for fact in facts if fact.kind == "numeric_summary"}
    assert "手机号" not in numeric_subjects
    assert "交易金额" in numeric_subjects


def test_temporal_columns_are_not_treated_as_numeric_metrics():
    profiles, _ = build_registry(DATA_DIR)
    profile = next(item for item in profiles if item.model_id == 14)
    rows = [
        {"row_id": 1, "data": {"交易时间": "2024-01-01 10:00:00", "交易金额": "120.5", "通话时长": "30"}},
        {"row_id": 2, "data": {"交易时间": "2024-01-02 10:00:00", "交易金额": "80.5", "通话时长": "45"}},
        {"row_id": 3, "data": {"交易时间": "2024-01-03 10:00:00", "交易金额": "90.5", "通话时长": "15"}},
    ]
    facts = compute_facts(profile, rows, "temporal-fuzz")
    numeric_subjects = {fact.subject for fact in facts if fact.kind == "numeric_summary"}
    assert "交易时间" not in numeric_subjects
    assert "交易金额" in numeric_subjects
    assert "通话时长" in numeric_subjects


def test_registry_loader_rejects_non_object_and_invalid_keys(tmp_path):
    (tmp_path / "ch_model_info.jsonl").write_text("{}\n", encoding="utf-8")
    (tmp_path / "ch_model_info_data_sample.jsonl").write_text("[]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Expected JSON object"):
        load_sample_rows(tmp_path)

    (tmp_path / "ch_model_info_data_sample.jsonl").write_text(
        json.dumps({"id": "bad", "model_id": 5, "case_id": 13, "json_data": "{}"}) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Invalid integer key"):
        load_sample_rows(tmp_path)


def test_all_profiles_tolerate_mixed_malformed_rows():
    profiles, _ = build_registry(DATA_DIR)
    rows = [None, "not-a-row", {"row_id": 7, "data": {"金额": "bad", "时间": 123}}]
    tools = ToolRegistry()
    for profile in profiles:
        facts = compute_facts(profile, rows, "fuzz")
        observations, _ = run_policy(profile, rows, facts, "fuzz", tools, 3)
        assert validate_claims(build_claims(profile, facts, observations), facts)["valid"] is True


def test_acceptance_summary_locks_coverage_and_fault_gate():
    result = acceptance_summary(DATA_DIR)
    assert result["valid"] is True
    assert result["registry"]["sampled_models"] == 61
    assert result["strategy_contract"]["valid"] is True


def test_report_schema_contract_rejects_structural_corruption():
    result = validate_report_contract({"schema_version": "0.9", "facts": []})
    assert result["valid"] is False
    assert any("missing top-level key" in error for error in result["errors"])
    assert any("schema_version" in error for error in result["errors"])

    report, _ = QualityEngine(data_dir=DATA_DIR).run(case_id=13)
    corrupted = deepcopy(report)
    corrupted["claims"].append(deepcopy(corrupted["claims"][0]))
    duplicate_result = validate_report_contract(corrupted)
    assert duplicate_result["valid"] is False
    assert "claim_id values must be unique" in duplicate_result["errors"]


def test_report_schema_contract_rejects_bad_fact_shape_and_orphan_references():
    report, _ = QualityEngine(data_dir=DATA_DIR).run(case_id=13)
    corrupted = deepcopy(report)
    corrupted["facts"][0]["fact_id"] = None
    corrupted["facts"][0]["complete"] = "yes"
    corrupted["claims"][0]["fact_ids"].append("missing_fact")
    corrupted["claims"][0]["model_ids"].append(999999)
    result = validate_report_contract(corrupted)
    assert result["valid"] is False
    assert any("fact_id must be a non-empty string" in error for error in result["errors"])
    assert any("complete must be boolean" in error for error in result["errors"])
    assert any("missing fact ids" in error for error in result["errors"])
    assert any("missing model ids" in error for error in result["errors"])


def test_report_schema_contract_rejects_model_set_and_coverage_drift():
    report, _ = QualityEngine(data_dir=DATA_DIR).run(case_id=13)
    corrupted = deepcopy(report)
    corrupted["models"][1]["model_id"] = corrupted["models"][0]["model_id"]
    corrupted["coverage"]["total_models"] += 1
    result = validate_report_contract(corrupted)
    assert result["valid"] is False
    assert "model_id values must be unique" in result["errors"]
    assert "coverage.total_models does not match models length" in result["errors"]


def test_benchmark_enforces_replay_fingerprint_stability(capsys):
    # The strict 1.3-second wall-clock gate belongs to the standalone release
    # benchmark/watchdog. This unit isolates the replay-fingerprint invariant
    # so host scheduling noise cannot turn a stable deterministic run into a
    # flaky full-suite failure.
    code = benchmark_main(["--runs", "2", "--threshold", "5.0"])
    payload = json.loads(capsys.readouterr().out)
    assert code == 0
    assert payload["valid"] is True
    assert payload["fingerprint_stable"] is True
    assert payload["replay_fingerprint"]


def test_malformed_fuzz_harness_entrypoint(capsys):
    code = fuzz_main(["--rounds", "1", "--seed", "20260907"])
    payload = json.loads(capsys.readouterr().out)
    assert code == 0
    assert payload["valid"] is True
    assert payload["profiles"] == 104
    assert payload["profile_runs"] == 104

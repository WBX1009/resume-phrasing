import json
import os
from pathlib import Path
import subprocess
import sys

from fastapi.testclient import TestClient

from investigation_support import snapshot, service, ScriptedReasoner, wait_done
from src.case_api import create_app
from src.case_execution import CaseExecutionConfig


def test_api_uses_server_repository_and_exposes_evidence_and_sse(tmp_path):
    active = service(tmp_path)
    with TestClient(create_app(lambda: active)) as client:
        assert client.get("/health").status_code == 200
        bad = client.post("/v1/case-runs", json={"case_id": "synthetic", "case_scope": {"entity_ids": ["outside"]}})
        assert bad.status_code == 422
        assert client.post("/v1/case-runs", json={"case_id": "missing"}).status_code == 404
        response = client.post("/v1/case-runs", json={"case_id": "synthetic", "run_id": "r"})
        assert response.status_code == 202
        assert wait_done(active)["status"] == "success"
        report = client.get("/v1/case-runs/r/report").json()
        fid = report["fact_ids"][0]
        assert client.get(f"/v1/case-runs/r/facts/{fid}").json()["fact_id"] == fid
        assert client.get("/v1/case-runs/r/facts/unknown").status_code == 404
        with active.store._connect() as db:
            db.execute("UPDATE case_task_outcomes SET spec_json='invalid json' WHERE run_id='r'")
        assert client.get("/v1/case-runs/r").status_code == 200
        events = client.get("/v1/case-runs/r/events")
        assert events.status_code == 200 and "event: complete" in events.text
        assert "prepare_facts" in events.text and "worker_tool" in events.text
        assert client.post("/v1/case-runs/r/resume").status_code == 409


def test_partial_report_remains_available_and_gaps_are_explicit(tmp_path):
    active = service(tmp_path, reasoner=ScriptedReasoner(rechecks=True), config=CaseExecutionConfig(max_recheck_rounds=0))
    with TestClient(create_app(lambda: active)) as client:
        client.post("/v1/case-runs", json={"case_id": "synthetic", "run_id": "r"})
        assert wait_done(active)["status"] == "partial"
        report = client.get("/v1/case-runs/r/report")
        assert report.status_code == 200 and len(report.json()["verification"]["gaps"]) == 2
        assert client.get("/v1/case-runs/unknown/report").status_code == 404
        assert client.get("/v1/case-runs/unknown/events").status_code == 404


def test_importing_api_does_not_open_databases_or_start_service_threads(tmp_path):
    project = Path(__file__).resolve().parents[1]
    script = "import src.case_api,threading; from pathlib import Path; assert not list(Path('.').iterdir()); assert 'case-runs' not in [t.name for t in threading.enumerate()]"
    result = subprocess.run([sys.executable, "-B", "-c", script], cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": str(project)}, capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stderr


def test_cli_uses_the_same_end_to_end_graph_without_private_data(tmp_path):
    project = Path(__file__).resolve().parents[1]
    repository = tmp_path / "cases"
    repository.mkdir()
    (repository / "synthetic.json").write_text(json.dumps(snapshot(rules=True)), encoding="utf-8")
    target = tmp_path / "report.json"
    result = subprocess.run([sys.executable, "-B", "-m", "src.case_cli", "--case-id", "synthetic",
        "--repository", str(repository), "--storage", str(tmp_path / "runtime"), "--offline", "--report", str(target)],
        cwd=project, env={**os.environ, "PYTHONIOENCODING": "utf-8"}, capture_output=True, text=True, encoding="utf-8", timeout=20)
    assert result.returncode == 0, result.stderr
    report = json.loads(target.read_text(encoding="utf-8"))
    assert report["runtime"] == "langgraph-native-v3" and report["status"] == "partial"
    assert report["tasks"] and report["verification"]["gaps"]
    assert report["verification"]["final_events"] == []

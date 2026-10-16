import time
import asyncio
from pathlib import Path
import sqlite3

from fastapi.testclient import TestClient

from src.case_api import _build_default_service, create_app
from src.candidate_events import CandidateEventBuilder
from src.case_run_service import CaseRunService
from src.case_task_scheduler import (
    CaseTaskScope,
    InvestigationTask,
    SQLiteRunStore,
    TaskSchedulerConfig,
)


def _service(tmp_path: Path) -> CaseRunService:
    store = SQLiteRunStore(tmp_path / "runs.sqlite3")

    async def worker(task, task_input, context):
        result = {"task_id": task.task_id, "value": task.payload.get("value", "ok")}
        if "candidate_events" in task.payload:
            result["candidate_events"] = task.payload["candidate_events"]
        return result

    async def planner(state, context):
        return [InvestigationTask.from_dict(item) for item in state["initial_tasks"]]

    async def verifier(state, context):
        return {"gaps": [], "accepted_events": ["synthetic-event"]}

    async def recheck_planner(state, context):
        return []

    async def recheck_allowed(state):
        return False

    async def summarizer(state, context):
        return {
            "case_id": state["case_id"],
            "run_id": state["run_id"],
            "task_runs": state["task_runs"],
            "verification": state["verification"],
            "candidate_events": state.get("candidate_events", []),
            "final_events": state.get("final_events", []),
        }

    return CaseRunService(
        store,
        worker=worker,
        planner=planner,
        verifier=verifier,
        recheck_planner=recheck_planner,
        recheck_allowed=recheck_allowed,
        summarizer=summarizer,
        checkpoint_path=tmp_path / "checkpoints.sqlite3",
    )


def _request_payload():
    return {
        "case_id": "case-api-1",
        "run_id": "run-api-1",
        "case_facts": {"fact-1": {"value": "synthetic"}},
        "case_scope": {
            "entity_ids": ["person-1"],
            "fact_ids": ["fact-1"],
            "allowed_tools": [],
        },
        "tasks": [
            {
                "task_id": "task-1",
                "question": "Review the synthetic fact",
                "entity_ids": ["person-1"],
                "basis_fact_ids": ["fact-1"],
                "allowed_tools": [],
                "payload": {"value": "evidence"},
            }
        ],
    }


def _wait_for_terminal(client: TestClient, run_id: str):
    for _ in range(100):
        response = client.get(f"/v1/case-runs/{run_id}")
        record = response.json()
        if record["status"] in {"success", "failed", "cancelled"}:
            return record
        time.sleep(0.02)
    raise AssertionError("case run did not reach a terminal status")


def test_case_api_submits_and_reports_with_durable_workflow(tmp_path: Path):
    service = _service(tmp_path)
    client = TestClient(create_app(service))

    accepted = client.post("/v1/case-runs", json=_request_payload())
    assert accepted.status_code == 202
    assert accepted.json() == {"run_id": "run-api-1", "status": "running"}

    record = _wait_for_terminal(client, "run-api-1")
    assert record["status"] == "success"
    report = client.get("/v1/case-runs/run-api-1/report")
    assert report.status_code == 200
    assert report.json()["verification"]["gaps"] == []

    events = client.get("/v1/case-runs/run-api-1/events")
    assert events.status_code == 200
    assert "event: complete" in events.text
    assert '"node": "plan"' in events.text
    assert '"node": "summarize"' in events.text
    assert (tmp_path / "checkpoints.sqlite3").exists()
    with sqlite3.connect(tmp_path / "checkpoints.sqlite3") as connection:
        assert connection.execute("SELECT COUNT(*) FROM checkpoints").fetchone()[0] > 0


def test_case_run_store_persists_cancel_and_failure_fields(tmp_path: Path):
    store = SQLiteRunStore(tmp_path / "runs.sqlite3")
    scope = {
        "entity_ids": ["person-1"],
        "fact_ids": ["fact-1"],
        "allowed_tools": [],
    }
    store.prepare_run("run-state", TaskSchedulerConfig(), CaseTaskScope(
        entity_ids=frozenset(scope["entity_ids"]),
        fact_ids=frozenset(scope["fact_ids"]),
        allowed_tools=frozenset(),
    ))
    assert store.request_cancel("run-state") is True
    assert store.is_cancel_requested("run-state") is True
    store.finish_run("run-state", "failed", error="TimeoutError")
    assert store.get_run("run-state")["error"] == "TimeoutError"


def test_case_api_cancel_is_cooperative_and_visible_in_status(tmp_path: Path):
    service = _service(tmp_path)
    original_worker = service._worker

    async def slow_worker(task, task_input, context):
        await asyncio.sleep(0.2)
        return await original_worker(task, task_input, context)

    service._worker = slow_worker
    client = TestClient(create_app(service))
    accepted = client.post("/v1/case-runs", json=_request_payload())
    assert accepted.status_code == 202
    cancel = client.post("/v1/case-runs/run-api-1/cancel")
    assert cancel.status_code == 200
    assert cancel.json()["cancel_requested"] is True

    record = _wait_for_terminal(client, "run-api-1")
    assert record["status"] == "cancelled"


def test_case_api_persists_explicit_candidate_events_in_report(tmp_path: Path):
    payload = _request_payload()
    payload["run_id"] = "run-api-candidate"
    payload["tasks"][0]["payload"]["candidate_events"] = [
        {"event_type": "synthetic-transfer", "subject_id": "person-1"}
    ]
    service = _service(tmp_path)
    client = TestClient(create_app(service))

    accepted = client.post("/v1/case-runs", json=payload)
    assert accepted.status_code == 202
    record = _wait_for_terminal(client, "run-api-candidate")
    assert record["status"] == "success"

    report = client.get("/v1/case-runs/run-api-candidate/report")
    assert report.status_code == 200
    expected = [{"event_type": "synthetic-transfer", "subject_id": "person-1"}]
    assert report.json()["candidate_events"] == expected


def test_default_service_builds_candidates_and_closes_rechecked_gap(tmp_path: Path):
    fact_pack = {
        "focus_entities": [
            {
                "entity_name": "person-1",
                "fact_id": "fact-1",
                "anomaly_flags": [{"risk_tag": "night_activity"}],
            }
        ]
    }
    candidate = CandidateEventBuilder().build(fact_pack)[0]
    service = _build_default_service(tmp_path)
    run_id = "default-recheck"
    service.submit(
        {
            "case_id": run_id,
            "run_id": run_id,
            "case_facts": {"fact_pack": fact_pack},
            "case_scope": {
                "entity_ids": ["person-1"],
                "fact_ids": ["fact-1"],
                "allowed_tools": [],
            },
            "initial_tasks": [
                {
                    "task_id": "initial",
                    "question": "形成候选并核验",
                    "entity_ids": ["person-1"],
                    "basis_fact_ids": ["fact-1"],
                    "allowed_tools": [],
                    "payload": {
                        "fact_pack": fact_pack,
                        "verification": {
                            "gaps": [
                                {
                                    "candidate_event_id": candidate["event_id"],
                                    "missing_fact": "关系",
                                }
                            ]
                        },
                    },
                }
            ],
        }
    )
    record = None
    for _ in range(100):
        time.sleep(0.02)
        record = service.status(run_id)
        if record and record["status"] in {"success", "failed", "cancelled"}:
            break
    assert record and record["status"] == "success"
    report = service.report(run_id)
    assert report is not None
    assert report["verification"]["gaps"] == []
    assert len(report["candidate_events"]) == 1
    assert len(report["final_events"]) == 1

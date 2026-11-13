import os
from pathlib import Path
import subprocess
import sys
import json
import pytest


def _run_child(script: str, checkpoint_path: Path) -> None:
    environment = os.environ.copy()
    environment["CASE_CHECKPOINT_PATH"] = str(checkpoint_path)
    subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        cwd=Path(__file__).resolve().parents[1],
        env=environment,
    )


def test_langgraph_sqlite_checkpoint_survives_process_boundary(tmp_path: Path):
    checkpoint_path = tmp_path / "cross-process-checkpoint.sqlite3"
    writer = """
import asyncio
import os
from typing import TypedDict
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import END, START, StateGraph

class State(TypedDict):
    value: str

def set_value(state):
    return {"value": "persisted-by-first-process"}

async def main():
    builder = StateGraph(State)
    builder.add_node("set_value", set_value)
    builder.add_edge(START, "set_value")
    builder.add_edge("set_value", END)
    async with AsyncSqliteSaver.from_conn_string(os.environ["CASE_CHECKPOINT_PATH"]) as saver:
        graph = builder.compile(checkpointer=saver)
        await graph.ainvoke(
            {"value": "initial"},
            {"configurable": {"thread_id": "cross-process-thread"}},
        )

asyncio.run(main())
"""
    reader = """
import os
import sqlite3
from langgraph.checkpoint.sqlite import SqliteSaver

with sqlite3.connect(os.environ["CASE_CHECKPOINT_PATH"], check_same_thread=False) as connection:
    saver = SqliteSaver(connection)
    item = saver.get_tuple({"configurable": {"thread_id": "cross-process-thread"}})
    assert item is not None
    assert item.checkpoint["channel_values"]["value"] == "persisted-by-first-process"
"""

    _run_child(writer, checkpoint_path)
    _run_child(reader, checkpoint_path)


@pytest.mark.parametrize("boundary", ["worker", "verify", "publish", "recheck"])
def test_case_service_recovers_after_process_exit_without_replaying_completed_tasks(tmp_path, boundary):
    script = r'''
import asyncio, json, os, sqlite3, time
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from pathlib import Path
from src.case_execution import InvestigationTask, SQLiteRunStore
from src.case_run_service import CaseRunService

root = Path(os.environ["RECOVERY_ROOT"])
phase = os.environ["RECOVERY_PHASE"]
boundary = os.environ["RECOVERY_BOUNDARY"]
store = SQLiteRunStore(root / "runs.db")
def log(name):
    with (root / "calls.jsonl").open("a") as f:
        f.write(json.dumps(name) + "\n")
def task(name):
    return InvestigationTask(name, "synthetic", entity_ids=("p",), basis_fact_ids=("f",))
async def planner(state, context):
    log("plan")
    return [task("A"), task("B")]
async def worker(t, data, context):
    log(t.task_id)
    if phase == "crash" and boundary == "worker" and t.task_id == "B":
        # Wait until A's parent task output is durably recorded as a pending write.
        for _ in range(1000):
            with sqlite3.connect(root / "checkpoints.db") as db:
                rows = db.execute("SELECT type,value FROM writes WHERE channel='outcomes'").fetchall()
                saved = any("A" in JsonPlusSerializer().loads_typed((kind, value)) for kind,value in rows)
            if saved:
                os._exit(71)
            await asyncio.sleep(.01)
        raise AssertionError("A never checkpointed")
    if phase == "crash" and boundary == "recheck" and t.task_id == "recheck":
        os._exit(71)
    return {"evidence": t.task_id}
async def verifier(state, context):
    log("verify")
    if phase == "crash" and boundary == "verify":
        os._exit(71)
    return {"gaps": ["synthetic gap"] if boundary == "recheck" and "recheck" not in state["task_outcomes"] else []}
async def rechecks(state, context):
    return [task("recheck")]
async def summary(state, context):
    return {"task_ids": sorted(state["task_outcomes"])}
if phase == "crash" and boundary == "publish":
    store.publish_report = lambda *args: os._exit(71)
service = CaseRunService(store, worker=worker, planner=planner, verifier=verifier,
    recheck_planner=rechecks, recheck_allowed=lambda s: True, summarizer=summary,
    checkpoint_path=root / "checkpoints.db")
if phase == "crash":
    service.submit({"run_id": "r", "case_id": "c", "case_facts": {"f": 1},
        "case_scope": {"entity_ids": ["p"], "fact_ids": ["f"], "allowed_tools": []}, "initial_tasks": []})
for _ in range(1000):
    record = service.status("r")
    if record and record["status"] != "running":
        assert phase == "resume", record
        assert record["status"] == "success", record
        assert sorted(service.store.load_task_projection("r")) == (["A", "B", "recheck"] if boundary == "recheck" else ["A", "B"])
        assert service.report("r")["task_ids"] == (["A", "B", "recheck"] if boundary == "recheck" else ["A", "B"])
        service.close()
        break
    time.sleep(.01)
else:
    raise AssertionError("service did not complete")
'''
    environment = {**os.environ, "RECOVERY_ROOT": str(tmp_path), "RECOVERY_BOUNDARY": boundary,
                   "LANGSMITH_TRACING": "false"}
    for phase in ("crash", "resume"):
        if phase == "resume" and boundary == "publish":
            import sqlite3
            # Even malformed task JSON must not block native checkpoint recovery.
            with sqlite3.connect(tmp_path / "runs.db") as connection:
                connection.execute("UPDATE case_task_outcomes SET spec_json='{}',status='pending'")
        environment["RECOVERY_PHASE"] = phase
        result = subprocess.run([sys.executable, "-B", "-c", script], env=environment,
                                cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True, timeout=30)
        assert result.returncode == (71 if phase == "crash" else 0), result.stdout + result.stderr
    calls = [json.loads(line) for line in (tmp_path / "calls.jsonl").read_text().splitlines()]
    assert calls.count("plan") == calls.count("A") == 1
    assert calls.count("B") == (2 if boundary == "worker" else 1)
    if boundary == "recheck":
        assert calls.count("recheck") == 2


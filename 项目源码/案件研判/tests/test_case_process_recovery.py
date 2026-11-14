"""Hard process exits test checkpoints, not SQLite saver table layouts."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
from investigation_support import snapshot


CHILD = r'''import asyncio, json, os, time
from pathlib import Path
from src.case_execution import SQLiteRunStore, CaseExecutionConfig
from src.case_run_service import CaseRunService
from src.case_repository import MappingCaseRepository
from investigation_support import ScriptedReasoner, wait_done
import src.case_workflow_graph as workflow
root = Path(os.environ["RECOVERY_ROOT"])
phase, boundary = os.environ["RECOVERY_PHASE"], os.environ["RECOVERY_BOUNDARY"]
def log(value):
    with (root / "calls.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(value)+"\n")
async def observe(stage,payload,context):
    log([stage,payload.get("task",{}).get("task_id"),payload.get("step")])
    if phase == "crash":
        if boundary == "judge" and stage == "worker_judge" and payload["step"] == 1:
            os._exit(74)
        if boundary == "recheck" and stage == "worker_plan" and payload["task"].get("gap_ids"):
            os._exit(74)
store = SQLiteRunStore(root / "runs.db")
if phase == "crash" and boundary == "verify":
    workflow.central_verify = lambda *a: os._exit(74)
if phase == "crash" and boundary == "publish":
    store.publish_report = lambda *a: os._exit(74)
cases = {"synthetic":json.loads((root/"source.json").read_text(encoding="utf-8"))} if phase == "crash" else {}
reasoner = ScriptedReasoner(rechecks=boundary=="recheck", observer=observe)
service = CaseRunService(store, repository=MappingCaseRepository(cases), reasoner=reasoner)
if phase == "crash": service.submit("synthetic",run_id="r")
record = wait_done(service,timeout=20)
report = service.report("r")
(root/"result.json").write_text(json.dumps({"record":record,"report":report}),encoding="utf-8")
service.close()
'''


@pytest.mark.parametrize("boundary", ["judge", "verify", "publish", "recheck"])
def test_case_recovers_across_processes_without_replaying_completed_steps(tmp_path, boundary):
    (tmp_path / "source.json").write_text(json.dumps(snapshot()), encoding="utf-8")
    env = {**os.environ, "RECOVERY_ROOT": str(tmp_path), "RECOVERY_BOUNDARY": boundary,
        "PYTHONPATH": os.pathsep.join([".", "tests"]), "PYTHONIOENCODING": "utf-8"}
    cwd = Path(__file__).resolve().parents[1]
    crash = subprocess.run([sys.executable, "-B", "-c", CHILD], cwd=cwd,
        env={**env,"RECOVERY_PHASE":"crash"}, capture_output=True, text=True, timeout=30, encoding="utf-8")
    assert crash.returncode == 74, crash.stdout+crash.stderr
    # Restart with an empty repository. The admitted immutable snapshot is
    # sufficient, so later edits/deletions cannot alter the original case.
    (tmp_path / "source.json").unlink()
    resumed = subprocess.run([sys.executable, "-B", "-c", CHILD], cwd=cwd,
        env={**env,"RECOVERY_PHASE":"resume"}, capture_output=True, text=True, timeout=30, encoding="utf-8")
    assert resumed.returncode == 0, resumed.stdout+resumed.stderr
    result = json.loads((tmp_path / "result.json").read_text(encoding="utf-8"))
    assert result["record"]["status"] == ("partial" if boundary == "recheck" else "success")
    calls = [json.loads(line) for line in (tmp_path/"calls.jsonl").read_text(encoding="utf-8").splitlines()]
    assert sum(stage == "discover" for stage,tid,step in calls) == 1
    assert sum(stage == "plan" for stage,tid,step in calls) == 1
    assert sum(stage == "worker_plan" and tid == "initial-0" for stage,tid,step in calls) == 2
    if boundary != "recheck":
        assert result["report"]["budget"]["tool_calls"] == 2
    if boundary == "judge":
        assert result["report"]["budget"]["llm_calls"] == 7


PENDING = r'''import asyncio,json,os
from pathlib import Path
from typing import Annotated,TypedDict
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import START,END,StateGraph
from langgraph.types import Send
root=Path(os.environ["RECOVERY_ROOT"])
phase=os.environ["RECOVERY_PHASE"]
class State(TypedDict,total=False):
    done: Annotated[dict,lambda a,b:{**a,**b}]
async def main():
    config={"configurable":{"thread_id":"parallel"}}
    async with AsyncSqliteSaver.from_conn_string(str(root/"parallel.db")) as saver:
        async def worker(state):
            with (root/"parallel.jsonl").open("a") as f: f.write(json.dumps(state["id"])+"\n")
            if state["id"]=="B" and phase=="crash":
                for _ in range(1000):
                    item=await saver.aget_tuple(config)
                    if item and any(channel=="done" and "A" in value for _,channel,value in item.pending_writes):
                        os._exit(74)
                    await asyncio.sleep(.01)
                raise AssertionError("A never checkpointed")
            return {"done":{state["id"]:True}}
        g=StateGraph(State)
        g.add_node("worker",worker)
        g.add_conditional_edges(START,lambda s:[Send("worker",{"id":"A"}),Send("worker",{"id":"B"})],["worker"])
        g.add_edge("worker",END)
        graph=g.compile(checkpointer=saver)
        result=await graph.ainvoke({"done":{}} if phase=="crash" else None,config,durability="sync")
        assert result["done"]=={"A":True,"B":True}
asyncio.run(main())
'''


def test_native_parallel_pending_write_reuses_successful_sibling(tmp_path):
    env = {**os.environ,"RECOVERY_ROOT":str(tmp_path)}
    for phase,code in (("crash",74),("resume",0)):
        result = subprocess.run([sys.executable,"-B","-c",PENDING], env={**env,"RECOVERY_PHASE":phase},
            capture_output=True,text=True,timeout=30)
        assert result.returncode==code,result.stdout+result.stderr
    calls = [json.loads(x) for x in (tmp_path/"parallel.jsonl").read_text().splitlines()]
    assert calls.count("A")==1 and calls.count("B")==2

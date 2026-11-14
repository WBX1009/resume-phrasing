import asyncio
from copy import deepcopy
import threading
import time

import httpx
import pytest

from investigation_support import snapshot, service, ScriptedReasoner, wait_done
from test_case_authorization import admitted
from src.case_execution import CaseExecutionConfig, SharedResourceLimits, ExternalOperationUncertain, _invoke_tool
from src.investigation_contracts import Contract
from src.investigation_tools import InvestigationTools, RegisteredTool
from src.case_worker_subgraph import retryable_error


@pytest.mark.parametrize("status,expected", [(400, False), (401, False), (408, True), (429, True), (503, True)])
def test_native_retry_classifier_distinguishes_permanent_provider_errors(status, expected):
    response = httpx.Response(status, request=httpx.Request("GET", "https://synthetic.invalid"))
    assert retryable_error(httpx.HTTPStatusError("synthetic", request=response.request, response=response)) is expected


def test_programming_error_stops_graph_and_can_resume_from_last_checkpoint(tmp_path):
    class Defect(ScriptedReasoner):
        def response(self, stage, payload):
            if stage == "worker_judge":
                raise RuntimeError("synthetic defect")
            return super().response(stage, payload)
    faulty = Defect()
    active = service(tmp_path, reasoner=faulty)
    active.submit("synthetic", run_id="r")
    assert wait_done(active)["status"] == "failed"
    assert sum(s == "worker_judge" for s,t,p in faulty.calls) == 1
    assert active.report("r") is None
    active.close()
    fixed = ScriptedReasoner()
    active = service(tmp_path, reasoner=fixed)
    try:
        active.resume("r")
        assert wait_done(active)["status"] == "success"
        assert not any(s in {"discover", "plan"} for s,t,p in fixed.calls)
        assert active.report("r")["budget"]["tool_calls"] == 2
    finally:
        active.close()


def test_native_node_timeout_is_bounded_and_blocks_dependents(tmp_path):
    async def slow(stage, payload, context):
        if stage == "worker_plan":
            await asyncio.sleep(10)
    active = service(tmp_path, reasoner=ScriptedReasoner(dependent=True, observer=slow),
        config=CaseExecutionConfig(task_timeout=1, max_attempts=2, max_recheck_rounds=0))
    try:
        active.submit("synthetic", run_id="r")
        assert wait_done(active)["status"] == "partial"
        outcomes = active.report("r")["task_outcomes"]
        assert outcomes["initial-0"]["error"] == "NodeTimeoutError"
        assert outcomes["dependent-0"]["status"] == "blocked"
    finally:
        active.close()


def test_multiple_cases_share_model_and_worker_resource_limits(tmp_path):
    active_workers, peak_workers, active_models, peak_models = set(), 0, 0, 0
    class Measured(ScriptedReasoner):
        async def decide(self, stage, payload, context):
            nonlocal peak_workers, active_models, peak_models
            async def callback():
                nonlocal peak_workers, active_models, peak_models
                active_models += 1
                peak_models = max(peak_models, active_models)
                if stage == "worker_plan":
                    active_workers.add(context.run_id)
                    peak_workers = max(peak_workers, len(active_workers))
                await asyncio.sleep(.01)
                result = self.response(stage, payload)
                if stage == "worker_judge" and result["done"]:
                    active_workers.discard(context.run_id)
                active_models -= 1
                return result
            return await context.llm_call(callback, max_output_tokens=10)
    active = service(tmp_path, reasoner=Measured(), resources=SharedResourceLimits(llm_request_cap=1, db_query_cap=1, worker_cap=1))
    try:
        active.submit("synthetic", run_id="a")
        active.submit("synthetic", run_id="b")
        assert wait_done(active, "a")["status"] == "success"
        assert wait_done(active, "b")["status"] == "success"
        assert peak_models == peak_workers == 1
    finally:
        active.close()


def test_cancelling_blocking_io_does_not_release_database_slot_early():
    async def scenario():
        gate = asyncio.Semaphore(1)
        started, finish = threading.Event(), threading.Event()
        def blocking():
            started.set()
            assert finish.wait(3)
        task = asyncio.create_task(_invoke_tool(gate, blocking, lambda: None))
        while not started.is_set():
            await asyncio.sleep(.005)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert gate.locked()
        finish.set()
        await asyncio.wait_for(gate.acquire(), 3)
        gate.release()
    asyncio.run(scenario())


@pytest.mark.parametrize("lose_ack", [False, True])
def test_external_write_idempotence_and_uncertain_acknowledgement(tmp_path, lose_ack):
    class Empty(Contract):
        pass
    writes = []
    def write(snapshot, request, *, idempotency_key):
        writes.append(idempotency_key)
        if lose_ack:
            raise ConnectionError("synthetic lost acknowledgement after external commit")
        return {"aggregation_complete": True, "receipt": "synthetic"}
    active = service(tmp_path)
    try:
        data = snapshot()
        data["allowed_tools"].append("write")
        from src.case_repository import MappingCaseRepository
        active.repository = MappingCaseRepository({"synthetic": data})
        active.tools = InvestigationTools(active.store, extra={"write": RegisteredTool(Empty, write, writes=True)})
        data, task, context = admitted(active)
        request = {"tool_name": "write", "entity_ids": ["p"], "time_range": data.time_range, "params": {}}
        async def scenario():
            if lose_ack:
                with pytest.raises(ConnectionError):
                    await context.execute_tool(request)
                with pytest.raises(ExternalOperationUncertain):
                    await context.execute_tool(request)
            else:
                first = await context.execute_tool(request)
                assert await context.execute_tool(request) == first
        asyncio.run(scenario())
        assert len(writes) == 1
        assert active.store.budget_usage("r")["tool_calls"] == 1
    finally:
        active.close()


def test_cancel_is_terminal_and_cannot_be_resumed(tmp_path):
    started = threading.Event()
    async def pause(stage, payload, context):
        if stage == "worker_plan":
            started.set()
            await asyncio.sleep(.1)
    active = service(tmp_path, reasoner=ScriptedReasoner(observer=pause))
    try:
        active.submit("synthetic", run_id="r")
        assert started.wait(5)
        active.cancel("r")
        assert wait_done(active)["status"] == "cancelled"
        with pytest.raises(ValueError, match="terminal"):
            active.resume("r")
        assert active.report("r") is None
    finally:
        active.close()


def test_one_sqlite_service_owner_prevents_duplicate_startup_recovery(tmp_path):
    active = service(tmp_path)
    try:
        with pytest.raises(ValueError, match="active service owner"):
            service(tmp_path)
        active.submit("synthetic", run_id="r")
        assert wait_done(active)["status"] == "success"
    finally:
        active.close()


def test_native_max_concurrency_limits_dynamic_workers_within_one_case(tmp_path):
    running, peak = set(), 0
    async def observe(stage, payload, context):
        nonlocal peak
        if stage == "worker_plan" and payload["step"] == 0:
            running.add(payload["task"]["task_id"])
            peak = max(peak, len(running))
        if stage == "worker_judge" and payload["step"] >= 2:
            running.discard(payload["task"]["task_id"])
        await asyncio.sleep(.01)
    class Fanout(ScriptedReasoner):
        def response(self, stage, payload):
            value = super().response(stage, payload)
            if stage == "plan":
                original = value["tasks"][0]
                value["tasks"] = [{**original, "task_id": str(i)} for i in range(3)]
            return value
    active = service(tmp_path, reasoner=Fanout(observer=observe), config=CaseExecutionConfig(case_task_limit=1))
    try:
        active.submit("synthetic", run_id="r")
        assert wait_done(active)["status"] == "success"
        assert active.status("r")["task_total"] == 3 and peak == 1
    finally:
        active.close()

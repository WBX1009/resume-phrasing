"""Behavioral contracts for native task execution, independent of API projections."""
import asyncio
from pathlib import Path

import pytest
from case_graph_support import run_tasks
import httpx

from src.case_execution import CaseTaskScope, InvestigationTask, SQLiteRunStore, CaseExecutionConfig
from src.case_task_graph import CaseTaskExecutor, merge_outcomes


def task(name, depends_on=()):
    return InvestigationTask(name, "synthetic check", entity_ids=("p",), basis_fact_ids=("f",), allowed_tools=("lookup",), depends_on=depends_on)


SCOPE = CaseTaskScope(frozenset({"p"}), frozenset({"f"}), frozenset({"lookup"}))


def test_native_retry_counts_actual_budget_and_preserves_success(tmp_path):
    async def scenario():
        calls = []
        async def worker(t, data, context):
            async def request():
                calls.append(t.task_id)
                if t.task_id == "retry" and calls.count("retry") == 1:
                    raise ConnectionError("synthetic transient")
                return {"ok": True}
            return await context.llm_call(request, max_output_tokens=10)
        store = SQLiteRunStore(tmp_path / "runs.db")
        result = await run_tasks(CaseTaskExecutor(store, worker), "r", [task("good"), task("retry")], {}, scope=SCOPE)
        assert all(item["status"] == "succeeded" for item in result["outcomes"].values())
        assert calls.count("good") == 1 and calls.count("retry") == 2
        assert store.budget_usage(result["run_id"])["llm_calls"] == 3
        assert store.load_task_projection("r")["retry"]["attempts"] == 2
    asyncio.run(scenario())


def test_native_timeout_exhausts_and_blocks_dependents(tmp_path):
    async def scenario():
        calls = []
        async def worker(t, data, context):
            calls.append(t.task_id)
            if t.task_id == "slow":
                await asyncio.sleep(5)
            return {}
        executor = CaseTaskExecutor(SQLiteRunStore(tmp_path / "runs.db"), worker,
                                    config=CaseExecutionConfig(task_timeout=1, max_attempts=2))
        result = await run_tasks(executor, "r", [task("slow"), task("dependent", ("slow",)), task("good")], {}, scope=SCOPE)
        assert result["outcomes"]["slow"]["error"] == "NodeTimeoutError"
        assert result["outcomes"]["dependent"]["status"] == "blocked"
        assert result["outcomes"]["good"]["status"] == "succeeded"
        assert calls.count("slow") == 2 and "dependent" not in calls
    asyncio.run(scenario())


@pytest.mark.parametrize("error", [KeyError, RuntimeError])
def test_programming_error_stops_graph_instead_of_becoming_business_failure(tmp_path, error):
    async def worker(t, data, context):
        raise error("synthetic implementation defect")
    with pytest.raises(error):
        asyncio.run(run_tasks(CaseTaskExecutor(SQLiteRunStore(tmp_path / "runs.db"), worker), "r", [task("a")], {}, scope=SCOPE))


@pytest.mark.parametrize("status,attempts", [(400, 1), (429, 2), (503, 2)])
def test_http_failures_use_native_retry_with_status_policy(tmp_path, status, attempts):
    async def scenario():
        calls = []
        async def worker(t, data, context):
            calls.append(t.task_id)
            request = httpx.Request("GET", "https://synthetic.invalid")
            response = httpx.Response(status, request=request)
            raise httpx.HTTPStatusError("synthetic provider failure", request=request, response=response)
        executor = CaseTaskExecutor(SQLiteRunStore(tmp_path / "runs.db"), worker,
                                    config=CaseExecutionConfig(max_attempts=2))
        result = await run_tasks(executor, "r", [task("a")], {}, scope=SCOPE)
        assert calls == ["a"] * attempts
        assert result["outcomes"]["a"]["status"] == "failed"
        assert result["outcomes"]["a"]["error"] == "HTTPStatusError"
    asyncio.run(scenario())


def test_missing_result_never_unlocks_a_dependent_task(tmp_path):
    calls = []
    async def worker(t, data, context):
        calls.append(t.task_id)
    with pytest.raises(TypeError, match="explicit result"):
        asyncio.run(run_tasks(CaseTaskExecutor(SQLiteRunStore(tmp_path / "runs.db"), worker),
            "r", [task("a"), task("b", ("a",))], {}, scope=SCOPE))
    assert calls == ["a"]


def test_terminal_reducer_deduplicates_and_rejects_conflicts():
    outcome = {"a": {"status": "succeeded", "result": 1}}
    assert merge_outcomes(outcome, outcome) == outcome
    with pytest.raises(ValueError, match="conflicting"):
        merge_outcomes(outcome, {"a": {"status": "failed"}})


def test_acknowledgement_loss_stays_uncertain_and_is_not_blindly_retried(tmp_path):
    async def scenario():
        calls = []
        async def worker(t, data, context):
            def write(*, idempotency_key):
                calls.append(idempotency_key)
                raise ConnectionError("acknowledgement lost")
            return await context.idempotent_tool_call(write, tool_name="lookup", operation="write")
        result = await run_tasks(CaseTaskExecutor(SQLiteRunStore(tmp_path / "runs.db"), worker), "r", [task("a")], {}, scope=SCOPE)
        assert len(calls) == 1
        assert result["outcomes"]["a"]["error"] == "ExternalOperationUncertain"
    asyncio.run(scenario())

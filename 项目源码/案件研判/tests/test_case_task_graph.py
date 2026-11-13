import asyncio
from pathlib import Path

import pytest
from case_graph_support import run_tasks
from src.case_task_graph import CaseTaskExecutor

from src.case_execution import (
    BudgetLimitExceeded,
    CaseTaskScope,
    InvestigationTask,
    PlanError,
    TaskExecutionError,
    SQLiteRunStore,
    SharedResourceLimits,
    TaskExecutionContext,
    CaseExecutionConfig,
    validate_task_plan,
)


def scope(*, entities=("person-1",), facts=("fact-1",), tools=("lookup",), time_range=None):
    return CaseTaskScope(frozenset(entities), frozenset(facts), frozenset(tools), time_range)


def task(task_id, *, depends_on=(), entities=("person-1",), tools=("lookup",)):
    return InvestigationTask(
        task_id=task_id,
        question=f"Check {task_id}",
        entity_ids=entities,
        basis_fact_ids=("fact-1",),
        allowed_tools=tools,
        depends_on=depends_on,
    )


def test_plan_validation_rejects_unknown_entities_facts_and_tools():
    with pytest.raises(PlanError, match="outside the case"):
        validate_task_plan([task("a", entities=("unknown",))], max_total_tasks=10, scope=scope())
    with pytest.raises(PlanError, match="unknown Fact"):
        validate_task_plan([task("a")], max_total_tasks=10, scope=scope(facts=("other-fact",)))
    with pytest.raises(PlanError, match="whitelist"):
        validate_task_plan([task("a")], max_total_tasks=10, scope=scope(tools=("read_only",)))


def test_plan_validation_rejects_missing_dependencies_cycles_and_task_cap():
    with pytest.raises(PlanError, match="unknown dependencies"):
        validate_task_plan([task("a", depends_on=("missing",))], max_total_tasks=10, scope=scope())
    with pytest.raises(PlanError, match="cycle"):
        validate_task_plan([task("a", depends_on=("b",)), task("b", depends_on=("a",))], max_total_tasks=10, scope=scope())
    with pytest.raises(PlanError, match="total task limit"):
        validate_task_plan([task("a"), task("b")], max_total_tasks=1, scope=scope())


def test_plan_requires_case_scope_fact_support_and_in_range_iso_dates():
    with pytest.raises(PlanError, match="CaseTaskScope is required"):
        validate_task_plan([task("a")], max_total_tasks=10, scope=None)
    with pytest.raises(PlanError, match="supporting Fact"):
        validate_task_plan(
            [InvestigationTask(task_id="a", question="Check", entity_ids=("person-1",))],
            max_total_tasks=10,
            scope=scope(),
        )
    bounded_scope = scope(time_range=("2025-02-01", "2025-04-01"))
    out_of_range = InvestigationTask(
        task_id="a", question="Check", entity_ids=("person-1",), basis_fact_ids=("fact-1",),
        time_range=("2025-01-01", "2025-03-01"),
    )
    with pytest.raises(PlanError, match="outside the case range"):
        validate_task_plan([out_of_range], max_total_tasks=10, scope=bounded_scope)
    with pytest.raises(PlanError, match="valid ISO-8601"):
        InvestigationTask(task_id="a", question="Check", time_range=("not-a-date", "2025-01-01"))


def test_config_rejects_bool_as_integer_and_case_limit_above_server_cap():
    with pytest.raises(ValueError, match="positive integer"):
        CaseExecutionConfig(case_task_limit=True)
    with pytest.raises(ValueError, match="server_task_cap"):
        CaseExecutionConfig(case_task_limit=3, server_task_cap=2)


def test_executor_never_exceeds_limit_and_fills_slot_when_one_task_finishes(tmp_path: Path):
    async def scenario():
        observed = {"active": 0, "maximum": 0, "started": [], "finished": []}
        lock = asyncio.Lock()
        pair_started = asyncio.Event()

        async def worker(current_task, task_input, context):
            async with lock:
                observed["active"] += 1
                observed["maximum"] = max(observed["maximum"], observed["active"])
                observed["started"].append(current_task.task_id)
            if len(observed["started"]) == 2:
                pair_started.set()
            await asyncio.wait_for(pair_started.wait(), timeout=5)
            await asyncio.sleep(0.01)
            async with lock:
                observed["active"] -= 1
                observed["finished"].append(current_task.task_id)
            return {"task_id": current_task.task_id}

        tasks = [task(f"t-{index}") for index in range(7)]
        executor = CaseTaskExecutor(
            SQLiteRunStore(tmp_path / "runs.sqlite3"),
            worker,
            config=CaseExecutionConfig(case_task_limit=2, server_task_cap=4),
        )
        result = await run_tasks(executor, "run-1", tasks, {"facts": ["safe"]}, scope=scope())
        assert all(item["status"] == "succeeded" for item in result["outcomes"].values())
        assert sum(item["status"] == "succeeded" for item in result["outcomes"].values()) == 7
        assert observed["maximum"] == 2
        assert len(observed["started"]) == len(observed["finished"]) == 7

    asyncio.run(scenario())


def test_executor_waits_for_dependencies_but_runs_independent_tasks(tmp_path: Path):
    async def scenario():
        order = []

        async def worker(current_task, task_input, context):
            order.append(("start", current_task.task_id))
            await asyncio.sleep(0.005)
            order.append(("end", current_task.task_id))
            return {"task": current_task.task_id}

        tasks = [task("a"), task("b"), task("c", depends_on=("a",))]
        result = await run_tasks(CaseTaskExecutor(
            SQLiteRunStore(tmp_path / "runs.sqlite3"),
            worker,
            config=CaseExecutionConfig(case_task_limit=2),
        ), "run-1", tasks, {"seed": 1}, scope=scope())

        assert all(item["status"] == "succeeded" for item in result["outcomes"].values())
        assert order.index(("end", "a")) < order.index(("start", "c"))
        assert ("start", "a") in order and ("start", "b") in order

    asyncio.run(scenario())


def test_failed_prerequisite_blocks_descendants_and_keeps_independent_work(tmp_path: Path):
    async def scenario():
        async def worker(current_task, task_input, context):
            if current_task.task_id == "bad":
                raise TaskExecutionError("tool unavailable")
            return {"ok": True}

        result = await run_tasks(CaseTaskExecutor(
            SQLiteRunStore(tmp_path / "runs.sqlite3"), worker, config=CaseExecutionConfig(case_task_limit=2)
        ), "run-1", [task("bad"), task("dependent", depends_on=("bad",)), task("independent")], {}, scope=scope())

        assert any(item["status"] != "succeeded" for item in result["outcomes"].values())
        assert result["outcomes"]["bad"]["status"] == "failed"
        assert result["outcomes"]["dependent"]["status"] == "blocked"
        assert result["outcomes"]["independent"]["status"] == "succeeded"

    asyncio.run(scenario())


def test_task_inputs_are_isolated_and_llm_budget_is_shared_and_atomic(tmp_path: Path):
    async def scenario():
        async def worker(current_task, task_input, context):
            task_input["case_facts"]["mutable"].append(current_task.task_id)
            async def model_call():
                await asyncio.sleep(0.005)
                return {"ok": True}
            return await context.llm_call(model_call, max_output_tokens=500)

        store = SQLiteRunStore(tmp_path / "runs.sqlite3")
        executor = CaseTaskExecutor(
            store,
            worker,
            config=CaseExecutionConfig(case_task_limit=2, max_llm_calls=1, max_reserved_tokens=500),
            resources=SharedResourceLimits(llm_request_cap=1, db_query_cap=1),
        )
        facts = {"mutable": []}
        result = await run_tasks(executor, "run-1", [task("a"), task("b")], facts, scope=scope())

        assert facts == {"mutable": []}
        assert sum(item["status"] == "failed" for item in result["outcomes"].values()) == 1
        assert sum(item["status"] == "succeeded" for item in result["outcomes"].values()) == 1
        assert store.budget_usage(result["run_id"]) == {"llm_calls": 1, "tool_calls": 0, "reserved_tokens": 500}
        assert any("BudgetLimitExceeded" in (run["error"] or "") for run in result["outcomes"].values())

    asyncio.run(scenario())


def test_tool_call_rejects_non_whitelisted_tool_before_spending_budget(tmp_path: Path):
    async def scenario():
        store = SQLiteRunStore(tmp_path / "runs.sqlite3")
        config = CaseExecutionConfig()
        task_spec = task("a", tools=("lookup",))
        store.prepare_task_projection("run-1", [task_spec], config, scope())
        context = TaskExecutionContext(
            run_id="run-1",
            task=task_spec,
            idempotency_key="run-1:a",
            store=store,
            config=config,
            resources=SharedResourceLimits(),
        )
        with pytest.raises(PlanError, match="non-whitelisted"):
            await context.tool_call(lambda: "must not run", tool_name="write_record")
        assert store.budget_usage("run-1")["tool_calls"] == 0

    asyncio.run(scenario())


def test_shared_database_cap_applies_across_concurrent_case_runs(tmp_path: Path):
    async def scenario():
        observed = {"active": 0, "maximum": 0, "calls": 0}
        lock = asyncio.Lock()
        resources = SharedResourceLimits(llm_request_cap=2, db_query_cap=1)

        async def database_call(case_id):
            async with lock:
                observed["active"] += 1
                observed["maximum"] = max(observed["maximum"], observed["active"])
            await asyncio.sleep(0.02)
            async with lock:
                observed["active"] -= 1
                observed["calls"] += 1
            return {"case": case_id}

        async def worker(current_task, task_input, context):
            return await context.tool_call(
                database_call,
                task_input["case_facts"]["case"],
                tool_name="lookup",
            )

        config = CaseExecutionConfig(case_task_limit=2, server_task_cap=2)
        store = SQLiteRunStore(tmp_path / "runs.sqlite3")
        executor = CaseTaskExecutor(store, worker, config=config, resources=resources)
        results = await asyncio.gather(
            run_tasks(executor, "case-a", [task("a1"), task("a2")], {"case": "a"}, scope=scope()),
            run_tasks(executor, "case-b", [task("b1"), task("b2")], {"case": "b"}, scope=scope()),
        )

        assert all(all(item["status"] == "succeeded" for item in result["outcomes"].values()) for result in results)
        assert observed["maximum"] == 1
        assert observed["calls"] == 4
        assert sum(sum(item["status"] == "succeeded" for item in result["outcomes"].values()) for result in results) == 4

    asyncio.run(scenario())


def test_admission_rejects_replaced_plan_or_budget(tmp_path):
    store = SQLiteRunStore(tmp_path / "runs.db")
    state = {"run_id": "r", "tasks": [task("a").to_dict()]}
    store.admit(state, CaseExecutionConfig(), scope())
    with pytest.raises(ValueError, match="different plan"):
        store.admit({**state, "tasks": [task("b").to_dict()]}, CaseExecutionConfig(), scope())
    with pytest.raises(ValueError, match="different scope or budget"):
        store.admit(state, CaseExecutionConfig(max_llm_calls=1), scope())


def test_failure_blocks_all_descendants_even_in_reverse_dependency_order(tmp_path):
    async def scenario():
        calls = []
        async def worker(t, data, context):
            calls.append(t.task_id)
            if t.task_id == "z":
                raise TaskExecutionError("synthetic tool failure")
            return {}
        result = await run_tasks(CaseTaskExecutor(SQLiteRunStore(tmp_path / "runs.sqlite3"), worker),
            "r", [task("a", depends_on=("b",)), task("b", depends_on=("z",)), task("z"), task("independent")], {}, scope=scope())
        assert result["outcomes"]["a"]["status"] == result["outcomes"]["b"]["status"] == "blocked"
        assert set(calls) == {"z", "independent"}
    asyncio.run(scenario())

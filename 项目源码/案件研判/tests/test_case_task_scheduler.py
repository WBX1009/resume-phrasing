import asyncio
from pathlib import Path

import pytest

from src.case_task_scheduler import (
    BudgetLimitExceeded,
    CaseTaskScheduler,
    CaseTaskScope,
    InvestigationTask,
    PlanError,
    SQLiteRunStore,
    SharedResourceLimits,
    TaskExecutionContext,
    TaskSchedulerConfig,
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
        TaskSchedulerConfig(case_task_limit=True)
    with pytest.raises(ValueError, match="server_task_cap"):
        TaskSchedulerConfig(case_task_limit=3, server_task_cap=2)


def test_scheduler_never_exceeds_limit_and_fills_slot_when_one_task_finishes(tmp_path: Path):
    async def scenario():
        observed = {"active": 0, "maximum": 0, "started": [], "finished": []}
        lock = asyncio.Lock()

        async def worker(current_task, task_input, context):
            async with lock:
                observed["active"] += 1
                observed["maximum"] = max(observed["maximum"], observed["active"])
                observed["started"].append(current_task.task_id)
            await asyncio.sleep(0.01)
            async with lock:
                observed["active"] -= 1
                observed["finished"].append(current_task.task_id)
            return {"task_id": current_task.task_id}

        tasks = [task(f"t-{index}") for index in range(7)]
        scheduler = CaseTaskScheduler(
            SQLiteRunStore(tmp_path / "runs.sqlite3"),
            worker,
            config=TaskSchedulerConfig(case_task_limit=2, server_task_cap=4),
        )
        result = await scheduler.run("run-1", tasks, {"facts": ["safe"]}, scope=scope())
        assert result.status == "success"
        assert result.succeeded_count == 7
        assert observed["maximum"] == result.max_active_tasks == 2
        assert len(observed["started"]) == len(observed["finished"]) == 7

    asyncio.run(scenario())


def test_scheduler_waits_for_dependencies_but_runs_independent_tasks(tmp_path: Path):
    async def scenario():
        order = []

        async def worker(current_task, task_input, context):
            order.append(("start", current_task.task_id))
            await asyncio.sleep(0.005)
            order.append(("end", current_task.task_id))
            return {"task": current_task.task_id}

        tasks = [task("a"), task("b"), task("c", depends_on=("a",))]
        result = await CaseTaskScheduler(
            SQLiteRunStore(tmp_path / "runs.sqlite3"),
            worker,
            config=TaskSchedulerConfig(case_task_limit=2),
        ).run("run-1", tasks, {"seed": 1}, scope=scope())

        assert result.status == "success"
        assert order.index(("end", "a")) < order.index(("start", "c"))
        assert ("start", "a") in order and ("start", "b") in order

    asyncio.run(scenario())


def test_failed_prerequisite_blocks_descendants_and_keeps_independent_work(tmp_path: Path):
    async def scenario():
        async def worker(current_task, task_input, context):
            if current_task.task_id == "bad":
                raise RuntimeError("tool unavailable")
            return {"ok": True}

        result = await CaseTaskScheduler(
            SQLiteRunStore(tmp_path / "runs.sqlite3"), worker, config=TaskSchedulerConfig(case_task_limit=2)
        ).run("run-1", [task("bad"), task("dependent", depends_on=("bad",)), task("independent")], {}, scope=scope())

        assert result.status == "failed"
        assert result.tasks["bad"]["status"] == "failed"
        assert result.tasks["dependent"]["status"] == "blocked"
        assert result.tasks["independent"]["status"] == "succeeded"

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
        scheduler = CaseTaskScheduler(
            store,
            worker,
            config=TaskSchedulerConfig(case_task_limit=2, max_llm_calls=1, max_reserved_tokens=500),
            resources=SharedResourceLimits(llm_request_cap=1, db_query_cap=1),
        )
        facts = {"mutable": []}
        result = await scheduler.run("run-1", [task("a"), task("b")], facts, scope=scope())

        assert facts == {"mutable": []}
        assert result.failed_count == 1
        assert result.succeeded_count == 1
        assert result.budget == {"llm_calls": 1, "tool_calls": 0, "reserved_tokens": 500}
        assert any("BudgetLimitExceeded" in (run["error"] or "") for run in result.tasks.values())

    asyncio.run(scenario())


def test_tool_call_rejects_non_whitelisted_tool_before_spending_budget(tmp_path: Path):
    async def scenario():
        store = SQLiteRunStore(tmp_path / "runs.sqlite3")
        config = TaskSchedulerConfig()
        task_spec = task("a", tools=("lookup",))
        store.start_run("run-1", [task_spec], config, scope())
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

        config = TaskSchedulerConfig(case_task_limit=2, server_task_cap=2)
        store = SQLiteRunStore(tmp_path / "runs.sqlite3")
        scheduler = CaseTaskScheduler(store, worker, config=config, resources=resources)
        results = await asyncio.gather(
            scheduler.run("case-a", [task("a1"), task("a2")], {"case": "a"}, scope=scope()),
            scheduler.run("case-b", [task("b1"), task("b2")], {"case": "b"}, scope=scope()),
        )

        assert all(result.status == "success" for result in results)
        assert observed["maximum"] == 1
        assert observed["calls"] == 4
        assert sum(result.succeeded_count for result in results) == 4

    asyncio.run(scenario())


def test_run_id_rejects_different_plan_or_budget(tmp_path: Path):
    async def scenario():
        async def worker(current_task, task_input, context):
            return {"ok": True}

        scheduler = CaseTaskScheduler(SQLiteRunStore(tmp_path / "runs.sqlite3"), worker)
        await scheduler.run("stable-id", [task("a")], {}, scope=scope())
        with pytest.raises(ValueError, match="different plan, scope, or budget"):
            await scheduler.run("stable-id", [task("b")], {}, scope=scope())

    asyncio.run(scenario())


def test_interrupted_running_task_is_retried_with_the_same_idempotency_key(tmp_path: Path):
    async def scenario():
        store = SQLiteRunStore(tmp_path / "runs.sqlite3")
        plan = [task("a")]
        config = TaskSchedulerConfig()
        store.start_run("run-1", plan, config, scope())
        store.save_task("run-1", "a", "running", increment_attempt=True)
        old_key = store.load_tasks("run-1")["a"]["idempotency_key"]
        seen = []

        async def worker(current_task, task_input, context):
            seen.append(context.idempotency_key)
            return {"recovered": True}

        result = await CaseTaskScheduler(store, worker, config=config).run("run-1", plan, {}, scope=scope())

        assert result.status == "success"
        assert seen == [old_key]
        assert result.tasks["a"]["attempts"] == 2

    asyncio.run(scenario())


def test_targeted_recheck_tasks_append_without_resetting_run_budget_or_results(tmp_path: Path):
    async def scenario():
        calls = []

        async def worker(current_task, task_input, context):
            calls.append(current_task.task_id)
            return {"task": current_task.task_id}

        scheduler = CaseTaskScheduler(SQLiteRunStore(tmp_path / "runs.sqlite3"), worker)
        original = [task("initial")]
        first = await scheduler.run("run-1", original, {}, scope=scope())
        with pytest.raises(ValueError, match="scope differs"):
            scheduler.append_recheck_tasks(
                "run-1",
                [task("outside-scope", entities=("person-2",))],
                scope=scope(entities=("person-1", "person-2")),
            )
        scheduler.append_recheck_tasks("run-1", [task("recheck", depends_on=("initial",))], scope=scope())
        second = await scheduler.run("run-1", original, {}, scope=scope())

        assert first.status == second.status == "success"
        assert calls == ["initial", "recheck"]
        assert second.tasks["initial"]["attempts"] == 1
        assert second.tasks["recheck"]["status"] == "succeeded"

    asyncio.run(scenario())

import asyncio
from pathlib import Path

import pytest

from src.case_task_scheduler import (
    BudgetLimitExceeded,
    CaseTaskScope,
    ExternalOperationUncertain,
    InvestigationTask,
    SharedResourceLimits,
    SQLiteRunStore,
    TaskExecutionContext,
    TaskSchedulerConfig,
)
from src.case_worker_subgraph import SupplementWorkerSubgraph


def _context(tmp_path: Path, run_id: str, task: InvestigationTask) -> TaskExecutionContext:
    store = SQLiteRunStore(tmp_path / f"{run_id}.sqlite3")
    scope = CaseTaskScope(
        entity_ids=frozenset({"person-1"}),
        fact_ids=frozenset({"fact-1"}),
        allowed_tools=frozenset({"lookup"}),
    )
    config = TaskSchedulerConfig()
    store.start_run(run_id, [task], config, scope)
    return TaskExecutionContext(
        run_id=run_id,
        task=task,
        idempotency_key=f"{run_id}:{task.task_id}",
        store=store,
        config=config,
        resources=SharedResourceLimits(),
    )


def test_reusable_supplement_subgraph_keeps_task_contexts_isolated(tmp_path: Path):
    async def scenario():
        task_a = InvestigationTask(
            task_id="a",
            question="question-a",
            entity_ids=("person-1",),
            basis_fact_ids=("fact-1",),
            allowed_tools=("lookup",),
        )
        task_b = InvestigationTask(
            task_id="b",
            question="question-b",
            entity_ids=("person-1",),
            basis_fact_ids=("fact-1",),
            allowed_tools=("lookup",),
        )

        async def worker(task, task_input, context):
            return {
                "task_id": task.task_id,
                "question": task_input["task"]["question"],
                "run_id": context.run_id,
            }

        graph = SupplementWorkerSubgraph(worker)
        result_a, result_b = await asyncio.gather(
            graph(task_a, {"task": task_a.to_dict()}, _context(tmp_path, "run-a", task_a)),
            graph(task_b, {"task": task_b.to_dict()}, _context(tmp_path, "run-b", task_b)),
        )

        assert result_a == {"task_id": "a", "question": "question-a", "run_id": "run-a"}
        assert result_b == {"task_id": "b", "question": "question-b", "run_id": "run-b"}

    asyncio.run(scenario())


def test_idempotent_tool_call_returns_committed_result_without_repeating_side_effect(tmp_path: Path):
    async def scenario():
        task = InvestigationTask(
            task_id="write",
            question="write one record",
            entity_ids=("person-1",),
            basis_fact_ids=("fact-1",),
            allowed_tools=("lookup",),
        )
        context = _context(tmp_path, "run-write", task)
        calls = []

        async def external_write(value: str, *, idempotency_key: str):
            calls.append((value, idempotency_key))
            return {"stored": value, "key": idempotency_key}

        first = await context.idempotent_tool_call(
            external_write,
            "payload",
            tool_name="lookup",
            operation="persist-result",
        )
        second = await context.idempotent_tool_call(
            external_write,
            "payload",
            tool_name="lookup",
            operation="persist-result",
        )

        assert first == second
        assert calls == [("payload", "run-write:write:persist-result")]
        assert context.store.budget_usage("run-write")["tool_calls"] == 1

    asyncio.run(scenario())


def test_idempotent_tool_call_fails_closed_for_unresolved_prior_operation(tmp_path: Path):
    async def scenario():
        task = InvestigationTask(
            task_id="write",
            question="write one record",
            entity_ids=("person-1",),
            basis_fact_ids=("fact-1",),
            allowed_tools=("lookup",),
        )
        context = _context(tmp_path, "run-uncertain", task)
        context.store.begin_effect("run-uncertain", "run-uncertain:write:persist-result")
        calls = []

        async def external_write(*, idempotency_key: str):
            calls.append(idempotency_key)
            return {"ok": True}

        with pytest.raises(ExternalOperationUncertain):
            await context.idempotent_tool_call(
                external_write,
                tool_name="lookup",
                operation="persist-result",
            )
        assert calls == []

    asyncio.run(scenario())


def test_idempotent_budget_rejection_does_not_poison_effect_state(tmp_path: Path):
    async def scenario():
        task = InvestigationTask(
            task_id="write",
            question="write one record",
            entity_ids=("person-1",),
            basis_fact_ids=("fact-1",),
            allowed_tools=("lookup",),
        )
        store = SQLiteRunStore(tmp_path / "runs.sqlite3")
        config = TaskSchedulerConfig(max_tool_calls=1)
        scope = CaseTaskScope(
            entity_ids=frozenset({"person-1"}),
            fact_ids=frozenset({"fact-1"}),
            allowed_tools=frozenset({"lookup"}),
        )
        store.start_run("run-budget", [task], config, scope)
        context = TaskExecutionContext(
            run_id="run-budget",
            task=task,
            idempotency_key="run-budget:write",
            store=store,
            config=config,
            resources=SharedResourceLimits(),
        )

        async def external_write(*, idempotency_key: str):
            return {"key": idempotency_key}

        await context.idempotent_tool_call(
            external_write,
            tool_name="lookup",
            operation="first",
        )
        with pytest.raises(BudgetLimitExceeded):
            await context.idempotent_tool_call(
                external_write,
                tool_name="lookup",
                operation="second",
            )

        with store._connect() as connection:
            row = connection.execute(
                "SELECT status, error FROM case_effects WHERE run_id = ? AND operation_key = ?",
                ("run-budget", "run-budget:write:second"),
            ).fetchone()
        assert dict(row) == {"status": "failed", "error": "BudgetLimitExceeded"}

    asyncio.run(scenario())

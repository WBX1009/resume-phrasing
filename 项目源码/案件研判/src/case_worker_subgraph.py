"""Reusable LangGraph subgraph for one isolated case-supplement task.

The parent case graph owns planning, scheduling, budgets, and durable task
results.  This subgraph owns only one Worker invocation, so each task receives
its own state object and cannot accidentally share another task's local
messages or intermediate result.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
import inspect
from typing import Any, TypedDict
from uuid import uuid4

from langgraph.graph import END, START, StateGraph

from .case_task_scheduler import InvestigationTask, TaskExecutionContext


class SupplementWorkerState(TypedDict, total=False):
    task: dict[str, Any]
    task_input: dict[str, Any]
    invocation_id: str
    result: Any


WorkerCallback = Callable[
    [InvestigationTask, dict[str, Any], TaskExecutionContext],
    Any | Awaitable[Any],
]


class SupplementWorkerSubgraph:
    """Compile once and invoke the same isolated Worker graph for each task."""

    def __init__(self, worker: WorkerCallback):
        if not inspect.iscoroutinefunction(worker):
            raise TypeError("supplement worker must be an async callable")
        self.worker = worker
        self._contexts: dict[str, TaskExecutionContext] = {}
        builder = StateGraph(SupplementWorkerState)
        builder.add_node("worker", self._worker_node)
        builder.add_edge(START, "worker")
        builder.add_edge("worker", END)
        # The parent TaskRun ledger persists the result.  This child graph is
        # intentionally invocation-scoped so task-local state is not shared.
        self.graph = builder.compile()

    async def __call__(
        self,
        task: InvestigationTask,
        task_input: dict[str, Any],
        context: TaskExecutionContext,
    ) -> Any:
        invocation_id = uuid4().hex
        self._contexts[invocation_id] = context
        try:
            result = await self.graph.ainvoke(
                {
                    "task": task.to_dict(),
                    "task_input": task_input,
                    "invocation_id": invocation_id,
                }
            )
        finally:
            self._contexts.pop(invocation_id, None)
        return result.get("result") if isinstance(result, Mapping) else None

    async def _worker_node(self, state: SupplementWorkerState) -> dict[str, Any]:
        task = InvestigationTask.from_dict(state["task"])
        task_input = dict(state["task_input"])
        context = self._contexts.get(state["invocation_id"])
        if context is None:
            raise TypeError("supplement worker context is missing")
        value = self.worker(task, task_input, context)
        if inspect.isawaitable(value):
            value = await value
        return {"result": value}

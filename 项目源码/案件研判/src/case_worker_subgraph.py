"""Reusable LangGraph subgraph for one isolated case-supplement task.

The parent graph persists the business plan and outcomes. LangGraph owns
execution, retries and recovery; application services own atomic budgets.  This subgraph owns only one Worker invocation, so each task receives
its own state object and cannot accidentally share another task's local
messages or intermediate result.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
import inspect
from typing import Any, TypedDict
from langgraph.runtime import Runtime
from langgraph.types import RetryPolicy, Command
from langgraph.errors import NodeError, NodeTimeoutError, NodeCancelledError
from copy import deepcopy

from langgraph.graph import END, START, StateGraph

from .case_execution import InvestigationTask, TaskExecutionContext, TaskExecutionError, PlanError, BudgetLimitExceeded, ExternalOperationUncertain


_NATIVE_RETRY_ON = RetryPolicy().retry_on


def http_status(error):
    return getattr(error, "status_code", None) or getattr(getattr(error, "response", None), "status_code", None)


def retryable_error(error):
    """Use LangGraph's classifier, with explicit timeout and HTTP policy."""
    if isinstance(error, (TimeoutError, NodeTimeoutError)):
        return True
    status = http_status(error)
    if isinstance(status, int):
        return status in {408, 429} or 500 <= status < 600
    return _NATIVE_RETRY_ON(error)


class SupplementWorkerState(TypedDict, total=False):
    task: dict[str, Any]
    task_input: dict[str, Any]
    status: str
    error: str | None
    result: Any


WorkerCallback = Callable[
    [InvestigationTask, dict[str, Any], TaskExecutionContext],
    Any | Awaitable[Any],
]


class SupplementWorkerSubgraph:
    """Compile once and invoke the same isolated Worker graph for each task."""

    def __init__(self, worker: WorkerCallback, *, max_attempts=3, timeout=120):
        if not inspect.iscoroutinefunction(worker):
            raise TypeError("supplement worker must be an async callable")
        self.worker = worker
        builder = StateGraph(SupplementWorkerState, context_schema=TaskExecutionContext)
        builder.add_node("worker", self._worker_node, retry_policy=RetryPolicy(max_attempts=max_attempts, retry_on=retryable_error, initial_interval=0.05), timeout=timeout, error_handler=self._failed)
        builder.add_edge(START, "worker")
        builder.add_edge("worker", END)
        self.graph = builder.compile()

    async def _worker_node(self, state: SupplementWorkerState, runtime: Runtime[TaskExecutionContext]) -> dict[str, Any]:
        import asyncio
        import json
        context = runtime.context
        if context.store.is_cancel_requested(context.run_id):
            raise asyncio.CancelledError
        context.store.record_task_status(context.run_id, context.task.task_id, "running", increment_attempt=True)
        value = await self.worker(InvestigationTask.from_dict(state["task"]), deepcopy(state["task_input"]), context)
        if value is None:
            raise TypeError("worker must return an explicit result; use structured no-data evidence when appropriate")
        json.dumps(value, allow_nan=False)
        return {"result": value, "status": "succeeded", "error": None}

    def _failed(self, state: SupplementWorkerState, error: NodeError):
        exc = error.error
        known_failure = isinstance(exc, (TaskExecutionError, PlanError, BudgetLimitExceeded, ExternalOperationUncertain))
        if isinstance(exc, NodeCancelledError) or not (known_failure or http_status(exc) is not None or retryable_error(exc)):
            raise exc
        return Command(update={"result": None, "status": "failed", "error": type(exc).__name__}, goto=END)

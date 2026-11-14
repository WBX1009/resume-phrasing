"""One Task, one native Planner -> Tool -> Judge graph with bounded feedback."""
from __future__ import annotations

import asyncio
from typing import Any, TypedDict
from langgraph.runtime import Runtime, get_runtime
from langgraph.func import task as durable_task
from langgraph.graph import START, END, StateGraph
from langgraph.types import RetryPolicy
from langgraph.errors import NodeTimeoutError, NodeCancelledError
from pydantic import ValidationError

from .case_execution import TaskExecutionContext, TaskExecutionError, PlanError, BudgetLimitExceeded, ExternalOperationUncertain
from .investigation_contracts import WorkerDecision, WorkerJudgment
from .investigation_domain import fact_context, validate_assessments

_NATIVE_RETRY_ON = RetryPolicy().retry_on


def http_status(error):
    return getattr(error, "status_code", None) or getattr(getattr(error, "response", None), "status_code", None)


def retryable_error(error):
    """Use LangGraph's classifier, with explicit timeout and HTTP policy."""
    if isinstance(error, (TaskExecutionError, PlanError, BudgetLimitExceeded, ExternalOperationUncertain, ValidationError)):
        return False
    if isinstance(error, (TimeoutError, NodeTimeoutError)):
        return True
    status = http_status(error)
    if isinstance(status, int):
        return status in {408, 429} or 500 <= status < 600
    if isinstance(error, (RuntimeError, KeyError, TypeError, ValueError, AssertionError)):
        return False
    return _NATIVE_RETRY_ON(error)


class SupplementWorkerState(TypedDict, total=False):
    task: dict[str, Any]
    context_fact_ids: list[str]
    observed_fact_ids: list[str]
    goal: dict[str, Any]
    step: int
    decision: dict[str, Any]
    assessments: list[dict[str, Any]]
    done: bool
    status: str
    result: dict[str, Any] | None
    error: str | None


class SupplementWorkerSubgraph:
    def __init__(self, reasoner, tools, *, max_attempts=3, timeout=120):
        self.reasoner, self.tools = reasoner, tools
        builder = StateGraph(SupplementWorkerState, context_schema=TaskExecutionContext)
        policy = RetryPolicy(max_attempts=max_attempts, retry_on=retryable_error, initial_interval=.05)
        for name, callback in (("plan_step", self._plan), ("tool", self._tool), ("judge", self._judge)):
            builder.add_node(name, self._operation_boundary(callback, name, policy, timeout))
        builder.add_node("finish", self._finish)
        builder.add_edge(START, "plan_step")
        builder.add_conditional_edges("plan_step", lambda s: END if s.get("status") == "failed" else
            "tool" if s["decision"]["action"] == "tool" else "judge", ["tool", "judge", END])
        builder.add_conditional_edges("tool", lambda s: END if s.get("status") == "failed" else "judge", ["judge", END])
        builder.add_conditional_edges("judge", lambda s: END if s.get("status") == "failed" else
            "finish" if s["done"] else "plan_step", ["finish", "plan_step", END])
        builder.add_edge("finish", END)
        self.graph = builder.compile()

    def _operation_boundary(self, callback, name, policy, timeout):
        # Native @task performs retries, timeout and durable result reuse. The
        # surrounding node translates only exhausted, known business/provider
        # failures into an outcome. Unknown defects propagate without turning
        # an error-handler invocation into a completed worker checkpoint.
        @durable_task(name=f"worker_{name}", retry_policy=policy, timeout=timeout)
        async def operation(state):
            return await callback(state, get_runtime(TaskExecutionContext))

        async def boundary(state):
            try:
                return await operation(state)
            except Exception as exc:
                return self._failed(exc, state)
        return boundary

    def _check(self, context, node):
        if context.store.is_cancel_requested(context.run_id):
            raise asyncio.CancelledError
        context.store.record_task_status(context.run_id, context.task.task_id, "running")
        context.store.record_event(context.run_id, f"worker:{context.task.task_id}:{node}", "started")

    def _payload(self, state, context):
        return {"task": state["task"], "goal": state["goal"], "step": state.get("step", 0),
                "observed_fact_ids": state.get("observed_fact_ids", []),
                "facts": fact_context(context.store, context.run_id, state["context_fact_ids"],
                    limit=context.config.max_context_facts, max_chars=context.config.max_context_chars),
                "tools": self.tools.describe(context.task.allowed_tools)}

    async def _plan(self, state, runtime: Runtime[TaskExecutionContext]):
        context = runtime.context
        self._check(context, "plan_step")
        if state.get("step", 0) >= context.config.max_worker_steps:
            return {"decision": {"action": "finish", "request": None, "reason": "达到单任务查询步数上限"}}
        raw = await self.reasoner.decide("worker_plan", self._payload(state, context), context)
        decision = WorkerDecision.model_validate(raw)
        if decision.request:
            self.tools.authorize(decision.request, context)
        return {"decision": decision.model_dump(mode="json")}

    async def _tool(self, state, runtime: Runtime[TaskExecutionContext]):
        context = runtime.context
        self._check(context, "tool")
        value = await context.execute_tool(state["decision"]["request"])
        observed = list(dict.fromkeys(state.get("observed_fact_ids", []) + value["fact_ids"]))
        return {"observed_fact_ids": observed, "context_fact_ids": list(dict.fromkeys(state["context_fact_ids"] + observed)),
                "step": state.get("step", 0)+1}

    async def _judge(self, state, runtime: Runtime[TaskExecutionContext]):
        context = runtime.context
        self._check(context, "judge")
        judgment = WorkerJudgment.model_validate(await self.reasoner.decide("worker_judge", self._payload(state, context), context))
        facts = context.store.facts(context.run_id, state["context_fact_ids"])
        assessments = validate_assessments([a.model_dump(mode="json") for a in judgment.assessments],
            goals=[state["goal"]], facts=facts, allowed_ids=state["context_fact_ids"], task=context.task)
        done = judgment.done or state["decision"]["action"] == "finish" or state.get("step", 0) >= context.config.max_worker_steps
        return {"assessments": assessments, "done": done}

    def _finish(self, state):
        return {"status": "succeeded", "error": None, "result": {
            "fact_ids": state.get("observed_fact_ids", []), "context_fact_ids": state["context_fact_ids"],
            "assessments": state.get("assessments", []), "steps": state.get("step", 0)}}

    def _failed(self, exc, state):
        known = isinstance(exc, (TaskExecutionError, PlanError, ValidationError, BudgetLimitExceeded, ExternalOperationUncertain))
        if isinstance(exc, NodeCancelledError) or not (known or http_status(exc) is not None or retryable_error(exc)):
            raise exc
        return {"result": {"fact_ids": state.get("observed_fact_ids", []),
            "context_fact_ids": state["context_fact_ids"], "assessments": [], "steps": state.get("step", 0)},
            "status": "failed", "error": type(exc).__name__}

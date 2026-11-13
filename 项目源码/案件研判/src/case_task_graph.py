"""Business readiness policy on LangGraph's durable execution runtime."""
from __future__ import annotations

import asyncio
from copy import deepcopy
from typing import Annotated, Any, TypedDict

from langchain_core.runnables import RunnableConfig
from langgraph.graph import START, END, StateGraph
from langgraph.types import Send

from .case_execution import (
    CaseTaskScope, InvestigationTask, PlanError,
    RunExecutionContext, SharedResourceLimits, TaskExecutionContext,
    CaseExecutionConfig, validate_task_plan,
)
from .case_worker_subgraph import SupplementWorkerSubgraph


def merge_outcomes(left, right):
    merged = dict(left)
    for task_id, outcome in right.items():
        if task_id in merged and merged[task_id] != outcome:
            raise PlanError(f"conflicting terminal outcome: {task_id}")
        merged[task_id] = outcome
    return merged


class TaskGraphState(TypedDict, total=False):
    run_id: str
    tasks: list[dict[str, Any]]
    case_facts: dict[str, Any]
    case_scope: dict[str, Any]
    outcomes: Annotated[dict[str, dict[str, Any]], merge_outcomes]


def scope_from_state(state):
    value = state["case_scope"]
    return CaseTaskScope(frozenset(value["entity_ids"]), frozenset(value["fact_ids"]),
                         frozenset(value["allowed_tools"]), tuple(value["time_range"]) if value.get("time_range") else None)


class CaseTaskExecutor:
    """Graph construction/dependency adapters; no application execution loop."""
    def __init__(self, store, worker, *, config=CaseExecutionConfig(), resources=None):
        self.store, self.config = store, config
        self.resources = resources or SharedResourceLimits()
        self.worker = SupplementWorkerSubgraph(worker, max_attempts=config.max_attempts, timeout=config.task_timeout)
        self.graph = self.build_graph()

    def workflow_context(self, run_id):
        return RunExecutionContext(run_id, self.store, self.config, self.resources)

    def prepare_run(self, run_id, scope):
        self.store.prepare_run(run_id, self.config, scope)

    def build_graph(self, checkpointer=None):
        builder = StateGraph(TaskGraphState)
        builder.add_node("resolve_ready", self._resolve)
        builder.add_node("worker", self._execute)
        builder.add_edge(START, "resolve_ready")
        builder.add_conditional_edges("resolve_ready", self._route, ["worker", END])
        builder.add_edge("worker", "resolve_ready")
        return builder.compile(checkpointer=checkpointer)

    def _resolve(self, state):
        if self.store.is_cancel_requested(state["run_id"]):
            raise asyncio.CancelledError
        tasks = [InvestigationTask.from_dict(t) for t in state["tasks"]]
        scope = scope_from_state(state)
        validate_task_plan(tasks, max_total_tasks=self.config.max_total_tasks, scope=scope)
        outcomes = dict(state.get("outcomes", {}))
        changed = True
        while changed:
            changed = False
            for task in tasks:
                if task.task_id not in outcomes and any(outcomes.get(d, {}).get("status") in {"failed", "blocked"} for d in task.depends_on):
                    outcomes[task.task_id] = {"status": "blocked", "result": None, "error": "a prerequisite task did not complete"}
                    changed = True
        self.store.project(state["run_id"], tasks, outcomes, self.config, scope)
        return {"outcomes": outcomes}

    def _route(self, state):
        outcomes = state.get("outcomes", {})
        pending = [t for t in state["tasks"] if t["task_id"] not in outcomes]
        ready = [t for t in pending if all(outcomes.get(d, {}).get("status") == "succeeded" for d in t.get("depends_on", []))]
        if pending and not ready:
            raise PlanError("unfinished plan has no ready task")
        return [Send("worker", {"run_id": state["run_id"], "task": t,
                    "task_input": {"task": t, "case_facts": deepcopy(state["case_facts"]),
                        "dependency_results": {d: deepcopy(outcomes[d]["result"]) for d in t.get("depends_on", [])}, "task_state": {}}})
                for t in ready] or END

    async def _execute(self, state, config: RunnableConfig):
        task = InvestigationTask.from_dict(state["task"])
        context = TaskExecutionContext(state["run_id"], task, f"{state['run_id']}:{task.task_id}", self.store, self.config, self.resources)
        output = await self.worker.graph.ainvoke(
            {"task": task.to_dict(), "task_input": state["task_input"]},
            config, context=context,
        )
        return {"outcomes": {task.task_id: {k: output.get(k) for k in ("status", "result", "error")}}}

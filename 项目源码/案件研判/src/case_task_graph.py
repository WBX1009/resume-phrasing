"""Business dependency policy; LangGraph owns invocation and recovery."""
from __future__ import annotations
import asyncio
from typing import Annotated, Any, TypedDict
from langchain_core.runnables import RunnableConfig
from langgraph.graph import START, END, StateGraph
from langgraph.types import Send
from .case_execution import (CaseTaskScope, InvestigationTask, PlanError, RunExecutionContext,
    SharedResourceLimits, TaskExecutionContext, CaseExecutionConfig, validate_task_plan)
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
    fact_ids: list[str]
    goals: list[dict[str, Any]]
    case_scope: dict[str, Any]
    outcomes: Annotated[dict[str, dict[str, Any]], merge_outcomes]


def scope_from_state(state):
    value = state["case_scope"]
    return CaseTaskScope(frozenset(value["entity_ids"]), frozenset(value["allowed_tools"]),
        tuple(value["time_range"]) if value.get("time_range") else None)


class CaseTaskExecutor:
    def __init__(self, store, reasoner, tools, *, config=CaseExecutionConfig(), resources=None):
        self.store, self.config, self.tools = store, config, tools
        self.resources = resources or SharedResourceLimits()
        self.worker = SupplementWorkerSubgraph(reasoner, tools, max_attempts=config.max_attempts, timeout=config.task_timeout)
        builder = StateGraph(TaskGraphState)
        builder.add_node("resolve_ready", self._resolve)
        builder.add_node("worker", self._execute)
        builder.add_edge(START, "resolve_ready")
        builder.add_conditional_edges("resolve_ready", self._route, ["worker", END])
        builder.add_edge("worker", "resolve_ready")
        self.graph = builder.compile()

    def workflow_context(self, run_id):
        return RunExecutionContext(run_id, self.store, self.config, self.resources)

    def _resolve(self, state):
        if self.store.is_cancel_requested(state["run_id"]):
            raise asyncio.CancelledError
        tasks = [InvestigationTask.from_dict(t) for t in state["tasks"]]
        scope = scope_from_state(state)
        order = validate_task_plan(tasks, max_total_tasks=self.config.max_total_tasks, scope=scope,
            known_fact_ids=set(state["fact_ids"]))
        by_id = {t.task_id: t for t in tasks}
        outcomes = dict(state.get("outcomes", {}))
        for tid in order:
            task = by_id[tid]
            if tid not in outcomes and any(outcomes.get(d, {}).get("status") in {"failed", "blocked"} for d in task.depends_on):
                outcomes[tid] = {"status": "blocked", "result": None, "error": "a prerequisite task did not complete"}
        self.store.project(state["run_id"], tasks, outcomes, self.config, scope)
        return {"outcomes": outcomes}

    def _route(self, state):
        outcomes = state.get("outcomes", {})
        pending = [t for t in state["tasks"] if t["task_id"] not in outcomes]
        ready = [t for t in pending if all(outcomes.get(d, {}).get("status") == "succeeded" for d in t.get("depends_on", []))]
        if pending and not ready:
            raise PlanError("unfinished plan has no ready task")
        goals = {g["goal_id"]: g for g in state["goals"]}
        sends = []
        for task in ready:
            dependencies = [fid for d in task.get("depends_on", []) for fid in outcomes[d]["result"]["fact_ids"]]
            ids = list(dict.fromkeys(task["basis_fact_ids"] + dependencies))
            if len(ids) > self.config.max_context_facts:
                raise PlanError("dependency evidence exceeds task context limit")
            goal = dict(goals[task["goal_id"]])
            if task.get("gap_ids"):
                from .investigation_contracts import stable_id
                goal["requirements"] = [r for r in goal["requirements"]
                    if stable_id("gap", [goal["goal_id"], r["requirement_id"]]) in task["gap_ids"]]
            sends.append(Send("worker", {"run_id": state["run_id"], "case_scope": state["case_scope"],
                "task": task, "goal": goal, "context_fact_ids": ids}))
        return sends or END

    async def _execute(self, state, config: RunnableConfig):
        task = InvestigationTask.from_dict(state["task"])
        context = TaskExecutionContext(state["run_id"], task, f"{state['run_id']}:{task.task_id}", self.store,
            self.config, self.resources, scope_from_state(state), self.tools)
        async with self.resources.workers:
            output = await self.worker.graph.ainvoke({"task": state["task"], "goal": state["goal"],
                "context_fact_ids": state["context_fact_ids"], "observed_fact_ids": [], "step": 0}, config, context=context)
        if output.get("status") not in {"succeeded", "failed"}:
            raise RuntimeError("worker exited without a terminal outcome")
        return {"outcomes": {task.task_id: {k: output.get(k) for k in ("status", "result", "error")}}}

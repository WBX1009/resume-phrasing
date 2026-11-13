"""LangGraph case flow with native dynamic task execution.

The callbacks are explicit application boundaries: they must be backed by the
case FactPack, the approved tool registry, the central verifier, and the final
report builder. This module does not invent business event semantics.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping, Sequence
import inspect
from typing import Any, TypedDict

from langchain_core.runnables import RunnableConfig
from copy import deepcopy
from langgraph.graph import END, START, StateGraph

from .case_task_graph import CaseTaskExecutor
from .case_execution import (
    CaseTaskScope,
    InvestigationTask,
    PlanError,
    RunExecutionContext,
)


class CaseWorkflowState(TypedDict, total=False):
    case_id: str
    run_id: str
    case_facts: dict[str, Any]
    case_scope: dict[str, Any]
    initial_tasks: list[dict[str, Any]]
    tasks: list[dict[str, Any]]
    task_outcomes: dict[str, dict[str, Any]]
    execution_status: str
    candidate_events: list[dict[str, Any]]
    verification: dict[str, Any]
    final_events: list[dict[str, Any]]
    human_review: list[Any]
    recheck_round: int
    recheck_added: bool
    report: dict[str, Any]


PlanCallback = Callable[[Mapping[str, Any], RunExecutionContext], Any | Awaitable[Any]]
VerifyCallback = Callable[[Mapping[str, Any], RunExecutionContext], Mapping[str, Any] | Awaitable[Mapping[str, Any]]]
SummaryCallback = Callable[[Mapping[str, Any], RunExecutionContext], Mapping[str, Any] | Awaitable[Mapping[str, Any]]]
RecheckAllowedCallback = Callable[[Mapping[str, Any]], bool | Awaitable[bool]]


def build_case_workflow_graph(
    *,
    planner: PlanCallback,
    executor: CaseTaskExecutor,
    verifier: VerifyCallback,
    recheck_planner: PlanCallback,
    recheck_allowed: RecheckAllowedCallback,
    summarizer: SummaryCallback,
    checkpointer: Any,
    max_recheck_rounds: int = 1,
) -> Any:
    """Build planner → bounded evidence tasks → central verify → limited recheck → summary.

    A checkpointer must be supplied explicitly. Use a durable implementation in
    deployment; an in-memory checkpointer is suitable only for isolated tests.
    Task outcomes and subgraph progress are persisted by this checkpointer.
    SQLite task rows are a rebuildable API projection.
    """
    if checkpointer is None:
        raise ValueError("an explicit LangGraph checkpointer is required")
    if not isinstance(max_recheck_rounds, int) or isinstance(max_recheck_rounds, bool) or max_recheck_rounds < 0:
        raise ValueError("max_recheck_rounds must be a non-negative integer")

    async def plan_node(state: CaseWorkflowState) -> dict[str, Any]:
        _require_input(state)
        scope = _scope_from_dict(state["case_scope"])
        executor.prepare_run(state["run_id"], scope)
        try:
            tasks = await _invoke(planner, state, executor.workflow_context(state["run_id"]))
            task_specs = _serialize_tasks(tasks)
            # The task graph validates the complete plan again before dispatch.
            from .case_execution import validate_task_plan

            validate_task_plan(task_specs, max_total_tasks=executor.config.max_total_tasks, scope=scope)
        except Exception:
            executor.store.finish_run(state["run_id"], "failed")
            raise
        serialized = [task.to_dict() for task in task_specs]
        return {"initial_tasks": serialized, "tasks": serialized, "recheck_round": 0}

    async def dispatch_node(state: CaseWorkflowState, config: RunnableConfig) -> dict[str, Any]:
        config = {**config, "max_concurrency": executor.config.case_task_limit,
                  "recursion_limit": 4*executor.config.max_total_tasks+20}
        tasks = _serialize_tasks(state.get("tasks") or state.get("initial_tasks", []))
        outcome = await executor.graph.ainvoke({
            "run_id": state["run_id"], "tasks": [t.to_dict() for t in tasks],
            "case_facts": deepcopy(state["case_facts"]), "case_scope": state["case_scope"],
            "outcomes": {k: {key: v.get(key) for key in ("status", "result", "error")}
                         for k,v in state.get("task_outcomes", {}).items()},
        }, config)
        records = {t.task_id: {"task": t.to_dict(), **outcome["outcomes"][t.task_id],
                    "idempotency_key": f"{state['run_id']}:{t.task_id}"} for t in tasks}
        status = "success" if all(v["status"] == "succeeded" for v in records.values()) else "failed"
        return {"task_outcomes": records, "execution_status": status}

    async def candidate_events_node(state: CaseWorkflowState) -> dict[str, Any]:
        """Collect only explicitly structured Worker candidate events.

        Evidence text is intentionally not interpreted here.  Production
        Workers must return ``candidate_events`` (or one ``candidate_event``)
        as a structured, traceable result; this node applies the frozen run
        limit before the central verifier sees the candidates.
        """
        executor.store.record_event(state["run_id"], "candidate_events", "started")
        try:
            events: list[dict[str, Any]] = []
            for task_id, record in sorted(state.get("task_outcomes", {}).items()):
                if not isinstance(record, Mapping) or record.get("status") != "succeeded":
                    continue
                result = record.get("result")
                if not isinstance(result, Mapping):
                    continue
                if "candidate_events" in result and "candidate_event" in result:
                    raise PlanError(f"task {task_id} returned both candidate_events and candidate_event")
                if "candidate_events" in result:
                    value = result["candidate_events"]
                    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, Mapping)):
                        raise PlanError(f"task {task_id} candidate_events must be a sequence of mappings")
                    raw_events = list(value)
                elif "candidate_event" in result:
                    raw_events = [result["candidate_event"]]
                else:
                    continue
                for event in raw_events:
                    if not isinstance(event, Mapping) or not event:
                        raise PlanError(f"task {task_id} candidate events must be non-empty mappings")
                    events.append(dict(event))

            if len(events) > executor.config.candidate_limit:
                raise PlanError(
                    "candidate event limit exceeded "
                    f"({len(events)}/{executor.config.candidate_limit})"
                )
            executor.store.record_event(
                state["run_id"],
                "candidate_events",
                "succeeded",
                {"count": len(events), "limit": executor.config.candidate_limit},
            )
            return {"candidate_events": events}
        except Exception:
            executor.store.record_event(state["run_id"], "candidate_events", "failed")
            executor.store.finish_run(state["run_id"], "failed")
            raise

    async def verify_node(state: CaseWorkflowState) -> dict[str, Any]:
        try:
            outcome = await _invoke(verifier, state, executor.workflow_context(state["run_id"]))
            if not isinstance(outcome, Mapping):
                raise TypeError("central verifier must return a mapping")
            if "gaps" not in outcome or not isinstance(outcome["gaps"], Sequence) or isinstance(outcome["gaps"], (str, bytes)):
                raise PlanError("central verifier must return an explicit gaps sequence")
            final_events = _event_sequence(outcome.get("final_events", []), "final_events")
            human_review = _sequence(outcome.get("human_review", []), "human_review")
            return {
                "verification": dict(outcome),
                "final_events": final_events,
                "human_review": human_review,
            }
        except Exception:
            executor.store.finish_run(state["run_id"], "failed")
            raise

    async def plan_rechecks_node(state: CaseWorkflowState) -> dict[str, Any]:
        try:
            scope = _scope_from_dict(state["case_scope"])
            proposed = await _invoke(recheck_planner, state, executor.workflow_context(state["run_id"]))
            tasks = _serialize_tasks(proposed)
            round_number = int(state.get("recheck_round", 0)) + 1
            if tasks:
                merged = [*state.get("tasks", []), *(task.to_dict() for task in tasks)]
                from .case_execution import validate_task_plan
                validate_task_plan(_serialize_tasks(merged), max_total_tasks=executor.config.max_total_tasks, scope=scope)
            else:
                merged = list(state.get("tasks", []))
            return {
                "tasks": merged,
                "recheck_round": round_number,
                "recheck_added": bool(tasks),
            }
        except Exception:
            executor.store.finish_run(state["run_id"], "failed")
            raise

    async def summarize_node(state: CaseWorkflowState) -> dict[str, Any]:
        try:
            report = await _invoke(summarizer, state, executor.workflow_context(state["run_id"]))
            if not isinstance(report, Mapping):
                raise TypeError("summarizer must return a mapping")
            return {"report": dict(report)}
        except Exception:
            executor.store.finish_run(state["run_id"], "failed")
            raise

    async def route_after_verify(state: CaseWorkflowState) -> str:
        try:
            verification = state.get("verification", {})
            gaps = verification.get("gaps", [])
            if not gaps or int(state.get("recheck_round", 0)) >= max_recheck_rounds:
                return "summarize"
            if await _invoke(recheck_allowed, state):
                return "plan_rechecks"
            return "summarize"
        except Exception:
            executor.store.finish_run(state["run_id"], "failed")
            raise

    builder = StateGraph(CaseWorkflowState)
    builder.add_node("plan", plan_node)
    builder.add_node("dispatch", dispatch_node)
    builder.add_node("candidate_events", candidate_events_node)
    builder.add_node("central_verify", verify_node)
    builder.add_node("plan_rechecks", plan_rechecks_node)
    builder.add_node("summarize", summarize_node)
    builder.add_edge(START, "plan")
    builder.add_edge("plan", "dispatch")
    builder.add_edge("dispatch", "candidate_events")
    builder.add_edge("candidate_events", "central_verify")
    builder.add_conditional_edges(
        "central_verify",
        route_after_verify,
        {"plan_rechecks": "plan_rechecks", "summarize": "summarize"},
    )
    builder.add_conditional_edges(
        "plan_rechecks",
        lambda state: "dispatch" if state.get("recheck_added") else "summarize",
        {"dispatch": "dispatch", "summarize": "summarize"},
    )
    builder.add_edge("summarize", END)
    return builder.compile(checkpointer=checkpointer)


async def _invoke(
    callback: Callable[..., Any],
    state: Mapping[str, Any],
    context: RunExecutionContext | None = None,
) -> Any:
    value = callback(state, context) if context is not None else callback(state)
    if inspect.isawaitable(value):
        return await value
    return value


def _serialize_tasks(value: Any) -> list[InvestigationTask]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise PlanError("planner must return a sequence of InvestigationTask objects")
    tasks: list[InvestigationTask] = []
    for item in value:
        tasks.append(item if isinstance(item, InvestigationTask) else InvestigationTask.from_dict(item))
    return tasks


def _sequence(value: Any, label: str) -> list[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, Mapping)):
        raise PlanError(f"central verifier {label} must be a sequence")
    return list(value)


def _event_sequence(value: Any, label: str) -> list[dict[str, Any]]:
    values = _sequence(value, label)
    events: list[dict[str, Any]] = []
    for event in values:
        if not isinstance(event, Mapping) or not event:
            raise PlanError(f"central verifier {label} must contain non-empty mappings")
        events.append(dict(event))
    return events


def _scope_from_dict(value: Any) -> CaseTaskScope:
    if not isinstance(value, Mapping):
        raise PlanError("case_scope must be a trusted mapping")
    try:
        return CaseTaskScope(
            entity_ids=frozenset(value["entity_ids"]),
            fact_ids=frozenset(value["fact_ids"]),
            allowed_tools=frozenset(value["allowed_tools"]),
            time_range=tuple(value["time_range"]) if value.get("time_range") is not None else None,
        )
    except (KeyError, TypeError) as exc:
        raise PlanError("case_scope is incomplete or malformed") from exc


def _require_input(state: Mapping[str, Any]) -> None:
    for key in ("case_id", "run_id", "case_facts", "case_scope"):
        if key not in state:
            raise PlanError(f"workflow input is missing {key}")
    if not isinstance(state["case_id"], str) or not state["case_id"].strip():
        raise PlanError("case_id must be a non-empty string")
    if not isinstance(state["run_id"], str) or not state["run_id"].strip():
        raise PlanError("run_id must be a non-empty string")
    if not isinstance(state["case_facts"], Mapping):
        raise PlanError("case_facts must be a mapping")
    _scope_from_dict(state["case_scope"])

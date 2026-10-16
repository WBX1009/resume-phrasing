"""LangGraph case flow around the bounded, durable task scheduler.

The callbacks are explicit application boundaries: they must be backed by the
case FactPack, the approved tool registry, the central verifier, and the final
report builder. This module does not invent business event semantics.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping, Sequence
import inspect
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from .case_task_scheduler import (
    CaseTaskScope,
    CaseTaskScheduler,
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
    task_runs: dict[str, dict[str, Any]]
    task_run_status: str
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
    scheduler: CaseTaskScheduler,
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
    The task ledger remains separately persisted by ``CaseTaskScheduler``.
    """
    if checkpointer is None:
        raise ValueError("an explicit LangGraph checkpointer is required")
    if not isinstance(max_recheck_rounds, int) or isinstance(max_recheck_rounds, bool) or max_recheck_rounds < 0:
        raise ValueError("max_recheck_rounds must be a non-negative integer")

    async def plan_node(state: CaseWorkflowState) -> dict[str, Any]:
        _require_input(state)
        scope = _scope_from_dict(state["case_scope"])
        scheduler.prepare_run(state["run_id"], scope)
        try:
            tasks = await _invoke(planner, state, scheduler.workflow_context(state["run_id"]))
            task_specs = _serialize_tasks(tasks)
            # The scheduler repeats this validation before creating TaskRuns.
            from .case_task_scheduler import validate_task_plan

            validate_task_plan(task_specs, max_total_tasks=scheduler.config.max_total_tasks, scope=scope)
        except Exception:
            scheduler.store.finish_run(state["run_id"], "failed")
            raise
        serialized = [task.to_dict() for task in task_specs]
        return {"initial_tasks": serialized, "tasks": serialized, "recheck_round": 0}

    async def dispatch_node(state: CaseWorkflowState) -> dict[str, Any]:
        scope = _scope_from_dict(state["case_scope"])
        # Recheck planning appends targeted tasks to ``tasks``. Dispatch the
        # merged plan so a verification gap actually drives another query.
        tasks = _serialize_tasks(state.get("tasks") or state.get("initial_tasks", []))
        result = await scheduler.run(
            state["run_id"],
            tasks,
            state["case_facts"],
            scope=scope,
            # The graph owns the terminal state: the report must be persisted
            # before CaseRunService exposes success to API clients.
            finalize=False,
        )
        task_runs = {
            task_id: {
                "task": value["task"].to_dict(),
                "status": value["status"],
                "result": value["result"],
                "error": value["error"],
                "attempts": value["attempts"],
                "idempotency_key": value["idempotency_key"],
            }
            for task_id, value in result.tasks.items()
        }
        return {"task_runs": task_runs, "task_run_status": result.status}

    async def candidate_events_node(state: CaseWorkflowState) -> dict[str, Any]:
        """Collect only explicitly structured Worker candidate events.

        Evidence text is intentionally not interpreted here.  Production
        Workers must return ``candidate_events`` (or one ``candidate_event``)
        as a structured, traceable result; this node applies the frozen run
        limit before the central verifier sees the candidates.
        """
        scheduler.store.record_event(state["run_id"], "candidate_events", "started")
        try:
            events: list[dict[str, Any]] = []
            for task_id, record in sorted(state.get("task_runs", {}).items()):
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

            if len(events) > scheduler.config.candidate_limit:
                raise PlanError(
                    "candidate event limit exceeded "
                    f"({len(events)}/{scheduler.config.candidate_limit})"
                )
            scheduler.store.record_event(
                state["run_id"],
                "candidate_events",
                "succeeded",
                {"count": len(events), "limit": scheduler.config.candidate_limit},
            )
            return {"candidate_events": events}
        except Exception:
            scheduler.store.record_event(state["run_id"], "candidate_events", "failed")
            scheduler.store.finish_run(state["run_id"], "failed")
            raise

    async def verify_node(state: CaseWorkflowState) -> dict[str, Any]:
        try:
            outcome = await _invoke(verifier, state, scheduler.workflow_context(state["run_id"]))
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
            scheduler.store.finish_run(state["run_id"], "failed")
            raise

    async def plan_rechecks_node(state: CaseWorkflowState) -> dict[str, Any]:
        try:
            scope = _scope_from_dict(state["case_scope"])
            proposed = await _invoke(recheck_planner, state, scheduler.workflow_context(state["run_id"]))
            tasks = _serialize_tasks(proposed)
            round_number = int(state.get("recheck_round", 0)) + 1
            if tasks:
                scheduler.append_recheck_tasks(state["run_id"], tasks, scope=scope)
                merged = [*state.get("tasks", []), *(task.to_dict() for task in tasks)]
            else:
                merged = list(state.get("tasks", []))
            return {
                "tasks": merged,
                "recheck_round": round_number,
                "recheck_added": bool(tasks),
            }
        except Exception:
            scheduler.store.finish_run(state["run_id"], "failed")
            raise

    async def summarize_node(state: CaseWorkflowState) -> dict[str, Any]:
        try:
            report = await _invoke(summarizer, state, scheduler.workflow_context(state["run_id"]))
            if not isinstance(report, Mapping):
                raise TypeError("summarizer must return a mapping")
            # Persist the report before publishing terminal success.  API
            # clients may observe the status immediately after the graph
            # finishes, so success must imply that the report is readable.
            scheduler.store.save_report(state["run_id"], report)
            scheduler.store.finish_run(
                state["run_id"],
                "success" if state.get("task_run_status") == "success" else "failed",
            )
            return {"report": dict(report)}
        except Exception:
            scheduler.store.finish_run(state["run_id"], "failed")
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
            scheduler.store.finish_run(state["run_id"], "failed")
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

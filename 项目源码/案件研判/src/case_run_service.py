"""Application service for submitting and observing case investigations.

The scheduler owns task-level durability and budgets.  This service adds the
missing process boundary: asynchronous submission, cooperative cancellation,
SQLite-backed status/report reads, a durable LangGraph checkpointer, and an
SSE-friendly event stream.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Iterator, Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from copy import deepcopy
import inspect
import json
from pathlib import Path
import threading
import time
from typing import Any
from uuid import uuid4

from .case_task_scheduler import (
    CaseTaskScope,
    CaseTaskScheduler,
    InvestigationTask,
    RunExecutionContext,
    SharedResourceLimits,
    SQLiteRunStore,
    TaskExecutionContext,
    TaskSchedulerConfig,
)
from .case_worker_subgraph import SupplementWorkerSubgraph


Callback = Callable[..., Any]


class CaseRunService:
    """Run one or more graph-backed case investigations in the background."""

    def __init__(
        self,
        store: SQLiteRunStore,
        *,
        worker: Callback,
        planner: Callback,
        verifier: Callback,
        recheck_planner: Callback,
        recheck_allowed: Callback,
        summarizer: Callback,
        config: TaskSchedulerConfig = TaskSchedulerConfig(),
        resources: SharedResourceLimits | None = None,
        checkpoint_path: Path | str | None = None,
        executor: ThreadPoolExecutor | None = None,
    ) -> None:
        if not inspect.iscoroutinefunction(worker):
            raise TypeError("worker must be an async callable")
        self.store = store
        self._worker = worker
        self._planner = planner
        self._verifier = verifier
        self._recheck_planner = recheck_planner
        self._recheck_allowed = recheck_allowed
        self._summarizer = summarizer
        self._config = config
        self._resources = resources or SharedResourceLimits()
        self.checkpoint_path = Path(checkpoint_path or store.path.with_name("workflow_checkpoints.sqlite3"))
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        self._worker_subgraph = SupplementWorkerSubgraph(self._guarded_worker)
        self.scheduler = CaseTaskScheduler(
            store,
            self._worker_subgraph,
            config=config,
            resources=self._resources,
        )
        self._executor = executor or ThreadPoolExecutor(max_workers=4, thread_name_prefix="case-run")
        self._futures: dict[str, Future[Any]] = {}
        self._lock = threading.RLock()

    def submit(self, state: Mapping[str, Any]) -> dict[str, Any]:
        payload = deepcopy(dict(state))
        run_id = str(payload.get("run_id") or uuid4().hex)
        payload["run_id"] = run_id
        self._validate_submission(payload)
        scope = _scope_from_state(payload)
        with self._lock:
            existing = self.store.get_run(run_id)
            if existing and existing["status"] == "running":
                raise ValueError(f"run_id is already running: {run_id}")
            self.store.prepare_run(run_id, self._config, scope)
            future = self._executor.submit(self._run_sync, payload)
            self._futures[run_id] = future
        return {"run_id": run_id, "status": "running"}

    def status(self, run_id: str) -> dict[str, Any] | None:
        return self.store.get_run(run_id)

    def cancel(self, run_id: str) -> bool:
        return self.store.request_cancel(run_id)

    def report(self, run_id: str) -> dict[str, Any] | None:
        return self.store.load_report(run_id)

    def iter_sse_events(
        self,
        run_id: str,
        *,
        poll_interval: float = 0.1,
        timeout_seconds: float = 3600.0,
    ) -> Iterator[str]:
        """Yield status/task changes until the run reaches a terminal state."""
        deadline = time.monotonic() + timeout_seconds
        last_snapshot: str | None = None
        while time.monotonic() <= deadline:
            record = self.status(run_id)
            if record is None:
                yield _sse("error", {"run_id": run_id, "error": "unknown_run_id"})
                return
            snapshot = self._event_snapshot(run_id, record)
            encoded = json.dumps(snapshot, ensure_ascii=False, sort_keys=True)
            if encoded != last_snapshot:
                last_snapshot = encoded
                yield _sse("status", snapshot)
            if record["status"] in {"success", "failed", "cancelled"}:
                yield _sse("complete", snapshot)
                return
            time.sleep(max(0.01, poll_interval))
        yield _sse("error", {"run_id": run_id, "error": "event_stream_timeout"})

    def _event_snapshot(self, run_id: str, record: Mapping[str, Any]) -> dict[str, Any]:
        tasks = self.store.load_tasks(run_id)
        return {
            "run": dict(record),
            "tasks": [
                {
                    "task_id": task_id,
                    "status": value["status"],
                    "error": value["error"],
                    "attempts": value["attempts"],
                    "result": value["result"],
                }
                for task_id, value in sorted(tasks.items())
            ],
            "events": self.store.list_events(run_id),
        }

    def _validate_submission(self, state: Mapping[str, Any]) -> None:
        for key in ("case_id", "case_facts", "case_scope"):
            if key not in state:
                raise ValueError(f"submission is missing {key}")
        if not isinstance(state["case_id"], str) or not state["case_id"].strip():
            raise ValueError("case_id must be a non-empty string")
        if not isinstance(state["case_facts"], Mapping):
            raise ValueError("case_facts must be a mapping")
        if "initial_tasks" not in state or not isinstance(state["initial_tasks"], Sequence):
            raise ValueError("initial_tasks must be a sequence")
        for item in state["initial_tasks"]:
            if not isinstance(item, (InvestigationTask, Mapping)):
                raise ValueError("initial_tasks must contain task mappings")
        _scope_from_state(state)

    async def _guarded_worker(
        self,
        task: InvestigationTask,
        task_input: dict[str, Any],
        context: TaskExecutionContext,
    ) -> Any:
        self._raise_if_cancelled(context.run_id)
        self.store.record_event(context.run_id, f"task:{task.task_id}", "started")
        try:
            result = await self._worker(task, task_input, context)
            self._raise_if_cancelled(context.run_id)
        except asyncio.CancelledError:
            self.store.record_event(context.run_id, f"task:{task.task_id}", "cancelled")
            raise
        except Exception as exc:
            self.store.record_event(
                context.run_id,
                f"task:{task.task_id}",
                "failed",
                {"error": type(exc).__name__},
            )
            raise
        self.store.record_event(context.run_id, f"task:{task.task_id}", "succeeded")
        return result

    async def _guarded_callback(
        self,
        node_name: str,
        callback: Callback,
        state: Mapping[str, Any],
        context: RunExecutionContext | None = None,
    ) -> Any:
        run_id = str(state["run_id"])
        self._raise_if_cancelled(run_id)
        self.store.record_event(run_id, node_name, "started")
        try:
            value = callback(state, context) if context is not None else callback(state)
            if inspect.isawaitable(value):
                value = await value
            self._raise_if_cancelled(run_id)
        except asyncio.CancelledError:
            self.store.record_event(run_id, node_name, "cancelled")
            raise
        except Exception as exc:
            self.store.record_event(
                run_id,
                node_name,
                "failed",
                {"error": type(exc).__name__},
            )
            raise
        self.store.record_event(run_id, node_name, "succeeded")
        return value

    def _raise_if_cancelled(self, run_id: str) -> None:
        if self.store.is_cancel_requested(run_id):
            raise asyncio.CancelledError

    def _run_sync(self, state: Mapping[str, Any]) -> Any:
        run_id = str(state["run_id"])
        try:
            return asyncio.run(self._run_graph(state))
        except asyncio.CancelledError:
            self.store.record_event(run_id, "run", "cancelled")
            self.store.finish_run(run_id, "cancelled")
            return None
        except Exception as exc:
            if self.store.is_cancel_requested(run_id):
                # LangGraph may wrap a cooperative callback cancellation in a
                # node-specific exception before it reaches this boundary.
                self.store.record_event(run_id, "run", "cancelled", {"error": type(exc).__name__})
                self.store.finish_run(run_id, "cancelled")
                return None
            self.store.record_event(run_id, "run", "failed", {"error": type(exc).__name__})
            self.store.finish_run(run_id, "failed", error=type(exc).__name__)
            return None

    async def _run_graph(self, state: Mapping[str, Any]) -> Any:
        # Import lazily so the scheduler and CLI remain usable without the
        # optional LangGraph/API extras installed.
        from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

        from .case_workflow_graph import build_case_workflow_graph

        graph = None
        async with AsyncSqliteSaver.from_conn_string(str(self.checkpoint_path)) as checkpointer:
            graph = build_case_workflow_graph(
                planner=lambda current, context: self._guarded_callback("plan", self._planner, current, context),
                scheduler=self.scheduler,
                verifier=lambda current, context: self._guarded_callback("central_verify", self._verifier, current, context),
                recheck_planner=lambda current, context: self._guarded_callback("plan_rechecks", self._recheck_planner, current, context),
                recheck_allowed=lambda current: self._guarded_callback("route_after_verify", self._recheck_allowed, current),
                summarizer=lambda current, context: self._guarded_callback("summarize", self._summarizer, current, context),
                checkpointer=checkpointer,
                max_recheck_rounds=1,
            )
            result = await graph.ainvoke(
                dict(state),
                {"configurable": {"thread_id": f"{state['case_id']}:{state['run_id']}"}},
            )
        report = result.get("report") if isinstance(result, Mapping) else None
        if isinstance(report, Mapping):
            self.store.save_report(str(state["run_id"]), report)
        return result


def _scope_from_state(state: Mapping[str, Any]) -> CaseTaskScope:
    value = state["case_scope"]
    if not isinstance(value, Mapping):
        raise ValueError("case_scope must be a mapping")
    return CaseTaskScope(
        entity_ids=frozenset(value.get("entity_ids") or ()),
        fact_ids=frozenset(value.get("fact_ids") or ()),
        allowed_tools=frozenset(value.get("allowed_tools") or ()),
        time_range=tuple(value["time_range"]) if value.get("time_range") is not None else None,
    )


def _sse(event: str, data: Mapping[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(dict(data), ensure_ascii=False)}\n\n"

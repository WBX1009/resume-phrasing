"""One case service: durable admission, native graph recovery, read projections."""
from __future__ import annotations

import asyncio
from collections.abc import Mapping
from typing import Any
from concurrent.futures import wait
import json
import logging
from pathlib import Path
import threading
import time
from uuid import uuid4

from langgraph.runtime import RunControl
from langgraph.errors import GraphDrained
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

from .case_task_graph import CaseTaskExecutor, scope_from_state
from .case_execution import CaseExecutionConfig, SQLiteRunStore, SharedResourceLimits, InvestigationTask, PlanError, _invoke_tool
from .case_repository import CaseRepository
from .investigation_tools import InvestigationTools

TERMINAL = {"success", "partial", "failed", "cancelled"}


class CaseRunService:
    def __init__(self, store: SQLiteRunStore, *, repository: CaseRepository, reasoner,
                 tools=None, config=CaseExecutionConfig(), resources=None, checkpoint_path=None):
        self.store, self.repository, self.reasoner = store, repository, reasoner
        self.tools = tools or InvestigationTools(store)
        self._config = config
        self._resources = resources or SharedResourceLimits()
        self.checkpoint_path = Path(checkpoint_path or store.path.with_name("langgraph-v3.sqlite3"))
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        self.executor = CaseTaskExecutor(store, reasoner, self.tools, config=config, resources=self._resources)
        self._claim_store()
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, name="case-runs", daemon=True)
        self._futures = {}
        self._controls = {}
        self._lock = threading.RLock()
        self._changed = threading.Condition()
        self._closing = False
        self._thread.start()
        try:
            self.recover_pending()
        except BaseException:
            self.close()
            raise

    def _claim_store(self):
        """One service owns startup recovery for a SQLite run database."""
        import os
        self._ownership = open(str(self.store.path) + ".service.lock", "a+b")
        if self._ownership.seek(0, 2) == 0:
            self._ownership.write(b"0")
            self._ownership.flush()
        self._ownership.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(self._ownership.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self._ownership.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            self._ownership.close()
            raise ValueError("run database already has an active service owner") from exc

    def _track(self, run_id, coroutine):
        future = asyncio.run_coroutine_threadsafe(coroutine, self._loop)
        self._futures[run_id] = future
        def finished(done):
            with self._lock:
                if self._futures.get(run_id) is done:
                    self._futures.pop(run_id, None)
            self._notify()
        future.add_done_callback(finished)

    def _notify(self):
        with self._changed:
            self._changed.notify_all()

    def submit(self, case_id: str, *, run_id: str | None = None):
        if not isinstance(case_id, str) or not case_id.strip():
            raise PlanError("case_id must be a non-empty string")
        run_id = run_id or uuid4().hex
        if not isinstance(run_id, str) or not run_id.strip():
            raise PlanError("run_id must be a non-empty string")
        if threading.current_thread() is self._thread:
            raise RuntimeError("submit must be called outside the service event loop")
        async def load_authorized_snapshot():
            return await _invoke_tool(self._resources.database, self.repository.load, lambda: None, case_id)
        with self._lock:
            if self._closing:
                raise ValueError("service is closed")
            if self.store.get_run(run_id):
                raise ValueError("run_id already exists")
        snapshot = asyncio.run_coroutine_threadsafe(load_authorized_snapshot(), self._loop).result()
        if snapshot.case_id != case_id:
            raise PlanError("repository returned another case")
        if not set(snapshot.allowed_tools) <= self.tools.names():
            raise PlanError("case permits an unregistered tool")
        state = {"case_id": case_id, "run_id": run_id, "snapshot_ref": snapshot.reference(),
                 "case_scope": snapshot.scope().to_dict()}
        with self._lock:
            if self._closing:
                raise ValueError("service is closed")
            if self.store.get_run(run_id):
                raise ValueError("run_id already exists")
            self.store.admit(state, self._config, snapshot.scope(), snapshot=snapshot.model_dump(mode="json"))
            self._track(run_id, self._run_guarded(state))
        return {"run_id": run_id, "status": "running"}

    def status(self, run_id):
        return self.store.get_run(run_id)

    def cancel(self, run_id):
        result = self.store.request_cancel(run_id)
        self._notify()
        return result

    def report(self, run_id):
        return self.store.load_report(run_id)

    def fact(self, run_id, fact_id):
        # Only graph-committed evidence is exposed, not orphan artifacts from a
        # crashed node before its checkpoint. Run access belongs to the host API.
        report = self.report(run_id)
        if not report or fact_id not in report["fact_ids"]:
            raise KeyError("fact not referenced by a published report")
        return self.store.facts(run_id, [fact_id])[fact_id]

    def iter_sse_events(self, run_id, *, poll_interval=.5, timeout_seconds=3600):
        deadline, last, cursor = time.monotonic()+timeout_seconds, None, 0
        while time.monotonic() <= deadline:
            record = self.status(run_id)
            if record is None:
                yield _sse("error", {"error": "unknown_run_id"})
                return
            # Projections are optional read models. Never deserialize their
            # task spec or use them for scheduling/recovery.
            with self.store._connect() as connection:
                rows = connection.execute("SELECT task_id,status,error FROM case_task_outcomes WHERE run_id=? ORDER BY task_id", (run_id,)).fetchall()
            events = self.store.list_events(run_id, after_id=cursor)
            if events:
                cursor = events[-1]["event_id"]
            snapshot = {"run": record, "tasks": [dict(r) for r in rows], "events": events}
            encoded = json.dumps(snapshot, ensure_ascii=False, sort_keys=True)
            if encoded != last:
                last = encoded
                yield _sse("status", snapshot)
            if record["status"] in TERMINAL:
                yield _sse("complete", snapshot)
                return
            with self._changed:
                self._changed.wait(timeout=max(.05,poll_interval))
        yield _sse("error", {"error": "event_stream_timeout"})

    def recover_pending(self):
        with self.store._connect() as connection:
            rows = connection.execute("SELECT run_id FROM case_runs WHERE status='running'").fetchall()
        for row in rows:
            self.store.admission(row[0])  # fail explicitly on an incompatible graph version
            self.resume(row[0], preserve_cancel=True)

    def resume(self, run_id, *, preserve_cancel=False):
        with self._lock:
            if self._closing:
                raise ValueError("service is closed")
            if run_id in self._futures:
                raise ValueError("run is already active")
            record = self.status(run_id)
            if record and record["status"] in {"success", "partial", "cancelled"}:
                raise ValueError("completed/cancelled runs are terminal; submit a new run")
            state = self.store.admission(run_id)
            self.store.prepare_run(run_id, self._config, scope_from_state(state))
            if not preserve_cancel:
                self.store.mark_running(run_id)
            self._track(run_id, self._run_guarded(state, resume=True))
        return {"run_id": run_id, "status": "running"}

    def close(self):
        """Drain native runs, then await cancelled I/O before releasing ownership."""
        async def stop():
            pending = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
            for task in pending:
                task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)
            await self._loop.shutdown_default_executor()
            if hasattr(self.reasoner, "aclose"):
                await self.reasoner.aclose()
        if self._loop.is_running():
            self._closing = True
            for control in list(self._controls.values()):
                control.request_drain("service shutdown")
            wait(list(self._futures.values()), timeout=5)
            asyncio.run_coroutine_threadsafe(stop(), self._loop).result(timeout=30)
            self._loop.call_soon_threadsafe(self._loop.stop)
            self._thread.join(timeout=30)
            self._loop.close()
            self._ownership.close()

    async def _run_guarded(self, state: Mapping[str, Any], *, resume=False) -> Any:
        run_id = str(state["run_id"])
        control = RunControl()
        self._controls[run_id] = control
        try:
            return await self._run_graph(state, resume=resume, control=control)
        except GraphDrained:
            return None
        except asyncio.CancelledError:
            if self._closing:
                return None
            self.store.record_event(run_id, "run", "cancelled")
            self.store.finish_run(run_id, "cancelled")
            return None
        except Exception as exc:
            logging.getLogger(__name__).exception("case graph failed: %s", run_id)
            if self.store.is_cancel_requested(run_id):
                # LangGraph may wrap a cooperative callback cancellation in a
                # node-specific exception before it reaches this boundary.
                self.store.record_event(run_id, "run", "cancelled", {"error": type(exc).__name__})
                self.store.finish_run(run_id, "cancelled")
                return None
            self.store.record_event(run_id, "run", "failed", {"error": type(exc).__name__})
            self.store.finish_run(run_id, "failed", error=type(exc).__name__)
            return None
        finally:
            self._controls.pop(run_id, None)

    async def _run_graph(self, state, *, resume=False, control=None):
        import aiosqlite
        from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
        from .case_workflow_graph import build_case_workflow_graph

        # Explicit strict deserialization. State contains JSON and LangGraph's
        # safe built-in Send type only; no arbitrary Python object restoration.
        serde = JsonPlusSerializer(pickle_fallback=False, allowed_msgpack_modules=None)
        async with aiosqlite.connect(str(self.checkpoint_path)) as connection:
            saver = AsyncSqliteSaver(connection, serde=serde)
            graph = build_case_workflow_graph(executor=self.executor, reasoner=self.reasoner, checkpointer=saver)
            config = {"configurable": {"thread_id": f"native-v3:{state['case_id']}:{state['run_id']}"},
                "max_concurrency": self._config.case_task_limit,
                "recursion_limit": (self._config.max_worker_steps*3+8)*self._config.max_total_tasks+50}
            snapshot = await graph.aget_state(config)
            async for namespace, event in graph.astream(None if resume and snapshot.values else state, config,
                    durability="sync", control=control, stream_mode="tasks", subgraphs=True):
                node = event.get("name", "graph")
                status = "failed" if event.get("error") else "succeeded" if "result" in event else "started"
                self.store.record_event(state["run_id"], node, status, {"namespace": list(namespace)})
                self._notify()
            result = (await graph.aget_state(config)).values
        if result.get("report"):
            self.store.project(state["run_id"], [InvestigationTask.from_dict(t) for t in result["tasks"]],
                result["task_outcomes"], self._config, scope_from_state(state))
            self.store.publish_report(state["run_id"], result["report"], result["report"]["status"])
            self._notify()
        return result


def _sse(event, data):
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"

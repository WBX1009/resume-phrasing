"""Business task contracts, atomic budgets, effects and API projections.

Execution and recovery belong to LangGraph, never to the task projection.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from functools import partial
import hashlib
import inspect
import json
from pathlib import Path
import sqlite3
from typing import Any


class PlanError(ValueError):
    """The proposed case task plan is malformed or outside its authorized scope."""


class BudgetLimitExceeded(RuntimeError):
    """A run-level LLM/tool/token budget has been exhausted."""


class TaskExecutionError(RuntimeError):
    """A known, terminal business task failure, distinct from a program defect."""


class ExternalOperationUncertain(RuntimeError):
    """An idempotent external operation has an unresolved prior attempt."""


@dataclass(frozen=True)
class CaseTaskScope:
    """Authoritative per-case bounds used to validate every planned task."""

    entity_ids: frozenset[str]
    fact_ids: frozenset[str]
    allowed_tools: frozenset[str]
    time_range: tuple[str, str] | None = None

    def __post_init__(self) -> None:
        for name in ("entity_ids", "fact_ids", "allowed_tools"):
            if isinstance(getattr(self, name), (str, bytes)):
                raise PlanError(f"scope {name} must be a collection of identifiers, not a string")
            values = frozenset(getattr(self, name))
            if any(not isinstance(value, str) or not value.strip() for value in values):
                raise PlanError(f"scope {name} must contain non-empty strings")
            object.__setattr__(self, name, values)
        if not self.entity_ids:
            raise PlanError("case scope must contain at least one authorized entity")
        if self.time_range is not None:
            bounds = tuple(self.time_range)
            if len(bounds) != 2:
                raise PlanError("case time_range must contain exactly two bounds")
            parsed = tuple(_parse_time(value, "case time_range") for value in bounds)
            _ensure_comparable(parsed[0], parsed[1], "case time_range")
            if parsed[0][0] > parsed[1][0]:
                raise PlanError("case time_range start must not be after its end")
            object.__setattr__(self, "time_range", bounds)

    def to_dict(self) -> dict[str, Any]:
        return {
            "entity_ids": sorted(self.entity_ids),
            "fact_ids": sorted(self.fact_ids),
            "allowed_tools": sorted(self.allowed_tools),
            "time_range": list(self.time_range) if self.time_range is not None else None,
        }


@dataclass(frozen=True)
class InvestigationTask:
    task_id: str
    question: str
    entity_ids: tuple[str, ...] = ()
    time_range: tuple[str, str] | None = None
    basis_fact_ids: tuple[str, ...] = ()
    allowed_tools: tuple[str, ...] = ()
    depends_on: tuple[str, ...] = ()
    payload: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("entity_ids", "basis_fact_ids", "allowed_tools", "depends_on"):
            values = tuple(getattr(self, name))
            if any(not isinstance(value, str) or not value.strip() for value in values):
                raise PlanError(f"{name} must contain non-empty strings")
            if len(values) != len(set(values)):
                raise PlanError(f"{name} must not contain duplicates")
            object.__setattr__(self, name, values)
        if not isinstance(self.task_id, str) or not self.task_id.strip():
            raise PlanError("task_id must be a non-empty string")
        if not isinstance(self.question, str) or not self.question.strip():
            raise PlanError("question must be a non-empty string")
        if self.time_range is not None:
            bounds = tuple(self.time_range)
            if len(bounds) != 2:
                raise PlanError("time_range must contain exactly two ISO-8601 bounds")
            parsed = tuple(_parse_time(value, "task time_range") for value in bounds)
            _ensure_comparable(parsed[0], parsed[1], "task time_range")
            if parsed[0][0] > parsed[1][0]:
                raise PlanError("time_range start must not be after its end")
            object.__setattr__(self, "time_range", bounds)
        if not isinstance(self.payload, Mapping):
            raise PlanError("payload must be a mapping")
        try:
            json.dumps(dict(self.payload), ensure_ascii=False, allow_nan=False, sort_keys=True)
        except (TypeError, ValueError) as exc:
            raise PlanError("payload must contain finite JSON values only") from exc
        object.__setattr__(self, "payload", deepcopy(dict(self.payload)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "question": self.question,
            "entity_ids": list(self.entity_ids),
            "time_range": list(self.time_range) if self.time_range is not None else None,
            "basis_fact_ids": list(self.basis_fact_ids),
            "allowed_tools": list(self.allowed_tools),
            "depends_on": list(self.depends_on),
            "payload": dict(self.payload),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "InvestigationTask":
        return cls(
            task_id=value["task_id"],
            question=value["question"],
            entity_ids=tuple(value.get("entity_ids") or ()),
            time_range=tuple(value["time_range"]) if value.get("time_range") is not None else None,
            basis_fact_ids=tuple(value.get("basis_fact_ids") or ()),
            allowed_tools=tuple(value.get("allowed_tools") or ()),
            depends_on=tuple(value.get("depends_on") or ()),
            payload=value.get("payload") or {},
        )


@dataclass(frozen=True)
class CaseExecutionConfig:
    case_task_limit: int = 2
    server_task_cap: int = 4
    max_total_tasks: int = 100
    max_llm_calls: int = 200
    max_tool_calls: int = 200
    max_reserved_tokens: int = 2_000_000
    candidate_limit: int = 100
    max_attempts: int = 3
    task_timeout: int = 120

    def __post_init__(self) -> None:
        values = {
            "case_task_limit": self.case_task_limit,
            "server_task_cap": self.server_task_cap,
            "max_total_tasks": self.max_total_tasks,
            "max_llm_calls": self.max_llm_calls,
            "max_tool_calls": self.max_tool_calls,
            "max_reserved_tokens": self.max_reserved_tokens,
            "candidate_limit": self.candidate_limit,
            "max_attempts": self.max_attempts,
            "task_timeout": self.task_timeout,
        }
        for name, value in values.items():
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.case_task_limit > self.server_task_cap:
            raise ValueError("case_task_limit exceeds server_task_cap")


class SharedResourceLimits:
    """Share one instance across case runs in a single application process."""

    def __init__(self, llm_request_cap: int = 2, db_query_cap: int = 2) -> None:
        for name, value in (("llm_request_cap", llm_request_cap), ("db_query_cap", db_query_cap)):
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        self.llm = asyncio.Semaphore(llm_request_cap)
        self.database = asyncio.Semaphore(db_query_cap)


class SQLiteRunStore:
    """Persist admissions, API projections, atomic budgets and external effect records."""

    _budget_caps = {
        "llm_calls": "max_llm_calls",
        "tool_calls": "max_tool_calls",
        "reserved_tokens": "max_reserved_tokens",
    }

    def __init__(self, path: Path | str) -> None:
        if str(path) == ":memory:":
            raise ValueError("a file-backed SQLite path is required for restart recovery")
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            # Detect incompatible databases; do not carry old task-ledger migrations.
            columns = {row[1] for row in connection.execute("PRAGMA table_info(case_runs)")}
            if columns and "plan_hash" in columns:
                raise ValueError("unsupported run database schema; use a fresh runtime database")
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS case_runs (
                    run_id TEXT PRIMARY KEY,
                    scope_hash TEXT NOT NULL,
                    config_json TEXT NOT NULL,
                    status TEXT NOT NULL,
                    llm_calls INTEGER NOT NULL DEFAULT 0,
                    tool_calls INTEGER NOT NULL DEFAULT 0,
                    reserved_tokens INTEGER NOT NULL DEFAULT 0,
                    cancel_requested INTEGER NOT NULL DEFAULT 0,
                    report_json TEXT,
                    error TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS case_task_outcomes (
                    run_id TEXT NOT NULL REFERENCES case_runs(run_id),
                    task_id TEXT NOT NULL,
                    spec_json TEXT NOT NULL,
                    status TEXT NOT NULL,
                    result_json TEXT,
                    error TEXT,
                    attempts INTEGER NOT NULL DEFAULT 0,
                    idempotency_key TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (run_id, task_id)
                );
                CREATE TABLE IF NOT EXISTS case_run_events (
                    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL REFERENCES case_runs(run_id),
                    node TEXT NOT NULL,
                    status TEXT NOT NULL,
                    payload_json TEXT,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS case_admissions (
                    run_id TEXT PRIMARY KEY REFERENCES case_runs(run_id),
                    input_json TEXT NOT NULL,
                    graph_version TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS case_effects (
                    run_id TEXT NOT NULL REFERENCES case_runs(run_id),
                    operation_key TEXT NOT NULL,
                    status TEXT NOT NULL,
                    result_json TEXT,
                    error TEXT,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (run_id, operation_key)
                );
                """
            )
    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30, isolation_level=None)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 30000")
        return connection

    def prepare_task_projection(self, run_id, tasks, config, scope):
        """Initialize a plan projection; never infer recovery from task rows."""
        validate_task_plan(tasks, max_total_tasks=config.max_total_tasks, scope=scope)
        self.prepare_run(run_id, config, scope)
        with self._connect() as connection:
            for task in tasks:
                connection.execute(
                    "INSERT INTO case_task_outcomes(run_id,task_id,spec_json,status,idempotency_key,updated_at) VALUES(?,?,?,'pending',?,?) ON CONFLICT(run_id,task_id) DO UPDATE SET spec_json=excluded.spec_json,idempotency_key=excluded.idempotency_key",
                    (run_id, task.task_id, _json(task.to_dict()), f"{run_id}:{task.task_id}", _now()),
                )

    def admit(self, state, config, scope):
        """Atomically persist admission and frozen limits before any execution."""
        encoded = _json(state)
        run_id = state["run_id"]
        scope_hash, config_json = _digest(scope.to_dict()), _json(asdict(config))
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            run = connection.execute("SELECT scope_hash,config_json FROM case_runs WHERE run_id=?", (run_id,)).fetchone()
            admission = connection.execute("SELECT input_json,graph_version FROM case_admissions WHERE run_id=?", (run_id,)).fetchone()
            if run and not admission:
                raise ValueError("legacy run requires its original application version")
            if run and (run[0] != scope_hash or run[1] != config_json):
                raise ValueError("run_id is already bound to a different scope or budget configuration")
            if admission and (admission[0] != encoded or admission[1] != "native-v2"):
                raise ValueError("run_id is already bound to a different plan, scope, or budget configuration")
            now = _now()
            connection.execute(
                "INSERT OR IGNORE INTO case_runs(run_id,scope_hash,config_json,status,created_at,updated_at) VALUES(?,?,?,'running',?,?)",
                (run_id, scope_hash, config_json, now, now),
            )
            connection.execute("INSERT OR IGNORE INTO case_admissions VALUES(?,?,?)", (run_id, encoded, "native-v2"))
            connection.commit()

    def admission(self, run_id):
        with self._connect() as connection:
            row = connection.execute("SELECT input_json,graph_version FROM case_admissions WHERE run_id=?", (run_id,)).fetchone()
        if not row or row[1] != "native-v2":
            raise ValueError("run has no native-v2 admission; legacy runs require their original application version")
        return json.loads(row[0])

    def mark_running(self, run_id):
        with self._connect() as connection:
            connection.execute("UPDATE case_runs SET status='running',cancel_requested=0,error=NULL,updated_at=? WHERE run_id=?", (_now(),run_id))

    def project(self, run_id, tasks, outcomes, config, scope):
        self.prepare_task_projection(run_id, tasks, config, scope)
        for task_id, record in outcomes.items():
            self.record_task_status(run_id, task_id, record["status"], result=record.get("result"), error=record.get("error"))

    def publish_report(self, run_id, report, status):
        with self._connect() as connection:
            connection.execute("UPDATE case_runs SET report_json=?,status=?,error=NULL,updated_at=? WHERE run_id=?", (_json(report),status,_now(),run_id))

    def prepare_run(self, run_id: str, config: CaseExecutionConfig, scope: CaseTaskScope) -> None:
        """Create the immutable budget/scope ledger before any graph LLM call."""
        if not isinstance(run_id, str) or not run_id.strip():
            raise ValueError("run_id must be a non-empty string")
        scope_hash = _digest(scope.to_dict())
        config_json = _json(asdict(config))
        now = _now()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute("SELECT scope_hash, config_json FROM case_runs WHERE run_id = ?", (run_id,)).fetchone()
            if row is None:
                connection.execute(
                    "INSERT INTO case_runs(run_id, scope_hash, config_json, status, created_at, updated_at) VALUES(?, ?, ?, 'running', ?, ?)",
                    (run_id, scope_hash, config_json, now, now),
                )
            elif row["scope_hash"] != scope_hash or row["config_json"] != config_json:
                connection.rollback()
                raise ValueError("run_id is already bound to a different scope or budget configuration")
            connection.commit()

    def load_task_projection(self, run_id: str) -> dict[str, dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute("SELECT * FROM case_task_outcomes WHERE run_id = ? ORDER BY task_id", (run_id,)).fetchall()
        return {
            row["task_id"]: {
                "task": InvestigationTask.from_dict(json.loads(row["spec_json"])),
                "status": row["status"],
                "result": json.loads(row["result_json"]) if row["result_json"] is not None else None,
                "error": row["error"],
                "attempts": row["attempts"],
                "idempotency_key": row["idempotency_key"],
            }
            for row in rows
        }

    def record_task_status(
        self,
        run_id: str,
        task_id: str,
        status: str,
        *,
        result: Any = None,
        error: str | None = None,
        increment_attempt: bool = False,
    ) -> None:
        if status not in {"pending", "running", "succeeded", "failed", "blocked"}:
            raise ValueError(f"unsupported task status: {status}")
        result_json = _json(result) if status == "succeeded" else None
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            if increment_attempt:
                connection.execute(
                    "UPDATE case_task_outcomes SET status = ?, result_json = ?, error = ?, attempts = attempts + 1, updated_at = ? WHERE run_id = ? AND task_id = ?",
                    (status, result_json, error, _now(), run_id, task_id),
                )
            else:
                connection.execute(
                    "UPDATE case_task_outcomes SET status = ?, result_json = ?, error = ?, updated_at = ? WHERE run_id = ? AND task_id = ?",
                    (status, result_json, error, _now(), run_id, task_id),
                )
            connection.commit()

    def reserve(self, run_id: str, resource: str, amount: int, config: CaseExecutionConfig) -> int:
        if resource not in self._budget_caps:
            raise ValueError(f"unknown budget resource: {resource}")
        if not isinstance(amount, int) or isinstance(amount, bool) or amount <= 0:
            raise ValueError("budget reservation must be a positive integer")
        cap = getattr(config, self._budget_caps[resource])
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(f"SELECT {resource} FROM case_runs WHERE run_id = ?", (run_id,)).fetchone()
            if row is None:
                connection.rollback()
                raise KeyError(f"unknown run_id: {run_id}")
            used = int(row[resource])
            if used + amount > cap:
                connection.rollback()
                raise BudgetLimitExceeded(f"{resource} budget exhausted ({used}/{cap})")
            updated = used + amount
            connection.execute(
                f"UPDATE case_runs SET {resource} = ?, updated_at = ? WHERE run_id = ?",
                (updated, _now(), run_id),
            )
            connection.commit()
            return updated

    def reserve_llm(self, run_id: str, max_output_tokens: int, config: CaseExecutionConfig) -> None:
        if not isinstance(max_output_tokens, int) or isinstance(max_output_tokens, bool) or max_output_tokens <= 0:
            raise ValueError("max_output_tokens must be a positive integer")
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT llm_calls, reserved_tokens FROM case_runs WHERE run_id = ?", (run_id,)
            ).fetchone()
            if row is None:
                connection.rollback()
                raise KeyError(f"unknown run_id: {run_id}")
            if int(row["llm_calls"]) + 1 > config.max_llm_calls:
                connection.rollback()
                raise BudgetLimitExceeded(
                    f"llm_calls budget exhausted ({row['llm_calls']}/{config.max_llm_calls})"
                )
            if int(row["reserved_tokens"]) + max_output_tokens > config.max_reserved_tokens:
                connection.rollback()
                raise BudgetLimitExceeded(
                    "reserved_tokens budget exhausted "
                    f"({row['reserved_tokens']}/{config.max_reserved_tokens})"
                )
            connection.execute(
                "UPDATE case_runs SET llm_calls = llm_calls + 1, reserved_tokens = reserved_tokens + ?, updated_at = ? WHERE run_id = ?",
                (max_output_tokens, _now(), run_id),
            )
            connection.commit()

    def finish_run(self, run_id: str, status: str, *, error: str | None = None) -> None:
        if status not in {"success", "failed", "cancelled"}:
            raise ValueError(f"unsupported run status: {status}")
        with self._connect() as connection:
            connection.execute(
                "UPDATE case_runs SET status = ?, error = ?, updated_at = ? WHERE run_id = ?",
                (status, error, _now(), run_id),
            )

    def request_cancel(self, run_id: str) -> bool:
        """Request cooperative cancellation without interrupting a task mid-write."""
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT status FROM case_runs WHERE run_id = ?", (run_id,)
            ).fetchone()
            if row is None:
                connection.rollback()
                raise KeyError(f"unknown run_id: {run_id}")
            if row["status"] in {"success", "failed", "cancelled"}:
                connection.rollback()
                return False
            connection.execute(
                "UPDATE case_runs SET cancel_requested = 1, updated_at = ? WHERE run_id = ?",
                (_now(), run_id),
            )
            connection.commit()
            return True

    def is_cancel_requested(self, run_id: str) -> bool:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT cancel_requested FROM case_runs WHERE run_id = ?", (run_id,)
            ).fetchone()
        if row is None:
            raise KeyError(f"unknown run_id: {run_id}")
        return bool(row["cancel_requested"])

    def load_report(self, run_id: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT report_json FROM case_runs WHERE run_id = ?", (run_id,)
            ).fetchone()
        if row is None:
            raise KeyError(f"unknown run_id: {run_id}")
        return json.loads(row["report_json"]) if row["report_json"] is not None else None

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM case_runs WHERE run_id = ?", (run_id,)
            ).fetchone()
        if row is None:
            return None
        # Status/recovery must not deserialize a possibly stale task projection.
        with self._connect() as connection:
            task_counts = connection.execute(
                "SELECT status,COUNT(*) FROM case_task_outcomes WHERE run_id=? GROUP BY status", (run_id,)
            ).fetchall()
        counts = {status: 0 for status in ("pending", "running", "succeeded", "failed", "blocked")}
        counts.update({status: count for status, count in task_counts})
        return {
            "run_id": run_id,
            "status": row["status"],
            "cancel_requested": bool(row["cancel_requested"]),
            "error": row["error"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "task_counts": counts,
            "task_total": sum(counts.values()),
            "budget": {
                "llm_calls": int(row["llm_calls"]),
                "tool_calls": int(row["tool_calls"]),
                "reserved_tokens": int(row["reserved_tokens"]),
            },
            "report": json.loads(row["report_json"]) if row["report_json"] is not None else None,
            "event_count": self._event_count(run_id),
        }

    def record_event(
        self,
        run_id: str,
        node: str,
        status: str,
        payload: Mapping[str, Any] | None = None,
    ) -> None:
        if not node or not status:
            raise ValueError("event node and status must be non-empty")
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO case_run_events(run_id, node, status, payload_json, created_at) VALUES(?, ?, ?, ?, ?)",
                (run_id, node, status, _json(dict(payload or {})), _now()),
            )

    def list_events(self, run_id: str) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT event_id, node, status, payload_json, created_at FROM case_run_events WHERE run_id = ? ORDER BY event_id",
                (run_id,),
            ).fetchall()
        return [
            {
                "event_id": int(row["event_id"]),
                "node": row["node"],
                "status": row["status"],
                "payload": json.loads(row["payload_json"]) if row["payload_json"] else {},
                "created_at": row["created_at"],
            }
            for row in rows
        ]

    def _event_count(self, run_id: str) -> int:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT COUNT(*) AS count FROM case_run_events WHERE run_id = ?", (run_id,)
            ).fetchone()
        return int(row["count"])

    def budget_usage(self, run_id: str) -> dict[str, int]:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT llm_calls, tool_calls, reserved_tokens FROM case_runs WHERE run_id = ?", (run_id,)
            ).fetchone()
        if row is None:
            raise KeyError(f"unknown run_id: {run_id}")
        return dict(row)

    def begin_effect(self, run_id: str, operation_key: str) -> tuple[str, Any | None]:
        """Atomically claim an external operation or return its cached result.

        A previously running operation is not retried automatically: its
        external side effect may have committed before the process stopped.
        The caller must reconcile that operation instead of risking a second
        side effect.
        """
        if not operation_key or not isinstance(operation_key, str):
            raise ValueError("operation_key must be a non-empty string")
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT status, result_json, error FROM case_effects WHERE run_id = ? AND operation_key = ?",
                (run_id, operation_key),
            ).fetchone()
            if row is not None:
                connection.commit()
                if row["status"] == "succeeded":
                    return "cached", json.loads(row["result_json"])
                if row["status"] == "running":
                    return "uncertain", None
                # A failed local attempt is safe to claim again only when the
                # caller explicitly uses the same stable operation key.
                connection.execute(
                    "UPDATE case_effects SET status = 'running', error = NULL, updated_at = ? WHERE run_id = ? AND operation_key = ?",
                    (_now(), run_id, operation_key),
                )
                connection.commit()
                return "claimed", None
            connection.execute(
                "INSERT INTO case_effects(run_id, operation_key, status, updated_at) VALUES(?, ?, 'running', ?)",
                (run_id, operation_key, _now()),
            )
            connection.commit()
            return "claimed", None

    def complete_effect(self, run_id: str, operation_key: str, result: Any) -> None:
        with self._connect() as connection:
            connection.execute(
                "UPDATE case_effects SET status = 'succeeded', result_json = ?, error = NULL, updated_at = ? WHERE run_id = ? AND operation_key = ?",
                (_json(result), _now(), run_id, operation_key),
            )
            connection.commit()

    def fail_effect(self, run_id: str, operation_key: str, error: str) -> None:
        with self._connect() as connection:
            connection.execute(
                "UPDATE case_effects SET status = 'failed', error = ?, updated_at = ? WHERE run_id = ? AND operation_key = ?",
                (error, _now(), run_id, operation_key),
            )
            connection.commit()


@dataclass
class TaskExecutionContext:
    run_id: str
    task: InvestigationTask
    idempotency_key: str
    store: SQLiteRunStore
    config: CaseExecutionConfig
    resources: SharedResourceLimits

    async def llm_call(
        self,
        callback: Callable[..., Awaitable[Any]],
        *args: Any,
        max_output_tokens: int,
        **kwargs: Any,
    ) -> Any:
        if not isinstance(max_output_tokens, int) or isinstance(max_output_tokens, bool) or max_output_tokens <= 0:
            raise ValueError("max_output_tokens must be a positive integer")
        if self.store.is_cancel_requested(self.run_id):
            raise asyncio.CancelledError
        async with self.resources.llm:
            if self.store.is_cancel_requested(self.run_id):
                raise asyncio.CancelledError
            self.store.reserve_llm(self.run_id, max_output_tokens, self.config)
            return await _invoke_async(callback, *args, **kwargs)

    async def tool_call(self, callback: Callable[..., Any], *args: Any, tool_name: str, **kwargs: Any) -> Any:
        if not isinstance(tool_name, str) or tool_name not in self.task.allowed_tools:
            raise PlanError(f"task {self.task.task_id} attempted a non-whitelisted tool")
        def authorize():
            if self.store.is_cancel_requested(self.run_id):
                raise asyncio.CancelledError
            self.store.reserve(self.run_id, "tool_calls", 1, self.config)

        return await _invoke_tool(self.resources.database, callback, authorize, *args, **kwargs)

    async def idempotent_tool_call(
        self,
        callback: Callable[..., Any],
        *args: Any,
        tool_name: str,
        operation: str,
        **kwargs: Any,
    ) -> Any:
        """Execute an external write once and cache its result by stable key.

        The downstream callback must accept ``idempotency_key`` and pass it to
        the external service.  If the process dies after that service accepts
        the request but before the local result commit, the next attempt is
        rejected as uncertain rather than duplicated.
        """
        if not isinstance(operation, str) or not operation.strip():
            raise ValueError("operation must be a non-empty string")
        if not isinstance(tool_name, str) or tool_name not in self.task.allowed_tools:
            raise PlanError(f"task {self.task.task_id} attempted a non-whitelisted tool")
        if self.store.is_cancel_requested(self.run_id):
            raise asyncio.CancelledError
        operation_key = f"{self.idempotency_key}:{operation.strip()}"
        state, cached = self.store.begin_effect(self.run_id, operation_key)
        if state == "cached":
            return cached
        if state == "uncertain":
            raise ExternalOperationUncertain(
                f"external operation requires reconciliation before retry: {operation_key}"
            )
        dispatched = False
        def authorize():
            nonlocal dispatched
            if self.store.is_cancel_requested(self.run_id):
                raise asyncio.CancelledError
            self.store.reserve(self.run_id, "tool_calls", 1, self.config)
            dispatched = True

        try:
            value = await _invoke_tool(
                self.resources.database, callback, authorize, *args,
                idempotency_key=operation_key, **kwargs,
            )
            self.store.complete_effect(self.run_id, operation_key, value)
            return value
        except BaseException as exc:
            # Before dispatch nothing reached the provider. After dispatch, even
            # cancellation/timeout may hide a committed write: keep it uncertain.
            if not dispatched:
                self.store.fail_effect(self.run_id, operation_key, type(exc).__name__)
            raise


@dataclass(frozen=True)
class RunExecutionContext:
    """Budgeted LLM access for graph-level planning, verification and summary nodes."""

    run_id: str
    store: SQLiteRunStore
    config: CaseExecutionConfig
    resources: SharedResourceLimits

    async def llm_call(
        self,
        callback: Callable[..., Awaitable[Any]],
        *args: Any,
        max_output_tokens: int,
        **kwargs: Any,
    ) -> Any:
        if not isinstance(max_output_tokens, int) or isinstance(max_output_tokens, bool) or max_output_tokens <= 0:
            raise ValueError("max_output_tokens must be a positive integer")
        if self.store.is_cancel_requested(self.run_id):
            raise asyncio.CancelledError
        async with self.resources.llm:
            if self.store.is_cancel_requested(self.run_id):
                raise asyncio.CancelledError
            self.store.reserve_llm(self.run_id, max_output_tokens, self.config)
            return await _invoke_async(callback, *args, **kwargs)


def validate_task_plan(
    tasks: Sequence[InvestigationTask],
    *,
    max_total_tasks: int,
    scope: CaseTaskScope,
) -> tuple[str, ...]:
    if not isinstance(scope, CaseTaskScope):
        raise PlanError("an authoritative CaseTaskScope is required")
    if len(tasks) > max_total_tasks:
        raise PlanError("total task limit exceeded")
    by_id: dict[str, InvestigationTask] = {}
    for task in tasks:
        if not isinstance(task, InvestigationTask):
            raise PlanError("every planned task must be an InvestigationTask")
        if task.task_id in by_id:
            raise PlanError(f"duplicate task_id: {task.task_id}")
        by_id[task.task_id] = task
        if not task.entity_ids:
            raise PlanError(f"task {task.task_id} must name at least one case entity")
        if not set(task.entity_ids).issubset(scope.entity_ids):
            raise PlanError(f"task {task.task_id} references an entity outside the case")
        if not task.basis_fact_ids:
            raise PlanError(f"task {task.task_id} must cite at least one supporting Fact")
        if not set(task.basis_fact_ids).issubset(scope.fact_ids):
            raise PlanError(f"task {task.task_id} references an unknown Fact")
        if not set(task.allowed_tools).issubset(scope.allowed_tools):
            raise PlanError(f"task {task.task_id} requests a tool outside the case whitelist")
        if task.time_range is not None:
            if scope.time_range is None:
                raise PlanError(f"task {task.task_id} has a time range but the case has no authorized time range")
            task_bounds = tuple(_parse_time(value, "task time_range") for value in task.time_range)
            case_bounds = tuple(_parse_time(value, "case time_range") for value in scope.time_range)
            _ensure_comparable(task_bounds[0], case_bounds[0], "task and case time_range")
            _ensure_comparable(task_bounds[1], case_bounds[1], "task and case time_range")
            if task_bounds[0][0] < case_bounds[0][0] or task_bounds[1][0] > case_bounds[1][0]:
                raise PlanError(f"task {task.task_id} time range is outside the case range")
    for task in tasks:
        missing = sorted(set(task.depends_on) - set(by_id))
        if missing:
            raise PlanError(f"task {task.task_id} has unknown dependencies: {missing}")
        if task.task_id in task.depends_on:
            raise PlanError(f"task {task.task_id} cannot depend on itself")

    indegree = {task_id: 0 for task_id in by_id}
    dependents: dict[str, list[str]] = {task_id: [] for task_id in by_id}
    for task in tasks:
        indegree[task.task_id] = len(task.depends_on)
        for dependency in task.depends_on:
            dependents[dependency].append(task.task_id)
    ready = sorted(task_id for task_id, degree in indegree.items() if degree == 0)
    order: list[str] = []
    while ready:
        task_id = ready.pop(0)
        order.append(task_id)
        for dependent in sorted(dependents[task_id]):
            indegree[dependent] -= 1
            if indegree[dependent] == 0:
                ready.append(dependent)
                ready.sort()
    if len(order) != len(by_id):
        raise PlanError("task dependency graph contains a cycle")
    return tuple(order)


def _parse_time(value: Any, label: str) -> tuple[datetime, bool]:
    if not isinstance(value, str) or not value.strip():
        raise PlanError(f"{label} bounds must be non-empty ISO-8601 strings")
    try:
        parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError as exc:
        raise PlanError(f"{label} bounds must be valid ISO-8601 strings") from exc
    aware = parsed.tzinfo is not None and parsed.utcoffset() is not None
    if aware:
        parsed = parsed.astimezone(timezone.utc)
    return parsed, aware


def _ensure_comparable(left: tuple[datetime, bool], right: tuple[datetime, bool], label: str) -> None:
    if left[1] != right[1]:
        raise PlanError(f"{label} must use either all timezone-aware or all timezone-naive bounds")


async def _invoke_async(callback: Callable[..., Awaitable[Any]], *args: Any, **kwargs: Any) -> Any:
    result = callback(*args, **kwargs)
    if not inspect.isawaitable(result):
        raise TypeError("async callback did not return an awaitable")
    return await result


async def _invoke_tool(gate, callback, authorize, *args, **kwargs):
    """Hold the resource permit until blocking I/O actually finishes.

    Cancelling an await cannot kill a thread. Releasing early would let a
    native retry exceed the database cap while the old query is still running.
    """
    await gate.acquire()
    release_here = True
    try:
        authorize()
        if inspect.iscoroutinefunction(callback) or inspect.iscoroutinefunction(getattr(callback, "__call__", None)):
            return await callback(*args, **kwargs)
        future = asyncio.get_running_loop().run_in_executor(None, partial(callback, *args, **kwargs))
        try:
            return await asyncio.shield(future)
        except asyncio.CancelledError:
            release_here = False
            def finished(completed):
                try:
                    completed.exception()
                finally:
                    gate.release()
            future.add_done_callback(finished)
            raise
    finally:
        if release_here:
            gate.release()


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":"))


def _digest(value: Any) -> str:
    return hashlib.sha256(_json(value).encode("utf-8")).hexdigest()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()

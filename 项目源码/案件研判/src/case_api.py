"""FastAPI boundary for asynchronous case runs.

The default app is deliberately deterministic and data-free for this public
repository.  Production callers can inject the same service with their
authorized planner, database adapters, verifier, and report builder.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from uuid import uuid4

from fastapi import FastAPI, HTTPException, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from .case_run_service import CaseRunService
from .candidate_events import CandidateEventBuilder
from .case_task_scheduler import (
    InvestigationTask,
    SQLiteRunStore,
    TaskExecutionContext,
    TaskSchedulerConfig,
)
from .directed_supplement import build_recheck_tasks, has_actionable_gaps


class CaseScopeRequest(BaseModel):
    entity_ids: list[str] = Field(min_length=1)
    fact_ids: list[str] = Field(default_factory=list)
    allowed_tools: list[str] = Field(default_factory=list)
    time_range: list[str] | None = None


class CaseRunRequest(BaseModel):
    case_id: str = Field(min_length=1)
    run_id: str | None = None
    case_facts: dict[str, Any] = Field(default_factory=dict)
    case_scope: CaseScopeRequest
    tasks: list[dict[str, Any]] = Field(default_factory=list)


def create_app(service: CaseRunService | None = None) -> FastAPI:
    app = FastAPI(title="Model Insight Agent Case Runs", version="0.1.0")
    active_service = service or _build_default_service()

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/v1/case-runs", status_code=status.HTTP_202_ACCEPTED)
    async def submit_case(request: CaseRunRequest) -> dict[str, Any]:
        run_id = request.run_id or uuid4().hex
        try:
            payload = request.model_dump()
            payload["run_id"] = run_id
            payload["case_scope"] = request.case_scope.model_dump()
            payload["initial_tasks"] = payload.pop("tasks", [])
            return active_service.submit(payload)
        except (ValueError, KeyError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/v1/case-runs/{run_id}")
    async def get_case_status(run_id: str) -> dict[str, Any]:
        record = active_service.status(run_id)
        if record is None:
            raise HTTPException(status_code=404, detail="unknown run_id")
        return record

    @app.post("/v1/case-runs/{run_id}/cancel")
    async def cancel_case(run_id: str) -> dict[str, Any]:
        try:
            accepted = active_service.cancel(run_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="unknown run_id") from exc
        record = active_service.status(run_id)
        return {"run_id": run_id, "cancel_requested": accepted, "status": record["status"]}

    @app.get("/v1/case-runs/{run_id}/report")
    async def get_case_report(run_id: str) -> dict[str, Any]:
        record = active_service.status(run_id)
        if record is None:
            raise HTTPException(status_code=404, detail="unknown run_id")
        report = active_service.report(run_id)
        if record["status"] != "success" or report is None:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail={"status": record["status"], "error": record["error"]},
            )
        return report

    @app.get("/v1/case-runs/{run_id}/events")
    async def case_events(run_id: str) -> StreamingResponse:
        if active_service.status(run_id) is None:
            raise HTTPException(status_code=404, detail="unknown run_id")
        return StreamingResponse(
            active_service.iter_sse_events(run_id),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    app.state.case_run_service = active_service
    return app


def _build_default_service(storage_root: Path | str | None = None) -> CaseRunService:
    root = Path(storage_root).resolve() if storage_root is not None else Path(__file__).resolve().parent.parent
    store = SQLiteRunStore(root / "runtime" / "case_runs.sqlite3")
    config = TaskSchedulerConfig()
    candidate_builder = CandidateEventBuilder(max_events=config.candidate_limit)

    async def worker(task: InvestigationTask, task_input: dict[str, Any], context: TaskExecutionContext) -> dict[str, Any]:
        # This public snapshot returns explicitly supplied facts only. The
        # injected production worker is where GaussDB/tool/LLM calls belong.
        result = {
            "task_id": task.task_id,
            "question": task.question,
            "evidence": task.payload.get("result", task_input["case_facts"]),
            "idempotency_key": context.idempotency_key,
        }
        # Explicit candidates remain a supported injection boundary. When a
        # public fact pack is supplied, deterministic rule flags and structured
        # model clues can build candidates without interpreting free-form text.
        if "candidate_events" in task.payload:
            result["candidate_events"] = task.payload["candidate_events"]
        elif "candidate_event" in task.payload:
            result["candidate_event"] = task.payload["candidate_event"]
        elif task.payload.get("source") != "verification_gap":
            case_facts = task_input.get("case_facts") or {}
            fact_pack = task.payload.get("fact_pack") or case_facts.get("fact_pack")
            if isinstance(case_facts, dict) and ("anomaly_flags" in case_facts or "focus_entities" in case_facts):
                fact_pack = fact_pack or case_facts
            model_clues = task.payload.get("model_clues") or case_facts.get("model_clues", [])
            if isinstance(fact_pack, dict):
                result["candidate_events"] = candidate_builder.build(
                    fact_pack,
                    model_clues=model_clues if isinstance(model_clues, list) else [],
                )
        # Structured verification output is an explicit seam for the public
        # synthetic fixture.  It keeps the default service runnable without
        # pretending to contain a production verifier or model.
        verification = task.payload.get("verification")
        if isinstance(verification, dict):
            for key in ("gaps", "verified_events", "final_events", "human_review"):
                if key in verification:
                    result[key] = verification[key]
        return result

    async def planner(state, context):
        return [InvestigationTask.from_dict(item) for item in state.get("initial_tasks", [])]

    async def verifier(state, context):
        gaps: list[dict[str, Any]] = []
        verified_events: list[dict[str, Any]] = []
        task_runs = state.get("task_runs", {})
        resolved_gap_keys: set[str] = set()
        for task_id, record in task_runs.items():
            task = record.get("task") if isinstance(record, dict) else None
            payload = task.get("payload") if isinstance(task, dict) else None
            if record.get("status") == "succeeded" and isinstance(payload, dict):
                if payload.get("source") == "verification_gap":
                    resolved_gap_keys.add(_gap_key(payload.get("gap"), task_id))
        for task_id, record in task_runs.items():
            if record.get("status") != "succeeded":
                gaps.append({"task_id": task_id, "reason": "task execution failed"})
                continue
            result = record.get("result")
            if not isinstance(result, dict):
                continue
            task_payload = (record.get("task") or {}).get("payload") or {}
            is_recheck = task_payload.get("source") == "verification_gap"
            for gap in result.get("gaps", []) or []:
                if isinstance(gap, dict):
                    normalized_gap = {**gap, "task_id": gap.get("task_id", task_id)}
                    if not is_recheck and _gap_key(normalized_gap, task_id) in resolved_gap_keys:
                        continue
                    gaps.append(normalized_gap)
            for event in result.get("verified_events", []) or []:
                if isinstance(event, dict):
                    verified_events.append(event)
        final_events = verified_events if not gaps and verified_events else ([] if gaps else list(state.get("candidate_events", [])))
        return {
            "gaps": gaps,
            "accepted_events": final_events,
            "final_events": final_events,
            "human_review": gaps,
        }

    async def recheck_planner(state, context):
        return build_recheck_tasks(state)

    async def recheck_allowed(state):
        return has_actionable_gaps(state)

    async def summarizer(state, context):
        return {
            "case_id": state["case_id"],
            "run_id": state["run_id"],
            "task_runs": state.get("task_runs", {}),
            "verification": state.get("verification", {}),
            "candidate_events": state.get("candidate_events", []),
            "final_events": state.get("final_events", []),
            "human_review": state.get("human_review", []),
            "limits": {"candidate_limit": context.config.candidate_limit},
            "evidence_status": "public_snapshot_synthetic_input",
        }

    return CaseRunService(
        store,
        worker=worker,
        planner=planner,
        verifier=verifier,
        recheck_planner=recheck_planner,
        recheck_allowed=recheck_allowed,
        summarizer=summarizer,
        config=config,
    )


def _gap_key(value: Any, fallback_task_id: str) -> str:
    """Return a stable key so a successful recheck can close its old gap."""

    if isinstance(value, dict):
        for key in ("candidate_event_id", "task_id"):
            item = value.get(key)
            if item:
                return str(item)
    return str(fallback_task_id)


app = create_app()

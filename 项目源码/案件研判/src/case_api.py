"""FastAPI entry using the same service as the CLI. Import creates no runtime."""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
import json
import os
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from fastapi import Response
from typing import Literal
from pydantic import Field

from .case_execution import SQLiteRunStore, CaseExecutionConfig
from .case_repository import FileCaseRepository
from .case_run_service import CaseRunService
from .investigation_contracts import Contract
from .investigation_reasoner import OfflineReasoner, OpenAIReasoner


class CaseRunRequest(Contract):
    case_id: str = Field(pattern=r"^[A-Za-z0-9_-]{1,100}$")
    run_id: str | None = Field(default=None, pattern=r"^[A-Za-z0-9_-]{1,100}$")


def build_service(*, repository_root=None, storage_root=None, offline=False, config=None):
    repository_root = repository_root or os.environ.get("CASE_REPOSITORY")
    if repository_root is None:
        raise ValueError("configure an administrator-approved CASE_REPOSITORY directory")
    root = Path(storage_root or os.environ.get("CASE_RUNTIME_DIR", "runtime/native-v3")).resolve()
    if config is None:
        config_path = os.environ.get("CASE_EXECUTION_CONFIG")
        config = CaseExecutionConfig(**json.loads(Path(config_path).read_text(encoding="utf-8"))) if config_path else CaseExecutionConfig()
    if offline or os.environ.get("CASE_REASONER") == "offline":
        reasoner = OfflineReasoner()
    else:
        model = os.environ.get("CASE_LLM_MODEL")
        api_key = os.environ.get("CASE_LLM_API_KEY")
        if not model or not api_key:
            raise ValueError("configure CASE_LLM_MODEL/CASE_LLM_API_KEY, or explicitly select offline mode")
        reasoner = OpenAIReasoner(model=model, api_key=api_key, base_url=os.environ.get("CASE_LLM_BASE_URL"))
    return CaseRunService(SQLiteRunStore(root / "runs.sqlite3"), repository=FileCaseRepository(repository_root),
        reasoner=reasoner, config=config)


def create_app(service_factory=None):
    @asynccontextmanager
    async def lifespan(app):
        app.state.case_run_service = await asyncio.to_thread(service_factory or build_service)
        try:
            yield
        finally:
            await asyncio.to_thread(app.state.case_run_service.close)

    app = FastAPI(title="Evidence-first case investigation", version="0.3.0", lifespan=lifespan)

    def service(request: Request):
        return request.app.state.case_run_service

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.post("/v1/case-runs", status_code=202)
    async def submit(body: CaseRunRequest, request: Request):
        try:
            return await asyncio.to_thread(service(request).submit, body.case_id, run_id=body.run_id)
        except (KeyError, FileNotFoundError) as exc:
            raise HTTPException(404, "case not available") from exc
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc

    @app.get("/v1/case-runs/{run_id}")
    async def status(run_id: str, request: Request):
        record = await asyncio.to_thread(service(request).status, run_id)
        if record is None:
            raise HTTPException(404, "unknown run_id")
        return record

    @app.post("/v1/case-runs/{run_id}/cancel")
    async def cancel(run_id: str, request: Request):
        try:
            return {"run_id": run_id, "cancel_requested": await asyncio.to_thread(service(request).cancel, run_id)}
        except KeyError as exc:
            raise HTTPException(404, "unknown run_id") from exc

    @app.post("/v1/case-runs/{run_id}/resume", status_code=202)
    async def resume(run_id: str, request: Request):
        try:
            return await asyncio.to_thread(service(request).resume, run_id)
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc

    @app.get("/v1/case-runs/{run_id}/report")
    async def report(run_id: str, request: Request, format: Literal["json", "md", "html", "docx"] = "json"):
        try:
            result = await asyncio.to_thread(service(request).report, run_id)
        except KeyError as exc:
            raise HTTPException(404, "unknown run_id") from exc
        if result is None:
            raise HTTPException(409, "report is not available yet")
        if format == "json":
            return result
        from .case_report import render_report
        content, media_type = await asyncio.to_thread(render_report, result, format)
        return Response(content, media_type=media_type)

    @app.get("/v1/case-runs/{run_id}/facts/{fact_id}")
    async def fact(run_id: str, fact_id: str, request: Request):
        try:
            return await asyncio.to_thread(service(request).fact, run_id, fact_id)
        except (KeyError, ValueError) as exc:
            raise HTTPException(404, "unknown published fact") from exc

    @app.get("/v1/case-runs/{run_id}/events")
    async def events(run_id: str, request: Request):
        if await asyncio.to_thread(service(request).status, run_id) is None:
            raise HTTPException(404, "unknown run_id")
        return StreamingResponse(service(request).iter_sse_events(run_id), media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    return app


app = create_app()

import sys
from pathlib import Path
from typing import List

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

if __package__ in (None, ""):
    project_root = str(Path(__file__).resolve().parents[2])
    if project_root not in sys.path:
        sys.path.append(project_root)

from backend.api.routers import chat_api, conversation_api, kb_api, runtime_api
from backend.common.logger import get_logger

logger = get_logger(__name__)


def create_app() -> FastAPI:
    app = FastAPI(
        title="DeepAssist API",
        description="API gateway for chat, RAG, agent orchestration, and knowledge-base ingestion.",
        version="2.0.0",
    )

    allow_origins: List[str] = ["*"]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allow_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(chat_api.router)
    app.include_router(conversation_api.router)
    app.include_router(kb_api.router)
    app.include_router(runtime_api.router)

    @app.get("/health", tags=["System"])
    def health_check() -> dict:
        return {
            "status": "ok",
            "service": "deepassist-api",
            "message": "DeepAssist backend is running.",
        }

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    logger.info("Starting FastAPI server...")
    uvicorn.run("backend.api.main:app", host="0.0.0.0", port=8000, reload=True)

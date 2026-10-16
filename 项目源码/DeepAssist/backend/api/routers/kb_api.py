from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile

from backend.api.dependencies import get_kb_app
from backend.application.kb_app import KnowledgeBaseApp

router = APIRouter(prefix="/api/kb", tags=["Knowledge Base"])

@router.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    collection_name: str = Query(default="tech_docs_kb"),
    app: KnowledgeBaseApp = Depends(get_kb_app),
) -> dict:
    raw_content = await file.read()
    try:
        text_content = raw_content.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise HTTPException(
            status_code=400,
            detail="Only UTF-8 text or Markdown files are supported by the current ingestion pipeline.",
        ) from exc

    return app.process_document(
        file_name=file.filename or "uploaded_document.md",
        content=text_content,
        collection_name=collection_name,
    )

@router.get("/files")
def list_files(
    collection_name: str = Query(default="tech_docs_kb"),
    app: KnowledgeBaseApp = Depends(get_kb_app),
) -> dict:
    return app.list_files(collection_name=collection_name)

@router.get("/collections")
def list_collections(
    app: KnowledgeBaseApp = Depends(get_kb_app),
) -> dict:
    return app.list_collections()


@router.get("/health")
def get_health(
    refresh: bool = Query(default=False),
    repair: bool = Query(default=False),
    batch_size: int = Query(default=256, ge=1, le=4096),
    collection_name: list[str] | None = Query(default=None),
    app: KnowledgeBaseApp = Depends(get_kb_app),
) -> dict:
    return app.get_health_report(
        refresh=refresh,
        collections=collection_name,
        repair=repair,
        batch_size=batch_size,
    )

@router.delete("/files/{source_file:path}")
def delete_file(
    source_file: str,
    collection_name: str = Query(default="tech_docs_kb"),
    app: KnowledgeBaseApp = Depends(get_kb_app),
) -> dict:
    return app.delete_file(
        file_name=source_file,
        collection_name=collection_name,
    )

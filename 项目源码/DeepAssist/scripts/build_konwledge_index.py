import hashlib
import json
import os
import re
import signal
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterator, List, Tuple

import pandas as pd
from tqdm import tqdm

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from backend.application.kb_app import KnowledgeBaseApp
from backend.common.config import settings
from backend.common.logger import get_logger
from backend.domain.entities.document import DocumentChunk
from backend.infrastructure.databases.chroma_store import ChromaStore
from backend.infrastructure.databases.kb_manifest_store import KnowledgeBaseManifestStore
from backend.infrastructure.databases.whoosh_store import WhooshStore
from backend.infrastructure.embeddings.bge_m3_local import BGEM3Local

# 🌟 修正点：正确的导入路径
from backend.domain.services.rag.chunking import DocumentChunker

logger = get_logger("build_knowledge_index")

INGEST_SCHEMA_VERSION = "kb_contract_v2"
MAX_SAMPLES_PER_FILE = int(os.getenv("DEEPASSIST_MAX_SAMPLES_PER_FILE", "50000"))
DEFAULT_BATCH_SIZE = int(os.getenv("DEEPASSIST_INGEST_BATCH_SIZE", "1024"))
SHUTDOWN_REQUESTED = False

def signal_handler(sig, frame):
    del sig, frame
    global SHUTDOWN_REQUESTED
    if SHUTDOWN_REQUESTED:
        return
    SHUTDOWN_REQUESTED = True
    print("\n" + "=" * 72)
    print("Received Ctrl+C. Finishing the current batch safely before shutdown...")
    print("=" * 72 + "\n")

signal.signal(signal.SIGINT, signal_handler)

@dataclass(frozen=True)
class IngestTask:
    collection_name: str
    path: Path
    adapter: Callable[["IngestTask", int], Iterator[Tuple[DocumentChunk, int]]]
    domain: str
    source_format: str
    language: str

class ProgressManager:
    def __init__(self, progress_path: str):
        self.progress_file = Path(progress_path)
        self.progress_file.parent.mkdir(parents=True, exist_ok=True)
        self.progress = self._load()

    def _load(self) -> dict:
        if not self.progress_file.exists():
            return {}
        try:
            with self.progress_file.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            return payload if isinstance(payload, dict) else {}
        except Exception:
            return {}

    def get_last_index(self, file_path: str) -> int:
        return int(self.progress.get(file_path, -1))

    def save_index(self, file_path: str, index: int) -> None:
        self.progress[file_path] = int(index)
        with self.progress_file.open("w", encoding="utf-8") as handle:
            json.dump(self.progress, handle, ensure_ascii=False, indent=2)

def stable_source_file(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except Exception:
        return path.as_posix()

def stable_chunk_id(source_file: str, record_index: int, local_key: str = "") -> str:
    raw = f"{source_file}::{record_index}::{local_key}"
    return hashlib.md5(raw.encode("utf-8")).hexdigest()

def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()

def truncate(text: str, limit: int = 160) -> str:
    text = normalize_whitespace(text)
    if len(text) <= limit:
        return text
    return f"{text[:limit]}..."

def sanitize_metadata(metadata: Dict[str, object]) -> Dict[str, object]:
    sanitized: Dict[str, object] = {}
    for key, value in metadata.items():
        if value is None:
            continue
        if isinstance(value, (str, int, float, bool)):
            sanitized[key] = value
        elif isinstance(value, Path):
            sanitized[key] = value.as_posix()
        elif isinstance(value, (list, tuple)):
            sanitized[key] = [str(item) for item in value if item is not None]
        elif isinstance(value, dict):
            sanitized[key] = {
                str(inner_key): str(inner_value)
                for inner_key, inner_value in value.items()
                if inner_value is not None
            }
        else:
            sanitized[key] = str(value)
    return sanitized

def build_metadata(
    task: IngestTask,
    record_index: int,
    source_file: str,
    source_type: str,
    source_title: str = "",
    extra: Dict[str, object] | None = None,
) -> Dict[str, object]:
    metadata = {
        "schema_version": INGEST_SCHEMA_VERSION,
        "collection_name": task.collection_name,
        "source_file": source_file,
        "source_path": source_file,
        "source_display_name": Path(source_file).name,
        "domain": task.domain,
        "source_format": task.source_format,
        "source_type": source_type,
        "language": task.language,
        "record_index": int(record_index),
        "source_title": truncate(source_title, 180),
        "ingestion_pipeline": "offline_build_script_v2",
    }
    if extra:
        metadata.update(extra)
    return sanitize_metadata(metadata)

def make_qa_chunk(
    task: IngestTask,
    source_file: str,
    record_index: int,
    question: str,
    answer: str,
    local_key: str = "qa",
    extra: Dict[str, object] | None = None,
) -> DocumentChunk:
    question_text = normalize_whitespace(question)
    answer_text = normalize_whitespace(answer)
    content = f"Q: {question_text}\nA: {answer_text}"
    metadata = build_metadata(
        task=task,
        record_index=record_index,
        source_file=source_file,
        source_type="qa",
        source_title=question_text,
        extra=extra,
    )
    return DocumentChunk(
        id=stable_chunk_id(source_file, record_index, local_key),
        content=content,
        metadata=metadata,
    )

def adapter_medical(task: IngestTask, start_idx: int) -> Iterator[Tuple[DocumentChunk, int]]:
    source_file = stable_source_file(task.path)
    with task.path.open("r", encoding="utf-8") as handle:
        for row_index, line in enumerate(handle):
            if row_index >= MAX_SAMPLES_PER_FILE:
                break
            if row_index <= start_idx or not line.strip():
                continue
            try:
                payload = json.loads(line)
            except Exception:
                continue
            question = f"{payload.get('instruction', '')}{payload.get('input', '')}"
            answer = payload.get("output", "")
            yield (
                make_qa_chunk(
                    task=task,
                    source_file=source_file,
                    record_index=row_index,
                    question=question,
                    answer=answer,
                    extra={"dataset_name": "medical_qa"},
                ),
                row_index,
            )

def adapter_legal(task: IngestTask, start_idx: int) -> Iterator[Tuple[DocumentChunk, int]]:
    source_file = stable_source_file(task.path)
    with task.path.open("r", encoding="utf-8") as handle:
        for row_index, line in enumerate(handle):
            if row_index >= MAX_SAMPLES_PER_FILE:
                break
            if row_index <= start_idx or not line.strip():
                continue
            try:
                payload = json.loads(line)
            except Exception:
                continue
            yield (
                make_qa_chunk(
                    task=task,
                    source_file=source_file,
                    record_index=row_index,
                    question=payload.get("input", ""),
                    answer=payload.get("output", ""),
                    local_key=str(payload.get("id", "qa")),
                    extra={"dataset_name": "disc_law_sft"},
                ),
                row_index,
            )

def adapter_financial(task: IngestTask, start_idx: int) -> Iterator[Tuple[DocumentChunk, int]]:
    source_file = stable_source_file(task.path)
    with task.path.open("r", encoding="utf-8") as handle:
        payloads = json.load(handle)

    for row_index, payload in enumerate(payloads):
        if row_index >= MAX_SAMPLES_PER_FILE:
            break
        if row_index <= start_idx:
            continue
        raw_output = payload.get("output", "")
        clean_output = re.sub(r"\[Calculator.*?\]", "", raw_output)
        yield (
            make_qa_chunk(
                task=task,
                source_file=source_file,
                record_index=row_index,
                question=f"{payload.get('instruction', '')}{payload.get('input', '')}",
                answer=clean_output,
                extra={"dataset_name": "disc_fin_sft"},
            ),
            row_index,
        )

def adapter_enron(task: IngestTask, start_idx: int) -> Iterator[Tuple[DocumentChunk, int]]:
    source_file = stable_source_file(task.path)
    chunker = DocumentChunker()
    frame = pd.read_parquet(task.path)

    for row_index, row in frame.iterrows():
        if row_index >= MAX_SAMPLES_PER_FILE:
            break
        if row_index <= start_idx:
            continue

        questions = row.get("questions", []) or []
        answers = row.get("gold_answers", []) or []
        if isinstance(questions, (list, tuple)) and isinstance(answers, (list, tuple)):
            for qa_index, (question, answer) in enumerate(zip(questions, answers)):
                yield (
                    make_qa_chunk(
                        task=task,
                        source_file=source_file,
                        record_index=int(row_index),
                        question=str(question),
                        answer=str(answer),
                        local_key=f"qa::{qa_index}",
                        extra={
                            "dataset_name": "enron_qa_0922",
                            "qa_pair_index": qa_index,
                        },
                    ),
                    int(row_index),
                )

        email_content = str(row.get("email", "") or "")
        if not email_content.strip():
            continue
        for chunk_index, chunk in enumerate(
            chunker.split_markdown(email_content, source_name=source_file)
        ):
            merged_metadata = {
                **chunk.metadata,
                **build_metadata(
                    task=task,
                    record_index=int(row_index),
                    source_file=source_file,
                    source_type="email",
                    source_title=str(row.get("subject", "") or row.get("file", "") or "email"),
                    extra={
                        "dataset_name": "enron_qa_0922",
                        "chunk_kind": "email_body",
                        "email_chunk_index": chunk_index,
                    },
                ),
            }
            chunk.id = stable_chunk_id(source_file, int(row_index), f"email::{chunk_index}")
            chunk.metadata = sanitize_metadata(merged_metadata)
            yield chunk, int(row_index)

def existing_manifest_chunk_count(
    manifest_store: KnowledgeBaseManifestStore,
    collection_name: str,
    source_file: str,
) -> int:
    for item in manifest_store.list_files(collection_name):
        if item.get("source_file") == source_file:
            return int(item.get("chunk_count", 0) or 0)
    return 0

def reset_file_state(
    task: IngestTask,
    source_file: str,
    vector_db: ChromaStore,
    keyword_db: WhooshStore,
) -> None:
    logger.info("Resetting historical chunks for %s", source_file)
    vector_db.delete_by_source(task.collection_name, source_file)
    keyword_db.delete_by_source(task.collection_name, source_file)

def flush_batch(
    collection_name: str,
    chunks: List[DocumentChunk],
    kb_app: KnowledgeBaseApp,
) -> int:
    if not chunks:
        return 0

    result = kb_app.process_chunks(collection_name, chunks)
    if result.get("status") == "success":
        return len(chunks)

    raise RuntimeError(
        f"Batch ingest failed for collection={collection_name}: {result.get('message')}"
    )

def ingest_file_to_db(
    task: IngestTask,
    embedding_model: BGEM3Local,
    vector_db: ChromaStore,
    keyword_db: WhooshStore,
    progress_mgr: ProgressManager,
    manifest_store: KnowledgeBaseManifestStore,
    kb_app: KnowledgeBaseApp,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> None:
    del embedding_model

    global SHUTDOWN_REQUESTED

    source_file = stable_source_file(task.path)
    progress_key = str(task.path.resolve())
    start_idx = progress_mgr.get_last_index(progress_key)
    previous_chunk_count = existing_manifest_chunk_count(
        manifest_store,
        task.collection_name,
        source_file,
    )

    if start_idx >= 0 and previous_chunk_count == 0:
        logger.warning(
            "Progress exists for %s but manifest is missing. Rebuilding this source from scratch.",
            source_file,
        )
        start_idx = -1

    if start_idx == -1:
        reset_file_state(task, source_file, vector_db, keyword_db)
        previous_chunk_count = 0
    else:
        logger.info(
            "Resuming %s from record index %s",
            source_file,
            start_idx + 1,
        )

    batch_chunks: List[DocumentChunk] = []
    last_row_idx = start_idx
    inserted_chunks = 0
    progress = tqdm(desc=f"Ingest {task.path.name}", unit="chunk")

    for chunk, row_index in task.adapter(task, start_idx):
        if SHUTDOWN_REQUESTED:
            break

        batch_chunks.append(chunk)
        last_row_idx = row_index
        if len(batch_chunks) < batch_size:
            continue

        inserted_chunks += flush_batch(
            task.collection_name,
            batch_chunks,
            kb_app,
        )
        progress_mgr.save_index(progress_key, last_row_idx)
        progress.update(len(batch_chunks))
        batch_chunks = []

    if batch_chunks:
        inserted_chunks += flush_batch(
            task.collection_name,
            batch_chunks,
            kb_app,
        )
        progress_mgr.save_index(progress_key, last_row_idx)
        progress.update(len(batch_chunks))

    progress.close()

    total_chunk_count = previous_chunk_count + inserted_chunks
    if start_idx == -1 and not SHUTDOWN_REQUESTED:
        total_chunk_count = inserted_chunks

    if total_chunk_count > 0:
        manifest_store.upsert_file(
            collection_name=task.collection_name,
            source_file=source_file,
            chunk_count=total_chunk_count,
            metadata={
                "collection_name": task.collection_name,
                "domain": task.domain,
                "source_format": task.source_format,
                "language": task.language,
                "source_path": source_file,
                "source_display_name": task.path.name,
                "schema_version": INGEST_SCHEMA_VERSION,
            },
        )

    logger.info(
        "Completed ingest for %s: inserted=%s total_manifest_chunks=%s",
        source_file,
        inserted_chunks,
        total_chunk_count,
    )

def build_tasks() -> List[IngestTask]:
    sources_dir = PROJECT_ROOT / "data" / "sources"
    return [
        IngestTask(
            collection_name="medical_kb",
            path=sources_dir / "medical" / "finetune" / "train_zh_0.json",
            adapter=adapter_medical,
            domain="medical",
            source_format="jsonl",
            language="zh",
        ),
        IngestTask(
            collection_name="legal_kb",
            path=sources_dir / "DISC-Law-SFT" / "DISC-Law-SFT-Pair-QA-released.jsonl",
            adapter=adapter_legal,
            domain="legal",
            source_format="jsonl",
            language="zh",
        ),
        IngestTask(
            collection_name="financial_kb",
            path=sources_dir / "DISC-FIN-SFT" / "data" / "total.json",
            adapter=adapter_financial,
            domain="financial",
            source_format="json",
            language="zh",
        ),
        IngestTask(
            collection_name="enron_kb",
            path=sources_dir / "enron_qa_0922" / "data" / "train-00000-of-00002.parquet",
            adapter=adapter_enron,
            domain="office",
            source_format="parquet",
            language="en",
        ),
        IngestTask(
            collection_name="enron_kb",
            path=sources_dir / "enron_qa_0922" / "data" / "train-00001-of-00002.parquet",
            adapter=adapter_enron,
            domain="office",
            source_format="parquet",
            language="en",
        ),
    ]

def main() -> None:
    logger.info(
        "Initializing offline KB build pipeline: max_samples_per_file=%s batch_size=%s",
        MAX_SAMPLES_PER_FILE,
        DEFAULT_BATCH_SIZE,
    )

    embedding_model = BGEM3Local()
    vector_db = ChromaStore()
    keyword_db = WhooshStore()
    progress_mgr = ProgressManager(settings.INGEST_PROGRESS_PATH)
    manifest_store = KnowledgeBaseManifestStore(settings.KB_MANIFEST_PATH)

    kb_app = KnowledgeBaseApp(
        chunker=DocumentChunker(),
        embedding_model=embedding_model,
        vector_db=vector_db,
        keyword_db=keyword_db,
    )

    for task in build_tasks():
        if SHUTDOWN_REQUESTED:
            break

        logger.info("=" * 72)

        if not task.path.exists():
            logger.error("Source file is missing, skip: %s", task.path)
            continue

        ingest_file_to_db(
            task=task,
            embedding_model=embedding_model,
            vector_db=vector_db,
            keyword_db=keyword_db,
            progress_mgr=progress_mgr,
            manifest_store=manifest_store,
            kb_app=kb_app,
            batch_size=DEFAULT_BATCH_SIZE,
        )

    if SHUTDOWN_REQUESTED:
        logger.info("Offline KB build stopped safely. Progress and manifest were preserved.")
    else:
        logger.info("Offline KB build finished successfully.")

if __name__ == "__main__":
    main()
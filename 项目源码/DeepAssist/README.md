# DeepAssist

DeepAssist is an agent-oriented LLM application backend for vertical knowledge-base scenarios. The project is being refactored toward an enterprise-style architecture: layered boundaries, replaceable adapters, measurable RAG and Agent behavior, structured streaming events, and room for production components without turning the codebase into a framework showcase.

## Current Status

Primary endpoint:

```text
POST /api/conversation/stream
```

Each user turn is routed by `ConversationRouter` into one of:

```text
direct_chat | rag | agent | clarify
```

The first SSE event from the unified endpoint is expected to be `routing_decision`.

Legacy debugging endpoints still exist:

```text
POST /api/chat/stream
POST /api/agent/stream
```

They are kept for compatibility and targeted debugging. New application flows should prefer `/api/conversation/stream`.

## Implemented Capabilities

- FastAPI backend with dependency assembly in `backend/api/dependencies.py`.
- Unified conversation routing with structured `RouteDecision`.
- SSE streaming through structured `StreamEvent` objects.
- DeepSeek LLM adapter behind `BaseLLM`, with an optional generic
  OpenAI-compatible endpoint adapter.
- RAG pipeline with BGE-M3 embeddings, Chroma vector search, Elasticsearch keyword search with a local Whoosh fallback, Weighted RRF fusion, context packing, and answer guard.
- Conversation memory, summaries, task snapshots, and routing logs through the memory-store port.
- Agent workflow with `ToolRegistry`, `ToolPolicy`, ReAct-style execution, supervisor/worker routing, duplicate tool-call protection, partial-result recovery, and task snapshots.
- LangGraph runtime adapter with explicit plan/gate/execute/final nodes, checkpoint-before-approval, and DeepAssist task snapshots for HITL resume.
- Basic Capability / ToolGroup presentation for runtime tool governance.
- Bidirectional MCP integration with `DeepAssistMCPServer` and synchronous/asynchronous `MCPToolClientAdapter` methods.
- Versioned Skill registry that injects task-selected instructions, constraints,
  check standards, tools, and resources into Agent context.
- Offline route/RAG/Agent-tool eval runner for local regression gates.
- Local structured tracing for `/api/conversation/stream`, queryable through runtime APIs.
- Local HITL approval flow for high-risk tool calls, including pending-action listing, approve, reject, and approved execution.
- Development Python sandbox tool for local calculation and experiment tasks, gated by HITL by default.
- Knowledge-base upload, file listing, collection listing, health checks, and delete flow.
- Core read-only tools for knowledge-base search/catalog and controlled local file reading.

Demo tools for weather lookup, Python execution, and SQL querying have been removed from the core runtime. Future non-core tools should be connected through controlled `ToolProvider`, MCP, LangChain-compatible adapters, and HITL policy gates when risk is high.

## Architecture Direction

DeepAssist follows a pragmatic ports-and-adapters design:

```text
backend/
  api/                 HTTP, request models, dependency wiring
  application/         use-case orchestration
  domain/
    entities/          stable business objects
    interfaces/        ports
    services/          domain policies and strategies
  infrastructure/      SDK, database, search, runtime, tool, MCP, and tracing adapters
evals/                 future route/RAG/Agent evaluation cases
tests/                 regression and security tests
```

Domain and Application code should not depend directly on LangGraph, MCP SDKs, Elasticsearch, MySQL, Milvus, Chroma, Whoosh, or vendor-specific LLM SDK types. Concrete SDK integration belongs in Infrastructure adapters.

## Implemented vs Planned

Implemented:

- SQLite memory store
- MySQL memory-store adapter
- Chroma vector-store adapter
- Whoosh keyword-search adapter
- Elasticsearch keyword-search adapter with resilient local fallback
- DeepSeek LLM adapter
- Optional generic OpenAI-compatible LLM adapter
- Local BGE-M3 embedding adapter
- Cross-Encoder reranker adapter (default `BAAI/bge-reranker-base`) with
  lexical/no-op fallback when the model backend is unavailable
- LangChain-compatible Retriever/Runnable boundary adapters that normalize
  framework results into project-owned domain objects, plus loader/splitter and
  vector-store adapters for the KB ingestion and retrieval ports
- Custom Agent runtime based on the existing `AgentEngine`
- Capability / ToolGroup runtime presentation
- Read-only knowledge-base and file tools
- Development high-risk tools for sandbox file writes and Python execution, visible to the agent but gated by HITL by default
- Offline eval runner with route, RAG expectation, and agent tool expectation metrics
- Structured local trace store and optional LangSmith trace sink for request,
  node, retrieval, model, and tool events (`LANGSMITH_TRACING=true` plus an API key)
- Local HITL approval store and runtime pending-action endpoints
- SQLite-backed LangGraph checkpoints by default (`LANGGRAPH_CHECKPOINT_BACKEND=sqlite`);
  `memory` is reserved for isolated tests
- Regression, security, and acceptance contract test suite

Remaining production work:

- Live route/RAG/Agent eval metrics against real LLM, embedding, retrieval, and tool adapters
- Persistent tracing backend and OpenTelemetry/Langfuse/Phoenix adapters
- Docker Compose with backend, MySQL, and Elasticsearch
- Docling / Unstructured / LlamaIndex readers for richer document ingestion
- Async background ingestion jobs
- Milvus or Qdrant vector-store adapter

## Environment

Recommended local environment:

```bash
conda activate llm-dev
pip install -r requirements.txt
```

Python 3.10 or 3.11 is recommended for deployment smoke tests because several ML
and retrieval dependencies can lag behind the newest Python releases.

Create `.env` from `.env.example` and fill in secrets locally. Do not commit `.env`, model files, vector indexes, keyword indexes, database files, or uploaded knowledge-base data.

DeepAssist uses DeepSeek through `DeepSeekClient` by default. If a deployment
places an OpenAI-compatible gateway in front of the configured DeepSeek model,
set `LLM_PROVIDER=openai_compatible` and point
`OPENAI_COMPATIBLE_BASE_URL` to its `/v1` endpoint. The protocol setting does
not change the project's model choice.

`OpenAI-compatible` describes the HTTP request/response contract, not a model
family. A locally deployed Qwen (or any other model) is compatible only when
its serving layer exposes the same chat-completions contract; the model name
and endpoint must then be configured explicitly. The DeepAssist project
remains DeepSeek-only unless that deployment choice is intentionally changed.

Reranking uses the Cross-Encoder backend by default with
`RERANKER_MODEL_NAME=BAAI/bge-reranker-base`. If the optional model dependency
or model load fails, the adapter records the reason and falls back to lexical
reranking; this fallback is a resilience path, not the normal application
configuration. LangSmith tracing is also optional and is used only when
`LANGSMITH_TRACING=true` and an API key are configured.

Local development tool policy:

```env
ALLOWED_TOOL_CATEGORIES=read_only,write_file,code_exec
ENABLE_DEVELOPMENT_TOOLS=true
TOOL_APPROVAL_MODE=hitl
TOOL_TIMEOUT_SECONDS=30
```

With `ENABLE_DEVELOPMENT_TOOLS=true`, local sandbox write and Python execution tools are visible to the model for testing. With `TOOL_APPROVAL_MODE=hitl`, high-risk tools pause at `pending_action` until approved through `/api/runtime/pending-actions/{action_id}/approve`. For one-machine development testing only, set `TOOL_APPROVAL_MODE=disabled` to execute enabled high-risk tools directly.

## Run

```bash
uvicorn backend.api.main:app --host 0.0.0.0 --port 8000 --reload
```

Health check:

```text
GET /health
```

OpenAPI docs:

```text
http://localhost:8000/docs
```

## Example Requests

Unified conversation stream:

```bash
curl -X POST "http://localhost:8000/api/conversation/stream" \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "demo-session",
    "query": "根据知识库说明 DeepAssist 的 RAG 流程，并给出引用。",
    "collection_name": "__all__",
    "model_name": "deepseek-chat",
    "temperature": 0.7,
    "top_p": 1.0,
    "history_rounds": 10,
    "use_user_memory": false
  }'
```

Knowledge-base upload:

```bash
curl -X POST "http://localhost:8000/api/kb/upload" \
  -F "file=@README.md" \
  -F "collection_name=tech_docs_kb"
```

Runtime capabilities:

```bash
curl "http://localhost:8000/api/runtime/capabilities"
```

## Quality Gate

Before ending a refactor stage, run:

```bash
python -m compileall backend tests evals
python -m pytest tests
```

For text-cleanup stages, also scan user-visible files for common mojibake markers and Unicode replacement characters.

## Public Benchmark Evaluation

DeepAssist includes an evaluation pipeline for public benchmark datasets. The
checked-in fixture under `evals/fixtures/processed/` is only a smoke test and
must not be used as a resume or README performance claim.

Install evaluation dependencies:

```bash
conda activate llm-dev
pip install -r requirements.txt
```

Optional LLM-as-judge metrics such as faithfulness and answer relevancy use
separate dependencies because Ragas/DeepEval can constrain LangChain-related
packages differently from the runtime stack:

```bash
pip install -r requirements-eval-judge.txt
```

Download public datasets to local disk:

```bash
python evals/download_datasets.py \
  --cache-dir data/eval_datasets/raw \
  --datasets beir_nfcorpus,beir_fiqa,squad,hotpotqa,gsm8k,bfcl,cmrc2018 \
  --sample-size 300
```

Convert downloaded datasets into DeepAssist eval cases:

```bash
python evals/prepare_datasets.py \
  --raw-dir data/eval_datasets/raw \
  --out-dir data/eval_datasets/processed \
  --profile local
```

Run deterministic processed-file metrics:

```bash
python evals/run_eval.py --suite all --sample-size 20 --offline
```

Run live backend metrics after starting the API server:

```bash
python evals/run_eval.py \
  --suite all \
  --sample-size 20 \
  --live \
  --api-base http://localhost:8000/api
```

Generate the Markdown report used for portfolio documentation:

```bash
python evals/report.py --run-id latest --output docs/evaluation_report.md
```

Only copy metrics from `docs/evaluation_report.md` after running public
datasets, not from the built-in smoke fixture.

## Resume-Oriented Description

DeepAssist uses layered architecture and ports/adapters to keep core Agent and RAG policies stable while allowing infrastructure components to be replaced. Domain defines route decisions, tool calls, tool results, document chunks, stream events, memory/search/LLM/rerank ports, and policy services. Infrastructure adapts DeepSeek, an optional OpenAI-compatible endpoint, BGE-M3, Chroma, Elasticsearch with Whoosh fallback, SQLite, MySQL, and Cross-Encoder reranking, with LangGraph, MCP, Skill, tracing, and evaluation adapters behind those boundaries. The outer task flow is a project-level planning and reflection design; it is not presented as a formal implementation of a paper-named Plan-and-Solve algorithm.

## License

This project is for learning, engineering practice, and portfolio demonstration.

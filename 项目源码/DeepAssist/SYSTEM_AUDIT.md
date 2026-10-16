# DeepAssist System Audit

## Status

This audit records the local acceptance surface before deployment. Production
observability and external service validation remain separate concerns.

## Fixed In This Pass

- Corrected runtime backend wiring so `AGENT_RUNTIME_BACKEND=custom` uses
  `CustomAgentRuntimeAdapter`; `supervisor` and `legacy` keep the previous
  supervisor path.
- Replaced mojibake in LLM, embedding, and lexical reranker code paths.
- Converted the LangChain tool adapter from a non-functional placeholder into a
  `ToolProviderPort` implementation.
- Reclassified local write-file capability as high risk instead of grouping it
  under read-only file access.
- Replaced the eval runner placeholder with a scaffold validator for eval JSONL
  case files.
- Clarified `LocalTraceStore` as an in-memory diagnostic store rather than a
  production tracing backend.
- Fixed runtime adapter terminal status propagation so failed AgentEngine runs
  are not reported as completed runtime nodes.
- Changed category-based `ToolPolicy` behavior to fail closed when tool category
  metadata is missing.
- Hid tools blocked by policy from OpenAI tool schemas so default high-risk tools
  are not advertised to the model.
- Added structured diagnostics for unavailable MySQL memory-store initialization.
- Removed hard-coded write-file instructions from the agent prompt.
- Added an optional generic OpenAI-compatible endpoint adapter while keeping
  DeepSeek as the project's configured model.
- Added LangGraph checkpoint-before-approval with SQLite task snapshots for
  application-level HITL resume.
- Added resilient Elasticsearch/Whoosh keyword retrieval, asynchronous MCP
  discovery/execution, and versioned Skill selection.
- Added structured model-call events across Agent, LangGraph, worker, and chat
  streaming paths, with trace metadata for model/tool/state correlation.
- Added a configurable ToolPolicy execution timeout with structured timeout
  diagnosis and a resume-facing feature matrix enforced by a test gate.

## Deliberately Deferred

- Live route/RAG/agent metrics against real LLM, embedding, retrieval, and
  tool adapters remain a separate evaluation phase.
- The default trace sink remains local/in-memory for the offline demo; an
  optional LangSmith adapter now mirrors request/node/model/tool events when
  explicitly enabled. Durable trace storage and OpenTelemetry/Langfuse/Phoenix
  remain outside this public snapshot.
- HITL approval is still required before high-risk write/code/db tools should be
  enabled by default.
- Real MySQL, Elasticsearch, and external MCP integration tests should run in a
  deployment or docker-compose validation environment.
- LangGraph graph-owned execution is intentionally coordinated with the
  application snapshot store: checkpoints pause before approval, while the
  approved request is rehydrated from the persisted task snapshot.
- The local runtime is suitable for deterministic acceptance tests; deployment
  smoke tests still need live Elasticsearch, model, and external MCP services.

## Validation Baseline

- `python -m compileall backend tests evals`
- `python -m pytest tests -q`
- `python -m pytest tests/test_openai_compatible.py -q`
- `python evals/run_eval.py`
- mojibake scan
- removed demo tool scan
- `git diff --check`

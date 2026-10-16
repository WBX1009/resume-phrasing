# DeepAssist acceptance contract

This file is the evidence checklist for the resume-facing capabilities. It is
deliberately deterministic: external model, Elasticsearch, embedding, and MCP
services are required only for deployment smoke tests, not for the local gate.

## Required gates

```powershell
python -m compileall backend tests evals
python -m pytest -q
```

The acceptance suite must cover:

- every row in `docs/resume_feature_matrix.md` is marked `✅` and names a
  passing unit or regression test;

- the domain/application boundary has no direct vendor SDK imports;
- the LangGraph plan/gate/execute/final graph emits a checkpointed pending
  approval event and the application snapshot store can rehydrate it;
- the DeepSeek adapter and optional OpenAI-compatible adapter send model,
  messages, tools, temperature, and streaming requests through
  `chat.completions`;
- Chroma/vector and Elasticsearch/Whoosh keyword adapters share the keyword
  port and RRF/citation code remains adapter-independent;
- MCP discovery and execution work through both synchronous and asynchronous
  client methods;
- versioned Skills are selected by task and injected into Agent context;
- SQLite task snapshots survive construction of a new store instance;
- the unified conversation endpoint continues to emit structured routing,
  retrieval/citation, Agent, trace, and SSE end events.
- model calls, tool results, and task snapshots are emitted as structured
  events and retained in the local trace stream;
- tool execution has argument, permission, approval, error-diagnosis, and
  configurable timeout gates.

## OpenAI-compatible terminology

DeepSeek is the project model. OpenAI-compatible describes only the HTTP
serving contract. The claim is valid when the configured endpoint exposes the
OpenAI chat-completions shape, usually at `/v1/chat/completions`.

A local Qwen deployment is compatible only when its serving layer exposes that
same contract; the model name itself does not imply compatibility.

## Deployment checks

For a real environment, repeat the gate with a live DeepSeek endpoint,
Elasticsearch, the embedding model, and the MCP transport. Record latency,
retrieval metrics, tool success, and HITL resume results separately from the
deterministic unit gate.

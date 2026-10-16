"""Small, deterministic gates for the capabilities described in the README/resume."""

import ast
from pathlib import Path
import sqlite3
from types import SimpleNamespace

from backend.common.config import settings
from backend.domain.entities.agent_run import AgentRunConfig
from backend.domain.interfaces.agent_runtime import AgentRuntimeRequest
from backend.domain.entities.task_snapshot import TaskSnapshot
from backend.domain.entities.tooling import ToolCall
from backend.domain.services.agent.engine import AgentEngine
from backend.domain.services.agent.tooling import ToolPolicy, ToolRegistry
from backend.infrastructure.agent_runtime.langgraph_runtime import LangGraphAgentRuntimeAdapter
from backend.infrastructure.databases.sqlite_memory import SQLiteMemoryStore
from backend.infrastructure.tools.python_sandbox import execute_python_sandbox


ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN = {
    "langgraph",
    "mcp",
    "chromadb",
    "whoosh",
    "elasticsearch",
    "pymysql",
    "openai",
    "fastapi",
}


def test_domain_and_application_keep_vendor_sdks_out_of_business_layers():
    violations = []
    for layer in (ROOT / "backend" / "domain", ROOT / "backend" / "application"):
        for path in layer.rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    names = [item.name.split(".")[0] for item in node.names]
                elif isinstance(node, ast.ImportFrom):
                    names = [(node.module or "").split(".")[0]]
                else:
                    continue
                for name in names:
                    if name in FORBIDDEN:
                        violations.append(f"{path.relative_to(ROOT)}:{node.lineno}:{name}")
    assert violations == []


def test_langgraph_checkpoint_gate_surfaces_pending_approval():
    class FakeLLM:
        def chat(self, messages, tools=None, **kwargs):
            return SimpleNamespace(
                content="I will calculate this.",
                reasoning_content=None,
                tool_calls=[
                    {
                        "id": "call-checkpoint",
                        "type": "function",
                        "function": {
                            "name": "execute_python_sandbox",
                            "arguments": '{"code":"print(2 + 2)","timeout_seconds":5}',
                        },
                    }
                ],
            )

    registry = ToolRegistry.from_callables(
        [execute_python_sandbox],
        policy=ToolPolicy(
            allowed_categories={"code_exec"},
            tool_categories={"execute_python_sandbox": "code_exec"},
            approval_mode="hitl",
        ),
    )
    runtime = LangGraphAgentRuntimeAdapter(
        AgentEngine(
            llm=FakeLLM(),
            tool_registry=registry,
            run_config=AgentRunConfig(max_iterations=3),
        )
    )
    events = list(
        runtime.stream_run(
            type(
                "Request",
                (),
                {
                    "session_id": "acceptance-checkpoint",
                    "query": "calculate",
                    "messages": [{"role": "user", "content": "calculate"}],
                    "model_options": {},
                    "resume_state": None,
                },
            )()
        )
    )
    snapshot = next(event for event in events if event.event == "task_snapshot")
    assert snapshot.data["status"] == "pending_approval"
    assert snapshot.data["checkpoint_id"]
    assert any(event.event == "pending_action" for event in events)
    assert any(event.event == "model_call" for event in events)


def test_sqlite_snapshot_survives_store_reconstruction(tmp_path, monkeypatch):
    db_path = tmp_path / "memory.db"
    monkeypatch.setattr(settings, "CONVERSATION_DB_PATH", str(db_path))
    first = SQLiteMemoryStore()
    expected = TaskSnapshot(
        session_id="persistent-session",
        query="resume this",
        route_worker="langgraph_agent_runtime",
        status="interrupted",
        payload={"messages": [{"role": "user", "content": "resume this"}]},
    )
    assert first.save_task_snapshot(expected) is True
    second = SQLiteMemoryStore()
    actual = second.get_task_snapshot("persistent-session")
    assert actual is not None
    assert actual.payload == expected.payload


def test_langgraph_runtime_persists_native_checkpoints_in_sqlite(tmp_path, monkeypatch):
    class FakeLLM:
        model_name = "checkpoint-test-model"

        def chat(self, messages, tools=None, **kwargs):
            return SimpleNamespace(content="done", reasoning_content=None, tool_calls=[])

    monkeypatch.setattr(settings, "LANGGRAPH_CHECKPOINT_BACKEND", "sqlite")
    monkeypatch.setattr(
        settings,
        "LANGGRAPH_CHECKPOINT_DB_PATH",
        str(tmp_path / "langgraph-checkpoints.db"),
    )
    runtime = LangGraphAgentRuntimeAdapter(
        AgentEngine(
            llm=FakeLLM(),
            tool_registry=ToolRegistry.from_callables([]),
            run_config=AgentRunConfig(max_iterations=2),
        )
    )
    list(
        runtime.stream_run(
            AgentRuntimeRequest(
                session_id="sqlite-checkpoint-session",
                query="say done",
                messages=[{"role": "user", "content": "say done"}],
            )
        )
    )

    assert type(runtime._checkpointer).__name__ == "SqliteSaver"
    with sqlite3.connect(tmp_path / "langgraph-checkpoints.db") as connection:
        assert connection.execute("SELECT COUNT(*) FROM checkpoints").fetchone()[0] > 0

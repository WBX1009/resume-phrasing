import os
from pathlib import Path
import subprocess
import sys


def _run_child(script: str, checkpoint_path: Path) -> None:
    environment = os.environ.copy()
    environment["CASE_CHECKPOINT_PATH"] = str(checkpoint_path)
    subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        cwd=Path(__file__).resolve().parents[1],
        env=environment,
    )


def test_langgraph_sqlite_checkpoint_survives_process_boundary(tmp_path: Path):
    checkpoint_path = tmp_path / "cross-process-checkpoint.sqlite3"
    writer = """
import asyncio
import os
from typing import TypedDict
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import END, START, StateGraph

class State(TypedDict):
    value: str

def set_value(state):
    return {"value": "persisted-by-first-process"}

async def main():
    builder = StateGraph(State)
    builder.add_node("set_value", set_value)
    builder.add_edge(START, "set_value")
    builder.add_edge("set_value", END)
    async with AsyncSqliteSaver.from_conn_string(os.environ["CASE_CHECKPOINT_PATH"]) as saver:
        graph = builder.compile(checkpointer=saver)
        await graph.ainvoke(
            {"value": "initial"},
            {"configurable": {"thread_id": "cross-process-thread"}},
        )

asyncio.run(main())
"""
    reader = """
import os
import sqlite3
from langgraph.checkpoint.sqlite import SqliteSaver

with sqlite3.connect(os.environ["CASE_CHECKPOINT_PATH"], check_same_thread=False) as connection:
    saver = SqliteSaver(connection)
    item = saver.get_tuple({"configurable": {"thread_id": "cross-process-thread"}})
    assert item is not None
    assert item.checkpoint["channel_values"]["value"] == "persisted-by-first-process"
"""

    _run_child(writer, checkpoint_path)
    _run_child(reader, checkpoint_path)


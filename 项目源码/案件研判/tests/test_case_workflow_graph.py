import asyncio
from pathlib import Path

import pytest

pytest.importorskip("langgraph")
from langgraph.checkpoint.memory import InMemorySaver

from src.case_task_scheduler import CaseTaskScheduler, InvestigationTask, SQLiteRunStore, TaskSchedulerConfig
from src.case_workflow_graph import build_case_workflow_graph


def task(task_id, *, depends_on=()):
    return InvestigationTask(
        task_id=task_id,
        question=f"Review {task_id}",
        entity_ids=("person-1",),
        basis_fact_ids=("fact-1",),
        allowed_tools=("lookup",),
        depends_on=depends_on,
    )


def test_langgraph_plans_dispatches_centrally_verifies_and_runs_limited_recheck(tmp_path: Path):
    async def scenario():
        calls = []
        verification_calls = []

        async def worker(current_task, task_input, context):
            calls.append(current_task.task_id)
            if current_task.task_id == "initial":
                await context.llm_call(model_request, "worker evidence", max_output_tokens=5)
            return {"evidence": current_task.task_id}

        async def model_request(prompt):
            return {"response": prompt}

        async def planner(state, context):
            await context.llm_call(model_request, "initial plan", max_output_tokens=5)
            return [task("initial")]

        async def verifier(state, context):
            verification_calls.append(state["task_runs"])
            if len(verification_calls) == 1:
                return {"gaps": [{"fact_id": "gap-1"}], "accepted_events": []}
            return {"gaps": [], "accepted_events": ["event-1"]}

        async def recheck_planner(state, context):
            await context.llm_call(model_request, "recheck plan", max_output_tokens=5)
            return [task("recheck", depends_on=("initial",))]

        async def recheck_allowed(state):
            return True

        async def summarizer(state, context):
            return {"ready": True, "unresolved": state["verification"]["gaps"]}

        scheduler = CaseTaskScheduler(
            SQLiteRunStore(tmp_path / "runs.sqlite3"),
            worker,
            config=TaskSchedulerConfig(
                case_task_limit=2,
                server_task_cap=4,
                max_llm_calls=3,
                max_reserved_tokens=15,
            ),
        )
        graph = build_case_workflow_graph(
            planner=planner,
            scheduler=scheduler,
            verifier=verifier,
            recheck_planner=recheck_planner,
            recheck_allowed=recheck_allowed,
            summarizer=summarizer,
            checkpointer=InMemorySaver(),
            max_recheck_rounds=1,
        )
        result = await graph.ainvoke(
            {
                "case_id": "case-1",
                "run_id": "run-1",
                "case_facts": {"fact-1": {"value": "synthetic"}},
                "case_scope": {
                    "entity_ids": ["person-1"],
                    "fact_ids": ["fact-1"],
                    "allowed_tools": ["lookup"],
                },
            },
            {"configurable": {"thread_id": "case-1-run-1"}},
        )

        assert calls == ["initial", "recheck"]
        assert len(verification_calls) == 2
        assert result["task_run_status"] == "success"
        assert result["report"] == {"ready": True, "unresolved": []}
        assert scheduler.store.budget_usage("run-1") == {
            "llm_calls": 3,
            "tool_calls": 0,
            "reserved_tokens": 15,
        }

    asyncio.run(scenario())


def test_langgraph_workflow_requires_explicit_checkpointer(tmp_path: Path):
    async def worker(current_task, task_input, context):
        return {}

    scheduler = CaseTaskScheduler(SQLiteRunStore(tmp_path / "runs.sqlite3"), worker)
    with pytest.raises(ValueError, match="explicit LangGraph checkpointer"):
        build_case_workflow_graph(
            planner=lambda state, context: [],
            scheduler=scheduler,
            verifier=lambda state, context: {"gaps": []},
            recheck_planner=lambda state, context: [],
            recheck_allowed=lambda state: False,
            summarizer=lambda state, context: {},
            checkpointer=None,
        )


def test_structured_candidate_events_reach_central_verifier_and_report(tmp_path: Path):
    async def scenario():
        candidate = {
            "event_type": "synthetic-transfer",
            "subject_id": "person-1",
            "object_id": "account-1",
        }
        seen = []

        async def worker(current_task, task_input, context):
            return {"candidate_event": candidate, "evidence": "explicit synthetic result"}

        async def planner(state, context):
            return [task("candidate-task")]

        async def verifier(state, context):
            seen.append(state["candidate_events"])
            return {
                "gaps": [],
                "final_events": state["candidate_events"],
                "human_review": [],
            }

        async def recheck_planner(state, context):
            return []

        async def recheck_allowed(state):
            return False

        async def summarizer(state, context):
            return {
                "candidate_events": state["candidate_events"],
                "final_events": state["final_events"],
                "human_review": state["human_review"],
            }

        scheduler = CaseTaskScheduler(
            SQLiteRunStore(tmp_path / "runs.sqlite3"),
            worker,
            config=TaskSchedulerConfig(candidate_limit=1),
        )
        graph = build_case_workflow_graph(
            planner=planner,
            scheduler=scheduler,
            verifier=verifier,
            recheck_planner=recheck_planner,
            recheck_allowed=recheck_allowed,
            summarizer=summarizer,
            checkpointer=InMemorySaver(),
        )
        result = await graph.ainvoke(
            {
                "case_id": "case-candidate",
                "run_id": "run-candidate",
                "case_facts": {"fact-1": {"value": "synthetic"}},
                "case_scope": {
                    "entity_ids": ["person-1"],
                    "fact_ids": ["fact-1"],
                    "allowed_tools": ["lookup"],
                },
            },
            {"configurable": {"thread_id": "case-candidate-run-candidate"}},
        )

        assert seen == [[candidate]]
        assert result["report"] == {
            "candidate_events": [candidate],
            "final_events": [candidate],
            "human_review": [],
        }

    asyncio.run(scenario())


def test_candidate_event_limit_fails_closed(tmp_path: Path):
    async def scenario():
        async def worker(current_task, task_input, context):
            return {"candidate_events": [{"event": "one"}, {"event": "two"}]}

        async def planner(state, context):
            return [task("too-many-candidates")]

        scheduler = CaseTaskScheduler(
            SQLiteRunStore(tmp_path / "runs.sqlite3"),
            worker,
            config=TaskSchedulerConfig(candidate_limit=1),
        )
        graph = build_case_workflow_graph(
            planner=planner,
            scheduler=scheduler,
            verifier=lambda state, context: {"gaps": []},
            recheck_planner=lambda state, context: [],
            recheck_allowed=lambda state: False,
            summarizer=lambda state, context: {},
            checkpointer=InMemorySaver(),
        )

        with pytest.raises(ValueError, match="candidate event limit exceeded"):
            await graph.ainvoke(
                {
                    "case_id": "case-limit",
                    "run_id": "run-limit",
                    "case_facts": {"fact-1": {"value": "synthetic"}},
                    "case_scope": {
                        "entity_ids": ["person-1"],
                        "fact_ids": ["fact-1"],
                        "allowed_tools": ["lookup"],
                    },
                },
                {"configurable": {"thread_id": "case-limit-run-limit"}},
            )
        assert scheduler.store.get_run("run-limit")["status"] == "failed"

    asyncio.run(scenario())

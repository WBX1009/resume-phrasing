"""Small native graph invocation fixture; production uses CaseRunService."""
from langgraph.checkpoint.memory import InMemorySaver


async def run_tasks(executor, run_id, tasks, case_facts, *, scope):
    executor.prepare_run(run_id, scope)
    graph = executor.build_graph(checkpointer=InMemorySaver())
    return await graph.ainvoke(
        {
            "run_id": run_id,
            "tasks": [task.to_dict() for task in tasks],
            "case_facts": case_facts,
            "case_scope": scope.to_dict(),
            "outcomes": {},
        },
        {
            "configurable": {"thread_id": run_id},
            "max_concurrency": executor.config.case_task_limit,
            "recursion_limit": 4 * executor.config.max_total_tasks + 10,
        },
        durability="sync",
    )

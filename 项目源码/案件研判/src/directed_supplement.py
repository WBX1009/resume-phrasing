"""Translate individual evidence gaps to tasks without granting new scope."""
from .case_execution import InvestigationTask, PlanError
from .investigation_contracts import EvidenceGap, stable_id, validate_fact_references


def build_recheck_tasks(state, *, max_tasks=20):
    candidates = {c["event_id"]: c for c in state["candidate_events"]}
    tasks = {t["task_id"]: t for t in state["tasks"]}
    output = []
    for raw in state["verification"]["gaps"]:
        gap = EvidenceGap.model_validate(raw)
        candidate = candidates[gap.candidate_id]
        sources = [t for t in tasks.values() if t["goal_id"] == gap.goal_id]
        # This is provenance, not a dependency on a failed prerequisite.
        basis = list(dict.fromkeys(gap.basis_fact_ids + [fid
            for t in sources for fid in (state["task_outcomes"].get(t["task_id"], {}).get("result") or {}).get("fact_ids", [])]))
        if not set(basis) <= set(state["fact_ids"]):
            raise PlanError("recheck gap references an unknown fact")
        tools = sorted(set(tool for t in sources for tool in t["allowed_tools"]))
        output.append(InvestigationTask(task_id=stable_id("recheck", [gap.gap_id, state["recheck_round"]+1]),
            question=gap.question, candidate_id=gap.candidate_id, goal_id=gap.goal_id,
            entity_ids=tuple(candidate["subject_ids"]), time_range=tuple(candidate["time_range"]),
            basis_fact_ids=tuple(basis), allowed_tools=tuple(tools), gap_ids=(gap.gap_id,),
            source_task_ids=tuple(sorted(t["task_id"] for t in sources)), payload={"requirement_id": gap.requirement_id}))
        if len(output) >= max_tasks:
            break
    return output

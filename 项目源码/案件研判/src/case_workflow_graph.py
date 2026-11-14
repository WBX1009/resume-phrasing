"""The single end-to-end case graph, including dual recall and rechecks."""
from __future__ import annotations

import asyncio
from typing import Any, TypedDict, Annotated
from langchain_core.runnables import RunnableConfig
from langgraph.graph import START, END, StateGraph
from langgraph.types import Send, RetryPolicy

from .case_execution import PlanError, InvestigationTask, validate_task_plan, _invoke_tool
from .case_task_graph import scope_from_state
from .case_repository import load_snapshot
from .candidate_events import CandidateEventBuilder
from .directed_supplement import build_recheck_tasks
from .investigation_contracts import stable_id, CaseReport, Candidate
from .investigation_domain import prepare_facts, fact_context, rule_candidates, validate_plan, central_verify
from .case_worker_subgraph import retryable_error


class CaseWorkflowState(TypedDict, total=False):
    case_id: str
    run_id: str
    snapshot_ref: str
    case_scope: dict[str, Any]
    data_label: str
    coverage: dict
    fact_ids: list[str]
    rule_clues: list[dict]
    model_clues: list[dict]
    discovery_audit: dict
    candidate_events: list[dict]
    goals: list[dict]
    tasks: list[dict]
    task_outcomes: dict
    verification: dict
    recheck_round: int
    recheck_added: bool
    report: dict


def _merge_batches(left, right):
    merged = dict(left)
    for key, value in right.items():
        if key in merged and merged[key] != value:
            raise PlanError("conflicting semantic batch")
        merged[key] = value
    return merged


class DiscoveryState(TypedDict, total=False):
    run_id: str
    case_scope: dict
    fact_ids: list[str]
    batch_id: str
    batches: Annotated[dict, _merge_batches]


def build_case_workflow_graph(*, executor, reasoner, checkpointer):
    if checkpointer is None:
        raise ValueError("a durable checkpointer is required")
    store, policy_config = executor.store, executor.config
    retry = RetryPolicy(max_attempts=policy_config.max_attempts, retry_on=retryable_error, initial_interval=.05)

    def check(state):
        if store.is_cancel_requested(state["run_id"]):
            raise asyncio.CancelledError

    async def prepare(state):
        check(state)
        snapshot = load_snapshot(store, state)
        facts = await _invoke_tool(executor.resources.database, prepare_facts, lambda: check(state), snapshot, store, state["run_id"])
        store.record_event(state["run_id"], "prepare_facts", "succeeded", {"fact_count": len(facts)})
        return {"fact_ids": sorted(facts), "data_label": snapshot.data_label, "recheck_round": 0,
            "coverage": {"complete": all(m.complete and m.records for m in snapshot.models),
                         "models": [{"model_id": m.model_id, "record_count": len(m.records), "complete": m.complete} for m in snapshot.models]}}

    async def discover_rules(state):
        check(state)
        return {"rule_clues": rule_candidates(store.facts(state["run_id"], state["fact_ids"]))}

    async def semantic_batch(state):
        check(state)
        payload = {"scope": state["case_scope"], "facts": fact_context(store, state["run_id"], state["fact_ids"],
            limit=policy_config.max_context_facts, max_chars=policy_config.max_context_chars)}
        raw = await reasoner.decide("discover", payload, executor.workflow_context(state["run_id"]))
        if not isinstance(raw, dict) or not isinstance(raw.get("candidates"), list):
            raise PlanError("discovery must return a candidates array")
        if len(raw["candidates"]) > policy_config.max_total_tasks*10:
            raise PlanError("semantic discovery exceeds input safety bound")
        valid, rejected = [], []
        for clue in raw["candidates"]:
            try:
                proposed = Candidate.model_validate(clue)
                if not set(proposed.fact_ids) <= set(state["fact_ids"]):
                    raise PlanError("unknown batch fact")
            except (ValueError, TypeError):
                rejected.append({"source": "llm", "reason": "unknown_batch_fact"})
            else:
                valid.append(proposed.model_dump(mode="json"))
        return {"batches": {state["batch_id"]: {"candidates": valid, "rejected": rejected,
                "channel_status": raw.get("channel_status", "complete")}}}

    def split_batches(state):
        ids = state["fact_ids"]
        chunks = [ids[i:i+policy_config.max_context_facts] for i in range(0, len(ids), policy_config.max_context_facts)]
        return [Send("recall", {"run_id": state["run_id"], "case_scope": state["case_scope"],
            "fact_ids": chunk, "batch_id": str(i)}) for i, chunk in enumerate(chunks)] or END

    semantic = StateGraph(DiscoveryState)
    semantic.add_node("split", lambda state: {})
    semantic.add_node("recall", semantic_batch, retry_policy=retry, timeout=policy_config.task_timeout)
    semantic.add_edge(START, "split")
    semantic.add_conditional_edges("split", split_batches, ["recall", END])
    semantic.add_edge("recall", END)
    semantic_graph = semantic.compile()

    async def discover_semantics(state, config: RunnableConfig):
        result = await semantic_graph.ainvoke({"run_id": state["run_id"], "case_scope": state["case_scope"],
            "fact_ids": state["fact_ids"], "batches": {}}, config)
        batches = result.get("batches", {})
        return {"model_clues": [c for b in batches.values() for c in b["candidates"]],
            "discovery_audit": {"semantic_batches": len(batches),
                "rejected": [r for b in batches.values() for r in b["rejected"]],
                "semantic_status": "complete" if all(b["channel_status"] == "complete" for b in batches.values()) else "unavailable_offline"}}

    async def select(state):
        check(state)
        selected = CandidateEventBuilder(policy_config.candidate_limit).build(state["rule_clues"], state["model_clues"],
            scope=scope_from_state(state), facts=store.facts(state["run_id"], state["fact_ids"]))
        audit = {**state["discovery_audit"], **{k: v for k,v in selected.items() if k != "events"}}
        audit["rejected"] = state["discovery_audit"].get("rejected", []) + selected["rejected"]
        store.record_event(state["run_id"], "select_candidates", "succeeded", audit)
        return {"candidate_events": selected["events"], "discovery_audit": audit,
                "rule_clues": [], "model_clues": []}

    async def plan(state):
        check(state)
        if not state["candidate_events"]:
            return {"tasks": [], "goals": [], "task_outcomes": {}}
        ids = list(dict.fromkeys(fid for c in state["candidate_events"] for fid in c["fact_ids"]))
        payload = {"candidates": state["candidate_events"], "scope": state["case_scope"],
            "facts": fact_context(store, state["run_id"], ids[:policy_config.max_context_facts],
                limit=policy_config.max_context_facts, max_chars=policy_config.max_context_chars),
            "fact_summary_omitted_count": max(0,len(ids)-policy_config.max_context_facts)}
        raw = await reasoner.decide("plan", payload, executor.workflow_context(state["run_id"]))
        validated, tasks = validate_plan(raw, state, store, policy_config, scope_from_state(state))
        store.record_event(state["run_id"], "plan", "succeeded", {"task_count": len(tasks)})
        return {"tasks": [t.to_dict() for t in tasks], "goals": [g.model_dump(mode="json") for g in validated.goals], "task_outcomes": {}}

    async def dispatch(state, config: RunnableConfig):
        check(state)
        output = await executor.graph.ainvoke({"run_id": state["run_id"], "case_scope": state["case_scope"],
            "tasks": state["tasks"], "goals": state["goals"], "fact_ids": state["fact_ids"],
            "outcomes": state.get("task_outcomes", {})}, config)
        produced = [fid for o in output["outcomes"].values() for fid in (o.get("result") or {}).get("fact_ids", [])]
        return {"task_outcomes": output["outcomes"], "fact_ids": sorted(set(state["fact_ids"]) | set(produced))}

    async def verify(state):
        check(state)
        result = central_verify(state, store)
        store.record_event(state["run_id"], "central_verify", "succeeded", {"gap_count": len(result["gaps"]),
            "verified_count": len(result["final_events"])})
        return {"verification": result}

    async def rechecks(state):
        check(state)
        slots = policy_config.max_total_tasks-len(state["tasks"])
        added = build_recheck_tasks(state, max_tasks=slots) if slots > 0 else []
        tasks = [InvestigationTask.from_dict(t) for t in state["tasks"]] + added
        validate_task_plan(tasks, max_total_tasks=policy_config.max_total_tasks, scope=scope_from_state(state), known_fact_ids=set(state["fact_ids"]))
        return {"tasks": [t.to_dict() for t in tasks], "recheck_added": bool(added), "recheck_round": state["recheck_round"]+1}

    async def summarize(state):
        check(state)
        partial = bool(not state["coverage"]["complete"] or state["verification"]["gaps"] or any(o["status"] != "succeeded" for o in state["task_outcomes"].values())
            or state["discovery_audit"].get("semantic_status") != "complete" or state["discovery_audit"].get("omitted_count")
            or state["discovery_audit"].get("rejected"))
        report = {"case_id": state["case_id"], "run_id": state["run_id"], "status": "partial" if partial else "success",
            "data_label": state["data_label"], "runtime": "langgraph-native-v3", "snapshot_ref": state["snapshot_ref"],
            "candidate_events": state["candidate_events"], "goals": state["goals"], "tasks": state["tasks"],
            "task_outcomes": state["task_outcomes"], "verification": state["verification"], "fact_ids": state["fact_ids"],
            "discovery": state["discovery_audit"], "recheck_rounds": state["recheck_round"],
            "coverage": state["coverage"],
            "budget": store.budget_usage(state["run_id"]), "evidence_boundary": "辅助研判推断，需人工复核；测试不证明生产准确率"}
        return {"report": CaseReport.model_validate(report).model_dump(mode="json")}

    graph = StateGraph(CaseWorkflowState)
    for name, callback in (("prepare_facts", prepare), ("discover_rules", discover_rules),
            ("discover_semantics", discover_semantics), ("select_candidates", select), ("plan", plan),
            ("dispatch", dispatch), ("central_verify", verify), ("plan_rechecks", rechecks), ("summarize", summarize)):
        kwargs = {"retry_policy": retry, "timeout": policy_config.task_timeout} if name == "plan" else {}
        graph.add_node(name, callback, **kwargs)
    graph.add_edge(START, "prepare_facts")
    graph.add_edge("prepare_facts", "discover_rules")
    graph.add_edge("prepare_facts", "discover_semantics")
    graph.add_edge(["discover_rules", "discover_semantics"], "select_candidates")
    graph.add_edge("select_candidates", "plan")
    graph.add_edge("plan", "dispatch")
    graph.add_edge("dispatch", "central_verify")
    graph.add_conditional_edges("central_verify", lambda s: "plan_rechecks" if s["verification"]["gaps"]
        and s["recheck_round"] < policy_config.max_recheck_rounds else "summarize", ["plan_rechecks", "summarize"])
    graph.add_conditional_edges("plan_rechecks", lambda s: "dispatch" if s["recheck_added"] else "summarize", ["dispatch", "summarize"])
    graph.add_edge("summarize", END)
    return graph.compile(checkpointer=checkpointer)

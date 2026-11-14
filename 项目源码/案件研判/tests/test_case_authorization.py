import asyncio
from copy import deepcopy
import json

import pytest
from pydantic import ValidationError

from investigation_support import service, snapshot, ScriptedReasoner, wait_done
from src.case_execution import TaskExecutionContext, InvestigationTask, PlanError, CaseExecutionConfig, validate_task_plan
from src.investigation_contracts import authorize_bounds, validate_fact_references, Assessment
from src.investigation_domain import prepare_facts
from src.investigation_tools import InvestigationTools
from src.candidate_events import CandidateEventBuilder


def admitted(active):
    data = active.repository.load("synthetic")
    state = {"case_id": data.case_id, "run_id": "r", "case_scope": data.scope().to_dict(), "snapshot_ref": data.reference()}
    active.store.admit(state, active._config, data.scope(), snapshot=data.model_dump(mode="json"))
    facts = prepare_facts(data, active.store, "r")
    task = InvestigationTask("T", "bounded query", entity_ids=("p",), time_range=data.time_range,
        basis_fact_ids=(next(iter(facts)),), allowed_tools=tuple(data.allowed_tools))
    context = TaskExecutionContext("r", task, "r:T", active.store, active._config, active._resources, data.scope(), active.tools)
    return data, task, context


@pytest.mark.parametrize("mutation", [
    {"entity_ids": ["outside"]}, {"time_range": ["2030-01-01", "2030-01-02"]},
    {"time_range": None}, {"params": {"model_id": 999}},
    {"params": {"model_id": 1, "entity": "outside"}},
    {"params": {"model_id": True}}, {"params": {"model_id": 1, "limit": 100_000}},
    {"tool_name": "unregistered"},
])
def test_tool_parameter_authorization_precedes_budget_and_io(tmp_path, mutation):
    active = service(tmp_path)
    try:
        data, task, context = admitted(active)
        request = {"tool_name": "query_records", "entity_ids": ["p"], "time_range": data.time_range, "params": {"model_id": 1}}
        request.update(mutation)
        with pytest.raises((PlanError, ValidationError)):
            asyncio.run(context.execute_tool(request))
        assert active.store.budget_usage("r")["tool_calls"] == 0
    finally:
        active.close()


@pytest.mark.parametrize("name", ["entity_model_slice", "same_amount_window_probe"])
def test_existing_business_tools_use_the_same_scope_boundary(tmp_path, name):
    active = service(tmp_path)
    try:
        data, task, context = admitted(active)
        request = {"tool_name": name, "entity_ids": ["p"], "time_range": data.time_range,
                   "params": {"model_id": 1, "entity_name": "outside"}}
        with pytest.raises(PlanError):
            asyncio.run(context.execute_tool(request))
        request["params"]["entity_name"] = "p"
        result = asyncio.run(context.execute_tool(request))
        fact = active.store.facts("r", result["fact_ids"])[result["fact_ids"][0]]
        assert fact["provenance"]["tool"] == name
        assert active.store.budget_usage("r")["tool_calls"] == 1
    finally:
        active.close()


def test_candidate_selection_rejects_unknown_facts_scope_and_identity_collisions(tmp_path):
    active = service(tmp_path)
    try:
        data, task, context = admitted(active)
        facts = active.store.facts("r")
        clue = {"event_id": "collision", "event_type": "A", "subject_ids": ["p"],
            "fact_ids": [next(iter(facts))], "time_range": data.time_range, "rationale": "candidate", "priority": 1}
        events = [clue, {**clue, "event_type": "B"}, {**clue, "subject_ids": ["outside"]},
                  {**clue, "fact_ids": ["invented"]}, {**clue, "fact_ids": []}]
        result = CandidateEventBuilder(1).build(events, [{**clue, "event_type": "C", "priority": 10}], scope=data.scope(), facts=facts)
        assert result["valid_count"] == 3 and len(result["rejected"]) == 3
        assert len(result["events"]) == 1 and result["omitted_count"] == 2
        result = CandidateEventBuilder(3).build(events[:2], [clue], scope=data.scope(), facts=facts)
        assert len(result["events"]) == 2
        assert len({c["event_id"] for c in result["events"]}) == 2
        assert next(c for c in result["events"] if c["event_type"] == "A")["source"] == ["llm", "rule"]
    finally:
        active.close()


def test_evidence_covers_subjects_without_leaking_other_subjects():
    facts = {"fp": {"entity_ids": ["p"]}, "fq": {"entity_ids": ["q"]}, "fpq": {"entity_ids": ["p", "q"]}}
    validate_fact_references(["fp", "fq"], facts, ["p", "q"])
    with pytest.raises(PlanError):
        validate_fact_references(["fpq"], facts, ["p"])
    with pytest.raises(PlanError):
        validate_fact_references(["fp"], facts, ["p", "q"])


@pytest.mark.parametrize("name,params", [
    ("relation_neighbor_subgraph", {"model_id": 86, "entity_name": "p"}),
    ("fund_flow_path_probe", {"model_id": 86, "source": "p", "target": "q"}),
    ("transfer_structure_analyzer", {"model_id": 86, "entity_name": "p"}),
    ("counterparty_overlap_probe", {"entity_name": "p", "model_ids": [86]}),
])
def test_original_relation_tools_run_through_authorized_registry(tmp_path, name, params):
    data = snapshot()
    data["entity_ids"] = ["p", "q"]
    data["allowed_tools"] = [name]
    model = data["models"][0]
    model["model_id"] = 86
    model["strategy"].update({"model_kind": "relation", "allowed_tools": [name],
        "relation": {"source": "起点", "target": "终点", "value": "交易金额"}})
    for row in model["records"]:
        row["entity_ids"] = ["p", "q"]
        row["data"].update({"起点": "p", "终点": "q"})
    data["models"] = [model]
    active = service(tmp_path, data=data)
    try:
        from src.case_execution import TaskExecutionContext, InvestigationTask
        approved = active.repository.load("synthetic")
        state = {"case_id": "synthetic", "run_id": "r", "snapshot_ref": approved.reference(), "case_scope": approved.scope().to_dict()}
        active.store.admit(state, active._config, approved.scope(), snapshot=approved.model_dump(mode="json"))
        facts = prepare_facts(approved, active.store, "r")
        task = InvestigationTask("T", "relation check", entity_ids=("p", "q"), time_range=approved.time_range,
            basis_fact_ids=(next(iter(facts)),), allowed_tools=(name,))
        context = TaskExecutionContext("r", task, "r:T", active.store, active._config, active._resources, approved.scope(), active.tools)
        output = asyncio.run(context.execute_tool({"tool_name": name, "entity_ids": ["p", "q"],
            "time_range": approved.time_range, "params": params}))
        assert output["fact_ids"] and active.store.budget_usage("r")["tool_calls"] == 1
    finally:
        active.close()


@pytest.mark.parametrize("fault", ["unknown_fact", "wrong_goal", "other_gap", "incomplete"])
def test_invalid_worker_judgment_never_enters_final_events(tmp_path, fault):
    class InvalidJudge(ScriptedReasoner):
        def response(self, stage, p):
            result = super().response(stage, p)
            if stage == "worker_judge" and p["step"] >= 2:
                if fault == "unknown_fact":
                    result["assessments"][0]["fact_ids"] = ["invented"]
                elif fault == "wrong_goal":
                    result["assessments"][0]["goal_id"] = "foreign"
                elif fault == "other_gap":
                    result["assessments"][0]["requirement_id"] = "not-planned"
            return result
    data = snapshot()
    if fault == "incomplete":
        for model in data["models"]:
            model["complete"] = False
    active = service(tmp_path, reasoner=InvalidJudge(), data=data, config=CaseExecutionConfig(max_recheck_rounds=0))
    try:
        active.submit("synthetic", run_id="r")
        assert wait_done(active)["status"] == "partial"
        report = active.report("r")
        assert report["verification"]["final_events"] == []
        assert report["task_outcomes"]["initial-0"]["status"] == "failed"
    finally:
        active.close()

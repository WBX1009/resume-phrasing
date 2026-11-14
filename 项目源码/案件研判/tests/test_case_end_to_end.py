import asyncio
from copy import deepcopy
import time

import pytest

from investigation_support import snapshot, ScriptedReasoner, service, wait_done
from src.case_execution import CaseExecutionConfig, PlanError
from src.case_repository import MappingCaseRepository
from src.investigation_domain import prepare_facts
from src.investigation_reasoner import OfflineReasoner


def test_full_chain_discovers_before_plan_without_preprovided_tasks(tmp_path):
    reasoner = ScriptedReasoner(dependent=True)
    active = service(tmp_path, reasoner=reasoner)
    try:
        active.submit("synthetic", run_id="r")
        assert wait_done(active)["status"] == "success"
        report = active.report("r")
        assert [c["source"] for c in report["candidate_events"]] == [["llm"]]
        assert [s for s,t,p in reasoner.calls[:2]] == ["discover", "plan"]
        assert len(report["goals"]) == 1
        assert report["verification"]["final_events"][0]["evidence_level"] == "inference"
        assert len(report["tasks"]) == 2
        first = report["task_outcomes"]["initial-0"]["result"]
        assert first["steps"] == 2
        dependent = next(p for s,t,p in reasoner.calls if s == "worker_plan" and t == "dependent-0")
        assert set(first["fact_ids"]) <= {f["fact_id"] for f in dependent["facts"]}
        facts = active.store.facts("r", report["fact_ids"])
        assert all(f["provenance"]["source"] in {"deterministic_operator", "registered_tool"} for f in facts.values())
        assert not {"case_facts", "task_input"} & set(dependent)
    finally:
        active.close()


def test_two_channels_share_budget_and_rule_only_mode_never_promotes_candidates(tmp_path):
    active = service(tmp_path, data=snapshot(rules=True), reasoner=OfflineReasoner(),
        config=CaseExecutionConfig(max_recheck_rounds=0))
    try:
        active.submit("synthetic", run_id="r")
        assert wait_done(active)["status"] == "partial"
        report = active.report("r")
        assert report["candidate_events"]
        assert all(c["source"] == ["rule"] for c in report["candidate_events"])
        assert report["verification"]["final_events"] == []
        assert report["verification"]["gaps"]
        assert report["discovery"]["semantic_status"] == "unavailable_offline"
    finally:
        active.close()


def test_recheck_references_new_evidence_and_closes_only_its_own_gap(tmp_path):
    reasoner = ScriptedReasoner(rechecks=True)
    active = service(tmp_path, reasoner=reasoner, config=CaseExecutionConfig(max_recheck_rounds=2))
    try:
        active.submit("synthetic", run_id="r")
        assert wait_done(active)["status"] == "partial"
        report = active.report("r")
        assert report["recheck_rounds"] == 2
        assert [g["requirement_id"] for g in report["verification"]["gaps"]] == ["relation"]
        assert report["verification"]["final_events"] == []
        initial = report["task_outcomes"]["initial-0"]["result"]
        rechecks = [t for t in report["tasks"] if t["gap_ids"]]
        assert len(rechecks) == 3
        assert len({t["gap_ids"][0] for t in rechecks}) == 2
        assert all(set(initial["fact_ids"]) <= set(t["basis_fact_ids"]) for t in rechecks)
        assert all(t["source_task_ids"] and not t["depends_on"] for t in rechecks)
        assert all(set(f["fact_id"] for f in p["facts"]) >= set(initial["fact_ids"])
            for s,t,p in reasoner.calls if s == "worker_plan" and t.startswith("recheck-"))
    finally:
        active.close()


def test_native_judge_retry_does_not_repeat_successful_tool_node(tmp_path):
    reasoner = ScriptedReasoner(fail_stage="worker_judge")
    active = service(tmp_path, reasoner=reasoner)
    try:
        active.submit("synthetic", run_id="r")
        assert wait_done(active)["status"] == "success"
        assert active.status("r")["budget"] == {"llm_calls": 7, "tool_calls": 2, "reserved_tokens": 70}
        assert sum(s == "worker_plan" for s,t,p in reasoner.calls) == 2
        assert sum(s == "worker_judge" for s,t,p in reasoner.calls) == 3
    finally:
        active.close()


def test_budget_failure_blocks_dependents_without_becoming_verified(tmp_path):
    active = service(tmp_path, reasoner=ScriptedReasoner(dependent=True),
        config=CaseExecutionConfig(max_llm_calls=2, max_recheck_rounds=0))
    try:
        active.submit("synthetic", run_id="r")
        assert wait_done(active)["status"] == "partial"
        report = active.report("r")
        assert report["task_outcomes"]["initial-0"]["error"] == "BudgetLimitExceeded"
        assert report["task_outcomes"]["dependent-0"]["status"] == "blocked"
        assert report["verification"]["final_events"] == []
        assert report["budget"]["llm_calls"] == 2 and report["budget"]["tool_calls"] == 0
    finally:
        active.close()


def test_missing_or_fabricated_assessment_cannot_confirm_candidate(tmp_path):
    class MissingJudge(ScriptedReasoner):
        def response(self, stage, payload):
            result = super().response(stage, payload)
            if stage == "worker_judge":
                result["assessments"] = []
            return result
    active = service(tmp_path, reasoner=MissingJudge(), config=CaseExecutionConfig(max_recheck_rounds=0))
    try:
        active.submit("synthetic", run_id="r")
        assert wait_done(active)["status"] == "partial"
        assert len(active.report("r")["verification"]["gaps"]) == 2
        assert active.report("r")["verification"]["final_events"] == []
    finally:
        active.close()


def test_completed_future_is_removed_and_sqlite_connections_close(tmp_path):
    active = service(tmp_path)
    active.submit("synthetic", run_id="r")
    assert wait_done(active)["status"] == "success"
    deadline = time.monotonic()+2
    while active._futures and time.monotonic() < deadline:
        time.sleep(.01)
    assert not active._futures
    active.close()
    with active.store._connect() as db:
        assert db.execute("SELECT 1").fetchone()[0] == 1
    import sqlite3
    with pytest.raises(sqlite3.ProgrammingError):
        db.execute("SELECT 1")


def test_repository_never_substitutes_another_case_or_accepts_foreign_rows(tmp_path):
    data = snapshot()
    data["models"][0]["records"][0]["entity_ids"] = ["outside"]
    with pytest.raises(ValueError):
        MappingCaseRepository({"synthetic": data})
    data = snapshot()
    data["models"][0]["records"][0]["occurred_at"] = "2030-01-01"
    with pytest.raises(ValueError):
        MappingCaseRepository({"synthetic": data})
    active = service(tmp_path)
    try:
        with pytest.raises(KeyError):
            active.submit("missing")
    finally:
        active.close()

"""Synthetic fixtures exercise behavior, not production accuracy."""
import asyncio
from copy import deepcopy
import time

from src.case_execution import SQLiteRunStore, CaseExecutionConfig
from src.case_repository import MappingCaseRepository
from src.case_run_service import CaseRunService, TERMINAL


def snapshot(*, rules=False):
    strategy = {"model_kind": "entity", "person_fields": ["姓名"], "counterparty_fields": ["交易对手"],
        "primary_amount_field": "交易金额", "time_fields": ["交易时间"],
        "anomaly_rules": ["same_day_same_amount"], "allowed_tools": ["entity_model_slice", "same_amount_window_probe"]}
    models = []
    for mid in (1, 2):
        records = [{"row_id": f"{mid}-{i}", "entity_ids": ["p"], "occurred_at": f"2025-01-01T10:0{i}:00",
                    "data": {"姓名": "p", "交易对手": f"peer-{i}", "交易金额": 100 if rules else 100+i,
                        "交易时间": f"2025-01-01T10:0{i}:00"}} for i in range(2)]
        models.append({"model_id": mid, "name": f"synthetic-{mid}", "records": records, "strategy": strategy})
    return {"case_id": "synthetic", "entity_ids": ["p"], "time_range": ["2025-01-01T00:00:00", "2025-01-02T00:00:00"],
        "allowed_tools": ["query_records", "entity_model_slice", "same_amount_window_probe"], "models": models,
        "data_label": "synthetic_test_only"}


class ScriptedReasoner:
    def __init__(self, *, rechecks=False, dependent=False, fail_stage=None, fail_count=1, observer=None):
        self.calls = []
        self.rechecks, self.dependent = rechecks, dependent
        self.fail_stage, self.fail_count, self.observer = fail_stage, fail_count, observer

    async def decide(self, stage, payload, context):
        async def request():
            tid = payload.get("task", {}).get("task_id", "main")
            self.calls.append((stage, tid, deepcopy(payload)))
            if self.observer:
                await self.observer(stage, payload, context)
            if stage == self.fail_stage and sum(s == stage for s,t,p in self.calls) <= self.fail_count:
                raise ConnectionError("synthetic transient provider failure")
            return self.response(stage, payload)
        return await context.llm_call(request, max_output_tokens=10)

    def response(self, stage, p):
        if stage == "discover":
            ids = [f["fact_id"] for f in p["facts"][:2]]
            return {"candidates": [{"event_type": "semantic_cross_model", "subject_ids": ["p"],
                "fact_ids": ids, "time_range": p["scope"]["time_range"], "rationale": "synthetic semantic lead", "priority": 9}]}
        if stage == "plan":
            goals, tasks = [], []
            for n,c in enumerate(p["candidates"]):
                gid = f"goal-{n}"
                goals.append({"goal_id": gid, "candidate_id": c["event_id"], "requirements": [
                    {"requirement_id": "flow", "question": "核对资金证据"}, {"requirement_id": "relation", "question": "核对关系证据"}]})
                spec = {"task_id": f"initial-{n}", "question": "核查两个来源", "candidate_id": c["event_id"], "goal_id": gid,
                    "entity_ids": c["subject_ids"], "time_range": c["time_range"], "basis_fact_ids": c["fact_ids"],
                    "allowed_tools": p["scope"]["allowed_tools"]}
                tasks.append(spec)
                if self.dependent:
                    tasks.append({**spec, "task_id": f"dependent-{n}", "depends_on": [spec["task_id"]]})
            return {"plan_version": 1, "goals": goals, "tasks": tasks}
        if stage == "worker_plan":
            t = p["task"]
            return {"action": "tool", "reason": "查询可核验来源", "request": {"tool_name": "query_records",
                "entity_ids": t["entity_ids"], "time_range": t["time_range"],
                "params": {"model_id": 1 if p["step"] == 0 else 2, "limit": 100}}}
        if stage == "worker_judge":
            recheck = bool(p["task"].get("gap_ids"))
            done = recheck or p["step"] >= 2
            return {"done": done, "reason": "synthetic evidence assessment", "assessments": [
                {"goal_id": p["goal"]["goal_id"], "requirement_id": r["requirement_id"],
                 "status": ("supported" if (not self.rechecks or (recheck and r["requirement_id"] == "flow")) and done else "unresolved"),
                 "fact_ids": p["observed_fact_ids"], "rationale": "synthetic judgment bound to observed facts"}
                for r in p["goal"]["requirements"]]}
        raise AssertionError(stage)


def service(root, *, reasoner=None, data=None, config=None, **kwargs):
    return CaseRunService(SQLiteRunStore(root / "runs.db"), repository=MappingCaseRepository({"synthetic": data or snapshot()}),
        reasoner=reasoner or ScriptedReasoner(), config=config or CaseExecutionConfig(), **kwargs)


def wait_done(active, run_id="r", timeout=20):
    deadline = time.monotonic()+timeout
    while time.monotonic() < deadline:
        record = active.status(run_id)
        if record and record["status"] in TERMINAL:
            return record
        time.sleep(.02)
    raise AssertionError("case did not reach a terminal state")

import asyncio
from copy import deepcopy
from dataclasses import replace
from io import BytesIO
from types import SimpleNamespace

from docx import Document
import pytest

from investigation_support import service, ScriptedReasoner, wait_done, snapshot
from test_case_authorization import admitted
from src.case_execution import CaseExecutionConfig, PlanError, BudgetLimitExceeded, validate_task_plan
from src.case_task_graph import merge_outcomes
from src.investigation_reasoner import OpenAIReasoner
from src.case_report import render_report
from src.row_source import RowQuery, _build_db_row_query


def test_business_plan_validates_dependencies_cycles_scope_and_new_facts(tmp_path):
    active = service(tmp_path)
    try:
        data, t, context = admitted(active)
        known = set(active.store.facts("r"))
        validate_task_plan([t], max_total_tasks=3, scope=data.scope(), known_fact_ids=known)
        for invalid in ([replace(t, depends_on=("missing",))],
                        [replace(t, depends_on=("T",))],
                        [replace(t, depends_on=("B",)), replace(t, task_id="B", depends_on=("T",))],
                        [replace(t, entity_ids=("outside",))], [replace(t, basis_fact_ids=("missing",))]):
            with pytest.raises(PlanError):
                validate_task_plan(invalid, max_total_tasks=3, scope=data.scope(), known_fact_ids=known)
        async def query():
            return await context.execute_tool({"tool_name": "query_records", "entity_ids": ["p"], "time_range": data.time_range, "params": {"model_id": 1}})
        result = asyncio.run(query())
        validate_task_plan([replace(t, basis_fact_ids=tuple(result["fact_ids"]))], max_total_tasks=3,
            scope=data.scope(), known_fact_ids=known | set(result["fact_ids"]))
    finally:
        active.close()


def test_terminal_reducer_rejects_conflicts_and_allows_idempotent_replay():
    result = {"T": {"status": "succeeded", "result": {"fact_ids": ["f"]}}}
    assert merge_outcomes(result, result) == result
    with pytest.raises(PlanError):
        merge_outcomes(result, {"T": {"status": "failed"}})


def test_real_model_adapter_disables_sdk_retries_and_reserves_input_plus_output(tmp_path):
    active = service(tmp_path)
    reasoner = OpenAIReasoner(model="synthetic-compatible-model", api_key="synthetic-not-a-real-key", max_output_tokens=10)
    requests = []
    async def request(**kwargs):
        requests.append(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content='{"candidates":[]}'))])
    reasoner.client.chat.completions.create = request
    try:
        data,t,context = admitted(active)
        async def scenario():
            assert await reasoner.decide("discover", {"facts": []}, context) == {"candidates": []}
            await reasoner.aclose()
        asyncio.run(scenario())
        assert reasoner.client.max_retries == 0 and len(requests) == 1
        assert requests[0]["max_tokens"] == 10
        assert active.store.budget_usage("r")["llm_calls"] == 1
        assert active.store.budget_usage("r")["reserved_tokens"] > 10
    finally:
        active.close()


def test_atomic_budget_does_not_overshoot_under_parallel_requests(tmp_path):
    active = service(tmp_path, config=CaseExecutionConfig(max_llm_calls=3))
    try:
        data,t,context = admitted(active)
        async def scenario():
            async def provider():
                await asyncio.sleep(.005)
                return {}
            results = await asyncio.gather(*[context.llm_call(provider, max_output_tokens=10) for _ in range(10)], return_exceptions=True)
            assert sum(isinstance(r, BudgetLimitExceeded) for r in results) == 7
        asyncio.run(scenario())
        assert active.store.budget_usage("r") == {"llm_calls": 3, "tool_calls": 0, "reserved_tokens": 30}
    finally:
        active.close()


def test_database_query_keeps_field_names_values_and_limits_parameterized():
    injection = "x' OR TRUE --"
    sql, params = _build_db_row_query(1, 2, RowQuery(any_field_equals=((injection, injection),), limit=5))
    assert injection not in sql and params == [1, 2, injection, injection, 5]
    with pytest.raises(ValueError):
        _build_db_row_query(1, 2, RowQuery(limit=0))


def test_report_formats_share_one_evidence_bound_projection(tmp_path):
    active = service(tmp_path)
    try:
        active.submit("synthetic", run_id="r")
        assert wait_done(active)["status"] == "success"
        report = active.report("r")
        for fmt in ("json", "md", "html", "docx"):
            content, _ = render_report(report, fmt)
            text = "\n".join(p.text for p in Document(BytesIO(content)).paragraphs) if fmt == "docx" else content.decode()
            assert report["fact_ids"][0] in text
        malicious = deepcopy(report)
        malicious["candidate_events"][0]["rationale"] = "<script>alert(1)</script>"
        html, _ = render_report(malicious, "html")
        assert b"<script>" not in html
    finally:
        active.close()


def test_publication_honors_cancellation_atomically(tmp_path):
    active = service(tmp_path)
    try:
        admitted(active)
        active.cancel("r")
        with pytest.raises(asyncio.CancelledError):
            active.store.publish_report("r", {}, "success")
        assert active.report("r") is None
    finally:
        active.close()

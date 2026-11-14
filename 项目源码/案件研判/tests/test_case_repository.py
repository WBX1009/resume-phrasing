from copy import deepcopy
import json

import pytest

from investigation_support import snapshot, service, ScriptedReasoner, wait_done
from test_case_authorization import admitted
from src.case_execution import CaseExecutionConfig, PlanError
from src.case_repository import RowSourceCaseRepository, FileCaseRepository
from src.investigation_domain import fact_context


def test_database_repository_uses_server_policy_bounded_queries_and_no_case_substitution():
    data = snapshot()
    data["case_id"] = "13"
    data["models"] = data["models"][:1]
    source_rows = [(r["row_id"], r["data"]) for r in data["models"][0]["records"]]
    data["models"][0]["records"] = []
    class Source:
        source_name = "gaussdb"
        def iter_model_rows_query(self, case_id, model_id, query):
            assert case_id == 13 and model_id == 1
            assert query.any_field_equals == (("姓名", "p"),)
            assert query.limit == 3
            yield from source_rows
            yield "bad", {"姓名": "p", "交易时间": "invalid"}
    repository = RowSourceCaseRepository(Source(), {"13": data}, max_rows_per_model=2)
    result = repository.load("13")
    assert len(result.models[0].records) == 2 and not result.models[0].complete
    with pytest.raises(KeyError):
        repository.load("missing")


def test_file_repository_rejects_path_traversal_and_mismatched_case(tmp_path):
    repository = FileCaseRepository(tmp_path)
    with pytest.raises(PlanError):
        repository.load("../outside")
    (tmp_path/"wrong.json").write_text(json.dumps(snapshot()), encoding="utf-8")
    with pytest.raises(PlanError):
        repository.load("wrong")


def test_context_preserves_references_and_marks_omitted_values_within_character_budget(tmp_path):
    active = service(tmp_path)
    try:
        data,t,context = admitted(active)
        ids = list(active.store.facts("r"))[:4]
        for fid in ids:
            # Use actual immutable facts; force a small context budget.
            assert active.store.facts("r",[fid])
        values = fact_context(active.store,"r",ids,limit=10,max_chars=1800)
        assert [v["fact_id"] for v in values] == ids
        assert len(json.dumps(values,ensure_ascii=False)) <= 1800
        assert any(v["content_omitted"] for v in values)
        with pytest.raises(PlanError,match="metadata"):
            fact_context(active.store,"r",ids,limit=10,max_chars=10)
    finally:
        active.close()


def test_completed_tool_evidence_survives_failed_judgment_for_recheck(tmp_path):
    class JudgeFailsInitially(ScriptedReasoner):
        def response(self,stage,p):
            if stage == "worker_judge" and not p["task"].get("gap_ids"):
                from src.case_execution import TaskExecutionError
                raise TaskExecutionError("synthetic model cannot finish this task")
            return super().response(stage,p)
    active = service(tmp_path, reasoner=JudgeFailsInitially(), config=CaseExecutionConfig(max_recheck_rounds=1))
    try:
        active.submit("synthetic",run_id="r")
        assert wait_done(active)["status"] == "partial"
        report = active.report("r")
        initial = report["task_outcomes"]["initial-0"]
        assert initial["status"] == "failed" and initial["result"]["fact_ids"]
        observed = set(initial["result"]["fact_ids"])
        assert observed <= set(report["fact_ids"])
        assert all(observed <= set(t["basis_fact_ids"]) for t in report["tasks"] if t["gap_ids"])
    finally:
        active.close()


@pytest.mark.parametrize("setup_error", [False, True])
def test_database_connection_is_closed_on_setup_or_body_failure(monkeypatch, setup_error):
    from types import SimpleNamespace
    from src import data_loader
    class Connection:
        closed = False
        def set_client_encoding(self, value):
            if setup_error:
                raise RuntimeError("synthetic setup failure")
        def set_session(self, **kwargs):
            assert kwargs == {"readonly": True, "autocommit": True}
        def close(self):
            self.closed = True
    connection = Connection()
    monkeypatch.setattr(data_loader.psycopg2, "connect", lambda **kwargs: connection)
    db = SimpleNamespace(is_complete=True, host="synthetic.invalid", port=5432, user="synthetic",
        password="synthetic-not-a-secret", database="synthetic", ssl_mode="require")
    with pytest.raises(RuntimeError):
        with data_loader._connect(SimpleNamespace(db=db)):
            raise RuntimeError("synthetic body failure")
    assert connection.closed

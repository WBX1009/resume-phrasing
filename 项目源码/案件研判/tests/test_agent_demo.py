from __future__ import annotations

import unittest
from datetime import datetime
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import patch

from src.agent_loop import (
    _call_key,
    _make_stop_step,
    _parse_initial_hypotheses,
    _validate_agent_request,
)
from src.execution_policy import FallbackPolicy, normalize_fallback_policy
from src.final_agent_report import _allowed_numeric_tokens, _localize_internal_terms, _validate_assertions
from src.judge import (
    _deterministic_judge,
    _is_clear_chinese_assessment,
    _judge_single_hypothesis,
    _localize_internal_terms as _localize_judge_terms,
)
from src.row_source import RowQuery, RowSource, _build_db_row_query, _matches_exact_filters, _matches_row_query
from src.stable_ids import stable_digest
from src.supplement_tools import ToolExecutionBudget, _finalize_tool_result, same_amount_window_probe
from src.tool_registry import validate_tool_request
from src.entity_profile import EntityAccumulator


class FakeRowSource(RowSource):
    def __init__(self, rows):
        self.rows = rows

    def iter_model_rows(self, case_id, model_id):
        for row_id, payload in self.rows:
            yield row_id, payload


class AgentDemoTests(unittest.TestCase):
    def test_initial_hypothesis_fact_ids_must_belong_to_entity(self):
        payload = {
            "hypotheses": [
                {
                    "hypothesis_id": "hyp_01",
                    "entity_name": "甲",
                    "statement": "甲需要进一步核查。",
                    "basis_fact_ids": ["fact_b"],
                    "target_model_ids": [70],
                }
            ]
        }
        result = _parse_initial_hypotheses(
            payload,
            {"甲", "乙"},
            {"甲": ["fact_a"], "乙": ["fact_b"]},
            [70],
            1,
        )
        self.assertEqual(result, [])

    def test_time_range_is_rejected(self):
        request = {
            "action": "call_tool",
            "tool_name": "entity_model_slice",
            "params": {"model_id": 70, "entity_name": "甲", "time_range": "2023-01"},
        }
        ok, reason = _validate_agent_request(
            request,
            {70: {"model_kind": "entity", "allowed_tools": ["entity_model_slice"]}},
            {"甲"},
            [70],
            ["entity_model_slice"],
            set(),
        )
        self.assertFalse(ok)
        self.assertIn("time_range", reason)

    def test_row_query_any_all_and_in_semantics(self):
        payload = {"source": "甲", "target": "乙", "kind": "转账", "level": "高"}
        self.assertTrue(_matches_row_query(payload, RowQuery(any_field_equals=(("source", "甲"), ("target", "甲")))))
        self.assertTrue(_matches_row_query(payload, RowQuery(all_field_equals=(("source", "甲"), ("kind", "转账")))))
        self.assertFalse(_matches_row_query(payload, RowQuery(all_field_equals=(("source", "甲"), ("kind", "消费")))))
        self.assertTrue(_matches_row_query(payload, RowQuery(field_in=(("level", ("中", "高")),))))
        self.assertTrue(_matches_exact_filters(payload, {"source": "不存在", "target": "乙"}))

    def test_db_row_query_preserves_group_semantics(self):
        query = RowQuery(
            any_field_equals=(("source", "甲"), ("target", "甲")),
            all_field_equals=(("kind", "转账"),),
            field_in=(("level", ("中", "高")),),
            limit=11,
        )
        sql, params = _build_db_row_query(13, 70, query)
        self.assertIn(" OR ", sql)
        self.assertIn(" AND ", sql)
        self.assertIn("ANY(%s)", sql)
        self.assertTrue(sql.endswith("LIMIT %s"))
        self.assertEqual(params[-1], 11)

    def test_duplicate_tool_call_is_rejected(self):
        request = {
            "action": "call_tool",
            "tool_name": "entity_model_slice",
            "params": {"model_id": 70, "entity_name": "甲"},
        }
        ok, reason = _validate_agent_request(
            request,
            {70: {"model_kind": "entity", "allowed_tools": ["entity_model_slice"]}},
            {"甲"},
            [70],
            ["entity_model_slice"],
            {_call_key(request)},
        )
        self.assertFalse(ok)
        self.assertIn("duplicate", reason)

    def test_stop_step_preserves_latest_status(self):
        step = _make_stop_step(
            "hyp_01",
            3,
            {"tool_name": "stop", "params": {}, "source": "llm_agent_planner"},
            "llm_stop",
            "证据已经足够。",
            "enhanced",
        )
        self.assertEqual(step["status"], "enhanced")

    def test_single_same_amount_record_is_not_enhanced(self):
        rows = [(1, {"name": "甲", "amount": 100.0, "date": "2024-01-01"})]
        output = same_amount_window_probe(
            None,
            13,
            {
                "person_fields": ["name"],
                "amount_fields": ["amount"],
                "primary_amount_field": "amount",
                "time_fields": ["date"],
                "counterparty_fields": [],
            },
            {"model_id": 70, "entity_name": "甲"},
            row_source=FakeRowSource(rows),
        )
        self.assertEqual(output["status_signal"], "unresolved")

    def test_tool_budget_marks_truncated_result_unresolved(self):
        rows = [
            (index, {"name": "甲", "amount": 100.0, "date": "2024-01-01"})
            for index in range(3)
        ]
        output = same_amount_window_probe(
            None,
            13,
            {
                "person_fields": ["name"],
                "amount_fields": ["amount"],
                "primary_amount_field": "amount",
                "time_fields": ["date"],
                "counterparty_fields": [],
            },
            {"model_id": 70, "entity_name": "甲"},
            row_source=FakeRowSource(rows),
            budget=ToolExecutionBudget(max_rows_per_call=2),
        )
        self.assertTrue(output["truncated"])
        self.assertFalse(output["aggregation_complete"])
        self.assertEqual(output["rows_read"], 2)
        self.assertEqual(output["status_signal"], "unresolved")

    def test_row_scanning_tool_requires_row_source(self):
        with self.assertRaises(RuntimeError):
            same_amount_window_probe(
                None,
                13,
                {"person_fields": ["name"], "amount_fields": ["amount"], "time_fields": ["date"]},
                {"model_id": 70, "entity_name": "甲"},
            )

    def test_tool_param_schema_rejects_unknown_type_and_large_list(self):
        strategies = {70: {"model_kind": "entity", "allowed_tools": ["entity_model_slice"]}}
        request = {
            "tool_name": "entity_model_slice",
            "params": {"model_id": 70, "entity_name": "甲", "unexpected": True},
        }
        self.assertFalse(validate_tool_request(request, strategies, {"甲"})[0])
        request = {"tool_name": "entity_model_slice", "params": {"model_id": True, "entity_name": "甲"}}
        self.assertFalse(validate_tool_request(request, strategies, {"甲"})[0])
        overlap_strategies = {
            value: {"model_kind": "entity", "allowed_tools": ["counterparty_overlap_probe"]}
            for value in range(1, 10)
        }
        request = {
            "tool_name": "counterparty_overlap_probe",
            "params": {"entity_name": "甲", "model_ids": list(range(1, 10))},
        }
        self.assertFalse(validate_tool_request(request, overlap_strategies, {"甲"})[0])

    def test_stable_digest_is_equal_across_processes(self):
        value = {"entity": "甲", "model_id": 70, "params": [1, 2]}
        expected = stable_digest(value)
        code = "from src.stable_ids import stable_digest; print(stable_digest({'entity':'甲','model_id':70,'params':[1,2]}))"
        actual = subprocess.check_output([sys.executable, "-c", code], text=True).strip()
        self.assertEqual(actual, expected)

    def test_judge_fallback_is_always_unresolved(self):
        result = _deterministic_judge(
            {"hypothesis_id": "hyp_1"},
            [{"supplement_fact_id": "sf_1", "output": {"status_signal": "enhanced"}}],
            "fallback",
            FallbackPolicy.REVIEW_SAFE,
        )
        self.assertEqual(result["status"], "unresolved")

    def test_truncated_llm_weakened_is_forced_unresolved(self):
        audit = {"judge_fallback_count": 0, "judge_success_count": 0}
        supplement = {
            "supplement_fact_id": "sf_1",
            "output": {
                "truncated": True,
                "aggregation_complete": False,
                "status_signal": "unresolved",
            },
        }
        payload = {
            "status": "weakened",
            "assessment": "当前补证结果削弱了该线索。",
            "supporting_supplement_fact_ids": ["sf_1"],
        }
        with patch("src.judge._ask_llm_for_judge", return_value=payload):
            result = _judge_single_hypothesis(
                SimpleNamespace(llm=SimpleNamespace(is_complete=True)),
                {"hypothesis_id": "hyp_1"},
                [supplement],
                [],
                audit,
                FallbackPolicy.REVIEW_SAFE,
            )
        self.assertEqual(result["status"], "unresolved")
        self.assertEqual(result["original_status"], "weakened")
        self.assertEqual(audit["incomplete_downgrade_count"], 1)

    def test_aggregation_incomplete_forces_tool_signal_unresolved(self):
        output = _finalize_tool_result(
            13,
            "test_tool",
            {"model_id": 70},
            {
                "summary": "测试结果。",
                "truncated": False,
                "aggregation_complete": False,
                "status_signal": "weakened",
            },
        )
        self.assertEqual(output["status_signal"], "unresolved")

    def test_fallback_policies_are_explicit(self):
        self.assertEqual(normalize_fallback_policy("strict_demo"), FallbackPolicy.STRICT_DEMO)
        self.assertEqual(normalize_fallback_policy("review_safe"), FallbackPolicy.REVIEW_SAFE)
        self.assertEqual(normalize_fallback_policy("facts_only"), FallbackPolicy.FACTS_ONLY)
        with self.assertRaises(ValueError):
            normalize_fallback_policy("unsafe")

    def test_internal_field_names_are_rejected_and_localized(self):
        assessment = "补证结果的top_outgoing包含高频关系。"
        self.assertFalse(_is_clear_chinese_assessment(assessment))
        self.assertEqual(_localize_internal_terms(assessment), "补证结果的主要流出关系包含高频关系。")
        repaired = _localize_judge_terms(assessment)
        self.assertEqual(repaired, "补证结果的主要流出关系包含高频关系。")
        self.assertTrue(_is_clear_chinese_assessment(repaired))

    def test_time_only_value_does_not_create_calendar_bucket(self):
        accumulator = EntityAccumulator("person:甲", "甲", "person", "name")
        accumulator.add_time(datetime(2026, 7, 2, 15, 30), True, False)
        self.assertEqual(accumulator.month_buckets, {})
        self.assertEqual(accumulator.day_buckets, {})
        self.assertEqual(accumulator.hour_buckets["15"], 1)

    def test_report_assertion_must_match_chain_and_references(self):
        context = {
            "known_fact_ids": ["fact_a"],
            "known_supplement_fact_ids": ["sf_001"],
            "known_model_ids": [70],
            "known_entities": ["甲"],
            "chains": [
                {
                    "evidence_chain_id": "chain_1",
                    "model_class": "多维分析",
                    "model_ids": [70],
                    "hypothesis": {"basis_fact_ids": ["fact_a"]},
                    "final_status": "unresolved",
                    "supplement_facts": [{"supplement_fact_id": "sf_001"}],
                }
            ],
        }
        payload = {
            "assertions": [
                {
                    "assertion_id": "assert_001",
                    "model_class": "多维分析",
                    "model_ids": [70],
                    "entity_name": "甲",
                    "title": "待核查线索",
                    "statement": "甲仍需进一步核查。",
                    "status": "enhanced",
                    "severity": "重点关注",
                    "fact_refs": ["fact_a"],
                    "supplement_fact_refs": ["sf_001"],
                    "evidence_chain_id": "chain_1",
                    "limitations": "当前证据不足。",
                }
            ]
        }
        assertions, errors = _validate_assertions(payload, context)
        self.assertEqual(assertions, [])
        self.assertTrue(any("status" in error for error in errors))

    def test_report_assertion_cannot_borrow_number_from_other_chain(self):
        context = {
            "known_fact_ids": ["fact_a", "fact_b"],
            "known_supplement_fact_ids": ["sf_a", "sf_b"],
            "known_model_ids": [70, 86],
            "known_entities": ["甲", "乙"],
            "chains": [
                {
                    "evidence_chain_id": "chain_a",
                    "model_class": "多维分析",
                    "model_ids": [70],
                    "hypothesis": {"entity_name": "甲", "basis_fact_ids": ["fact_a"]},
                    "final_status": "unresolved",
                    "supplement_facts": [{"supplement_fact_id": "sf_a", "truncated": False}],
                    "allowed_numeric_tokens": ["70", "100"],
                },
                {
                    "evidence_chain_id": "chain_b",
                    "model_class": "金融画像",
                    "model_ids": [86],
                    "hypothesis": {"entity_name": "乙", "basis_fact_ids": ["fact_b"]},
                    "final_status": "unresolved",
                    "supplement_facts": [{"supplement_fact_id": "sf_b", "truncated": False}],
                    "allowed_numeric_tokens": ["86", "200"],
                },
            ],
        }
        payload = {"assertions": [
            self._assertion("chain_a", "多维分析", [70], "甲", "fact_a", "sf_a", "甲涉及200条记录。"),
            self._assertion("chain_b", "金融画像", [86], "乙", "fact_b", "sf_b", "乙涉及200条记录。"),
        ]}
        assertions, errors = _validate_assertions(payload, context)
        self.assertEqual(assertions, [])
        self.assertTrue(any("当前证据链" in error for error in errors))

    def test_report_assertion_accepts_allowed_amount_unit_conversion(self):
        context = {
            "known_fact_ids": ["fact_a"],
            "known_supplement_fact_ids": ["sf_a"],
            "known_model_ids": [70],
            "known_entities": ["甲"],
            "chains": [{
                "evidence_chain_id": "chain_a",
                "model_class": "多维分析",
                "model_ids": [70],
                "hypothesis": {"entity_name": "甲", "basis_fact_ids": ["fact_a"]},
                "final_status": "enhanced",
                "supplement_facts": [{"supplement_fact_id": "sf_a", "truncated": False}],
                "allowed_numeric_tokens": ["70", "34170000", "3417", "0.34"],
            }],
        }
        payload = {"assertions": [
            self._assertion("chain_a", "多维分析", [70], "甲", "fact_a", "sf_a", "甲涉及金额3417万元。", "enhanced")
        ]}
        assertions, errors = _validate_assertions(payload, context)
        self.assertEqual(len(assertions), 1, errors)

    def test_report_assertion_requires_complete_chain_model_ids(self):
        context = {
            "known_fact_ids": ["fact_a"],
            "known_supplement_fact_ids": ["sf_a"],
            "known_model_ids": [70, 86],
            "known_entities": ["甲"],
            "chains": [{
                "evidence_chain_id": "chain_a",
                "model_class": "多维分析",
                "model_ids": [70, 86],
                "hypothesis": {"entity_name": "甲", "basis_fact_ids": ["fact_a"]},
                "final_status": "enhanced",
                "supplement_facts": [{"supplement_fact_id": "sf_a", "truncated": False}],
                "allowed_numeric_tokens": ["70", "86"],
            }],
        }
        payload = {"assertions": [
            self._assertion("chain_a", "多维分析", [70], "甲", "fact_a", "sf_a", "甲的线索获得支持。", "enhanced")
        ]}
        assertions, errors = _validate_assertions(payload, context)
        self.assertEqual(assertions, [])
        self.assertTrue(any("完整匹配" in error for error in errors))

    def test_report_assertion_rejects_weakened_for_incomplete_chain(self):
        context = {
            "known_fact_ids": ["fact_a"],
            "known_supplement_fact_ids": ["sf_a"],
            "known_model_ids": [70],
            "known_entities": ["甲"],
            "chains": [{
                "evidence_chain_id": "chain_a",
                "model_class": "多维分析",
                "model_ids": [70],
                "hypothesis": {"entity_name": "甲", "basis_fact_ids": ["fact_a"]},
                "final_status": "weakened",
                "supplement_facts": [{
                    "supplement_fact_id": "sf_a",
                    "truncated": False,
                    "aggregation_complete": False,
                }],
                "allowed_numeric_tokens": ["70"],
            }],
        }
        assertion = self._assertion(
            "chain_a", "多维分析", [70], "甲", "fact_a", "sf_a", "当前线索被削弱。", "weakened"
        )
        assertion["limitations"] = "当前聚合未完成。"
        assertions, errors = _validate_assertions({"assertions": [assertion]}, context)
        self.assertEqual(assertions, [])
        self.assertTrue(any("只能生成待核查断言" in error for error in errors))

    def test_amount_numeric_tokens_include_wan_and_yi_conversions(self):
        tokens = _allowed_numeric_tokens({"amount_sum": 34170000})
        self.assertIn("3417", tokens)
        self.assertIn("0.34", tokens)

    @staticmethod
    def _assertion(chain_id, model_class, model_ids, entity, fact_id, supplement_id, statement, status="unresolved"):
        return {
            "assertion_id": f"assert_{chain_id}",
            "model_class": model_class,
            "model_ids": model_ids,
            "entity_name": entity,
            "title": "研判线索",
            "statement": statement,
            "status": status,
            "severity": "重点关注" if status == "enhanced" else "待核查",
            "fact_refs": [fact_id],
            "supplement_fact_refs": [supplement_id],
            "evidence_chain_id": chain_id,
            "limitations": "当前结论仅限已有证据。",
        }


if __name__ == "__main__":
    unittest.main()

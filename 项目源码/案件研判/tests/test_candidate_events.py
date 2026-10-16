import pytest

from src.candidate_events import CandidateEventBuilder, CandidateEventError
from src.directed_supplement import build_recheck_tasks


def test_candidate_builder_merges_rule_flags_and_structured_model_clues():
    fact_pack = {
        "focus_entities": [
            {
                "entity_name": "主体A",
                "fact_id": "fact-1",
                "anomaly_flags": [
                    {
                        "risk_tag": "night_activity",
                        "trigger_rule": "night_count >= threshold",
                        "report_expression": "夜间交易占比较高",
                        "metrics": {"night_count": 12},
                    }
                ],
            }
        ]
    }
    events = CandidateEventBuilder(max_events=5).build(
        fact_pack,
        model_clues=[
            {
                "event_type": "risk_night_activity",
                "subject_id": "主体A",
                "fact_ids": ["fact-1"],
                "rationale": "模型线索建议核验夜间交易上下文",
            }
        ],
    )

    assert len(events) == 1
    assert events[0]["source"] == ["rule", "model"]
    assert events[0]["fact_ids"] == ["fact-1"]
    assert events[0]["subject_ids"] == ["主体A"]


def test_candidate_builder_rejects_unstructured_model_clues_and_limits():
    with pytest.raises(CandidateEventError):
        CandidateEventBuilder().build({}, model_clues=[{"event_type": "risk_unknown"}])

    fact_pack = {
        "anomaly_flags": [
            {"risk_tag": "a", "fact_id": "f-a"},
            {"risk_tag": "b", "fact_id": "f-b"},
        ]
    }
    with pytest.raises(CandidateEventError, match="candidate event limit"):
        CandidateEventBuilder(max_events=1).build(fact_pack)


def test_gap_becomes_scope_aware_recheck_task():
    state = {
        "case_scope": {
            "entity_ids": ["主体A"],
            "fact_ids": ["fact-1"],
            "allowed_tools": ["lookup_relation"],
        },
        "candidate_events": [
            {
                "event_id": "candidate-1",
                "subject_ids": ["主体A"],
                "fact_ids": ["fact-1"],
            }
        ],
        "verification": {
            "gaps": [
                {
                    "candidate_event_id": "candidate-1",
                    "missing_fact": "双方关系",
                    "allowed_tools": ["lookup_relation"],
                }
            ]
        },
    }

    tasks = build_recheck_tasks(state)

    assert len(tasks) == 1
    assert tasks[0].entity_ids == ("主体A",)
    assert tasks[0].basis_fact_ids == ("fact-1",)
    assert tasks[0].allowed_tools == ("lookup_relation",)
    assert "双方关系" in tasks[0].question

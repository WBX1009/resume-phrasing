from src.quality_context import pack_context
from src.quality_models import Fact, ModelProfile


def test_context_packing_preserves_every_fact_reference():
    profile = ModelProfile(86, "关系", "关系分析", "graph_cross_domain", "sampled", "plan_react_reflect", strategy_key="m86", specialized_tools=("edge_summary",))
    facts = [Fact(f"fact_{i}", 86, 13, "numeric_summary", f"字段{i}", {"payload": "x" * 500}) for i in range(30)]
    packed = pack_context(profile, facts, [], max_chars=800)
    assert packed["raw_chars"] > packed["packed_chars"]
    assert set(packed["fact_ids"]) == {fact.fact_id for fact in facts}
    assert len(packed["fact_summaries"]) == len(facts)
    assert packed["omitted_value_fact_ids"]


def test_context_packing_reports_budget_overflow_without_dropping_ids():
    profile = ModelProfile(86, "关系", "关系分析", "graph_cross_domain", "sampled", "plan_react_reflect", strategy_key="m86")
    facts = [Fact("fact_1", 86, 13, "numeric_summary", "字段", {"payload": "x" * 500})]
    packed = pack_context(profile, facts, [], max_chars=10)
    assert packed["max_chars"] == 10
    assert packed["budget_exceeded"] is True
    assert packed["fact_ids"] == ["fact_1"]

from concurrent.futures import ThreadPoolExecutor
import time

import pytest

from src.quality_llm import BoundedLLM, BudgetLedger
import json


def test_bounded_llm_cache_and_whitelist(tmp_path):
    calls = []

    def transport(system, user, max_tokens):
        calls.append((system, user, max_tokens))
        return '{"actions":["edge_summary"]}', 12

    llm = BoundedLLM(tmp_path, ledger=BudgetLedger(max_calls=2, max_tokens=10000), transport=transport)
    profile = {"model_id": 86, "description": "关系", "specialized_tools": ["edge_summary"], "reasoning_mode": "plan_react_reflect"}
    first = llm.advisory_plan(profile, [{"fact_id": "f1", "kind": "relation_summary"}])
    second = llm.advisory_plan(profile, [{"fact_id": "f1", "kind": "relation_summary"}])
    assert first.status == "online"
    assert second.status == "cache"
    assert len(calls) == 1
    assert llm.ledger.calls_used == 1


def test_bounded_llm_rejects_tool_whitelist(tmp_path):
    def transport(system, user, max_tokens):
        return '{"actions":["not_allowed"]}', 4

    llm = BoundedLLM(tmp_path, transport=transport)
    result = llm.advisory_plan(
        {"model_id": 86, "specialized_tools": ["edge_summary"], "reasoning_mode": "react"},
        [],
    )
    assert result.status == "invalid"
    assert "whitelist" in (result.reason or "")


def test_bounded_llm_rejects_duplicate_actions(tmp_path):
    def transport(system, user, max_tokens):
        return '{"actions":["edge_summary","edge_summary"]}', 4

    llm = BoundedLLM(tmp_path, transport=transport)
    result = llm.advisory_plan(
        {"model_id": 86, "specialized_tools": ["edge_summary"], "reasoning_mode": "react"},
        [],
    )
    assert result.status == "invalid"
    assert "duplicate" in (result.reason or "")


def test_budget_stops_before_transport(tmp_path):
    with pytest.raises(ValueError, match="max_calls"):
        BudgetLedger(max_calls=-1)
    with pytest.raises(ValueError, match="max_tokens"):
        BudgetLedger(max_tokens=True)

    called = False

    def transport(system, user, max_tokens):
        nonlocal called
        called = True
        return '{"actions":[]}', 1

    llm = BoundedLLM(tmp_path, ledger=BudgetLedger(max_calls=0, max_tokens=0), transport=transport)
    result = llm.advisory_plan({"model_id": 1, "specialized_tools": []}, [])
    assert result.status == "unavailable"
    assert called is False

    invalid_tokens_result = llm.complete_json("system", "invalid", max_tokens=0)
    assert invalid_tokens_result.status == "unavailable"
    assert "positive integer" in (invalid_tokens_result.reason or "")
    assert called is False

    def concurrent_transport(system, user, max_tokens):
        time.sleep(0.01)
        return '{"actions":[]}', 1

    concurrent_ledger = BudgetLedger(max_calls=4, max_tokens=100_000)
    concurrent_llm = BoundedLLM(
        tmp_path / "budget-race",
        ledger=concurrent_ledger,
        transport=concurrent_transport,
    )
    with ThreadPoolExecutor(max_workers=32) as pool:
        results = list(
            pool.map(
                lambda index: concurrent_llm.complete_json("system", f"user-{index}", max_tokens=32),
                range(32),
            )
        )
    assert concurrent_ledger.calls_used == 4
    assert sum(result.status == "online" for result in results) == 4
    assert all(result.status in {"online", "unavailable"} for result in results)


def test_provider_failure_is_visible_but_safe(tmp_path):
    def transport(system, user, max_tokens):
        raise RuntimeError("provider down")

    ledger = BudgetLedger(max_calls=2, max_tokens=10000)
    llm = BoundedLLM(tmp_path, ledger=ledger, transport=transport)
    result = llm.advisory_plan({"model_id": 86, "specialized_tools": []}, [])
    assert result.status == "unavailable"
    assert ledger.calls_used == 1
    assert "provider down" in (result.reason or "")

    def invalid_usage_transport(system, user, max_tokens):
        return '{"actions":[]}', -1

    usage_ledger = BudgetLedger(max_calls=1, max_tokens=10_000)
    usage_llm = BoundedLLM(
        tmp_path / "invalid-usage",
        ledger=usage_ledger,
        transport=invalid_usage_transport,
    )
    invalid_usage = usage_llm.complete_json("system", "invalid-usage", max_tokens=32)
    assert invalid_usage.status == "unavailable"
    assert "actual token count" in (invalid_usage.reason or "")
    assert usage_ledger.calls_used == 1
    assert list((tmp_path / "invalid-usage").glob("*.json")) == []


def test_openai_compatible_primary_env_configures_local_qwen(monkeypatch, tmp_path):
    monkeypatch.setenv("LLM_PRIMARY_PROVIDER", "openai-compatible")
    monkeypatch.setenv("LLM_PRIMARY_MODEL", "Qwen/Qwen3-32B-Instruct")
    monkeypatch.setenv("LLM_PRIMARY_API_KEY", "EMPTY")
    monkeypatch.setenv("LLM_PRIMARY_BASE_URL", "http://127.0.0.1:8000/v1")
    llm = BoundedLLM(tmp_path)
    assert llm.provider == "openai-compatible"
    assert llm.model == "Qwen/Qwen3-32B-Instruct"
    assert llm.api_key == "EMPTY"
    assert llm.base_url == "http://127.0.0.1:8000/v1"


def test_corrupt_cache_payload_is_discarded_and_recovered(tmp_path):
    calls = []

    def transport(system, user, max_tokens):
        calls.append(1)
        return '{"actions":[]}', 3

    llm = BoundedLLM(tmp_path, transport=transport)
    key = llm._key("system", "user", 32)
    llm._cache_path(key).write_text(json.dumps({"payload": ["not-an-object"]}), encoding="utf-8")
    result = llm.complete_json("system", "user", max_tokens=32)
    assert result.status == "online"
    assert calls == [1]
    assert json.loads(llm._cache_path(key).read_text(encoding="utf-8"))["payload"] == {"actions": []}
    assert list(tmp_path.glob("*.tmp")) == []

    shared_path = tmp_path / "parallel-cache.json"

    def publish(index):
        BoundedLLM._write_cache(shared_path, {"payload": {"writer": index, "actions": []}})

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(publish, range(64)))
    parallel_payload = json.loads(shared_path.read_text(encoding="utf-8"))
    assert parallel_payload["payload"]["actions"] == []
    assert list(tmp_path.glob("*.tmp")) == []

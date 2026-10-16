from __future__ import annotations

import argparse
import json
import math
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests


LEGACY_CASE_FILES = {
    "route": "route_cases.jsonl",
    "rag": "rag_cases.jsonl",
    "agent_tool": "agent_tool_cases.jsonl",
}

BENCHMARK_CASE_FILES = {
    "retrieval": "retrieval_cases.jsonl",
    "rag": "rag_cases.jsonl",
    "agent": "agent_cases.jsonl",
}

ROUTE_KEYWORDS = {
    "agent": [
        "列出",
        "读取",
        "文件",
        "工具",
        "执行",
        "list",
        "read",
        "tool",
        "agent",
        "python",
        "calculate",
    ],
    "rag": [
        "知识库",
        "根据资料",
        "引用",
        "文档",
        "rag",
        "knowledge",
        "source",
        "citation",
    ],
}


def load_cases(path: Path) -> List[Dict[str, Any]]:
    cases: List[Dict[str, Any]] = []
    if not path.exists():
        return cases

    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path.name}:{line_number} is not valid JSON: {exc}") from exc
        if not isinstance(item, dict):
            raise ValueError(f"{path.name}:{line_number} must be a JSON object")
        if not item.get("id") or not item.get("question"):
            raise ValueError(f"{path.name}:{line_number} requires id and question")
        cases.append(item)
    return cases


def build_eval_report(eval_dir: Path) -> Dict[str, Any]:
    """Backward-compatible offline scaffold used by existing regression tests."""

    cases = {
        suite: load_cases(eval_dir / file_name)
        for suite, file_name in LEGACY_CASE_FILES.items()
    }
    route_metrics = run_legacy_route_eval(cases["route"])
    rag_metrics = run_legacy_rag_eval(cases["rag"])
    agent_tool_metrics = run_legacy_agent_tool_eval(cases["agent_tool"])
    failures = _collect_failures(route_metrics, rag_metrics, agent_tool_metrics)

    return {
        "status": "success" if not failures else "failed",
        "mode": "offline_static",
        "full_eval_executed": True,
        "live_adapter_eval_executed": False,
        "message": (
            "Offline deterministic eval completed. It validates route fixtures, "
            "RAG expectations, and agent tool expectations without calling LLM, "
            "embedding, vector, keyword, or external tool adapters."
        ),
        "case_counts": {suite: len(items) for suite, items in cases.items()},
        "metrics": {
            "route_accuracy": route_metrics["accuracy"],
            "rag_case_validity": rag_metrics["case_validity"],
            "agent_tool_case_validity": agent_tool_metrics["case_validity"],
        },
        "suites": {
            "route": route_metrics,
            "rag": rag_metrics,
            "agent_tool": agent_tool_metrics,
        },
        "failures": failures,
    }


def build_scaffold_report(eval_dir: Path) -> Dict[str, Any]:
    return build_eval_report(eval_dir)


def run_benchmark_eval(
    processed_dir: Path,
    suite: str = "all",
    sample_size: int = 20,
    live: bool = False,
    api_base: str = "http://localhost:8000/api",
    results_dir: Optional[Path] = None,
    run_id: Optional[str] = None,
) -> Dict[str, Any]:
    selected_suites = _selected_suites(suite)
    cases_by_suite = {
        item: _limit(load_cases(processed_dir / BENCHMARK_CASE_FILES[item]), sample_size)
        for item in selected_suites
    }
    live_outputs = (
        collect_live_outputs(cases_by_suite, api_base=api_base)
        if live
        else {}
    )

    suites: Dict[str, Dict[str, Any]] = {}
    if "retrieval" in cases_by_suite:
        suites["retrieval"] = run_retrieval_eval(
            cases_by_suite["retrieval"],
            live_outputs.get("retrieval", {}),
        )
    if "rag" in cases_by_suite:
        suites["rag"] = run_rag_benchmark_eval(
            cases_by_suite["rag"],
            live_outputs.get("rag", {}),
        )
    if "agent" in cases_by_suite:
        suites["agent"] = run_agent_benchmark_eval(
            cases_by_suite["agent"],
            live_outputs.get("agent", {}),
        )

    failures = _collect_failures(*suites.values())
    status = "success" if not failures else "failed"
    report = {
        "status": status,
        "mode": "live_backend" if live else "offline_processed",
        "full_eval_executed": True,
        "live_adapter_eval_executed": live,
        "run_id": run_id or _default_run_id(),
        "processed_dir": str(processed_dir.as_posix()),
        "case_counts": {name: len(items) for name, items in cases_by_suite.items()},
        "metrics": flatten_suite_metrics(suites),
        "suites": suites,
        "failures": failures,
        "config_snapshot": {
            "suite": suite,
            "sample_size": sample_size,
            "api_base": api_base if live else None,
            "created_at": datetime.now(timezone.utc).isoformat(),
        },
    }
    if results_dir is not None:
        write_run_artifacts(report, results_dir)
    return report


def collect_live_outputs(
    cases_by_suite: Dict[str, List[Dict[str, Any]]],
    api_base: str,
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    outputs: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for suite, cases in cases_by_suite.items():
        outputs[suite] = {}
        for case in cases:
            outputs[suite][case["id"]] = run_backend_case(case, api_base=api_base)
    return outputs


def run_backend_case(case: Dict[str, Any], api_base: str) -> Dict[str, Any]:
    session_id = f"eval_{case['id']}_{int(time.time() * 1000)}"
    payload = {
        "session_id": session_id,
        "query": case["question"],
        "collection_name": "__all__",
        "history_rounds": 3,
        "use_user_memory": False,
    }
    started = time.perf_counter()
    events: List[Dict[str, Any]] = []
    final_answer = ""
    try:
        with requests.post(
            f"{api_base.rstrip('/')}/conversation/stream",
            json=payload,
            stream=True,
            timeout=180,
        ) as response:
            response.raise_for_status()
            for line in response.iter_lines(decode_unicode=True):
                if not line or not line.startswith("data:"):
                    continue
                raw = line[5:].strip()
                if not raw or raw == "[DONE]":
                    continue
                event = json.loads(raw)
                if event.get("event") == "done":
                    break
                events.append(event)
                if event.get("event") == "message_delta":
                    final_answer += event.get("content", "")
                elif event.get("event") == "final_answer":
                    final_answer = event.get("content", "") or final_answer
    except Exception as exc:
        return {
            "success": False,
            "error": f"{type(exc).__name__}: {exc}",
            "latency_ms": round((time.perf_counter() - started) * 1000, 3),
            "events": events,
            "final_answer": final_answer,
        }

    return {
        "success": True,
        "latency_ms": round((time.perf_counter() - started) * 1000, 3),
        "events": events,
        "final_answer": final_answer,
        "route": _extract_route(events),
        "called_tools": _extract_called_tools(events),
        "citation_doc_ids": _extract_citation_doc_ids(events),
        "ranked_doc_ids": _extract_ranked_doc_ids(events),
    }


def run_retrieval_eval(
    cases: List[Dict[str, Any]],
    live_outputs: Dict[str, Dict[str, Any]] | None = None,
) -> Dict[str, Any]:
    details: List[Dict[str, Any]] = []
    accum = MetricAccumulator()
    for case in cases:
        output = (live_outputs or {}).get(case["id"], {})
        gold = _list_text(case.get("gold_doc_ids"))
        ranked = _list_text(output.get("ranked_doc_ids") or case.get("ranked_doc_ids"))
        pre_rerank = _list_text(case.get("pre_rerank_doc_ids") or ranked)
        if not gold or not ranked:
            details.append({"id": case["id"], "passed": False, "reason": "missing_gold_or_ranked_docs"})
            continue

        metrics = {
            "recall@5": recall_at_k(ranked, gold, 5),
            "recall@10": recall_at_k(ranked, gold, 10),
            "hit_rate@5": hit_rate_at_k(ranked, gold, 5),
            "hit_rate@10": hit_rate_at_k(ranked, gold, 10),
            "mrr@10": mrr_at_k(ranked, gold, 10),
            "ndcg@10": ndcg_at_k(ranked, gold, 10),
            "before_rerank_ndcg@10": ndcg_at_k(pre_rerank, gold, 10),
            "after_rerank_ndcg@10": ndcg_at_k(ranked, gold, 10),
        }
        accum.add(metrics)
        details.append({"id": case["id"], "passed": metrics["hit_rate@10"] > 0, **metrics})
    return accum.report("retrieval", len(cases), details)


def run_rag_benchmark_eval(
    cases: List[Dict[str, Any]],
    live_outputs: Dict[str, Dict[str, Any]] | None = None,
) -> Dict[str, Any]:
    details: List[Dict[str, Any]] = []
    accum = MetricAccumulator()
    for case in cases:
        output = (live_outputs or {}).get(case["id"], {})
        prediction = str(output.get("final_answer") or case.get("predicted_answer") or "")
        gold = str(case.get("gold_answer") or "")
        citation_doc_ids = _list_text(output.get("citation_doc_ids") or case.get("citation_doc_ids"))
        gold_doc_ids = _list_text(case.get("gold_doc_ids"))
        metrics = {
            "exact_match": exact_match(prediction, gold),
            "token_f1": token_f1(prediction, gold),
            "citation_presence": 1.0 if citation_doc_ids or "[C" in prediction else 0.0,
            "source_hit_rate": hit_rate_at_k(citation_doc_ids, gold_doc_ids, len(citation_doc_ids) or 1)
            if gold_doc_ids
            else 0.0,
        }
        accum.add(metrics)
        details.append({"id": case["id"], "passed": metrics["token_f1"] > 0, **metrics})
    return accum.report("rag", len(cases), details)


def run_agent_benchmark_eval(
    cases: List[Dict[str, Any]],
    live_outputs: Dict[str, Dict[str, Any]] | None = None,
) -> Dict[str, Any]:
    details: List[Dict[str, Any]] = []
    accum = MetricAccumulator()
    for case in cases:
        output = (live_outputs or {}).get(case["id"], {})
        predicted_route = str(output.get("route") or case.get("predicted_route") or "")
        called_tools = _list_text(output.get("called_tools") or case.get("called_tools"))
        expected_tools = _list_text(case.get("expected_tools"))
        prediction = str(output.get("final_answer") or case.get("predicted_answer") or "")
        gold = str(case.get("gold_answer") or "")
        metrics = {
            "route_accuracy": 1.0 if predicted_route == str(case.get("expected_route") or "") else 0.0,
            "tool_choice_accuracy": 1.0 if set(expected_tools).issubset(set(called_tools)) else 0.0,
            "argument_validity": 1.0 if bool(case.get("arguments_valid", False)) else 0.0,
            "tool_execution_success": 1.0 if bool(case.get("tool_execution_success", False)) else 0.0,
            "final_answer_accuracy": 1.0 if answer_contains_gold(prediction, gold) else 0.0,
            "hitl_resume_success": 1.0 if bool(case.get("hitl_resume_success", False)) else 0.0,
        }
        if output:
            metrics["tool_execution_success"] = 1.0 if _any_successful_tool(output.get("events", [])) else 0.0
            metrics["hitl_resume_success"] = 1.0 if _has_event(output.get("events", []), "approval_result") else metrics["hitl_resume_success"]
        accum.add(metrics)
        details.append({"id": case["id"], "passed": metrics["final_answer_accuracy"] > 0, **metrics})
    return accum.report("agent", len(cases), details)


class MetricAccumulator:
    def __init__(self) -> None:
        self.totals: Dict[str, float] = {}
        self.counts: Dict[str, int] = {}

    def add(self, metrics: Dict[str, float]) -> None:
        for key, value in metrics.items():
            self.totals[key] = self.totals.get(key, 0.0) + float(value)
            self.counts[key] = self.counts.get(key, 0) + 1

    def report(
        self,
        suite: str,
        case_count: int,
        details: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        metrics = {
            key: round(total / max(self.counts.get(key, 0), 1), 4)
            for key, total in sorted(self.totals.items())
        }
        passed = sum(1 for item in details if item.get("passed"))
        return {
            "suite": suite,
            "evaluated": case_count,
            "passed": passed,
            "pass_rate": _ratio(passed, case_count),
            "metrics": metrics,
            "details": details,
        }


def write_run_artifacts(report: Dict[str, Any], results_root: Path) -> Path:
    run_id = str(report.get("run_id") or _default_run_id())
    run_dir = results_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "metrics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    failures = report.get("failures") or []
    (run_dir / "failed_cases.jsonl").write_text(
        "\n".join(json.dumps(item, ensure_ascii=False) for item in failures)
        + ("\n" if failures else ""),
        encoding="utf-8",
    )
    (run_dir / "config_snapshot.json").write_text(
        json.dumps(report.get("config_snapshot", {}), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    latest_path = results_root / "latest_run.txt"
    latest_path.write_text(run_id, encoding="utf-8")
    return run_dir


def flatten_suite_metrics(suites: Dict[str, Dict[str, Any]]) -> Dict[str, float]:
    flattened: Dict[str, float] = {}
    for suite, payload in suites.items():
        flattened[f"{suite}_pass_rate"] = payload.get("pass_rate", 0.0)
        for metric_name, value in (payload.get("metrics") or {}).items():
            flattened[f"{suite}_{metric_name}"] = value
    return flattened


def run_legacy_route_eval(cases: List[Dict[str, Any]]) -> Dict[str, Any]:
    details: List[Dict[str, Any]] = []
    correct = 0
    evaluated = 0
    for case in cases:
        expected = str(case.get("expected_route") or "").strip()
        if not expected:
            continue
        evaluated += 1
        predicted = predict_static_route(str(case["question"]))
        passed = predicted == expected
        correct += int(passed)
        details.append({"id": case["id"], "expected": expected, "predicted": predicted, "passed": passed})
    return {"evaluated": evaluated, "passed": correct, "accuracy": _ratio(correct, evaluated), "details": details}


def run_legacy_rag_eval(cases: List[Dict[str, Any]]) -> Dict[str, Any]:
    details: List[Dict[str, Any]] = []
    passed = 0
    for case in cases:
        keywords = case.get("expected_keywords") or []
        valid_keywords = isinstance(keywords, list) and all(isinstance(item, str) and item.strip() for item in keywords)
        expected_route = str(case.get("expected_route") or "")
        case_passed = expected_route == "rag" and valid_keywords
        passed += int(case_passed)
        details.append({"id": case["id"], "expected_route": expected_route, "expected_keyword_count": len(keywords) if isinstance(keywords, list) else 0, "passed": case_passed})
    return {"evaluated": len(cases), "passed": passed, "case_validity": _ratio(passed, len(cases)), "details": details}


def run_legacy_agent_tool_eval(cases: List[Dict[str, Any]]) -> Dict[str, Any]:
    details: List[Dict[str, Any]] = []
    passed = 0
    for case in cases:
        expected_tools = case.get("expected_tools") or []
        valid_tools = isinstance(expected_tools, list) and all(isinstance(item, str) and item.strip() for item in expected_tools)
        expected_route = str(case.get("expected_route") or "")
        case_passed = expected_route == "agent" and valid_tools
        passed += int(case_passed)
        details.append({"id": case["id"], "expected_route": expected_route, "expected_tools": expected_tools if isinstance(expected_tools, list) else [], "passed": case_passed})
    return {"evaluated": len(cases), "passed": passed, "case_validity": _ratio(passed, len(cases)), "details": details}


def predict_static_route(question: str) -> str:
    normalized = question.lower()
    for route, keywords in ROUTE_KEYWORDS.items():
        if any(keyword.lower() in normalized for keyword in keywords):
            return route
    return "direct_chat"


def exact_match(prediction: str, gold: str) -> float:
    return 1.0 if normalize_answer(prediction) == normalize_answer(gold) else 0.0


def token_f1(prediction: str, gold: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(gold).split()
    if not pred_tokens or not gold_tokens:
        return 0.0
    common = 0
    gold_counts: Dict[str, int] = {}
    for token in gold_tokens:
        gold_counts[token] = gold_counts.get(token, 0) + 1
    for token in pred_tokens:
        if gold_counts.get(token, 0) > 0:
            common += 1
            gold_counts[token] -= 1
    if common == 0:
        return 0.0
    precision = common / len(pred_tokens)
    recall = common / len(gold_tokens)
    return round(2 * precision * recall / (precision + recall), 4)


def normalize_answer(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\u4e00-\u9fff.\- ]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def answer_contains_gold(prediction: str, gold: str) -> bool:
    if not gold:
        return False
    return normalize_answer(gold) in normalize_answer(prediction)


def recall_at_k(ranked: List[str], gold: List[str], k: int) -> float:
    gold_set = set(gold)
    if not gold_set:
        return 0.0
    return round(len(set(ranked[:k]) & gold_set) / len(gold_set), 4)


def hit_rate_at_k(ranked: List[str], gold: List[str], k: int) -> float:
    return 1.0 if set(ranked[:k]) & set(gold) else 0.0


def mrr_at_k(ranked: List[str], gold: List[str], k: int) -> float:
    gold_set = set(gold)
    for index, doc_id in enumerate(ranked[:k], start=1):
        if doc_id in gold_set:
            return round(1.0 / index, 4)
    return 0.0


def ndcg_at_k(ranked: List[str], gold: List[str], k: int) -> float:
    gold_set = set(gold)
    dcg = 0.0
    for index, doc_id in enumerate(ranked[:k], start=1):
        if doc_id in gold_set:
            dcg += 1.0 / math.log2(index + 1)
    ideal_hits = min(len(gold_set), k)
    idcg = sum(1.0 / math.log2(index + 1) for index in range(1, ideal_hits + 1))
    return round(dcg / idcg, 4) if idcg else 0.0


def _extract_route(events: List[Dict[str, Any]]) -> str:
    for event in events:
        if event.get("event") == "routing_decision":
            return str((event.get("data") or {}).get("route") or "")
    return ""


def _extract_called_tools(events: List[Dict[str, Any]]) -> List[str]:
    return [str(event.get("name")) for event in events if event.get("event") == "tool_call" and event.get("name")]


def _extract_citation_doc_ids(events: List[Dict[str, Any]]) -> List[str]:
    ids: List[str] = []
    for event in events:
        if event.get("event") != "citation_trace":
            continue
        for citation in (event.get("data") or {}).get("citations", []) or []:
            doc_id = citation.get("doc_id") or citation.get("source_file") or citation.get("source")
            if doc_id:
                ids.append(str(doc_id))
    return ids


def _extract_ranked_doc_ids(events: List[Dict[str, Any]]) -> List[str]:
    for event in events:
        if event.get("event") != "retrieval_trace":
            continue
        documents = (event.get("data") or {}).get("documents") or []
        ids = [str(item.get("id") or item.get("source_file") or "") for item in documents if isinstance(item, dict)]
        return [item for item in ids if item]
    return []


def _any_successful_tool(events: List[Dict[str, Any]]) -> bool:
    return any(event.get("event") == "tool_result" and (event.get("data") or {}).get("success") for event in events)


def _has_event(events: List[Dict[str, Any]], event_name: str) -> bool:
    return any(event.get("event") == event_name for event in events)


def _list_text(value: Any) -> List[str]:
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return []


def _selected_suites(suite: str) -> List[str]:
    normalized = (suite or "all").strip().lower()
    if normalized == "all":
        return ["retrieval", "rag", "agent"]
    if normalized not in BENCHMARK_CASE_FILES:
        available = ", ".join(["all", *BENCHMARK_CASE_FILES])
        raise ValueError(f"Unknown suite '{suite}'. Available suites: {available}")
    return [normalized]


def _limit(items: List[Dict[str, Any]], sample_size: int) -> List[Dict[str, Any]]:
    if sample_size <= 0:
        return items
    return items[:sample_size]


def _collect_failures(*suite_metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
    failures: List[Dict[str, Any]] = []
    for metrics in suite_metrics:
        for detail in metrics.get("details", []):
            if not detail.get("passed"):
                failures.append({"suite": metrics.get("suite"), **detail})
    return failures


def _ratio(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return round(numerator / denominator, 4)


def _default_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _default_processed_dir() -> Path:
    public_dir = Path("data/eval_datasets/processed")
    if any((public_dir / file_name).exists() for file_name in BENCHMARK_CASE_FILES.values()):
        return public_dir
    return Path("evals/fixtures/processed")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run DeepAssist public benchmark evaluation.")
    parser.add_argument("--suite", choices=["retrieval", "rag", "agent", "all"], default="all")
    parser.add_argument("--processed-dir", default=None)
    parser.add_argument("--results-dir", default="evals/results")
    parser.add_argument("--sample-size", type=int, default=20)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--api-base", default="http://localhost:8000/api")
    parser.add_argument("--offline", action="store_true", help="Force processed-file evaluation only.")
    parser.add_argument("--live", action="store_true", help="Call the local backend SSE endpoint.")
    parser.add_argument(
        "--legacy",
        action="store_true",
        help="Run the backward-compatible route/RAG/agent-tool JSONL scaffold.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.legacy:
        report = build_eval_report(Path("evals"))
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0 if report["status"] == "success" else 1

    processed_dir = Path(args.processed_dir) if args.processed_dir else _default_processed_dir()
    report = run_benchmark_eval(
        processed_dir=processed_dir,
        suite=args.suite,
        sample_size=args.sample_size,
        live=bool(args.live and not args.offline),
        api_base=args.api_base,
        results_dir=Path(args.results_dir),
        run_id=args.run_id,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

try:
    from .dataset_catalog import DATASET_SPECS
except ImportError:  # pragma: no cover - direct script execution
    from dataset_catalog import DATASET_SPECS


SUITE_FILES = {
    "retrieval": "retrieval_cases.jsonl",
    "rag": "rag_cases.jsonl",
    "agent": "agent_cases.jsonl",
}


def prepare_datasets(raw_dir: Path, out_dir: Path, profile: str = "local") -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    grouped: Dict[str, List[Dict[str, Any]]] = {"retrieval": [], "rag": [], "agent": []}
    stats: Dict[str, Dict[str, int]] = {}

    for dataset_dir in sorted(raw_dir.iterdir()) if raw_dir.exists() else []:
        if not dataset_dir.is_dir():
            continue
        metadata_path = dataset_dir / "download_metadata.json"
        if not metadata_path.exists():
            continue
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        dataset_name = str(metadata.get("name") or dataset_dir.name)
        spec = DATASET_SPECS.get(dataset_name)
        suite = str(metadata.get("suite") or (spec.suite if spec else ""))
        if not suite:
            continue
        dataset_path = dataset_dir / "dataset"
        rows = list(_iter_dataset_rows(dataset_path))
        converted = convert_rows(dataset_name, suite, rows, profile=profile)
        grouped.setdefault(suite, []).extend(converted)
        stats[dataset_name] = {
            "input_rows": len(rows),
            "converted_rows": len(converted),
            "skipped_rows": max(len(rows) - len(converted), 0),
        }

    for suite, file_name in SUITE_FILES.items():
        write_jsonl(out_dir / file_name, grouped.get(suite, []))

    return {
        "status": "success",
        "profile": profile,
        "out_dir": str(out_dir.as_posix()),
        "case_counts": {suite: len(items) for suite, items in grouped.items()},
        "datasets": stats,
    }


def convert_rows(
    dataset_name: str,
    suite: str,
    rows: List[Dict[str, Any]],
    profile: str = "local",
) -> List[Dict[str, Any]]:
    if suite == "rag":
        return [_case for row in rows if (_case := convert_rag_row(dataset_name, row))]
    if suite == "agent":
        return [_case for row in rows if (_case := convert_agent_row(dataset_name, row))]
    if suite == "retrieval":
        return [_case for row in rows if (_case := convert_retrieval_row(dataset_name, row))]
    return []


def convert_rag_row(dataset_name: str, row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    question = _first_text(row, ["question", "query"])
    if not question:
        return None
    contexts = _contexts_from_row(row)
    answer = _answer_from_row(row)
    if not contexts or not answer:
        return None
    case_id = _case_id(dataset_name, row)
    return {
        "id": case_id,
        "dataset": dataset_name,
        "suite": "rag",
        "question": question,
        "contexts": contexts,
        "gold_answer": answer,
        "gold_doc_ids": [f"{case_id}_ctx_{index}" for index in range(len(contexts))],
        "expected_route": "rag",
        "expected_tools": ["search_knowledge_base"],
        "metadata": {"source_dataset": dataset_name},
    }


def convert_agent_row(dataset_name: str, row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    question = _first_text(row, ["question", "prompt", "user_prompt", "query"])
    if not question:
        return None
    case_id = _case_id(dataset_name, row)
    if dataset_name == "gsm8k":
        answer = _answer_from_row(row)
        numeric_answer = extract_final_number(answer)
        return {
            "id": case_id,
            "dataset": dataset_name,
            "suite": "agent",
            "question": question,
            "gold_answer": numeric_answer or answer,
            "expected_route": "agent",
            "expected_tools": ["execute_python_sandbox"],
            "expected_capabilities": ["python_execution"],
            "metadata": {"source_dataset": dataset_name},
        }

    expected_tools = _tools_from_row(row)
    if not expected_tools:
        return None
    return {
        "id": case_id,
        "dataset": dataset_name,
        "suite": "agent",
        "question": question,
        "gold_answer": _answer_from_row(row),
        "expected_route": "agent",
        "expected_tools": expected_tools,
        "expected_capabilities": ["tool_calling"],
        "metadata": {"source_dataset": dataset_name},
    }


def convert_retrieval_row(dataset_name: str, row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    question = _first_text(row, ["question", "query", "text"])
    gold_doc_ids = _list_text(row.get("gold_doc_ids") or row.get("relevant_doc_ids") or row.get("doc_ids"))
    ranked_doc_ids = _list_text(row.get("ranked_doc_ids") or row.get("retrieved_doc_ids"))
    contexts = _contexts_from_row(row)
    if not question or not gold_doc_ids:
        return None
    return {
        "id": _case_id(dataset_name, row),
        "dataset": dataset_name,
        "suite": "retrieval",
        "question": question,
        "contexts": contexts,
        "gold_doc_ids": gold_doc_ids,
        "ranked_doc_ids": ranked_doc_ids or gold_doc_ids,
        "pre_rerank_doc_ids": _list_text(row.get("pre_rerank_doc_ids")) or ranked_doc_ids or gold_doc_ids,
        "expected_route": "rag",
        "metadata": {"source_dataset": dataset_name},
    }


def validate_case(case: Dict[str, Any]) -> None:
    if not case.get("id") or not case.get("question"):
        raise ValueError("Eval case requires id and question")
    suite = case.get("suite")
    if suite == "retrieval" and not case.get("gold_doc_ids"):
        raise ValueError(f"Retrieval case {case.get('id')} requires gold_doc_ids")
    if suite == "rag" and not (case.get("contexts") or case.get("gold_doc_ids")):
        raise ValueError(f"RAG case {case.get('id')} requires contexts or gold_doc_ids")
    if suite == "agent" and not (case.get("expected_route") or case.get("expected_tools")):
        raise ValueError(f"Agent case {case.get('id')} requires expected_route or expected_tools")


def write_jsonl(path: Path, cases: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: List[str] = []
    for case in cases:
        validate_case(case)
        lines.append(json.dumps(case, ensure_ascii=False, sort_keys=True))
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def _iter_dataset_rows(dataset_path: Path) -> Iterable[Dict[str, Any]]:
    try:
        from datasets import load_from_disk
    except Exception as exc:
        raise RuntimeError("The 'datasets' package is required to prepare raw datasets.") from exc

    dataset = load_from_disk(str(dataset_path))
    if hasattr(dataset, "keys") and not hasattr(dataset, "features"):
        for split_name in dataset.keys():
            for row in dataset[split_name]:
                item = dict(row)
                item.setdefault("_split", split_name)
                yield item
        return
    for row in dataset:
        yield dict(row)


def _contexts_from_row(row: Dict[str, Any]) -> List[str]:
    if isinstance(row.get("context"), str):
        return [row["context"]]
    if isinstance(row.get("contexts"), list):
        return [str(item) for item in row["contexts"] if str(item).strip()]
    if isinstance(row.get("paragraphs"), list):
        return [str(item) for item in row["paragraphs"] if str(item).strip()]
    hotpot_context = row.get("context")
    if isinstance(hotpot_context, dict):
        sentences = hotpot_context.get("sentences")
        titles = hotpot_context.get("title") or hotpot_context.get("titles") or []
        if isinstance(sentences, list):
            rendered = []
            for index, sentence_group in enumerate(sentences):
                title = titles[index] if isinstance(titles, list) and index < len(titles) else ""
                if isinstance(sentence_group, list):
                    rendered.append(f"{title}\n" + " ".join(str(item) for item in sentence_group))
            return [item for item in rendered if item.strip()]
    return []


def _answer_from_row(row: Dict[str, Any]) -> str:
    answers = row.get("answers")
    if isinstance(answers, dict):
        text = answers.get("text")
        if isinstance(text, list) and text:
            return str(text[0])
        if isinstance(text, str):
            return text
    if isinstance(answers, list) and answers:
        return str(answers[0])
    return _first_text(row, ["answer", "gold_answer", "target", "final_answer"])


def _tools_from_row(row: Dict[str, Any]) -> List[str]:
    for key in ("expected_tools", "tools", "tool", "function", "functions"):
        value = row.get(key)
        items = _list_text(value)
        if items:
            return items
    return []


def _first_text(row: Dict[str, Any], keys: List[str]) -> str:
    for key in keys:
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _list_text(value: Any) -> List[str]:
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return []


def _case_id(dataset_name: str, row: Dict[str, Any]) -> str:
    raw_id = row.get("id") or row.get("_id") or row.get("qid") or row.get("question_id")
    if raw_id is None:
        raw_id = abs(hash(json.dumps(row, ensure_ascii=False, sort_keys=True))) % 10_000_000
    return f"{dataset_name}_{raw_id}"


def extract_final_number(answer: str) -> str:
    if not answer:
        return ""
    marker_match = re.search(r"####\s*([-+]?\d[\d,]*(?:\.\d+)?)", answer)
    if marker_match:
        return marker_match.group(1).replace(",", "")
    matches = re.findall(r"[-+]?\d[\d,]*(?:\.\d+)?", answer)
    return matches[-1].replace(",", "") if matches else ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert downloaded public benchmark datasets into DeepAssist eval cases."
    )
    parser.add_argument("--raw-dir", default="data/eval_datasets/raw")
    parser.add_argument("--out-dir", default="data/eval_datasets/processed")
    parser.add_argument("--profile", default="local")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = prepare_datasets(Path(args.raw_dir), Path(args.out_dir), profile=args.profile)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

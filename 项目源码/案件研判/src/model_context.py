from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path
from typing import Any


def load_categories(path: Path) -> dict[int, dict[str, str]]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        rows = csv.DictReader(fh)
        return {
            int(row["模型ID"]): row
            for row in rows
            if row.get("模型ID") and row["模型ID"].isdigit()
        }


def build_model_contexts(
    case_id: int,
    model_ids: list[int],
    records: list[dict[str, Any]],
    model_info: dict[int, dict[str, Any]],
    categories: dict[int, dict[str, str]],
    row_counts: dict[int, int] | None = None,
) -> list[dict[str, Any]]:
    counts = Counter(int(row["model_id"]) for row in records)
    row_counts = row_counts or {}
    contexts: list[dict[str, Any]] = []
    for model_id in model_ids:
        info = model_info.get(model_id, {})
        category = categories.get(model_id, {})
        contexts.append(
            {
                "case_id": case_id,
                "model_id": model_id,
                "model_name": info.get("model_name") or category.get("模型名称") or f"model_{model_id}",
                "model_category": category.get("模型类") or "未分类",
                "model_desc": info.get("model_desc") or "",
                "required_data": info.get("required_data") or "",
                "result_count": row_counts.get(model_id, counts.get(model_id, 0)),
                "loaded_result_count": counts.get(model_id, 0),
                "business_goal": category.get("模型类内总结目标") or info.get("model_desc") or "",
                "auxiliary_topic": category.get("辅助主题") or "",
            }
        )
    return contexts

"""Turn verifier evidence gaps into bounded, scope-aware recheck tasks."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
from typing import Any

from .case_task_scheduler import InvestigationTask


def build_recheck_tasks(state: Mapping[str, Any], *, max_tasks: int = 20) -> list[InvestigationTask]:
    verification = state.get("verification") or {}
    gaps = verification.get("gaps") or []
    if not isinstance(gaps, Sequence) or isinstance(gaps, (str, bytes, Mapping)):
        return []

    task_runs = state.get("task_runs") or {}
    candidates = {
        str(event.get("event_id")): event
        for event in state.get("candidate_events", []) or []
        if isinstance(event, Mapping) and event.get("event_id")
    }
    scope = state.get("case_scope") or {}
    scope_entities = set(str(item) for item in scope.get("entity_ids", []) or [])
    scope_tools = set(str(item) for item in scope.get("allowed_tools", []) or [])
    tasks: list[InvestigationTask] = []
    seen: set[str] = set()

    for raw_gap in gaps:
        if not isinstance(raw_gap, Mapping):
            continue
        gap = dict(raw_gap)
        source_task = task_runs.get(str(gap.get("task_id"))) or {}
        source_spec = source_task.get("task") if isinstance(source_task, Mapping) else {}
        source_spec = source_spec if isinstance(source_spec, Mapping) else {}
        candidate = candidates.get(str(gap.get("candidate_event_id")))

        entity_ids = _unique(
            gap.get("entity_ids")
            or gap.get("subject_ids")
            or source_spec.get("entity_ids")
            or (candidate or {}).get("subject_ids")
        )
        entity_ids = [item for item in entity_ids if not scope_entities or item in scope_entities]
        if not entity_ids:
            continue

        basis_fact_ids = _unique(
            gap.get("basis_fact_ids")
            or gap.get("fact_ids")
            or source_spec.get("basis_fact_ids")
            or (candidate or {}).get("fact_ids")
        )
        allowed_tools = _unique(gap.get("allowed_tools") or source_spec.get("allowed_tools") or [])
        allowed_tools = [item for item in allowed_tools if not scope_tools or item in scope_tools]
        time_range = gap.get("time_range") or source_spec.get("time_range") or (candidate or {}).get("time_range")
        if time_range is not None:
            time_range = tuple(str(item) for item in time_range)
            if len(time_range) != 2:
                continue

        question = str(
            gap.get("question")
            or gap.get("reason")
            or gap.get("missing_fact")
            or "补充核验当前候选事件缺失的证据"
        ).strip()
        if not question:
            continue
        payload = {
            "source": "verification_gap",
            "gap": gap,
            "candidate_event_id": (candidate or {}).get("event_id"),
        }
        # A public synthetic fixture may describe what the next verification
        # should return.  Production verifiers can ignore this field and
        # produce their own structured result.
        if isinstance(gap.get("recheck_result"), Mapping):
            payload["verification"] = dict(gap["recheck_result"])
        task_id = _task_id(entity_ids, basis_fact_ids, time_range, question)
        if task_id in seen:
            continue
        seen.add(task_id)
        tasks.append(
            InvestigationTask(
                task_id=task_id,
                question=question,
                entity_ids=tuple(entity_ids),
                time_range=time_range,
                basis_fact_ids=tuple(basis_fact_ids),
                allowed_tools=tuple(allowed_tools),
                payload=payload,
            )
        )
        if len(tasks) >= max_tasks:
            break
    return tasks


def has_actionable_gaps(state: Mapping[str, Any]) -> bool:
    return bool(build_recheck_tasks(state, max_tasks=1))


def _unique(values: Any) -> list[str]:
    if values is None or isinstance(values, (str, bytes)):
        values = [values]
    result: list[str] = []
    for value in values:
        if value is None:
            continue
        text = str(value).strip()
        if text and text not in result:
            result.append(text)
    return result


def _task_id(
    entity_ids: Sequence[str],
    fact_ids: Sequence[str],
    time_range: Sequence[str] | None,
    question: str,
) -> str:
    payload = {
        "entity_ids": sorted(entity_ids),
        "fact_ids": sorted(fact_ids),
        "time_range": list(time_range) if time_range else None,
        "question": question,
    }
    digest = hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]
    return f"recheck-{digest}"

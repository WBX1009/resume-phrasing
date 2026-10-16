from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from .stable_ids import stable_id


TERMINAL_STATUSES = {"enhanced", "weakened", "unresolved", "skipped"}


def is_incomplete_tool_output(output: dict[str, Any] | None) -> bool:
    output = output or {}
    return output.get("truncated") is True or output.get("aggregation_complete") is False


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def make_hypothesis(
    index: int,
    entity_name: str | None,
    statement: str,
    basis_fact_ids: list[str],
    source: str,
    target_model_ids: list[int] | None = None,
) -> dict[str, Any]:
    return {
        "hypothesis_id": f"hyp_{index:02d}",
        "entity_name": entity_name,
        "statement": statement,
        "basis_fact_ids": basis_fact_ids,
        "target_model_ids": target_model_ids or [],
        "status": "candidate",
        "source": source,
        "created_at": utc_now(),
    }


def make_planner_decision(
    hypothesis_id: str,
    round_index: int,
    tool_name: str,
    params: dict[str, Any],
    reason: str,
    source: str,
) -> dict[str, Any]:
    return {
        "decision_id": f"{hypothesis_id}_r{round_index:02d}_{tool_name}",
        "hypothesis_id": hypothesis_id,
        "round": round_index,
        "tool_name": tool_name,
        "params": params,
        "reason": reason,
        "source": source,
        "created_at": utc_now(),
    }


def make_supplement_fact(
    index: int,
    hypothesis_id: str,
    decision: dict[str, Any],
    output: dict[str, Any],
) -> dict[str, Any]:
    stable_key = stable_id(
        "sf",
        {
            "hypothesis_id": hypothesis_id,
            "decision_id": decision.get("decision_id"),
            "tool_name": decision.get("tool_name"),
            "params": decision.get("params", {}),
            "request_fingerprint": output.get("request_fingerprint"),
            "fact_id": output.get("fact_id"),
        },
    )
    return {
        "supplement_fact_id": stable_key,
        "stable_key": stable_key,
        "sequence_index": index,
        "hypothesis_id": hypothesis_id,
        "decision_id": decision.get("decision_id"),
        "tool_name": decision.get("tool_name"),
        "params": decision.get("params", {}),
        "output": output,
        "fact_refs": output.get("fact_refs", []),
        "evidence_refs": output.get("evidence_refs", []),
        "created_at": utc_now(),
    }


def make_chain_step(
    hypothesis_id: str,
    round_index: int,
    decision: dict[str, Any],
    supplement_fact: dict[str, Any] | None,
    status: str,
    assessment: str,
) -> dict[str, Any]:
    if status not in TERMINAL_STATUSES and status != "running":
        status = "unresolved"
    return {
        "hypothesis_id": hypothesis_id,
        "round": round_index,
        "decision": decision,
        "supplement_fact_id": supplement_fact.get("supplement_fact_id") if supplement_fact else None,
        "status": status,
        "assessment": assessment,
        "created_at": utc_now(),
    }


def build_evidence_chains(
    hypotheses: list[dict[str, Any]],
    chain_steps: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    steps_by_hyp: dict[str, list[dict[str, Any]]] = {}
    for step in chain_steps:
        steps_by_hyp.setdefault(str(step.get("hypothesis_id")), []).append(step)

    chains = []
    for hypothesis in hypotheses:
        hypothesis_id = str(hypothesis.get("hypothesis_id"))
        steps = sorted(steps_by_hyp.get(hypothesis_id, []), key=lambda item: int(item.get("round") or 0))
        final_status = hypothesis.get("status", "candidate")
        if steps:
            final_status = steps[-1].get("status", "unresolved")
        if final_status == "candidate":
            final_status = "unresolved"
        chains.append(
            {
                "chain_id": f"chain_{hypothesis_id}",
                "hypothesis": hypothesis,
                "steps": steps,
                "final_status": final_status,
                "updated_at": utc_now(),
            }
        )
    return chains

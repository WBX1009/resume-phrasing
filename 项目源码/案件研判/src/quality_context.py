"""Progressive, reference-preserving context packing for advisory LLM calls."""

from __future__ import annotations

import json
from typing import Any, Iterable

from .quality_models import Fact, ModelProfile, Observation


def pack_context(
    profile: ModelProfile,
    facts: Iterable[Fact],
    observations: Iterable[Observation],
    max_chars: int = 8_000,
) -> dict[str, Any]:
    if max_chars < 1:
        raise ValueError("max_chars must be positive")
    facts = list(facts)
    observations = list(observations)
    raw_payload = {
        "model_id": profile.model_id,
        "description": profile.description,
        "facts": [fact.to_dict() for fact in facts],
        "observations": [observation.to_dict() for observation in observations],
    }
    raw_text = json.dumps(raw_payload, ensure_ascii=False, sort_keys=True)
    header = {
        "model_id": profile.model_id,
        "strategy_key": profile.strategy_key,
        "reasoning_mode": profile.reasoning_mode,
        "allowed_tools": list(profile.specialized_tools),
    }
    lines = [json.dumps(header, ensure_ascii=False, sort_keys=True)]
    omitted: list[str] = []
    retained: list[str] = []
    fact_summaries: list[dict[str, Any]] = []
    omitted_observations: list[str] = []
    fact_references = [f"[fact_id={fact.fact_id}]" for fact in facts]
    observation_subset = observations[:5]
    observation_references = [
        json.dumps({"observation_id": observation.observation_id}, ensure_ascii=False)
        for observation in observation_subset
    ]

    def joined_length(extra: list[str]) -> int:
        return len("\n".join([*lines, *extra]))

    for fact in facts:
        compact = json.dumps(
            {"fact_id": fact.fact_id, "kind": fact.kind, "subject": fact.subject, "value": fact.value},
            ensure_ascii=False,
            sort_keys=True,
        )
        reference = fact_references[len(retained)]
        candidate = reference + compact
        remaining_minimum = fact_references[len(retained) + 1 :] + observation_references
        if joined_length([candidate, *remaining_minimum]) <= max_chars:
            lines.append(candidate)
            retained.append(fact.fact_id)
            fact_summaries.append({"fact_id": fact.fact_id, "kind": fact.kind, "subject": fact.subject, "value": fact.value})
        else:
            # Preserve the stable identifier even when the value is omitted.
            # If the descriptive omission would itself exceed the budget,
            # reduce it to the reference token before declaring overflow.
            omitted_candidate = reference + json.dumps(
                {"kind": fact.kind, "subject": fact.subject, "value": "<omitted>"},
                ensure_ascii=False,
            )
            if joined_length([omitted_candidate, *remaining_minimum]) <= max_chars:
                lines.append(omitted_candidate)
            else:
                lines.append(reference)
            retained.append(fact.fact_id)
            omitted.append(fact.fact_id)
            fact_summaries.append({"fact_id": fact.fact_id, "kind": fact.kind, "subject": fact.subject, "value": "<omitted>"})
    for index, observation in enumerate(observation_subset):
        observation_candidate = json.dumps(
            {"observation_id": observation.observation_id, "tool": observation.tool_name, "status": observation.status},
            ensure_ascii=False,
        )
        remaining_minimum = observation_references[index + 1 :]
        if joined_length([observation_candidate, *remaining_minimum]) <= max_chars:
            lines.append(observation_candidate)
            continue
        observation_reference = json.dumps(
            {"observation_id": observation.observation_id, "status": observation.status},
            ensure_ascii=False,
        )
        if joined_length([observation_reference, *remaining_minimum]) <= max_chars:
            lines.append(observation_reference)
        else:
            lines.append(observation_references[index])
        omitted_observations.append(observation.observation_id)
    text = "\n".join(lines)
    return {
        "text": text,
        "raw_chars": len(raw_text),
        "packed_chars": len(text),
        "max_chars": max_chars,
        "budget_exceeded": len(text) > max_chars,
        "compression_ratio": round(len(text) / max(1, len(raw_text)), 6),
        "fact_ids": retained,
        "fact_summaries": fact_summaries,
        "omitted_value_fact_ids": omitted,
        "observation_ids": [observation.observation_id for observation in observations[:5]],
        "omitted_observation_ids": omitted_observations,
    }

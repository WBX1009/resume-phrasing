"""Validate, merge and select both candidate channels before planning."""
from __future__ import annotations

from dataclasses import dataclass
from .case_execution import PlanError
from .investigation_contracts import Candidate, authorize_bounds, validate_fact_references


@dataclass(frozen=True)
class CandidateEventBuilder:
    max_events: int = 100

    def __post_init__(self):
        if not isinstance(self.max_events, int) or isinstance(self.max_events, bool) or self.max_events <= 0:
            raise ValueError("max_events must be a positive integer")

    def build(self, rule_clues, model_clues, *, scope, facts):
        merged = {}
        rejected = []
        for source, clues in (("rule", rule_clues), ("llm", model_clues)):
            for raw in clues:
                try:
                    candidate = Candidate.model_validate(raw)
                    authorize_bounds(candidate.subject_ids, candidate.time_range, scope)
                    validate_fact_references(candidate.fact_ids, facts, candidate.subject_ids)
                    candidate.source = [source]
                    event = candidate.model_dump(mode="json")
                    old = merged.get(candidate.event_id)
                    if old:
                        old["fact_ids"] = sorted(set(old["fact_ids"]) | set(event["fact_ids"]))
                        old["source"] = sorted(set(old["source"]) | {source})
                        old["priority"] = max(old["priority"], event["priority"])
                    else:
                        merged[candidate.event_id] = event
                except ValueError as exc:
                    rejected.append({"source": source, "reason": type(exc).__name__})
        # Stable bounded selection; a noisy channel cannot crash the case or
        # permanently exclude the other channel when both have valid hits.
        ranked = sorted(merged.values(), key=lambda e: (-e["priority"], e["event_id"]))
        selected = []
        for source in ("rule", "llm"):
            first = next((e for e in ranked if source in e["source"]), None)
            if first and first not in selected and len(selected) < self.max_events:
                selected.append(first)
        for event in ranked:
            if len(selected) >= self.max_events:
                break
            if event not in selected:
                selected.append(event)
        return {"events": selected, "rejected": rejected,
                "omitted_count": len(ranked)-len(selected), "valid_count": len(ranked)}

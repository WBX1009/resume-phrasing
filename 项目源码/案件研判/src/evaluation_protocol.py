"""Local baseline/full/ablation protocol for event-level comparisons.

This is an execution and metadata guard, not a source of production metrics.
It makes it difficult to compare strategy variants under different datasets,
matching rules or budgets, while leaving actual business labels outside the
public repository.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .event_evaluation import EventMatchPolicy, evaluate_events
from .langsmith_event_evaluation import compare_experiment_metadata


@dataclass(frozen=True)
class EvaluationVariant:
    name: str
    strategy_version: str
    disabled_components: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.name.strip() or not self.strategy_version.strip():
            raise ValueError("variant name and strategy_version must be non-empty")
        if len(self.disabled_components) != len(set(self.disabled_components)):
            raise ValueError("disabled_components must be unique")


@dataclass(frozen=True)
class EvaluationProtocol:
    dataset_version: str
    policy: EventMatchPolicy
    candidate_limit: int
    llm_call_budget: int
    tool_call_budget: int
    token_budget: int
    protocol_version: str = "case-event-v1"
    code_revision: str = "working-tree"
    model_revision: str = "unspecified"
    prompt_revision: str = "unspecified"

    def __post_init__(self) -> None:
        if not self.dataset_version.strip():
            raise ValueError("dataset_version must be non-empty")
        for name in ("candidate_limit", "llm_call_budget", "tool_call_budget", "token_budget"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")

    def metadata(self, variant: EvaluationVariant) -> dict[str, Any]:
        return {
            "protocol_version": self.protocol_version,
            "dataset_version": self.dataset_version,
            "matching_policy_version": self.policy.version,
            "candidate_limit": self.candidate_limit,
            "llm_call_budget": self.llm_call_budget,
            "tool_call_budget": self.tool_call_budget,
            "token_budget": self.token_budget,
            "code_revision": self.code_revision,
            "model_revision": self.model_revision,
            "prompt_revision": self.prompt_revision,
            "strategy_version": variant.strategy_version,
            "variant_name": variant.name,
            "disabled_components": list(variant.disabled_components),
        }

    def validate_variants(self, variants: Sequence[EvaluationVariant]) -> None:
        if not variants:
            raise ValueError("at least one evaluation variant is required")
        names = [variant.name for variant in variants]
        if len(names) != len(set(names)):
            raise ValueError("evaluation variant names must be unique")
        metadata = [self.metadata(variant) for variant in variants]
        for current in metadata[1:]:
            compare_experiment_metadata(metadata[0], current)
        for variant in variants:
            if variant.name.lower().startswith("ablation") and len(variant.disabled_components) != 1:
                raise ValueError("each ablation variant must disable exactly one component")

    def evaluate_outputs(
        self,
        variants: Sequence[EvaluationVariant],
        outputs_by_variant: Mapping[str, Mapping[str, Any]],
        gold_events: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any]:
        self.validate_variants(variants)
        result: dict[str, Any] = {
            "protocol_version": self.protocol_version,
            "dataset_version": self.dataset_version,
            "matching_policy_version": self.policy.version,
            "variants": {},
        }
        for variant in variants:
            output = outputs_by_variant.get(variant.name)
            if not isinstance(output, Mapping):
                raise ValueError(f"missing outputs for variant {variant.name!r}")
            expected_limits = {
                "candidate_limit": self.candidate_limit,
                "llm_call_budget": self.llm_call_budget,
                "tool_call_budget": self.tool_call_budget,
                "token_budget": self.token_budget,
            }
            if output.get("limits") != expected_limits:
                raise ValueError(f"variant {variant.name!r} does not use frozen limits")
            usage = output.get("usage")
            if not isinstance(usage, Mapping):
                raise ValueError(f"variant {variant.name!r} must report actual usage")
            for usage_key, limit_key in (
                ("llm_calls", "llm_call_budget"),
                ("tool_calls", "tool_call_budget"),
                ("reserved_tokens", "token_budget"),
            ):
                amount = usage.get(usage_key)
                if not isinstance(amount, int) or isinstance(amount, bool) or amount < 0:
                    raise ValueError(f"variant {variant.name!r} usage {usage_key} must be non-negative")
                if amount > getattr(self, limit_key):
                    raise ValueError(f"variant {variant.name!r} exceeded {limit_key}")
            candidate_events = output.get("candidate_events", [])
            final_events = output.get("final_events", [])
            if len(candidate_events) > self.candidate_limit:
                raise ValueError(f"variant {variant.name!r} exceeded candidate_limit")
            candidate = evaluate_events(candidate_events, gold_events, self.policy)
            final = evaluate_events(final_events, gold_events, self.policy)
            result["variants"][variant.name] = {
                "metadata": self.metadata(variant),
                "candidate": candidate.to_dict(),
                "final": final.to_dict(),
            }
        return result

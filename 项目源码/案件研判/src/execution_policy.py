from __future__ import annotations

from enum import Enum


class FallbackPolicy(str, Enum):
    STRICT_DEMO = "strict_demo"
    REVIEW_SAFE = "review_safe"
    FACTS_ONLY = "facts_only"


def normalize_fallback_policy(value: str | FallbackPolicy | None) -> FallbackPolicy:
    if isinstance(value, FallbackPolicy):
        return value
    try:
        return FallbackPolicy(str(value or FallbackPolicy.REVIEW_SAFE.value))
    except ValueError as exc:
        raise ValueError(f"unsupported fallback policy: {value}") from exc

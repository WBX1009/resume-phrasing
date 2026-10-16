"""Conservative cross-model evidence linking."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable

from .quality_models import Claim, Fact
from .quality_operators import stable_id


def _normalise(value: Any) -> str:
    return " ".join(str(value).strip().lower().split())


def build_cross_model_links(facts: Iterable[Fact]) -> tuple[list[dict[str, Any]], list[Claim]]:
    """Join only explicit entity examples from independent model facts.

    This is intentionally conservative: the output is a candidate link, not
    a conclusion. Numeric-only identifiers and very short strings are ignored
    to reduce accidental joins in tiny samples.
    """
    index: dict[str, list[Fact]] = defaultdict(list)
    for fact in facts:
        if fact.kind != "entity_cardinality" or not isinstance(fact.value, dict):
            continue
        for example in fact.value.get("examples", []):
            value = _normalise(example)
            if len(value) < 3 or value.isdigit():
                continue
            index[value].append(fact)

    links: list[dict[str, Any]] = []
    claims: list[Claim] = []
    for value, value_facts in sorted(index.items()):
        unique_models = sorted({fact.model_id for fact in value_facts})
        if len(unique_models) < 2:
            continue
        chosen: list[Fact] = []
        seen_models: set[int] = set()
        for fact in value_facts:
            if fact.model_id in seen_models:
                continue
            chosen.append(fact)
            seen_models.add(fact.model_id)
        fact_ids = [fact.fact_id for fact in chosen]
        link_id = stable_id("cross_link", value, *fact_ids)
        links.append(
            {
                "link_id": link_id,
                "entity_value": value,
                "model_ids": sorted(seen_models),
                "fact_ids": fact_ids,
                "status": "candidate_needs_verification",
            }
        )
        claims.append(
            Claim(
                claim_id=stable_id("claim", link_id),
                level="inference",
                text=f"不同模型结果中均出现实体线索“{value}”，可作为跨模型复核入口，不代表关系已经成立。",
                model_ids=sorted(seen_models),
                fact_ids=fact_ids,
                status="downgraded",
                reason="cross_model_candidate_requires_human_verification",
            )
        )
        if len(links) >= 100:
            break
    return links, claims

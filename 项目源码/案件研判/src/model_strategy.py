from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .config import ROOT_DIR


DEFAULT_STRATEGY_PATH = ROOT_DIR / "configs" / "example_strategy.yaml"


@dataclass(frozen=True)
class StrategyBundle:
    case_id: int
    sample_limit: int
    top_entity_limit: int
    top_relation_limit: int
    evidence_per_item: int
    observation_model_ids: list[int]
    entity_dictionary_model_id: int
    thresholds: dict[str, Any]
    models: list[dict[str, Any]]
    by_model_id: dict[int, dict[str, Any]]

    @property
    def model_ids(self) -> list[int]:
        return [int(item["model_id"]) for item in self.models]


def load_strategy(path: Path | None = None) -> StrategyBundle:
    strategy_path = path or DEFAULT_STRATEGY_PATH
    payload = yaml.safe_load(strategy_path.read_text(encoding="utf-8"))
    models = payload.get("models") or []
    by_model_id = {int(item["model_id"]): item for item in models}
    return StrategyBundle(
        case_id=int(payload.get("case_id", 1)),
        sample_limit=int(payload.get("sample_limit", 200)),
        top_entity_limit=int(payload.get("top_entity_limit", 5)),
        top_relation_limit=int(payload.get("top_relation_limit", 5)),
        evidence_per_item=int(payload.get("evidence_per_item", 3)),
        observation_model_ids=[int(item) for item in payload.get("observation_model_ids", [])],
        entity_dictionary_model_id=int(payload.get("entity_dictionary_model_id", 98)),
        thresholds=dict(payload.get("thresholds") or {}),
        models=models,
        by_model_id=by_model_id,
    )


def strategy_for(bundle: StrategyBundle, model_id: int) -> dict[str, Any]:
    return bundle.by_model_id.get(int(model_id), {"model_id": model_id, "model_kind": "unknown"})

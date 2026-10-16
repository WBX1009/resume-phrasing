"""Domain-specific reasoning policies with an offline deterministic backend.

The policy layer mirrors the concepts learned from HelloAgents without taking
that project as a runtime dependency.  A future LLM adapter can implement the
same `choose_actions` contract; the evidence ledger and validators remain
authoritative either way.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Iterable, Protocol

from .quality_models import Claim, Fact, ModelProfile, Observation, TraceEvent
from .quality_operators import fact_index, stable_id


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class ToolSpec:
    name: str
    families: tuple[str, ...]
    description: str
    handler: Callable[..., dict[str, Any]]


class ToolRegistry:
    def __init__(self, specs: Iterable[ToolSpec] | None = None) -> None:
        self._specs = {spec.name: spec for spec in (specs or _default_specs())}

    def names(self) -> list[str]:
        return sorted(self._specs)

    def execute(
        self,
        name: str,
        profile: ModelProfile,
        rows: list[dict[str, Any]],
        facts: list[Fact],
        case_id: int | str,
    ) -> Observation:
        spec = self._specs.get(name)
        observation_id = stable_id("observation", profile.model_id, case_id, name)
        if spec is None:
            return Observation(
                observation_id=observation_id,
                tool_name=name,
                status="blocked",
                input={"model_id": profile.model_id, "case_id": case_id},
                output={},
                reason="unknown_tool",
            )
        if profile.family not in spec.families:
            return Observation(
                observation_id=observation_id,
                tool_name=name,
                status="blocked",
                input={"model_id": profile.model_id, "case_id": case_id},
                output={},
                reason="tool_not_allowed_for_family",
            )
        try:
            output = spec.handler(rows=rows, facts=facts)
            status = "ok" if output.get("count", output.get("edge_count", 0)) else "empty"
            return Observation(
                observation_id=observation_id,
                tool_name=name,
                status=status,
                input={"model_id": profile.model_id, "case_id": case_id},
                output=output,
                fact_ids=[fact.fact_id for fact in facts if fact.kind in output.get("supports", [])],
            )
        except Exception as exc:  # defensive boundary for one bad tool
            return Observation(
                observation_id=observation_id,
                tool_name=name,
                status="error",
                input={"model_id": profile.model_id, "case_id": case_id},
                output={},
                reason=f"{type(exc).__name__}: {exc}",
            )


def _top_entities(*, rows: list[dict[str, Any]], facts: list[Fact]) -> dict[str, Any]:
    cardinalities = [fact for fact in facts if fact.kind == "entity_cardinality"]
    return {
        "count": len(cardinalities),
        "entities": [
            {"field": fact.subject, **(fact.value if isinstance(fact.value, dict) else {"value": fact.value})}
            for fact in cardinalities[:8]
        ],
        "supports": ["entity_cardinality"],
    }


def _temporal_window_summary(*, rows: list[dict[str, Any]], facts: list[Fact]) -> dict[str, Any]:
    temporal = [fact for fact in facts if fact.kind in {"temporal_summary", "sequence_summary"}]
    return {
        "count": len(temporal),
        "windows": [{"subject": fact.subject, "value": fact.value} for fact in temporal[:5]],
        "supports": [fact.kind for fact in temporal],
    }


def _overlap_summary(*, rows: list[dict[str, Any]], facts: list[Fact]) -> dict[str, Any]:
    values_by_field: dict[str, set[str]] = {}
    for row in rows:
        data = row.get("data", {})
        if not isinstance(data, dict):
            continue
        for key, value in data.items():
            if isinstance(value, list):
                values_by_field.setdefault(key, set()).update(str(item) for item in value)
            elif isinstance(value, str) and ("列表" in key or "联系人" in key or "账号" in key):
                values_by_field.setdefault(key, set()).update(item.strip() for item in value.split(",") if item.strip())
    pairs: list[dict[str, Any]] = []
    fields = sorted(values_by_field)
    for index, left in enumerate(fields):
        for right in fields[index + 1 :]:
            overlap = values_by_field[left] & values_by_field[right]
            if overlap:
                pairs.append({"left": left, "right": right, "overlap_count": len(overlap)})
    return {"count": len(pairs), "pairs": pairs[:10], "supports": ["entity_cardinality"]}


def _same_amount_window_summary(*, rows: list[dict[str, Any]], facts: list[Fact]) -> dict[str, Any]:
    amount_values: dict[float, int] = {}
    for row in rows:
        data = row.get("data", {})
        if not isinstance(data, dict):
            continue
        for key, value in data.items():
            if "金额" not in key and "余额" not in key:
                continue
            try:
                numeric = float(str(value).replace(",", ""))
            except (TypeError, ValueError):
                continue
            amount_values[round(numeric, 2)] = amount_values.get(round(numeric, 2), 0) + 1
    repeated = [
        {"amount": amount, "count": count}
        for amount, count in sorted(amount_values.items(), key=lambda item: (-item[1], item[0]))
        if count > 1
    ]
    return {"count": len(repeated), "repeated_amounts": repeated[:10], "supports": ["numeric_summary"]}


def _threshold_event_summary(*, rows: list[dict[str, Any]], facts: list[Fact]) -> dict[str, Any]:
    numeric = [fact for fact in facts if fact.kind == "numeric_summary"]
    events = []
    for fact in numeric:
        value = fact.value if isinstance(fact.value, dict) else {}
        if value.get("max", 0) >= 50000:
            events.append({"field": fact.subject, "max": value["max"], "count": value.get("count", 0)})
    return {"count": len(events), "events": events[:10], "supports": ["numeric_summary"]}


def _edge_summary(*, rows: list[dict[str, Any]], facts: list[Fact]) -> dict[str, Any]:
    relation = [fact for fact in facts if fact.kind == "relation_summary"]
    return {
        "edge_count": sum((fact.value or {}).get("edge_count", 0) for fact in relation),
        "edges": [fact.value for fact in relation[:5]],
        "count": len(relation),
        "supports": ["relation_summary"],
    }


def _graph_neighbor_summary(*, rows: list[dict[str, Any]], facts: list[Fact]) -> dict[str, Any]:
    relation = [fact for fact in facts if fact.kind == "relation_summary"]
    neighbors: set[str] = set()
    for fact in relation:
        for edge in (fact.value or {}).get("top_edges", []):
            neighbors.add(str(edge.get("source")))
            neighbors.add(str(edge.get("target")))
    neighbors.discard("None")
    return {"count": len(neighbors), "neighbors": sorted(neighbors)[:20], "supports": ["relation_summary"]}


def _default_specs() -> list[ToolSpec]:
    all_families = tuple({
        "account_profile",
        "communication_temporal",
        "contact_device_relation",
        "distribution_keyword",
        "threshold_event",
        "temporal_sequence",
        "graph_cross_domain",
    })
    return [
        ToolSpec("top_entities", all_families, "Summarise deterministic entity cardinality facts", _top_entities),
        ToolSpec("temporal_window_summary", ("communication_temporal", "temporal_sequence"), "Inspect temporal facts", _temporal_window_summary),
        ToolSpec("overlap_summary", ("contact_device_relation",), "Compare set-like contact/device fields", _overlap_summary),
        ToolSpec("same_amount_window_summary", ("temporal_sequence",), "Find repeated transaction amounts", _same_amount_window_summary),
        ToolSpec("threshold_event_summary", ("threshold_event",), "Summarise explicit high-value numeric events", _threshold_event_summary),
        ToolSpec("edge_summary", ("graph_cross_domain",), "Summarise relation edges", _edge_summary),
        ToolSpec("graph_neighbor_summary", ("graph_cross_domain",), "Summarise graph neighbors", _graph_neighbor_summary),
    ]


class Policy(Protocol):
    name: str

    def run(
        self,
        profile: ModelProfile,
        rows: list[dict[str, Any]],
        facts: list[Fact],
        case_id: int | str,
        tools: ToolRegistry,
        max_steps: int,
    ) -> tuple[list[Observation], list[TraceEvent]]: ...


def _trace(event_type: str, profile: ModelProfile, case_id: int | str, step: int, payload: dict[str, Any]) -> TraceEvent:
    return TraceEvent(
        event_id=stable_id("trace", profile.model_id, case_id, step, event_type, payload),
        event_type=event_type,
        timestamp=_now(),
        step=step,
        payload=payload,
    )


class DeterministicPolicy:
    name = "deterministic"

    def run(self, profile, rows, facts, case_id, tools, max_steps):
        return [], [_trace("policy_selected", profile, case_id, 0, {"policy": self.name})]


class ReActPolicy:
    name = "react"

    def run(self, profile, rows, facts, case_id, tools, max_steps):
        observations: list[Observation] = []
        trace = [_trace("policy_selected", profile, case_id, 0, {"policy": self.name})]
        # The offline policy makes the same bounded choice a live planner would
        # make after seeing the strategy profile.  No hidden free-form thought
        # is persisted or rendered into the report.
        for step, name in enumerate(profile.specialized_tools[:max_steps], start=1):
            trace.append(_trace("action", profile, case_id, step, {"tool": name}))
            observation = tools.execute(name, profile, rows, facts, case_id)
            observations.append(observation)
            trace.append(_trace("observation", profile, case_id, step, {"tool": name, "status": observation.status}))
        trace.append(_trace("stop", profile, case_id, len(observations) + 1, {"reason": "bounded_policy"}))
        return observations, trace


class PlanSolvePolicy(ReActPolicy):
    name = "plan_solve"

    def run(self, profile, rows, facts, case_id, tools, max_steps):
        observations: list[Observation] = []
        trace = [_trace("policy_selected", profile, case_id, 0, {"policy": self.name})]
        plan = [
            {"step": index, "tool": name, "depends_on": [index - 1] if index > 1 else []}
            for index, name in enumerate(profile.specialized_tools[:max_steps], start=1)
        ]
        trace.append(_trace("plan_created", profile, case_id, 1, {"steps": plan}))
        for item in plan:
            name = item["tool"]
            observation = tools.execute(name, profile, rows, facts, case_id)
            observations.append(observation)
            trace.append(_trace("plan_step", profile, case_id, item["step"], {"tool": name, "status": observation.status}))
        trace.append(_trace("stop", profile, case_id, len(plan) + 1, {"reason": "plan_complete"}))
        return observations, trace


class PlanReactReflectPolicy(PlanSolvePolicy):
    name = "plan_react_reflect"

    def run(self, profile, rows, facts, case_id, tools, max_steps):
        observations, trace = super().run(profile, rows, facts, case_id, tools, max_steps)
        fact_ids = set(fact_index(facts))
        orphaned = sorted({fact_id for observation in observations for fact_id in observation.fact_ids if fact_id not in fact_ids})
        trace.append(
            _trace(
                "reflection",
                profile,
                case_id,
                len(trace) + 1,
                {"orphaned_fact_ids": orphaned, "decision": "accept" if not orphaned else "downgrade"},
            )
        )
        return observations, trace


POLICIES: dict[str, Policy] = {
    "deterministic": DeterministicPolicy(),
    "react": ReActPolicy(),
    "plan_solve": PlanSolvePolicy(),
    "plan_react_reflect": PlanReactReflectPolicy(),
}


def run_policy(
    profile: ModelProfile,
    rows: list[dict[str, Any]],
    facts: list[Fact],
    case_id: int | str,
    tools: ToolRegistry | None = None,
    max_steps: int = 3,
) -> tuple[list[Observation], list[TraceEvent]]:
    policy = POLICIES[profile.reasoning_mode]
    return policy.run(profile, rows, facts, case_id, tools or ToolRegistry(), max_steps)


def build_claims(
    profile: ModelProfile,
    facts: list[Fact],
    observations: list[Observation],
) -> list[Claim]:
    claims: list[Claim] = []
    for fact in facts:
        if fact.kind == "empty_result":
            claims.append(
                Claim(
                    claim_id=stable_id("claim", fact.fact_id),
                    level="unresolved",
                    text=f"模型 {profile.model_id} 当前没有可用结果，无法形成研判结论。",
                    model_ids=[profile.model_id],
                    fact_ids=[fact.fact_id],
                    status="downgraded",
                    reason="empty_or_incomplete_result",
                )
            )
            continue
        if fact.kind == "row_count":
            count = (fact.value or {}).get("count", 0)
            claims.append(
                Claim(
                    claim_id=stable_id("claim", fact.fact_id),
                    level="fact",
                    text=f"模型 {profile.model_id} 在当前快照中提供 {count} 条结果记录。",
                    model_ids=[profile.model_id],
                    fact_ids=[fact.fact_id],
                )
            )
            break
    for observation in observations:
        if observation.status != "ok" or not observation.fact_ids:
            continue
        claims.append(
            Claim(
                claim_id=stable_id("claim", observation.observation_id),
                level="rule_hit",
                text=f"工具 {observation.tool_name} 完成了受控补充分析；结果仅表示规则/统计命中，仍需人工核验。",
                model_ids=[profile.model_id],
                fact_ids=observation.fact_ids,
                observation_ids=[observation.observation_id],
            )
        )
    return claims


def validate_claims(claims: Iterable[Claim], facts: Iterable[Fact]) -> dict[str, Any]:
    facts = list(facts)
    claims = list(claims)
    fact_ids = {fact.fact_id for fact in facts}
    errors: list[str] = []
    checked = 0
    seen_fact_ids: set[str] = set()
    for fact in facts:
        if fact.fact_id in seen_fact_ids:
            errors.append(f"duplicate fact id {fact.fact_id}")
        seen_fact_ids.add(fact.fact_id)
    seen_claim_ids: set[str] = set()
    for claim in claims:
        checked += 1
        if claim.claim_id in seen_claim_ids:
            errors.append(f"duplicate claim id {claim.claim_id}")
        seen_claim_ids.add(claim.claim_id)
        missing = sorted(set(claim.fact_ids) - fact_ids)
        if missing:
            errors.append(f"{claim.claim_id}: missing fact ids {missing}")
    return {"valid": not errors, "checked_claims": checked, "errors": errors}

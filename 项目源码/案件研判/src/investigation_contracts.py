"""JSON contracts shared by the case graph, model adapter and API.

Models propose decisions; only the repository and registered tools create facts.
"""
from __future__ import annotations

from hashlib import sha256
import json
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .case_execution import CaseTaskScope, PlanError, _parse_time, _ensure_comparable


def stable_id(prefix: str, value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False, separators=(",", ":"))
    return f"{prefix}-{sha256(encoded.encode()).hexdigest()[:24]}"


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class EvidenceFact(Contract):
    fact_id: str = Field(min_length=1)
    entity_ids: list[str] = Field(min_length=1)
    time_range: tuple[str, str] | None = None
    kind: str = Field(min_length=1)
    summary: str = Field(min_length=1, max_length=2000)
    content: dict[str, Any]
    provenance: dict[str, Any]
    complete: bool = True


class Candidate(Contract):
    event_type: str = Field(min_length=1, max_length=120)
    subject_ids: list[str] = Field(min_length=1, max_length=30)
    fact_ids: list[str] = Field(min_length=1, max_length=100)
    time_range: tuple[str, str] | None = None
    rationale: str = Field(min_length=1, max_length=2000)
    priority: int = Field(default=0, ge=0, le=10)
    event_id: str = ""
    source: list[Literal["rule", "llm"]] = Field(default_factory=list)
    status: Literal["candidate"] = "candidate"

    @model_validator(mode="after")
    def canonical_identity(self):
        self.subject_ids = sorted(set(self.subject_ids))
        self.fact_ids = sorted(set(self.fact_ids))
        # Ignore model supplied identities: dissimilar events cannot collide.
        self.event_id = stable_id("candidate", [self.event_type, self.subject_ids, self.time_range])
        return self


class Requirement(Contract):
    requirement_id: str = Field(min_length=1, max_length=120)
    question: str = Field(min_length=1, max_length=2000)


class InvestigationGoal(Contract):
    goal_id: str = Field(min_length=1)
    candidate_id: str = Field(min_length=1)
    requirements: list[Requirement] = Field(min_length=1, max_length=10)

    @model_validator(mode="after")
    def unique_requirements(self):
        keys = [r.requirement_id for r in self.requirements]
        if len(keys) != len(set(keys)):
            raise ValueError("duplicate requirement_id")
        return self


class ToolRequest(Contract):
    tool_name: str = Field(min_length=1)
    entity_ids: list[str] = Field(min_length=1, max_length=30)
    time_range: tuple[str, str] | None = None
    params: dict[str, Any] = Field(default_factory=dict)


class WorkerDecision(Contract):
    action: Literal["tool", "finish"]
    request: ToolRequest | None = None
    reason: str = Field(min_length=1, max_length=2000)

    @model_validator(mode="after")
    def consistent(self):
        if (self.action == "tool") != (self.request is not None):
            raise ValueError("a tool decision requires exactly one request")
        return self


class Assessment(Contract):
    goal_id: str
    requirement_id: str
    status: Literal["supported", "refuted", "unresolved"]
    fact_ids: list[str] = Field(default_factory=list, max_length=100)
    rationale: str = Field(min_length=1, max_length=2000)


class EvidenceGap(Contract):
    gap_id: str
    goal_id: str
    candidate_id: str
    requirement_id: str
    question: str
    basis_fact_ids: list[str]
    source_task_ids: list[str] = Field(default_factory=list)


class TaskSpec(Contract):
    task_id: str = Field(min_length=1)
    question: str = Field(min_length=1, max_length=2000)
    candidate_id: str
    goal_id: str
    entity_ids: list[str] = Field(min_length=1, max_length=30)
    time_range: tuple[str, str]
    basis_fact_ids: list[str] = Field(min_length=1, max_length=100)
    allowed_tools: list[str] = Field(max_length=30)
    depends_on: list[str] = Field(default_factory=list, max_length=100)
    gap_ids: list[str] = Field(default_factory=list, max_length=10)
    source_task_ids: list[str] = Field(default_factory=list, max_length=100)
    payload: dict[str, Any] = Field(default_factory=dict)


class TaskPlan(Contract):
    plan_version: int = Field(default=1, ge=1)
    goals: list[InvestigationGoal] = Field(max_length=1000)
    tasks: list[TaskSpec] = Field(max_length=1000)


class WorkerJudgment(Contract):
    done: bool
    assessments: list[Assessment] = Field(default_factory=list, max_length=10)
    reason: str = Field(min_length=1, max_length=2000)


class CaseReport(Contract):
    case_id: str
    run_id: str
    status: Literal["success", "partial"]
    data_label: str
    runtime: Literal["langgraph-native-v3"]
    snapshot_ref: str
    candidate_events: list[Candidate]
    goals: list[InvestigationGoal]
    tasks: list[TaskSpec]
    task_outcomes: dict[str, dict[str, Any]]
    verification: dict[str, Any]
    fact_ids: list[str]
    discovery: dict[str, Any]
    coverage: dict[str, Any]
    recheck_rounds: int
    budget: dict[str, int]
    evidence_boundary: str


def authorize_bounds(entity_ids, time_range, scope: CaseTaskScope, *, task=None):
    entities = set(entity_ids)
    allowed = set(task.entity_ids) if task is not None else scope.entity_ids
    if not entities or not entities <= set(allowed) or not entities <= scope.entity_ids:
        raise PlanError("entity parameters are outside the authorized scope")
    bounds = task.time_range if task is not None else scope.time_range
    if bounds is None and task is not None:
        bounds = scope.time_range
    if bounds is not None and time_range is None:
        raise PlanError("an explicit bounded time_range is required")
    if time_range is not None:
        if len(time_range) != 2:
            raise PlanError("time_range requires two bounds")
        start, end = [_parse_time(v, "request time_range") for v in time_range]
        _ensure_comparable(start, end, "request time_range")
        if start[0] > end[0]:
            raise PlanError("time_range is reversed")
        if bounds is not None:
            lower, upper = [_parse_time(v, "scope time_range") for v in bounds]
            _ensure_comparable(start, lower, "request/scope time_range")
            if start[0] < lower[0] or end[0] > upper[0]:
                raise PlanError("time parameters are outside the authorized scope")


def validate_fact_references(ids, facts, entities=None):
    if not ids or not set(ids) <= set(facts):
        raise PlanError("unknown or empty fact references")
    if entities is not None:
        covered = set()
        for fid in ids:
            fact_entities = set(facts[fid]["entity_ids"])
            if not fact_entities <= set(entities):
                raise PlanError("fact references expose entities outside the requested scope")
            covered.update(fact_entities)
        if not set(entities) <= covered:
            raise PlanError("fact references do not support the named entities")

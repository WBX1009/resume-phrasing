"""Public data contracts for the evidence-first quality pipeline.

The legacy demo uses a number of loosely shaped dictionaries.  The quality
pipeline deliberately keeps its contracts small and serialisable so that the
same objects can be written to JSON, traced, replayed, and projected into
Markdown/HTML/DOCX without asking an LLM to invent a wire format.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal


EvidenceLevel = Literal["fact", "rule_hit", "inference", "unresolved"]
CoverageStatus = Literal["sampled", "unsupported_no_sample"]
ReasoningMode = Literal[
    "deterministic", "react", "plan_solve", "plan_react_reflect"
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class ModelProfile:
    model_id: int
    name: str
    description: str
    family: str
    coverage: CoverageStatus
    reasoning_mode: ReasoningMode
    observed_fields: tuple[str, ...] = ()
    observed_row_count: int = 0
    observed_case_ids: tuple[int, ...] = ()
    operators: tuple[str, ...] = ()
    specialized_tools: tuple[str, ...] = ()
    required_field_hints: tuple[str, ...] = ()
    strategy_key: str = ""
    report_sections: tuple[str, ...] = ()
    cross_model_dependencies: tuple[int, ...] = ()
    unsupported_reason: str | None = None

    @property
    def is_supported(self) -> bool:
        return self.coverage == "sampled"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Fact:
    fact_id: str
    model_id: int
    case_id: int | str
    kind: str
    subject: str
    value: Any
    source_row_ids: list[int] = field(default_factory=list)
    source_fields: list[str] = field(default_factory=list)
    complete: bool = True
    provenance: str = "deterministic_operator"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Observation:
    observation_id: str
    tool_name: str
    status: Literal["ok", "empty", "blocked", "error"]
    input: dict[str, Any]
    output: dict[str, Any]
    fact_ids: list[str] = field(default_factory=list)
    reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Claim:
    claim_id: str
    level: EvidenceLevel
    text: str
    model_ids: list[int]
    fact_ids: list[str]
    observation_ids: list[str] = field(default_factory=list)
    status: Literal["accepted", "downgraded", "rejected"] = "accepted"
    reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TraceEvent:
    event_id: str
    event_type: str
    timestamp: str
    step: int
    payload: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RunManifest:
    run_id: str
    started_at: str
    finished_at: str | None
    command: str
    case_ids: list[int]
    snapshot_sha256: str
    profile_count: int
    sampled_profile_count: int
    unsupported_profile_count: int
    reasoning_modes: dict[str, int]
    online: bool
    llm_calls: int = 0
    llm_tokens: int = 0
    status: Literal["running", "success", "failed"] = "running"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

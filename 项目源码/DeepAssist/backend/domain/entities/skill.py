from typing import Any, Dict, List

from pydantic import BaseModel, Field, field_validator


class SkillSpec(BaseModel):
    """Versioned prompt/tool/resource bundle selected for an agent task."""

    name: str
    version: str
    description: str = ""
    instructions: str
    trigger_keywords: List[str] = Field(default_factory=list)
    tool_names: List[str] = Field(default_factory=list)
    constraints: List[str] = Field(default_factory=list)
    check_standards: List[str] = Field(default_factory=list)
    resources: Dict[str, Any] = Field(default_factory=dict)
    enabled: bool = True

    @field_validator("name", "version", "instructions")
    @classmethod
    def _required_text(cls, value: str) -> str:
        value = str(value or "").strip()
        if not value:
            raise ValueError("skill name, version, and instructions are required")
        return value

from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class CapabilitySpec(BaseModel):
    """A coarse capability exposed to runtime diagnostics and tool selection."""

    name: str
    description: str = ""
    risk_level: str = "low"
    enabled_by_default: bool = True
    metadata: Dict[str, str] = Field(default_factory=dict)


class ToolGroupSpec(BaseModel):
    """A lightweight grouping of tools without introducing a full skill framework."""

    name: str
    description: str = ""
    tools: List[str] = Field(default_factory=list)
    policy: str = "read_only"
    capability: Optional[str] = None
    enabled_by_default: bool = True

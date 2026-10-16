from __future__ import annotations

import json
import re
from typing import Iterable, List, Optional

from backend.domain.entities.skill import SkillSpec


class SkillRegistry:
    """Register immutable skill versions and select bundles per task."""

    def __init__(self, skills: Iterable[SkillSpec] | None = None):
        self._skills: dict[str, dict[str, SkillSpec]] = {}
        for skill in skills or []:
            self.register(skill)

    def register(self, skill: SkillSpec) -> SkillSpec:
        versions = self._skills.setdefault(skill.name, {})
        versions[skill.version] = skill
        return skill

    def get(self, name: str, version: str | None = None) -> Optional[SkillSpec]:
        versions = self._skills.get(name, {})
        if version:
            return versions.get(version)
        if not versions:
            return None
        return max(versions.values(), key=lambda item: self._version_key(item.version))

    def select_for_task(
        self,
        query: str,
        *,
        tool_names: Iterable[str] | None = None,
    ) -> List[SkillSpec]:
        normalized = str(query or "").casefold()
        available = set(tool_names or [])
        selected: List[SkillSpec] = []
        for name in sorted(self._skills):
            skill = self.get(name)
            if skill is None or not skill.enabled:
                continue
            if skill.trigger_keywords and not any(
                str(keyword).casefold() in normalized for keyword in skill.trigger_keywords
            ):
                continue
            if available and not set(skill.tool_names).issubset(available):
                continue
            selected.append(skill)
        return selected

    def render_for_task(
        self,
        query: str,
        *,
        tool_names: Iterable[str] | None = None,
    ) -> str:
        selected = self.select_for_task(query, tool_names=tool_names)
        if not selected:
            return ""
        blocks = []
        for skill in selected:
            lines = [f"[{skill.name}@{skill.version}]", skill.instructions.strip()]
            if skill.tool_names:
                lines.append("Tools: " + ", ".join(skill.tool_names))
            if skill.constraints:
                lines.append("Constraints: " + "; ".join(skill.constraints))
            if skill.check_standards:
                lines.append("Checks: " + "; ".join(skill.check_standards))
            if skill.resources:
                lines.append(
                    "Resources: "
                    + json.dumps(skill.resources, ensure_ascii=False, sort_keys=True)
                )
            blocks.append("\n".join(lines))
        return "\n\n".join(blocks)

    def manifest(self) -> List[dict]:
        return [
            skill.model_dump()
            for name in sorted(self._skills)
            for skill in sorted(
                self._skills[name].values(),
                key=lambda item: self._version_key(item.version),
            )
        ]

    @staticmethod
    def _version_key(version: str) -> tuple[int, ...]:
        values = [int(item) for item in re.findall(r"\d+", str(version))]
        return tuple(values or [0])


def build_default_skill_registry() -> SkillRegistry:
    return SkillRegistry(
        [
            SkillSpec(
                name="agent_core",
                version="1.0.0",
                description="Core tool-use and evidence discipline.",
                instructions=(
                    "Keep tool calls within the supplied schema and state what each "
                    "observation supports before forming the final answer."
                ),
                constraints=[
                    "Do not claim an observation that is not present in the tool result.",
                ],
                check_standards=[
                    "Check that the final answer is supported by the collected evidence.",
                ],
            ),
            SkillSpec(
                name="knowledge_grounding",
                version="1.0.0",
                description="Knowledge-base answer discipline.",
                trigger_keywords=["知识库", "文档", "引用", "rag", "knowledge"],
                tool_names=["search_knowledge_base"],
                instructions=(
                    "Prefer knowledge-base retrieval for factual claims and retain "
                    "the returned citation markers in the answer."
                ),
                constraints=[
                    "Do not replace a missing citation with an unsupported guess.",
                ],
                check_standards=[
                    "Check that every citation marker refers to a retrieved source.",
                ],
            ),
        ]
    )

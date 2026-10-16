from collections import Counter
from typing import List

from backend.domain.entities.context_window import ContextSummary, ContextTurn


class ConversationSummaryCompressor:
    """Compress dropped turns into one synthetic system message."""

    def __init__(
        self,
        max_turns: int = 4,
        max_chars_per_line: int = 140,
    ):
        self.max_turns = max_turns
        self.max_chars_per_line = max_chars_per_line

    def compress(self, dropped_turns: List[ContextTurn]) -> ContextSummary | None:
        if not dropped_turns:
            return None

        reason_counts: Counter[str] = Counter()
        lines: List[str] = []

        for turn in dropped_turns[: self.max_turns]:
            reason_counts.update(turn.reasons)
            lines.append(self._render_turn(turn))

        remaining_turns = max(0, len(dropped_turns) - self.max_turns)
        if remaining_turns:
            lines.append(f"- ... and {remaining_turns} more earlier turns.")

        content_lines = [
            "[Conversation Summary]",
            "Compressed earlier context retained for continuity:",
            *lines,
        ]
        content = "\n".join(content_lines)

        return ContextSummary(
            content=content,
            source_turn_ids=[turn.turn_id for turn in dropped_turns],
            dropped_turn_count=len(dropped_turns),
            dropped_message_count=sum(len(turn.messages) for turn in dropped_turns),
            reason_counts=dict(reason_counts),
        )

    def _render_turn(self, turn: ContextTurn) -> str:
        user_parts = [
            self._clip(str(message.get("content", "")))
            for message in turn.messages
            if message.get("role") == "user" and str(message.get("content", "")).strip()
        ]
        assistant_parts = [
            self._clip(str(message.get("content", "")))
            for message in turn.messages
            if message.get("role") == "assistant" and str(message.get("content", "")).strip()
        ]
        tool_names = [
            message.get("name") or "tool"
            for message in turn.messages
            if message.get("role") == "tool"
        ]

        fragments: List[str] = []
        if user_parts:
            fragments.append(f"user: {' | '.join(user_parts[:2])}")
        if assistant_parts:
            fragments.append(f"assistant: {' | '.join(assistant_parts[:2])}")
        if tool_names:
            unique_tools = ", ".join(dict.fromkeys(str(name) for name in tool_names))
            fragments.append(f"tools: {unique_tools}")
        if turn.reasons:
            fragments.append(f"priority: {', '.join(turn.reasons)}")

        if not fragments:
            fragments.append("Earlier exchange without textual content.")

        return "- " + " ; ".join(fragments)

    def _clip(self, text: str) -> str:
        compact = " ".join(text.split())
        if len(compact) <= self.max_chars_per_line:
            return compact
        return compact[: self.max_chars_per_line - 3] + "..."

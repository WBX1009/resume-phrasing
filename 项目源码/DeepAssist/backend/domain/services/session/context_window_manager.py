import heapq
import re
from typing import Any, Dict, List

# Token counting utilities for context budgeting
try:
    import tiktoken  # type: ignore

    _TOKEN_ENCODER = tiktoken.get_encoding("cl100k_base")
except Exception:
    _TOKEN_ENCODER = None

from backend.domain.entities.context_window import (
    ContextPriorityBand,
    ContextTurn,
    ContextWindowPlan,
)


def _count_tokens(text: str) -> int:
    """
    Estimate token count for the given text using tiktoken if available.
    Fallback heuristics count CJK characters, words, and symbols as token-like units.
    """
    if _TOKEN_ENCODER:
        try:
            return len(_TOKEN_ENCODER.encode(text))
        except Exception:
            pass

    # Conservative fallback: count CJK characters, words, and symbols
    token_like_units = re.findall(r"[\u4e00-\u9fff]|[A-Za-z0-9_]+|[^\s]", text or "")
    return len(token_like_units)


class PriorityContextWindowManager:
    """Select a coherent context window using turn priorities."""

    _PROFILE_PATTERN = re.compile(
        r"(?:\b(i am|i'm|my name is|remember|prefer|must|always|do not|don't)\b|"
        r"(?:\u6211\u662f|\u6211\u53eb|\u8bf7\u8bb0\u4f4f|\u8bb0\u4f4f|"
        r"\u504f\u597d|\u4e60\u60ef|\u5fc5\u987b|\u4e0d\u8981|\u8bf7\u7528))",
        flags=re.IGNORECASE,
    )

    def __init__(
        self,
        recent_turns_to_keep: int = 2,
        max_overshoot_messages: int = 2,
    ):
        self.recent_turns_to_keep = recent_turns_to_keep
        self.max_overshoot_messages = max_overshoot_messages

    def _count_tokens(self, text: str) -> int:
        """
        Estimate token count for the given text by delegating to the module-level function.
        """
        return _count_tokens(text)

    def _estimate_message_tokens(self, message: Dict[str, Any]) -> int:
        """
        Estimate the token cost of a single chat message.
        Currently counts tokens in the message content only.
        """
        content = str(message.get("content", ""))
        return self._count_tokens(content)

    def plan(self, history: List[Dict[str, Any]], budget: int) -> ContextWindowPlan:
        safe_budget = max(0, budget)
        turns = self._build_turns(history)
        if not turns:
            return ContextWindowPlan(budget=safe_budget)

        scored_turns = self._score_turns(turns)
        if safe_budget == 0:
            return ContextWindowPlan(
                budget=safe_budget,
                dropped_turns=scored_turns,
            )

        total_cost = sum(turn.estimated_cost for turn in scored_turns)
        if total_cost <= safe_budget:
            return ContextWindowPlan(
                budget=safe_budget,
                selected_turns=scored_turns,
            )

        selected_flags = [True] * len(scored_turns)
        removable: list[tuple[int, int, int]] = []
        for index, turn in enumerate(scored_turns):
            if turn.pinned:
                continue
            heapq.heappush(
                removable,
                (turn.priority_score, turn.started_at_index, index),
            )

        remaining_cost = total_cost
        while removable and remaining_cost > safe_budget:
            _, _, index = heapq.heappop(removable)
            if not selected_flags[index]:
                continue
            selected_flags[index] = False
            remaining_cost -= scored_turns[index].estimated_cost

        selected_turns = [
            turn for turn, keep in zip(scored_turns, selected_flags) if keep
        ]
        dropped_turns = [
            turn for turn, keep in zip(scored_turns, selected_flags) if not keep
        ]

        if remaining_cost > safe_budget and selected_turns and not self._has_pinned_or_profile_turn(selected_turns):
            selected_turns = self._compress_to_recent_turns(selected_turns, safe_budget)
            selected_ids = {turn.turn_id for turn in selected_turns}
            dropped_turns = [
                turn for turn in scored_turns if turn.turn_id not in selected_ids
            ]

        return ContextWindowPlan(
            budget=safe_budget,
            selected_turns=selected_turns,
            dropped_turns=dropped_turns,
        )

    def trim(self, history: List[Dict[str, Any]], budget: int) -> List[Dict[str, Any]]:
        return self.plan(history, budget).flattened_messages()

    def _build_turns(self, history: List[Dict[str, Any]]) -> List[ContextTurn]:
        turns: List[ContextTurn] = []
        current_messages: List[Dict[str, Any]] = []
        current_start = 0
        turn_index = 0

        for index, message in enumerate(history):
            role = message.get("role")
            if role == "user":
                if current_messages:
                    turns.append(
                        ContextTurn(
                            turn_id=f"turn-{turn_index}",
                            started_at_index=current_start,
                            messages=current_messages,
                            estimated_cost=max(
                                1,
                                sum(
                                    self._estimate_message_tokens(msg)
                                    for msg in current_messages
                                ),
                            ),
                        )
                    )
                    turn_index += 1
                current_messages = [message]
                current_start = index
                continue

            if not current_messages:
                current_start = index
            current_messages.append(message)

        if current_messages:
            turns.append(
                ContextTurn(
                    turn_id=f"turn-{turn_index}",
                    started_at_index=current_start,
                    messages=current_messages,
                    estimated_cost=max(
                        1,
                        sum(
                            self._estimate_message_tokens(msg)
                            for msg in current_messages
                        ),
                    ),
                )
            )

        return turns

    def _score_turns(self, turns: List[ContextTurn]) -> List[ContextTurn]:
        scored: List[ContextTurn] = []
        total_turns = len(turns)
        recent_start = max(0, total_turns - self.recent_turns_to_keep)

        for index, turn in enumerate(turns):
            reasons: List[str] = []
            score = 100 + index
            band = ContextPriorityBand.LOW
            pinned = index >= recent_start

            if pinned:
                score += 1000
                band = ContextPriorityBand.CRITICAL
                reasons.append("recent_turn")

            if self._contains_tool_trace(turn):
                score += 220
                reasons.append("tool_trace")
                if band in {ContextPriorityBand.LOW, ContextPriorityBand.MEDIUM}:
                    band = ContextPriorityBand.HIGH

            if self._contains_profile_signal(turn):
                score += 180
                reasons.append("preference_or_profile")
                pinned = True
                if band == ContextPriorityBand.LOW:
                    band = ContextPriorityBand.HIGH

            if self._contains_question(turn):
                score += 60
                reasons.append("question")
                if band == ContextPriorityBand.LOW:
                    band = ContextPriorityBand.MEDIUM

            if self._is_user_only_turn(turn):
                score += 30
                reasons.append("user_only")
                if band == ContextPriorityBand.LOW:
                    band = ContextPriorityBand.MEDIUM

            scored.append(
                turn.model_copy(
                    update={
                        "priority_score": score,
                        "priority_band": band,
                        "pinned": pinned,
                        "reasons": reasons,
                    }
                )
            )

        return scored

    def _compress_to_recent_turns(
        self,
        turns: List[ContextTurn],
        budget: int,
    ) -> List[ContextTurn]:
        if budget <= 0:
            return []

        kept: List[ContextTurn] = []
        consumed = 0
        for turn in reversed(turns):
            turn_cost = turn.estimated_cost
            would_fit = consumed + turn_cost <= budget
            can_overshoot = not kept and turn_cost <= budget + self.max_overshoot_messages
            if would_fit or can_overshoot:
                kept.append(turn)
                consumed += turn_cost
            elif kept:
                break

        return list(reversed(kept))

    def _contains_tool_trace(self, turn: ContextTurn) -> bool:
        for message in turn.messages:
            if message.get("role") == "tool":
                return True
            if message.get("tool_calls"):
                return True
        return False

    def _contains_profile_signal(self, turn: ContextTurn) -> bool:
        for message in turn.messages:
            content = str(message.get("content", ""))
            if self._PROFILE_PATTERN.search(content):
                return True
        return False

    def _contains_question(self, turn: ContextTurn) -> bool:
        for message in turn.messages:
            if message.get("role") != "user":
                continue
            content = str(message.get("content", "")).strip()
            if content.endswith(("?", "\uFF1F", "\u5417", "\u5462")):
                return True
        return False

    def _is_user_only_turn(self, turn: ContextTurn) -> bool:
        return bool(turn.messages) and all(
            message.get("role") == "user" for message in turn.messages
        )

    def _has_pinned_or_profile_turn(self, turns: List[ContextTurn]) -> bool:
        return any(
            turn.pinned or "preference_or_profile" in turn.reasons
            for turn in turns
        )

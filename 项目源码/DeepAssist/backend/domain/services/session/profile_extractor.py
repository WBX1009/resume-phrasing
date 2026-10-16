import json
import re
from typing import Any, Dict, Iterable, List, Set

from backend.common.logger import get_logger
from backend.domain.interfaces.memory_db import BaseMemoryStore

logger = get_logger(__name__)


class ProfileExtractor:
    """Maintains a lightweight long-term user profile from completed sessions.

    This is intentionally deterministic infrastructure for the memory loop. A
    later LLM summarizer can replace the extraction rules without changing the
    application workflow or storage contract.
    """

    def __init__(self, memory_store: BaseMemoryStore):
        self.store = memory_store

    def handle_conversation_completed(self, payload: Dict[str, Any]) -> None:
        messages = self._payload_messages(payload)
        if not messages:
            return

        user_texts = [
            str(msg.get("content", "")).strip()
            for msg in messages
            if msg.get("role") == "user" and msg.get("content")
        ]
        if not user_texts:
            return

        session_id = str(payload.get("session_id", "")).strip()
        last_user_query = user_texts[-1][:800]

        interaction_count = self._read_int("interaction_count") + 1
        self.store.set_profile("interaction_count", str(interaction_count))
        self.store.set_profile("last_user_query", last_user_query)
        if session_id:
            self.store.set_profile("last_session_id", session_id)

        topics = self._read_json_list("topics")
        topics.update(self._infer_topics(" ".join(user_texts)))
        if topics:
            self.store.set_profile(
                "topics",
                json.dumps(sorted(topics), ensure_ascii=False),
            )

        facts = self._read_json_list("user_facts")
        facts.update(self._infer_user_facts(user_texts))
        if facts:
            self.store.set_profile(
                "user_facts",
                json.dumps(sorted(facts), ensure_ascii=False),
            )

        logger.info(
            "Updated user profile from session %s; interactions=%s",
            session_id or "<unknown>",
            interaction_count,
        )

    def render_profile(self) -> str:
        profiles = self.store.get_all_profiles()
        if not profiles:
            return ""

        lines: List[str] = []
        interaction_count = profiles.get("interaction_count")
        if interaction_count:
            lines.append(f"- interaction_count: {interaction_count}")

        topics = self._parse_json_iterable(profiles.get("topics", "[]"))
        if topics:
            lines.append(f"- recurring_topics: {', '.join(topics)}")

        facts = self._parse_json_iterable(profiles.get("user_facts", "[]"))
        if facts:
            lines.append("- user_facts:")
            lines.extend(f"  - {fact}" for fact in facts)

        last_user_query = profiles.get("last_user_query")
        if last_user_query:
            lines.append(f"- last_user_query: {last_user_query}")

        return "\n".join(lines)

    def _payload_messages(self, payload: Dict[str, Any]) -> List[Dict[str, Any]]:
        messages = payload.get("messages")
        if isinstance(messages, list):
            return [msg for msg in messages if isinstance(msg, dict)]

        fallback: List[Dict[str, Any]] = []
        user_query = payload.get("user_query")
        assistant_response = payload.get("assistant_response")
        if user_query:
            fallback.append({"role": "user", "content": user_query})
        if assistant_response:
            fallback.append({"role": "assistant", "content": assistant_response})
        return fallback

    def _read_int(self, key: str) -> int:
        value = self.store.get_profile(key)
        try:
            return int(value or 0)
        except ValueError:
            return 0

    def _read_json_list(self, key: str) -> Set[str]:
        return set(self._parse_json_iterable(self.store.get_profile(key) or "[]"))

    def _parse_json_iterable(self, value: str) -> List[str]:
        try:
            parsed = json.loads(value)
        except (TypeError, json.JSONDecodeError):
            return []
        if not isinstance(parsed, list):
            return []
        return [str(item) for item in parsed if item]

    def _infer_topics(self, text: str) -> Iterable[str]:
        lower_text = text.lower()
        candidates = {
            "agent": ("agent", "\u667a\u80fd\u4f53", "\u5de5\u5177"),
            "architecture": ("architecture", "\u67b6\u6784", "\u5206\u5c42", "\u89e3\u8026"),
            "python": ("python",),
            "rag": ("rag", "\u77e5\u8bc6\u5e93", "\u68c0\u7d22", "\u5411\u91cf"),
            "zh": ("\u4e2d\u6587",),
        }
        for topic, needles in candidates.items():
            if any(needle in lower_text for needle in needles):
                yield topic

    def _infer_user_facts(self, user_texts: List[str]) -> Iterable[str]:
        patterns = [
            r"(?:请记住|记住|添加|保存|更新)[：:\s]+(.{4,120})",
            r"(我是[^。！？\n]{2,80})",
            r"(我是一名[^。！？\n]{2,80})",
            r"(我的职业是[^。！？\n]{2,80})",
            r"(我的身份是[^。！？\n]{2,80})",
            r"(我有[^。！？\n]{2,80}经验)",
        ]
        for text in user_texts:
            normalized = " ".join(str(text).split())
            for pattern in patterns:
                for match in re.findall(pattern, normalized):
                    fact = self._clean_fact(match)
                    if fact:
                        yield fact

    def _clean_fact(self, fact: str) -> str:
        fact = fact.strip(" ，,。.;；：:")
        fact = re.sub(r"^(如果没有|请|帮我|给我)", "", fact).strip(" ，,。.;；：:")
        if len(fact) < 4:
            return ""
        if any(needle in fact for needle in ("查询我的长期记忆画像", "长期记忆画像是什么")):
            return ""
        return fact[:120]

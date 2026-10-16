from typing import Any, Callable, Dict, List

from backend.common.logger import get_logger
from backend.domain.entities.context_window import ContextWindowPlan
from backend.domain.entities.message import AIMessage, Message
from backend.domain.entities.task_snapshot import TaskSnapshot
from backend.domain.entities.tooling import ToolCall
from backend.domain.interfaces.memory_db import BaseMemoryStore
from backend.domain.services.session.context_window_manager import PriorityContextWindowManager
from backend.domain.services.session.long_term_memory_recall import LongTermMemoryRecallService
from backend.domain.services.session.summary_compressor import ConversationSummaryCompressor

logger = get_logger(__name__)


class SessionManager:
    """Conversation history orchestration for persistence and context windows."""

    def __init__(
        self,
        memory_store: BaseMemoryStore,
        context_window_manager: PriorityContextWindowManager | None = None,
        summary_compressor: ConversationSummaryCompressor | None = None,
        memory_recall: LongTermMemoryRecallService | None = None,
        event_publisher: Callable[[str, Dict[str, Any]], None] | None = None,
    ):
        self.store = memory_store
        self.context_window_manager = context_window_manager or PriorityContextWindowManager()
        self.summary_compressor = summary_compressor or ConversationSummaryCompressor()
        self.memory_recall = memory_recall or LongTermMemoryRecallService(memory_store)
        self.event_publisher = event_publisher

    def get_session_history(self, session_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        raw_messages = self._load_and_normalize_history(session_id, limit=max(1, limit))
        return self._sanitize_tool_history(raw_messages)

    def plan_chat_context(
        self,
        session_id: str,
        max_rounds: int = 5,
        query: str = "",
        use_long_term_memory: bool = False,
    ) -> ContextWindowPlan:
        safe_budget = max(1, max_rounds)
        persisted_summary = self.store.get_session_summary(session_id)
        recalled_memories = (
            self._recall_long_term_memories(query, safe_budget)
            if use_long_term_memory
            else []
        )
        history_budget = max(1, safe_budget - len(recalled_memories))

        raw_history = self._load_and_normalize_history(
            session_id,
            limit=max(history_budget * 3, safe_budget),
        )
        valid_messages = self._sanitize_tool_history(raw_history)
        plan = self.context_window_manager.plan(valid_messages, budget=history_budget)
        if plan.dropped_turns:
            summary = self.summary_compressor.compress(plan.dropped_turns)
            if summary is not None:
                persisted_copy = summary.model_copy(update={"source": "persisted"})
                self.store.save_session_summary(session_id, persisted_copy)
                plan = plan.model_copy(update={"summary": summary})
        elif persisted_summary is not None:
            plan = plan.model_copy(
                update={"summary": persisted_summary.model_copy(update={"source": "persisted"})}
            )
        if recalled_memories:
            plan = plan.model_copy(update={"recalled_memories": recalled_memories, "budget": safe_budget})
        else:
            plan = plan.model_copy(update={"budget": safe_budget})
        return plan

    def get_chat_context(
        self,
        session_id: str,
        max_rounds: int = 5,
        query: str = "",
        use_long_term_memory: bool = False,
    ) -> List[Dict[str, Any]]:
        plan = self.plan_chat_context(
            session_id,
            max_rounds=max_rounds,
            query=query,
            use_long_term_memory=use_long_term_memory,
        )
        messages = plan.flattened_messages()
        logger.debug(
            "Context window selected %s messages for session %s under budget=%s; recalled_memories=%s, dropped_turns=%s, summary=%s",
            len(messages),
            session_id,
            plan.budget,
            len(plan.recalled_memories),
            len(plan.dropped_turns),
            bool(plan.summary),
        )
        return messages

    def save_interaction(self, session_id: str, user_query: str, ai_response: str):
        messages: List[Dict[str, Any]] = []
        if user_query:
            self.store.add_message(session_id, Message(role="user", content=user_query))
            messages.append({"role": "user", "content": user_query})
        if ai_response:
            self.store.add_message(session_id, Message(role="assistant", content=ai_response))
            messages.append({"role": "assistant", "content": ai_response})
        if messages:
            self._publish_event(
                "conversation.completed",
                {
                    "session_id": session_id,
                    "messages": messages,
                    "user_query": user_query,
                    "assistant_response": ai_response,
                },
            )

    def add_messages(self, session_id: str, messages: List[Dict[str, Any]]):
        for msg in messages:
            if "tool_calls" in msg and msg["tool_calls"]:
                entity = AIMessage(**msg)
            else:
                entity = Message(**msg)
            self.store.add_message(session_id, entity)

        if messages:
            self._publish_event(
                "conversation.completed",
                {
                    "session_id": session_id,
                    "messages": messages,
                },
            )

        logger.info("Persisted %s conversation/tool trace messages", len(messages))

    def delete_session(self, session_id: str) -> bool:
        success = self.store.clear_history(session_id)
        if success:
            self.store.clear_task_snapshot(session_id)
            self.store.clear_session_summary(session_id)
            logger.info("Deleted session lifecycle for %s", session_id)
        return success

    def list_sessions(self) -> List[Dict[str, Any]]:
        return self.store.get_all_sessions()

    def get_task_snapshot(self, session_id: str) -> TaskSnapshot | None:
        return self.store.get_task_snapshot(session_id)

    def save_task_snapshot(self, snapshot: TaskSnapshot) -> bool:
        return self.store.save_task_snapshot(snapshot)

    def clear_task_snapshot(self, session_id: str) -> bool:
        return self.store.clear_task_snapshot(session_id)

    def save_routing_log(self, log_data: Dict[str, Any]) -> bool:
        save_method = getattr(self.store, "save_routing_log", None)
        if not callable(save_method):
            return False
        try:
            return bool(save_method(log_data))
        except Exception as exc:
            logger.error("Failed to save routing log: %s", exc)
            return False

    def _load_and_normalize_history(self, session_id: str, limit: int) -> List[Dict[str, Any]]:
        history_entities = self.store.get_history(session_id, limit=limit)
        raw_messages = [msg.model_dump(exclude_none=True) for msg in history_entities]
        return [self._normalize_message_for_llm(msg) for msg in raw_messages]

    def _sanitize_tool_history(self, raw_messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        valid_messages: List[Dict[str, Any]] = []
        expected_tool_call_ids = set()

        for msg in raw_messages:
            role = msg.get("role")

            if role in {"user", "system"}:
                if expected_tool_call_ids:
                    while valid_messages and (
                        valid_messages[-1].get("role") == "tool"
                        or "tool_calls" in valid_messages[-1]
                    ):
                        valid_messages.pop()
                    expected_tool_call_ids.clear()
                valid_messages.append(msg)
                continue

            if role == "assistant":
                if expected_tool_call_ids:
                    while valid_messages and (
                        valid_messages[-1].get("role") == "tool"
                        or "tool_calls" in valid_messages[-1]
                    ):
                        valid_messages.pop()
                    expected_tool_call_ids.clear()

                valid_messages.append(msg)
                if "tool_calls" in msg and msg["tool_calls"]:
                    expected_tool_call_ids = {tc["id"] for tc in msg["tool_calls"]}
                continue

            if role == "tool":
                if not expected_tool_call_ids:
                    continue

                tool_id = msg.get("tool_call_id")
                if tool_id in expected_tool_call_ids:
                    valid_messages.append(msg)
                    expected_tool_call_ids.remove(tool_id)

        if expected_tool_call_ids:
            while valid_messages and (
                valid_messages[-1].get("role") == "tool"
                or "tool_calls" in valid_messages[-1]
            ):
                valid_messages.pop()

        return valid_messages

    def _normalize_message_for_llm(self, msg: Dict[str, Any]) -> Dict[str, Any]:
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            normalized = dict(msg)
            normalized["tool_calls"] = [
                ToolCall.model_validate(tool_call).to_openai_tool_call()
                for tool_call in msg["tool_calls"]
            ]
            return normalized
        return msg

    def _recall_long_term_memories(self, query: str, budget: int):
        if not query.strip() or budget <= 1:
            return []
        memory_limit = min(2, max(1, budget // 3))
        return self.memory_recall.recall(query, max_items=memory_limit)

    def _publish_event(self, event_name: str, payload: Dict[str, Any]) -> None:
        if self.event_publisher is not None:
            self.event_publisher(event_name, payload)

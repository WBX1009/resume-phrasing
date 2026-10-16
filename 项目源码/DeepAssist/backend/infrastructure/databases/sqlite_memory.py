import json
import os
import sqlite3
from typing import Any, Dict, List, Optional

from backend.common.config import settings
from backend.common.logger import get_logger
from backend.domain.entities.context_window import ContextSummary
from backend.domain.entities.message import AIMessage, Message
from backend.domain.entities.task_snapshot import TaskSnapshot
from backend.domain.interfaces.memory_db import BaseMemoryStore

logger = get_logger(__name__)


class SQLiteMemoryStore(BaseMemoryStore):
    """SQLite-backed local memory store used by default."""

    def __init__(self):
        self.db_path = settings.CONVERSATION_DB_PATH
        os.makedirs(os.path.dirname(self.db_path) or ".", exist_ok=True)
        self._init_db()
        logger.info("SQLite memory store initialized: %s", self.db_path)

    def _get_connection(self):
        return sqlite3.connect(self.db_path, check_same_thread=False)

    def _init_db(self) -> None:
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS messages (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        session_id TEXT NOT NULL,
                        role TEXT NOT NULL,
                        content TEXT,
                        name TEXT,
                        tool_call_id TEXT,
                        tool_calls TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                    """
                )
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS user_profiles (
                        key TEXT PRIMARY KEY,
                        value TEXT,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                    """
                )
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS task_snapshots (
                        session_id TEXT PRIMARY KEY,
                        query TEXT NOT NULL,
                        route_worker TEXT NOT NULL,
                        status TEXT NOT NULL,
                        payload TEXT,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                    """
                )
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS session_summaries (
                        session_id TEXT PRIMARY KEY,
                        content TEXT NOT NULL,
                        source TEXT NOT NULL,
                        source_turn_ids TEXT,
                        dropped_turn_count INTEGER NOT NULL,
                        dropped_message_count INTEGER NOT NULL,
                        reason_counts TEXT,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                    """
                )
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS routing_logs (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        session_id TEXT NOT NULL,
                        query TEXT NOT NULL,
                        history_preview TEXT,
                        route TEXT NOT NULL,
                        confidence REAL NOT NULL,
                        reason TEXT,
                        rewritten_query TEXT,
                        source_policy TEXT,
                        required_capabilities TEXT,
                        risk_level TEXT,
                        final_used_route TEXT,
                        error TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                    """
                )
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_messages_session_id ON messages(session_id)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_routing_logs_session_id ON routing_logs(session_id)")
                conn.commit()
        except Exception as exc:
            logger.error("Failed to initialize SQLite memory schema: %s", exc)

    def get_history(self, session_id: str, limit: int = 10) -> List[Message]:
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT role, content, name, tool_call_id, tool_calls
                    FROM messages
                    WHERE session_id = ?
                    ORDER BY id DESC
                    LIMIT ?
                    """,
                    (session_id, max(1, limit * 4)),
                )
                rows = cursor.fetchall()

            history: List[Message] = []
            for role, content, name, tool_call_id, tool_calls_json in reversed(rows):
                if tool_calls_json:
                    history.append(
                        AIMessage(
                            role=role,
                            content=content or "",
                            tool_calls=json.loads(tool_calls_json),
                        )
                    )
                else:
                    history.append(
                        Message(
                            role=role,
                            content=content or "",
                            name=name,
                            tool_call_id=tool_call_id,
                        )
                    )
            return history
        except Exception as exc:
            logger.error("Failed to read SQLite history [%s]: %s", session_id, exc)
            return []

    def add_message(self, session_id: str, message: Message) -> bool:
        try:
            tool_calls_json = None
            if isinstance(message, AIMessage) and message.tool_calls:
                tool_calls_json = json.dumps([tool_call.model_dump() for tool_call in message.tool_calls], ensure_ascii=False)

            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    INSERT INTO messages (session_id, role, content, name, tool_call_id, tool_calls)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (session_id, message.role, message.content, message.name, message.tool_call_id, tool_calls_json),
                )
                conn.commit()
            return True
        except Exception as exc:
            logger.error("Failed to write SQLite history [%s]: %s", session_id, exc)
            return False

    def clear_history(self, session_id: str) -> bool:
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                for table in ("messages", "session_summaries", "task_snapshots", "routing_logs"):
                    cursor.execute(f"DELETE FROM {table} WHERE session_id = ?", (session_id,))
                conn.commit()
            return True
        except Exception as exc:
            logger.error("Failed to clear SQLite history [%s]: %s", session_id, exc)
            return False

    def get_all_sessions(self) -> List[Dict[str, Any]]:
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT session_id, MIN(created_at) AS start_time, MAX(created_at) AS last_time
                    FROM messages
                    GROUP BY session_id
                    ORDER BY last_time DESC
                    """
                )
                rows = cursor.fetchall()
                sessions: List[Dict[str, Any]] = []
                for session_id, _start_time, last_time in rows:
                    cursor.execute(
                        """
                        SELECT content FROM messages
                        WHERE session_id = ? AND role = 'user'
                        ORDER BY id ASC
                        LIMIT 1
                        """,
                        (session_id,),
                    )
                    title_row = cursor.fetchone()
                    title = "New chat"
                    if title_row and title_row[0]:
                        title = title_row[0][:15].replace("\n", " ")
                        if len(title_row[0]) > 15:
                            title += "..."
                    sessions.append({"session_id": session_id, "title": title, "updated_at": last_time})
                return sessions
        except Exception as exc:
            logger.error("Failed to list SQLite sessions: %s", exc)
            return []

    def get_profile(self, key: str) -> Optional[str]:
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT value FROM user_profiles WHERE key = ?", (key,))
                row = cursor.fetchone()
                return row[0] if row else None
        except Exception as exc:
            logger.error("Failed to read SQLite profile [%s]: %s", key, exc)
            return None

    def set_profile(self, key: str, value: str) -> bool:
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    INSERT INTO user_profiles (key, value, updated_at)
                    VALUES (?, ?, CURRENT_TIMESTAMP)
                    ON CONFLICT(key) DO UPDATE SET
                        value = excluded.value,
                        updated_at = CURRENT_TIMESTAMP
                    """,
                    (key, value),
                )
                conn.commit()
            return True
        except Exception as exc:
            logger.error("Failed to write SQLite profile [%s]: %s", key, exc)
            return False

    def get_all_profiles(self) -> Dict[str, str]:
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT key, value FROM user_profiles ORDER BY key ASC")
                return {key: value for key, value in cursor.fetchall()}
        except Exception as exc:
            logger.error("Failed to read SQLite profiles: %s", exc)
            return {}

    def get_task_snapshot(self, session_id: str) -> Optional[TaskSnapshot]:
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT session_id, query, route_worker, status, payload, updated_at
                    FROM task_snapshots
                    WHERE session_id = ?
                    """,
                    (session_id,),
                )
                row = cursor.fetchone()
            if not row:
                return None
            payload = json.loads(row[4]) if row[4] else {}
            return TaskSnapshot(
                session_id=row[0],
                query=row[1],
                route_worker=row[2],
                status=row[3],
                payload=payload if isinstance(payload, dict) else {},
                updated_at=row[5],
            )
        except Exception as exc:
            logger.error("Failed to read SQLite task snapshot [%s]: %s", session_id, exc)
            return None

    def save_task_snapshot(self, snapshot: TaskSnapshot) -> bool:
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    INSERT INTO task_snapshots (session_id, query, route_worker, status, payload, updated_at)
                    VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                    ON CONFLICT(session_id) DO UPDATE SET
                        query = excluded.query,
                        route_worker = excluded.route_worker,
                        status = excluded.status,
                        payload = excluded.payload,
                        updated_at = CURRENT_TIMESTAMP
                    """,
                    (
                        snapshot.session_id,
                        snapshot.query,
                        snapshot.route_worker,
                        snapshot.status,
                        json.dumps(snapshot.payload, ensure_ascii=False),
                    ),
                )
                conn.commit()
            return True
        except Exception as exc:
            logger.error("Failed to save SQLite task snapshot [%s]: %s", snapshot.session_id, exc)
            return False

    def clear_task_snapshot(self, session_id: str) -> bool:
        return self._delete_by_session("task_snapshots", session_id)

    def get_session_summary(self, session_id: str) -> Optional[ContextSummary]:
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT content, source, source_turn_ids, dropped_turn_count,
                           dropped_message_count, reason_counts
                    FROM session_summaries
                    WHERE session_id = ?
                    """,
                    (session_id,),
                )
                row = cursor.fetchone()
            if not row:
                return None
            return ContextSummary(
                content=row[0],
                source=row[1] or "persisted",
                source_turn_ids=json.loads(row[2]) if row[2] else [],
                dropped_turn_count=int(row[3] or 0),
                dropped_message_count=int(row[4] or 0),
                reason_counts=json.loads(row[5]) if row[5] else {},
            )
        except Exception as exc:
            logger.error("Failed to read SQLite session summary [%s]: %s", session_id, exc)
            return None

    def save_session_summary(self, session_id: str, summary: ContextSummary) -> bool:
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    INSERT INTO session_summaries (
                        session_id, content, source, source_turn_ids, dropped_turn_count,
                        dropped_message_count, reason_counts, updated_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                    ON CONFLICT(session_id) DO UPDATE SET
                        content = excluded.content,
                        source = excluded.source,
                        source_turn_ids = excluded.source_turn_ids,
                        dropped_turn_count = excluded.dropped_turn_count,
                        dropped_message_count = excluded.dropped_message_count,
                        reason_counts = excluded.reason_counts,
                        updated_at = CURRENT_TIMESTAMP
                    """,
                    (
                        session_id,
                        summary.content,
                        summary.source,
                        json.dumps(summary.source_turn_ids, ensure_ascii=False),
                        summary.dropped_turn_count,
                        summary.dropped_message_count,
                        json.dumps(summary.reason_counts, ensure_ascii=False),
                    ),
                )
                conn.commit()
            return True
        except Exception as exc:
            logger.error("Failed to save SQLite session summary [%s]: %s", session_id, exc)
            return False

    def clear_session_summary(self, session_id: str) -> bool:
        return self._delete_by_session("session_summaries", session_id)

    def save_routing_log(self, log_data: Dict[str, Any]) -> bool:
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    INSERT INTO routing_logs (
                        session_id, query, history_preview, route, confidence, reason,
                        rewritten_query, source_policy, required_capabilities,
                        risk_level, final_used_route, error
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        str(log_data.get("session_id") or ""),
                        str(log_data.get("query") or ""),
                        json.dumps(log_data.get("history_preview", []), ensure_ascii=False),
                        str(log_data.get("route") or ""),
                        float(log_data.get("confidence") or 0.0),
                        str(log_data.get("reason") or ""),
                        str(log_data.get("rewritten_query") or ""),
                        str(log_data.get("source_policy") or ""),
                        json.dumps(log_data.get("required_capabilities", []), ensure_ascii=False),
                        str(log_data.get("risk_level") or ""),
                        str(log_data.get("final_used_route") or ""),
                        str(log_data.get("error") or ""),
                    ),
                )
                conn.commit()
            return True
        except Exception as exc:
            logger.error("Failed to save SQLite routing log [%s]: %s", log_data.get("session_id"), exc)
            return False

    def _delete_by_session(self, table: str, session_id: str) -> bool:
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(f"DELETE FROM {table} WHERE session_id = ?", (session_id,))
                conn.commit()
            return True
        except Exception as exc:
            logger.error("Failed to delete SQLite rows from %s [%s]: %s", table, session_id, exc)
            return False

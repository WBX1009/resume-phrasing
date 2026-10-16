import json
from collections.abc import Callable
from typing import Any, Dict, List, Optional

from backend.common.config import settings
from backend.common.logger import get_logger
from backend.domain.entities.context_window import ContextSummary
from backend.domain.entities.message import AIMessage, Message
from backend.domain.entities.task_snapshot import TaskSnapshot
from backend.domain.interfaces.memory_db import BaseMemoryStore

logger = get_logger(__name__)

ConnectionFactory = Callable[[], Any]


class MySQLMemoryStoreAdapter(BaseMemoryStore):
    """MySQL-backed memory store adapter for enterprise deployments."""

    def __init__(
        self,
        host: str | None = None,
        port: int | None = None,
        user: str | None = None,
        password: str | None = None,
        database: str | None = None,
        connection_factory: ConnectionFactory | None = None,
        initialize_schema: bool = True,
    ):
        self.host = host or settings.MYSQL_HOST
        self.port = port or settings.MYSQL_PORT
        self.user = user or settings.MYSQL_USER
        self.password = password if password is not None else settings.MYSQL_PASSWORD
        self.database = database or settings.MYSQL_DATABASE
        self._connection_factory = connection_factory
        if initialize_schema:
            self._init_db()

    def _get_connection(self):
        if self._connection_factory is not None:
            return self._connection_factory()

        try:
            import pymysql
        except ImportError as exc:
            raise RuntimeError(
                "pymysql is required when MEMORY_BACKEND=mysql. Install requirements.txt first."
            ) from exc

        return pymysql.connect(
            host=self.host,
            port=int(self.port),
            user=self.user,
            password=self.password,
            database=self.database,
            charset="utf8mb4",
            autocommit=False,
        )

    def _init_db(self) -> None:
        statements = [
            """
            CREATE TABLE IF NOT EXISTS messages (
                id BIGINT PRIMARY KEY AUTO_INCREMENT,
                session_id VARCHAR(191) NOT NULL,
                role VARCHAR(32) NOT NULL,
                content LONGTEXT,
                name VARCHAR(191),
                tool_call_id VARCHAR(191),
                tool_calls JSON,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                INDEX idx_messages_session_id_id (session_id, id)
            ) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci
            """,
            """
            CREATE TABLE IF NOT EXISTS user_profiles (
                `key` VARCHAR(191) PRIMARY KEY,
                value LONGTEXT,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
            ) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci
            """,
            """
            CREATE TABLE IF NOT EXISTS task_snapshots (
                session_id VARCHAR(191) PRIMARY KEY,
                query LONGTEXT NOT NULL,
                route_worker VARCHAR(191) NOT NULL,
                status VARCHAR(64) NOT NULL,
                payload JSON,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
            ) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci
            """,
            """
            CREATE TABLE IF NOT EXISTS session_summaries (
                session_id VARCHAR(191) PRIMARY KEY,
                content LONGTEXT NOT NULL,
                source VARCHAR(64) NOT NULL,
                source_turn_ids JSON,
                dropped_turn_count INT NOT NULL,
                dropped_message_count INT NOT NULL,
                reason_counts JSON,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
            ) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci
            """,
            """
            CREATE TABLE IF NOT EXISTS routing_logs (
                id BIGINT PRIMARY KEY AUTO_INCREMENT,
                session_id VARCHAR(191) NOT NULL,
                query LONGTEXT NOT NULL,
                history_preview JSON,
                route VARCHAR(64) NOT NULL,
                confidence DOUBLE NOT NULL,
                reason LONGTEXT,
                rewritten_query LONGTEXT,
                source_policy VARCHAR(64),
                required_capabilities JSON,
                risk_level VARCHAR(64),
                final_used_route VARCHAR(64),
                error LONGTEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                INDEX idx_routing_logs_session_id (session_id)
            ) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci
            """,
        ]
        self._execute_write(lambda cursor: [cursor.execute(statement) for statement in statements])

    def get_history(self, session_id: str, limit: int = 10) -> List[Message]:
        safe_limit = max(1, limit * 4)
        rows = self._fetch_all(
            """
            SELECT role, content, name, tool_call_id, tool_calls
            FROM messages
            WHERE session_id = %s
            ORDER BY id DESC
            LIMIT %s
            """,
            (session_id, safe_limit),
        )
        history: List[Message] = []
        for row in reversed(rows):
            role, content, name, tool_call_id, tool_calls_json = row
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

    def add_message(self, session_id: str, message: Message) -> bool:
        tool_calls_json = None
        if isinstance(message, AIMessage) and message.tool_calls:
            tool_calls_json = json.dumps([tool_call.model_dump() for tool_call in message.tool_calls], ensure_ascii=False)

        return self._execute_write(
            lambda cursor: cursor.execute(
                """
                INSERT INTO messages (session_id, role, content, name, tool_call_id, tool_calls)
                VALUES (%s, %s, %s, %s, %s, %s)
                """,
                (session_id, message.role, message.content, message.name, message.tool_call_id, tool_calls_json),
            )
        )

    def clear_history(self, session_id: str) -> bool:
        return self._execute_write(
            lambda cursor: [
                cursor.execute("DELETE FROM messages WHERE session_id = %s", (session_id,)),
                cursor.execute("DELETE FROM session_summaries WHERE session_id = %s", (session_id,)),
                cursor.execute("DELETE FROM task_snapshots WHERE session_id = %s", (session_id,)),
                cursor.execute("DELETE FROM routing_logs WHERE session_id = %s", (session_id,)),
            ]
        )

    def get_all_sessions(self) -> List[Dict[str, Any]]:
        rows = self._fetch_all(
            """
            SELECT session_id, MIN(created_at) AS start_time, MAX(created_at) AS last_time
            FROM messages
            GROUP BY session_id
            ORDER BY last_time DESC
            """
        )
        sessions: List[Dict[str, Any]] = []
        for session_id, _start_time, last_time in rows:
            title_row = self._fetch_one(
                """
                SELECT content FROM messages
                WHERE session_id = %s AND role = 'user'
                ORDER BY id ASC
                LIMIT 1
                """,
                (session_id,),
            )
            title = "New chat"
            if title_row and title_row[0]:
                title = title_row[0][:15].replace("\n", " ")
                if len(title_row[0]) > 15:
                    title += "..."
            sessions.append({"session_id": session_id, "title": title, "updated_at": str(last_time)})
        return sessions

    def get_profile(self, key: str) -> Optional[str]:
        row = self._fetch_one("SELECT value FROM user_profiles WHERE `key` = %s", (key,))
        return row[0] if row else None

    def set_profile(self, key: str, value: str) -> bool:
        return self._execute_write(
            lambda cursor: cursor.execute(
                """
                INSERT INTO user_profiles (`key`, value, updated_at)
                VALUES (%s, %s, CURRENT_TIMESTAMP)
                ON DUPLICATE KEY UPDATE value = VALUES(value), updated_at = CURRENT_TIMESTAMP
                """,
                (key, value),
            )
        )

    def get_all_profiles(self) -> Dict[str, str]:
        return {key: value for key, value in self._fetch_all("SELECT `key`, value FROM user_profiles ORDER BY `key` ASC")}

    def get_task_snapshot(self, session_id: str) -> Optional[TaskSnapshot]:
        row = self._fetch_one(
            """
            SELECT session_id, query, route_worker, status, payload, updated_at
            FROM task_snapshots
            WHERE session_id = %s
            """,
            (session_id,),
        )
        if not row:
            return None
        payload = json.loads(row[4]) if row[4] else {}
        return TaskSnapshot(
            session_id=row[0],
            query=row[1],
            route_worker=row[2],
            status=row[3],
            payload=payload if isinstance(payload, dict) else {},
            updated_at=str(row[5]) if row[5] is not None else None,
        )

    def save_task_snapshot(self, snapshot: TaskSnapshot) -> bool:
        return self._execute_write(
            lambda cursor: cursor.execute(
                """
                INSERT INTO task_snapshots (session_id, query, route_worker, status, payload, updated_at)
                VALUES (%s, %s, %s, %s, %s, CURRENT_TIMESTAMP)
                ON DUPLICATE KEY UPDATE
                    query = VALUES(query),
                    route_worker = VALUES(route_worker),
                    status = VALUES(status),
                    payload = VALUES(payload),
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
        )

    def clear_task_snapshot(self, session_id: str) -> bool:
        return self._execute_write(
            lambda cursor: cursor.execute("DELETE FROM task_snapshots WHERE session_id = %s", (session_id,))
        )

    def get_session_summary(self, session_id: str) -> Optional[ContextSummary]:
        row = self._fetch_one(
            """
            SELECT content, source, source_turn_ids, dropped_turn_count,
                   dropped_message_count, reason_counts
            FROM session_summaries
            WHERE session_id = %s
            """,
            (session_id,),
        )
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

    def save_session_summary(self, session_id: str, summary: ContextSummary) -> bool:
        return self._execute_write(
            lambda cursor: cursor.execute(
                """
                INSERT INTO session_summaries (
                    session_id, content, source, source_turn_ids, dropped_turn_count,
                    dropped_message_count, reason_counts, updated_at
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, CURRENT_TIMESTAMP)
                ON DUPLICATE KEY UPDATE
                    content = VALUES(content),
                    source = VALUES(source),
                    source_turn_ids = VALUES(source_turn_ids),
                    dropped_turn_count = VALUES(dropped_turn_count),
                    dropped_message_count = VALUES(dropped_message_count),
                    reason_counts = VALUES(reason_counts),
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
        )

    def clear_session_summary(self, session_id: str) -> bool:
        return self._execute_write(
            lambda cursor: cursor.execute("DELETE FROM session_summaries WHERE session_id = %s", (session_id,))
        )

    def save_routing_log(self, log_data: Dict[str, Any]) -> bool:
        return self._execute_write(
            lambda cursor: cursor.execute(
                """
                INSERT INTO routing_logs (
                    session_id, query, history_preview, route, confidence, reason,
                    rewritten_query, source_policy, required_capabilities,
                    risk_level, final_used_route, error
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
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
        )

    def _fetch_one(self, sql: str, params: tuple[Any, ...] = ()) -> Optional[tuple[Any, ...]]:
        rows = self._fetch_all(sql, params)
        return rows[0] if rows else None

    def _fetch_all(self, sql: str, params: tuple[Any, ...] = ()) -> List[tuple[Any, ...]]:
        connection = self._get_connection()
        try:
            cursor = connection.cursor()
            cursor.execute(sql, params)
            return list(cursor.fetchall())
        except Exception as exc:
            logger.error("MySQL memory query failed: %s", exc)
            return []
        finally:
            self._close_connection(connection)

    def _execute_write(self, operation: Callable[[Any], Any]) -> bool:
        connection = self._get_connection()
        try:
            cursor = connection.cursor()
            operation(cursor)
            connection.commit()
            return True
        except Exception as exc:
            rollback = getattr(connection, "rollback", None)
            if callable(rollback):
                rollback()
            logger.error("MySQL memory write failed: %s", exc)
            return False
        finally:
            self._close_connection(connection)

    def _close_connection(self, connection: Any) -> None:
        close = getattr(connection, "close", None)
        if callable(close):
            close()

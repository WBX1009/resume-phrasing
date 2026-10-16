"""Optional LangSmith trace sink backed by the local trace port.

The application remains usable offline: local traces are written first and a
remote sink failure is logged instead of changing the request result.  This
keeps the observability adapter outside the domain/application layers while
making the LangSmith integration explicit when enabled by configuration.
"""

from datetime import datetime, timedelta, timezone
from typing import Any, List, Optional
from uuid import uuid4

from backend.common.logger import get_logger
from backend.domain.entities.trace_event import TraceEvent
from backend.domain.interfaces.trace_store import TraceStorePort
from backend.infrastructure.tracing.local_trace_store import LocalTraceStore

logger = get_logger(__name__)


class LangSmithTraceStore(TraceStorePort):
    """Persist local trace events and mirror them as LangSmith runs."""

    def __init__(
        self,
        client: Any,
        project_name: str = "deepassist",
        local_store: Optional[LocalTraceStore] = None,
    ):
        self.client = client
        self.project_name = project_name
        self.local_store = local_store or LocalTraceStore()

    @classmethod
    def from_settings(cls, settings: Any, local_store: Optional[LocalTraceStore] = None):
        from langsmith import Client

        client_kwargs = {
            "api_key": settings.LANGSMITH_API_KEY,
            "auto_batch_tracing": True,
        }
        if settings.LANGSMITH_ENDPOINT:
            client_kwargs["api_url"] = settings.LANGSMITH_ENDPOINT
        client = Client(**client_kwargs)
        return cls(
            client=client,
            project_name=settings.LANGSMITH_PROJECT,
            local_store=local_store,
        )

    def append(self, event: TraceEvent) -> None:
        self.local_store.append(event)
        try:
            self.client.create_run(
                id=uuid4(),
                name=event.span_name,
                run_type=self._run_type(event.event_type),
                project_name=self.project_name,
                inputs={"summary": event.input_summary or ""},
                outputs={"summary": event.output_summary or ""},
                error=event.error_type,
                start_time=self._start_time(event),
                end_time=self._end_time(event),
                extra={
                    "metadata": {
                        "deepassist_trace_id": event.trace_id,
                        "deepassist_session_id": event.session_id,
                        "deepassist_route": event.route,
                        "deepassist_event_type": event.event_type,
                        **event.metadata,
                    }
                },
                tags=[
                    "deepassist",
                    event.event_type,
                    *([event.route] if event.route else []),
                ],
            )
        except Exception as exc:  # pragma: no cover - exercised with a live service
            logger.warning("LangSmith trace upload failed: %s", exc)

    def list_by_trace_id(self, trace_id: str) -> List[TraceEvent]:
        return self.local_store.list_by_trace_id(trace_id)

    def list_recent(self, limit: int = 50) -> List[TraceEvent]:
        return self.local_store.list_recent(limit=limit)

    def clear(self) -> None:
        self.local_store.clear()

    def _run_type(self, event_type: str) -> str:
        if event_type in {"model_call", "llm_call"}:
            return "llm"
        if event_type in {"tool_call", "tool_result"}:
            return "tool"
        return "chain"

    def _start_time(self, event: TraceEvent) -> datetime:
        end_time = self._event_time(event)
        latency_ms = max(float(event.latency_ms or 0.0), 0.0)
        return end_time - timedelta(milliseconds=latency_ms)

    def _end_time(self, event: TraceEvent) -> datetime:
        return self._event_time(event)

    def _event_time(self, event: TraceEvent) -> datetime:
        try:
            value = datetime.fromisoformat(event.created_at.replace("Z", "+00:00"))
            if value.tzinfo is None:
                return value.replace(tzinfo=timezone.utc)
            return value.astimezone(timezone.utc)
        except (TypeError, ValueError):
            return datetime.now(timezone.utc)

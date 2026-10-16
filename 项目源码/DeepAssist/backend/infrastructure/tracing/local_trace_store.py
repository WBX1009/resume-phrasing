from threading import Lock
from typing import List

from backend.domain.entities.trace_event import TraceEvent
from backend.domain.interfaces.trace_store import TraceStorePort


class LocalTraceStore(TraceStorePort):
    """In-memory trace store for local diagnostics and tests."""

    def __init__(self, max_events: int = 1000):
        self._events: List[TraceEvent] = []
        self._max_events = max_events
        self._lock = Lock()

    def append(self, event: TraceEvent) -> None:
        with self._lock:
            self._events.append(event)
            if len(self._events) > self._max_events:
                self._events = self._events[-self._max_events :]

    def list_by_trace_id(self, trace_id: str) -> List[TraceEvent]:
        with self._lock:
            return [event for event in self._events if event.trace_id == trace_id]

    def list_recent(self, limit: int = 50) -> List[TraceEvent]:
        safe_limit = max(1, min(limit, self._max_events))
        with self._lock:
            return list(self._events[-safe_limit:])

    def clear(self) -> None:
        with self._lock:
            self._events.clear()

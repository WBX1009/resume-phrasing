from typing import List, Protocol

from backend.domain.entities.trace_event import TraceEvent


class TraceStorePort(Protocol):
    """Port for persisting and querying structured trace events."""

    def append(self, event: TraceEvent) -> None:
        ...

    def list_by_trace_id(self, trace_id: str) -> List[TraceEvent]:
        ...

    def list_recent(self, limit: int = 50) -> List[TraceEvent]:
        ...

    def clear(self) -> None:
        ...

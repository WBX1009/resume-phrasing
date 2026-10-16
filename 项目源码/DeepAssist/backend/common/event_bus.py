from collections import defaultdict
from typing import Any, Callable, DefaultDict, Dict, List

from backend.common.logger import get_logger

logger = get_logger(__name__)
Handler = Callable[[Dict[str, Any]], None]


class EventBus:
    """Small in-process pub/sub bus for decoupled background reactions."""

    def __init__(self):
        self._handlers: DefaultDict[str, List[Handler]] = defaultdict(list)

    def subscribe(self, event_name: str, handler: Handler) -> None:
        self._handlers[event_name].append(handler)

    def publish(self, event_name: str, payload: Dict[str, Any]) -> None:
        for handler in self._handlers.get(event_name, []):
            try:
                handler(payload)
            except Exception as exc:
                logger.error("Event handler failed for %s: %s", event_name, exc)


event_bus = EventBus()

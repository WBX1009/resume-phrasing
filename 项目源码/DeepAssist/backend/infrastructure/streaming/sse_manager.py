import json
from typing import Any, Dict, Union

from backend.domain.entities.stream_event import StreamEvent


class SSEManager:
    """SSE transport wrapper for structured JSON events."""

    @staticmethod
    def format_event(event: Union[StreamEvent, Dict[str, Any]]) -> str:
        payload = event.to_payload() if isinstance(event, StreamEvent) else event
        return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

    @staticmethod
    def format_chunk(text: str) -> str:
        return SSEManager.format_event(StreamEvent.message_delta(text))

    @staticmethod
    def format_end() -> str:
        return SSEManager.format_event(StreamEvent.done())

    @staticmethod
    def format_error(err_msg: str) -> str:
        return SSEManager.format_event(StreamEvent.error(err_msg))

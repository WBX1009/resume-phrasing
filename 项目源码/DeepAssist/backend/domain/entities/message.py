from typing import List, Optional

from pydantic import BaseModel, Field

from backend.domain.entities.tooling import ToolCall


class Message(BaseModel):
    """Canonical chat message used by API, services, and persistence."""

    role: str = Field(..., description="Message role: user, assistant, system, or tool")
    content: str = Field(..., description="Message content")
    name: Optional[str] = Field(default=None, description="Tool name for tool messages")
    tool_call_id: Optional[str] = Field(default=None, description="Matching tool call id")


class ChatSession(BaseModel):
    """A complete chat session for history views."""

    session_id: str
    title: str = Field(default="New chat", description="Session title")
    messages: List[Message] = Field(default_factory=list, description="Session messages")


class AIMessage(Message):
    """Assistant message that may contain tool calls."""

    tool_calls: Optional[List[ToolCall]] = Field(default=None, description="Requested tool calls")

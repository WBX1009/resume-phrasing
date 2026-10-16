from abc import ABC, abstractmethod
from typing import Any, Dict, Iterator, List, Optional


class BaseLLM(ABC):
    """Port for chat model adapters."""

    @abstractmethod
    def chat(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Any]] = None,
        model_name: Optional[str] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
    ) -> Any:
        raise NotImplementedError

    @abstractmethod
    def chat_stream(
        self,
        messages: List[Dict[str, Any]],
        model_name: Optional[str] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
    ) -> Iterator[Any]:
        raise NotImplementedError

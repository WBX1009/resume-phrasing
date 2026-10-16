"""Optional OpenAI-compatible chat completion adapter for DeepAssist."""

from __future__ import annotations

from typing import Any, Iterator, Optional

from openai import OpenAI

from backend.common.config import settings
from backend.common.exceptions import ConfigurationError
from backend.common.logger import get_logger
from backend.domain.interfaces.llm import BaseLLM

logger = get_logger(__name__)


class OpenAICompatibleClient(BaseLLM):
    """Call a configured endpoint implementing the OpenAI chat-completions contract.

    DeepAssist's project model remains DeepSeek. This adapter only describes
    the wire protocol used by an endpoint; it is not a second model choice.
    """

    def __init__(
        self,
        model_name: str | None = None,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        client: Any | None = None,
    ):
        self.model_name = model_name or settings.LLM_CHAT_MODEL
        self.base_url = (base_url or settings.OPENAI_COMPATIBLE_BASE_URL or "").strip()
        # The SDK requires a non-empty key even for gateways that authenticate
        # elsewhere; ``EMPTY`` is an explicit placeholder, not a credential.
        self.api_key = (api_key or settings.OPENAI_COMPATIBLE_API_KEY or "EMPTY").strip()
        self.client = client
        logger.info(
            "OpenAI-compatible LLM initialized: model=%s base_url=%s",
            self.model_name,
            self.base_url or "(SDK default)",
        )

    def chat(
        self,
        messages,
        tools=None,
        model_name=None,
        temperature=None,
        top_p=None,
    ) -> Any:
        kwargs: dict[str, Any] = {
            "model": self._resolve_model_name(model_name),
            "messages": messages,
            "temperature": self._resolve_temperature(temperature),
        }
        if top_p is not None:
            kwargs["top_p"] = float(top_p)
        if tools:
            kwargs["tools"] = tools
        try:
            response = self._get_client().chat.completions.create(**kwargs)
            return response.choices[0].message
        except Exception:
            logger.exception("OpenAI-compatible chat call failed")
            raise

    def chat_stream(
        self,
        messages,
        model_name=None,
        temperature=None,
        top_p=None,
    ) -> Iterator[Any]:
        kwargs: dict[str, Any] = {
            "model": self._resolve_model_name(model_name),
            "messages": messages,
            "stream": True,
            "temperature": self._resolve_temperature(temperature),
        }
        if top_p is not None:
            kwargs["top_p"] = float(top_p)
        try:
            response = self._get_client().chat.completions.create(**kwargs)
            for chunk in response:
                if not getattr(chunk, "choices", None):
                    continue
                delta = chunk.choices[0].delta
                reasoning = getattr(delta, "reasoning_content", None)
                if reasoning:
                    yield {"type": "reasoning", "content": reasoning}
                content = getattr(delta, "content", None)
                if content:
                    yield {"type": "content", "content": content}
        except Exception:
            logger.exception("OpenAI-compatible streaming call failed")
            raise

    def _get_client(self) -> OpenAI:
        if self.client is not None:
            return self.client
        if not self.base_url:
            raise ConfigurationError(
                "OPENAI_COMPATIBLE_BASE_URL is not configured. Set it to the "
                "server's OpenAI-compatible /v1 endpoint."
            )
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        return self.client

    def _resolve_model_name(self, model_name: Optional[str]) -> str:
        return (model_name or self.model_name or "").strip()

    def _resolve_temperature(self, temperature: Optional[float]) -> float:
        if temperature is None:
            return 0.7
        return max(0.0, min(2.0, float(temperature)))

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
import re
import time
from typing import Any

from openai import OpenAI


_LLM_JSON_CACHE: dict[tuple[str, str, str], dict[str, Any]] = {}


@dataclass(frozen=True)
class LlmCallResult:
    payload: dict[str, Any] | None
    content: str
    reasoning_content: str
    finish_reason: str
    cached: bool = False


def model_candidates(default_model: str, role: str) -> list[str]:
    """Return ordered model candidates for a task role.

    Role-specific env vars can override the front of the list without removing
    conservative defaults. This keeps old .env files working while allowing the
    report pipeline to route Planner/Judge/Narrative tasks differently.
    """

    role_key = role.upper().replace("-", "_")
    role_models = _split_models(os.getenv(f"LLM_{role_key}_MODELS"))
    single_role_models = _split_models(os.getenv(f"LLM_{role_key}_MODEL"))
    legacy_role_models = _split_models(os.getenv(f"QIANFAN_{role_key}_MODEL"))
    fallback_models = _split_models(os.getenv("QIANFAN_FALLBACK_MODELS"))

    if role_key == "PLANNER":
        defaults = ["deepseek-v4-pro", "deepseek-v3.2", "glm-5.1", "deepseek-v4-flash"]
    elif role_key == "JUDGE":
        defaults = ["deepseek-v4-pro", "glm-5.1", "deepseek-v3.2", "deepseek-v4-flash"]
    elif role_key == "NARRATIVE":
        defaults = ["glm-5.1", "deepseek-v4-pro", "ernie-5.1", "deepseek-v4-flash"]
    else:
        defaults = ["glm-5.1", "deepseek-v4-pro", "deepseek-v4-flash"]

    configured = _unique(
        [
            *role_models,
            *single_role_models,
            *legacy_role_models,
            default_model,
        ]
    )
    if os.getenv("LLM_STRICT_MODEL_ROUTING", "").strip().lower() in {"1", "true", "yes"}:
        return _unique([*role_models, *single_role_models, default_model])

    return _unique(
        [
            *configured,
            *fallback_models,
            *defaults,
        ]
    )


def cached_chat_json(
    config: Any,
    *,
    model_name: str,
    role: str,
    messages: list[dict[str, str]],
    max_tokens: int,
    temperature: float | None = None,
    timeout: float = 60.0,
) -> LlmCallResult:
    """Call chat.completions and parse JSON with process-level caching."""

    cache_key = (
        model_name,
        role,
        hashlib.sha256(json.dumps(messages, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest(),
    )
    cached = _LLM_JSON_CACHE.get(cache_key)
    if cached is not None:
        return LlmCallResult(
            payload=cached.get("payload"),
            content=str(cached.get("content") or ""),
            reasoning_content=str(cached.get("reasoning_content") or ""),
            finish_reason=str(cached.get("finish_reason") or ""),
            cached=True,
        )

    client = OpenAI(api_key=config.llm.api_key, base_url=config.llm.base_url, timeout=timeout)
    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=config.llm.temperature if temperature is None else temperature,
        max_tokens=max_tokens,
    )
    choice = response.choices[0]
    message = choice.message
    content = message.content or ""
    reasoning = getattr(message, "reasoning_content", "") or ""
    payload = extract_json(content)
    result = {
        "payload": payload,
        "content": content,
        "reasoning_content": reasoning,
        "finish_reason": str(getattr(choice, "finish_reason", "") or ""),
    }
    _LLM_JSON_CACHE[cache_key] = result
    return LlmCallResult(
        payload=payload,
        content=content,
        reasoning_content=reasoning,
        finish_reason=result["finish_reason"],
        cached=False,
    )


def update_audit_from_result(audit: dict[str, Any], result: LlmCallResult, model_name: str, attempt: int) -> None:
    audit["llm_content_chars"] = len(result.content)
    audit["llm_reasoning_chars"] = len(result.reasoning_content)
    audit["llm_finish_reason"] = result.finish_reason
    audit["llm_attempts"] = attempt
    audit["last_attempt_model"] = model_name
    audit["llm_cache_hit"] = result.cached
    if result.finish_reason == "length":
        audit["llm_output_warning"] = "finish_reason=length; output may be truncated"


def retry_delay_seconds(exc: Exception, attempt: int) -> int:
    retry_after = _retry_after_from_exception(exc)
    if retry_after is not None:
        return min(max(retry_after, 1), 120)
    text = f"{type(exc).__name__}: {exc}".lower()
    if "429" in text or "rate" in text or "too many" in text:
        return min(15 * max(attempt, 1), 90)
    return min(4 * max(attempt, 1), 20)


def extract_json(text: str) -> dict[str, Any]:
    text = text.strip()
    if not text:
        raise ValueError("LLM returned empty content")
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text).strip()
        text = re.sub(r"```$", "", text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.S)
        if match:
            return json.loads(match.group(0))
        raise


def sleep_before_retry(exc: Exception, attempt: int) -> int:
    delay = retry_delay_seconds(exc, attempt)
    time.sleep(delay)
    return delay


def _split_models(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def _unique(items: list[str]) -> list[str]:
    result: list[str] = []
    for item in items:
        if item and item not in result:
            result.append(item)
    return result


def _retry_after_from_exception(exc: Exception) -> int | None:
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if not headers:
        return None
    value = None
    try:
        value = headers.get("retry-after") or headers.get("Retry-After")
    except Exception:
        return None
    try:
        return int(float(value))
    except Exception:
        return None

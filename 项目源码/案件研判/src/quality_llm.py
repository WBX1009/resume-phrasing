"""Optional bounded LLM adapter used only for advisory structured planning."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from threading import Lock
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable


@dataclass
class BudgetLedger:
    max_calls: int = 200
    max_tokens: int = 2_000_000
    calls_used: int = 0
    tokens_used: int = 0
    _lock: Lock = field(default_factory=Lock, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        for name in ("max_calls", "max_tokens", "calls_used", "tokens_used"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")

    def reserve(self, estimated_tokens: int) -> None:
        with self._lock:
            if not isinstance(estimated_tokens, int) or isinstance(estimated_tokens, bool) or estimated_tokens < 0:
                raise ValueError("estimated token count must be a non-negative integer")
            if self.calls_used + 1 > self.max_calls:
                raise RuntimeError("LLM call budget exhausted")
            if self.tokens_used + estimated_tokens > self.max_tokens:
                raise RuntimeError("LLM token budget exhausted")
            self.calls_used += 1
            self.tokens_used += estimated_tokens

    def reconcile(self, estimated_tokens: int, actual_tokens: int) -> None:
        with self._lock:
            if not isinstance(actual_tokens, int) or isinstance(actual_tokens, bool) or actual_tokens < 0:
                raise ValueError("actual token count must be a non-negative integer")
            self.tokens_used = max(0, self.tokens_used - estimated_tokens + max(0, actual_tokens))
            if self.tokens_used > self.max_tokens:
                raise RuntimeError("LLM token budget exceeded by provider usage")

    def to_dict(self) -> dict[str, int]:
        with self._lock:
            return {
                "max_calls": self.max_calls,
                "max_tokens": self.max_tokens,
                "calls_used": self.calls_used,
                "tokens_used": self.tokens_used,
            }


@dataclass
class LLMResult:
    status: str
    payload: dict[str, Any]
    cache_key: str
    calls_used: int
    tokens_used: int
    reason: str | None = None


class BoundedLLM:
    """A cache-first OpenAI-compatible adapter with a strict budget ledger.

    The adapter is deliberately advisory.  Its output is never accepted as a
    Fact; callers may only use the returned action names after validating them
    against a profile's tool whitelist.
    """

    def __init__(
        self,
        cache_dir: Path | str,
        ledger: BudgetLedger | None = None,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        transport: Callable[[str, str, int], tuple[str, int]] | None = None,
    ) -> None:
        try:
            from dotenv import load_dotenv

            load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=False)
        except ImportError:
            pass
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.ledger = ledger or BudgetLedger()
        # Prefer the explicit primary-provider variables used by the current
        # .env.example, while retaining the legacy QUALITY_LLM/QIANFAN names.
        self.provider = (
            os.getenv("QUALITY_LLM_PROVIDER")
            or os.getenv("LLM_PRIMARY_PROVIDER")
            or os.getenv("LLM_PROVIDER")
            or "openai-compatible"
        )
        self.model = (
            model
            or os.getenv("QUALITY_LLM_MODEL")
            or os.getenv("LLM_PRIMARY_MODEL")
            or os.getenv("OPENAI_MODEL_NAME")
            or os.getenv("QIANFAN_DEFAULT_MODEL")
            or "qianfan-code-latest"
        )
        self.api_key = (
            api_key
            or os.getenv("QUALITY_LLM_API_KEY")
            or os.getenv("LLM_PRIMARY_API_KEY")
            or os.getenv("OPENAI_API_KEY")
            or os.getenv("QIANFAN_API_KEY")
        )
        self.base_url = (
            base_url
            or os.getenv("QUALITY_LLM_BASE_URL")
            or os.getenv("LLM_PRIMARY_BASE_URL")
            or os.getenv("OPENAI_BASE_URL")
            or os.getenv("QIANFAN_OPENAI_BASE_URL")
        )
        self.transport = transport

    def _key(self, system: str, user: str, max_tokens: int) -> str:
        raw = json.dumps({"model": self.model, "system": system, "user": user, "max_tokens": max_tokens}, ensure_ascii=False, sort_keys=True)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _cache_path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.json"

    @staticmethod
    def _write_cache(path: Path, payload: dict[str, Any]) -> None:
        """Publish a cache entry atomically, even when processes race."""
        fd, temporary_name = tempfile.mkstemp(
            prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False, indent=2)
            # Windows can briefly deny a replace while another process is
            # publishing the same key. Retry a bounded number of times; the
            # unique temporary file keeps each attempt isolated.
            for attempt in range(6):
                try:
                    temporary.replace(path)
                    break
                except PermissionError:
                    if attempt == 5:
                        raise
                    time.sleep(0.01 * (attempt + 1))
        except Exception:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass
            raise

    def _transport(self, system: str, user: str, max_tokens: int) -> tuple[str, int]:
        if self.transport is not None:
            return self.transport(system, user, max_tokens)
        if not self.api_key:
            raise RuntimeError("no LLM API key configured")
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("openai package is not installed") from exc
        client = OpenAI(api_key=self.api_key, base_url=self.base_url or None)
        response = client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=0,
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
        )
        text = response.choices[0].message.content or "{}"
        usage = getattr(response, "usage", None)
        tokens = int(getattr(usage, "total_tokens", 0) or 0)
        return text, tokens

    def complete_json(self, system: str, user: str, max_tokens: int = 1024) -> LLMResult:
        if not isinstance(max_tokens, int) or isinstance(max_tokens, bool) or max_tokens <= 0:
            invalid_key = hashlib.sha256(f"invalid-max-tokens:{max_tokens!r}".encode("utf-8")).hexdigest()
            return LLMResult(
                "unavailable",
                {},
                invalid_key,
                0,
                0,
                "max_tokens must be a positive integer",
            )
        key = self._key(system, user, max_tokens)
        cache_path = self._cache_path(key)
        if cache_path.exists():
            try:
                cached = json.loads(cache_path.read_text(encoding="utf-8"))
                payload = cached["payload"]
                if not isinstance(payload, dict):
                    raise ValueError("cached LLM payload must be an object")
                return LLMResult("cache", payload, key, 0, 0)
            except (OSError, KeyError, json.JSONDecodeError):
                cache_path.unlink(missing_ok=True)
            except ValueError:
                cache_path.unlink(missing_ok=True)
        estimate = min(max_tokens + max(1, len(user) // 4) + len(system) // 4, 20_000)
        try:
            self.ledger.reserve(estimate)
            text, actual_tokens = self._transport(system, user, max_tokens)
            payload = json.loads(text)
            if not isinstance(payload, dict):
                raise ValueError("LLM JSON response must be an object")
            self.ledger.reconcile(estimate, actual_tokens)
            self._write_cache(cache_path, {"payload": payload})
            return LLMResult("online", payload, key, 1, actual_tokens)
        except Exception as exc:
            # Keep the failure visible to the trace while allowing the
            # deterministic pipeline to complete safely.
            return LLMResult("unavailable", {}, key, 0, 0, f"{type(exc).__name__}: {exc}")

    def advisory_plan(self, profile: dict[str, Any], facts: list[dict[str, Any]]) -> LLMResult:
        system = (
            "You are a constrained investigation planner. Return JSON only with an actions array. "
            "Choose only tool names present in allowed_tools. Never calculate facts or invent entities."
        )
        user = json.dumps(
            {
                "model_id": profile.get("model_id"),
                "description": profile.get("description", ""),
                "allowed_tools": profile.get("specialized_tools", []),
                "reasoning_mode": profile.get("reasoning_mode"),
                "facts": facts[:20],
            },
            ensure_ascii=False,
        )
        result = self.complete_json(system, user)
        if result.status in {"online", "cache"}:
            actions = result.payload.get("actions", [])
            allowed = set(profile.get("specialized_tools", []))
            if not isinstance(actions, list) or any(action not in allowed for action in actions):
                return LLMResult("invalid", {}, result.cache_key, result.calls_used, result.tokens_used, "tool whitelist violation")
            if len(actions) != len(set(actions)):
                return LLMResult("invalid", {}, result.cache_key, result.calls_used, result.tokens_used, "duplicate tool action")
            result.payload = {"actions": actions}
        return result

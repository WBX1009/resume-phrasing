from __future__ import annotations

from hashlib import sha256
from typing import Any
import json


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def stable_digest(value: Any, length: int = 16) -> str:
    return sha256(canonical_json(value).encode("utf-8")).hexdigest()[:length]


def stable_id(prefix: str, value: Any, length: int = 16) -> str:
    return f"{prefix}_{stable_digest(value, length)}"

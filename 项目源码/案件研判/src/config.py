from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os

from dotenv import dotenv_values


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_IDS = [5, 29, 43, 48, 71]


@dataclass(frozen=True)
class DbConfig:
    host: str | None
    port: int | None
    user: str | None
    password: str | None
    database: str | None
    schema: str
    ssl_mode: str

    @property
    def is_complete(self) -> bool:
        return all([self.host, self.port, self.user, self.password, self.database])


@dataclass(frozen=True)
class LlmConfig:
    """单个 LLM 接入配置。"""

    provider: str
    api_key: str | None
    base_url: str | None
    model: str
    max_tokens: int
    temperature: float

    @property
    def is_complete(self) -> bool:
        """模型名 + api_key + base_url 全部非空才算配置完整。"""
        return bool(self.model and self.api_key and self.base_url)


@dataclass(frozen=True)
class AppConfig:
    root_dir: Path
    output_dir: Path
    snapshot_dir: Path | None
    categories_path: Path
    default_model_ids: list[int]
    db: DbConfig
    primary_llm: LlmConfig
    fallback_llm: LlmConfig | None

    @property
    def llm(self) -> LlmConfig:
        """向后兼容：返回主模型配置。"""
        return self.primary_llm


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _int_or_none(value: str | None) -> int | None:
    if not value:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _float_or_default(value: str | None, default: float) -> float:
    if not value:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _int_or_default(value: str | None, default: int) -> int:
    parsed = _int_or_none(value)
    return parsed if parsed is not None else default


def find_snapshot_dir(root_dir: Path = ROOT_DIR) -> Path | None:
    candidates = [
        path
        for path in root_dir.iterdir()
        if path.is_dir() and (path / "ch_model_info_data_sample.jsonl").exists()
    ]
    return candidates[0] if candidates else None


def _load_env() -> None:
    """将 .env 写入 os.environ（已有值不覆盖）。"""
    for key, value in dotenv_values(ROOT_DIR / ".env", encoding="utf-8-sig").items():
        if key and value is not None:
            os.environ.setdefault(key.lstrip("﻿"), value)


def _build_llm_config(
    prefix: str,
    *,
    default_provider: str = "",
    default_model: str = "",
    default_max_tokens: int = 4096,
    default_temperature: float = 0.0,
) -> LlmConfig:
    """从 {prefix}_PROVIDER / _MODEL / _API_KEY / _BASE_URL / _MAX_TOKENS 构建 LlmConfig。"""
    provider = os.getenv(f"{prefix}_PROVIDER") or default_provider
    model = os.getenv(f"{prefix}_MODEL") or default_model
    api_key = os.getenv(f"{prefix}_API_KEY")
    base_url = os.getenv(f"{prefix}_BASE_URL")
    max_tokens = _int_or_default(os.getenv(f"{prefix}_MAX_TOKENS"), default_max_tokens)
    temperature = _float_or_default(os.getenv("LLM_TEMPERATURE"), default_temperature)

    return LlmConfig(
        provider=provider,
        api_key=api_key,
        base_url=base_url,
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
    )


def _resolve_llm_configs() -> tuple[LlmConfig, LlmConfig | None]:
    """解析主模型和备用模型配置，如果新版变量未设置则回退到旧版兼容变量。"""

    temperature = _float_or_default(os.getenv("LLM_TEMPERATURE"), 0.0)
    default_max_tokens = _int_or_default(os.getenv("LLM_DEFAULT_MAX_TOKENS"), 4096)

    # -- 主模型 --
    if os.getenv("LLM_PRIMARY_PROVIDER") and os.getenv("LLM_PRIMARY_MODEL"):
        primary = _build_llm_config(
            "LLM_PRIMARY",
            default_max_tokens=default_max_tokens,
            default_temperature=temperature,
        )
    else:
        # 回退到旧版兼容变量
        provider = os.getenv("LLM_PROVIDER") or "qianfan"
        model = (
            os.getenv("QIANFAN_DEFAULT_MODEL")
            or os.getenv("OPENAI_MODEL_NAME")
            or "qianfan-code-latest"
        )
        api_key = os.getenv("QIANFAN_API_KEY") or os.getenv("OPENAI_API_KEY")
        base_url = os.getenv("QIANFAN_OPENAI_BASE_URL") or os.getenv("OPENAI_BASE_URL")
        primary = LlmConfig(
            provider=provider,
            api_key=api_key,
            base_url=base_url,
            model=model,
            max_tokens=default_max_tokens,
            temperature=temperature,
        )

    # -- 备用模型 --
    if os.getenv("LLM_FALLBACK_PROVIDER") and os.getenv("LLM_FALLBACK_MODEL"):
        fallback = _build_llm_config(
            "LLM_FALLBACK",
            default_max_tokens=default_max_tokens,
            default_temperature=temperature,
        )
    else:
        fallback = None

    return primary, fallback


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------


def load_config() -> AppConfig:
    _load_env()

    primary_llm, fallback_llm = _resolve_llm_configs()

    return AppConfig(
        root_dir=ROOT_DIR,
        output_dir=ROOT_DIR / "output",
        snapshot_dir=find_snapshot_dir(ROOT_DIR),
        categories_path=ROOT_DIR / "output" / "case2_model_categories.csv",
        default_model_ids=DEFAULT_MODEL_IDS,
        db=DbConfig(
            host=os.getenv("GENERAL_DB_HOST"),
            port=_int_or_none(os.getenv("GENERAL_DB_PORT")),
            user=os.getenv("GENERAL_DB_USERNAME"),
            password=os.getenv("GENERAL_DB_PASSWORD"),
            database=os.getenv("GENERAL_DB_DATABASE"),
            schema=os.getenv("GENERAL_DB_SCHEMA") or "public",
            ssl_mode=os.getenv("DB_SSL_MODE") or "prefer",
        ),
        primary_llm=primary_llm,
        fallback_llm=fallback_llm,
    )
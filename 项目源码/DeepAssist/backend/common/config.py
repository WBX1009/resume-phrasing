from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class Settings(BaseSettings):
    # Model credentials and endpoints.
    # DeepAssist runs with DeepSeek by default. ``openai_compatible`` is an
    # optional transport setting for another endpoint and does not change the
    # project model declaration.
    LLM_PROVIDER: str = Field(default="deepseek")
    OPENAI_COMPATIBLE_API_KEY: str = Field(default="EMPTY")
    OPENAI_COMPATIBLE_BASE_URL: str = Field(default="http://localhost:8000/v1")
    DEEPSEEK_API_KEY: str = Field(default="")
    DEEPSEEK_BASE_URL: str = Field(default="https://api.deepseek.com")

    # Model names.
    LLM_CHAT_MODEL: str = "deepseek-chat"
    LLM_REASONER_MODEL: str = "deepseek-reasoner"
    ROUTER_MODEL_NAME: str = ""
    ROUTER_TEMPERATURE: float = Field(default=0.0)
    ROUTER_MIN_CONFIDENCE: float = Field(default=0.65)
    ENABLE_ROUTING_LOG: bool = Field(default=True)

    # Local paths.
    EMBEDDING_MODEL_PATH: str = Field(default=str(PROJECT_ROOT / "data" / "models" / "bge-m3"))
    VECTOR_DB_PATH: str = Field(default=str(PROJECT_ROOT / "data" / "indexes" / "vector_store"))
    KEYWORD_DB_PATH: str = Field(default=str(PROJECT_ROOT / "data" / "indexes" / "keyword_store"))
    CONVERSATION_DB_PATH: str = Field(default=str(PROJECT_ROOT / "data" / "application_db" / "conversations.db"))
    USER_PROFILE_DB_PATH: str = Field(default=str(PROJECT_ROOT / "data" / "application_db" / "user_profiles.db"))
    INGEST_PROGRESS_PATH: str = Field(default=str(PROJECT_ROOT / "data" / "application_db" / "ingest_progress.json"))
    KB_MANIFEST_PATH: str = Field(default=str(PROJECT_ROOT / "data" / "application_db" / "knowledge_base_manifest.json"))
    VECTOR_HEALTH_REPORT_PATH: str = Field(default=str(PROJECT_ROOT / "data" / "application_db" / "vector_index_doctor_report.json"))
    AGENT_SANDBOX_DIR: str = Field(default=str(PROJECT_ROOT / "workspace" / "agent_sandbox"))

    # Embedding runtime settings. auto prefers CUDA when available and falls back to CPU.
    EMBEDDING_DEVICE: str = Field(default="auto")
    EMBEDDING_CUDA_VISIBLE_DEVICES: str = Field(default="")
    EMBEDDING_BATCH_SIZE: int = Field(default=64)

    # Retrieval settings.
    KEYWORD_BACKEND: str = Field(default="elasticsearch")
    ELASTICSEARCH_HOSTS: str = Field(default="http://localhost:9200")
    ELASTICSEARCH_INDEX_PREFIX: str = Field(default="deepassist-kb")
    ELASTICSEARCH_USERNAME: str = Field(default="")
    ELASTICSEARCH_PASSWORD: str = Field(default="")

    # Memory storage settings.
    MEMORY_BACKEND: str = Field(default="sqlite")
    MYSQL_HOST: str = Field(default="localhost")
    MYSQL_PORT: int = Field(default=3306)
    MYSQL_USER: str = Field(default="deepassist")
    MYSQL_PASSWORD: str = Field(default="")
    MYSQL_DATABASE: str = Field(default="deepassist")

    RRF_K: int = 60
    RETRIEVAL_TOP_K: int = 5
    ENABLE_LLM_REWRITE: bool = Field(default=True)
    RAG_LOW_RELEVANCE_THRESHOLD: float = Field(default=0.2)

    # Reranker backend: cross_encoder | lexical | noop.
    RERANKER_BACKEND: str = Field(default="cross_encoder")
    RERANKER_MODEL_NAME: str = Field(default="BAAI/bge-reranker-base")
    RERANKER_DEVICE: str = Field(default="auto")
    RERANKER_BATCH_SIZE: int = Field(default=16)

    MAX_AGENT_STEPS: int = 10
    AGENT_RUNTIME_BACKEND: str = Field(default="langgraph")
    LANGGRAPH_CHECKPOINT_BACKEND: str = Field(default="sqlite")
    LANGGRAPH_CHECKPOINT_DB_PATH: str = Field(
        default=str(PROJECT_ROOT / "data" / "application_db" / "langgraph_checkpoints.db")
    )
    # Local development exposes high-risk tool schemas, but HITL still blocks
    # execution unless TOOL_APPROVAL_MODE=disabled.
    ENABLE_DEVELOPMENT_TOOLS: bool = Field(default=True)
    ALLOWED_TOOL_CATEGORIES: str = "read_only,write_file,code_exec"
    TOOL_APPROVAL_MODE: str = Field(default="hitl")
    TOOL_TIMEOUT_SECONDS: float = Field(default=30.0)

    # Observability is local/offline by default.  When enabled with an API
    # key, the infrastructure trace adapter mirrors request/node/tool events
    # to LangSmith while retaining the local queryable trace store.
    LANGSMITH_TRACING: bool = Field(default=False)
    LANGSMITH_API_KEY: str = Field(default="")
    LANGSMITH_PROJECT: str = Field(default="deepassist")
    LANGSMITH_ENDPOINT: str = Field(default="")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @field_validator(
        "EMBEDDING_MODEL_PATH",
        "VECTOR_DB_PATH",
        "KEYWORD_DB_PATH",
        "CONVERSATION_DB_PATH",
        "USER_PROFILE_DB_PATH",
        "INGEST_PROGRESS_PATH",
        "KB_MANIFEST_PATH",
        "VECTOR_HEALTH_REPORT_PATH",
        "AGENT_SANDBOX_DIR",
        "LANGGRAPH_CHECKPOINT_DB_PATH",
        mode="after",
    )
    @classmethod
    def resolve_relative_path(cls, value: str) -> str:
        if value and value.startswith("./"):
            return str((PROJECT_ROOT / value[2:]).resolve())
        return value


settings = Settings()

"""Centralised, validated, environment-driven configuration for AKGPT.

All secrets are read from environment variables (or a local ``.env`` file) and are
held as ``SecretStr`` so they never appear in ``repr()``, logs or API responses.
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from exceptions import ConfigurationError

DEFAULT_GROQ_MODELS = "openai/gpt-oss-120b,openai/gpt-oss-20b,qwen/qwen3.6-27b"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # --- LLM provider -------------------------------------------------------
    llm_provider: str = "groq"
    groq_api_key: SecretStr | None = None
    groq_model: str = "openai/gpt-oss-120b"
    groq_models: str = DEFAULT_GROQ_MODELS  # comma separated selectable models
    llm_timeout_seconds: float = Field(60.0, gt=0, le=600)
    llm_max_retries: int = Field(2, ge=0, le=6)

    # --- External tools -----------------------------------------------------
    tavily_api_key: SecretStr | None = None
    tavily_max_results: int = Field(5, ge=1, le=10)
    openweather_api_key: SecretStr | None = None
    weather_units: Literal["metric", "imperial", "standard"] = "metric"
    tool_timeout_seconds: float = Field(20.0, gt=0, le=120)

    # --- LangSmith (optional) ----------------------------------------------
    langsmith_tracing: bool = True
    langsmith_endpoint: str = "https://api.smith.langchain.com"
    langsmith_api_key: SecretStr | None = None
    langsmith_project: str = "AKGPT"
    langsmith_hide_inputs: bool = True
    langsmith_hide_outputs: bool = False

    # --- Storage ------------------------------------------------------------
    database_url: str = "sqlite:///./data/akgpt.db"
    checkpoint_db_path: str = "./data/langgraph_checkpoints.sqlite"
    faiss_index_path: str = "./faiss_db"
    upload_dir: str = "./uploads"

    # --- RAG ----------------------------------------------------------------
    embedding_provider: Literal["sentence-transformers", "hash"] = "sentence-transformers"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dimension: int = Field(384, ge=8, le=4096)  # used by the offline "hash" provider
    reranker_model: str | None = None  # optional cross-encoder, e.g. cross-encoder/ms-marco-MiniLM-L-6-v2
    chunk_size: int = Field(1000, ge=100, le=8000)
    chunk_overlap: int = Field(200, ge=0)
    top_k: int = Field(5, ge=1, le=20)
    rag_min_score: float = Field(0.15, ge=-1.0, le=1.0)
    max_upload_size_mb: int = Field(20, ge=1, le=200)
    max_pdf_pages: int = Field(500, ge=1)
    max_extracted_chars: int = Field(5_000_000, ge=1000)

    # --- Server -------------------------------------------------------------
    host: str = "0.0.0.0"
    port: int = Field(8080, ge=1, le=65535)
    app_env: Literal["development", "production", "test"] = "development"
    log_level: str = "INFO"
    allowed_origins: str = "http://localhost:8080"
    cookie_secure: bool = False
    max_request_bytes: int = Field(256 * 1024, ge=1024)  # non-upload JSON bodies
    rate_limit_per_minute: int = Field(60, ge=1)  # per client, chat/upload endpoints

    # --- Agent --------------------------------------------------------------
    max_history_messages: int = Field(20, ge=2, le=200)
    max_agent_steps: int = Field(8, ge=1, le=25)
    max_message_chars: int = Field(8000, ge=1)
    chat_timeout_seconds: float = Field(180.0, gt=0)
    default_temperature: float = Field(0.3, ge=0.0, le=1.0)

    # ------------------------------------------------------------------ validators
    @field_validator("log_level")
    @classmethod
    def _upper_log(cls, v: str) -> str:
        v = v.upper()
        if v not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
            raise ValueError("LOG_LEVEL must be one of DEBUG, INFO, WARNING, ERROR, CRITICAL")
        return v

    @field_validator("groq_api_key", "tavily_api_key", "openweather_api_key", "langsmith_api_key", mode="before")
    @classmethod
    def _blank_secret_is_none(cls, v):
        if v is None:
            return None
        if isinstance(v, str) and not v.strip():
            return None
        return v

    @field_validator("reranker_model", mode="before")
    @classmethod
    def _blank_reranker(cls, v):
        return v or None

    @model_validator(mode="after")
    def _cross_checks(self) -> "Settings":
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE")
        if not self.database_url.startswith("sqlite"):
            raise ValueError("DATABASE_URL must be a sqlite:/// URL (AKGPT uses SQLite)")
        return self

    # ------------------------------------------------------------------ helpers
    @property
    def is_production(self) -> bool:
        return self.app_env == "production"

    @property
    def cors_origins(self) -> list[str]:
        origins = [o.strip().rstrip("/") for o in self.allowed_origins.split(",") if o.strip()]
        if "*" in origins and self.is_production:
            raise ConfigurationError("ALLOWED_ORIGINS may not contain '*' in production.")
        return origins

    @property
    def model_options(self) -> list[str]:
        models = [m.strip() for m in self.groq_models.split(",") if m.strip()]
        if self.groq_model not in models:
            models.insert(0, self.groq_model)
        return models

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_size_mb * 1024 * 1024

    @property
    def sqlite_path(self) -> Path:
        """Filesystem path of the SQLite database (``:memory:`` is not supported)."""
        url = self.database_url
        for prefix in ("sqlite+pysqlite:///", "sqlite:///"):
            if url.startswith(prefix):
                return Path(url[len(prefix):]).expanduser()
        raise ConfigurationError("Unsupported DATABASE_URL; use sqlite:///relative/or/absolute/path.db")

    @property
    def upload_path(self) -> Path:
        return Path(self.upload_dir).expanduser()

    @property
    def faiss_path(self) -> Path:
        return Path(self.faiss_index_path).expanduser()

    def key_status(self) -> dict[str, bool]:
        """Which optional integrations are configured (booleans only - never the keys)."""
        return {
            "groq": self.groq_api_key is not None,
            "tavily": self.tavily_api_key is not None,
            "openweather": self.openweather_api_key is not None,
            "langsmith": self.langsmith_tracing and self.langsmith_api_key is not None,
        }

    def startup_warnings(self) -> list[str]:
        warnings: list[str] = []
        if self.groq_api_key is None:
            warnings.append("GROQ_API_KEY is not set: chat requests will fail until it is configured (see .env.example).")
        if self.tavily_api_key is None:
            warnings.append("TAVILY_API_KEY is not set: the web_search tool is disabled.")
        if self.openweather_api_key is None:
            warnings.append("OPENWEATHER_API_KEY is not set: the get_weather tool is disabled.")
        if self.langsmith_tracing and self.langsmith_api_key is None:
            warnings.append("LANGSMITH_TRACING=true but LANGSMITH_API_KEY is empty: tracing is disabled.")
        if self.is_production and not self.cookie_secure:
            warnings.append("COOKIE_SECURE=false in production: enable it when serving over HTTPS.")
        return warnings

    def __repr__(self) -> str:  # never leak secrets, even through accidental repr
        return f"Settings(app_env={self.app_env!r}, keys={self.key_status()})"

    __str__ = __repr__


def configure_langsmith(settings: Settings) -> bool:
    """Apply LangSmith tracing configuration to the process environment.

    LangSmith is strictly optional: if tracing is off or the key is missing we
    explicitly switch tracing off so nothing is sent. Returns True when enabled.
    """
    enabled = settings.langsmith_tracing and settings.langsmith_api_key is not None
    if enabled:
        os.environ["LANGSMITH_TRACING"] = "true"
        os.environ["LANGSMITH_API_KEY"] = settings.langsmith_api_key.get_secret_value()  # type: ignore[union-attr]
        os.environ["LANGSMITH_ENDPOINT"] = settings.langsmith_endpoint
        os.environ["LANGSMITH_PROJECT"] = settings.langsmith_project
        # Privacy: do not ship raw inputs (user prompts / documents) by default.
        os.environ["LANGSMITH_HIDE_INPUTS"] = "true" if settings.langsmith_hide_inputs else "false"
        os.environ["LANGSMITH_HIDE_OUTPUTS"] = "true" if settings.langsmith_hide_outputs else "false"
    else:
        os.environ["LANGSMITH_TRACING"] = "false"
        os.environ["LANGCHAIN_TRACING_V2"] = "false"
    return enabled


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    try:
        return Settings()
    except ConfigurationError:
        raise
    except Exception as exc:  # pydantic ValidationError -> actionable message, no secret values
        errors = getattr(exc, "errors", None)
        if callable(errors):
            lines = [f"  - {'.'.join(str(p) for p in e['loc'])}: {e['msg']}" for e in errors()]
            raise ConfigurationError("Invalid AKGPT configuration:\n" + "\n".join(lines)) from None
        raise ConfigurationError(f"Invalid AKGPT configuration: {type(exc).__name__}") from None


def reset_settings_cache() -> None:
    get_settings.cache_clear()

"""Application configuration via pydantic-settings.

All runtime configuration is centralized here and loaded from environment
variables (or a `.env` file). No module may read `os.environ` directly.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Root settings object. See `.env.example` for documentation."""

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # --- Application ---
    APP_NAME: str = "Multi-Agent Research & Intelligence Platform"
    APP_ENV: Literal["development", "production", "test"] = "development"
    APP_DEBUG: bool = True
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    CORS_ORIGINS: str = "http://localhost:5173,http://localhost:3000"

    # --- Database ---
    DATABASE_URL: str = "sqlite+aiosqlite:///./data/research_platform.db"

    # --- LLM ---
    LLM_PROVIDER: Literal["openai", "groq", "mock"] = "mock"
    LLM_MODEL: str = "gpt-4o-mini"
    LLM_API_KEY: str = ""
    LLM_TEMPERATURE: float = 0.2
    LLM_MAX_OUTPUT_TOKENS: int = 2000
    LLM_TIMEOUT_SECONDS: int = 60
    LLM_MAX_RETRIES: int = 2

    # Alternate provider (used when LLM_PROVIDER=groq)
    GROQ_API_KEY: str = ""
    GROQ_MODEL: str = "llama-3.3-70b-versatile"

    # --- Embeddings ---
    EMBEDDING_PROVIDER: Literal["openai", "mock"] = "mock"
    EMBEDDING_MODEL: str = "text-embedding-3-small"
    EMBEDDING_DIM: int = 256

    # --- Web search ---
    WEB_SEARCH_PROVIDER: Literal["duckduckgo", "offline"] = "offline"
    WEB_SEARCH_MAX_RESULTS: int = 8
    WEB_FETCH_TIMEOUT_SECONDS: int = 15

    # --- Research budget / cost control ---
    RESEARCH_MAX_ITERATIONS: int = 3
    RESEARCH_MAX_SEARCHES: int = 12
    RESEARCH_MAX_SOURCES: int = 15
    RESEARCH_MAX_TOKENS: int = 150_000
    RESEARCH_MAX_RUNTIME_SECONDS: int = 600

    # --- Storage ---
    UPLOAD_DIR: str = "./data/uploads"
    MAX_UPLOAD_SIZE_MB: int = 20
    VECTOR_STORE_DIR: str = "./data/vector_store"

    # --- Observability ---
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: Literal["json", "console"] = "json"

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.CORS_ORIGINS.split(",") if o.strip()]

    @property
    def max_upload_bytes(self) -> int:
        return self.MAX_UPLOAD_SIZE_MB * 1024 * 1024


@lru_cache
def get_settings() -> Settings:
    """Cached settings accessor (import this everywhere)."""
    return Settings()


# Depth profiles are plain data — keep them in a module-level registry so they
# are versioned with code and trivially testable.
DEPTH_PROFILES: dict[str, dict] = {
    "quick": {
        "max_subquestions": 3,
        "queries_per_subquestion": 1,
        "max_sources_per_query": 3,
        "max_evidence_per_subquestion": 5,
        "critic_passes": 1,
        "allow_followup_search": False,
    },
    "standard": {
        "max_subquestions": 5,
        "queries_per_subquestion": 2,
        "max_sources_per_query": 5,
        "max_evidence_per_subquestion": 8,
        "critic_passes": 2,
        "allow_followup_search": True,
    },
    "deep": {
        "max_subquestions": 8,
        "queries_per_subquestion": 3,
        "max_sources_per_query": 6,
        "max_evidence_per_subquestion": 12,
        "critic_passes": 3,
        "allow_followup_search": True,
    },
}

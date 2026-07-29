from __future__ import annotations

from functools import lru_cache

from pydantic import Field, PositiveFloat, PositiveInt, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings loaded from environment variables and CLI overrides."""

    model_config = SettingsConfigDict(env_prefix="FLUME_", env_file=".env", extra="ignore")

    host: str = "127.0.0.1"
    port: int = 8080
    database_url: str = "sqlite:///./flume.db"
    vllm_workers: list[str] = Field(default_factory=lambda: ["http://localhost:8000"])
    model_id: str = "local-model"
    tokenizer_id: str = "local-tokenizer"
    tokenizer_revision: str = "main"
    allow_remote_tokenizer: bool = False
    request_timeout_seconds: PositiveFloat = 120.0
    connect_timeout_seconds: PositiveFloat = 5.0
    health_timeout_seconds: PositiveFloat = 2.0
    health_refresh_seconds: PositiveFloat = 5.0
    max_pack_tokens: int | None = None
    max_request_body_bytes: PositiveInt = 1_048_576
    max_output_tokens: PositiveInt = 4_096
    max_in_flight: PositiveInt = 256
    pack_cache_bytes: PositiveInt = 256 * 1024 * 1024
    sqlite_busy_timeout_ms: PositiveInt = 5_000
    cache_salt_secret: str = "development-only-change-before-production"
    metrics_enabled: bool = True

    @field_validator("vllm_workers", mode="before")
    @classmethod
    def parse_workers(cls, value: object) -> list[str]:
        if isinstance(value, str):
            return [item.strip().rstrip("/") for item in value.split(",") if item.strip()]
        if isinstance(value, list):
            return [str(item).rstrip("/") for item in value]
        raise TypeError("vllm_workers must be a comma-separated string or list")


@lru_cache
def get_settings() -> Settings:
    return Settings()

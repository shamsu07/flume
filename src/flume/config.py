from __future__ import annotations

from functools import lru_cache

from pydantic import Field, field_validator
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
    allow_remote_tokenizer: bool = False
    request_timeout_seconds: float = 120.0
    max_pack_tokens: int | None = None
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

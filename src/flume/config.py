from __future__ import annotations

import json
import math
from functools import lru_cache
from typing import Annotated, Literal

from pydantic import (
    Field,
    NonNegativeInt,
    PositiveFloat,
    PositiveInt,
    field_validator,
    model_validator,
)
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

INSECURE_CACHE_SALT_SECRET = "development-only-change-before-production"


class Settings(BaseSettings):
    """Runtime settings loaded from environment variables and CLI overrides."""

    model_config = SettingsConfigDict(env_prefix="FLUME_", env_file=".env", extra="ignore")

    host: str = "127.0.0.1"
    port: int = 8080
    database_url: str = "sqlite:///./flume.db"
    vllm_workers: Annotated[list[str], NoDecode] = Field(
        default_factory=lambda: ["http://localhost:8000"]
    )
    model_id: str = "local-model"
    tokenizer_id: str = "local-tokenizer"
    tokenizer_revision: str = "0000000000000000000000000000000000000000"
    allow_remote_tokenizer: bool = False
    request_timeout_seconds: PositiveFloat = 120.0
    connect_timeout_seconds: PositiveFloat = 5.0
    health_timeout_seconds: PositiveFloat = 2.0
    health_refresh_seconds: PositiveFloat = 5.0
    routing_policy: Literal["hrw", "bounded_hrw"] = "hrw"
    routing_load_slack: NonNegativeInt = 2
    routing_spill_hold_ms: NonNegativeInt = 2_000
    worker_load_refresh_ms: PositiveInt = 500
    worker_load_stale_ms: PositiveInt = 2_000
    worker_capacity_weights: dict[str, float] = Field(default_factory=dict)
    routing_state_max_entries: PositiveInt = 10_000
    routing_state_ttl_seconds: PositiveFloat = 600.0
    max_pack_tokens: int | None = None
    max_request_body_bytes: PositiveInt = 1_048_576
    max_output_tokens: PositiveInt = 4_096
    max_in_flight: PositiveInt = 256
    pack_cache_bytes: PositiveInt = 256 * 1024 * 1024
    sqlite_busy_timeout_ms: PositiveInt = 5_000
    cache_salt_secret: str = Field(
        default=INSECURE_CACHE_SALT_SECRET,
        min_length=32,
        repr=False,
    )
    metrics_enabled: bool = True

    @field_validator("vllm_workers", mode="before")
    @classmethod
    def parse_workers(cls, value: object) -> list[str]:
        if isinstance(value, str):
            stripped = value.strip()
            if stripped.startswith("["):
                try:
                    value = json.loads(stripped)
                except json.JSONDecodeError as exc:
                    raise ValueError("vllm_workers contains invalid JSON") from exc
            else:
                value = [item for item in stripped.split(",") if item.strip()]
        if isinstance(value, list):
            normalized = [str(item).strip().rstrip("/") for item in value]
            if not normalized or any(not item for item in normalized):
                raise ValueError("vllm_workers must contain at least one non-empty URL")
            return normalized
        raise TypeError("vllm_workers must be a comma-separated string or JSON array")

    @field_validator("cache_salt_secret")
    @classmethod
    def validate_cache_salt_secret(cls, value: str) -> str:
        if value != value.strip():
            raise ValueError("cache_salt_secret cannot have leading or trailing whitespace")
        return value

    @field_validator("worker_capacity_weights")
    @classmethod
    def validate_worker_capacity_weights(cls, value: dict[str, float]) -> dict[str, float]:
        normalized: dict[str, float] = {}
        for worker_url, weight in value.items():
            worker = worker_url.strip().rstrip("/")
            if not worker:
                raise ValueError("worker capacity weight URL cannot be empty")
            if not math.isfinite(weight) or weight <= 0:
                raise ValueError("worker capacity weights must be finite and greater than zero")
            if worker in normalized:
                raise ValueError("worker capacity weight URLs must be unique")
            normalized[worker] = weight
        return normalized

    @model_validator(mode="after")
    def validate_routing_settings(self) -> Settings:
        workers = set(self.vllm_workers)
        unknown_workers = set(self.worker_capacity_weights).difference(workers)
        if unknown_workers:
            raise ValueError("worker capacity weights contain an unknown vLLM worker")
        if self.worker_load_stale_ms < self.worker_load_refresh_ms:
            raise ValueError("worker_load_stale_ms must be at least worker_load_refresh_ms")
        if self.routing_spill_hold_ms > self.routing_state_ttl_seconds * 1_000:
            raise ValueError("routing_spill_hold_ms cannot exceed routing_state_ttl_seconds")
        self.worker_capacity_weights = {
            worker: self.worker_capacity_weights.get(worker, 1.0)
            for worker in self.vllm_workers
        }
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()

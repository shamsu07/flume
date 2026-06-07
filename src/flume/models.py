from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


def utc_now() -> datetime:
    return datetime.now(UTC)


class OrderPolicy(StrEnum):
    stable = "stable"
    input = "input"


class DocumentChunk(BaseModel):
    doc_id: str
    chunk_id: str = "0"
    version: str = "1"
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("doc_id", "chunk_id", "version")
    @classmethod
    def non_empty(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("value cannot be empty")
        return value


class PackCreateRequest(BaseModel):
    tenant_id: str = "default"
    model_id: str
    tokenizer_id: str
    template_id: str = "default-rag-v1"
    chunks: list[DocumentChunk]
    template: str | None = None
    order_policy: OrderPolicy = OrderPolicy.stable
    ttl_seconds: int | None = None
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("chunks")
    @classmethod
    def require_chunks(cls, value: list[DocumentChunk]) -> list[DocumentChunk]:
        if not value:
            raise ValueError("at least one chunk is required")
        return value


class ContextPack(BaseModel):
    pack_id: str
    tenant_id: str
    model_id: str
    tokenizer_id: str
    template_id: str
    document_hash: str
    token_hash: str
    token_count: int
    compiled_prefix: str
    created_at: datetime = Field(default_factory=utc_now)
    ttl_seconds: int | None = None
    order_policy: OrderPolicy = OrderPolicy.stable
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(use_enum_values=True)

    @property
    def expired(self) -> bool:
        if self.ttl_seconds is None:
            return False
        age = (utc_now() - self.created_at).total_seconds()
        return age > self.ttl_seconds


class AskRequest(BaseModel):
    pack_id: str
    question: str
    stream: bool = False
    max_tokens: int = 256
    temperature: float = 0.0
    top_p: float = 1.0
    stop: list[str] | None = None
    extra_body: dict[str, Any] = Field(default_factory=dict)


class AskResponse(BaseModel):
    pack_id: str
    worker_url: str
    text: str
    ttft_ms: float | None
    latency_ms: float
    prompt_tokens: int
    output_tokens: int | None = None
    finish_reason: str | None = None


class WarmRequest(BaseModel):
    pack_id: str
    strategy: Literal["one_token", "echo_question"] = "one_token"
    question: str = "warm"


class WarmResponse(BaseModel):
    pack_id: str
    worker_url: str
    warmed: bool
    latency_ms: float


class WorkerStats(BaseModel):
    worker_url: str
    healthy: bool | None = None
    assigned_packs: int = 0
    affinity_hits: int = 0
    affinity_misses: int = 0


class StatsResponse(BaseModel):
    packs: int
    routes: int
    workers: list[WorkerStats]


class BenchmarkRunRequest(BaseModel):
    baseline: Literal[
        "plain_vllm",
        "vllm_apc",
        "flume_stable",
        "flume_warm_affinity",
    ] = "flume_warm_affinity"
    context_lengths: list[int] = Field(default_factory=lambda: [4096, 16384])
    concurrency: int = 1
    iterations: int = 3
    questions: list[str] = Field(
        default_factory=lambda: [
            "Summarize the key policy.",
            "What should support verify?",
            "When is the refund window?",
        ]
    )


class BenchmarkRun(BaseModel):
    run_id: str
    request: BenchmarkRunRequest
    created_at: datetime = Field(default_factory=utc_now)
    results: dict[str, Any] = Field(default_factory=dict)
    status: Literal["created", "running", "completed", "failed"] = "created"
    error: str | None = None

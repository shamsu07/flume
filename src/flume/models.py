from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from string import Formatter
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from flume.hashing import canonical_identifier, canonical_json_value, canonical_text


def utc_now() -> datetime:
    return datetime.now(UTC)


class OrderPolicy(StrEnum):
    stable = "stable"
    input = "input"


class DocumentChunk(BaseModel):
    doc_id: str = Field(min_length=1, max_length=512)
    chunk_id: str = Field(default="0", min_length=1, max_length=512)
    version: str = Field(default="1", min_length=1, max_length=512)
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(extra="forbid")

    @field_validator("doc_id", "chunk_id", "version")
    @classmethod
    def non_empty(cls, value: str) -> str:
        return canonical_identifier(value)

    @field_validator("text")
    @classmethod
    def normalize_text(cls, value: str) -> str:
        return canonical_text(value)

    @field_validator("metadata")
    @classmethod
    def normalize_metadata(cls, value: dict[str, Any]) -> dict[str, Any]:
        normalized = canonical_json_value(value)
        if not isinstance(normalized, dict):
            raise TypeError("metadata must be an object")
        return normalized


class PackCreateRequest(BaseModel):
    tenant_id: str = Field(default="default", min_length=1, max_length=128)
    model_id: str = Field(min_length=1, max_length=512)
    tokenizer_id: str = Field(min_length=1, max_length=512)
    template_id: str = Field(default="default-rag-v1", min_length=1, max_length=128)
    chunks: list[DocumentChunk] = Field(min_length=1)
    template: str | None = None
    order_policy: OrderPolicy = OrderPolicy.stable
    ttl_seconds: int | None = Field(default=None, gt=0)
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(extra="forbid")

    @field_validator("tenant_id", "model_id", "tokenizer_id", "template_id")
    @classmethod
    def normalize_identifiers(cls, value: str) -> str:
        return canonical_identifier(value)

    @field_validator("template")
    @classmethod
    def validate_template(cls, value: str | None) -> str | None:
        if value is None:
            return value
        normalized = canonical_text(value)
        fields: list[tuple[str, str, str | None]] = []
        try:
            for _, field_name, format_spec, conversion in Formatter().parse(normalized):
                if field_name is not None:
                    fields.append((field_name, format_spec, conversion))
        except ValueError as exc:
            raise ValueError("template contains invalid formatting syntax") from exc
        if fields != [("context", "", None)]:
            raise ValueError("template must contain exactly one unmodified {context} field")
        return normalized

    @field_validator("tags")
    @classmethod
    def normalize_tags(cls, value: list[str]) -> list[str]:
        normalized = [canonical_identifier(tag) for tag in value]
        if len(normalized) != len(set(normalized)):
            raise ValueError("tags must be unique")
        return sorted(normalized)

    @field_validator("metadata")
    @classmethod
    def normalize_metadata(cls, value: dict[str, Any]) -> dict[str, Any]:
        normalized = canonical_json_value(value)
        if not isinstance(normalized, dict):
            raise TypeError("metadata must be an object")
        return normalized

    @model_validator(mode="after")
    def reject_duplicate_chunks(self) -> PackCreateRequest:
        seen: set[tuple[str, str]] = set()
        for chunk in self.chunks:
            key = (chunk.doc_id, chunk.chunk_id)
            if key in seen:
                raise ValueError(f"duplicate chunk key: {chunk.doc_id}/{chunk.chunk_id}")
            seen.add(key)
        return self


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

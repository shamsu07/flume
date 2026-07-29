from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from string import Formatter
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from flume.hashing import (
    canonical_identifier,
    canonical_json_value,
    canonical_text,
    sha256_text,
    sha256_token_ids,
    stable_pack_id,
)


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


class PackRegistrationRequest(BaseModel):
    """Public pack input. Runtime identity is supplied by the server and tenant header."""

    template_id: str = Field(default="default-rag-v1", min_length=1, max_length=128)
    chunks: list[DocumentChunk] = Field(min_length=1)
    template: str | None = None
    order_policy: OrderPolicy = OrderPolicy.stable
    ttl_seconds: int | None = Field(default=None, gt=0)
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(extra="forbid")

    @field_validator("template_id")
    @classmethod
    def normalize_identifiers(cls, value: str) -> str:
        return canonical_identifier(value)

    @field_validator("template")
    @classmethod
    def validate_template(cls, value: str | None) -> str | None:
        if value is None:
            return value
        normalized = canonical_text(value)
        fields: list[tuple[str, str | None, str | None]] = []
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
    def reject_duplicate_chunks(self) -> PackRegistrationRequest:
        seen: set[tuple[str, str]] = set()
        for chunk in self.chunks:
            key = (chunk.doc_id, chunk.chunk_id)
            if key in seen:
                raise ValueError(f"duplicate chunk key: {chunk.doc_id}/{chunk.chunk_id}")
            seen.add(key)
        return self


class PackCreateRequest(PackRegistrationRequest):
    """Internal compiler input with server-authoritative runtime identity."""

    tenant_id: str = Field(default="default", min_length=1, max_length=128)
    model_id: str = Field(min_length=1, max_length=512)
    tokenizer_id: str = Field(min_length=1, max_length=512)

    @field_validator("tenant_id", "model_id", "tokenizer_id")
    @classmethod
    def normalize_authority_identifiers(cls, value: str) -> str:
        return canonical_identifier(value)


class ContextPack(BaseModel):
    pack_id: str
    compiler_format_version: str
    tenant_id: str
    model_id: str
    tokenizer_id: str
    tokenizer_revision: str
    tokenizer_fingerprint: str
    template_id: str
    template_digest: str
    document_hash: str
    canonical_prefix_hash: str
    token_hash: str
    token_count: int
    compiled_prefix: str
    prefix_token_ids: tuple[int, ...]
    created_at: datetime = Field(default_factory=utc_now)
    ttl_seconds: int | None = None
    order_policy: OrderPolicy = OrderPolicy.stable
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(extra="forbid", frozen=True)

    @field_validator(
        "template_digest",
        "document_hash",
        "canonical_prefix_hash",
        "token_hash",
        "tokenizer_fingerprint",
    )
    @classmethod
    def require_sha256(cls, value: str) -> str:
        if len(value) != 64:
            raise ValueError("digest must contain 64 hexadecimal characters")
        try:
            int(value, 16)
        except ValueError as exc:
            raise ValueError("digest must contain 64 hexadecimal characters") from exc
        return value.lower()

    @field_validator("compiled_prefix")
    @classmethod
    def require_canonical_prefix(cls, value: str) -> str:
        if canonical_text(value) != value:
            raise ValueError("compiled prefix is not canonical text")
        return value

    @model_validator(mode="after")
    def enforce_manifest_invariants(self) -> ContextPack:
        if not self.prefix_token_ids:
            raise ValueError("compiled prefix token ids cannot be empty")
        if self.token_count != len(self.prefix_token_ids):
            raise ValueError("token_count does not match compiled prefix token ids")
        if self.token_hash != sha256_token_ids(list(self.prefix_token_ids)):
            raise ValueError("token_hash does not match compiled prefix token ids")
        if self.canonical_prefix_hash != sha256_text(self.compiled_prefix):
            raise ValueError("canonical_prefix_hash does not match compiled prefix")
        if self.pack_id != stable_pack_id(self.immutable_manifest()):
            raise ValueError("pack_id does not match immutable manifest")
        return self

    def immutable_manifest(self) -> dict[str, str | int]:
        return {
            "compiler_format_version": self.compiler_format_version,
            "tenant_id": self.tenant_id,
            "model_id": self.model_id,
            "tokenizer_id": self.tokenizer_id,
            "tokenizer_revision": self.tokenizer_revision,
            "tokenizer_fingerprint": self.tokenizer_fingerprint,
            "template_id": self.template_id,
            "template_digest": self.template_digest,
            "order_policy": self.order_policy.value,
            "document_hash": self.document_hash,
            "canonical_prefix_hash": self.canonical_prefix_hash,
            "token_hash": self.token_hash,
            "token_count": self.token_count,
        }

    def operational_annotations(self) -> dict[str, Any]:
        return {
            "ttl_seconds": self.ttl_seconds,
            "tags": list(self.tags),
            "metadata": self.metadata,
        }

    @property
    def expired(self) -> bool:
        if self.ttl_seconds is None:
            return False
        age = (utc_now() - self.created_at).total_seconds()
        return age > self.ttl_seconds

    def to_summary(self) -> PackSummary:
        return PackSummary(
            pack_id=self.pack_id,
            compiler_format_version=self.compiler_format_version,
            model_id=self.model_id,
            tokenizer_id=self.tokenizer_id,
            tokenizer_revision=self.tokenizer_revision,
            tokenizer_fingerprint=self.tokenizer_fingerprint,
            template_id=self.template_id,
            document_hash=self.document_hash,
            canonical_prefix_hash=self.canonical_prefix_hash,
            token_count=self.token_count,
            created_at=self.created_at,
            ttl_seconds=self.ttl_seconds,
            tags=self.tags,
        )


class PackSummary(BaseModel):
    """Safe public representation which never contains prefix text or token ids."""

    pack_id: str
    compiler_format_version: str
    model_id: str
    tokenizer_id: str
    tokenizer_revision: str
    tokenizer_fingerprint: str
    template_id: str
    document_hash: str
    canonical_prefix_hash: str
    token_count: int
    created_at: datetime
    ttl_seconds: int | None = None
    tags: list[str] = Field(default_factory=list)

    model_config = ConfigDict(frozen=True)


class PackPage(BaseModel):
    items: list[PackSummary]
    next_cursor: str | None = None

    model_config = ConfigDict(frozen=True)


class CompletionRequest(BaseModel):
    """OpenAI completions request with a required Flume context-pack extension."""

    pack_id: str = Field(min_length=1, max_length=128)
    prompt: str = Field(min_length=1, max_length=1_000_000)
    stream: bool = False
    max_tokens: int = Field(default=256, ge=1, le=1_048_576)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    top_p: float = Field(default=1.0, gt=0.0, le=1.0)
    stop: str | list[str] | None = None
    extra_body: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(extra="forbid", strict=True)

    @field_validator("pack_id")
    @classmethod
    def normalize_pack_id(cls, value: str) -> str:
        return canonical_identifier(value)

    @field_validator("prompt")
    @classmethod
    def normalize_prompt(cls, value: str) -> str:
        normalized = canonical_text(value)
        if not normalized:
            raise ValueError("prompt cannot be empty")
        return normalized

    @field_validator("stop")
    @classmethod
    def validate_stop(cls, value: str | list[str] | None) -> str | list[str] | None:
        if value is None:
            return None
        values = [value] if isinstance(value, str) else value
        if not values or len(values) > 4:
            raise ValueError("stop must contain between one and four strings")
        if any(not item or len(item) > 512 for item in values):
            raise ValueError("stop strings must contain between 1 and 512 characters")
        return value

    @field_validator("extra_body")
    @classmethod
    def bound_extra_body(cls, value: dict[str, Any]) -> dict[str, Any]:
        if len(value) > 32:
            raise ValueError("extra_body cannot contain more than 32 fields")
        return value


class CompletionChoice(BaseModel):
    text: str
    index: int
    logprobs: Any | None = None
    finish_reason: str | None = None

    model_config = ConfigDict(extra="allow", frozen=True)


class CompletionUsage(BaseModel):
    prompt_tokens: int = Field(ge=0)
    completion_tokens: int = Field(ge=0)
    total_tokens: int = Field(ge=0)

    model_config = ConfigDict(extra="allow", frozen=True)


class CompletionResponse(BaseModel):
    id: str
    object: Literal["text_completion"]
    created: int
    model: str
    choices: list[CompletionChoice]
    usage: CompletionUsage

    model_config = ConfigDict(extra="allow", frozen=True)


class WarmRequest(BaseModel):
    strategy: Literal["one_token", "echo_question"] = "one_token"
    question: str = Field(default="warm", min_length=1, max_length=16_384)

    model_config = ConfigDict(extra="forbid", strict=True)


class WarmResponse(BaseModel):
    pack_id: str
    worker_id: str
    warmed: bool
    latency_ms: float

    model_config = ConfigDict(extra="forbid", frozen=True)


class WorkerStats(BaseModel):
    worker_id: str
    healthy: bool | None = None
    assigned_packs: int = 0
    affinity_hits: int = 0
    affinity_misses: int = 0

    model_config = ConfigDict(extra="forbid", frozen=True)


class StatsResponse(BaseModel):
    packs: int
    routes: int
    workers: list[WorkerStats]

    model_config = ConfigDict(extra="forbid", frozen=True)


# Kept for one compatibility commit while the SDK migrates in the following commit.
class AskRequest(BaseModel):
    pack_id: str
    question: str
    stream: bool = False
    max_tokens: int = 256
    temperature: float = 0.0
    top_p: float = 1.0
    stop: list[str] | None = None
    extra_body: dict[str, Any] = Field(default_factory=dict)


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
    questions: list[str] = Field(default_factory=lambda: ["Summarize the key policy."])


class BenchmarkRun(BaseModel):
    run_id: str
    request: BenchmarkRunRequest
    created_at: datetime = Field(default_factory=utc_now)
    results: dict[str, Any] = Field(default_factory=dict)
    status: Literal["created", "running", "completed", "failed"] = "created"
    error: str | None = None

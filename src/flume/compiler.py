from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from flume.hashing import (
    canonical_json,
    canonical_text,
    sha256_json,
    sha256_text,
    sha256_token_ids,
    stable_pack_id,
)
from flume.models import ContextPack, DocumentChunk, OrderPolicy, PackCreateRequest

COMPILER_FORMAT_VERSION = "1"

DEFAULT_TEMPLATE = """You are answering using a cache-stable RAG context pack.

{context}

Answer the user's question using only the context when possible.

Question:
"""


class Tokenizer(Protocol):
    tokenizer_id: str
    revision: str
    fingerprint: str

    def encode(self, text: str) -> list[int]:
        ...


@dataclass(frozen=True, slots=True)
class DeterministicByteTokenizer:
    """Explicit test tokenizer. It is never selected as a production fallback."""

    tokenizer_id: str = "deterministic-byte-tokenizer"
    revision: str = "byte-v1"
    fingerprint: str = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "fingerprint",
            sha256_json(
                {
                    "implementation": "flume.deterministic-byte-tokenizer",
                    "revision": self.revision,
                }
            ),
        )

    def encode(self, text: str) -> list[int]:
        return list(text.encode("utf-8"))


class HuggingFaceTokenizer:
    def __init__(
        self,
        tokenizer_id: str,
        revision: str,
        *,
        allow_remote: bool = False,
    ):
        from transformers import AutoTokenizer

        if revision.casefold() in {"main", "master", "latest"}:
            raise ValueError("tokenizer revision must be immutable, not a floating alias")
        self.tokenizer_id = tokenizer_id
        self._tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_id,
            revision=revision,
            local_files_only=not allow_remote,
            trust_remote_code=False,
        )
        resolved_revision = getattr(self._tokenizer, "_commit_hash", None)
        if resolved_revision is None:
            resolved_revision = self._tokenizer.init_kwargs.get("_commit_hash")
        self.revision = resolved_revision or revision
        serialized_backend = self._tokenizer.backend_tokenizer.to_str()
        self.fingerprint = sha256_json(
            {
                "tokenizer_id": tokenizer_id,
                "revision": self.revision,
                "backend_sha256": sha256_text(serialized_backend),
            }
        )

    def encode(self, text: str) -> list[int]:
        return list(self._tokenizer.encode(text, add_special_tokens=False))


def load_tokenizer(
    tokenizer_id: str,
    revision: str,
    *,
    allow_remote: bool = False,
) -> Tokenizer:
    """Load exactly the configured tokenizer or propagate the loading failure."""
    return HuggingFaceTokenizer(
        tokenizer_id,
        revision,
        allow_remote=allow_remote,
    )


class ContextPackCompiler:
    def __init__(
        self,
        tokenizer: Tokenizer,
        *,
        model_id: str | None = None,
    ):
        self.tokenizer = tokenizer
        self.model_id = model_id
        self._validate_tokenizer_identity()

    @classmethod
    def from_pretrained(
        cls,
        *,
        tokenizer_id: str,
        tokenizer_revision: str,
        model_id: str | None = None,
        allow_remote_tokenizer: bool = False,
    ) -> ContextPackCompiler:
        return cls(
            load_tokenizer(
                tokenizer_id,
                tokenizer_revision,
                allow_remote=allow_remote_tokenizer,
            ),
            model_id=model_id,
        )

    def compile(self, request: PackCreateRequest) -> ContextPack:
        self._validate_request_authority(request)
        ordered_chunks = self._order_chunks(request.chunks, request.order_policy)
        context = self._render_context(ordered_chunks)
        template = canonical_text(request.template or DEFAULT_TEMPLATE)
        compiled_prefix = template.format(context=context)

        token_ids = self.tokenizer.encode(compiled_prefix)

        document_hash = sha256_json(
            {
                "compiler_format_version": COMPILER_FORMAT_VERSION,
                "chunks": [chunk.model_dump(mode="json") for chunk in ordered_chunks],
            }
        )
        template_digest = sha256_text(template)
        canonical_prefix_hash = sha256_text(compiled_prefix)
        token_hash = sha256_token_ids(token_ids)
        manifest = {
            "compiler_format_version": COMPILER_FORMAT_VERSION,
            "tenant_id": request.tenant_id,
            "model_id": request.model_id,
            "tokenizer_id": self.tokenizer.tokenizer_id,
            "tokenizer_revision": self.tokenizer.revision,
            "tokenizer_fingerprint": self.tokenizer.fingerprint,
            "template_id": request.template_id,
            "template_digest": template_digest,
            "order_policy": request.order_policy.value,
            "document_hash": document_hash,
            "canonical_prefix_hash": canonical_prefix_hash,
            "token_hash": token_hash,
            "token_count": len(token_ids),
        }

        return ContextPack(
            pack_id=stable_pack_id(manifest),
            compiler_format_version=COMPILER_FORMAT_VERSION,
            tenant_id=request.tenant_id,
            model_id=request.model_id,
            tokenizer_id=self.tokenizer.tokenizer_id,
            tokenizer_revision=self.tokenizer.revision,
            tokenizer_fingerprint=self.tokenizer.fingerprint,
            template_id=request.template_id,
            template_digest=template_digest,
            document_hash=document_hash,
            canonical_prefix_hash=canonical_prefix_hash,
            token_hash=token_hash,
            token_count=len(token_ids),
            compiled_prefix=compiled_prefix,
            prefix_token_ids=tuple(token_ids),
            ttl_seconds=request.ttl_seconds,
            order_policy=request.order_policy,
            tags=request.tags,
            metadata=request.metadata,
        )

    def completion_token_ids(self, pack: ContextPack, question: str) -> list[int]:
        """Append separately encoded question/answer tokens to the immutable prefix."""
        if (
            pack.tokenizer_id,
            pack.tokenizer_revision,
            pack.tokenizer_fingerprint,
        ) != (
            self.tokenizer.tokenizer_id,
            self.tokenizer.revision,
            self.tokenizer.fingerprint,
        ):
            raise ValueError("pack tokenizer identity does not match compiler runtime")
        suffix = f"{canonical_text(question)}\n\nAnswer:\n"
        return [*pack.prefix_token_ids, *self.tokenizer.encode(suffix)]

    def _validate_tokenizer_identity(self) -> None:
        if not self.tokenizer.tokenizer_id.strip():
            raise ValueError("tokenizer_id cannot be empty")
        if not self.tokenizer.revision.strip():
            raise ValueError("tokenizer revision cannot be empty")
        if self.tokenizer.revision.casefold() in {"main", "master", "latest"}:
            raise ValueError("tokenizer revision must be immutable, not a floating alias")
        if len(self.tokenizer.fingerprint) != 64:
            raise ValueError("tokenizer fingerprint must be a SHA-256 hex digest")
        try:
            int(self.tokenizer.fingerprint, 16)
        except ValueError as exc:
            raise ValueError("tokenizer fingerprint must be a SHA-256 hex digest") from exc

    def _validate_request_authority(self, request: PackCreateRequest) -> None:
        if request.tokenizer_id != self.tokenizer.tokenizer_id:
            raise ValueError(
                "request tokenizer_id does not match the server-authoritative tokenizer"
            )
        if self.model_id is not None and request.model_id != self.model_id:
            raise ValueError("request model_id does not match the server-authoritative model")

    def _order_chunks(
        self,
        chunks: list[DocumentChunk],
        order_policy: OrderPolicy,
    ) -> list[DocumentChunk]:
        if order_policy == OrderPolicy.input:
            return list(chunks)
        return sorted(chunks, key=lambda c: (c.doc_id, c.chunk_id, c.version))

    def _render_context(self, chunks: list[DocumentChunk]) -> str:
        rendered = []
        for chunk in chunks:
            metadata = ""
            if chunk.metadata:
                metadata = f"\nmetadata: {canonical_json(chunk.metadata)}"
            doc_id = canonical_json(chunk.doc_id)
            chunk_id = canonical_json(chunk.chunk_id)
            version = canonical_json(chunk.version)
            rendered.append(
                "\n".join(
                    [
                        (
                            f"<chunk doc_id={doc_id} "
                            f"chunk_id={chunk_id} "
                            f"version={version}>"
                        ),
                        canonical_text(chunk.text),
                        f"</chunk>{metadata}",
                    ]
                )
            )
        return "\n\n---\n\n".join(rendered)

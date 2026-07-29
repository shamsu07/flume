from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from flume.hashing import (
    canonical_json,
    canonical_text,
    sha256_json,
    sha256_text,
    sha256_token_ids,
)
from flume.models import ContextPack, DocumentChunk, OrderPolicy, PackCreateRequest

COMPILER_FORMAT_VERSION = "1"

DEFAULT_TEMPLATE = """You are answering using a cache-stable RAG context pack.

{context}

Answer the user's question using only the context when possible.

Question:
"""


class Tokenizer(Protocol):
    name: str

    def encode(self, text: str) -> list[int]:
        ...


@dataclass(slots=True)
class DeterministicByteTokenizer:
    """Offline fallback tokenizer used when a Hugging Face tokenizer is unavailable."""

    name: str = "deterministic-byte-tokenizer"

    def encode(self, text: str) -> list[int]:
        return list(text.encode("utf-8"))


class HuggingFaceTokenizer:
    def __init__(self, tokenizer_id: str, allow_remote: bool = False):
        from transformers import AutoTokenizer

        self.name = tokenizer_id
        self._tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_id,
            local_files_only=not allow_remote,
            trust_remote_code=False,
        )

    def encode(self, text: str) -> list[int]:
        return list(self._tokenizer.encode(text, add_special_tokens=False))


def load_tokenizer(tokenizer_id: str, allow_remote: bool = False) -> Tokenizer:
    try:
        return HuggingFaceTokenizer(tokenizer_id, allow_remote=allow_remote)
    except Exception:
        return DeterministicByteTokenizer(name=f"fallback:{tokenizer_id}")


class ContextPackCompiler:
    def __init__(self, allow_remote_tokenizer: bool = False):
        self.allow_remote_tokenizer = allow_remote_tokenizer

    def compile(self, request: PackCreateRequest) -> ContextPack:
        ordered_chunks = self._order_chunks(request.chunks, request.order_policy)
        context = self._render_context(ordered_chunks)
        template = canonical_text(request.template or DEFAULT_TEMPLATE)
        compiled_prefix = template.format(context=context)

        tokenizer = load_tokenizer(request.tokenizer_id, self.allow_remote_tokenizer)
        token_ids = tokenizer.encode(compiled_prefix)

        document_hash = sha256_json(
            {
                "compiler_format_version": COMPILER_FORMAT_VERSION,
                "chunks": [chunk.model_dump(mode="json") for chunk in ordered_chunks],
            }
        )
        template_digest = sha256_text(template)
        canonical_prefix_hash = sha256_text(compiled_prefix)
        token_hash = sha256_token_ids(token_ids)
        pack_id = self._pack_id(
            tenant_id=request.tenant_id,
            model_id=request.model_id,
            tokenizer_id=request.tokenizer_id,
            template_id=request.template_id,
            template_digest=template_digest,
            order_policy=request.order_policy,
            document_hash=document_hash,
            canonical_prefix_hash=canonical_prefix_hash,
            token_hash=token_hash,
        )

        return ContextPack(
            pack_id=pack_id,
            compiler_format_version=COMPILER_FORMAT_VERSION,
            tenant_id=request.tenant_id,
            model_id=request.model_id,
            tokenizer_id=request.tokenizer_id,
            template_id=request.template_id,
            template_digest=template_digest,
            document_hash=document_hash,
            canonical_prefix_hash=canonical_prefix_hash,
            token_hash=token_hash,
            token_count=len(token_ids),
            compiled_prefix=compiled_prefix,
            ttl_seconds=request.ttl_seconds,
            order_policy=request.order_policy,
            tags=request.tags,
            metadata={
                **request.metadata,
                "tokenizer_runtime": tokenizer.name,
                "chunk_count": len(ordered_chunks),
            },
        )

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

    def _pack_id(
        self,
        *,
        tenant_id: str,
        model_id: str,
        tokenizer_id: str,
        template_id: str,
        template_digest: str,
        order_policy: OrderPolicy,
        document_hash: str,
        canonical_prefix_hash: str,
        token_hash: str,
    ) -> str:
        digest = sha256_text(
            canonical_json(
                {
                    "compiler_format_version": COMPILER_FORMAT_VERSION,
                    "tenant_id": tenant_id,
                    "model_id": model_id,
                    "tokenizer_id": tokenizer_id,
                    "template_id": template_id,
                    "template_digest": template_digest,
                    "order_policy": order_policy.value,
                    "document_hash": document_hash,
                    "canonical_prefix_hash": canonical_prefix_hash,
                    "token_hash": token_hash,
                }
            )
        )
        return f"pack_{digest[:24]}"

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import httpx

from flume.compiler import ContextPackCompiler
from flume.hashing import canonical_identifier, canonical_text, sha256_text
from flume.models import (
    AskRequest,
    BenchmarkRun,
    BenchmarkRunRequest,
    ContextPack,
    DocumentChunk,
    PackCreateRequest,
    StatsResponse,
    WarmResponse,
)


def compile_pack(
    *,
    chunks: list[DocumentChunk],
    tenant_id: str,
    model_id: str,
    tokenizer_id: str,
    tokenizer_revision: str,
    template_id: str = "default-rag-v1",
    allow_remote_tokenizer: bool = False,
) -> ContextPack:
    compiler = ContextPackCompiler.from_pretrained(
        tokenizer_id=tokenizer_id,
        tokenizer_revision=tokenizer_revision,
        model_id=model_id,
        allow_remote_tokenizer=allow_remote_tokenizer,
    )
    return compiler.compile(
        PackCreateRequest(
            tenant_id=tenant_id,
            model_id=model_id,
            tokenizer_id=tokenizer_id,
            template_id=template_id,
            chunks=chunks,
        )
    )


def chunks_from_files(
    paths: list[Path],
    *,
    logical_names: Mapping[Path, str] | None = None,
) -> list[DocumentChunk]:
    """Create path-independent chunks using stable logical names and content versions."""
    chunks: list[DocumentChunk] = []
    seen_names: set[str] = set()
    logical_names = logical_names or {}
    for path in paths:
        logical_name = canonical_identifier(logical_names.get(path, path.name))
        if logical_name in seen_names:
            raise ValueError(f"duplicate logical file name: {logical_name}")
        seen_names.add(logical_name)
        text = canonical_text(path.read_text(encoding="utf-8"))
        chunks.append(
            DocumentChunk(
                doc_id=logical_name,
                chunk_id="0",
                version=sha256_text(text),
                text=text,
                metadata={"source_name": logical_name},
            )
        )
    return sorted(chunks, key=lambda chunk: (chunk.doc_id, chunk.chunk_id, chunk.version))


class FlumeClient:
    def __init__(self, base_url: str = "http://localhost:8080", timeout_seconds: float = 120.0):
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds

    def register_pack(self, request: PackCreateRequest) -> ContextPack:
        with httpx.Client(timeout=self.timeout_seconds) as client:
            response = client.post(f"{self.base_url}/packs", json=request.model_dump(mode="json"))
            response.raise_for_status()
            return ContextPack.model_validate(response.json())

    def list_packs(self, tenant_id: str | None = None) -> list[ContextPack]:
        params = {"tenant_id": tenant_id} if tenant_id else None
        with httpx.Client(timeout=self.timeout_seconds) as client:
            response = client.get(f"{self.base_url}/packs", params=params)
            response.raise_for_status()
            return [ContextPack.model_validate(item) for item in response.json()]

    def get_pack(self, pack_id: str) -> ContextPack:
        with httpx.Client(timeout=self.timeout_seconds) as client:
            response = client.get(f"{self.base_url}/packs/{pack_id}")
            response.raise_for_status()
            return ContextPack.model_validate(response.json())

    def warm_pack(self, pack_id: str) -> WarmResponse:
        with httpx.Client(timeout=self.timeout_seconds) as client:
            response = client.post(
                f"{self.base_url}/packs/{pack_id}/warm",
                json={"pack_id": pack_id},
            )
            response.raise_for_status()
            return WarmResponse.model_validate(response.json())

    def ask(self, pack_id: str, question: str, **kwargs: Any) -> dict[str, Any]:
        request = AskRequest(pack_id=pack_id, question=question, **kwargs)
        with httpx.Client(timeout=self.timeout_seconds) as client:
            response = client.post(f"{self.base_url}/ask", json=request.model_dump(mode="json"))
            response.raise_for_status()
            return response.json()

    def cache_stats(self) -> StatsResponse:
        with httpx.Client(timeout=self.timeout_seconds) as client:
            response = client.get(f"{self.base_url}/stats")
            response.raise_for_status()
            return StatsResponse.model_validate(response.json())

    def run_benchmark(self, request: BenchmarkRunRequest) -> BenchmarkRun:
        with httpx.Client(timeout=None) as client:
            response = client.post(
                f"{self.base_url}/bench/run",
                json=request.model_dump(mode="json"),
            )
            response.raise_for_status()
            return BenchmarkRun.model_validate(response.json())

    def list_benchmarks(self) -> list[BenchmarkRun]:
        with httpx.Client(timeout=self.timeout_seconds) as client:
            response = client.get(f"{self.base_url}/bench/runs")
            response.raise_for_status()
            return [BenchmarkRun.model_validate(item) for item in response.json()]

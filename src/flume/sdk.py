from __future__ import annotations

from pathlib import Path
from typing import Any

import httpx

from flume.compiler import ContextPackCompiler
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
    template_id: str = "default-rag-v1",
) -> ContextPack:
    compiler = ContextPackCompiler()
    return compiler.compile(
        PackCreateRequest(
            tenant_id=tenant_id,
            model_id=model_id,
            tokenizer_id=tokenizer_id,
            template_id=template_id,
            chunks=chunks,
        )
    )


def chunks_from_files(paths: list[Path]) -> list[DocumentChunk]:
    chunks = []
    for index, path in enumerate(paths):
        chunks.append(
            DocumentChunk(
                doc_id=path.stem,
                chunk_id=str(index),
                version=str(int(path.stat().st_mtime)),
                text=path.read_text(encoding="utf-8"),
                metadata={"path": str(path)},
            )
        )
    return chunks


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

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator, Mapping
from pathlib import Path
from typing import Any, TypeVar

import httpx
from pydantic import BaseModel

from flume.compiler import ContextPackCompiler
from flume.hashing import canonical_identifier, canonical_text, sha256_text
from flume.models import (
    CompletionRequest,
    CompletionResponse,
    ContextPack,
    DocumentChunk,
    PackCreateRequest,
    PackPage,
    PackRegistrationRequest,
    PackSummary,
    StatsResponse,
    WarmRequest,
    WarmResponse,
)

ResponseModel = TypeVar("ResponseModel", bound=BaseModel)


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


class FlumeError(RuntimeError):
    """Base class for SDK failures."""


class FlumeAPIError(FlumeError):
    """A non-success response returned by Flume."""

    def __init__(
        self,
        status_code: int,
        detail: Any,
        *,
        request_id: str | None = None,
    ) -> None:
        self.status_code = status_code
        self.detail = detail
        self.request_id = request_id
        message = detail if isinstance(detail, str) else f"Flume API returned HTTP {status_code}"
        super().__init__(message)


class FlumeTransportError(FlumeError):
    """A network or protocol failure before a valid Flume response."""


def _raise_for_status(response: httpx.Response) -> None:
    if not response.is_error:
        return
    detail: Any = None
    try:
        body = response.json()
        detail = body.get("detail") if isinstance(body, dict) else None
    except ValueError:
        pass
    if detail is None:
        detail = f"Flume API returned HTTP {response.status_code}"
    raise FlumeAPIError(
        response.status_code,
        detail,
        request_id=response.headers.get("X-Request-Id"),
    )


def _validate_response(response: httpx.Response, model: type[ResponseModel]) -> ResponseModel:
    _raise_for_status(response)
    try:
        return model.model_validate(response.json())
    except ValueError as exc:
        raise FlumeTransportError("Flume API returned an invalid JSON response") from exc


class FlumeClient:
    """Persistent synchronous Flume v1 client."""

    def __init__(
        self,
        base_url: str = "http://localhost:8080",
        *,
        tenant_id: str = "default",
        timeout_seconds: float = 120.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.tenant_id = canonical_identifier(tenant_id)
        self._client = httpx.Client(
            timeout=timeout_seconds,
            transport=transport,
            limits=httpx.Limits(max_connections=100, max_keepalive_connections=20),
        )

    def __enter__(self) -> FlumeClient:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    @property
    def is_closed(self) -> bool:
        return self._client.is_closed

    def close(self) -> None:
        self._client.close()

    def register_pack(self, request: PackRegistrationRequest) -> PackSummary:
        response = self._request(
            "POST",
            "/v1/packs",
            json=request.model_dump(mode="json"),
        )
        return _validate_response(response, PackSummary)

    def list_packs(self, *, limit: int = 100, cursor: str | None = None) -> PackPage:
        params: dict[str, str | int] = {"limit": limit}
        if cursor is not None:
            params["cursor"] = cursor
        response = self._request("GET", "/v1/packs", params=params)
        return _validate_response(response, PackPage)

    def get_pack(self, pack_id: str) -> PackSummary:
        response = self._request("GET", f"/v1/packs/{pack_id}")
        return _validate_response(response, PackSummary)

    def warm_pack(
        self,
        pack_id: str,
        request: WarmRequest | None = None,
    ) -> WarmResponse:
        response = self._request(
            "POST",
            f"/v1/packs/{pack_id}/warm",
            json=(request or WarmRequest()).model_dump(mode="json"),
        )
        return _validate_response(response, WarmResponse)

    def complete(self, request: CompletionRequest) -> CompletionResponse:
        payload = request.model_copy(update={"stream": False})
        response = self._request(
            "POST",
            "/v1/completions",
            json=payload.model_dump(mode="json"),
        )
        return _validate_response(response, CompletionResponse)

    def stream(self, request: CompletionRequest) -> Iterator[bytes]:
        payload = request.model_copy(update={"stream": True}).model_dump(mode="json")

        def iterate() -> Iterator[bytes]:
            try:
                with self._client.stream(
                    "POST",
                    f"{self.base_url}/v1/completions",
                    headers=self._headers(),
                    json=payload,
                ) as response:
                    _raise_for_status(response)
                    yield from response.iter_raw()
            except FlumeError:
                raise
            except httpx.HTTPError as exc:
                raise FlumeTransportError("Flume streaming request failed") from exc

        return iterate()

    def stats(self) -> StatsResponse:
        response = self._request("GET", "/v1/stats")
        return _validate_response(response, StatsResponse)

    def _request(self, method: str, path: str, **kwargs: Any) -> httpx.Response:
        try:
            return self._client.request(
                method,
                f"{self.base_url}{path}",
                headers=self._headers(),
                **kwargs,
            )
        except httpx.HTTPError as exc:
            raise FlumeTransportError("Flume API request failed") from exc

    def _headers(self) -> dict[str, str]:
        return {"X-Flume-Tenant": self.tenant_id}


class AsyncFlumeClient:
    """Persistent asynchronous Flume v1 client."""

    def __init__(
        self,
        base_url: str = "http://localhost:8080",
        *,
        tenant_id: str = "default",
        timeout_seconds: float = 120.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.tenant_id = canonical_identifier(tenant_id)
        self._client = httpx.AsyncClient(
            timeout=timeout_seconds,
            transport=transport,
            limits=httpx.Limits(max_connections=100, max_keepalive_connections=20),
        )

    async def __aenter__(self) -> AsyncFlumeClient:
        return self

    async def __aexit__(self, *args: object) -> None:
        await self.close()

    @property
    def is_closed(self) -> bool:
        return self._client.is_closed

    async def close(self) -> None:
        await self._client.aclose()

    async def register_pack(self, request: PackRegistrationRequest) -> PackSummary:
        response = await self._request(
            "POST",
            "/v1/packs",
            json=request.model_dump(mode="json"),
        )
        return _validate_response(response, PackSummary)

    async def list_packs(self, *, limit: int = 100, cursor: str | None = None) -> PackPage:
        params: dict[str, str | int] = {"limit": limit}
        if cursor is not None:
            params["cursor"] = cursor
        response = await self._request("GET", "/v1/packs", params=params)
        return _validate_response(response, PackPage)

    async def get_pack(self, pack_id: str) -> PackSummary:
        response = await self._request("GET", f"/v1/packs/{pack_id}")
        return _validate_response(response, PackSummary)

    async def warm_pack(
        self,
        pack_id: str,
        request: WarmRequest | None = None,
    ) -> WarmResponse:
        response = await self._request(
            "POST",
            f"/v1/packs/{pack_id}/warm",
            json=(request or WarmRequest()).model_dump(mode="json"),
        )
        return _validate_response(response, WarmResponse)

    async def complete(self, request: CompletionRequest) -> CompletionResponse:
        payload = request.model_copy(update={"stream": False})
        response = await self._request(
            "POST",
            "/v1/completions",
            json=payload.model_dump(mode="json"),
        )
        return _validate_response(response, CompletionResponse)

    def stream(self, request: CompletionRequest) -> AsyncIterator[bytes]:
        payload = request.model_copy(update={"stream": True}).model_dump(mode="json")

        async def iterate() -> AsyncIterator[bytes]:
            try:
                async with self._client.stream(
                    "POST",
                    f"{self.base_url}/v1/completions",
                    headers=self._headers(),
                    json=payload,
                ) as response:
                    _raise_for_status(response)
                    async for chunk in response.aiter_raw():
                        yield chunk
            except FlumeError:
                raise
            except httpx.HTTPError as exc:
                raise FlumeTransportError("Flume streaming request failed") from exc

        return iterate()

    async def stats(self) -> StatsResponse:
        response = await self._request("GET", "/v1/stats")
        return _validate_response(response, StatsResponse)

    async def _request(self, method: str, path: str, **kwargs: Any) -> httpx.Response:
        try:
            return await self._client.request(
                method,
                f"{self.base_url}{path}",
                headers=self._headers(),
                **kwargs,
            )
        except httpx.HTTPError as exc:
            raise FlumeTransportError("Flume API request failed") from exc

    def _headers(self) -> dict[str, str]:
        return {"X-Flume-Tenant": self.tenant_id}

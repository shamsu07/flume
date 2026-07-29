import json
from typing import Any

import httpx
import pytest

from flume.models import (
    CompletionRequest,
    CompletionResponse,
    DocumentChunk,
    PackPage,
    PackRegistrationRequest,
    PackSummary,
    StatsResponse,
    WarmResponse,
)
from flume.sdk import (
    AsyncFlumeClient,
    FlumeAPIError,
    FlumeClient,
    FlumeTransportError,
)


def pack_summary() -> dict[str, Any]:
    return {
        "pack_id": "pack_123",
        "compiler_format_version": "1",
        "model_id": "model",
        "tokenizer_id": "tokenizer",
        "tokenizer_revision": "deadbeef",
        "tokenizer_fingerprint": "a" * 64,
        "template_id": "default-rag-v1",
        "document_hash": "b" * 64,
        "canonical_prefix_hash": "c" * 64,
        "token_count": 42,
        "created_at": "2026-07-29T00:00:00Z",
        "ttl_seconds": None,
        "tags": [],
    }


def completion_response() -> dict[str, Any]:
    return {
        "id": "cmpl-123",
        "object": "text_completion",
        "created": 1,
        "model": "model",
        "choices": [
            {
                "text": "answer",
                "index": 0,
                "logprobs": None,
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 42,
            "completion_tokens": 1,
            "total_tokens": 43,
        },
    }


def sdk_handler(requests: list[httpx.Request]):
    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        path = request.url.path
        if path == "/v1/packs" and request.method == "POST":
            return httpx.Response(200, json=pack_summary())
        if path == "/v1/packs":
            return httpx.Response(
                200,
                json={"items": [pack_summary()], "next_cursor": "next"},
            )
        if path == "/v1/packs/pack_123":
            return httpx.Response(200, json=pack_summary())
        if path == "/v1/packs/pack_123/warm":
            return httpx.Response(
                200,
                json={
                    "pack_id": "pack_123",
                    "worker_id": "worker123",
                    "warmed": True,
                    "latency_ms": 1.25,
                },
            )
        if path == "/v1/completions":
            body = json.loads(request.content)
            if body["stream"]:
                return httpx.Response(
                    200,
                    stream=httpx.ByteStream(
                        b': keepalive\r\n\r\ndata: {"choices":[{"text":"a"}]}\r\n\r\n'
                    ),
                    headers={"content-type": "text/event-stream"},
                )
            return httpx.Response(200, json=completion_response())
        if path == "/v1/stats":
            return httpx.Response(
                200,
                json={
                    "packs": 1,
                    "routes": 0,
                    "workers": [
                        {
                            "worker_id": "worker123",
                            "healthy": True,
                            "assigned_packs": 1,
                            "affinity_hits": 2,
                            "affinity_misses": 0,
                        }
                    ],
                },
            )
        return httpx.Response(404, json={"detail": "missing"})

    return handler


def registration() -> PackRegistrationRequest:
    return PackRegistrationRequest(chunks=[DocumentChunk(doc_id="doc", text="hello")])


def completion() -> CompletionRequest:
    return CompletionRequest(pack_id="pack_123", prompt="question")


def test_sync_client_uses_v1_typed_responses_tenant_header_and_lifecycle() -> None:
    requests: list[httpx.Request] = []
    client = FlumeClient(
        "http://flume",
        tenant_id="tenant-a",
        transport=httpx.MockTransport(sdk_handler(requests)),
    )

    with client:
        registered = client.register_pack(registration())
        page = client.list_packs(limit=2, cursor="cursor")
        loaded = client.get_pack("pack_123")
        warmed = client.warm_pack("pack_123")
        completed = client.complete(completion())
        streamed = b"".join(client.stream(completion()))
        stats = client.stats()

    assert isinstance(registered, PackSummary)
    assert isinstance(page, PackPage)
    assert isinstance(loaded, PackSummary)
    assert isinstance(warmed, WarmResponse)
    assert isinstance(completed, CompletionResponse)
    assert isinstance(stats, StatsResponse)
    assert completed.choices[0].text == "answer"
    assert b"keepalive" in streamed
    assert client.is_closed
    assert {request.headers["X-Flume-Tenant"] for request in requests} == {"tenant-a"}
    assert dict(requests[1].url.params) == {"limit": "2", "cursor": "cursor"}
    registration_body = json.loads(requests[0].content)
    assert "tenant_id" not in registration_body
    assert "model_id" not in registration_body
    assert "tokenizer_id" not in registration_body


@pytest.mark.asyncio
async def test_async_client_uses_typed_responses_streaming_and_lifecycle() -> None:
    requests: list[httpx.Request] = []
    client = AsyncFlumeClient(
        "http://flume",
        tenant_id="tenant-b",
        transport=httpx.MockTransport(sdk_handler(requests)),
    )

    async with client:
        registered = await client.register_pack(registration())
        page = await client.list_packs()
        loaded = await client.get_pack("pack_123")
        warmed = await client.warm_pack("pack_123")
        completed = await client.complete(completion())
        streamed = b"".join([chunk async for chunk in client.stream(completion())])
        stats = await client.stats()

    assert isinstance(registered, PackSummary)
    assert isinstance(page, PackPage)
    assert isinstance(loaded, PackSummary)
    assert isinstance(warmed, WarmResponse)
    assert isinstance(completed, CompletionResponse)
    assert isinstance(stats, StatsResponse)
    assert b"data:" in streamed
    assert client.is_closed
    assert {request.headers["X-Flume-Tenant"] for request in requests} == {"tenant-b"}


@pytest.mark.parametrize(
    "payload",
    [
        {"pack_id": "pack", "prompt": "question", "unknown": True},
        {"pack_id": "pack", "prompt": "question", "max_tokens": "2"},
        {"pack_id": "pack", "prompt": "question", "temperature": 2.1},
        {"pack_id": "pack", "prompt": "question", "top_p": 0},
        {"pack_id": "pack", "prompt": "question", "stop": []},
        {"pack_id": "pack", "prompt": "question", "stop": ["x"] * 5},
    ],
)
def test_completion_request_is_strict_and_bounded(payload: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        CompletionRequest.model_validate(payload)


def test_sdk_exposes_typed_api_errors_without_raw_bodies() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            404,
            json={"detail": "context pack not found", "secret": "do-not-expose"},
            headers={"X-Request-Id": "request-123"},
        )

    with FlumeClient("http://flume", transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(FlumeAPIError, match="context pack not found") as caught:
            client.get_pack("missing")

    assert caught.value.status_code == 404
    assert caught.value.detail == "context pack not found"
    assert caught.value.request_id == "request-123"
    assert "do-not-expose" not in str(caught.value)


def test_sdk_wraps_invalid_responses_and_transport_failures() -> None:
    def invalid(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"not-json")

    with FlumeClient("http://flume", transport=httpx.MockTransport(invalid)) as client:
        with pytest.raises(FlumeTransportError, match="invalid JSON"):
            client.get_pack("pack")

    def unavailable(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("sensitive network detail", request=request)

    with FlumeClient("http://flume", transport=httpx.MockTransport(unavailable)) as client:
        with pytest.raises(FlumeTransportError, match="API request failed") as caught:
            client.get_pack("pack")

    assert "sensitive" not in str(caught.value)

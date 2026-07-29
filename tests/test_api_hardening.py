import asyncio
import json
from collections.abc import AsyncIterator
from typing import Any, cast
from unittest.mock import AsyncMock, Mock

import httpx
import pytest
from fastapi.testclient import TestClient

from flume.api import AdmissionController, _stream_with_release, create_app
from flume.compiler import ContextPackCompiler, DeterministicByteTokenizer
from flume.config import INSECURE_CACHE_SALT_SECRET, Settings
from flume.store import PackConflictError

SECRET = "test-secret-that-is-at-least-thirty-two-bytes"


class FragmentedStream(httpx.AsyncByteStream):
    def __init__(self, chunks: list[bytes]) -> None:
        self.chunks = chunks

    async def __aiter__(self) -> AsyncIterator[bytes]:
        for chunk in self.chunks:
            yield chunk


class FailingStream(httpx.AsyncByteStream):
    async def __aiter__(self) -> AsyncIterator[bytes]:
        yield b'data: {"choices":[{"text":"partial"}]}\n\n'
        raise RuntimeError("upstream stream failed")


def default_handler(request: httpx.Request) -> httpx.Response:
    if request.url.path == "/health":
        return httpx.Response(200)
    return httpx.Response(
        200,
        json={
            "choices": [{"text": "answer", "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 12, "completion_tokens": 1},
        },
    )


def build_app(
    tmp_path: Any,
    *,
    handler: Any = default_handler,
    compiler: ContextPackCompiler | None | object = ...,
    **settings_overrides: Any,
):
    settings_values: dict[str, Any] = {
        "database_url": f"sqlite:///{tmp_path / 'flume.db'}",
        "vllm_workers": ["http://worker"],
        "model_id": "model",
        "tokenizer_id": "tokenizer",
        "tokenizer_revision": "byte-v1",
        "cache_salt_secret": SECRET,
        **settings_overrides,
    }
    runtime_compiler = compiler
    if compiler is ...:
        tokenizer = DeterministicByteTokenizer(
            tokenizer_id="tokenizer",
            revision="byte-v1",
        )
        runtime_compiler = ContextPackCompiler(tokenizer, model_id="model")
    return create_app(
        Settings(**settings_values),
        compiler=cast(ContextPackCompiler | None, runtime_compiler),
        vllm_transport=httpx.MockTransport(handler),
    )


def register_pack(client: TestClient, tenant: str = "demo") -> dict[str, Any]:
    response = client.post(
        "/v1/packs",
        headers={"X-Flume-Tenant": tenant},
        json={"chunks": [{"doc_id": "doc", "text": "hello"}]},
    )
    assert response.status_code == 200, response.text
    return response.json()


def test_readiness_reports_all_failed_dependencies(tmp_path: Any, monkeypatch: Any) -> None:
    def fail_tokenizer(**_: Any) -> None:
        raise RuntimeError("tokenizer unavailable")

    monkeypatch.setattr(
        ContextPackCompiler,
        "from_pretrained",
        staticmethod(fail_tokenizer),
    )

    def unhealthy(_: httpx.Request) -> httpx.Response:
        return httpx.Response(503)

    app = build_app(
        tmp_path,
        compiler=None,
        handler=unhealthy,
        cache_salt_secret=INSECURE_CACHE_SALT_SECRET,
    )
    with TestClient(app) as client:
        app.state.store.ping = AsyncMock(return_value=False)
        response = client.get("/readyz")

    assert response.status_code == 503
    assert response.json() == {
        "status": "not_ready",
        "checks": {
            "database": False,
            "tokenizer": False,
            "worker": False,
            "cache_salt_secret": False,
        },
    }


def test_request_body_and_content_length_limits(tmp_path: Any) -> None:
    app = build_app(tmp_path, max_request_body_bytes=32)
    with TestClient(app) as client:
        invalid = client.get("/livez", headers={"content-length": "invalid"})
        declared = client.post(
            "/v1/packs",
            headers={"X-Flume-Tenant": "demo", "content-length": "33"},
            content=b"{}",
        )
        request = client.build_request(
            "POST",
            "/v1/packs",
            headers={"X-Flume-Tenant": "demo", "transfer-encoding": "chunked"},
            content=iter([b"x" * 20, b"y" * 20]),
        )
        request.headers.pop("content-length", None)
        actual = client.send(request)

    assert invalid.status_code == 400
    assert invalid.json()["detail"] == "invalid content-length"
    assert declared.status_code == 413
    assert actual.status_code == 413


@pytest.mark.parametrize(
    ("method", "path", "body"),
    [
        ("POST", "/v1/packs", {"chunks": [{"doc_id": "doc", "text": "hello"}]}),
        ("GET", "/v1/packs", None),
        ("GET", "/v1/packs/missing", None),
        ("PATCH", "/v1/packs/missing/annotations", {"tag": "value"}),
        ("POST", "/v1/packs/missing/warm", {}),
        ("POST", "/v1/completions", {"pack_id": "missing", "prompt": "question"}),
        ("GET", "/v1/stats", None),
    ],
)
def test_tenant_header_is_enforced(
    tmp_path: Any,
    method: str,
    path: str,
    body: dict[str, Any] | None,
) -> None:
    app = build_app(tmp_path)
    with TestClient(app) as client:
        response = client.request(method, path, json=body)

    assert response.status_code == 422


def test_missing_pack_annotation_warmup_and_completion_paths(tmp_path: Any) -> None:
    app = build_app(tmp_path)
    headers = {"X-Flume-Tenant": "demo"}
    with TestClient(app) as client:
        loaded = client.get("/v1/packs/missing", headers=headers)
        annotated = client.patch(
            "/v1/packs/missing/annotations",
            headers=headers,
            json={"tag": "value"},
        )
        warmed = client.post("/v1/packs/missing/warm", headers=headers, json={})
        completed = client.post(
            "/v1/completions",
            headers=headers,
            json={"pack_id": "missing", "prompt": "question"},
        )
        invalid_cursor = client.get(
            "/v1/packs",
            headers=headers,
            params={"cursor": "not-a-cursor"},
        )

    assert loaded.status_code == 404
    assert annotated.status_code == 404
    assert warmed.status_code == 404
    assert completed.status_code == 404
    assert invalid_cursor.status_code == 400


def test_pack_compile_limit_conflict_and_annotation_success_paths(
    tmp_path: Any,
) -> None:
    app = build_app(tmp_path)
    headers = {"X-Flume-Tenant": "demo"}
    with TestClient(app) as client:
        app.state.compiler.compile = Mock(side_effect=ValueError("invalid pack"))
        invalid = client.post(
            "/v1/packs",
            headers=headers,
            json={"chunks": [{"doc_id": "doc", "text": "hello"}]},
        )

    assert invalid.status_code == 422
    assert invalid.json()["detail"] == "invalid pack"

    limited_app = build_app(tmp_path, max_pack_tokens=1)
    with TestClient(limited_app) as client:
        limited = client.post(
            "/v1/packs",
            headers=headers,
            json={"chunks": [{"doc_id": "doc", "text": "hello"}]},
        )
    assert limited.status_code == 422
    assert "max is 1" in limited.json()["detail"]

    conflict_app = build_app(tmp_path)
    with TestClient(conflict_app) as client:
        conflict_app.state.store.save_pack = AsyncMock(side_effect=PackConflictError("conflict"))
        conflict = client.post(
            "/v1/packs",
            headers=headers,
            json={"chunks": [{"doc_id": "doc", "text": "hello"}]},
        )
    assert conflict.status_code == 409

    annotation_app = build_app(tmp_path)
    with TestClient(annotation_app) as client:
        pack = register_pack(client)
        annotated = client.patch(
            f"/v1/packs/{pack['pack_id']}/annotations",
            headers=headers,
            json={"owner": "team"},
        )
    assert annotated.status_code == 200
    assert annotated.json()["annotations"] == {"owner": "team"}


@pytest.mark.parametrize(
    ("failure", "expected_status"),
    [
        ("upstream", 502),
        ("connection", 503),
    ],
)
def test_warmup_upstream_and_unavailable_worker_paths(
    tmp_path: Any,
    failure: str,
    expected_status: int,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(200)
        if failure == "connection":
            raise httpx.ConnectError("unavailable", request=request)
        return httpx.Response(503, content=b"sensitive upstream body")

    app = build_app(tmp_path, handler=handler)
    with TestClient(app) as client:
        pack = register_pack(client)
        response = client.post(
            f"/v1/packs/{pack['pack_id']}/warm",
            headers={"X-Flume-Tenant": "demo"},
            json={},
        )

    assert response.status_code == expected_status
    assert "sensitive upstream body" not in response.text


@pytest.mark.parametrize(
    ("failure", "expected_status"),
    [
        ("upstream", 502),
        ("connection", 503),
    ],
)
def test_completion_upstream_and_unavailable_worker_paths(
    tmp_path: Any,
    failure: str,
    expected_status: int,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(200)
        if failure == "connection":
            raise httpx.ConnectError("unavailable", request=request)
        return httpx.Response(503, content=b"sensitive upstream body")

    app = build_app(tmp_path, handler=handler)
    with TestClient(app) as client:
        pack = register_pack(client)
        response = client.post(
            "/v1/completions",
            headers={"X-Flume-Tenant": "demo"},
            json={"pack_id": pack["pack_id"], "prompt": "question"},
        )

    assert response.status_code == expected_status
    assert "sensitive upstream body" not in response.text


def test_fragmented_stream_is_forwarded_and_releases_admission(tmp_path: Any) -> None:
    raw = b': keepalive\r\n\r\ndata: {"choices":[{"text":"answer"}]}\r\n\r\ndata: [DONE]\r\n\r\n'

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(200)
        payload = json.loads(request.content)
        if payload["stream"]:
            return httpx.Response(
                200,
                stream=FragmentedStream([raw[:5], raw[5:29], raw[29:53], raw[53:]]),
            )
        return default_handler(request)

    app = build_app(tmp_path, handler=handler, max_in_flight=1)
    with TestClient(app) as client:
        pack = register_pack(client)
        streamed = client.post(
            "/v1/completions",
            headers={"X-Flume-Tenant": "demo"},
            json={"pack_id": pack["pack_id"], "prompt": "question", "stream": True},
        )
        following = client.post(
            "/v1/completions",
            headers={"X-Flume-Tenant": "demo"},
            json={"pack_id": pack["pack_id"], "prompt": "question"},
        )

    assert streamed.status_code == 200
    assert streamed.content == raw
    assert following.status_code == 200
    assert app.state.router.local_in_flight("http://worker") == 0


def test_streaming_error_releases_admission(tmp_path: Any) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(200)
        payload = json.loads(request.content)
        if payload["stream"]:
            return httpx.Response(200, stream=FailingStream())
        return default_handler(request)

    app = build_app(tmp_path, handler=handler, max_in_flight=1)
    with TestClient(app) as client:
        pack = register_pack(client)
        with pytest.raises(Exception, match="upstream stream failed"):
            client.post(
                "/v1/completions",
                headers={"X-Flume-Tenant": "demo"},
                json={"pack_id": pack["pack_id"], "prompt": "question", "stream": True},
            )
        following = client.post(
            "/v1/completions",
            headers={"X-Flume-Tenant": "demo"},
            json={"pack_id": pack["pack_id"], "prompt": "question"},
        )

    assert following.status_code == 200
    assert app.state.router.local_in_flight("http://worker") == 0


@pytest.mark.asyncio
async def test_streaming_cancellation_releases_admission() -> None:
    async def cancelled() -> AsyncIterator[bytes]:
        if False:
            yield b""
        raise asyncio.CancelledError

    admission = AdmissionController(1)
    assert admission.acquire()
    stream = _stream_with_release(
        cast(Any, cancelled()),
        admission,
        "http://worker",
    )

    with pytest.raises(asyncio.CancelledError):
        await anext(stream)

    assert admission.current == 0


def test_overload_and_metrics_disabled_paths(tmp_path: Any, monkeypatch: Any) -> None:
    app = build_app(tmp_path, metrics_enabled=False)
    with TestClient(app) as client:
        pack = register_pack(client)
        monkeypatch.setattr(AdmissionController, "acquire", lambda _: False)
        overloaded = client.post(
            "/v1/completions",
            headers={"X-Flume-Tenant": "demo"},
            json={"pack_id": pack["pack_id"], "prompt": "question"},
        )
        metrics = client.get("/metrics")

    assert overloaded.status_code == 429
    assert metrics.status_code == 404
    assert metrics.json()["detail"] == "metrics are disabled"

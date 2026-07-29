import json
import logging
from unittest.mock import AsyncMock

import httpx
from fastapi.testclient import TestClient

from flume.api import create_app
from flume.compiler import ContextPackCompiler, DeterministicByteTokenizer
from flume.config import Settings


def build_app(tmp_path, payloads: list[dict] | None = None):
    payloads = payloads if payloads is not None else []

    async def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(200)
        payloads.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "choices": [{"text": "answer", "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 12, "completion_tokens": 1},
            },
        )

    tokenizer = DeterministicByteTokenizer(
        tokenizer_id="tokenizer",
        revision="byte-v1",
    )
    compiler = ContextPackCompiler(tokenizer, model_id="model")
    settings = Settings(
        database_url=f"sqlite:///{tmp_path / 'flume.db'}",
        vllm_workers=["http://worker"],
        model_id="model",
        tokenizer_id="tokenizer",
        tokenizer_revision="byte-v1",
        cache_salt_secret="test-secret-that-is-at-least-thirty-two-bytes",
    )
    return create_app(
        settings,
        compiler=compiler,
        vllm_transport=httpx.MockTransport(handler),
    )


def register_pack(client: TestClient, tenant: str = "demo") -> dict:
    response = client.post(
        "/v1/packs",
        headers={"X-Flume-Tenant": tenant},
        json={
            "tenant_id": "untrusted-body-tenant",
            "model_id": "untrusted-model",
            "tokenizer_id": "untrusted-tokenizer",
            "chunks": [{"doc_id": "doc", "text": "hello"}],
        },
    )
    assert response.status_code == 200, response.text
    return response.json()


def test_liveness_and_readiness(tmp_path) -> None:
    app = build_app(tmp_path)

    with TestClient(app) as client:
        assert client.get("/livez").json() == {"status": "ok"}
        ready = client.get("/readyz")

    assert ready.status_code == 200
    assert ready.json()["status"] == "ready"


def test_pack_registration_is_tenant_scoped_and_safe(tmp_path) -> None:
    app = build_app(tmp_path)
    with TestClient(app) as client:
        pack = register_pack(client)
        repeated = register_pack(client)

        assert repeated == pack
        assert "compiled_prefix" not in pack
        assert "prefix_token_ids" not in pack
        loaded = client.get(
            f"/v1/packs/{pack['pack_id']}",
            headers={"X-Flume-Tenant": "demo"},
        )
        hidden = client.get(
            f"/v1/packs/{pack['pack_id']}",
            headers={"X-Flume-Tenant": "other"},
        )

    assert loaded.status_code == 200
    assert hidden.status_code == 404


def test_completion_uses_integer_prompt_and_hmac_tenant_salt(tmp_path) -> None:
    payloads: list[dict] = []
    app = build_app(tmp_path, payloads)
    with TestClient(app) as client:
        pack = register_pack(client)
        response = client.post(
            "/v1/completions",
            headers={"X-Flume-Tenant": "demo"},
            json={
                "pack_id": pack["pack_id"],
                "question": "What is this?",
                "max_tokens": 8,
            },
        )

    assert response.status_code == 200, response.text
    assert response.json()["object"] == "text_completion"
    assert response.headers["X-Flume-Pack-Id"] == pack["pack_id"]
    assert isinstance(payloads[-1]["prompt"], list)
    assert all(isinstance(token_id, int) for token_id in payloads[-1]["prompt"])
    assert payloads[-1]["cache_salt"] != "demo"
    assert len(payloads[-1]["cache_salt"]) == 64


def test_tenant_header_is_required_and_reserved_fields_are_rejected(tmp_path) -> None:
    app = build_app(tmp_path)
    with TestClient(app) as client:
        assert client.get("/v1/packs").status_code == 422
        pack = register_pack(client)
        response = client.post(
            "/v1/completions",
            headers={"X-Flume-Tenant": "demo"},
            json={
                "pack_id": pack["pack_id"],
                "question": "question",
                "extra_body": {"cache_salt": "attacker"},
            },
        )

    assert response.status_code == 422
    assert "reserved" in response.json()["detail"]


def test_hot_completion_avoids_store_and_health_and_logs_are_redacted(
    tmp_path,
    caplog,
) -> None:
    payloads: list[dict] = []
    health_calls = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal health_calls
        if request.url.path == "/health":
            health_calls += 1
            return httpx.Response(200)
        payloads.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "choices": [{"text": "answer", "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 12, "completion_tokens": 1},
            },
        )

    tokenizer = DeterministicByteTokenizer(tokenizer_id="tokenizer", revision="byte-v1")
    app = create_app(
        Settings(
            database_url=f"sqlite:///{tmp_path / 'flume.db'}",
            vllm_workers=["http://worker"],
            model_id="model",
            tokenizer_id="tokenizer",
            tokenizer_revision="byte-v1",
            cache_salt_secret="test-secret-that-is-at-least-thirty-two-bytes",
            health_refresh_seconds=60,
        ),
        compiler=ContextPackCompiler(tokenizer, model_id="model"),
        vllm_transport=httpx.MockTransport(handler),
    )
    caplog.set_level(logging.INFO, logger="flume.request")
    with TestClient(app) as client:
        pack = register_pack(client, tenant="sensitive-customer")
        startup_health_calls = health_calls
        app.state.store.get_pack = AsyncMock(side_effect=AssertionError("hot path hit SQLite"))

        response = client.post(
            "/v1/completions",
            headers={"X-Flume-Tenant": "sensitive-customer"},
            json={
                "pack_id": pack["pack_id"],
                "question": "secret question content",
                "max_tokens": 1,
            },
        )
        metrics = client.get("/metrics").text

    assert response.status_code == 200
    assert health_calls == startup_health_calls
    logs = "\n".join(record.getMessage() for record in caplog.records)
    assert "secret question content" not in logs
    assert "sensitive-customer" not in logs
    assert "sensitive-customer" not in metrics
    assert "http://worker" not in metrics

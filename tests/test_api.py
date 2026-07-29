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


def test_pack_registration_rejects_runtime_identity_in_body(tmp_path) -> None:
    app = build_app(tmp_path)
    with TestClient(app) as client:
        response = client.post(
            "/v1/packs",
            headers={"X-Flume-Tenant": "demo"},
            json={
                "tenant_id": "body-tenant",
                "model_id": "body-model",
                "tokenizer_id": "body-tokenizer",
                "chunks": [{"doc_id": "doc", "text": "hello"}],
            },
        )

    assert response.status_code == 422
    locations = {tuple(error["loc"]) for error in response.json()["detail"]}
    assert ("body", "tenant_id") in locations
    assert ("body", "model_id") in locations
    assert ("body", "tokenizer_id") in locations


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
                "prompt": "What is this?",
                "max_tokens": 8,
            },
        )

    assert response.status_code == 200, response.text
    assert response.json()["object"] == "text_completion"
    assert response.headers["X-Flume-Pack-Id"] == pack["pack_id"]
    assert response.headers["X-Flume-Worker-Id"]
    assert "X-Flume-Worker-Url" not in response.headers
    assert "http://worker" not in response.text
    assert isinstance(payloads[-1]["prompt"], list)
    assert all(isinstance(token_id, int) for token_id in payloads[-1]["prompt"])
    assert payloads[-1]["cache_salt"] != "demo"
    assert len(payloads[-1]["cache_salt"]) == 64
    assert app.state.router.local_in_flight("http://worker") == 0


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
                "prompt": "question",
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
                "prompt": "secret question content",
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
    assert 'flume_router_decisions_total{decision="primary",policy="hrw"}' in metrics
    assert "flume_router_decision_duration_seconds_count" in metrics
    assert 'flume_router_worker_load{source="local",worker_id="' in metrics


def test_bounded_routing_refreshes_load_only_in_background(tmp_path) -> None:
    metrics_calls = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal metrics_calls
        if request.url.path == "/health":
            return httpx.Response(200)
        if request.url.path == "/metrics":
            metrics_calls += 1
            return httpx.Response(
                200,
                text=(
                    "vllm:num_requests_running 0\n"
                    "vllm:num_requests_waiting 0\n"
                ),
            )
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
            vllm_workers=["http://worker-a", "http://worker-b"],
            model_id="model",
            tokenizer_id="tokenizer",
            tokenizer_revision="byte-v1",
            cache_salt_secret="test-secret-that-is-at-least-thirty-two-bytes",
            health_refresh_seconds=60,
            routing_policy="bounded_hrw",
            worker_load_refresh_ms=60_000,
            worker_load_stale_ms=60_000,
        ),
        compiler=ContextPackCompiler(tokenizer, model_id="model"),
        vllm_transport=httpx.MockTransport(handler),
    )
    with TestClient(app) as client:
        pack = register_pack(client)
        startup_metrics_calls = metrics_calls
        response = client.post(
            "/v1/completions",
            headers={"X-Flume-Tenant": "demo"},
            json={"pack_id": pack["pack_id"], "prompt": "question"},
        )

    assert response.status_code == 200
    assert startup_metrics_calls == 2
    assert metrics_calls == startup_metrics_calls


def test_warm_and_stats_expose_only_safe_worker_ids(tmp_path) -> None:
    app = build_app(tmp_path)
    with TestClient(app) as client:
        pack = register_pack(client)
        warmed = client.post(
            f"/v1/packs/{pack['pack_id']}/warm",
            headers={"X-Flume-Tenant": "demo"},
            json={},
        )
        stats = client.get("/v1/stats", headers={"X-Flume-Tenant": "demo"})

    assert warmed.status_code == 200
    assert warmed.json()["worker_id"]
    assert "worker_url" not in warmed.json()
    assert "http://" not in warmed.text
    assert stats.status_code == 200
    worker_stats = stats.json()["workers"][0]
    assert worker_stats["worker_id"]
    assert worker_stats["affinity_misses"] == 1
    assert worker_stats["affinity_hits"] == 0
    assert worker_stats["local_in_flight"] == 0
    assert worker_stats["capacity_weight"] == 1.0
    assert "worker_url" not in worker_stats
    assert "http://" not in stats.text


def test_api_failover_tries_two_distinct_workers_once(tmp_path) -> None:
    completion_hosts: list[str] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(200)
        completion_hosts.append(request.url.host or "")
        if len(completion_hosts) == 1:
            raise httpx.ConnectError("first worker unavailable", request=request)
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
            vllm_workers=["http://worker-a", "http://worker-b"],
            model_id="model",
            tokenizer_id="tokenizer",
            tokenizer_revision="byte-v1",
            cache_salt_secret="test-secret-that-is-at-least-thirty-two-bytes",
        ),
        compiler=ContextPackCompiler(tokenizer, model_id="model"),
        vllm_transport=httpx.MockTransport(handler),
    )
    with TestClient(app) as client:
        pack = register_pack(client)
        response = client.post(
            "/v1/completions",
            headers={"X-Flume-Tenant": "demo"},
            json={"pack_id": pack["pack_id"], "prompt": "question"},
        )

    assert response.status_code == 200
    assert len(completion_hosts) == 2
    assert len(set(completion_hosts)) == 2
    assert response.headers["X-Flume-Worker-Id"]
    assert all(host not in response.text for host in completion_hosts)
    assert app.state.router.local_in_flight("http://worker-a") == 0
    assert app.state.router.local_in_flight("http://worker-b") == 0


def test_stream_upstream_error_is_translated_before_sse_headers(tmp_path) -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(200)
        return httpx.Response(503, content=b"sensitive upstream body")

    tokenizer = DeterministicByteTokenizer(tokenizer_id="tokenizer", revision="byte-v1")
    app = create_app(
        Settings(
            database_url=f"sqlite:///{tmp_path / 'flume.db'}",
            vllm_workers=["http://worker"],
            model_id="model",
            tokenizer_id="tokenizer",
            tokenizer_revision="byte-v1",
            cache_salt_secret="test-secret-that-is-at-least-thirty-two-bytes",
        ),
        compiler=ContextPackCompiler(tokenizer, model_id="model"),
        vllm_transport=httpx.MockTransport(handler),
    )
    with TestClient(app) as client:
        pack = register_pack(client)
        response = client.post(
            "/v1/completions",
            headers={"X-Flume-Tenant": "demo"},
            json={"pack_id": pack["pack_id"], "prompt": "question", "stream": True},
        )

    assert response.status_code == 502
    assert response.headers["content-type"].startswith("application/json")
    assert "sensitive upstream body" not in response.text


def test_completion_schema_rejects_unknown_and_out_of_bounds_fields(tmp_path) -> None:
    app = build_app(tmp_path)
    with TestClient(app) as client:
        pack = register_pack(client)
        unknown = client.post(
            "/v1/completions",
            headers={"X-Flume-Tenant": "demo"},
            json={"pack_id": pack["pack_id"], "prompt": "question", "model": "override"},
        )
        too_many = client.post(
            "/v1/completions",
            headers={"X-Flume-Tenant": "demo"},
            json={"pack_id": pack["pack_id"], "prompt": "question", "max_tokens": 4097},
        )

    assert unknown.status_code == 422
    assert too_many.status_code == 422

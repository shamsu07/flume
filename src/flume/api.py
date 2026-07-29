from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import logging
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import httpx
from fastapi import FastAPI, Header, HTTPException, Query, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from flume.compiler import ContextPackCompiler
from flume.config import INSECURE_CACHE_SALT_SECRET, Settings, get_settings
from flume.metrics import (
    ASK_LATENCY,
    ASKS_TOTAL,
    IN_FLIGHT,
    OVERLOADS,
    PACK_CACHE,
    PACKS_CREATED,
    REQUEST_LATENCY,
    REQUESTS_TOTAL,
    ROUTER_FAILOVERS,
    TTFT,
    WARMUP_LATENCY,
    WARMUPS_TOTAL,
)
from flume.models import (
    CompletionRequest,
    ContextPack,
    PackCreateRequest,
    PackPage,
    PackRegistrationRequest,
    PackSummary,
    StatsResponse,
    WarmRequest,
    WarmResponse,
    WorkerStats,
)
from flume.router import NoHealthyWorkers, PackRouter, WarmupSingleFlight
from flume.store import ByteBoundedPackCache, FlumeStore, PackConflictError
from flume.vllm import (
    CompletionResult,
    VLLMClient,
    VLLMConnectionError,
    VLLMError,
    VLLMStream,
)

TENANT_HEADER = Header(alias="X-Flume-Tenant", min_length=1, max_length=128)
LOGGER = logging.getLogger("flume.request")


class AdmissionController:
    def __init__(self, maximum: int):
        self.maximum = maximum
        self.current = 0

    def acquire(self) -> bool:
        if self.current >= self.maximum:
            OVERLOADS.inc()
            return False
        self.current += 1
        IN_FLIGHT.inc()
        return True

    def release(self) -> None:
        if self.current > 0:
            self.current -= 1
            IN_FLIGHT.dec()


def create_app(
    settings: Settings | None = None,
    *,
    compiler: ContextPackCompiler | None = None,
    vllm_transport: httpx.AsyncBaseTransport | None = None,
) -> FastAPI:
    settings = settings or get_settings()
    store = FlumeStore(
        settings.database_url,
        busy_timeout_ms=settings.sqlite_busy_timeout_ms,
    )
    vllm = VLLMClient(
        timeout_seconds=settings.request_timeout_seconds,
        connect_timeout_seconds=settings.connect_timeout_seconds,
        transport=vllm_transport,
    )
    router = PackRouter(
        settings.vllm_workers,
        health_checker=vllm.health,
        refresh_seconds=settings.health_refresh_seconds,
        routing_policy=settings.routing_policy,
        load_slack=settings.routing_load_slack,
        spill_hold_seconds=settings.routing_spill_hold_ms / 1_000,
        capacity_weights=settings.worker_capacity_weights,
        state_max_entries=settings.routing_state_max_entries,
        state_ttl_seconds=settings.routing_state_ttl_seconds,
    )
    cache = ByteBoundedPackCache(settings.pack_cache_bytes)
    warmups: WarmupSingleFlight[tuple[CompletionResult, str]] = WarmupSingleFlight()
    admission = AdmissionController(settings.max_in_flight)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        await store.init_schema()
        await vllm.start()
        app.state.compiler_error = None
        if app.state.compiler is None:
            try:
                app.state.compiler = ContextPackCompiler.from_pretrained(
                    tokenizer_id=settings.tokenizer_id,
                    tokenizer_revision=settings.tokenizer_revision,
                    model_id=settings.model_id,
                    allow_remote_tokenizer=settings.allow_remote_tokenizer,
                )
            except Exception:
                app.state.compiler_error = "configured tokenizer could not be loaded"
        await router.start()
        try:
            yield
        finally:
            await router.close()
            await vllm.close()
            await store.close()

    app = FastAPI(
        title="Flume",
        version="0.2.0",
        description="Cache-stable long-context compiler and vLLM serving proxy.",
        lifespan=lifespan,
    )
    app.state.settings = settings
    app.state.store = store
    app.state.compiler = compiler
    app.state.vllm = vllm
    app.state.router = router
    app.state.pack_cache = cache

    @app.middleware("http")
    async def enforce_body_limit(request: Request, call_next: Any) -> Response:
        started = time.perf_counter()
        request_id = uuid.uuid4().hex
        status_code = 500
        content_length = request.headers.get("content-length")
        if content_length is not None:
            try:
                if int(content_length) > settings.max_request_body_bytes:
                    _record_request(request, 413, started, request_id)
                    return JSONResponse(
                        status_code=413,
                        content={"detail": "request body is too large"},
                    )
            except ValueError:
                _record_request(request, 400, started, request_id)
                return JSONResponse(status_code=400, content={"detail": "invalid content-length"})
        body = await request.body()
        if len(body) > settings.max_request_body_bytes:
            _record_request(request, 413, started, request_id)
            return JSONResponse(status_code=413, content={"detail": "request body is too large"})
        request._body = body
        try:
            response = await call_next(request)
            status_code = response.status_code
            response.headers["X-Request-Id"] = request_id
            return response
        finally:
            _record_request(request, status_code, started, request_id)

    @app.get("/livez")
    async def livez() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/readyz")
    async def readyz() -> Response:
        checks = {
            "database": await store.ping(),
            "tokenizer": app.state.compiler is not None,
            "worker": any(router.health().values()),
            "cache_salt_secret": settings.cache_salt_secret != INSECURE_CACHE_SALT_SECRET,
        }
        status_code = 200 if all(checks.values()) else 503
        return JSONResponse(
            status_code=status_code,
            content={"status": "ready" if status_code == 200 else "not_ready", "checks": checks},
        )

    @app.post("/v1/packs", response_model=PackSummary)
    async def create_pack(
        request: PackRegistrationRequest,
        tenant_id: str = TENANT_HEADER,
    ) -> PackSummary:
        runtime_compiler = _require_compiler(app)
        authoritative = PackCreateRequest(
            **request.model_dump(),
            tenant_id=tenant_id,
            model_id=settings.model_id,
            tokenizer_id=settings.tokenizer_id,
        )
        try:
            pack = runtime_compiler.compile(authoritative)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        if settings.max_pack_tokens and pack.token_count > settings.max_pack_tokens:
            raise HTTPException(
                status_code=422,
                detail=f"pack has {pack.token_count} tokens, max is {settings.max_pack_tokens}",
            )
        try:
            saved = await store.save_pack(pack)
        except PackConflictError as exc:
            raise HTTPException(status_code=409, detail="pack identity conflict") from exc
        cache.put(saved)
        PACKS_CREATED.inc()
        return saved.to_summary()

    @app.get("/v1/packs", response_model=PackPage)
    async def list_packs(
        tenant_id: str = TENANT_HEADER,
        limit: int = Query(default=100, ge=1, le=200),
        cursor: str | None = Query(default=None),
    ) -> PackPage:
        try:
            page = await store.list_packs(tenant_id, limit=limit, cursor=cursor)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail="invalid pagination cursor") from exc
        return PackPage(
            items=[pack.to_summary() for pack in page.items],
            next_cursor=page.next_cursor,
        )

    @app.get("/v1/packs/{pack_id}", response_model=PackSummary)
    async def get_pack(pack_id: str, tenant_id: str = TENANT_HEADER) -> PackSummary:
        return (await _load_pack(store, cache, tenant_id, pack_id)).to_summary()

    @app.patch("/v1/packs/{pack_id}/annotations")
    async def update_annotations(
        pack_id: str,
        annotations: dict[str, Any],
        tenant_id: str = TENANT_HEADER,
    ) -> dict[str, Any]:
        updated = await store.update_annotations(tenant_id, pack_id, annotations)
        if updated is None:
            raise HTTPException(status_code=404, detail="context pack not found")
        return {"pack_id": pack_id, "annotations": updated}

    @app.post("/v1/packs/{pack_id}/warm", response_model=WarmResponse)
    async def warm_pack(
        pack_id: str,
        request: WarmRequest | None = None,
        tenant_id: str = TENANT_HEADER,
    ) -> WarmResponse:
        runtime_compiler = _require_compiler(app)
        request = request or WarmRequest()
        pack = await _load_pack(store, cache, tenant_id, pack_id)
        worker_url = await _choose_worker(router, pack_id)
        question = request.question if request.strategy == "echo_question" else "warm"
        prompt = runtime_compiler.completion_token_ids(pack, question)
        salt = _cache_salt(settings, tenant_id)
        started = time.perf_counter()

        async def perform() -> tuple[CompletionResult, str]:
            return await _complete_with_failover(
                router=router,
                vllm=vllm,
                worker_url=worker_url,
                pack_id=pack_id,
                model=pack.model_id,
                prompt=prompt,
                max_tokens=1,
                temperature=0.0,
                top_p=1.0,
                stop=None,
                extra_body=None,
                cache_salt=salt,
            )

        try:
            _, actual_worker = await warmups.run((tenant_id, pack_id, worker_url), perform)
        except VLLMError as exc:
            WARMUPS_TOTAL.labels(worker_id=_worker_id(worker_url), outcome="failed").inc()
            raise HTTPException(status_code=502, detail="vLLM warmup failed") from exc
        except NoHealthyWorkers as exc:
            WARMUPS_TOTAL.labels(worker_id=_worker_id(worker_url), outcome="unavailable").inc()
            raise HTTPException(status_code=503, detail="no healthy vLLM workers") from exc
        latency_ms = (time.perf_counter() - started) * 1000
        WARMUPS_TOTAL.labels(worker_id=_worker_id(actual_worker), outcome="ok").inc()
        WARMUP_LATENCY.observe(latency_ms / 1000)
        return WarmResponse(
            pack_id=pack_id,
            worker_id=_worker_id(actual_worker),
            warmed=True,
            latency_ms=latency_ms,
        )

    @app.post("/v1/completions", response_model=None)
    async def completions(
        request: CompletionRequest,
        tenant_id: str = TENANT_HEADER,
    ) -> Response:
        if request.max_tokens < 1 or request.max_tokens > settings.max_output_tokens:
            raise HTTPException(
                status_code=422,
                detail=f"max_tokens must be between 1 and {settings.max_output_tokens}",
            )
        if not admission.acquire():
            raise HTTPException(status_code=429, detail="server is at its in-flight limit")

        stream_handed_off = False
        metric_worker = "unassigned"
        outcome = "failed"
        try:
            runtime_compiler = _require_compiler(app)
            pack = await _load_pack(store, cache, tenant_id, request.pack_id)
            worker_url = await router.choose(request.pack_id)
            metric_worker = _worker_id(worker_url)
            prompt = runtime_compiler.completion_token_ids(pack, request.prompt)
            salt = _cache_salt(settings, tenant_id)
            headers = {
                "X-Flume-Pack-Id": pack.pack_id,
                "X-Flume-Worker-Id": metric_worker,
                "X-Flume-Prompt-Tokens": str(len(prompt)),
            }
            if request.stream:
                stream, actual_worker = await _open_stream_with_failover(
                    router=router,
                    vllm=vllm,
                    worker_url=worker_url,
                    pack_id=pack.pack_id,
                    model=pack.model_id,
                    prompt=prompt,
                    request=request,
                    cache_salt=salt,
                )
                metric_worker = _worker_id(actual_worker)
                headers["X-Flume-Worker-Id"] = metric_worker
                stream_handed_off = True
                return StreamingResponse(
                    _stream_with_release(stream, admission, actual_worker),
                    media_type="text/event-stream",
                    headers=headers,
                )

            started = time.perf_counter()
            result, actual_worker = await _complete_with_failover(
                router=router,
                vllm=vllm,
                worker_url=worker_url,
                pack_id=pack.pack_id,
                model=pack.model_id,
                prompt=prompt,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                stop=request.stop,
                extra_body=request.extra_body,
                cache_salt=salt,
            )
            ASK_LATENCY.observe(time.perf_counter() - started)
            metric_worker = _worker_id(actual_worker)
            outcome = "ok"
            headers["X-Flume-Worker-Id"] = metric_worker
            body = {
                "id": f"cmpl-{uuid.uuid4().hex}",
                "object": "text_completion",
                "created": int(time.time()),
                "model": pack.model_id,
                "choices": [
                    {
                        "text": result.text,
                        "index": 0,
                        "logprobs": None,
                        "finish_reason": result.finish_reason,
                    }
                ],
                "usage": {
                    "prompt_tokens": result.prompt_tokens or len(prompt),
                    "completion_tokens": result.output_tokens or 0,
                    "total_tokens": (result.prompt_tokens or len(prompt))
                    + (result.output_tokens or 0),
                },
            }
            return JSONResponse(body, headers=headers)
        except NoHealthyWorkers as exc:
            outcome = "unavailable"
            raise HTTPException(status_code=503, detail="no healthy vLLM workers") from exc
        except ValueError as exc:
            outcome = "rejected"
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except VLLMError as exc:
            outcome = "upstream_error"
            raise HTTPException(status_code=502, detail="vLLM request failed") from exc
        finally:
            if not stream_handed_off:
                ASKS_TOTAL.labels(
                    worker_id=metric_worker,
                    stream="false",
                    outcome=outcome,
                ).inc()
                admission.release()

    @app.get("/v1/stats", response_model=StatsResponse)
    async def stats(tenant_id: str = TENANT_HEADER) -> StatsResponse:
        pack_count, route_count = await store.counts(tenant_id)
        route_counts = router.stats()
        health = router.health()
        workers = []
        for worker in settings.vllm_workers:
            counts = route_counts.get(
                worker,
                {"assigned_packs": 0, "affinity_hits": 0, "affinity_misses": 0},
            )
            workers.append(
                WorkerStats(
                    worker_id=_worker_id(worker),
                    healthy=health.get(worker),
                    assigned_packs=counts["assigned_packs"],
                    affinity_hits=counts["affinity_hits"],
                    affinity_misses=counts["affinity_misses"],
                )
            )
        return StatsResponse(packs=pack_count, routes=route_count, workers=workers)

    @app.get("/metrics")
    async def metrics() -> Response:
        if not settings.metrics_enabled:
            raise HTTPException(status_code=404, detail="metrics are disabled")
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    return app


def _require_compiler(app: FastAPI) -> ContextPackCompiler:
    compiler = app.state.compiler
    if compiler is None:
        raise HTTPException(status_code=503, detail="tokenizer runtime is unavailable")
    return compiler


async def _load_pack(
    store: FlumeStore,
    cache: ByteBoundedPackCache,
    tenant_id: str,
    pack_id: str,
) -> ContextPack:
    pack = cache.get(tenant_id, pack_id)
    if pack is not None:
        PACK_CACHE.labels(result="hit").inc()
        return pack
    PACK_CACHE.labels(result="miss").inc()
    pack = await store.get_pack(tenant_id, pack_id)
    if pack is None:
        raise HTTPException(status_code=404, detail="context pack not found")
    cache.put(pack)
    return pack


async def _choose_worker(router: PackRouter, pack_id: str) -> str:
    try:
        return await router.choose(pack_id)
    except NoHealthyWorkers as exc:
        raise HTTPException(status_code=503, detail="no healthy vLLM workers") from exc


def _cache_salt(settings: Settings, tenant_id: str) -> str:
    if settings.cache_salt_secret == INSECURE_CACHE_SALT_SECRET:
        raise HTTPException(status_code=503, detail="cache salt secret is not configured")
    return hmac.new(
        settings.cache_salt_secret.encode("utf-8"),
        tenant_id.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()


async def _complete_with_failover(
    *,
    router: PackRouter,
    vllm: VLLMClient,
    worker_url: str,
    pack_id: str,
    model: str,
    prompt: list[int],
    max_tokens: int,
    temperature: float,
    top_p: float,
    stop: str | list[str] | None,
    extra_body: dict[str, Any] | None,
    cache_salt: str,
) -> tuple[CompletionResult, str]:
    active_worker = worker_url
    router.acquire(active_worker)
    try:
        try:
            result = await vllm.complete(
                worker_url=active_worker,
                model=model,
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stop=stop,
                extra_body=extra_body,
                cache_salt=cache_salt,
            )
            return result, active_worker
        except VLLMConnectionError:
            ROUTER_FAILOVERS.labels(operation="completion").inc()
            router.mark_unhealthy(active_worker)
            router.release(active_worker)
            replacement = await router.choose(pack_id, exclude={active_worker})
            active_worker = replacement
            router.acquire(active_worker)
            result = await vllm.complete(
                worker_url=active_worker,
                model=model,
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stop=stop,
                extra_body=extra_body,
                cache_salt=cache_salt,
            )
            return result, active_worker
    finally:
        router.release(active_worker)


async def _open_stream_with_failover(
    *,
    router: PackRouter,
    vllm: VLLMClient,
    worker_url: str,
    pack_id: str,
    model: str,
    prompt: list[int],
    request: CompletionRequest,
    cache_salt: str,
) -> tuple[VLLMStream, str]:
    async def open_stream(target_worker: str) -> VLLMStream:
        return await vllm.open_stream_completion(
            worker_url=target_worker,
            model=model,
            prompt=prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            stop=request.stop,
            extra_body=request.extra_body,
            cache_salt=cache_salt,
            on_first_token=lambda value: TTFT.observe(value / 1000),
            on_done=lambda value: _finish_worker_stream(router, target_worker, value),
        )

    active_worker = worker_url
    router.acquire(active_worker)
    try:
        stream = await open_stream(active_worker)
        return stream, active_worker
    except VLLMConnectionError:
        ROUTER_FAILOVERS.labels(operation="stream").inc()
        router.mark_unhealthy(active_worker)
        router.release(active_worker)
        replacement = await router.choose(pack_id, exclude={active_worker})
        active_worker = replacement
        router.acquire(active_worker)
        try:
            stream = await open_stream(active_worker)
            return stream, active_worker
        except Exception:
            router.release(active_worker)
            raise
    except Exception:
        router.release(active_worker)
        raise


def _finish_worker_stream(router: PackRouter, worker_url: str, latency_ms: float) -> None:
    ASK_LATENCY.observe(latency_ms / 1000)
    router.release(worker_url)


async def _stream_with_release(
    stream: VLLMStream,
    admission: AdmissionController,
    worker_url: str,
) -> AsyncIterator[bytes]:
    outcome = "ok"
    try:
        async for chunk in stream:
            yield chunk
    except asyncio.CancelledError:
        outcome = "cancelled"
        raise
    except Exception:
        outcome = "failed"
        raise
    finally:
        ASKS_TOTAL.labels(
            worker_id=_worker_id(worker_url),
            stream="true",
            outcome=outcome,
        ).inc()
        admission.release()


def _worker_id(worker_url: str) -> str:
    return hashlib.sha256(worker_url.encode()).hexdigest()[:12]


def _record_request(
    request: Request,
    status_code: int,
    started: float,
    request_id: str,
) -> None:
    route = request.scope.get("route")
    route_path = getattr(route, "path", "unmatched")
    route_label = route_path if route_path.startswith("/") else "unmatched"
    duration = time.perf_counter() - started
    REQUESTS_TOTAL.labels(
        route=route_label,
        method=request.method,
        status_class=f"{status_code // 100}xx",
    ).inc()
    REQUEST_LATENCY.labels(route=route_label).observe(duration)
    LOGGER.info(
        json.dumps(
            {
                "event": "request_completed",
                "request_id": request_id,
                "method": request.method,
                "route": route_label,
                "status": status_code,
                "duration_ms": round(duration * 1000, 3),
            },
            separators=(",", ":"),
            sort_keys=True,
        )
    )

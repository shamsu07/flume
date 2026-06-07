from __future__ import annotations

import time
import uuid
from collections.abc import AsyncIterator

from fastapi import FastAPI, HTTPException
from fastapi.responses import Response, StreamingResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from flume.bench import BenchmarkRunner
from flume.compiler import ContextPackCompiler
from flume.config import Settings, get_settings
from flume.metrics import (
    ASK_LATENCY,
    ASKS_TOTAL,
    PACKS_CREATED,
    TTFT,
    WARMUP_LATENCY,
    WARMUPS_TOTAL,
)
from flume.models import (
    AskRequest,
    AskResponse,
    BenchmarkRun,
    BenchmarkRunRequest,
    ContextPack,
    PackCreateRequest,
    StatsResponse,
    WarmRequest,
    WarmResponse,
    WorkerStats,
)
from flume.router import PackRouter
from flume.store import FlumeStore
from flume.vllm import VLLMClient


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    store = FlumeStore(settings.database_url)
    store.init_schema()
    compiler = ContextPackCompiler(allow_remote_tokenizer=settings.allow_remote_tokenizer)
    vllm = VLLMClient(timeout_seconds=settings.request_timeout_seconds)
    router = PackRouter(settings.vllm_workers, store, health_checker=vllm.health)
    bench = BenchmarkRunner(
        store=store,
        compiler=compiler,
        router=router,
        vllm=vllm,
        settings=settings,
    )

    app = FastAPI(
        title="Flume",
        version="0.1.0",
        description="RAG cache compiler and vLLM serving proxy.",
    )
    app.state.settings = settings
    app.state.store = store
    app.state.compiler = compiler
    app.state.vllm = vllm
    app.state.router = router

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/packs", response_model=ContextPack)
    async def create_pack(request: PackCreateRequest) -> ContextPack:
        pack = compiler.compile(request)
        if settings.max_pack_tokens and pack.token_count > settings.max_pack_tokens:
            raise HTTPException(
                status_code=422,
                detail=f"pack has {pack.token_count} tokens, max is {settings.max_pack_tokens}",
            )
        store.save_pack(pack)
        PACKS_CREATED.labels(tenant_id=pack.tenant_id).inc()
        return pack

    @app.get("/packs", response_model=list[ContextPack])
    async def list_packs(tenant_id: str | None = None) -> list[ContextPack]:
        return store.list_packs(tenant_id=tenant_id)

    @app.get("/packs/{pack_id}", response_model=ContextPack)
    async def get_pack(pack_id: str) -> ContextPack:
        pack = store.get_pack(pack_id)
        if pack is None:
            raise HTTPException(status_code=404, detail="context pack not found")
        return pack

    @app.post("/packs/{pack_id}/warm", response_model=WarmResponse)
    async def warm_pack(pack_id: str, request: WarmRequest | None = None) -> WarmResponse:
        request = request or WarmRequest(pack_id=pack_id)
        pack = _load_pack(store, pack_id)
        worker_url = await router.choose(pack_id)
        warm_question = request.question if request.strategy == "echo_question" else "warm"
        prompt = _build_prompt(pack, warm_question)
        started = time.perf_counter()
        try:
            await vllm.complete(
                worker_url=worker_url,
                model=pack.model_id,
                prompt=prompt,
                max_tokens=1,
                temperature=0.0,
                top_p=1.0,
            )
        except Exception as exc:
            WARMUPS_TOTAL.labels(worker_url=worker_url, status="failed").inc()
            raise HTTPException(status_code=502, detail=f"vLLM warmup failed: {exc}") from exc
        latency_ms = (time.perf_counter() - started) * 1000
        WARMUPS_TOTAL.labels(worker_url=worker_url, status="ok").inc()
        WARMUP_LATENCY.observe(latency_ms / 1000)
        return WarmResponse(
            pack_id=pack_id,
            worker_url=worker_url,
            warmed=True,
            latency_ms=latency_ms,
        )

    @app.post("/ask", response_model=None)
    async def ask(request: AskRequest) -> AskResponse | StreamingResponse:
        pack = _load_pack(store, request.pack_id)
        worker_url = await router.choose(request.pack_id)
        prompt = _build_prompt(pack, request.question)

        if request.stream:
            return StreamingResponse(
                _stream_answer(request, pack, worker_url, prompt, vllm),
                media_type="text/event-stream",
                headers={
                    "X-Flume-Pack-Id": pack.pack_id,
                    "X-Flume-Worker-Url": worker_url,
                    "X-Flume-Prompt-Tokens": str(pack.token_count),
                },
            )

        started = time.perf_counter()
        try:
            result = await vllm.complete(
                worker_url=worker_url,
                model=pack.model_id,
                prompt=prompt,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                stop=request.stop,
                extra_body=request.extra_body,
            )
        except Exception as exc:
            raise HTTPException(status_code=502, detail=f"vLLM request failed: {exc}") from exc

        latency_ms = (time.perf_counter() - started) * 1000
        ASK_LATENCY.observe(latency_ms / 1000)
        ASKS_TOTAL.labels(worker_url=worker_url, stream="false").inc()
        return AskResponse(
            pack_id=pack.pack_id,
            worker_url=worker_url,
            text=result.text,
            ttft_ms=result.ttft_ms,
            latency_ms=result.latency_ms,
            prompt_tokens=result.prompt_tokens or pack.token_count,
            output_tokens=result.output_tokens,
            finish_reason=result.finish_reason,
        )

    @app.get("/stats", response_model=StatsResponse)
    async def stats() -> StatsResponse:
        pack_count, route_count = store.counts()
        route_counts = store.route_counts_by_worker()
        health = await router.health()
        workers = []
        for worker in settings.vllm_workers:
            counts = route_counts.get(
                worker,
                {"assigned_packs": 0, "affinity_hits": 0, "affinity_misses": 0},
            )
            workers.append(
                WorkerStats(
                    worker_url=worker,
                    healthy=health.get(worker),
                    assigned_packs=counts["assigned_packs"],
                    affinity_hits=counts["affinity_hits"],
                    affinity_misses=counts["affinity_misses"],
                )
            )
        return StatsResponse(packs=pack_count, routes=route_count, workers=workers)

    @app.post("/bench/run", response_model=BenchmarkRun)
    async def run_benchmark(request: BenchmarkRunRequest) -> BenchmarkRun:
        run = BenchmarkRun(
            run_id=f"bench_{uuid.uuid4().hex[:16]}",
            request=request,
            status="running",
        )
        store.save_benchmark(run)
        run = await bench.run(run)
        store.save_benchmark(run)
        return run

    @app.get("/bench/runs", response_model=list[BenchmarkRun])
    async def list_benchmarks() -> list[BenchmarkRun]:
        return store.list_benchmarks()

    @app.get("/bench/runs/{run_id}", response_model=BenchmarkRun)
    async def get_benchmark(run_id: str) -> BenchmarkRun:
        run = store.get_benchmark(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail="benchmark run not found")
        return run

    @app.get("/metrics")
    async def metrics() -> Response:
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    return app


def _load_pack(store: FlumeStore, pack_id: str) -> ContextPack:
    pack = store.get_pack(pack_id)
    if pack is None:
        raise HTTPException(status_code=404, detail="context pack not found")
    return pack


def _build_prompt(pack: ContextPack, question: str) -> str:
    return f"{pack.compiled_prefix}{question.strip()}\n\nAnswer:\n"


async def _stream_answer(
    request: AskRequest,
    pack: ContextPack,
    worker_url: str,
    prompt: str,
    vllm: VLLMClient,
) -> AsyncIterator[bytes]:
    started = time.perf_counter()

    def on_first_token(ttft_ms: float) -> None:
        TTFT.observe(ttft_ms / 1000)

    def on_done(latency_ms: float) -> None:
        ASK_LATENCY.observe(latency_ms / 1000)

    try:
        async for chunk in vllm.stream_completion(
            worker_url=worker_url,
            model=pack.model_id,
            prompt=prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            stop=request.stop,
            extra_body=request.extra_body,
            on_first_token=on_first_token,
            on_done=on_done,
        ):
            yield chunk
    finally:
        ASKS_TOTAL.labels(worker_url=worker_url, stream="true").inc()
        ASK_LATENCY.observe(time.perf_counter() - started)

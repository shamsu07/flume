#!/usr/bin/env python3
"""Measure Flume's local proxy overhead against a counted mock vLLM upstream.

The default mode starts both the mock upstream and the Flume application in this
checkout. Pass ``--proxy-url`` to benchmark an already running Flume instance.
Results are written as JSON plus a short Markdown companion.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import hmac
import json
import os
import platform
import socket
import subprocess
import tempfile
import time
from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import Response
from sqlalchemy import event

from flume.api import create_app
from flume.compiler import ContextPackCompiler, DeterministicByteTokenizer
from flume.config import Settings
from flume.models import ContextPack, PackCreateRequest, PackRegistrationRequest

TENANT_ID = "benchmark"
MODEL_ID = "mock-model"
TOKENIZER_ID = "mock-tokenizer"
TOKENIZER_REVISION = "byte-v1"
QUESTION = "benchmark question"
CACHE_SALT_SECRET = "proxy-overhead-benchmark-secret-00000000000000000000000000000000"


@dataclass(slots=True)
class SampleSummary:
    requests: int
    errors: int
    duration_seconds: float
    throughput_rps: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    minimum_ms: float
    maximum_ms: float


@dataclass(slots=True)
class UpstreamCounts:
    health: int = 0
    completions: int = 0


@dataclass(slots=True)
class RuntimeCounts:
    database_statements: int = 0
    database_writes: int = 0

    def reset(self) -> None:
        self.database_statements = 0
        self.database_writes = 0


def percentile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def summarize(latencies_ms: list[float], errors: int, duration_seconds: float) -> SampleSummary:
    completed = len(latencies_ms) + errors
    return SampleSummary(
        requests=completed,
        errors=errors,
        duration_seconds=duration_seconds,
        throughput_rps=completed / duration_seconds if duration_seconds else 0.0,
        p50_ms=percentile(latencies_ms, 0.50),
        p95_ms=percentile(latencies_ms, 0.95),
        p99_ms=percentile(latencies_ms, 0.99),
        minimum_ms=min(latencies_ms, default=0.0),
        maximum_ms=max(latencies_ms, default=0.0),
    )


def mock_vllm_app(counts: UpstreamCounts) -> FastAPI:
    app = FastAPI()

    @app.get("/health")
    async def health() -> dict[str, str]:
        counts.health += 1
        return {"status": "ok"}

    @app.post("/v1/completions")
    async def completions(request: Request) -> dict[str, Any]:
        body = await request.json()
        counts.completions += 1
        prompt = body.get("prompt", "")
        prompt_tokens = len(prompt) if isinstance(prompt, list) else len(str(prompt).split())
        return {
            "id": "cmpl_mock",
            "object": "text_completion",
            "created": 0,
            "model": body.get("model", "mock-model"),
            "choices": [{"index": 0, "text": "ok", "finish_reason": "length"}],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": 1,
                "total_tokens": prompt_tokens + 1,
            },
        }

    return app


def reference_proxy_app(upstream_url: str) -> FastAPI:
    """Minimal pooled forward proxy used as the fair throughput comparator."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.client = httpx.AsyncClient(
            timeout=30.0,
            limits=httpx.Limits(max_connections=512, max_keepalive_connections=128),
        )
        try:
            yield
        finally:
            await app.state.client.aclose()

    app = FastAPI(lifespan=lifespan)

    @app.get("/livez")
    async def livez() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/v1/completions")
    async def completions(request: Request) -> Response:
        body = await request.body()
        upstream = await app.state.client.post(
            f"{upstream_url}/v1/completions",
            content=body,
            headers={"content-type": request.headers.get("content-type", "application/json")},
        )
        return Response(
            content=upstream.content,
            status_code=upstream.status_code,
            media_type=upstream.headers.get("content-type", "application/json"),
        )

    return app


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


async def wait_until_ready(url: str, timeout_seconds: float = 10.0) -> None:
    deadline = time.monotonic() + timeout_seconds
    async with httpx.AsyncClient(timeout=0.5) as client:
        while time.monotonic() < deadline:
            try:
                response = await client.get(url)
                if response.status_code < 500:
                    return
            except httpx.HTTPError:
                pass
            await asyncio.sleep(0.02)
    raise RuntimeError(f"server did not become ready: {url}")


async def serve(
    app: FastAPI,
    port: int,
    *,
    ready_path: str = "/health",
) -> tuple[uvicorn.Server, asyncio.Task[None]]:
    config = uvicorn.Config(
        app,
        host="127.0.0.1",
        port=port,
        log_level="error",
        access_log=False,
    )
    server = uvicorn.Server(config)
    task = asyncio.create_task(server.serve())
    await wait_until_ready(f"http://127.0.0.1:{port}{ready_path}")
    return server, task


async def stop_server(server: uvicorn.Server, task: asyncio.Task[None]) -> None:
    server.should_exit = True
    await task


async def measure(
    request: Callable[[], Awaitable[httpx.Response]],
    *,
    requests: int,
    concurrency: int,
) -> SampleSummary:
    semaphore = asyncio.Semaphore(concurrency)
    latencies: list[float] = []
    errors = 0

    async def one() -> None:
        nonlocal errors
        async with semaphore:
            started = time.perf_counter()
            try:
                response = await request()
                response.raise_for_status()
            except (httpx.HTTPError, RuntimeError):
                errors += 1
            else:
                latencies.append((time.perf_counter() - started) * 1000)

    started = time.perf_counter()
    await asyncio.gather(*(one() for _ in range(requests)))
    return summarize(latencies, errors, time.perf_counter() - started)


def pack_registration() -> PackRegistrationRequest:
    return PackRegistrationRequest(
        template_id="proxy-overhead-v1",
        chunks=[
            {
                "doc_id": "fixture",
                "chunk_id": "0",
                "version": "1",
                "text": "alpha beta",
            }
        ],
    )


def compile_pack(
    registration: PackRegistrationRequest,
    compiler: ContextPackCompiler,
) -> ContextPack:
    return compiler.compile(
        PackCreateRequest(
            **registration.model_dump(),
            tenant_id=TENANT_ID,
            model_id=MODEL_ID,
            tokenizer_id=TOKENIZER_ID,
        )
    )


async def register_pack(
    client: httpx.AsyncClient,
    proxy_url: str,
    compiler: ContextPackCompiler,
) -> ContextPack:
    registration = pack_registration()
    expected_pack = compile_pack(registration, compiler)
    response = await client.post(
        f"{proxy_url}/v1/packs",
        json=registration.model_dump(mode="json"),
        headers={"X-Flume-Tenant": TENANT_ID},
    )
    response.raise_for_status()
    payload = response.json()
    if payload["pack_id"] != expected_pack.pack_id:
        raise RuntimeError("registered pack identity differs from deterministic local fixture")
    return expected_pack


async def proxy_request(
    client: httpx.AsyncClient,
    proxy_url: str,
    pack_id: str,
) -> httpx.Response:
    return await client.post(
        f"{proxy_url}/v1/completions",
        headers={"X-Flume-Tenant": TENANT_ID},
        json={
            "pack_id": pack_id,
            "prompt": QUESTION,
            "max_tokens": 1,
            "temperature": 0.0,
            "stream": False,
        },
    )


def cache_salt() -> str:
    return hmac.new(
        CACHE_SALT_SECRET.encode(),
        TENANT_ID.encode(),
        hashlib.sha256,
    ).hexdigest()


def instrument_database(app: FastAPI, counts: RuntimeCounts) -> None:
    @event.listens_for(app.state.store.engine.sync_engine, "before_cursor_execute")
    def count_statement(
        _connection: Any,
        _cursor: Any,
        statement: str,
        _parameters: Any,
        _context: Any,
        _executemany: bool,
    ) -> None:
        counts.database_statements += 1
        operation = statement.lstrip().split(maxsplit=1)[0].upper()
        if operation in {"DELETE", "INSERT", "REPLACE", "UPDATE"}:
            counts.database_writes += 1


def git_metadata() -> dict[str, Any]:
    def git(*args: str) -> str:
        return subprocess.run(
            ["git", *args],
            check=False,
            capture_output=True,
            text=True,
        ).stdout.strip()

    return {
        "commit": git("rev-parse", "HEAD") or "unknown",
        "dirty": bool(git("status", "--porcelain")),
        "branch": git("branch", "--show-current") or "detached",
    }


def environment_metadata() -> dict[str, Any]:
    return {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "hostname": platform.node(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or "unknown",
        "python": platform.python_version(),
        "logical_cpu_count": os.cpu_count(),
        "git": git_metadata(),
    }


def markdown_report(result: dict[str, Any]) -> str:
    direct = result["measurements"]["direct_upstream"]
    reference = result["measurements"]["reference_proxy"]
    proxy = result["measurements"]["flume"]
    reference_comparison = result["comparison"]["flume_vs_reference_proxy"]
    baseline_comparison = result["comparison"]["flume_vs_direct_upstream"]
    baseline = result["baseline_comparison"]
    instrumentation = result["instrumentation"]
    gates = result["gates"]
    metadata = result["environment"]
    lines = [
        "# Flume local proxy overhead",
        "",
        "This run uses a deterministic local mock vLLM and the Flume application from the",
        "recorded checkout. It is a proxy-path benchmark, not a GPU inference result.",
        "",
        f"- Commit: `{metadata['git']['commit']}` (dirty during capture: "
        f"`{str(metadata['git']['dirty']).lower()}`)",
        f"- Host: `{metadata['hostname']}` / `{metadata['machine']}`",
        f"- Python: `{metadata['python']}`",
        (
            f"- Requests: `{proxy['requests']}` at concurrency "
            f"`{result['configuration']['concurrency']}`"
        ),
        f"- Gate: **{'PASS' if gates['passed'] else 'FAIL'}**",
        "",
        "| Path | Throughput (req/s) | p50 (ms) | p95 (ms) | p99 (ms) | Errors |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
        f"| Direct mock vLLM (diagnostic) | {direct['throughput_rps']:.2f} | "
        f"{direct['p50_ms']:.3f} | "
        f"{direct['p95_ms']:.3f} | {direct['p99_ms']:.3f} | {direct['errors']} |",
        f"| Minimal pooled proxy (gate reference) | {reference['throughput_rps']:.2f} | "
        f"{reference['p50_ms']:.3f} | {reference['p95_ms']:.3f} | "
        f"{reference['p99_ms']:.3f} | {reference['errors']} |",
        f"| Through Flume | {proxy['throughput_rps']:.2f} | {proxy['p50_ms']:.3f} | "
        f"{proxy['p95_ms']:.3f} | {proxy['p99_ms']:.3f} | {proxy['errors']} |",
        "",
        f"Incremental p99 over direct proxying: "
        f"**{reference_comparison['incremental_p99_ms']:.3f} ms**. "
        f"Flume/reference throughput ratio: "
        f"**{reference_comparison['throughput_ratio']:.3f}**.",
        "",
        "## Committed-baseline comparison",
        "",
        f"Baseline source: `{baseline['source']}` at commit `{baseline['baseline_commit']}`.",
        " The legacy baseline compared Flume directly with the upstream, so this section",
        " uses the current direct-upstream diagnostic for compatibility; release gates use",
        " the fair two-hop reference proxy above.",
        "",
        "| Metric | Committed baseline | Current | Change |",
        "| --- | ---: | ---: | ---: |",
        (
            f"| Incremental p99 overhead | {baseline['baseline_incremental_p99_ms']:.3f} ms | "
            f"{baseline_comparison['incremental_p99_ms']:.3f} ms | "
            f"{baseline['incremental_p99_reduction_percent']:.2f}% lower |"
        ),
        (
            f"| Proxy/direct throughput ratio | {baseline['baseline_throughput_ratio']:.3f} | "
            f"{baseline_comparison['throughput_ratio']:.3f} | "
            f"{baseline['throughput_ratio_multiplier']:.2f}x |"
        ),
        (
            f"| Health probes during measured window | {baseline['baseline_health_probes']} | "
            f"{instrumentation['mock_upstream']['health_probes']} | "
            f"{baseline['health_probe_reduction_percent']:.2f}% lower |"
        ),
        "",
        "## Measured hot-path instrumentation",
        "",
        f"- Mock completions: `{instrumentation['mock_upstream']['completions']}`",
        f"- Health probes: `{instrumentation['mock_upstream']['health_probes']}`",
        (
            f"- Database statements/writes: "
            f"`{instrumentation['embedded_proxy']['database_statements']}` / "
            f"`{instrumentation['embedded_proxy']['database_writes']}`"
        ),
        (
            f"- Upstream client instances observed: "
            f"`{instrumentation['embedded_proxy']['upstream_client_instances_observed']}` "
            f"(reused: "
            f"`{str(instrumentation['embedded_proxy']['upstream_client_reused']).lower()}`)"
        ),
        "",
    ]
    return "\n".join(lines)


def load_baseline(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def percentage_reduction(previous: float, current: float) -> float:
    if previous == 0:
        return 0.0
    return (previous - current) / previous * 100


async def run(args: argparse.Namespace) -> dict[str, Any]:
    counts = UpstreamCounts()
    runtime_counts = RuntimeCounts()
    compiler = ContextPackCompiler(
        DeterministicByteTokenizer(
            tokenizer_id=TOKENIZER_ID,
            revision=TOKENIZER_REVISION,
        ),
        model_id=MODEL_ID,
    )
    upstream_port = free_port()
    upstream_url = f"http://127.0.0.1:{upstream_port}"
    upstream_server, upstream_task = await serve(mock_vllm_app(counts), upstream_port)
    reference_port = free_port()
    reference_url = f"http://127.0.0.1:{reference_port}"
    reference_server, reference_task = await serve(
        reference_proxy_app(upstream_url),
        reference_port,
        ready_path="/livez",
    )

    proxy_server: uvicorn.Server | None = None
    proxy_task: asyncio.Task[None] | None = None
    temporary_database: tempfile.TemporaryDirectory[str] | None = None
    embedded_app: FastAPI | None = None
    initial_upstream_client_id: int | None = None
    try:
        proxy_url = args.proxy_url.rstrip("/") if args.proxy_url else ""
        if not proxy_url:
            proxy_port = free_port()
            proxy_url = f"http://127.0.0.1:{proxy_port}"
            temporary_database = tempfile.TemporaryDirectory(prefix="flume-proxy-bench-")
            database_path = Path(temporary_database.name) / "flume.db"
            embedded_app = create_app(
                Settings(
                    database_url=f"sqlite:///{database_path}",
                    vllm_workers=[upstream_url],
                    model_id=MODEL_ID,
                    tokenizer_id=TOKENIZER_ID,
                    tokenizer_revision=TOKENIZER_REVISION,
                    cache_salt_secret=CACHE_SALT_SECRET,
                    health_refresh_seconds=3600.0,
                    metrics_enabled=False,
                ),
                compiler=compiler,
            )
            instrument_database(embedded_app, runtime_counts)
            proxy_server, proxy_task = await serve(
                embedded_app,
                proxy_port,
                ready_path="/livez",
            )
            initial_upstream_client_id = id(embedded_app.state.vllm.client)

        limits = httpx.Limits(
            max_connections=max(args.concurrency * 2, 20),
            max_keepalive_connections=max(args.concurrency, 10),
        )
        async with httpx.AsyncClient(timeout=args.timeout, limits=limits) as client:
            pack = await register_pack(client, proxy_url, compiler)
            prompt_token_ids = compiler.completion_token_ids(pack, QUESTION)
            direct_payload = {
                "model": MODEL_ID,
                "prompt": prompt_token_ids,
                "max_tokens": 1,
                "temperature": 0.0,
                "top_p": 1.0,
                "stream": False,
                "cache_salt": cache_salt(),
            }

            for _ in range(args.warmup):
                direct_response = await client.post(
                    f"{upstream_url}/v1/completions",
                    json=direct_payload,
                )
                direct_response.raise_for_status()
                reference_response = await client.post(
                    f"{reference_url}/v1/completions",
                    json=direct_payload,
                )
                reference_response.raise_for_status()
                (await proxy_request(client, proxy_url, pack.pack_id)).raise_for_status()

            counts.health = 0
            counts.completions = 0
            runtime_counts.reset()
            direct_upstream = await measure(
                lambda: client.post(f"{upstream_url}/v1/completions", json=direct_payload),
                requests=args.requests,
                concurrency=args.concurrency,
            )
            reference_proxy = await measure(
                lambda: client.post(f"{reference_url}/v1/completions", json=direct_payload),
                requests=args.requests,
                concurrency=args.concurrency,
            )
            flume = await measure(
                lambda: proxy_request(client, proxy_url, pack.pack_id),
                requests=args.requests,
                concurrency=args.concurrency,
            )

        direct_data = asdict(direct_upstream)
        reference_data = asdict(reference_proxy)
        proxy_data = asdict(flume)
        final_upstream_client_id = (
            id(embedded_app.state.vllm.client) if embedded_app is not None else None
        )
        observed_client_instances = (
            1
            if initial_upstream_client_id is not None
            and initial_upstream_client_id == final_upstream_client_id
            else None
        )
        reference_comparison = {
            "incremental_p50_ms": flume.p50_ms - reference_proxy.p50_ms,
            "incremental_p95_ms": flume.p95_ms - reference_proxy.p95_ms,
            "incremental_p99_ms": flume.p99_ms - reference_proxy.p99_ms,
            "throughput_ratio": (
                flume.throughput_rps / reference_proxy.throughput_rps
                if reference_proxy.throughput_rps
                else 0.0
            ),
        }
        direct_comparison = {
            "incremental_p50_ms": flume.p50_ms - direct_upstream.p50_ms,
            "incremental_p95_ms": flume.p95_ms - direct_upstream.p95_ms,
            "incremental_p99_ms": flume.p99_ms - direct_upstream.p99_ms,
            "throughput_ratio": (
                flume.throughput_rps / direct_upstream.throughput_rps
                if direct_upstream.throughput_rps
                else 0.0
            ),
        }
        baseline = load_baseline(args.baseline)
        baseline_comparison = baseline["comparison"]
        baseline_health = int(baseline["mock_upstream_counts"]["health"])
        baseline_data = {
            "source": str(args.baseline),
            "baseline_commit": baseline["environment"]["git"]["commit"],
            "baseline_incremental_p99_ms": baseline_comparison["incremental_p99_ms"],
            "baseline_throughput_ratio": baseline_comparison["throughput_ratio"],
            "baseline_health_probes": baseline_health,
            "incremental_p99_reduction_percent": percentage_reduction(
                baseline_comparison["incremental_p99_ms"],
                direct_comparison["incremental_p99_ms"],
            ),
            "throughput_ratio_multiplier": (
                direct_comparison["throughput_ratio"] / baseline_comparison["throughput_ratio"]
                if baseline_comparison["throughput_ratio"]
                else 0.0
            ),
            "health_probe_reduction_percent": percentage_reduction(
                float(baseline_health),
                float(counts.health),
            ),
        }
        gate_checks = {
            "concurrency_is_32": args.concurrency == 32,
            "at_least_1000_requests_per_path": args.requests >= 1000,
            "zero_request_errors": (
                direct_upstream.errors == 0 and reference_proxy.errors == 0 and flume.errors == 0
            ),
            "zero_request_path_health_probes": counts.health == 0,
            "zero_request_path_database_writes": (
                runtime_counts.database_writes == 0 if embedded_app is not None else None
            ),
            "single_reused_upstream_client": (
                observed_client_instances == 1 if embedded_app is not None else None
            ),
            "incremental_p99_at_most_10_ms": reference_comparison["incremental_p99_ms"] <= 10.0,
            "throughput_ratio_at_least_0_9": reference_comparison["throughput_ratio"] >= 0.9,
        }
        applicable_checks = [value for value in gate_checks.values() if value is not None]
        return {
            "schema_version": 3,
            "benchmark": "local_proxy_overhead",
            "environment": environment_metadata(),
            "configuration": {
                "requests_per_path": args.requests,
                "concurrency": args.concurrency,
                "warmup_requests_per_path": args.warmup,
                "timeout_seconds": args.timeout,
                "embedded_proxy": args.proxy_url is None,
                "mock_upstream": True,
                "gate_reference": "minimal_pooled_two_hop_proxy",
                "proxy_api_mode": "v1",
                "tenant_header": {"X-Flume-Tenant": TENANT_ID},
                "model_id": MODEL_ID,
                "tokenizer_id": TOKENIZER_ID,
                "tokenizer_revision": TOKENIZER_REVISION,
                "prompt_tokens": len(prompt_token_ids),
            },
            "measurements": {
                "direct_upstream": direct_data,
                "reference_proxy": reference_data,
                "flume": proxy_data,
            },
            "comparison": {
                "flume_vs_reference_proxy": reference_comparison,
                "flume_vs_direct_upstream": direct_comparison,
            },
            "baseline_comparison": baseline_data,
            "instrumentation": {
                "mock_upstream": {
                    "health_probes": counts.health,
                    "completions": counts.completions,
                },
                "embedded_proxy": {
                    "database_statements": (
                        runtime_counts.database_statements if embedded_app is not None else None
                    ),
                    "database_writes": (
                        runtime_counts.database_writes if embedded_app is not None else None
                    ),
                    "upstream_client_instances_observed": observed_client_instances,
                    "upstream_client_reused": (
                        initial_upstream_client_id == final_upstream_client_id
                        if embedded_app is not None
                        else None
                    ),
                },
            },
            "mock_upstream_counts": asdict(counts),
            "gates": {
                "passed": all(applicable_checks),
                "checks": gate_checks,
            },
        }
    finally:
        if proxy_server is not None and proxy_task is not None:
            await stop_server(proxy_server, proxy_task)
        await stop_server(reference_server, reference_task)
        await stop_server(upstream_server, upstream_task)
        if temporary_database is not None:
            temporary_database.cleanup()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requests", type=int, default=1000)
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument(
        "--proxy-url",
        help="Use an existing Flume server instead of an embedded one.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/results/proxy-overhead.json"),
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        default=Path("benchmarks/results/proxy-overhead-baseline.json"),
        help="Committed baseline JSON used for the comparison report.",
    )
    args = parser.parse_args()
    if args.requests < 1 or args.concurrency < 1 or args.warmup < 0:
        parser.error("requests/concurrency must be positive and warmup must be non-negative")
    if not args.baseline.is_file():
        parser.error(f"baseline file does not exist: {args.baseline}")
    return args


def main() -> None:
    args = parse_args()
    result = asyncio.run(run(args))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    markdown_path = args.output.with_suffix(".md")
    markdown_path.write_text(markdown_report(result), encoding="utf-8")
    print(json.dumps(result["comparison"], indent=2))
    print(f"wrote {args.output} and {markdown_path}")


if __name__ == "__main__":
    main()

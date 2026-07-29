#!/usr/bin/env python3
"""Measure Flume's local proxy overhead against a counted mock vLLM upstream.

The default mode starts both the mock upstream and the Flume application in this
checkout. Pass ``--proxy-url`` to benchmark an already running Flume instance.
Results are written as JSON plus a short Markdown companion.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import platform
import socket
import subprocess
import tempfile
import time
from collections.abc import Awaitable, Callable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import uvicorn
from fastapi import FastAPI, Request

from flume.api import create_app
from flume.config import Settings


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


async def serve(app: FastAPI, port: int) -> tuple[uvicorn.Server, asyncio.Task[None]]:
    config = uvicorn.Config(
        app,
        host="127.0.0.1",
        port=port,
        log_level="error",
        access_log=False,
    )
    server = uvicorn.Server(config)
    task = asyncio.create_task(server.serve())
    await wait_until_ready(f"http://127.0.0.1:{port}/health")
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


async def register_pack(client: httpx.AsyncClient, proxy_url: str) -> tuple[str, str]:
    body = {
        "tenant_id": "benchmark",
        "model_id": "mock-model",
        "tokenizer_id": "mock-tokenizer",
        "template_id": "proxy-overhead-v1",
        "chunks": [{"doc_id": "fixture", "chunk_id": "0", "version": "1", "text": "alpha beta"}],
    }
    headers = {"X-Flume-Tenant": "benchmark"}
    for path in ("/v1/packs", "/packs"):
        response = await client.post(f"{proxy_url}{path}", json=body, headers=headers)
        if response.status_code != 404:
            response.raise_for_status()
            payload = response.json()
            return str(payload["pack_id"]), str(payload.get("compiled_prefix", "alpha beta"))
    raise RuntimeError("Flume exposes neither /v1/packs nor /packs")


async def proxy_request(
    client: httpx.AsyncClient,
    proxy_url: str,
    pack_id: str,
    api_mode: str,
) -> httpx.Response:
    headers = {"X-Flume-Tenant": "benchmark"}
    if api_mode == "v1":
        return await client.post(
            f"{proxy_url}/v1/completions",
            headers=headers,
            json={
                "model": "mock-model",
                "prompt": "benchmark question",
                "pack_id": pack_id,
                "max_tokens": 1,
                "temperature": 0.0,
                "stream": False,
            },
        )
    return await client.post(
        f"{proxy_url}/ask",
        headers=headers,
        json={
            "pack_id": pack_id,
            "question": "benchmark question",
            "max_tokens": 1,
            "temperature": 0.0,
            "stream": False,
        },
    )


async def detect_proxy_api(client: httpx.AsyncClient, proxy_url: str) -> str:
    response = await client.get(f"{proxy_url}/openapi.json")
    response.raise_for_status()
    paths = response.json().get("paths", {})
    return "v1" if "/v1/completions" in paths else "legacy"


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
    direct = result["measurements"]["direct"]
    proxy = result["measurements"]["proxy"]
    comparison = result["comparison"]
    metadata = result["environment"]
    lines = [
        "# Flume local proxy overhead baseline",
        "",
        "This run uses a deterministic local mock vLLM and the Flume application from the",
        "recorded commit. It is a proxy-path benchmark, not a GPU inference result.",
        "",
        f"- Commit: `{metadata['git']['commit']}` (dirty during capture: "
        f"`{str(metadata['git']['dirty']).lower()}`)",
        f"- Host: `{metadata['hostname']}` / `{metadata['machine']}`",
        f"- Python: `{metadata['python']}`",
        (
            f"- Requests: `{proxy['requests']}` at concurrency "
            f"`{result['configuration']['concurrency']}`"
        ),
        "",
        "| Path | Throughput (req/s) | p50 (ms) | p95 (ms) | p99 (ms) | Errors |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
        f"| Direct mock vLLM | {direct['throughput_rps']:.2f} | {direct['p50_ms']:.3f} | "
        f"{direct['p95_ms']:.3f} | {direct['p99_ms']:.3f} | {direct['errors']} |",
        f"| Through Flume | {proxy['throughput_rps']:.2f} | {proxy['p50_ms']:.3f} | "
        f"{proxy['p95_ms']:.3f} | {proxy['p99_ms']:.3f} | {proxy['errors']} |",
        "",
        f"Incremental p99 overhead: **{comparison['incremental_p99_ms']:.3f} ms**. "
        f"Proxy/direct throughput ratio: **{comparison['throughput_ratio']:.3f}**.",
        "",
        "The mock upstream counters make per-request health probes visible. This baseline",
        "is intentionally retained so the optimized runtime can be compared on the same host.",
        "",
    ]
    return "\n".join(lines)


async def run(args: argparse.Namespace) -> dict[str, Any]:
    counts = UpstreamCounts()
    upstream_port = free_port()
    upstream_url = f"http://127.0.0.1:{upstream_port}"
    upstream_server, upstream_task = await serve(mock_vllm_app(counts), upstream_port)

    proxy_server: uvicorn.Server | None = None
    proxy_task: asyncio.Task[None] | None = None
    temporary_database: tempfile.TemporaryDirectory[str] | None = None
    try:
        proxy_url = args.proxy_url.rstrip("/") if args.proxy_url else ""
        if not proxy_url:
            proxy_port = free_port()
            proxy_url = f"http://127.0.0.1:{proxy_port}"
            temporary_database = tempfile.TemporaryDirectory(prefix="flume-proxy-bench-")
            database_path = Path(temporary_database.name) / "flume.db"
            app = create_app(
                Settings(
                    database_url=f"sqlite:///{database_path}",
                    vllm_workers=[upstream_url],
                    model_id="mock-model",
                    tokenizer_id="mock-tokenizer",
                    metrics_enabled=False,
                )
            )
            proxy_server, proxy_task = await serve(app, proxy_port)

        limits = httpx.Limits(
            max_connections=max(args.concurrency * 2, 20),
            max_keepalive_connections=max(args.concurrency, 10),
        )
        async with httpx.AsyncClient(timeout=args.timeout, limits=limits) as client:
            pack_id, compiled_prefix = await register_pack(client, proxy_url)
            api_mode = await detect_proxy_api(client, proxy_url)
            direct_payload = {
                "model": "mock-model",
                "prompt": f"{compiled_prefix}benchmark question\n\nAnswer:\n",
                "max_tokens": 1,
                "temperature": 0.0,
                "stream": False,
            }

            for _ in range(args.warmup):
                direct_response = await client.post(
                    f"{upstream_url}/v1/completions",
                    json=direct_payload,
                )
                direct_response.raise_for_status()
                (await proxy_request(client, proxy_url, pack_id, api_mode)).raise_for_status()

            counts.health = 0
            counts.completions = 0
            direct = await measure(
                lambda: client.post(f"{upstream_url}/v1/completions", json=direct_payload),
                requests=args.requests,
                concurrency=args.concurrency,
            )
            proxy = await measure(
                lambda: proxy_request(client, proxy_url, pack_id, api_mode),
                requests=args.requests,
                concurrency=args.concurrency,
            )

        direct_data = asdict(direct)
        proxy_data = asdict(proxy)
        return {
            "schema_version": 1,
            "benchmark": "local_proxy_overhead",
            "environment": environment_metadata(),
            "configuration": {
                "requests_per_path": args.requests,
                "concurrency": args.concurrency,
                "warmup_requests_per_path": args.warmup,
                "timeout_seconds": args.timeout,
                "embedded_proxy": args.proxy_url is None,
                "mock_upstream": True,
                "proxy_api_mode": api_mode,
            },
            "measurements": {"direct": direct_data, "proxy": proxy_data},
            "comparison": {
                "incremental_p50_ms": proxy.p50_ms - direct.p50_ms,
                "incremental_p95_ms": proxy.p95_ms - direct.p95_ms,
                "incremental_p99_ms": proxy.p99_ms - direct.p99_ms,
                "throughput_ratio": (
                    proxy.throughput_rps / direct.throughput_rps if direct.throughput_rps else 0.0
                ),
            },
            "mock_upstream_counts": asdict(counts),
        }
    finally:
        if proxy_server is not None and proxy_task is not None:
            await stop_server(proxy_server, proxy_task)
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
    args = parser.parse_args()
    if args.requests < 1 or args.concurrency < 1 or args.warmup < 0:
        parser.error("requests/concurrency must be positive and warmup must be non-negative")
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

"""Process-isolated local benchmark topology for Flume."""

from __future__ import annotations

import asyncio
import socket
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import httpx

TENANT_ID = "local-benchmark"


@dataclass(frozen=True, slots=True)
class LocalSample:
    index: int
    phase: str
    status_code: int | None
    worker_id: str | None
    prompt_tokens: int | None
    output_tokens: int | None
    latency_ms: float
    error: str | None


def reserve_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def start_server(*arguments: str) -> subprocess.Popen[bytes]:
    return subprocess.Popen(
        [sys.executable, "-m", "flume.benchmark_server", *arguments],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


async def wait_for_server(
    client: httpx.AsyncClient,
    url: str,
    process: subprocess.Popen[bytes],
    *,
    timeout: float,
) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"benchmark server exited early with status {process.returncode}")
        try:
            response = await client.get(url)
            if response.status_code < 500:
                return
        except httpx.HTTPError:
            pass
        await asyncio.sleep(0.025)
    raise TimeoutError(f"timed out waiting for {url}")


def stop_servers(processes: list[subprocess.Popen[bytes]]) -> None:
    for process in reversed(processes):
        if process.poll() is None:
            process.terminate()
    for process in reversed(processes):
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)


async def register_pack(client: httpx.AsyncClient, base_url: str) -> dict[str, Any]:
    response = await client.post(
        f"{base_url}/v1/packs",
        headers={"X-Flume-Tenant": TENANT_ID},
        json={
            "template_id": "local-benchmark-v1",
            "chunks": [
                {
                    "doc_id": "benchmark-document",
                    "chunk_id": "0",
                    "version": "1",
                    "text": "Deterministic Flume benchmark context. " * 32,
                }
            ],
        },
    )
    response.raise_for_status()
    return dict(response.json())


async def complete(
    client: httpx.AsyncClient,
    base_url: str,
    pack_id: str,
    *,
    phase: str,
    index: int,
) -> LocalSample:
    started = time.perf_counter()
    status_code: int | None = None
    worker_id: str | None = None
    prompt_tokens: int | None = None
    output_tokens: int | None = None
    error: str | None = None
    try:
        response = await client.post(
            f"{base_url}/v1/completions",
            headers={"X-Flume-Tenant": TENANT_ID},
            json={
                "pack_id": pack_id,
                "prompt": f"Question {index}: summarize the policy.",
                "max_tokens": 1,
                "temperature": 0.0,
            },
        )
        status_code = response.status_code
        worker_id = response.headers.get("X-Flume-Worker-Id")
        response.raise_for_status()
        payload = response.json()
        usage = payload.get("usage") or {}
        prompt_tokens = usage.get("prompt_tokens")
        output_tokens = usage.get("completion_tokens")
    except (httpx.HTTPError, ValueError) as exc:
        error = f"{type(exc).__name__}: {exc}"
    return LocalSample(
        index=index,
        phase=phase,
        status_code=status_code,
        worker_id=worker_id,
        prompt_tokens=prompt_tokens,
        output_tokens=output_tokens,
        latency_ms=(time.perf_counter() - started) * 1000,
        error=error,
    )


async def run_local_benchmark(args: Any) -> dict[str, Any]:
    ports = [reserve_port(), reserve_port(), reserve_port()]
    worker_urls = [f"http://127.0.0.1:{port}" for port in ports[:2]]
    flume_url = f"http://127.0.0.1:{ports[2]}"
    processes: list[subprocess.Popen[bytes]] = []
    with tempfile.TemporaryDirectory(prefix="flume-benchmark-") as temp_dir:
        database_url = f"sqlite:///{Path(temp_dir) / 'flume.db'}"
        try:
            for index, port in enumerate(ports[:2]):
                process = start_server(
                    "worker",
                    "--port",
                    str(port),
                    "--worker-id",
                    f"worker-{index + 1}",
                )
                processes.append(process)
            timeout = httpx.Timeout(args.timeout)
            async with httpx.AsyncClient(timeout=timeout) as client:
                for process, worker_url in zip(processes, worker_urls, strict=True):
                    await wait_for_server(
                        client,
                        f"{worker_url}/health",
                        process,
                        timeout=args.startup_timeout,
                    )
                flume_process = start_server(
                    "flume",
                    "--port",
                    str(ports[2]),
                    "--database-url",
                    database_url,
                    *(
                        argument
                        for worker_url in worker_urls
                        for argument in ("--worker-url", worker_url)
                    ),
                )
                processes.append(flume_process)
                await wait_for_server(
                    client,
                    f"{flume_url}/readyz",
                    flume_process,
                    timeout=args.startup_timeout,
                )
                pack = await register_pack(client, flume_url)
                cold = await complete(client, flume_url, pack["pack_id"], phase="cold", index=0)
                warm_response = await client.post(
                    f"{flume_url}/v1/packs/{pack['pack_id']}/warm",
                    headers={"X-Flume-Tenant": TENANT_ID},
                )
                warm_response.raise_for_status()
                for index in range(args.warmups):
                    await complete(
                        client,
                        flume_url,
                        pack["pack_id"],
                        phase="warmup",
                        index=index,
                    )
                measured = [
                    await complete(
                        client,
                        flume_url,
                        pack["pack_id"],
                        phase="measured",
                        index=index,
                    )
                    for index in range(args.samples)
                ]
        finally:
            stop_servers(processes)
    return {
        "schema_version": 1,
        "status": "completed",
        "configuration": {
            "benchmark": "local",
            "workers": 2,
            "warmups": args.warmups,
            "samples": args.samples,
        },
        "results": {
            "pack_id": pack["pack_id"],
            "pack_tokens": pack["token_count"],
            "warm": warm_response.json(),
            "cold": asdict(cold),
            "measured": [asdict(sample) for sample in measured],
        },
    }

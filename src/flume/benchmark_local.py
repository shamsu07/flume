"""Process-isolated local benchmark topology for Flume."""

from __future__ import annotations

import asyncio
import socket
import subprocess
import sys
import tempfile
import time
from collections import Counter
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

import httpx

TENANT_ID = "local-benchmark"


class WorkloadKind(StrEnum):
    uniform = "uniform"
    hot_80_20 = "hot_80_20"
    shuffled_equivalent = "shuffled_equivalent"
    worker_delay = "worker_delay"
    worker_failure = "worker_failure"
    all_unavailable = "all_unavailable"


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


@dataclass(frozen=True, slots=True)
class WorkloadFixture:
    kind: WorkloadKind
    pack_indexes: tuple[int, ...]


def build_workload_fixture(kind: WorkloadKind, samples: int) -> WorkloadFixture:
    if samples < 1:
        raise ValueError("samples must be positive")
    if kind == WorkloadKind.uniform:
        pack_count = min(10, samples)
        return WorkloadFixture(
            kind=kind,
            pack_indexes=tuple(index % pack_count for index in range(samples)),
        )
    if kind == WorkloadKind.hot_80_20:
        hot_samples = round(samples * 0.8)
        return WorkloadFixture(
            kind=kind,
            pack_indexes=tuple(0 if index < hot_samples else 1 for index in range(samples)),
        )
    if kind == WorkloadKind.shuffled_equivalent:
        return WorkloadFixture(
            kind=kind,
            pack_indexes=tuple(index % 2 for index in range(samples)),
        )
    return WorkloadFixture(kind=kind, pack_indexes=(0,))


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


def pack_chunks(pack_index: int) -> list[dict[str, str]]:
    return [
        {
            "doc_id": f"document-{pack_index}-a",
            "chunk_id": "0",
            "version": "1",
            "text": f"Deterministic context A for pack {pack_index}. " * 16,
        },
        {
            "doc_id": f"document-{pack_index}-b",
            "chunk_id": "0",
            "version": "1",
            "text": f"Deterministic context B for pack {pack_index}. " * 16,
        },
    ]


async def register_pack(
    client: httpx.AsyncClient,
    base_url: str,
    *,
    pack_index: int = 0,
    reverse_chunks: bool = False,
) -> dict[str, Any]:
    chunks = pack_chunks(pack_index)
    if reverse_chunks:
        chunks.reverse()
    response = await client.post(
        f"{base_url}/v1/packs",
        headers={"X-Flume-Tenant": TENANT_ID},
        json={
            "template_id": "local-benchmark-v1",
            "chunks": chunks,
        },
    )
    response.raise_for_status()
    return dict(response.json())


async def register_fixture_packs(
    client: httpx.AsyncClient,
    base_url: str,
    fixture: WorkloadFixture,
) -> list[dict[str, Any]]:
    if fixture.kind == WorkloadKind.shuffled_equivalent:
        forward = await register_pack(client, base_url, pack_index=100)
        reverse = await register_pack(
            client,
            base_url,
            pack_index=100,
            reverse_chunks=True,
        )
        if forward["pack_id"] != reverse["pack_id"]:
            raise AssertionError("stable ordering produced different pack ids")
        return [forward, reverse]
    pack_count = max(fixture.pack_indexes) + 1
    return [
        await register_pack(client, base_url, pack_index=pack_index)
        for pack_index in range(pack_count)
    ]


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


async def complete_fixture(
    client: httpx.AsyncClient,
    base_url: str,
    fixture: WorkloadFixture,
    packs: list[dict[str, Any]],
    *,
    phase: str,
    concurrency: int,
) -> list[LocalSample]:
    semaphore = asyncio.Semaphore(concurrency)

    async def bounded(index: int, pack_index: int) -> LocalSample:
        async with semaphore:
            return await complete(
                client,
                base_url,
                str(packs[pack_index]["pack_id"]),
                phase=phase,
                index=index,
            )

    return await asyncio.gather(
        *(
            bounded(index, pack_index)
            for index, pack_index in enumerate(fixture.pack_indexes)
        )
    )


async def configure_worker(
    client: httpx.AsyncClient,
    worker_url: str,
    *,
    delay_ms: float,
) -> None:
    response = await client.post(
        f"{worker_url}/benchmark/control",
        json={"delay_ms": delay_ms},
    )
    response.raise_for_status()


async def find_pack_for_worker(
    client: httpx.AsyncClient,
    flume_url: str,
    target_worker_id: str,
) -> tuple[dict[str, Any], LocalSample]:
    for pack_index in range(200, 264):
        pack = await register_pack(client, flume_url, pack_index=pack_index)
        sample = await complete(
            client,
            flume_url,
            str(pack["pack_id"]),
            phase="fixture_setup",
            index=pack_index,
        )
        if sample.worker_id == target_worker_id:
            return pack, sample
    raise RuntimeError("could not create a deterministic pack for the target worker")


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
                stats_response = await client.get(
                    f"{flume_url}/v1/stats",
                    headers={"X-Flume-Tenant": TENANT_ID},
                )
                stats_response.raise_for_status()
                public_worker_ids = [
                    worker["worker_id"] for worker in stats_response.json()["workers"]
                ]
                if len(public_worker_ids) != 2:
                    raise AssertionError("local benchmark requires exactly two workers")

                results: dict[str, Any] = {}
                workload_names = [WorkloadKind(value) for value in args.workload]
                for kind in (
                    WorkloadKind.uniform,
                    WorkloadKind.hot_80_20,
                    WorkloadKind.shuffled_equivalent,
                ):
                    if kind not in workload_names:
                        continue
                    fixture = build_workload_fixture(kind, args.samples)
                    packs = await register_fixture_packs(client, flume_url, fixture)
                    measured = await complete_fixture(
                        client,
                        flume_url,
                        fixture,
                        packs,
                        phase="measured",
                        concurrency=args.concurrency,
                    )
                    results[kind] = {
                        "pack_ids": [pack["pack_id"] for pack in packs],
                        "request_distribution": dict(Counter(fixture.pack_indexes)),
                        "samples": [asdict(sample) for sample in measured],
                    }
                    if kind == WorkloadKind.shuffled_equivalent:
                        results[kind]["equivalent_pack_id"] = (
                            packs[0]["pack_id"] == packs[1]["pack_id"]
                        )

                delay_pack: dict[str, Any] | None = None
                delay_setup: LocalSample | None = None
                if WorkloadKind.worker_delay in workload_names:
                    delay_pack, delay_setup = await find_pack_for_worker(
                        client,
                        flume_url,
                        public_worker_ids[0],
                    )
                    await configure_worker(
                        client,
                        worker_urls[0],
                        delay_ms=args.worker_delay_ms,
                    )
                    delayed = await complete(
                        client,
                        flume_url,
                        str(delay_pack["pack_id"]),
                        phase="measured",
                        index=0,
                    )
                    await configure_worker(client, worker_urls[0], delay_ms=0.0)
                    results[WorkloadKind.worker_delay] = {
                        "configured_delay_ms": args.worker_delay_ms,
                        "target_worker_id": public_worker_ids[0],
                        "setup_sample": asdict(delay_setup),
                        "sample": asdict(delayed),
                    }

                failure_pack: dict[str, Any] | None = None
                failure_setup: LocalSample | None = None
                if (
                    WorkloadKind.worker_failure in workload_names
                    or WorkloadKind.all_unavailable in workload_names
                ):
                    failure_pack, failure_setup = await find_pack_for_worker(
                        client,
                        flume_url,
                        public_worker_ids[0],
                    )
                if WorkloadKind.worker_failure in workload_names:
                    if failure_pack is None or failure_setup is None:
                        raise AssertionError("worker failure fixture was not initialized")
                    processes[0].terminate()
                    processes[0].wait(timeout=5)
                    failed_over = await complete(
                        client,
                        flume_url,
                        str(failure_pack["pack_id"]),
                        phase="measured",
                        index=0,
                    )
                    results[WorkloadKind.worker_failure] = {
                        "failed_worker_id": public_worker_ids[0],
                        "setup_sample": asdict(failure_setup),
                        "sample": asdict(failed_over),
                        "failover_observed": (
                            failed_over.status_code == 200
                            and failed_over.worker_id != public_worker_ids[0]
                        ),
                    }
                if WorkloadKind.all_unavailable in workload_names:
                    if failure_pack is None:
                        raise AssertionError("unavailable fixture was not initialized")
                    if processes[0].poll() is None:
                        processes[0].terminate()
                        processes[0].wait(timeout=5)
                    processes[1].terminate()
                    processes[1].wait(timeout=5)
                    await asyncio.sleep(0.25)
                    unavailable = await complete(
                        client,
                        flume_url,
                        str(failure_pack["pack_id"]),
                        phase="measured",
                        index=0,
                    )
                    results[WorkloadKind.all_unavailable] = {
                        "expected_status": 503,
                        "sample": asdict(unavailable),
                    }
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
            "concurrency": args.concurrency,
            "workloads": args.workload,
        },
        "results": results,
    }

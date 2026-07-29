"""Process-isolated local benchmark topology for Flume."""

from __future__ import annotations

import asyncio
import json
import os
import random
import re
import socket
import subprocess
import sys
import tempfile
import time
from collections import Counter
from collections.abc import Callable
from dataclasses import asdict, dataclass
from enum import StrEnum
from functools import partial
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


class ReadinessKind(StrEnum):
    worker = "worker"
    flume = "flume"


WORKLOAD_PACK_BASE = {
    WorkloadKind.uniform: 0,
    WorkloadKind.hot_80_20: 1_000,
    WorkloadKind.shuffled_equivalent: 2_000,
}


@dataclass(frozen=True, slots=True)
class LocalSample:
    index: int
    phase: str
    status_code: int | None
    worker_id: str | None
    prompt_tokens: int | None
    output_tokens: int | None
    ttft_ms: float | None
    e2e_ms: float
    error: str | None


@dataclass(frozen=True, slots=True)
class WorkloadFixture:
    kind: WorkloadKind
    pack_indexes: tuple[int, ...]


def build_workload_fixture(
    kind: WorkloadKind,
    samples: int,
    *,
    seed: int | None = None,
) -> WorkloadFixture:
    if samples < 1:
        raise ValueError("samples must be positive")
    if kind == WorkloadKind.uniform:
        pack_count = min(10, samples)
        pack_indexes = [index % pack_count for index in range(samples)]
        if seed is not None:
            random.Random(seed).shuffle(pack_indexes)
        return WorkloadFixture(kind=kind, pack_indexes=tuple(pack_indexes))
    if kind == WorkloadKind.hot_80_20:
        hot_samples = round(samples * 0.8)
        pack_indexes = [0 if index < hot_samples else 1 for index in range(samples)]
        if seed is not None:
            random.Random(seed).shuffle(pack_indexes)
        return WorkloadFixture(kind=kind, pack_indexes=tuple(pack_indexes))
    if kind == WorkloadKind.shuffled_equivalent:
        pack_indexes = [index % 2 for index in range(samples)]
        if seed is not None:
            random.Random(seed).shuffle(pack_indexes)
        return WorkloadFixture(kind=kind, pack_indexes=tuple(pack_indexes))
    return WorkloadFixture(kind=kind, pack_indexes=(0,))


def reserve_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def start_server(
    *arguments: str,
    source_tree: Path | None = None,
) -> subprocess.Popen[bytes]:
    command = [sys.executable, "-m", "flume.benchmark_server", *arguments]
    environment = None
    if source_tree is not None:
        command = [
            sys.executable,
            str(Path(__file__).with_name("benchmark_server.py")),
            *arguments,
        ]
        environment = os.environ.copy()
        source_path = str(source_tree / "src")
        existing_pythonpath = environment.get("PYTHONPATH")
        environment["PYTHONPATH"] = (
            f"{source_path}{os.pathsep}{existing_pythonpath}"
            if existing_pythonpath
            else source_path
        )
    return subprocess.Popen(
        command,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        env=environment,
    )


def worker_server_arguments(port: int, *, worker_index: int) -> tuple[str, ...]:
    return (
        "worker",
        "--port",
        str(port),
        "--worker-id",
        f"worker-{worker_index + 1}",
    )


def flume_server_arguments(
    port: int,
    *,
    database_url: str,
    worker_urls: list[str],
) -> tuple[str, ...]:
    return (
        "flume",
        "--port",
        str(port),
        "--database-url",
        database_url,
        *(
            argument
            for worker_url in worker_urls
            for argument in ("--worker-url", worker_url)
        ),
    )


def readiness_payload_is_valid(
    response: httpx.Response,
    readiness: ReadinessKind,
) -> bool:
    if response.status_code != 200:
        return False
    try:
        payload = response.json()
    except (json.JSONDecodeError, ValueError):
        return False
    if readiness == ReadinessKind.worker:
        return payload == {"status": "ok"}
    checks = payload.get("checks")
    return (
        payload.get("status") == "ready"
        and isinstance(checks, dict)
        and bool(checks)
        and all(value is True for value in checks.values())
    )


def process_stderr(process: subprocess.Popen[bytes]) -> str:
    if process.stderr is None:
        return ""
    return process.stderr.read().decode("utf-8", errors="replace")[-4_000:].strip()


async def wait_for_server(
    client: httpx.AsyncClient,
    url: str,
    process: subprocess.Popen[bytes],
    *,
    timeout: float,
    readiness: ReadinessKind,
) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            stderr = process_stderr(process)
            detail = f": {stderr}" if stderr else ""
            raise RuntimeError(
                f"benchmark server exited early with status {process.returncode}{detail}"
            )
        try:
            response = await client.get(url)
            if readiness_payload_is_valid(response, readiness):
                return
        except httpx.HTTPError:
            pass
        await asyncio.sleep(0.025)
    raise TimeoutError(f"timed out waiting for {url}")


def is_port_collision(error: str) -> bool:
    normalized = error.casefold()
    return any(
        marker in normalized
        for marker in (
            "address already in use",
            "errno 48",
            "errno 98",
            "error while attempting to bind",
        )
    )


async def start_ready_process(
    client: httpx.AsyncClient,
    arguments_for_port: Callable[[int], tuple[str, ...]],
    *,
    readiness_path: str,
    readiness: ReadinessKind,
    timeout: float,
    attempts: int = 3,
    source_tree: Path | None = None,
) -> tuple[subprocess.Popen[bytes], str]:
    if attempts < 1:
        raise ValueError("attempts must be positive")
    for attempt in range(attempts):
        port = reserve_port()
        process = start_server(
            *arguments_for_port(port),
            source_tree=source_tree,
        )
        base_url = f"http://127.0.0.1:{port}"
        try:
            await wait_for_server(
                client,
                f"{base_url}{readiness_path}",
                process,
                timeout=timeout,
                readiness=readiness,
            )
        except RuntimeError as exc:
            stop_servers([process])
            if attempt + 1 < attempts and is_port_collision(str(exc)):
                continue
            raise
        return process, base_url
    raise AssertionError("unreachable startup retry state")


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
    pack_base = WORKLOAD_PACK_BASE[fixture.kind]
    if fixture.kind == WorkloadKind.shuffled_equivalent:
        forward = await register_pack(client, base_url, pack_index=pack_base)
        reverse = await register_pack(
            client,
            base_url,
            pack_index=pack_base,
            reverse_chunks=True,
        )
        if forward["pack_id"] != reverse["pack_id"]:
            raise AssertionError("stable ordering produced different pack ids")
        return [forward, reverse]
    pack_count = max(fixture.pack_indexes) + 1
    if fixture.kind == WorkloadKind.hot_80_20:
        pack_count = max(pack_count, 2)
    return [
        await register_pack(client, base_url, pack_index=pack_base + pack_index)
        for pack_index in range(pack_count)
    ]


async def warm_packs(
    client: httpx.AsyncClient,
    base_url: str,
    packs: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[LocalSample], float]:
    responses: list[dict[str, Any]] = []
    samples: list[LocalSample] = []
    seen: set[str] = set()
    phase_started = time.perf_counter()
    for pack in packs:
        pack_id = str(pack["pack_id"])
        if pack_id in seen:
            continue
        seen.add(pack_id)
        started = time.perf_counter()
        response = await client.post(
            f"{base_url}/v1/packs/{pack_id}/warm",
            headers={"X-Flume-Tenant": TENANT_ID},
        )
        response.raise_for_status()
        ended = time.perf_counter()
        payload = dict(response.json())
        responses.append(payload)
        samples.append(
            LocalSample(
                index=len(samples),
                phase="setup",
                status_code=response.status_code,
                worker_id=str(payload["worker_id"]),
                prompt_tokens=None,
                output_tokens=1,
                ttft_ms=None,
                e2e_ms=(ended - started) * 1000,
                error=None,
            )
        )
    return responses, samples, time.perf_counter() - phase_started


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
    first_token_at: float | None = None
    error: str | None = None
    saw_token = False
    saw_usage = False
    saw_done = False
    try:
        async with client.stream(
            "POST",
            f"{base_url}/v1/completions",
            headers={"X-Flume-Tenant": TENANT_ID},
            json={
                "pack_id": pack_id,
                "prompt": f"Question {index}: summarize the policy.",
                "max_tokens": 1,
                "temperature": 0.0,
                "stream": True,
                "extra_body": {"stream_options": {"include_usage": True}},
            },
        ) as response:
            status_code = response.status_code
            worker_id = response.headers.get("X-Flume-Worker-Id")
            prompt_tokens_header = response.headers.get("X-Flume-Prompt-Tokens")
            if prompt_tokens_header is not None:
                prompt_tokens = int(prompt_tokens_header)
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line.startswith("data:"):
                    continue
                raw_data = line.removeprefix("data:").strip()
                if not raw_data:
                    continue
                if raw_data == "[DONE]":
                    saw_done = True
                    continue
                try:
                    payload = json.loads(raw_data)
                except json.JSONDecodeError:
                    continue
                if first_token_at is None and any(
                    choice.get("text") for choice in payload.get("choices", [])
                ):
                    first_token_at = time.perf_counter()
                    saw_token = True
                usage = payload.get("usage") or {}
                if usage.get("prompt_tokens") is not None:
                    prompt_tokens = int(usage["prompt_tokens"])
                if usage.get("completion_tokens") is not None:
                    output_tokens = int(usage["completion_tokens"])
                    saw_usage = True
    except (httpx.HTTPError, ValueError) as exc:
        error = f"{type(exc).__name__}: {exc}"
    if error is None and not (saw_token and saw_usage and saw_done):
        missing = [
            name
            for name, seen in (
                ("token", saw_token),
                ("usage", saw_usage),
                ("done", saw_done),
            )
            if not seen
        ]
        error = f"InvalidSSE: missing {', '.join(missing)}"
    ended = time.perf_counter()
    return LocalSample(
        index=index,
        phase=phase,
        status_code=status_code,
        worker_id=worker_id,
        prompt_tokens=prompt_tokens,
        output_tokens=output_tokens,
        ttft_ms=(first_token_at - started) * 1000 if first_token_at is not None else None,
        e2e_ms=(ended - started) * 1000,
        error=error,
    )


def metric_delta(
    before: dict[str, dict[str, float]],
    after: dict[str, dict[str, float]],
) -> dict[str, dict[str, float]]:
    return {
        worker: {
            metric: after.get(worker, {}).get(metric, 0.0)
            - before.get(worker, {}).get(metric, 0.0)
            for metric in set(before.get(worker, {})) | set(after.get(worker, {}))
        }
        for worker in set(before) | set(after)
    }


async def synthetic_metrics_snapshot(
    client: httpx.AsyncClient,
    worker_urls: list[str],
) -> dict[str, dict[str, float]]:
    snapshot: dict[str, dict[str, float]] = {}
    for worker_url in worker_urls:
        try:
            response = await client.get(f"{worker_url}/metrics")
            response.raise_for_status()
        except httpx.HTTPError:
            snapshot[worker_url] = {}
            continue
        metrics: dict[str, float] = {}
        for line in response.text.splitlines():
            name, separator, raw_value = line.rpartition(" ")
            if not separator or not name.startswith("flume_benchmark_synthetic_"):
                continue
            try:
                metrics[name] = float(raw_value)
            except ValueError:
                continue
        snapshot[worker_url] = metrics
    return snapshot


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


def latency_summary(values: list[float]) -> dict[str, float]:
    summary = {
        "p50": percentile(values, 0.50),
        "p95": percentile(values, 0.95),
    }
    if len(values) >= 1000:
        summary["p99"] = percentile(values, 0.99)
    return summary


def phase_snapshot(
    samples: list[LocalSample],
    *,
    duration_seconds: float | None = None,
    metrics_before: dict[str, dict[str, float]] | None = None,
    metrics_after: dict[str, dict[str, float]] | None = None,
) -> dict[str, Any]:
    if duration_seconds is None:
        duration_seconds = max((sample.e2e_ms for sample in samples), default=0.0) / 1000
    successful = [sample for sample in samples if sample.error is None]
    ttft = [sample.ttft_ms for sample in successful if sample.ttft_ms is not None]
    e2e = [sample.e2e_ms for sample in successful]
    snapshot = {
        "raw_sample_count": len(samples),
        "duration_seconds": duration_seconds,
        "throughput_rps": len(samples) / duration_seconds if duration_seconds else 0.0,
        "errors": len(samples) - len(successful),
        "ttft_ms": latency_summary(ttft),
        "e2e_ms": latency_summary(e2e),
        "status_counts": dict(Counter(str(sample.status_code) for sample in samples)),
        "route_counts": dict(
            Counter(
                sample.worker_id or "unassigned"
                for sample in samples
            )
        ),
        "samples": [asdict(sample) for sample in samples],
    }
    if metrics_before is not None and metrics_after is not None:
        snapshot["synthetic_worker_metrics"] = {
            "before": metrics_before,
            "after": metrics_after,
            "delta": metric_delta(metrics_before, metrics_after),
        }
    return snapshot


def validate_local_results(
    results: dict[str, Any],
    workload_names: list[WorkloadKind],
) -> list[str]:
    issues: list[str] = []
    zero_error_workloads = {
        WorkloadKind.uniform,
        WorkloadKind.hot_80_20,
        WorkloadKind.shuffled_equivalent,
        WorkloadKind.worker_delay,
    }
    for workload in zero_error_workloads.intersection(workload_names):
        for phase, snapshot in results[workload]["phase_snapshots"].items():
            if snapshot["errors"]:
                issues.append(f"{workload.value}/{phase} recorded errors")

    if WorkloadKind.worker_failure in workload_names:
        failure = results[WorkloadKind.worker_failure]
        if not failure["failover_observed"]:
            issues.append("worker_failure did not observe failover")
        if failure["phase_snapshots"]["measured"]["errors"]:
            issues.append("worker_failure measured request failed")

    if WorkloadKind.all_unavailable in workload_names:
        unavailable = results[WorkloadKind.all_unavailable]["phase_snapshots"]["measured"]
        statuses = [sample["status_code"] for sample in unavailable["samples"]]
        if statuses != [503]:
            issues.append("all_unavailable did not return exactly one intentional 503")
    return issues


async def complete_fixture(
    client: httpx.AsyncClient,
    base_url: str,
    fixture: WorkloadFixture,
    packs: list[dict[str, Any]],
    *,
    phase: str,
    concurrency: int,
) -> tuple[list[LocalSample], float]:
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

    started = time.perf_counter()
    samples = await asyncio.gather(
        *(
            bounded(index, pack_index)
            for index, pack_index in enumerate(fixture.pack_indexes)
        )
    )
    return samples, time.perf_counter() - started


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
    *,
    start_index: int,
) -> tuple[dict[str, Any], LocalSample, int]:
    for pack_index in range(start_index, start_index + 64):
        pack = await register_pack(client, flume_url, pack_index=pack_index)
        sample = await complete(
            client,
            flume_url,
            str(pack["pack_id"]),
            phase="cold",
            index=pack_index,
        )
        if sample.worker_id == target_worker_id:
            return pack, sample, pack_index - start_index + 1
    raise RuntimeError("could not create a deterministic pack for the target worker")


async def run_local_benchmark(args: Any) -> dict[str, Any]:
    source_tree_value = getattr(args, "source_tree", None)
    source_tree = (
        Path(source_tree_value).expanduser().resolve()
        if source_tree_value is not None
        else None
    )
    seed = int(getattr(args, "seed", 20260729))
    worker_urls: list[str] = []
    processes: list[subprocess.Popen[bytes]] = []
    service_counters: dict[str, Any] = {}
    with tempfile.TemporaryDirectory(prefix="flume-benchmark-") as temp_dir:
        database_url = f"sqlite:///{Path(temp_dir) / 'flume.db'}"
        try:
            timeout = httpx.Timeout(args.timeout)
            async with httpx.AsyncClient(timeout=timeout) as client:
                for index in range(2):
                    process, worker_url = await start_ready_process(
                        client,
                        partial(
                            worker_server_arguments,
                            worker_index=index,
                        ),
                        readiness_path="/health",
                        readiness=ReadinessKind.worker,
                        timeout=args.startup_timeout,
                        source_tree=source_tree,
                    )
                    processes.append(process)
                    worker_urls.append(worker_url)
                flume_process, flume_url = await start_ready_process(
                    client,
                    partial(
                        flume_server_arguments,
                        database_url=database_url,
                        worker_urls=worker_urls,
                    ),
                    readiness_path="/readyz",
                    readiness=ReadinessKind.flume,
                    timeout=args.startup_timeout,
                    source_tree=source_tree,
                )
                processes.append(flume_process)
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
                raw_sample_counts = {
                    "cold": 0,
                    "setup": 0,
                    "warmup": 0,
                    "measured": 0,
                }
                phase_samples: dict[str, list[LocalSample]] = {
                    "cold": [],
                    "setup": [],
                    "warmup": [],
                    "measured": [],
                }
                phase_durations = {
                    "cold": 0.0,
                    "setup": 0.0,
                    "warmup": 0.0,
                    "measured": 0.0,
                }
                setup_request_counts: dict[str, int] = {}
                workload_names = [WorkloadKind(value) for value in args.workload]
                for kind in (
                    WorkloadKind.uniform,
                    WorkloadKind.hot_80_20,
                    WorkloadKind.shuffled_equivalent,
                ):
                    if kind not in workload_names:
                        continue
                    fixture = build_workload_fixture(
                        kind,
                        args.samples,
                        seed=seed,
                    )
                    packs = await register_fixture_packs(client, flume_url, fixture)
                    unique_pack_indexes: list[int] = []
                    seen_pack_ids: set[str] = set()
                    for pack_index in fixture.pack_indexes:
                        pack_id = str(packs[pack_index]["pack_id"])
                        if pack_id not in seen_pack_ids:
                            seen_pack_ids.add(pack_id)
                            unique_pack_indexes.append(pack_index)
                    cold_fixture = WorkloadFixture(
                        kind=kind,
                        pack_indexes=tuple(unique_pack_indexes),
                    )
                    before_cold = await synthetic_metrics_snapshot(client, worker_urls)
                    cold, cold_duration = await complete_fixture(
                        client,
                        flume_url,
                        cold_fixture,
                        packs,
                        phase="cold",
                        concurrency=args.concurrency,
                    )
                    after_cold = await synthetic_metrics_snapshot(client, worker_urls)
                    (
                        explicit_warmups,
                        setup_samples,
                        setup_duration,
                    ) = await warm_packs(client, flume_url, packs)
                    after_setup = await synthetic_metrics_snapshot(client, worker_urls)
                    if args.warmups:
                        raw_warmup_fixture = build_workload_fixture(
                            kind,
                            args.warmups,
                            seed=seed + 1,
                        )
                        warmup_fixture = WorkloadFixture(
                            kind=kind,
                            pack_indexes=tuple(
                                pack_index % len(packs)
                                for pack_index in raw_warmup_fixture.pack_indexes
                            ),
                        )
                        warmup, warmup_duration = await complete_fixture(
                            client,
                            flume_url,
                            warmup_fixture,
                            packs,
                            phase="warmup",
                            concurrency=args.concurrency,
                        )
                    else:
                        warmup = []
                        warmup_duration = 0.0
                    after_warmup = await synthetic_metrics_snapshot(client, worker_urls)
                    measured, measured_duration = await complete_fixture(
                        client,
                        flume_url,
                        fixture,
                        packs,
                        phase="measured",
                        concurrency=args.concurrency,
                    )
                    after_measured = await synthetic_metrics_snapshot(client, worker_urls)
                    results[kind] = {
                        "pack_ids": [pack["pack_id"] for pack in packs],
                        "request_distribution": dict(Counter(fixture.pack_indexes)),
                        "explicit_warmups": explicit_warmups,
                        "phase_snapshots": {
                            "cold": phase_snapshot(
                                cold,
                                duration_seconds=cold_duration,
                                metrics_before=before_cold,
                                metrics_after=after_cold,
                            ),
                            "setup": phase_snapshot(
                                setup_samples,
                                duration_seconds=setup_duration,
                                metrics_before=after_cold,
                                metrics_after=after_setup,
                            ),
                            "warmup": phase_snapshot(
                                warmup,
                                duration_seconds=warmup_duration,
                                metrics_before=after_setup,
                                metrics_after=after_warmup,
                            ),
                            "measured": phase_snapshot(
                                measured,
                                duration_seconds=measured_duration,
                                metrics_before=after_warmup,
                                metrics_after=after_measured,
                            ),
                        },
                        "raw_sample_counts": {
                            "cold": len(cold),
                            "setup": len(setup_samples),
                            "warmup": len(warmup),
                            "measured": len(measured),
                        },
                    }
                    for phase, samples, duration in (
                        ("cold", cold, cold_duration),
                        ("setup", setup_samples, setup_duration),
                        ("warmup", warmup, warmup_duration),
                        ("measured", measured, measured_duration),
                    ):
                        phase_samples[phase].extend(samples)
                        phase_durations[phase] += duration
                        raw_sample_counts[phase] += len(samples)
                    if kind == WorkloadKind.shuffled_equivalent:
                        results[kind]["equivalent_pack_id"] = (
                            packs[0]["pack_id"] == packs[1]["pack_id"]
                        )

                delay_pack: dict[str, Any] | None = None
                delay_setup: LocalSample | None = None
                if WorkloadKind.worker_delay in workload_names:
                    delay_pack, delay_setup, delay_setup_requests = await find_pack_for_worker(
                        client,
                        flume_url,
                        public_worker_ids[0],
                        start_index=3_000,
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
                        "phase_snapshots": {
                            "cold": phase_snapshot(
                                [delay_setup],
                                duration_seconds=delay_setup.e2e_ms / 1000,
                            ),
                            "setup": phase_snapshot([], duration_seconds=0.0),
                            "warmup": phase_snapshot([], duration_seconds=0.0),
                            "measured": phase_snapshot(
                                [delayed],
                                duration_seconds=delayed.e2e_ms / 1000,
                            ),
                        },
                        "raw_sample_counts": {
                            "cold": 1,
                            "setup": 0,
                            "warmup": 0,
                            "measured": 1,
                        },
                    }
                    setup_request_counts[WorkloadKind.worker_delay] = delay_setup_requests
                    phase_samples["cold"].append(delay_setup)
                    phase_samples["measured"].append(delayed)
                    phase_durations["cold"] += delay_setup.e2e_ms / 1000
                    phase_durations["measured"] += delayed.e2e_ms / 1000
                    raw_sample_counts["cold"] += 1
                    raw_sample_counts["measured"] += 1

                failure_pack: dict[str, Any] | None = None
                failure_setup: LocalSample | None = None
                if (
                    WorkloadKind.worker_failure in workload_names
                    or WorkloadKind.all_unavailable in workload_names
                ):
                    (
                        failure_pack,
                        failure_setup,
                        failure_setup_requests,
                    ) = await find_pack_for_worker(
                        client,
                        flume_url,
                        public_worker_ids[0],
                        start_index=4_000,
                    )
                    setup_request_counts["failure_target_selection"] = failure_setup_requests
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
                        "failover_observed": (
                            failed_over.status_code == 200
                            and failed_over.worker_id != public_worker_ids[0]
                        ),
                        "phase_snapshots": {
                            "cold": phase_snapshot(
                                [failure_setup],
                                duration_seconds=failure_setup.e2e_ms / 1000,
                            ),
                            "setup": phase_snapshot([], duration_seconds=0.0),
                            "warmup": phase_snapshot([], duration_seconds=0.0),
                            "measured": phase_snapshot(
                                [failed_over],
                                duration_seconds=failed_over.e2e_ms / 1000,
                            ),
                        },
                        "raw_sample_counts": {
                            "cold": 1,
                            "setup": 0,
                            "warmup": 0,
                            "measured": 1,
                        },
                    }
                    phase_samples["cold"].append(failure_setup)
                    phase_samples["measured"].append(failed_over)
                    phase_durations["cold"] += failure_setup.e2e_ms / 1000
                    phase_durations["measured"] += failed_over.e2e_ms / 1000
                    raw_sample_counts["cold"] += 1
                    raw_sample_counts["measured"] += 1
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
                        "phase_snapshots": {
                            "cold": phase_snapshot([], duration_seconds=0.0),
                            "setup": phase_snapshot([], duration_seconds=0.0),
                            "warmup": phase_snapshot([], duration_seconds=0.0),
                            "measured": phase_snapshot(
                                [unavailable],
                                duration_seconds=unavailable.e2e_ms / 1000,
                            ),
                        },
                        "raw_sample_counts": {
                            "cold": 0,
                            "setup": 0,
                            "warmup": 0,
                            "measured": 1,
                        },
                    }
                    phase_samples["measured"].append(unavailable)
                    phase_durations["measured"] += unavailable.e2e_ms / 1000
                    raw_sample_counts["measured"] += 1
                final_stats_response = await client.get(
                    f"{flume_url}/v1/stats",
                    headers={"X-Flume-Tenant": TENANT_ID},
                )
                final_stats_response.raise_for_status()
                final_stats = final_stats_response.json()
                service_counters = {
                    "database_pack_rows": int(final_stats["packs"]),
                    "router_route_rows": int(final_stats["routes"]),
                    "workers": final_stats["workers"],
                }
        finally:
            stop_servers(processes)
    from flume.benchmark_cli import build_provenance, environment_metadata

    environment = environment_metadata()
    explicit_tested_commit = getattr(args, "tested_commit", None)
    local_commit = str(
        explicit_tested_commit
        or environment.get("working_directory_commit", "unknown")
    )
    commit_known = re.fullmatch(r"[0-9a-fA-F]{7,64}", local_commit) is not None
    release_valid = commit_known and not bool(environment.get("dirty", False))
    validation_issues = validate_local_results(results, workload_names)
    return {
        "schema_version": 2,
        "status": "failed" if validation_issues else "completed",
        "validation": {
            "passed": not validation_issues,
            "issues": validation_issues,
            "intentional_statuses": {"all_unavailable": 503},
        },
        "environment": environment,
        "provenance": {
            **build_provenance(
                environment,
                benchmark_mode="local",
                gpu_validated=False,
                revisions={
                    "model": "local-benchmark",
                    "tokenizer": "deterministic-byte-tokenizer",
                    "tokenizer_revision": "byte-v1",
                    "mock_worker_protocol": "openai-completions-v1",
                    "mock_prefix_cache_metrics": "synthetic",
                },
                tested_commit=local_commit if commit_known else None,
                tested_commit_source=(
                    "external_target"
                    if explicit_tested_commit and commit_known
                    else ("local_repo_head" if commit_known else "unknown")
                ),
                release_valid=bool(explicit_tested_commit and commit_known)
                or release_valid,
            ),
            "topology": {
                "flume_processes": 1,
                "mock_worker_processes": 2,
                "process_isolated": True,
            },
            "setup_request_counts": setup_request_counts,
        },
        "configuration": {
            "benchmark": "local",
            "workers": 2,
            "warmups": args.warmups,
            "samples": args.samples,
            "concurrency": args.concurrency,
            "workloads": args.workload,
            "seed": seed,
        },
        "service_counters": service_counters,
        "results": results,
        "phase_snapshots": {
            phase: phase_snapshot(
                samples,
                duration_seconds=phase_durations[phase],
            )
            for phase, samples in phase_samples.items()
        },
        "raw_sample_counts": raw_sample_counts,
    }

"""Offline vLLM benchmark for cache-stability and worker-affinity scenarios."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import platform
import random
import subprocess
import time
import uuid
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, Protocol

import httpx


class Tokenizer(Protocol):
    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]: ...


class Scenario(StrEnum):
    apc_disabled = "apc_disabled"
    unstable_prefix_random_workers = "unstable_prefix_random_workers"
    stable_prefix_random_workers = "stable_prefix_random_workers"
    stable_warmed_prefix_affinity = "stable_warmed_prefix_affinity"


@dataclass(frozen=True, slots=True)
class SamplePlan:
    index: int
    phase: str
    worker_url: str
    prompt_token_ids: list[int]
    max_tokens: int
    cache_salt: str


@dataclass(slots=True)
class Sample:
    index: int
    phase: str
    worker_url: str
    max_tokens: int
    prompt_tokens: int
    output_tokens: int | None
    ttft_ms: float | None
    e2e_ms: float
    status_code: int | None
    error: str | None


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
        "min": min(values, default=0.0),
        "max": max(values, default=0.0),
    }
    if len(values) >= 1000:
        summary["p99"] = percentile(values, 0.99)
    return summary


def exact_prefix_tokens(tokenizer: Tokenizer, target_tokens: int) -> list[int]:
    if target_tokens < 1:
        raise ValueError("context length must be positive")
    seed = tokenizer.encode(
        " Flume deterministic benchmark context for automatic prefix caching.",
        add_special_tokens=False,
    )
    if not seed:
        raise ValueError("tokenizer produced no benchmark seed tokens")
    repetitions = (target_tokens + len(seed) - 1) // len(seed)
    result = (seed * repetitions)[:target_tokens]
    if len(result) != target_tokens:
        raise AssertionError("failed to generate the requested token length")
    return result


def question_tokens(tokenizer: Tokenizer, index: int) -> list[int]:
    tokens = tokenizer.encode(
        f"\n\nQuestion {index % 17}: summarize the applicable rule.\n\nAnswer:\n",
        add_special_tokens=False,
    )
    if not tokens:
        raise ValueError("tokenizer produced no question tokens")
    return tokens


def unstable_prefix(
    tokenizer: Tokenizer,
    stable_prefix: list[int],
    index: int,
) -> list[int]:
    identity = uuid.uuid5(uuid.NAMESPACE_OID, str(index))
    unique = tokenizer.encode(
        f" request-{index}-{identity} ",
        add_special_tokens=False,
    )
    if not unique:
        raise ValueError("tokenizer produced no unstable-prefix tokens")
    return (unique + stable_prefix)[: len(stable_prefix)]


def affinity_worker(workers: list[str], identity: str) -> str:
    return max(
        workers,
        key=lambda worker: hashlib.sha256(f"{identity}\0{worker}".encode()).digest(),
    )


def choose_worker(
    scenario: Scenario,
    workers: list[str],
    *,
    index: int,
    affinity_identity: str,
    seed: int,
) -> str:
    if scenario == Scenario.stable_warmed_prefix_affinity:
        return affinity_worker(workers, affinity_identity)
    return random.Random(seed + index).choice(workers)


def build_sample_plans(
    *,
    tokenizer: Tokenizer,
    scenario: Scenario,
    workers: list[str],
    context_length: int,
    phase: str,
    samples: int,
    max_output_tokens: int,
    run_salt: str,
    seed: int,
) -> list[SamplePlan]:
    stable_prefix = exact_prefix_tokens(tokenizer, context_length)
    plans: list[SamplePlan] = []
    affinity_identity = f"{run_salt}:{context_length}"
    for index in range(samples):
        prefix = stable_prefix
        if scenario == Scenario.unstable_prefix_random_workers:
            prefix = unstable_prefix(tokenizer, stable_prefix, index)
        prompt = prefix + question_tokens(tokenizer, index)
        worker = choose_worker(
            scenario,
            workers,
            index=index,
            affinity_identity=affinity_identity,
            seed=seed,
        )
        sample_salt = f"{run_salt}:{scenario}:{context_length}"
        if phase == "cold" or scenario == Scenario.unstable_prefix_random_workers:
            sample_salt = f"{sample_salt}:{phase}:{index}"
        plans.append(
            SamplePlan(
                index=index,
                phase=phase,
                worker_url=worker,
                prompt_token_ids=prompt,
                max_tokens=(index % max_output_tokens) + 1,
                cache_salt=sample_salt,
            )
        )
    return plans


def line_has_token(line: str) -> bool:
    if not line.startswith("data:"):
        return False
    data = line.removeprefix("data:").strip()
    if not data or data == "[DONE]":
        return False
    try:
        payload = json.loads(data)
    except json.JSONDecodeError:
        return False
    return any(choice.get("text") for choice in payload.get("choices", []))


async def execute_sample(
    client: httpx.AsyncClient,
    plan: SamplePlan,
    *,
    model: str,
) -> Sample:
    started = time.perf_counter()
    first_token_at: float | None = None
    status_code: int | None = None
    output_tokens: int | None = None
    error: str | None = None
    payload = {
        "model": model,
        "prompt": plan.prompt_token_ids,
        "max_tokens": plan.max_tokens,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},
        "cache_salt": plan.cache_salt,
    }
    try:
        async with client.stream(
            "POST",
            f"{plan.worker_url}/v1/completions",
            json=payload,
        ) as response:
            status_code = response.status_code
            response.raise_for_status()
            async for line in response.aiter_lines():
                now = time.perf_counter()
                if first_token_at is None and line_has_token(line):
                    first_token_at = now
                if line.startswith("data:") and line.removeprefix("data:").strip() != "[DONE]":
                    try:
                        event = json.loads(line.removeprefix("data:").strip())
                    except json.JSONDecodeError:
                        continue
                    usage = event.get("usage") or {}
                    if usage.get("completion_tokens") is not None:
                        output_tokens = int(usage["completion_tokens"])
    except (httpx.HTTPError, ValueError) as exc:
        error = f"{type(exc).__name__}: {exc}"
    ended = time.perf_counter()
    return Sample(
        index=plan.index,
        phase=plan.phase,
        worker_url=plan.worker_url,
        max_tokens=plan.max_tokens,
        prompt_tokens=len(plan.prompt_token_ids),
        output_tokens=output_tokens,
        ttft_ms=(first_token_at - started) * 1000 if first_token_at is not None else None,
        e2e_ms=(ended - started) * 1000,
        status_code=status_code,
        error=error,
    )


async def execute_plans(
    client: httpx.AsyncClient,
    plans: list[SamplePlan],
    *,
    model: str,
    concurrency: int,
) -> tuple[list[Sample], float]:
    semaphore = asyncio.Semaphore(concurrency)

    async def bounded(plan: SamplePlan) -> Sample:
        async with semaphore:
            return await execute_sample(client, plan, model=model)

    started = time.perf_counter()
    samples = await asyncio.gather(*(bounded(plan) for plan in plans))
    return samples, time.perf_counter() - started


def parse_prometheus(text: str) -> dict[str, float]:
    wanted = ("prefix_cache_queries", "prefix_cache_hits")
    totals: dict[str, float] = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        metric, separator, raw_value = line.rpartition(" ")
        if not separator:
            continue
        base = metric.split("{", 1)[0]
        if not any(name in base for name in wanted):
            continue
        try:
            totals[base] = totals.get(base, 0.0) + float(raw_value)
        except ValueError:
            continue
    return totals


async def metrics_snapshot(
    client: httpx.AsyncClient,
    workers: Iterable[str],
) -> dict[str, dict[str, float]]:
    snapshots: dict[str, dict[str, float]] = {}
    for worker in workers:
        try:
            response = await client.get(f"{worker}/metrics")
            response.raise_for_status()
        except httpx.HTTPError:
            snapshots[worker] = {}
        else:
            snapshots[worker] = parse_prometheus(response.text)
    return snapshots


def metric_delta(
    before: dict[str, dict[str, float]],
    after: dict[str, dict[str, float]],
) -> dict[str, dict[str, float]]:
    return {
        worker: {
            metric: after.get(worker, {}).get(metric, 0.0) - before.get(worker, {}).get(metric, 0.0)
            for metric in set(before.get(worker, {})) | set(after.get(worker, {}))
        }
        for worker in set(before) | set(after)
    }


def summarize_samples(samples: list[Sample], duration_seconds: float) -> dict[str, Any]:
    successful = [sample for sample in samples if sample.error is None]
    ttft = [sample.ttft_ms for sample in successful if sample.ttft_ms is not None]
    e2e = [sample.e2e_ms for sample in successful]
    output_tokens = sum(sample.output_tokens or 0 for sample in successful)
    return {
        "requests": len(samples),
        "successful_requests": len(successful),
        "errors": len(samples) - len(successful),
        "duration_seconds": duration_seconds,
        "throughput_rps": len(samples) / duration_seconds if duration_seconds else 0.0,
        "output_tokens_per_second": output_tokens / duration_seconds if duration_seconds else 0.0,
        "ttft_ms": latency_summary(ttft),
        "e2e_ms": latency_summary(e2e),
        "samples": [asdict(sample) for sample in samples],
    }


def load_tokenizer(tokenizer_id: str, revision: str, allow_remote: bool) -> Tokenizer:
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(
        tokenizer_id,
        revision=revision,
        local_files_only=not allow_remote,
        trust_remote_code=False,
    )


def environment_metadata() -> dict[str, Any]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "commit": commit or "unknown",
        "hostname": platform.node(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "logical_cpu_count": os.cpu_count(),
    }


async def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    tokenizer = load_tokenizer(args.tokenizer, args.tokenizer_revision, args.allow_remote_tokenizer)
    scenarios = [Scenario(value) for value in args.scenario]
    result: dict[str, Any] = {
        "schema_version": 1,
        "status": "completed",
        "environment": environment_metadata(),
        "configuration": {
            "model": args.model,
            "tokenizer": args.tokenizer,
            "tokenizer_revision": args.tokenizer_revision,
            "vllm_revision": args.vllm_revision,
            "apc_workers": args.apc_worker,
            "apc_disabled_workers": args.apc_disabled_worker,
            "context_lengths": args.context_length,
            "concurrency": args.concurrency,
            "discarded_warmups": args.warmups,
            "warm_samples": args.warm_samples,
            "cold_samples": args.cold_samples,
            "max_output_tokens": args.max_output_tokens,
            "streaming": True,
            "seed": args.seed,
        },
        "results": {},
        "skipped": {},
    }
    run_salt = uuid.uuid4().hex
    timeout = httpx.Timeout(args.timeout, connect=min(args.timeout, 10.0))
    limits = httpx.Limits(max_connections=max(args.concurrency) * 2)
    async with httpx.AsyncClient(timeout=timeout, limits=limits) as client:
        for scenario in scenarios:
            workers = (
                args.apc_disabled_worker if scenario == Scenario.apc_disabled else args.apc_worker
            )
            if not workers:
                result["skipped"][scenario] = "no matching worker pool configured"
                continue
            scenario_results: dict[str, Any] = {}
            for context_length in args.context_length:
                for concurrency in args.concurrency:
                    cell = f"context_{context_length}/concurrency_{concurrency}"
                    before = await metrics_snapshot(client, workers) if args.collect_metrics else {}
                    cold_plans = build_sample_plans(
                        tokenizer=tokenizer,
                        scenario=scenario,
                        workers=workers,
                        context_length=context_length,
                        phase="cold",
                        samples=args.cold_samples,
                        max_output_tokens=args.max_output_tokens,
                        run_salt=run_salt,
                        seed=args.seed,
                    )
                    cold_samples, cold_duration = await execute_plans(
                        client,
                        cold_plans,
                        model=args.model,
                        concurrency=concurrency,
                    )
                    warmup_plans = build_sample_plans(
                        tokenizer=tokenizer,
                        scenario=scenario,
                        workers=workers,
                        context_length=context_length,
                        phase="warmup",
                        samples=args.warmups,
                        max_output_tokens=args.max_output_tokens,
                        run_salt=run_salt,
                        seed=args.seed,
                    )
                    await execute_plans(
                        client,
                        warmup_plans,
                        model=args.model,
                        concurrency=concurrency,
                    )
                    warm_plans = build_sample_plans(
                        tokenizer=tokenizer,
                        scenario=scenario,
                        workers=workers,
                        context_length=context_length,
                        phase="warm",
                        samples=args.warm_samples,
                        max_output_tokens=args.max_output_tokens,
                        run_salt=run_salt,
                        seed=args.seed,
                    )
                    warm_samples, warm_duration = await execute_plans(
                        client,
                        warm_plans,
                        model=args.model,
                        concurrency=concurrency,
                    )
                    after = await metrics_snapshot(client, workers) if args.collect_metrics else {}
                    scenario_results[cell] = {
                        "context_prefix_tokens": context_length,
                        "cold": summarize_samples(cold_samples, cold_duration),
                        "warm": summarize_samples(warm_samples, warm_duration),
                        "discarded_warmups": args.warmups,
                        "vllm_metric_delta": metric_delta(before, after),
                    }
            result["results"][scenario] = scenario_results
    return result


def markdown_report(result: dict[str, Any]) -> str:
    config = result["configuration"]
    lines = [
        "# Flume GPU benchmark",
        "",
        f"- Commit: `{result['environment']['commit']}`",
        f"- Model: `{config['model']}`",
        f"- Tokenizer: `{config['tokenizer']}@{config['tokenizer_revision']}`",
        f"- vLLM revision: `{config['vllm_revision']}`",
        f"- APC-enabled workers: `{len(config['apc_workers'])}`",
        f"- APC-disabled workers: `{len(config['apc_disabled_workers'])}`",
        "",
        "| Scenario | Cell | Phase | Requests | Errors | req/s | TTFT p50/p95 (ms) | "
        "E2E p50/p95 (ms) |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for scenario, cells in result["results"].items():
        for cell, measurement in cells.items():
            for phase in ("cold", "warm"):
                phase_result = measurement[phase]
                ttft = phase_result["ttft_ms"]
                e2e = phase_result["e2e_ms"]
                lines.append(
                    f"| {scenario} | {cell} | {phase} | {phase_result['requests']} | "
                    f"{phase_result['errors']} | {phase_result['throughput_rps']:.2f} | "
                    f"{ttft['p50']:.2f}/{ttft['p95']:.2f} | "
                    f"{e2e['p50']:.2f}/{e2e['p95']:.2f} |"
                )
    if result["skipped"]:
        lines.extend(["", "## Skipped scenarios", ""])
        lines.extend(f"- `{scenario}`: {reason}" for scenario, reason in result["skipped"].items())
    lines.append("")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--tokenizer-revision", required=True)
    parser.add_argument("--vllm-revision", required=True)
    parser.add_argument("--apc-worker", action="append", default=[])
    parser.add_argument("--apc-disabled-worker", action="append", default=[])
    parser.add_argument(
        "--scenario",
        action="append",
        choices=[scenario.value for scenario in Scenario],
        default=[],
    )
    parser.add_argument("--context-length", action="append", type=int, default=[])
    parser.add_argument("--concurrency", action="append", type=int, default=[])
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--warm-samples", type=int, default=100)
    parser.add_argument("--cold-samples", type=int, default=10)
    parser.add_argument("--max-output-tokens", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--seed", type=int, default=20260729)
    parser.add_argument("--collect-metrics", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--allow-remote-tokenizer", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("benchmark-results.json"))
    return parser


def parse_args(arguments: list[str] | None = None) -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args(arguments)
    args.scenario = args.scenario or [scenario.value for scenario in Scenario]
    args.context_length = args.context_length or [4096, 16384, 65536]
    args.concurrency = args.concurrency or [1, 8, 32]
    if not args.apc_worker and not args.apc_disabled_worker:
        parser.error("configure at least one --apc-worker or --apc-disabled-worker")
    positive = [
        *args.context_length,
        *args.concurrency,
        args.warm_samples,
        args.cold_samples,
        args.max_output_tokens,
    ]
    if any(value < 1 for value in positive) or args.warmups < 0:
        parser.error("lengths, concurrency, samples, and output tokens must be positive")
    if args.max_output_tokens > 8:
        parser.error("--max-output-tokens must be between 1 and 8")
    return args


def main() -> None:
    args = parse_args()
    result = asyncio.run(run_benchmark(args))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    report_path = args.output.with_suffix(".md")
    report_path.write_text(markdown_report(result), encoding="utf-8")
    print(f"wrote {args.output} and {report_path}")


if __name__ == "__main__":
    main()

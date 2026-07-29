"""Paired, local-only Apple M5 performance gate built on the Flume local harness."""

from __future__ import annotations

import hashlib
import json
import platform
import random
import re
import subprocess
from argparse import Namespace
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from flume.benchmark_local import latency_summary, run_local_benchmark

SCHEMA_VERSION = 1
DEFAULT_SEEDS = (20260729, 20260730, 20260731)
REQUIRED_CONCURRENCY = (1, 8, 32)


@dataclass(frozen=True, slots=True)
class Target:
    label: str
    source_tree: Path
    commit: str


def git_output(source_tree: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(source_tree), *arguments],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode:
        detail = result.stderr.strip() or result.stdout.strip()
        raise ValueError(f"could not inspect {source_tree.name}: {detail}")
    return result.stdout.strip()


def resolve_target(label: str, source_tree: Path, expected_commit: str) -> Target:
    source_tree = source_tree.expanduser().resolve()
    if not (source_tree / "pyproject.toml").is_file():
        raise ValueError(f"{label} source tree does not contain pyproject.toml")
    if re.fullmatch(r"[0-9a-fA-F]{7,64}", expected_commit) is None:
        raise ValueError(f"{label} commit must contain 7-64 hexadecimal characters")
    head = git_output(source_tree, "rev-parse", "HEAD").lower()
    if not head.startswith(expected_commit.lower()):
        raise ValueError(f"{label} source tree HEAD does not match its declared commit")
    if git_output(source_tree, "status", "--porcelain"):
        raise ValueError(f"{label} source tree must be clean")
    return Target(label=label, source_tree=source_tree, commit=head)


def mac_host_profile() -> dict[str, Any]:
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        raise RuntimeError("mac-gate requires an Apple Silicon Mac")
    result = subprocess.run(
        ["sysctl", "-n", "machdep.cpu.brand_string"],
        check=False,
        capture_output=True,
        text=True,
    )
    brand = result.stdout.strip()
    if result.returncode or "M5" not in brand:
        raise RuntimeError("mac-gate requires an Apple M5 host")
    return {
        "os_family": "macOS",
        "architecture": "arm64",
        "chip_family": "Apple M5",
        "machine_identity": "redacted",
    }


def bootstrap_mean_ci(
    values: list[float],
    *,
    seed: int,
    iterations: int = 2_000,
) -> dict[str, float]:
    if not values:
        return {"low": 0.0, "high": 0.0}
    generator = random.Random(seed)
    means = [
        sum(generator.choice(values) for _ in values) / len(values)
        for _ in range(iterations)
    ]
    ordered = sorted(means)
    low_index = int((len(ordered) - 1) * 0.025)
    high_index = int((len(ordered) - 1) * 0.975)
    return {"low": ordered[low_index], "high": ordered[high_index]}


def redacted_artifact(result: dict[str, Any]) -> dict[str, Any]:
    redacted = json.loads(json.dumps(result))
    redacted.get("environment", {})["hostname"] = "redacted"
    runtime = redacted.get("provenance", {}).get("runtime", {})
    runtime["hostname"] = "redacted"
    return redacted


def write_raw_artifact(
    raw_directory: Path,
    *,
    repetition: int,
    concurrency: int,
    target: Target,
    result: dict[str, Any],
) -> dict[str, Any]:
    raw_directory.mkdir(parents=True, exist_ok=True)
    name = f"rep-{repetition}-c{concurrency}-{target.label}.json"
    payload = (json.dumps(redacted_artifact(result), indent=2, sort_keys=True) + "\n").encode()
    path = raw_directory / name
    path.write_bytes(payload)
    return {
        "file": name,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "bytes": len(payload),
    }


def numeric_counter_summary(result: dict[str, Any]) -> dict[str, Any]:
    workload = result["results"]["uniform"]
    prefix_counters: dict[str, float] = {}
    client_requests = 0
    tokenizer_prompt_tokens = 0
    metrics_snapshots = 0
    for snapshot in workload["phase_snapshots"].values():
        client_requests += int(snapshot["raw_sample_count"])
        tokenizer_prompt_tokens += sum(
            int(sample["prompt_tokens"] or 0) for sample in snapshot["samples"]
        )
        metrics = snapshot.get("synthetic_worker_metrics")
        if metrics is not None:
            metrics_snapshots += 2
            for worker_metrics in metrics["delta"].values():
                for name, value in worker_metrics.items():
                    prefix_counters[name] = (
                        prefix_counters.get(name, 0.0) + float(value)
                    )
    service = result.get("service_counters", {})
    return {
        "health_checks": prefix_counters.get(
            "flume_benchmark_synthetic_health_checks",
            "not_observed",
        ),
        "metrics_scrapes": prefix_counters.get(
            "flume_benchmark_synthetic_metrics_scrapes",
            "not_observed",
        ),
        "database_pack_rows": int(service.get("database_pack_rows", 0)),
        "router_route_rows": int(service.get("router_route_rows", 0)),
        "client_completion_requests": client_requests,
        "tokenizer_prompt_tokens": tokenizer_prompt_tokens,
        "prefix_cache_queries": prefix_counters.get(
            "flume_benchmark_synthetic_prefix_cache_queries",
            "not_observed",
        ),
        "prefix_cache_hits": prefix_counters.get(
            "flume_benchmark_synthetic_prefix_cache_hits",
            "not_observed",
        ),
    }


def extract_run(result: dict[str, Any]) -> dict[str, Any]:
    measured = result["results"]["uniform"]["phase_snapshots"]["measured"]
    if measured["raw_sample_count"] < 1_000:
        raise ValueError("gate raw artifact contains fewer than 1000 measured samples")
    for latency_name in ("ttft_ms", "e2e_ms"):
        if "p99" not in measured[latency_name]:
            raise ValueError(f"gate raw artifact is missing {latency_name} p99")
    all_phase_errors = sum(
        int(snapshot["errors"])
        for snapshot in result["results"]["uniform"]["phase_snapshots"].values()
    )
    return {
        "status": result["status"],
        "errors": all_phase_errors,
        "raw_sample_count": int(measured["raw_sample_count"]),
        "duration_seconds": float(measured["duration_seconds"]),
        "throughput_rps": float(measured["throughput_rps"]),
        "ttft_ms": measured["ttft_ms"],
        "e2e_ms": measured["e2e_ms"],
        "routes": measured["route_counts"],
        "samples": measured["samples"],
        "counters": numeric_counter_summary(result),
    }


def aggregate_runs(runs: list[dict[str, Any]]) -> dict[str, Any]:
    samples = [sample for run in runs for sample in run["samples"]]
    successful = [sample for sample in samples if sample["error"] is None]
    ttft = [
        float(sample["ttft_ms"])
        for sample in successful
        if sample["ttft_ms"] is not None
    ]
    e2e = [float(sample["e2e_ms"]) for sample in successful]
    duration = sum(float(run["duration_seconds"]) for run in runs)
    routes: Counter[str] = Counter()
    counters: dict[str, float | str] = {}
    for run in runs:
        routes.update(run["routes"])
        for name, value in run["counters"].items():
            if isinstance(value, int | float):
                previous = counters.get(name, 0.0)
                if isinstance(previous, int | float):
                    counters[name] = float(previous) + float(value)
            else:
                counters.setdefault(name, "not_observed")
    return {
        "raw_sample_count": len(samples),
        "duration_seconds": duration,
        "throughput_rps": len(samples) / duration if duration else 0.0,
        "errors": sum(int(run["errors"]) for run in runs),
        "ttft_ms": latency_summary(ttft),
        "e2e_ms": latency_summary(e2e),
        "routes": dict(routes),
        "counters": counters,
    }


def paired_comparison(
    reference: dict[str, Any],
    target: dict[str, Any],
    *,
    thresholds: dict[str, float],
) -> dict[str, Any]:
    ratios = {
        "throughput": target["throughput_rps"] / reference["throughput_rps"],
        "ttft_p99": target["ttft_ms"]["p99"] / reference["ttft_ms"]["p99"],
        "e2e_p99": target["e2e_ms"]["p99"] / reference["e2e_ms"]["p99"],
    }
    zero_errors = reference["errors"] == 0 and target["errors"] == 0
    passed = (
        zero_errors
        and ratios["throughput"] >= thresholds["min_throughput_ratio"]
        and ratios["ttft_p99"] <= thresholds["max_ttft_p99_ratio"]
        and ratios["e2e_p99"] <= thresholds["max_e2e_p99_ratio"]
    )
    return {"ratios": ratios, "zero_errors": zero_errors, "passed": passed}


async def run_target_once(
    target: Target,
    *,
    seed: int,
    concurrency: int,
    warmups: int,
    samples: int,
    timeout: float,
    startup_timeout: float,
) -> dict[str, Any]:
    return await run_local_benchmark(
        Namespace(
            timeout=timeout,
            startup_timeout=startup_timeout,
            samples=samples,
            warmups=warmups,
            concurrency=concurrency,
            worker_delay_ms=25.0,
            workload=["uniform"],
            seed=seed,
            source_tree=target.source_tree,
            tested_commit=target.commit,
        )
    )


async def run_mac_gate(args: Namespace) -> dict[str, Any]:
    if args.warmups < 25 or args.samples < 1_000:
        raise ValueError("mac-gate requires at least 25 warmups and 1000 measured samples")
    if tuple(args.concurrency) != REQUIRED_CONCURRENCY:
        raise ValueError("mac-gate concurrency cells must be exactly 1, 8, and 32")
    if len(args.seed) != 3 or len(set(args.seed)) != 3:
        raise ValueError("mac-gate requires exactly three unique seeds")

    host = mac_host_profile()
    reference = resolve_target("reference", args.reference_source, args.reference_commit)
    target = resolve_target("target", args.target_source, args.target_commit)
    thresholds = {
        "min_throughput_ratio": args.min_throughput_ratio,
        "max_ttft_p99_ratio": args.max_ttft_p99_ratio,
        "max_e2e_p99_ratio": args.max_e2e_p99_ratio,
    }
    raw_directory = args.output.parent / f"{args.output.stem}-raw"
    cells: dict[str, Any] = {}
    all_raw_artifacts: list[dict[str, Any]] = []
    overall_passed = True

    for concurrency in args.concurrency:
        repetition_rows: list[dict[str, Any]] = []
        runs_by_label: dict[str, list[dict[str, Any]]] = {
            "reference": [],
            "target": [],
        }
        ratios: dict[str, list[float]] = {
            "throughput": [],
            "ttft_p99": [],
            "e2e_p99": [],
        }
        for repetition, seed in enumerate(args.seed, start=1):
            order = [reference, target]
            random.Random(seed + concurrency).shuffle(order)
            paired_runs: dict[str, dict[str, Any]] = {}
            artifacts: dict[str, dict[str, Any]] = {}
            for selected in order:
                raw_result = await run_target_once(
                    selected,
                    seed=seed,
                    concurrency=concurrency,
                    warmups=args.warmups,
                    samples=args.samples,
                    timeout=args.timeout,
                    startup_timeout=args.startup_timeout,
                )
                artifact = write_raw_artifact(
                    raw_directory,
                    repetition=repetition,
                    concurrency=concurrency,
                    target=selected,
                    result=raw_result,
                )
                artifacts[selected.label] = artifact
                all_raw_artifacts.append(artifact)
                paired_runs[selected.label] = extract_run(raw_result)
                runs_by_label[selected.label].append(paired_runs[selected.label])
            comparison = paired_comparison(
                paired_runs["reference"],
                paired_runs["target"],
                thresholds=thresholds,
            )
            for name, value in comparison["ratios"].items():
                ratios[name].append(value)
            repetition_rows.append(
                {
                    "repetition": repetition,
                    "seed": seed,
                    "order": [selected.label for selected in order],
                    "artifacts": artifacts,
                    "comparison": comparison,
                }
            )

        zero_errors = all(
            run["errors"] == 0
            for target_runs in runs_by_label.values()
            for run in target_runs
        )
        passing_repetitions = sum(
            row["comparison"]["passed"] for row in repetition_rows
        )
        cell_passed = zero_errors and passing_repetitions >= 2
        overall_passed = overall_passed and cell_passed
        cells[f"concurrency_{concurrency}"] = {
            "passed": cell_passed,
            "passing_repetitions": passing_repetitions,
            "required_passing_repetitions": 2,
            "zero_errors": zero_errors,
            "repetitions": repetition_rows,
            "aggregate": {
                "reference": aggregate_runs(runs_by_label["reference"]),
                "target": aggregate_runs(runs_by_label["target"]),
            },
            "bootstrap_ci": {
                name: bootstrap_mean_ci(
                    values,
                    seed=args.seed[0] + concurrency + index,
                )
                for index, (name, values) in enumerate(ratios.items())
            },
        }

    return {
        "schema_version": SCHEMA_VERSION,
        "benchmark": "mac_m5_paired_gate",
        "status": "completed" if overall_passed else "failed",
        "gpu_validated": False,
        "host": host,
        "package_under_test": {
            "reference_commit": reference.commit,
            "target_commit": target.commit,
        },
        "configuration": {
            "seeds": list(args.seed),
            "concurrency": list(args.concurrency),
            "warmups": args.warmups,
            "measured_samples": args.samples,
            "workload": "uniform",
            "paired_order": "seeded_randomized",
            "bootstrap_iterations": 2_000,
            "thresholds": thresholds,
        },
        "raw_artifacts": all_raw_artifacts,
        "cells": cells,
    }


def markdown_report(result: dict[str, Any]) -> str:
    lines = [
        "# Flume Apple M5 paired performance gate",
        "",
        f"- Status: `{result['status']}`",
        "- GPU validated: `false`",
        "- Machine identity: `redacted`",
        "",
        "| Cell | Passing pairs | Zero errors | Throughput target/reference | "
        "TTFT p99 target/reference | E2E p99 target/reference | Result |",
        "| --- | ---: | --- | ---: | ---: | ---: | --- |",
    ]
    for cell, measurement in result["cells"].items():
        reference = measurement["aggregate"]["reference"]
        target = measurement["aggregate"]["target"]
        lines.append(
            f"| {cell} | {measurement['passing_repetitions']}/3 | "
            f"{str(measurement['zero_errors']).lower()} | "
            f"{target['throughput_rps'] / reference['throughput_rps']:.3f} | "
            f"{target['ttft_ms']['p99'] / reference['ttft_ms']['p99']:.3f} | "
            f"{target['e2e_ms']['p99'] / reference['e2e_ms']['p99']:.3f} | "
            f"{'pass' if measurement['passed'] else 'fail'} |"
        )
    lines.append("")
    return "\n".join(lines)

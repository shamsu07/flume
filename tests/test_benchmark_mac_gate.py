import hashlib
import json
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import pytest

import flume.benchmark_mac_gate as gate
from flume.benchmark_mac_gate import (
    Target,
    bootstrap_mean_ci,
    git_output,
    mac_host_profile,
    markdown_report,
    paired_comparison,
    resolve_target,
    run_mac_gate,
)


def raw_result(*, commit: str, throughput: float = 100.0) -> dict[str, object]:
    measured_sample = {
        "index": 0,
        "phase": "measured",
        "status_code": 200,
        "worker_id": "route-a",
        "prompt_tokens": 100,
        "output_tokens": 1,
        "ttft_ms": 1.0,
        "e2e_ms": 2.0,
        "error": None,
    }
    metric_delta = {
        "worker": {
            "flume_benchmark_synthetic_health_checks": 2.0,
            "flume_benchmark_synthetic_metrics_scrapes": 2.0,
            "flume_benchmark_synthetic_prefix_cache_queries": 1_000.0,
            "flume_benchmark_synthetic_prefix_cache_hits": 999.0,
        }
    }

    def phase(count: int, samples: list[dict[str, object]]) -> dict[str, object]:
        return {
            "raw_sample_count": count,
            "duration_seconds": count / throughput if count else 0.0,
            "throughput_rps": throughput if count else 0.0,
            "errors": 0,
            "ttft_ms": {"p50": 1.0, "p95": 1.0, "p99": 1.0},
            "e2e_ms": {"p50": 2.0, "p95": 2.0, "p99": 2.0},
            "route_counts": {"route-a": count},
            "samples": samples,
            "synthetic_worker_metrics": {
                "before": {},
                "after": {},
                "delta": metric_delta,
            },
        }

    measured_samples = [
        {**measured_sample, "index": index}
        for index in range(1_000)
    ]
    return {
        "status": "completed",
        "environment": {"hostname": "secret-host"},
        "provenance": {
            "runtime": {"hostname": "secret-host"},
            "package_under_test": {"commit": commit},
        },
        "service_counters": {
            "database_pack_rows": 10,
            "router_route_rows": 10,
        },
        "results": {
            "uniform": {
                "phase_snapshots": {
                    "cold": phase(1, [{**measured_sample, "phase": "cold"}]),
                    "setup": phase(1, [{**measured_sample, "phase": "setup"}]),
                    "warmup": phase(25, [{**measured_sample, "phase": "warmup"}] * 25),
                    "measured": phase(1_000, measured_samples),
                }
            }
        },
    }


def test_bootstrap_and_threshold_logic_are_deterministic() -> None:
    first = bootstrap_mean_ci([0.9, 1.0, 1.1], seed=7)
    second = bootstrap_mean_ci([0.9, 1.0, 1.1], seed=7)
    assert first == second
    assert bootstrap_mean_ci([], seed=7) == {"low": 0.0, "high": 0.0}

    comparison = paired_comparison(
        {
            "throughput_rps": 100.0,
            "ttft_ms": {"p99": 10.0},
            "e2e_ms": {"p99": 20.0},
            "errors": 0,
        },
        {
            "throughput_rps": 95.0,
            "ttft_ms": {"p99": 10.5},
            "e2e_ms": {"p99": 21.0},
            "errors": 0,
        },
        thresholds={
            "min_throughput_ratio": 0.9,
            "max_ttft_p99_ratio": 1.1,
            "max_e2e_p99_ratio": 1.1,
        },
    )
    assert comparison["passed"] is True


def test_target_and_host_validation(monkeypatch, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="pyproject.toml"):
        resolve_target("missing", tmp_path / "missing", "abcdef0")
    (tmp_path / "pyproject.toml").write_text("[project]\nname='test'\n")
    with pytest.raises(ValueError, match="hexadecimal"):
        resolve_target("target", tmp_path, "not-a-commit")

    def fake_git(source: Path, *arguments: str) -> str:
        del source
        return "abcdef0123456789" if arguments[0] == "rev-parse" else ""

    monkeypatch.setattr(gate, "git_output", fake_git)
    target = resolve_target("target", tmp_path, "abcdef0")
    assert target.commit == "abcdef0123456789"
    with pytest.raises(ValueError, match="does not match"):
        resolve_target("target", tmp_path, "1234567")
    monkeypatch.setattr(
        gate,
        "git_output",
        lambda source, *arguments: (
            "abcdef0123456789" if arguments[0] == "rev-parse" else " M changed.py"
        ),
    )
    with pytest.raises(ValueError, match="must be clean"):
        resolve_target("target", tmp_path, "abcdef0")

    monkeypatch.setattr(gate.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(gate.platform, "machine", lambda: "arm64")
    monkeypatch.setattr(
        gate.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=0,
            stdout="Apple M5 Max\n",
        ),
    )
    assert mac_host_profile()["machine_identity"] == "redacted"
    monkeypatch.setattr(gate.platform, "system", lambda: "Linux")
    with pytest.raises(RuntimeError, match="Apple Silicon"):
        mac_host_profile()
    monkeypatch.setattr(gate.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(
        gate.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=0,
            stdout="Apple M4 Max\n",
        ),
    )
    with pytest.raises(RuntimeError, match="Apple M5"):
        mac_host_profile()

    monkeypatch.setattr(
        gate.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=0,
            stdout="value\n",
            stderr="",
        ),
    )
    assert git_output(tmp_path, "rev-parse", "HEAD") == "value"
    monkeypatch.setattr(
        gate.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=1,
            stdout="",
            stderr="not a repository",
        ),
    )
    with pytest.raises(ValueError, match="not a repository"):
        git_output(tmp_path, "status", "--porcelain")


@pytest.mark.asyncio
async def test_mac_gate_writes_checksummed_redacted_raw_artifacts(
    monkeypatch,
    tmp_path: Path,
) -> None:
    reference = Target("reference", tmp_path / "reference", "a" * 40)
    target = Target("target", tmp_path / "target", "b" * 40)
    monkeypatch.setattr(
        gate,
        "mac_host_profile",
        lambda: {
            "os_family": "macOS",
            "architecture": "arm64",
            "chip_family": "Apple M5",
            "machine_identity": "redacted",
        },
    )
    monkeypatch.setattr(
        gate,
        "resolve_target",
        lambda label, source, commit: reference if label == "reference" else target,
    )

    async def fake_run(selected: Target, **kwargs: object) -> dict[str, object]:
        del kwargs
        return raw_result(commit=selected.commit)

    monkeypatch.setattr(gate, "run_target_once", fake_run)
    output = tmp_path / "mac-gate.json"
    result = await run_mac_gate(
        Namespace(
            reference_source=reference.source_tree,
            reference_commit=reference.commit,
            target_source=target.source_tree,
            target_commit=target.commit,
            seed=[11, 22, 33],
            concurrency=[1, 8, 32],
            warmups=25,
            samples=1_000,
            timeout=10.0,
            startup_timeout=10.0,
            min_throughput_ratio=0.9,
            max_ttft_p99_ratio=1.1,
            max_e2e_p99_ratio=1.1,
            output=output,
        )
    )

    assert result["status"] == "completed"
    assert result["gpu_validated"] is False
    assert set(result["cells"]) == {
        "concurrency_1",
        "concurrency_8",
        "concurrency_32",
    }
    assert all(cell["passing_repetitions"] == 3 for cell in result["cells"].values())
    assert len(result["raw_artifacts"]) == 18
    for artifact in result["raw_artifacts"]:
        payload = (tmp_path / "mac-gate-raw" / artifact["file"]).read_bytes()
        assert hashlib.sha256(payload).hexdigest() == artifact["sha256"]
        assert json.loads(payload)["environment"]["hostname"] == "redacted"
    report = markdown_report(result)
    assert "concurrency_1" in report
    assert "3/3" in report

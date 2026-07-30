import io
import sys
from types import SimpleNamespace

import httpx
import pytest

import flume.benchmark_server as benchmark_server
from flume.benchmark_cli import local_markdown_report
from flume.benchmark_local import (
    LocalSample,
    ReadinessKind,
    WorkloadKind,
    build_workload_fixture,
    complete,
    is_port_collision,
    latency_summary,
    phase_snapshot,
    run_local_benchmark,
    start_server,
    wait_for_server,
)


@pytest.mark.asyncio
async def test_mock_worker_exposes_explicitly_synthetic_metrics() -> None:
    app = benchmark_server.create_mock_worker("worker-1")
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://worker",
    ) as client:
        assert (await client.get("/health")).status_code == 200
        initial_metrics = await client.get("/metrics")
        assert "flume_benchmark_synthetic_prefix_cache_queries 0" in initial_metrics.text

        control = await client.post("/benchmark/control", json={"delay_ms": 0})
        assert control.json() == {"delay_ms": 0.0}
        rejected = await client.post("/benchmark/control", json={"delay_ms": -1})
        assert rejected.status_code == 422

        completion = await client.post(
            "/v1/completions",
            json={"model": "model", "prompt": [1, 2], "max_tokens": 2},
        )
        assert completion.json()["usage"] == {
            "prompt_tokens": 2,
            "completion_tokens": 2,
            "total_tokens": 4,
        }
        assert "flume_benchmark_synthetic_prefix_cache_hits 0" in (
            await client.get("/metrics")
        ).text
        async with client.stream(
            "POST",
            "/v1/completions",
            json={"model": "model", "prompt": [1, 2], "max_tokens": 1, "stream": True},
        ) as stream:
            assert stream.headers["content-type"].startswith("text/event-stream")
            body = (await stream.aread()).decode()
        assert '"choices": [{"text": "x"' in body
        assert '"completion_tokens": 1' in body
        assert body.endswith("data: [DONE]\n\n")


@pytest.mark.asyncio
async def test_wait_for_server_requires_success_status() -> None:
    responses = iter(
        [
            httpx.Response(404),
            httpx.Response(503, json={"status": "ok"}),
            httpx.Response(204),
            httpx.Response(200, text="not-json"),
            httpx.Response(200, json={"status": "wrong"}),
            httpx.Response(200, json={"status": "ok"}),
        ]
    )

    async def handler(_: httpx.Request) -> httpx.Response:
        return next(responses)

    process = SimpleNamespace(poll=lambda: None, returncode=None)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        await wait_for_server(
            client,
            "http://server/ready",
            process,
            timeout=1.0,
            readiness=ReadinessKind.worker,
        )


@pytest.mark.asyncio
async def test_wait_for_server_reports_child_stderr() -> None:
    process = SimpleNamespace(
        poll=lambda: 1,
        returncode=1,
        stderr=io.BytesIO(b"address already in use"),
    )
    async with httpx.AsyncClient() as client:
        with pytest.raises(RuntimeError, match="address already in use"):
            await wait_for_server(
                client,
                "http://server/ready",
                process,
                timeout=1.0,
                readiness=ReadinessKind.worker,
            )
    assert is_port_collision("ERROR: [Errno 48] Address already in use")


@pytest.mark.asyncio
async def test_local_completion_rejects_incomplete_sse() -> None:
    async def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={
                "X-Flume-Worker-Id": "worker",
                "X-Flume-Prompt-Tokens": "12",
            },
            content=b'data: {"choices":[{"text":"x"}]}\n\n',
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        sample = await complete(
            client,
            "http://flume",
            "pack",
            phase="measured",
            index=0,
        )

    assert sample.ttft_ms is not None
    assert sample.error == "InvalidSSE: missing usage, done"


def test_benchmark_server_runners_build_apps(monkeypatch, tmp_path) -> None:
    calls: list[tuple[object, str, int]] = []

    def fake_run(app: object, *, host: str, port: int, **kwargs: object) -> None:
        del kwargs
        calls.append((app, host, port))

    monkeypatch.setattr(benchmark_server.uvicorn, "run", fake_run)
    benchmark_server.run_worker(
        SimpleNamespace(worker_id="worker-1", delay_ms=0.0, port=8101)
    )
    benchmark_server.run_flume(
        SimpleNamespace(
            port=8102,
            database_url=f"sqlite:///{tmp_path / 'benchmark.db'}",
            worker_url=["http://worker-1", "http://worker-2"],
        )
    )

    assert [(host, port) for _, host, port in calls] == [
        ("127.0.0.1", 8101),
        ("127.0.0.1", 8102),
    ]

    parsed = benchmark_server.build_parser().parse_args(
        ["worker", "--port", "8103", "--worker-id", "worker-3"]
    )
    assert parsed.worker_id == "worker-3"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "flume.benchmark_server",
            "worker",
            "--port",
            "8104",
            "--worker-id",
            "worker-4",
        ],
    )
    benchmark_server.main()
    assert calls[-1][2] == 8104


def test_external_harness_selects_target_source_tree(monkeypatch, tmp_path) -> None:
    seen: dict[str, object] = {}

    def fake_popen(command: list[str], **kwargs: object) -> SimpleNamespace:
        seen["command"] = command
        seen.update(kwargs)
        return SimpleNamespace()

    monkeypatch.setattr("flume.benchmark_local.subprocess.Popen", fake_popen)
    start_server("worker", "--port", "8101", source_tree=tmp_path)

    assert str(seen["command"][1]).endswith("benchmark_server.py")
    assert str(seen["env"]["PYTHONPATH"]).startswith(str(tmp_path / "src"))
    first = build_workload_fixture(WorkloadKind.uniform, 20, seed=7)
    second = build_workload_fixture(WorkloadKind.uniform, 20, seed=7)
    assert first == second


@pytest.mark.asyncio
async def test_process_isolated_local_benchmark_contract() -> None:
    result = await run_local_benchmark(
        SimpleNamespace(
            timeout=10.0,
            startup_timeout=10.0,
            samples=5,
            warmups=2,
            concurrency=2,
            worker_delay_ms=10.0,
            workload=[
                "uniform",
                "hot_80_20",
                "shuffled_equivalent",
                "worker_delay",
                "worker_failure",
                "all_unavailable",
            ],
        )
    )

    assert result["schema_version"] == 2
    assert result["status"] == "completed"
    assert result["validation"]["passed"] is True
    assert result["provenance"]["gpu_validated"] is False
    assert result["provenance"]["package_under_test"]["source"] in {
        "local_repo_head",
        "unknown",
    }
    assert result["provenance"]["topology"] == {
        "flume_processes": 1,
        "mock_worker_processes": 2,
        "process_isolated": True,
    }
    assert result["results"]["uniform"]["raw_sample_counts"] == {
        "cold": 5,
        "setup": 5,
        "warmup": 2,
        "measured": 5,
    }
    assert result["results"]["hot_80_20"]["request_distribution"] == {0: 4, 1: 1}
    assert result["results"]["shuffled_equivalent"]["equivalent_pack_id"] is True
    assert result["results"]["worker_failure"]["failover_observed"] is True
    unavailable = result["results"]["all_unavailable"]["phase_snapshots"]["measured"]
    assert unavailable["samples"][0]["status_code"] == 503
    assert set(result["phase_snapshots"]) == {"cold", "setup", "warmup", "measured"}
    workload_pack_ids = [
        set(result["results"][workload]["pack_ids"])
        for workload in ("uniform", "hot_80_20", "shuffled_equivalent")
    ]
    assert not (workload_pack_ids[0] & workload_pack_ids[1])
    assert not (workload_pack_ids[0] & workload_pack_ids[2])
    assert not (workload_pack_ids[1] & workload_pack_ids[2])
    for phase in result["results"]["uniform"]["phase_snapshots"].values():
        assert {
            "raw_sample_count",
            "duration_seconds",
            "throughput_rps",
            "errors",
            "ttft_ms",
            "e2e_ms",
            "route_counts",
            "samples",
            "synthetic_worker_metrics",
        } <= set(phase)
        assert set(phase["ttft_ms"]) == {"p50", "p95"}
        assert set(phase["e2e_ms"]) == {"p50", "p95"}
    measured_sample = result["results"]["uniform"]["phase_snapshots"]["measured"][
        "samples"
    ][0]
    assert measured_sample["ttft_ms"] is not None
    assert measured_sample["e2e_ms"] >= measured_sample["ttft_ms"]
    report = local_markdown_report(result)
    assert "req/s" in report
    assert "TTFT p50/p95" in report
    assert "E2E p50/p95" in report
    for workload in result["results"]:
        assert f"| {workload} |" in report


def test_phase_summary_only_reports_p99_with_enough_samples() -> None:
    sample = LocalSample(0, "measured", 200, "worker", 10, 1, 1.0, 2.0, None)
    snapshot = phase_snapshot([sample], duration_seconds=0.002)

    assert snapshot["throughput_rps"] == 500.0
    assert snapshot["route_counts"] == {"worker": 1}
    assert "p99" not in snapshot["ttft_ms"]
    assert latency_summary([1.0] * 1000)["p99"] == 1.0

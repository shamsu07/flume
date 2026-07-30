import json
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

import flume.benchmark_cli as benchmark
from flume.benchmark_cli import (
    Sample,
    SamplePlan,
    Scenario,
    build_sample_plans,
    choose_worker,
    environment_metadata,
    exact_prefix_tokens,
    execute_plans,
    execute_sample,
    latency_summary,
    line_has_token,
    markdown_report,
    metric_delta,
    metrics_snapshot,
    parse_args,
    parse_prometheus,
    percentile,
    run_benchmark,
    summarize_samples,
)


class FakeTokenizer:
    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        assert add_special_tokens is False
        return [ord(character) for character in text]


def test_exact_prefix_has_requested_token_length() -> None:
    tokens = exact_prefix_tokens(FakeTokenizer(), 4096)

    assert len(tokens) == 4096


def test_unstable_scenario_changes_prefix_but_not_length() -> None:
    plans = build_sample_plans(
        tokenizer=FakeTokenizer(),
        scenario=Scenario.unstable_prefix_random_workers,
        workers=["http://worker-1", "http://worker-2"],
        context_length=32,
        phase="warm",
        samples=2,
        max_output_tokens=8,
        run_salt="test",
        seed=1,
    )

    assert len(plans[0].prompt_token_ids) == len(plans[1].prompt_token_ids)
    assert plans[0].prompt_token_ids[:32] != plans[1].prompt_token_ids[:32]


def test_affinity_scenario_keeps_one_worker() -> None:
    plans = build_sample_plans(
        tokenizer=FakeTokenizer(),
        scenario=Scenario.stable_warmed_prefix_affinity,
        workers=["http://worker-1", "http://worker-2"],
        context_length=16,
        phase="warm",
        samples=20,
        max_output_tokens=8,
        run_salt="test",
        seed=1,
    )

    assert len({plan.worker_url for plan in plans}) == 1
    assert len({tuple(plan.prompt_token_ids[:16]) for plan in plans}) == 1


def test_p99_is_only_published_with_enough_samples() -> None:
    assert "p99" not in latency_summary([1.0] * 999)
    assert latency_summary([1.0] * 1000)["p99"] == 1.0


def test_prometheus_parser_sums_labeled_series() -> None:
    parsed = parse_prometheus(
        """
# HELP vllm:prefix_cache_hits Prefix cache hits
vllm:prefix_cache_hits{model_name="one"} 2
vllm:prefix_cache_hits{model_name="two"} 3
vllm:prefix_cache_queries 8
ignored_metric 99
"""
    )

    assert parsed == {
        "vllm:prefix_cache_hits": 5.0,
        "vllm:prefix_cache_queries": 8.0,
    }


@pytest.mark.asyncio
async def test_stream_sample_sends_integer_tokens_and_records_ttft() -> None:
    seen: dict[str, object] = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        seen.update(json.loads(request.content))
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=(
                b'data: {"choices":[{"text":"x"}]}\n\n'
                b'data: {"choices":[],"usage":{"completion_tokens":1}}\n\n'
                b"data: [DONE]\n\n"
            ),
        )

    plan = SamplePlan(
        index=0,
        phase="warm",
        worker_url="http://worker",
        prompt_token_ids=[1, 2, 3],
        max_tokens=1,
        cache_salt="isolated",
    )
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        sample = await execute_sample(client, plan, model="model")

    assert seen["prompt"] == [1, 2, 3]
    assert seen["cache_salt"] == "isolated"
    assert sample.ttft_ms is not None
    assert sample.output_tokens == 1
    assert sample.error is None


def test_percentiles_and_tokenizer_failures() -> None:
    assert percentile([], 0.5) == 0.0
    assert percentile([3.0], 0.5) == 3.0
    assert percentile([1.0, 3.0], 0.5) == 2.0

    class EmptyTokenizer:
        def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
            return []

    with pytest.raises(ValueError, match="positive"):
        exact_prefix_tokens(FakeTokenizer(), 0)
    with pytest.raises(ValueError, match="seed"):
        exact_prefix_tokens(EmptyTokenizer(), 2)
    with pytest.raises(ValueError, match="question"):
        benchmark.question_tokens(EmptyTokenizer(), 0)
    with pytest.raises(ValueError, match="unstable"):
        benchmark.unstable_prefix(EmptyTokenizer(), [1], 0)


def test_worker_selection_and_salt_policies() -> None:
    workers = ["http://one", "http://two"]
    affinity = choose_worker(
        Scenario.stable_warmed_prefix_affinity,
        workers,
        index=1,
        affinity_identity="pack",
        seed=4,
    )
    assert affinity in workers
    assert (
        choose_worker(
            Scenario.stable_warmed_prefix_affinity,
            workers,
            index=99,
            affinity_identity="pack",
            seed=100,
        )
        == affinity
    )
    cold = build_sample_plans(
        tokenizer=FakeTokenizer(),
        scenario=Scenario.stable_prefix_random_workers,
        workers=workers,
        context_length=8,
        phase="cold",
        samples=2,
        max_output_tokens=2,
        run_salt="run",
        seed=1,
    )
    warm = build_sample_plans(
        tokenizer=FakeTokenizer(),
        scenario=Scenario.stable_prefix_random_workers,
        workers=workers,
        context_length=8,
        phase="warm",
        samples=2,
        max_output_tokens=2,
        run_salt="run",
        seed=1,
    )
    assert cold[0].cache_salt != cold[1].cache_salt
    assert warm[0].cache_salt == warm[1].cache_salt


@pytest.mark.parametrize(
    ("line", "expected"),
    [
        ("event: token", False),
        ("data:", False),
        ("data: [DONE]", False),
        ("data: {bad", False),
        ('data: {"choices":[]}', False),
        ('data: {"choices":[{"text":"x"}]}', True),
    ],
)
def test_token_event_detection(line: str, expected: bool) -> None:
    assert line_has_token(line) is expected


@pytest.mark.asyncio
async def test_sample_errors_and_concurrent_execution() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        if "bad" in request.url.host:
            raise httpx.ConnectError("offline", request=request)
        return httpx.Response(
            200,
            content=b'data: {"choices":[{"text":"x"}]}\n\ndata: {bad\n\ndata: [DONE]\n\n',
        )

    plans = [
        SamplePlan(0, "warm", "http://good", [1], 1, "salt"),
        SamplePlan(1, "warm", "http://bad", [1], 1, "salt"),
    ]
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        samples, duration = await execute_plans(client, plans, model="model", concurrency=2)

    assert duration >= 0
    assert samples[0].ttft_ms is not None
    assert samples[1].status_code is None
    assert samples[1].error is not None


@pytest.mark.asyncio
async def test_metrics_snapshots_tolerate_unavailable_workers() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "bad":
            return httpx.Response(503)
        return httpx.Response(200, text="vllm:prefix_cache_hits 4\ninvalid\nx nope\n")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        snapshot = await metrics_snapshot(client, ["http://good", "http://bad"])

    assert snapshot["http://good"] == {"vllm:prefix_cache_hits": 4.0}
    assert snapshot["http://bad"] == {}
    assert metric_delta(
        {"one": {"hits": 2.0}},
        {"one": {"hits": 5.0}, "two": {"queries": 1.0}},
    ) == {"one": {"hits": 3.0}, "two": {"queries": 1.0}}


def test_sample_summary_includes_raw_measurements() -> None:
    samples = [
        Sample(0, "warm", "worker", 1, 10, 1, 2.0, 4.0, 200, None),
        Sample(1, "warm", "worker", 1, 10, None, None, 6.0, None, "offline"),
    ]

    result = summarize_samples(samples, 2.0)

    assert result["requests"] == 2
    assert result["successful_requests"] == 1
    assert result["errors"] == 1
    assert result["throughput_rps"] == 1.0
    assert result["output_tokens_per_second"] == 0.5
    assert result["samples"][1]["error"] == "offline"
    assert summarize_samples([], 0.0)["throughput_rps"] == 0.0


def benchmark_args() -> Namespace:
    return Namespace(
        tokenizer="tokenizer",
        tokenizer_revision="revision",
        allow_remote_tokenizer=False,
        scenario=[scenario.value for scenario in Scenario],
        model="model",
        vllm_revision="v0.22.0",
        apc_worker=["http://apc"],
        apc_disabled_worker=[],
        context_length=[8],
        concurrency=[2],
        warmups=1,
        warm_samples=2,
        cold_samples=1,
        max_output_tokens=2,
        seed=1,
        timeout=2.0,
        collect_metrics=True,
    )


@pytest.mark.asyncio
async def test_full_benchmark_orchestration_and_report(monkeypatch) -> None:
    monkeypatch.setattr(benchmark, "load_tokenizer", lambda *args: FakeTokenizer())
    monkeypatch.setattr(benchmark, "environment_metadata", lambda: {"commit": "abc"})

    async def fake_snapshot(*args) -> dict[str, dict[str, float]]:
        return {"http://apc": {"prefix_cache_hits": 1.0}}

    async def fake_execute(
        client: object,
        plans: list[SamplePlan],
        *,
        model: str,
        concurrency: int,
    ) -> tuple[list[Sample], float]:
        del client, model, concurrency
        return (
            [
                Sample(
                    plan.index,
                    plan.phase,
                    plan.worker_url,
                    plan.max_tokens,
                    len(plan.prompt_token_ids),
                    1,
                    1.0,
                    2.0,
                    200,
                    None,
                )
                for plan in plans
            ],
            0.5,
        )

    monkeypatch.setattr(benchmark, "metrics_snapshot", fake_snapshot)
    monkeypatch.setattr(benchmark, "execute_plans", fake_execute)

    result = await run_benchmark(benchmark_args())
    report = markdown_report(result)

    assert result["status"] == "completed"
    assert result["skipped"][Scenario.apc_disabled] == "no matching worker pool configured"
    assert "stable_warmed_prefix_affinity" in result["results"]
    assert "Commit: `abc`" in report
    assert "Skipped scenarios" in report
    assert "| warm |" in report


def test_environment_metadata_and_tokenizer_loader(monkeypatch) -> None:
    monkeypatch.setattr(benchmark.platform, "platform", lambda: "test-platform")
    monkeypatch.setattr(benchmark.platform, "node", lambda: "test-host")
    monkeypatch.setattr(benchmark.platform, "machine", lambda: "arm64")
    monkeypatch.setattr(
        benchmark.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(stdout="deadbeef\n"),
    )

    assert environment_metadata()["commit"] == "deadbeef"

    seen: dict[str, object] = {}

    class AutoTokenizer:
        @staticmethod
        def from_pretrained(tokenizer_id: str, **kwargs: object) -> FakeTokenizer:
            seen.update(tokenizer_id=tokenizer_id, **kwargs)
            return FakeTokenizer()

    monkeypatch.setattr("transformers.AutoTokenizer", AutoTokenizer)
    loaded = benchmark.load_tokenizer("tokenizer", "revision", False)
    assert isinstance(loaded, FakeTokenizer)
    assert seen["local_files_only"] is True
    assert seen["trust_remote_code"] is False


def test_parser_defaults_and_validation() -> None:
    args = parse_args(
        [
            "--model",
            "model",
            "--tokenizer",
            "tokenizer",
            "--tokenizer-revision",
            "revision",
            "--vllm-revision",
            "v0.22.0",
            "--apc-worker",
            "http://worker",
        ]
    )
    assert args.context_length == [4096, 16384, 65536]
    assert args.concurrency == [1, 8, 32]
    assert len(args.scenario) == 4

    base = [
        "--model",
        "model",
        "--tokenizer",
        "tokenizer",
        "--tokenizer-revision",
        "revision",
        "--vllm-revision",
        "v0.22.0",
    ]
    with pytest.raises(SystemExit):
        parse_args(base)
    with pytest.raises(SystemExit):
        parse_args([*base, "--apc-worker", "worker", "--warm-samples", "0"])
    with pytest.raises(SystemExit):
        parse_args([*base, "--apc-worker", "worker", "--max-output-tokens", "9"])


def test_main_writes_json_and_markdown(monkeypatch, tmp_path: Path, capsys) -> None:
    output = tmp_path / "nested" / "results.json"
    args = SimpleNamespace(output=output)
    result = {
        "environment": {"commit": "abc"},
        "configuration": {
            "model": "model",
            "tokenizer": "tokenizer",
            "tokenizer_revision": "revision",
            "vllm_revision": "v0.22.0",
            "apc_workers": [],
            "apc_disabled_workers": [],
        },
        "results": {},
        "skipped": {},
    }

    async def fake_run(_: object) -> dict[str, object]:
        return result

    monkeypatch.setattr(benchmark, "parse_args", lambda: args)
    monkeypatch.setattr(benchmark, "run_benchmark", fake_run)

    benchmark.main()

    assert json.loads(output.read_text())["environment"]["commit"] == "abc"
    assert output.with_suffix(".md").read_text().startswith("# Flume GPU benchmark")
    assert "wrote" in capsys.readouterr().out

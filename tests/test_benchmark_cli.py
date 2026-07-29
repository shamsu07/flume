import json

import httpx
import pytest

from flume.benchmark_cli import (
    SamplePlan,
    Scenario,
    build_sample_plans,
    exact_prefix_tokens,
    execute_sample,
    latency_summary,
    parse_prometheus,
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

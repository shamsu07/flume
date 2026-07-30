from benchmarks.proxy_overhead import percentile, summarize


def test_percentile_interpolates() -> None:
    assert percentile([1.0, 2.0, 3.0], 0.50) == 2.0
    assert percentile([1.0, 2.0, 3.0], 0.95) == 2.9


def test_summarize_includes_errors_in_throughput() -> None:
    result = summarize([1.0, 2.0, 3.0], errors=1, duration_seconds=2.0)

    assert result.requests == 4
    assert result.errors == 1
    assert result.throughput_rps == 2.0
    assert result.p99_ms > result.p95_ms

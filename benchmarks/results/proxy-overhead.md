# Flume local proxy overhead

This run uses a deterministic local mock vLLM and the Flume application from the
recorded checkout. It is a proxy-path benchmark, not a GPU inference result.

- Commit: `1f4315919019630b552c694b25c74b26a1524f99` (dirty during capture: `false`)
- Host: `macpro` / `arm64`
- Python: `3.12.13`
- Requests: `1000` at concurrency `32`
- Gate: **PASS**

| Path | Throughput (req/s) | p50 (ms) | p95 (ms) | p99 (ms) | Errors |
| --- | ---: | ---: | ---: | ---: | ---: |
| Direct mock vLLM (diagnostic) | 651.18 | 35.758 | 137.780 | 230.637 | 0 |
| Minimal pooled proxy (gate reference) | 288.30 | 86.700 | 231.581 | 320.432 | 0 |
| Through Flume | 280.42 | 99.388 | 198.004 | 257.115 | 0 |

Incremental p99 over direct proxying: **-63.317 ms**. Flume/reference throughput ratio: **0.973**.

## Committed-baseline comparison

Baseline source: `benchmarks/results/proxy-overhead-baseline.json` at commit `5b3dd47a9320af1ad267c5f92fe6683eb1e59d48`.

The legacy baseline compared Flume directly with the upstream, so this section
uses the current direct-upstream diagnostic for compatibility; release gates use
the fair two-hop reference proxy above.

| Metric | Committed baseline | Current | Change |
| --- | ---: | ---: | ---: |
| Incremental p99 overhead | 918.828 ms | 26.479 ms | 97.12% lower |
| Proxy/direct throughput ratio | 0.180 | 0.431 | 2.39x |
| Health probes during measured window | 1000 | 0 | 100.00% lower |

## Measured hot-path instrumentation

- Mock completions: `3000`
- Health probes: `0`
- Database statements/writes: `0` / `0`
- Upstream client instances observed: `1` (reused: `true`)

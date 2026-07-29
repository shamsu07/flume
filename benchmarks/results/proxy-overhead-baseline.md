# Flume local proxy overhead baseline

This run uses a deterministic local mock vLLM and the Flume application from the
recorded commit. It is a proxy-path benchmark, not a GPU inference result.

- Commit: `5b3dd47a9320af1ad267c5f92fe6683eb1e59d48` (dirty during capture: `true`)
- Host: `macpro` / `arm64`
- Python: `3.12.13`
- Requests: `1000` at concurrency `32`

| Path | Throughput (req/s) | p50 (ms) | p95 (ms) | p99 (ms) | Errors |
| --- | ---: | ---: | ---: | ---: | ---: |
| Direct mock vLLM | 637.81 | 36.907 | 134.665 | 205.024 | 0 |
| Through Flume | 114.88 | 195.464 | 566.140 | 1123.852 | 0 |

Incremental p99 overhead: **918.828 ms**. Proxy/direct throughput ratio: **0.180**.

The mock upstream counters make per-request health probes visible. This baseline
is intentionally retained so the optimized runtime can be compared on the same host.

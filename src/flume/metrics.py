from __future__ import annotations

from prometheus_client import Counter, Histogram

PACKS_CREATED = Counter("flume_packs_created_total", "Context packs created", ["tenant_id"])
ASKS_TOTAL = Counter("flume_asks_total", "Ask requests handled", ["worker_url", "stream"])
WARMUPS_TOTAL = Counter("flume_warmups_total", "Warmup requests handled", ["worker_url", "status"])
ROUTER_AFFINITY = Counter("flume_router_affinity_total", "Router affinity decisions", ["result"])

ASK_LATENCY = Histogram(
    "flume_ask_latency_seconds",
    "End-to-end ask latency through Flume",
    buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60, 120),
)
TTFT = Histogram(
    "flume_ttft_seconds",
    "Time to first token for streamed responses",
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30),
)
WARMUP_LATENCY = Histogram(
    "flume_warmup_latency_seconds",
    "Warmup latency",
    buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60, 120),
)

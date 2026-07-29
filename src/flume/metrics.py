from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram

REQUESTS_TOTAL = Counter(
    "flume_http_requests_total",
    "HTTP requests handled by route and status class",
    ["route", "method", "status_class"],
)
REQUEST_LATENCY = Histogram(
    "flume_http_request_duration_seconds",
    "Time until response headers by route",
    ["route"],
    buckets=(0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5),
)
PACKS_CREATED = Counter("flume_packs_created_total", "Immutable context packs registered")
PACK_CACHE = Counter(
    "flume_pack_cache_total",
    "Pack hot-cache lookups",
    ["result"],
)
ASKS_TOTAL = Counter(
    "flume_completions_total",
    "Completion requests handled",
    ["worker_id", "stream", "outcome"],
)
WARMUPS_TOTAL = Counter(
    "flume_warmups_total",
    "Warmup requests handled",
    ["worker_id", "outcome"],
)
ROUTER_AFFINITY = Counter(
    "flume_router_affinity_total",
    "Rendezvous-router affinity decisions",
    ["result"],
)
ROUTER_DECISIONS = Counter(
    "flume_router_decisions_total",
    "Worker routing decisions by policy and decision class",
    ["policy", "decision"],
)
ROUTER_DECISION_LATENCY = Histogram(
    "flume_router_decision_duration_seconds",
    "Cached worker routing decision latency",
    ["policy"],
    buckets=(0.00001, 0.000025, 0.00005, 0.0001, 0.00025, 0.0005, 0.001, 0.0025, 0.005),
)
ROUTER_WORKER_LOAD = Gauge(
    "flume_router_worker_load",
    "Cached worker load by safe worker ID and fixed load source",
    ["worker_id", "source"],
)
ROUTER_WORKER_LOAD_FRESH = Gauge(
    "flume_router_worker_load_fresh",
    "Whether cached upstream worker load is within the configured stale window",
    ["worker_id"],
)
ROUTER_FAILOVERS = Counter(
    "flume_router_failovers_total",
    "Requests rerouted before an upstream response",
    ["operation"],
)
ROUTER_UNAVAILABLE = Counter(
    "flume_router_unavailable_total",
    "Requests rejected because the worker pool is unavailable",
)
OVERLOADS = Counter(
    "flume_overload_rejections_total",
    "Requests rejected by admission control",
)
IN_FLIGHT = Gauge(
    "flume_in_flight_requests",
    "Completion requests admitted and not yet released",
)

ASK_LATENCY = Histogram(
    "flume_completion_latency_seconds",
    "End-to-end completion latency through Flume",
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60, 120),
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

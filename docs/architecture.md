# Architecture

Flume is a single-instance control and proxy layer for a homogeneous pool of
vLLM workers serving one pinned model/tokenizer pair. It turns documents into an
immutable integer-token prefix, assigns each pack to a healthy worker, and
forwards OpenAI-compatible completions without re-encoding that prefix.

```text
authenticated private gateway
  | X-Flume-Tenant
  v
Flume /v1
  |-- deterministic compiler + pinned tokenizer
  |-- bounded in-memory pack/token caches
  |-- SQLite WAL metadata + migrations
  |-- health-aware rendezvous router
  |-- pooled streaming/non-streaming proxy
  `-- low-cardinality Prometheus metrics
         |
         v
homogeneous vLLM workers with APC enabled
```

Pack identity covers canonical content, compiler format, template digest,
tokenizer ID and immutable revision, and the compiled token prefix. Tags and
operational annotations are mutable but are not identity inputs.

The warm path is intentionally memory-only: it must not reload the tokenizer,
deserialize a prefix, probe a worker, or write SQLite for each completion.
Background health refresh and Prometheus counters keep those operations off the
request path.

## Routing policies

`hrw` is the default and always chooses the healthy worker with the highest
rendezvous score for a pack. The experimental `bounded_hrw` policy preserves
that worker as the primary while allowing a temporary spill when it is outside
the configured load bound.

Bounded routing refreshes vLLM running and waiting request gauges in the
background. Its effective load is:

`max(local in-flight, upstream running + upstream waiting) / capacity weight`

The primary is retained when it is within the configured slack of the global
minimum effective load. Otherwise, the router selects the highest-ranked
non-primary worker within that bound and holds that single spill target for the
configured window while it remains within the load bound. If any healthy
candidate lacks a fresh upstream snapshot, the decision falls back to pure HRW;
local load alone never causes a spill.

The request path performs no network I/O. Affinity state is bounded by both an
entry limit and an idle TTL, and worker metrics use hashed worker IDs rather
than URLs.

This release is not a distributed control plane. Run one Flume process with one
SQLite database. PostgreSQL, replicated Flume instances, multi-model discovery,
and Kubernetes coordination are future work.

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

This release is not a distributed control plane. Run one Flume process with one
SQLite database. PostgreSQL, replicated Flume instances, multi-model discovery,
and Kubernetes coordination are future work.

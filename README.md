# Flume

Flume is a RAG cache compiler and vLLM serving proxy. It is designed to help
repeated long-context RAG workloads reuse vLLM Automatic Prefix Caching by
turning retrieved documents into deterministic, versioned, cache-stable context
packs.

Flume does **not** replace vLLM's KV cache, LMCache, SGLang HiCache, vector
databases, or retrieval systems. Flume sits above vLLM and focuses on prompt
determinism, warmup, worker affinity, and observability.

## Why

In a common RAG workload, many requests repeatedly include the same long
document context:

```text
system prompt + retrieved document pack + user question
```

vLLM can reuse KV cache for shared prefixes, but applications can lose
those hits through unstable chunk ordering, drifting templates, routing across
replicas, or lack of warmup/metrics. Flume provides controls intended to make
that reuse deliberate; the repository does not yet contain a validated real-GPU
comparison proving an improvement.

## Architecture

```text
Client / SDK
    |
    v
Flume FastAPI proxy
    |-- Context pack compiler
    |-- SQLite metadata store
    |-- Pack-aware router
    |-- Prometheus metrics
    v
vLLM OpenAI-compatible workers with prefix caching enabled
```

The default router is deterministic health-aware HRW. The load-informed
`bounded_hrw` policy is experimental and disabled by default; enable it only
with `FLUME_ROUTING_POLICY=bounded_hrw`. It falls back to pure HRW whenever any
healthy candidate lacks a fresh vLLM load snapshot.

## Evidence limits

Current repository evidence is limited to local process-isolated mock workers.
A paired loopback methodology for a Mac is documented, but no numeric result is
asserted here. Local runs can validate harness accounting, failure handling,
topology, and proxy behavior. They do not validate CUDA/GPU execution, vLLM APC
hit-rate gains, production tail latency, throughput gains, or cost improvements.

Real GPU/APC comparisons remain unvalidated and benchmark provenance must remain
`gpu_validated=false` unless an operator runs and attests the documented GPU
matrix. Do not use the local evidence to claim that Flume is faster or otherwise
superior to direct vLLM, NVIDIA Dynamo, LMCache, or another cache/routing system.
See [Benchmarks](docs/benchmarks.md) for the paired methodology and required
provenance.

## Quickstart

Install locally:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e ".[dev]"
```

Start vLLM separately with prefix caching enabled:

```bash
python -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --enable-prefix-caching
```

Start Flume:

```bash
export FLUME_TOKENIZER_REVISION=<immutable-hugging-face-commit>
export FLUME_CACHE_SALT_SECRET=<high-entropy-secret>
flume serve \
  --vllm-worker http://localhost:8000 \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --tokenizer meta-llama/Llama-3.1-8B-Instruct
```

Register a pack:

```bash
flume pack create \
  --tenant demo \
  examples/policy.txt
```

Warm and ask:

```bash
flume warm --tenant demo <pack_id>
flume ask --tenant demo <pack_id> "What are the refund rules?"
```

The service owns the model, tokenizer, and immutable tokenizer revision.
Clients cannot override them while registering packs or completing prompts.
Production deployments should use the tenant-scoped `/v1` interface behind an
authenticated private gateway:

```bash
curl http://127.0.0.1:8080/v1/completions \
  -H 'Content-Type: application/json' \
  -H 'X-Flume-Tenant: demo' \
  -d '{"pack_id":"<pack_id>","prompt":"What are the refund rules?","max_tokens":64}'
```

See:

- [Architecture](docs/architecture.md)
- [Configuration](docs/configuration.md)
- [Deployment and security](docs/deployment-security.md)
- [Failure semantics](docs/failure-semantics.md)
- [Benchmarks](docs/benchmarks.md)
- [Changelog](CHANGELOG.md)

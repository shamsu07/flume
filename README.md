# Flume

Flume is a RAG cache compiler and vLLM serving proxy. It helps repeated
long-context RAG workloads get more value from vLLM Automatic Prefix Caching by
turning retrieved documents into deterministic, versioned, cache-stable context
packs.

Flume does **not** replace vLLM's KV cache, LMCache, SGLang HiCache, vector
databases, or retrieval systems. V0 sits above vLLM and focuses on prompt
determinism, warmup, worker affinity, and observability.

## Why

In a common RAG workload, many requests repeatedly include the same long
document context:

```text
system prompt + retrieved document pack + user question
```

vLLM can reuse KV cache for shared prefixes, but real applications often lose
those hits through unstable chunk ordering, drifting templates, routing across
replicas, or lack of warmup/metrics. Flume makes that reuse intentional.

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
flume serve --vllm-worker http://localhost:8000 --model meta-llama/Llama-3.1-8B-Instruct
```

Register a pack:

```bash
flume pack create \
  --tenant demo \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --tokenizer meta-llama/Llama-3.1-8B-Instruct \
  examples/policy.txt
```

Warm and ask:

```bash
flume warm <pack_id>
flume ask <pack_id> "What are the refund rules?"
```

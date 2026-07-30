# Benchmarks

Benchmarks are offline operator tools. The serving API does not run long-lived
benchmark jobs or persist benchmark rows.

## Local proxy overhead

The local harness starts a counted mock vLLM and, unless `--proxy-url` is given,
an embedded Flume process:

```bash
python benchmarks/proxy_overhead.py \
  --requests 1000 \
  --concurrency 32 \
  --warmup 25 \
  --output benchmarks/results/proxy-overhead.json
```

The acceptance gate is zero errors, incremental p99 proxy overhead at most 10 ms,
and proxy throughput at least 90% of direct. The upstream health counter must
not increase once per warm proxy request. Keep the committed unoptimized
baseline immutable; write later comparisons to a new file.

## GPU and APC scenarios

Provide distinct workers started with APC disabled and enabled. Pin every
revision in the result metadata:

```bash
flume-benchmark \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --tokenizer meta-llama/Llama-3.1-8B-Instruct \
  --tokenizer-revision <immutable-hugging-face-commit> \
  --vllm-revision v0.22.0 \
  --apc-disabled-worker http://127.0.0.1:8001 \
  --apc-worker http://127.0.0.1:8002 \
  --output benchmark-results.json
```

Defaults cover 4K/16K/64K exact integer-token prefixes, concurrency 1/8/32, five
discarded warmups, 100 measured warm samples, ten isolated cold samples, and
streaming outputs of 1–8 tokens. The scenarios are:

1. APC disabled on the explicitly configured disabled worker pool.
2. Unstable prefixes distributed across APC-enabled workers.
3. Stable prefixes distributed across APC-enabled workers.
4. Stable warmed prefixes held to one affinity worker.

Each cell records raw samples, route choice, errors, TTFT, end-to-end latency,
throughput, and optional vLLM prefix-cache query/hit deltas. p99 is emitted only
when a phase contains at least 1,000 successful samples. Compare runs only when
hardware, model/tokenizer/vLLM revisions, topology, context lengths, concurrency,
and cache state are equivalent.

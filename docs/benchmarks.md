# Benchmarks

Benchmarks are offline operator tools. The serving API does not run long-lived
benchmark jobs or persist benchmark rows.

## Apple M5 paired performance gate

`mac-gate` is a local-only comparison gate for clean source worktrees on an
Apple M5 host. One external harness verifies each declared commit against
worktree `HEAD`, then runs three seeded, randomized reference/target pairs at
concurrency 1, 8, and 32:

```bash
flume-benchmark mac-gate \
  --reference-source /path/to/reference-worktree \
  --reference-commit <reference-commit> \
  --target-source /path/to/target-worktree \
  --target-commit <target-commit> \
  --output benchmark-mac-gate.json
```

Each run uses at least 25 discarded warmups and 1,000 measured streaming
requests. Aggregation records p50/p95/p99 TTFT and E2E, throughput, errors,
routes, observed local counters, deterministic paired bootstrap intervals, and
SHA-256 checksums for every raw JSON artifact. A cell passes when at least two
of three paired repetitions meet the configured throughput and latency ratios,
and every paired run has zero errors. All three concurrency cells must pass.

Output records `gpu_validated=false`, redacts host identity, and makes no
measurement claim until the command is actually run. Raw artifacts are written
beside the aggregate in `<output-stem>-raw/`; measurement files should not be
committed.

## Process-isolated local workloads

The default local command starts one Flume process and exactly two mock-worker
processes, then exercises only the public pack, warmup, completion, and stats
endpoints:

```bash
flume-benchmark local \
  --samples 100 \
  --warmups 10 \
  --concurrency 8 \
  --output benchmark-local.json
```

It runs deterministic uniform, 80/20 hot-pack, shuffled-equivalent-document,
worker-delay, worker-failure, and all-workers-unavailable workloads. The mock
workers' cache counters are explicitly named and recorded as synthetic; they
validate benchmark accounting and topology, not GPU cache performance.
Completions stream through Flume, and each raw sample records TTFT and
end-to-end latency. Phase summaries separately report cold, explicit setup
warmups, discarded warmups, and measured traffic with duration, throughput,
errors, percentiles, route counts, and raw sample counts.
Local subprocess readiness requires an exact HTTP 200 plus the expected
endpoint-specific JSON. Ephemeral ports are reserved locally before each child
starts; the small bind race is bounded by three collision retries, so this
harness is intended for a single-host local benchmark rather than distributed
or adversarial multi-process orchestration.

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
flume-benchmark gpu \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --tokenizer meta-llama/Llama-3.1-8B-Instruct \
  --tokenizer-revision <immutable-hugging-face-commit> \
  --vllm-revision v0.22.0 \
  --tested-commit <installed-flume-git-commit> \
  --apc-disabled-worker http://127.0.0.1:8001 \
  --apc-worker http://127.0.0.1:8002 \
  --flume-url http://127.0.0.1:8080 \
  --output benchmark-results.json
```

The Flume URL must point to a separate Flume deployment configured with the
declared APC-enabled workers. The affinity scenario registers, warms, and
completes through Flume's public `/v1` API and records the worker id returned by
Flume; the harness does not reproduce Flume's private routing hash. The legacy
flat invocation remains accepted with a deprecation warning.
Cold Flume samples use isolated tenant identities and registrations. The
harness fails the run if Flume's returned compiled token count differs from the
requested context length.

Defaults cover 4K/16K/64K exact integer-token prefixes, concurrency 1/8/32, five
discarded warmups, 100 measured warm samples, ten isolated cold samples, and
streaming outputs of 1–8 tokens. The scenarios are:

1. APC disabled on the explicitly configured disabled worker pool.
2. Unstable prefixes distributed across APC-enabled workers.
3. Stable prefixes distributed across APC-enabled workers.
4. Stable warmed prefixes held to one affinity worker.

Each cell records raw samples, route choice, errors, TTFT, end-to-end latency,
throughput, and separate cold, discarded-warmup, and measured vLLM metric
snapshots. Schema-versioned provenance defaults to `gpu_validated=false`; pass
`--gpu-validated` only as an operator attestation that the declared workers are
the GPU environment being measured. The result labels that basis as
`operator_attested`; the harness does not claim to verify GPU hardware.
Runtime and build provenance comes from the benchmark process and its local Git
checkout. `--tested-commit` separately records the installed Flume package
revision under test; the working-directory Git state is never presented as
package provenance. A run with any requested scenario skipped is written with
`status=partial` and exits nonzero, while any sample error produces
`status=failed` and a nonzero exit. p99 is emitted only
when a phase contains at least 1,000 successful samples. Compare runs only when
hardware, model/tokenizer/vLLM revisions, topology, context lengths, concurrency,
and cache state are equivalent.

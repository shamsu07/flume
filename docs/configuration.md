# Configuration

Configuration uses `FLUME_` environment variables. Secrets should come from the
deployment secret manager, not an image or Compose file.

| Variable | Required | Meaning |
| --- | --- | --- |
| `FLUME_MODEL_ID` | yes | Server-authoritative vLLM model ID. |
| `FLUME_TOKENIZER_ID` | yes | Tokenizer repository or local path. |
| `FLUME_TOKENIZER_REVISION` | yes | Immutable tokenizer commit revision. |
| `FLUME_CACHE_SALT_SECRET` | yes | High-entropy secret used to derive tenant cache salts. |
| `FLUME_VLLM_WORKERS` | yes | Comma-separated homogeneous worker base URLs. |
| `FLUME_DATABASE_URL` | yes | Single-instance SQLite URL under a persistent volume. |
| `FLUME_REQUEST_TIMEOUT_SECONDS` | no | Upstream request timeout. |
| `FLUME_CONNECT_TIMEOUT_SECONDS` | no | Upstream connection timeout. |
| `FLUME_HEALTH_TIMEOUT_SECONDS` | no | Per-worker background health timeout. |
| `FLUME_ROUTING_POLICY` | no | `hrw` (default) or experimental `bounded_hrw`. |
| `FLUME_ROUTING_LOAD_SLACK` | no | Normalized load slack before spilling from the HRW primary; default `2`. |
| `FLUME_ROUTING_SPILL_HOLD_MS` | no | Minimum spill stickiness window; default `2000`. |
| `FLUME_WORKER_LOAD_REFRESH_MS` | no | Background vLLM load refresh interval; default `500`. |
| `FLUME_WORKER_LOAD_STALE_MS` | no | Age after which upstream load is ignored; default `2000`. |
| `FLUME_WORKER_CAPACITY_WEIGHTS` | no | JSON map of worker URL to positive capacity weight; omitted workers use `1.0`. |
| `FLUME_ROUTING_STATE_MAX_ENTRIES` | no | In-memory affinity state bound; default `10000`. |
| `FLUME_ROUTING_STATE_TTL_SECONDS` | no | Idle affinity-state TTL; default `600`. |
| `FLUME_MAX_PACK_TOKENS` | no | Maximum compiled prefix length. |
| `FLUME_MAX_REQUEST_BODY_BYTES` | no | Maximum accepted request body. |
| `FLUME_MAX_OUTPUT_TOKENS` | no | Maximum generated tokens per completion. |
| `FLUME_MAX_IN_FLIGHT` | no | Global in-flight completion bound. |
| `FLUME_PACK_CACHE_BYTES` | no | Byte bound for decoded in-memory packs. |
| `FLUME_HEALTH_REFRESH_SECONDS` | no | Background worker-health refresh interval. |
| `FLUME_SQLITE_BUSY_TIMEOUT_MS` | no | SQLite contention wait before failure. |
| `FLUME_METRICS_ENABLED` | no | Enable the Prometheus endpoint. |

Tokenizer downloads should be disabled in production after the pinned revision
is baked or mounted. Changing model, tokenizer, revision, compiler format, or
template invalidates existing packs and requires registration under the new
identity.

The service binds to `127.0.0.1` by default. Put it behind an authenticated
private gateway that overwrites, rather than merely forwards,
`X-Flume-Tenant`.

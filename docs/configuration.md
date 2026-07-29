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
| `FLUME_MAX_PACK_TOKENS` | no | Maximum compiled prefix length. |
| `FLUME_MAX_REQUEST_BYTES` | no | Maximum accepted request body. |
| `FLUME_MAX_IN_FLIGHT` | no | Global in-flight completion bound. |
| `FLUME_PACK_CACHE_BYTES` | no | Byte bound for decoded in-memory packs. |
| `FLUME_HEALTH_INTERVAL_SECONDS` | no | Background worker-health refresh interval. |
| `FLUME_METRICS_ENABLED` | no | Enable the Prometheus endpoint. |

Tokenizer downloads should be disabled in production after the pinned revision
is baked or mounted. Changing model, tokenizer, revision, compiler format, or
template invalidates existing packs and requires registration under the new
identity.

The service binds to `127.0.0.1` by default. Put it behind an authenticated
private gateway that overwrites, rather than merely forwards,
`X-Flume-Tenant`.

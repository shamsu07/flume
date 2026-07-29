# Failure semantics

Flume returns structured JSON errors before streaming starts. After SSE headers
have been sent, upstream failure ends the stream; clients must treat a stream
without `[DONE]` as incomplete.

| Status | Meaning | Retry guidance |
| --- | --- | --- |
| `400` | Missing/invalid tenant or malformed request. | Fix the request. |
| `404` | Pack does not exist for this tenant. | Do not retry unchanged. |
| `409` | Immutable pack registration conflicts with existing identity. | Reconcile configuration. |
| `413` | Request body exceeds the configured limit. | Reduce the request. |
| `422` | Invalid pack/template/token or reserved completion field. | Fix the request. |
| `429` | In-flight capacity is exhausted. | Retry with jitter/backoff. |
| `502` | Selected vLLM failed before a valid response. | Retry only when safe. |
| `503` | No healthy worker or service not ready. | Retry with backoff. |
| `504` | Upstream timeout before response. | Retry non-streaming idempotent work. |

Non-streaming requests may be retried once only when a connection fails before
any response. Streaming requests are never retried after streaming begins,
because doing so could duplicate output. Client cancellation closes the upstream
response promptly.

`/livez` means the process event loop is alive. `/readyz` additionally requires
the database, pinned tokenizer, and at least one healthy vLLM worker. A failed
readiness probe is not evidence that every in-flight request failed.

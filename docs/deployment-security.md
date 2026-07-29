# Deployment and security

Flume's v1 tenant boundary assumes a trusted private gateway authenticates the
caller and supplies an authoritative `X-Flume-Tenant` header. Do not expose the
service directly to untrusted networks. The gateway must delete any incoming
tenant header before setting its own.

Use a unique, high-entropy `FLUME_CACHE_SALT_SECRET`. Flume derives an unguessable
vLLM `cache_salt` from this secret and the tenant identity so KV-cache timing and
reuse cannot cross tenants. Rotate the secret during a maintenance window;
rotation intentionally cold-starts tenant prefix caches.

Operational requirements:

- Keep SQLite and its WAL on persistent local storage; do not place it on NFS.
- Run one Flume application process per database.
- Give Flume network access only to its gateway and vLLM workers.
- Terminate TLS at the private gateway or service mesh.
- Never log prompts, document text, questions, credentials, cache salts, or raw
  upstream response bodies.
- Protect `/metrics` at the network layer and keep label values low-cardinality.
- Set request, token, body, and in-flight limits for the available GPU capacity.
- Back up SQLite with its online backup mechanism, not by copying a live main
  file without its WAL.

The container runs as an unprivileged user, drops Linux capabilities in Compose,
uses a read-only root filesystem, and writes only `/data` and `/tmp`. Prometheus,
Grafana, vLLM, and Flume ports bind to loopback in the example Compose topology.

Before rollout, verify `/livez`, then `/readyz`. Remove an instance from service
when readiness fails. Send `SIGTERM` and allow the request drain period before
forcing termination.

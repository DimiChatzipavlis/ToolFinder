# Security Posture

ToolFinder is an MCP **routing bridge**: it embeds the tool catalogs of one or
more downstream MCP servers and forwards the agent's chosen call to the server
that owns the tool. It does **selection and dispatch only** — it does not execute
code itself, and routing needs no credentials (it is a local embedding model).
For OpenAPI downstreams it injects auth resolved from **environment variables** at
call time — never inlined in config or logged.

## What the bridge actually protects

| Property | How | Where |
| --- | --- | --- |
| No shell injection when spawning downstream servers | `create_subprocess_exec` with argument lists; never `shell=True` | `toolfinder/mcp_adapter.py` |
| No hang on a stalled/garbage downstream | bounded `stdin.drain()`, per-request timeouts, pending-request draining on shutdown | `toolfinder/mcp_adapter.py` |
| Speculative argument keys rejected | `additionalProperties: false` injected into every object schema at ingest | `toolfinder/dynamic_faiss_router.py` |
| Refuses out-of-scope requests | similarity threshold (`min_cosine_similarity`) — abstains instead of force-routing | `toolfinder/dynamic_faiss_router.py` |
| Survives a downstream crash | one-shot reconnect + retry on a failed call | `toolfinder/mcp_server.py` |
| No unsafe deserialization | weights load via safetensors (sentence-transformers); FAISS index is built in-process, never loaded from disk | — |
| Downstream API credentials kept out of config & logs | OpenAPI `auth` (bearer / API-key header or query) resolved from env vars at call time | `toolfinder/openapi_adapter.py` |

## What it relies on (not the bridge's job)

- **Downstream sandboxing.** Filesystem/git/etc. access limits are enforced by
  the downstream servers themselves (e.g. the filesystem server's allowed-root).
  The bridge does not add its own sandbox.
- **Trusted configuration.** `TOOLFINDER_CONFIG` lists the servers to spawn; only
  point it at servers you trust.

## Known residual risks (honest)

1. **No authentication on the bridge.** It is a local stdio MCP server; do not
   expose it over a network without an external auth layer.
2. **Tool-description poisoning.** A hostile downstream server could craft a tool
   description to attract unrelated queries. Mitigations (length cap, embedding
   anomaly score, reranking) were *studied* (archived under `legacy/experiments/`)
   but are **not implemented in the server** — treat untrusted downstream servers
   with caution.
3. **Tool-name collisions.** If two downstream servers expose the same tool name,
   `call_tool` resolves to the first; `route_and_call` is unambiguous (dispatches
   by the routed server).
4. **No persistence / single process.** The index is in-memory; there is no
   horizontal scaling or audit log beyond `get_stats()` and process logging.
5. **OpenAPI adapter makes outbound HTTP calls.** Pointing it at an untrusted
   spec or endpoint is an SSRF-style risk, and it does no response-schema
   validation — configure only OpenAPI servers you trust.

## Reporting

Open a GitHub issue for any finding. This is an early-stage OSS tool, not a
supported product.

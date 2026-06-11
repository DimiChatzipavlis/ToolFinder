# Security Posture

Threat model, implemented mitigations, and residual risks. Verified by code
sweep on 2026-06-12 (no `eval`/`exec`/`pickle`/`shell=True`/`os.system`/unsafe
YAML/`verify=False` anywhere; no hardcoded credentials — API keys are read from
environment variables only).

## Threat model

**Selection layer (the router).** A hostile or compromised MCP server can
publish tool descriptions crafted to attract unrelated queries
(description poisoning), or rely on ambiguous queries being force-routed to
destructive tools. Out-of-scope user requests must be rejected, not routed.

**Execution layer (the agent/runtime).** A model can emit malformed or
malicious arguments: path traversal, oversized payloads, repeated destructive
actions, or non-JSON output crafted to break parsing.

## Implemented mitigations

| Risk | Mitigation | Where |
| --- | --- | --- |
| Description poisoning | Measured attack + 3 mitigations (length cap, embedding-centroid anomaly score, cross-encoder rerank); see `experiments/results/poisoning.json` | `experiments/attacks/poisoning.py` |
| Force-routing of out-of-scope queries | Similarity threshold with measured operating points (risk-coverage, AUROC) | `toolfinder/dynamic_faiss_router.py`, `experiments/evaluation/ood.py` |
| Speculative/unknown arguments | `additionalProperties: false` injected into every object schema at ingest; arguments validated against the tool's JSON Schema before execution | router `_inject_additional_properties_false`, agent `jsonschema.validate` |
| Path traversal | Path-like arguments resolve via `realpath` anchored at `workspace_root` (never process cwd), strict containment + parent-segment denial + allowed-roots check; null bytes rejected | `Enterprise/runtime/policy.py` |
| Repeated/destructive duplicate actions | Canonicalized (sorted-key JSON) SHA256 action signatures in the ReAct loop; per-response dedup in the hybrid pipeline | `toolfinder/autonomous_agent.py`, `Enterprise/runtime/openclaw_hybrid_pipeline.py` |
| Malicious model output parsing | Strict JSON first; recovery limited to `ast.literal_eval` (literals only, no calls/imports) with nesting-depth guard against parser DoS | `toolfinder/utils.py` |
| Oversized payloads | Byte/string/collection limits on arguments | `Enterprise/runtime/policy.py` |
| Stdio backpressure DoS | Bounded `stdin.drain()` with request timeout; pending-request map drained on timeout/shutdown | `toolfinder/mcp_adapter.py` |
| Command injection at server spawn | `create_subprocess_exec` with argument lists everywhere; no `shell=True` in the codebase | `toolfinder/mcp_adapter.py`, `Enterprise/runtime/openclaw_backend.py` |
| Unsafe deserialization | No `pickle`/`torch.load` of untrusted files; model weights load via safetensors through sentence-transformers; FAISS indexes are built in-process, never loaded from disk in the runtime | — |
| Supply-chain drift of artifacts | SHA256 manifest of datasets, results, and model artifacts | `experiments/build_manifest.py` |
| SSRF-ish hangs in outbound HTTP | Explicit timeouts on every `httpx` client/request | `Enterprise/runtime/openclaw_backend.py`, `experiments/*` |

## Residual risks (known, accepted, documented)

1. **The HTTP API has no authentication.** `POST /execute`
   (`Enterprise/runtime/api.py`) is a local validation runtime. Bind it to
   `127.0.0.1` only; any network exposure requires an external auth/proxy
   layer, which is explicitly out of scope here.
2. **Error responses include exception text** (`execution failed: {exc}`),
   which can disclose internal paths to a caller. Acceptable for a local
   runtime; scrub before any exposure.
3. **A global similarity threshold cannot fully separate adversarial
   near-misses** (in-domain vocabulary, absent capability) from valid queries
   — quantified in `experiments/results/ood_eval.json`. Destructive tools
   should carry stricter per-tool margins and human confirmation.
4. **Poisoning mitigations are partial individually** (measured in
   `experiments/results/poisoning.json`); deploy the combination (length cap +
   anomaly screen + rerank) and treat server allowlisting as the primary
   control.
5. **Resource exhaustion via very large literal payloads** remains possible
   below the configured limits; limits are caps, not proofs.

## Reporting

This is a university research project, not a supported product. Open an issue
for any finding; do not expect a formal SLA.

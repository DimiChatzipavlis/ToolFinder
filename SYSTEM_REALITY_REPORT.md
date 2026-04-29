# SYSTEM_REALITY_REPORT

Date: 2026-04-07
Auditor: Principal Systems Auditor / Red Team Lead
Scope: `toolfinder/`, `Enterprise/`, `tests/`, `pyproject.toml`

## Executive Verdict
The architecture is hardened in multiple critical areas, but it is **not mathematically sound end-to-end** for duplicate action prevention and path traversal invariance under all policy configurations.

- Runtime tests: `pytest -q` completed with **42 passed**.
- No destructive or hidden bypass path was found that executes arbitrary Python code via `ast.literal_eval`.
- Two audit blockers remain:
  1. Duplicate tool-call prevention in `toolfinder/autonomous_agent.py` is hash-based but not canonicalized.
  2. `Enterprise/runtime/policy.py` path traversal prevention is policy-dependent, not invariant across all configuration states.

## Section 1: Production Runtime Mocks / Hardcoded Answers / Cheat Detection

### 1.1 Production runtime (`toolfinder/`, `Enterprise/runtime/`)
- No explicit mock client classes or synthetic "always success" stubs were found in production runtime directories.
- Mock implementations are present in non-production example paths (`Enterprise/examples/*`) and test files (`tests/*`).
- Deterministic fallback text exists (for example planner fallback completion messages), but these are explicit fallback behaviors, not hidden mocked LLM outputs.

### 1.2 Test realism vs physical layers
- **FAISS retrieval is physically exercised** in `tests/test_dynamic_faiss_router.py` (real router code path, real FAISS index), but with a dummy embedder.
- **AST parsing path is not directly tested**: no tests invoke `toolfinder/utils.py` JSON extraction fallback (`ast.literal_eval`).
- **Locking coverage is partial**:
  - `Enterprise/runtime/registry.py` lock paths are hit indirectly by route/upsert calls.
  - No dedicated stress tests cover `toolfinder/mcp_adapter.py` pending-map locks under massive timeout churn.
- Test suite uses substantial stubbing/mocking in `tests/test_hybrid_pipeline.py`, `tests/test_autonomous_agent.py`, and portions of enterprise runtime tests.

## ReAct Loop and Duplicate-Action Proof Check

### A) `toolfinder/autonomous_agent.py`
Observed mechanism:
- Signature source: `raw_signature = server_name + tool_name + str(arguments)`
- Lock key: `sha256(raw_signature)`
- Set membership: reject if digest already in `executed_actions`

Mathematical issue:
- Equality is over **string form**, not canonical object semantics.
- Counterexample (same semantic arguments, different insertion order):
  - $A = {'x':1,'y':2}$
  - $B = {'y':2,'x':1}$
  - `str(A) != str(B)` in Python insertion-order representation
  - therefore $sha256(str(A)) \neq sha256(str(B))$
- Result: semantically identical call parameters can evade the loop-breaker by key-order variance.

Conclusion: duplicate execution prevention is strong against exact repeated serialized payloads, but not mathematically invariant for semantic identity.

### B) `Enterprise/runtime/openclaw_hybrid_pipeline.py`
- No cryptographic action-signature loop-breaker exists in this module.
- Tool calls from OpenClaw output are executed sequentially in a loop (`for idx, call in enumerate(agent_response.tool_calls, 1)`), with no deduplication set.

Conclusion: identical action+arguments can execute more than once if repeated in `agent_response.tool_calls`.

## Section 2: Execution Boundary Reality (AST Safety, Path Traversal)

### 2.1 AST fallback (`toolfinder/utils.py`)
- Flow: strict JSON decode first, then fallback to `ast.literal_eval`.
- `ast.literal_eval` only evaluates Python literals; it does not execute function calls/imports/system commands.
- Nesting-depth guard (`>100`) reduces parser recursion DoS risk.

Boundary status:
- **RCE via syntactically valid malicious payload through `ast.literal_eval`: not found.**
- Residual risk: very large literal payloads can still cause memory/CPU pressure (resource exhaustion), even if not code execution.

### 2.2 File-system traversal guard (`Enterprise/runtime/policy.py`)
For payload like `"path": "../../../../etc/passwd"`:
- Default policy (`deny_parent_path_segments=True`) correctly raises `SecurityPolicyViolation`.

But under other valid policy states:
- If `deny_parent_path_segments=False` and `allowed_path_roots=()`, parent traversal is not blocked.
- If `deny_parent_path_segments=False`, `allowed_path_roots` set, and path is relative, `_enforce_allowed_path_roots` returns early for non-absolute paths.

Boundary status:
- **Traversal prevention is not guaranteed under all configuration states.**
- Therefore the claim "physically cannot bypass under any configuration" is **false**.

## Section 3: Remaining Theoretical Memory-Leak / Deadlock Vectors

### 3.1 `_pending_requests` map in `toolfinder/mcp_adapter.py`
What is robust:
- Request future inserted under lock.
- On timeout/exception: request removed and future canceled.
- On normal response: request popped by stdout loop.
- On shutdown/error: `_fail_pending_requests` drains all pending futures.

Result for 10,000 timeout failures:
- No persistent map-growth leak was identified from timeout path alone.

### 3.2 Residual theoretical risks
- Large **concurrent** in-flight timeouts can spike memory transiently (many futures alive until timeout).
- `_send_message()` has no explicit timeout; if stdio backpressure stalls indefinitely, request tasks can block on `stdin.drain()` before timeout logic is reached.
- Event loop deadlock from lock-order inversion was not observed in current locking design.

## Section 4: Dependency and Hardware Reality

### 4.1 Dependency legitimacy (`pyproject.toml`)
All listed dependencies resolved via PyPI JSON API and returned package metadata:
- `faiss-cpu`, `sentence-transformers`, `jsonschema`, `httpx`, `pydantic`
- optional: `langchain`, `langgraph`, `langchain-ollama`, `openclaw`, `cmdop`
- dev: `pytest`, `pytest-asyncio`

### 4.2 Inference and vector hardware defaults
- Router embedding device default: `cuda` if available, else `cpu`.
- Vector index implementation is `faiss.IndexFlatIP` from `faiss-cpu`, so FAISS similarity search is CPU-backed by default.
- Net effect:
  - Embedding generation may use GPU when CUDA is available.
  - Dense vector index/search path defaults to CPU.

## Final Determination
- Production runtime does not contain hidden mock runtime shims masquerading as real tool execution.
- Security boundaries are improved but not absolute across all policy configurations.
- Duplicate-action prevention is not mathematically invariant across semantic-equivalent argument encodings.

**Overall: Reality check does not fully pass under strict red-team criteria.**

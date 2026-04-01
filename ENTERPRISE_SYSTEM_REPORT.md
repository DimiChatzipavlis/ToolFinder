# Enterprise System Report

Date: 2026-04-01
Scope: ToolFinder core plus Enterprise OpenClaw runtime

## 1. Final Architectural State

### 1.1 Semantic Routing Layer

Core implementation:
- Dense embedding model: sentence-transformers/all-mpnet-base-v2 in [toolfinder/dynamic_faiss_router.py](toolfinder/dynamic_faiss_router.py).
- Runtime FAISS index: IndexFlatIP in [toolfinder/dynamic_faiss_router.py](toolfinder/dynamic_faiss_router.py).
- Query-time thresholding via min_score filter in [toolfinder/dynamic_faiss_router.py](toolfinder/dynamic_faiss_router.py).

Complexity profile:
- Current runtime behavior is exact dense retrieval over IndexFlatIP (effective O(N) scan per query).
- Architectural scaling trajectory remains compatible with ANN indexes (targeting sublinear/O(log N)-like behavior at larger tool inventories).

Context payload compression and latency metrics:
- Focused benchmark headline: ~95% prompt payload reduction (9110 -> 485 chars) and ~84% first-turn latency reduction (85.52s -> 13.71s), from [README.md](README.md).
- Auto-updating multi-task benchmark block (current snapshot):
  - Average tools in context: 14 -> 2
  - Average payload: 9106 -> 1450 chars
  - Average total latency: 57.51s -> 14.47s
  - Average inference latency: 57.47s -> 14.39s

Dynamic thresholding:
- Core threshold control: min_score argument in router route_top_k.
- Enterprise config threshold: ENTERPRISE_MIN_SCORE in [Enterprise/runtime/config.py](Enterprise/runtime/config.py).
- Degraded keyword fallback is now fail-closed by default and opt-in via ENTERPRISE_ALLOW_KEYWORD_LOW_CONFIDENCE_FALLBACK in [Enterprise/runtime/registry.py](Enterprise/runtime/registry.py).

### 1.2 Execution Middleware

Resilient parsing:
- Strict JSON parse first, then AST literal_eval fallback in [toolfinder/utils.py](toolfinder/utils.py).
- Nesting-depth guard prevents parser recursion abuse in [toolfinder/utils.py](toolfinder/utils.py).

Schema hardening:
- Recursive additionalProperties: false injection in both:
  - [toolfinder/dynamic_faiss_router.py](toolfinder/dynamic_faiss_router.py)
  - [toolfinder/mcp_adapter.py](toolfinder/mcp_adapter.py)

ReAct idempotency:
- Action signature hashing via hashlib.sha256 and executed-action guard in [toolfinder/autonomous_agent.py](toolfinder/autonomous_agent.py).

### 1.3 Enterprise Hybrid Pipeline

Fail-closed structured parsing:
- StructuredParsingError introduced and raw-text success fallback removed in [Enterprise/runtime/openclaw_hybrid_pipeline.py](Enterprise/runtime/openclaw_hybrid_pipeline.py).
- OpenClaw session now requires structured JSON completion/tool-action payloads.

Degraded fallback telemetry and result semantics:
- Primary OpenClaw failure + successful fallback now returns status=degraded_fallback.
- fallback_triggered metadata flag is present in [Enterprise/runtime/contracts.py](Enterprise/runtime/contracts.py) and used by pipeline return paths in [Enterprise/runtime/openclaw_hybrid_pipeline.py](Enterprise/runtime/openclaw_hybrid_pipeline.py).
- pipeline_fallback_triggered telemetry counter added in [Enterprise/runtime/openclaw_hybrid_pipeline.py](Enterprise/runtime/openclaw_hybrid_pipeline.py).

Terminal policy enforcement:
- SecurityPolicyViolation introduced in [Enterprise/runtime/policy.py](Enterprise/runtime/policy.py).
- Orchestrator now treats SecurityPolicyViolation and PolicyViolation as terminal failures (no retry) in [Enterprise/runtime/orchestrator.py](Enterprise/runtime/orchestrator.py).

Health and unrouted mode hardening:
- Backend probe requires strict HTTP 200 in [Enterprise/runtime/openclaw_backend.py](Enterprise/runtime/openclaw_backend.py).
- Orchestrator emits critical warning when unrouted execution bypasses schema in [Enterprise/runtime/orchestrator.py](Enterprise/runtime/orchestrator.py).

### 1.4 Operational Readiness and Strict Runtime Reality

Strict execution mode:
- Added STRICT_MODE/--strict in:
  - [Enterprise/examples/run_realtime_openclaw.py](Enterprise/examples/run_realtime_openclaw.py)
  - [Enterprise/examples/run_e2e_hybrid.py](Enterprise/examples/run_e2e_hybrid.py)
- In strict mode, mock clients are disabled and live MCP availability is mandatory.

Shift in test posture:
- Runtime tests now expect fail-closed behavior instead of permissive fallback success.
- Fallback path now asserts degraded_fallback/fallback_triggered telemetry semantics.

## 2. Micro-Flaw Audit (Post-Hardening)

### 2.1 Concurrency and Async State

1) DynamicMCPClient startup and tool-cache path has no lock around _started/_tools_cache checks.
- Risk: concurrent initialize_and_get_tools callers can race process startup/cache initialization.

2) AutonomousMCPAgent register_server mutates shared clients map without synchronization.
- Risk: concurrent registration can produce duplicate/ordering races.

### 2.2 Telemetry and Observability Blindspots

3) OpenClawSessionDriver zeroes raw_output in exception paths.
- Risk: backend response/debug evidence is discarded at the exact failure point.

4) OpenClawHybridPipeline tool-call execution path collapses SecurityPolicyViolation into generic Exception.
- Risk: security failures are downgraded to generic tool errors in pipeline telemetry.

5) RealTimeHybridService catches broad exceptions and prints only str(exc).
- Risk: traceback and root network/process cause are lost in long-running loops.

6) OpenClaw backend request retries capture only a condensed last_error string.
- Risk: no structured per-attempt diagnostics for outage triage.

7) Event bus defaults to continue_on_error=True and only stores string errors.
- Risk: handler failures can be silently tolerated with limited forensics.

### 2.3 Configuration Drift

8) EnterpriseConfig does not validate min_score bounds or planner_timeout_s positivity.
- Risk: invalid environment values can silently create pathological routing/timeout behavior.

9) STRICT_MODE and LIVE_FILESYSTEM_ROOT are implemented in scripts but absent from markdown docs.
- Risk: runtime behavior can differ across operators due to undocumented controls.

10) Autonomous agent still contains hardcoded request timeout=300 for Ollama calls.
- Risk: timeout drift against enterprise config and inconsistent failure semantics.

### 2.4 Type Integrity

11) Enterprise contracts and pipeline payloads rely heavily on JsonDict/dict[str, Any].
- Risk: schema shape drift between router, planner, and OpenClaw payloads is not type-constrained.

12) HybridToolExecutor accepts clients typed as DynamicMCPClient | Any.
- Risk: interface mismatches are deferred to runtime and fail late.

13) OpenClawSessionDriver line-based JSON salvage accepts partial mixed-output streams if any action/completion fragment parses.
- Risk: malformed mixed output may still pass as success with incomplete semantics.

## 3. Enterprise Verdict

Current system state is significantly hardened versus prior revisions: strict structured parsing, explicit degraded fallback signaling, terminal policy enforcement, and strict mode execution gates are now in place.

Residual risk is now concentrated in operational discipline and long-run reliability details: startup race windows, weakly typed payload boundaries, and observability granularity rather than fundamental security posture.

## 4. Recommended Next Wave (No Code Changes in This Report)

1) Add startup locks in DynamicMCPClient and AutonomousMCPAgent registration paths.
2) Preserve and redact raw backend payload snippets in failure telemetry rather than dropping them.
3) Promote SecurityPolicyViolation as a first-class pipeline event type in OpenClaw tool-call execution path.
4) Add strict config validation for min_score in [0,1] and planner_timeout_s > 0.
5) Replace wide Any payload typing with pydantic/dataclass schema models at router-planner-pipeline boundaries.
6) Document STRICT_MODE/LIVE_FILESYSTEM_ROOT in markdown operator runbooks.
7) Move autonomous-agent timeout from hardcoded constant to explicit configuration.

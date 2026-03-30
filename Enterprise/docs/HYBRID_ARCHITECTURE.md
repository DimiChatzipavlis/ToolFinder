# Hybrid Enterprise Architecture

## Why Hybrid Instead Of Full Generative Planning

Tool routing and tool execution failures are different error classes.

- Selection error: picking the wrong tool.
- Argument error: passing malformed or incomplete parameters.
- Execution error: runtime tool failure, permissions, or side effects.

A retrieval-first gate minimizes selection error before the planner reasons about tool usage. The planner then focuses on argument quality and multi-step adaptation.

## Runtime Data Flow

1. User intent enters orchestrator.
2. Tool registry routes top-k candidates by semantic similarity.
3. Planner receives constrained candidate set and history.
4. Policy layer authorizes selected server/tool and argument envelope.
5. Executor validates schema and performs MCP call.
6. Observation is fed back to planner for iterative completion.
7. Events and telemetry are emitted for monitoring and audit.

## Real-Time Enterprise Features

- Live catalog refresh via `HybridToolRegistry.upsert_server_tools`.
- Event-driven integration points through `EnterpriseEventBus`.
- Real OpenClaw backend adapters:
	- `OpenClawHttpBackend` for Ollama/OpenAI-compatible HTTP endpoints.
	- `OpenClawCliBackend` for direct `openclaw` CLI execution.
- Continuous workspace monitoring via `WorkspaceChangeTracker` and `RealTimeHybridService`.
- Deterministic policy controls for security and compliance.

## Concrete Runtime Entry Points

- `Enterprise/examples/run_hybrid_demo.py`: deterministic local demo run.
- `Enterprise/examples/run_realtime_openclaw.py`: real-time OpenClaw-connected runtime loop.

## Production Hardening Recommendations

1. Back planner with high-throughput OSS serving (for example vLLM/TGI).
2. Add distributed event broker adapter (Kafka/NATS) behind event bus interface.
3. Persist session traces to durable storage for audits.
4. Add canary planner strategy and shadow evaluation before cutover.
5. Track p95 routing/planning/execution latency and tool success SLOs.

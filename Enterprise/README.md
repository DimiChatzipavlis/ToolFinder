# Enterprise Hybrid Runtime

This directory contains an enterprise-grade hybrid architecture built on ToolFinder's retrieval-first design.

## Verified Status (2026-03-31)

Current verdict: the Enterprise hybrid edition is working on this repository checkout.

Verified on this machine:

- Enterprise-focused tests passed: `pytest -q tests/test_enterprise_runtime.py tests/test_hybrid_pipeline.py tests/test_enterprise_backend.py`
- Deterministic orchestrator demo passed: `python Enterprise/examples/run_hybrid_demo.py`
- End-to-end OpenClaw pipeline demo passed: `python Enterprise/examples/run_e2e_hybrid.py --max-cycles 1 --fallback-strategy heuristic_planner`
- Realtime finite-cycle loop passed: `python Enterprise/examples/run_realtime_openclaw.py --max-cycles 1 --tool-runtime mock`

Known operational limit:

- In live filesystem mode, model-generated arguments may still request out-of-scope paths. Policy and runtime checks block unsafe calls, but you should still treat this as an expected operational failure mode to monitor.

## Design Goals

- Keep low tool-selection error via semantic retrieval gates.
- Add OpenClaw-style planner behavior using OSS LLM backends.
- Enable real-time catalog updates and policy-governed execution.
- Produce auditable telemetry for SLO and governance controls.

## Core Architecture

1. Retrieval gate: picks top-k tools from semantic index.
2. Planner: OpenClaw-style decision engine over routed tool subset.
3. Policy layer: validates authorization, argument size, and tool guardrails.
4. Executor: schema-validates and invokes MCP tools.
5. Event bus + telemetry: emits runtime events and performance metrics.
6. Registry: supports live tool-catalog updates without changing planner logic.

## New Hardening Features

- Strict completion semantics: execution errors are no longer promoted into successful completions.
- Policy parity across paths: OpenClaw tool calls use the same policy checks as orchestrator execution.
- Tool-call-first safety: final OpenClaw answers are only accepted after planned tool calls execute.
- Realtime precision: changed-file tracking uses added/modified/deleted deltas.
- Durable telemetry: optional JSONL sink via `ENTERPRISE_TELEMETRY_SINK` and merged fallback telemetry.
- Event bus isolation: one failing event subscriber does not stop event publication.

## Quick Start

Install base and enterprise extras:

```bash
pip install -e .[dev,enterprise]
```

Optional OpenClaw CLI backend (recommended for local edge workflow):

```bash
npm install -g openclaw
```

Run deterministic hybrid demo:

```bash
python Enterprise/examples/run_hybrid_demo.py
```

## Real-Time OpenClaw Runtime

This repository now includes a concrete real-time OpenClaw-connected loop:

```bash
python Enterprise/examples/run_realtime_openclaw.py \
	--workspace . \
	--endpoint http://127.0.0.1:11434/api/generate \
	--model llama3.2 \
	--api-mode ollama-generate \
	--backend-kind http \
	--tool-runtime auto
```

Environment variables:

- `OPENCLAW_BACKEND_KIND`: `http` or `cli`.
- `OPENCLAW_ENDPOINT`: planner endpoint URL.
- `OPENCLAW_MODEL`: OSS model id.
- `OPENCLAW_API_MODE`: `ollama-generate` or `openai-chat`.
- `OPENCLAW_API_KEY`: optional bearer token for protected endpoints.
- `OPENCLAW_CLI_BIN`: optional path/name for the CLI binary.
- `ENTERPRISE_DISABLE_SEMANTIC_ROUTER`: set `1` to force keyword fallback routing (useful offline/air-gapped).
- `ENTERPRISE_TELEMETRY_SINK`: optional JSONL file path for durable telemetry snapshots.

Finite-cycle smoke test:

```bash
python Enterprise/examples/run_realtime_openclaw.py --max-cycles 2
```

CLI backend smoke test:

```bash
python Enterprise/examples/run_realtime_openclaw.py --backend-kind cli --max-cycles 2
```

## End-to-End Hybrid Pipeline

The new `OpenClawHybridPipeline` combines ToolFinder's semantic retrieval with
OpenClaw's native agent execution for full end-to-end autonomy:

```bash
python Enterprise/examples/run_e2e_hybrid.py \
	--endpoint http://127.0.0.1:11434/api/generate \
	--model llama3.2 \
	--api-mode ollama-generate \
	--fallback-strategy heuristic_planner
```

Custom query:

```bash
python Enterprise/examples/run_e2e_hybrid.py \
	--query "Read main.py and summarize its architecture" \
	--max-agent-steps 5
```

Fallback strategies: `heuristic_planner` (default), `error`, `best_effort`.

Pipeline safety defaults:

- OpenClaw tool calls are policy-validated before execution.
- Tool-call plans are executed before accepting a final OpenClaw answer.
- If any planned tool call fails, the pipeline marks the OpenClaw path failed and applies fallback strategy.

Live MCP filesystem mode:

```bash
python Enterprise/examples/run_realtime_openclaw.py \
	--workspace . \
	--live-filesystem-root ./examples/langgraph_integration/sandbox
```

For implementation details, see `Enterprise/docs/HYBRID_ARCHITECTURE.md`.

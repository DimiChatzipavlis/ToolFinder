# Enterprise Hybrid Runtime

This directory contains an enterprise-grade hybrid architecture built on ToolFinder's retrieval-first design.

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

Finite-cycle smoke test:

```bash
python Enterprise/examples/run_realtime_openclaw.py --max-cycles 2
```

CLI backend smoke test:

```bash
python Enterprise/examples/run_realtime_openclaw.py --backend-kind cli --max-cycles 2
```

Live MCP filesystem mode:

```bash
python Enterprise/examples/run_realtime_openclaw.py \
	--workspace . \
	--live-filesystem-root ./examples/langgraph_integration/sandbox
```

For implementation details, see `Enterprise/docs/HYBRID_ARCHITECTURE.md`.

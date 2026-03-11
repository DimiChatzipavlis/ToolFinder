# Neural MCP Router & Autonomous Execution Agent

**Abstract:**
A two-phase system designed to resolve the LLM context-stuffing bottleneck for Model Context Protocol (MCP) integrations. It transitions from a static dense-retrieval prototype to a fully autonomous, multi-server ReAct agent capable of zero-shot tool execution.

## Phase 1: The Academic Prototype (Semester Project)
This phase established the semantic routing baseline used to map natural-language requests to MCP tool schemas without exposing all tool definitions to the language model.

- **Methodology:** Contrastive learning using `MultipleNegativesRankingLoss` on a synthetic dataset of 1,500 queries mapping to GitHub MCP schemas.
- **Model:** 109M parameter Bi-Encoder (`all-mpnet-base-v2`).
- **Empirical Validation:** 94.13% Zero-Shot Recall@1 on held-out tools with <50ms latency.
- **Limitations Identified:** $O(N \cdot d)$ retrieval scaling, static CSV dependency, and absence of execution-layer schema validation.

## Phase 2: Scalable Middleware (Enterprise Architecture)
This phase refactors the routing prototype into a runtime system that can discover tools dynamically, route across multiple MCP servers, and validate execution requests before dispatch.

- **Dynamic Ingestion (stdio):** Bypasses static datasets by connecting to ephemeral MCP servers over standard I/O, normalizing `tools/list` payloads at runtime.
- **Logarithmic Scaling (FAISS):** Replaces flat tensor scans with a FAISS Inner Product index for N-server scalability.
- **Strict Validation Boundary:** Recursively injects `{"additionalProperties": false}` into dynamically ingested schemas. Intercepts LLM outputs via Regex and enforces compliance via `jsonschema` prior to JSON-RPC dispatch, ensuring zero-hallucination execution.
- **ReAct State Machine:** Implements a multi-iteration Reason+Act loop. Utilizes "Semantic Anchoring" (Goal + Last Observation) to prevent vector dilution during dense retrieval, enabling cross-server workflow execution (e.g., read from SQLite $\rightarrow$ write to Memory).

## Usage & Quickstart
Install dependencies from the repository root before running either workflow.

```bash
cd ToolFinder
pip install -r requirements.txt
```

1. The Zero-Shot Academic Benchmark (`evaluate_zero_shot.py`).

```bash
cd ToolFinder
python evaluate_zero_shot.py
```

2. The Dynamic Multi-Server ReAct Agent (`verify_discovery.py`).

```bash
cd ToolFinder
python scalable_router/verify_discovery.py
```

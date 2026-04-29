# ToolFinder: Neural Semantic Router for MCP

🧠 Stop stuffing your LLM context windows. ToolFinder is a zero-hallucination routing middleware that dynamically connects Local SLMs to massive Model Context Protocol (MCP) ecosystems without OOM crashes.

It separates tool selection from tool execution, so your model only sees the few schemas it actually needs. The result is lower latency, tighter prompts, and a much safer path from local-model experimentation to production-grade MCP orchestration.

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![MCP](https://img.shields.io/badge/MCP-compatible-black)](https://modelcontextprotocol.io/)
[![FAISS](https://img.shields.io/badge/retrieval-FAISS-orange)](https://github.com/facebookresearch/faiss)
[![LangGraph](https://img.shields.io/badge/integration-LangGraph-green)](https://github.com/langchain-ai/langgraph)

## 🚨 The Problem: Context Bloat & Lost in the Middle

Blindly binding `50` MCP tools to a small local model like `llama3.2` is a systems mistake.

- The prompt fills with irrelevant schemas before reasoning even begins.
- The model spends tokens comparing tools instead of using them.
- Similar APIs start colliding in-context, which increases tool-selection errors and malformed calls.
- Even when the right tool is chosen, smaller models often emit partially invalid JSON under long prompt pressure.

This is the classic "lost in the middle" failure mode applied to MCP orchestration: the right tool may exist in context, but the model has to wade through too much irrelevant structure to use it reliably.

## 🎯 The Solution: Semantic Anchoring

ToolFinder turns MCP tool usage into a retrieval problem first.

- A contrastive bi-encoder built on `sentence-transformers/all-mpnet-base-v2` embeds queries and MCP schemas into the same vector space.
- A FAISS-backed similarity index retrieves only the top-k candidate tools for the user’s query.
- The model then reasons over a tiny, relevant tool surface instead of the entire ecosystem.

The rigorous data science, datasets, and semester-project evaluation pipeline live in [academic_research](academic_research). That folder contains the training corpora, notebooks, and zero-shot evaluation stack behind the retrieval layer.

> Technical note:
> The current runtime implementation uses FAISS `IndexFlatIP` for exact dense retrieval with very low observed latency. The architectural scaling story generalizes cleanly to ANN indexes when larger tool graphs justify sublinear search.

## 📊 Empirical Benchmarks: The Proof

The repository contains two proof surfaces:

- A focused LangGraph A/B benchmark showing the first-turn efficiency win of semantic routing.
- A self-bootstrapping multi-task evaluator that continuously refreshes the benchmark table below.

### Headline Result

- ~95% prompt payload reduction in the focused LangGraph proof (`9110` chars to `485` chars)
- ~84% latency reduction in the same first-turn comparison (`85.52s` to `13.71s`)
- Sub-20ms to low-double-digit-ms routing in the LangGraph routing path

### Auto-Updating Benchmark

The block below is updated only when you run `python examples/eval_toolfinder.py --update-readme`. Preserve the markers so the automated suite can continue injecting the latest metrics.

<!-- EVAL_TABLE_START -->
_Last auto-updated: 2026-03-16 17:01:57_

| Metric | Naive Baseline | ToolFinder Enabled |
| --- | --- | --- |
| Tasks Run | 3 | 3 |
| Average Tools In Context | 14 | 2 |
| Average Context Payload (Chars) | 9106 | 1450 |
| Average Total Latency (s) | 57.51 | 14.47 |
| Average Inference Latency (s) | 57.47 | 14.39 |
| Successful Tool Calls | 3/3 | 3/3 |
| Expected Tool Matches | 3/3 | 3/3 |
| State Verified | 3/3 | 3/3 |

Task outcomes:
- T1_READ: naive=`read_text_file` verified=`True`, toolfinder=`read_text_file` verified=`True`
- T2_WRITE: naive=`write_file` verified=`True`, toolfinder=`write_file` verified=`True`
- T3_LIST: naive=`list_directory` verified=`True`, toolfinder=`list_directory` verified=`True`
<!-- EVAL_TABLE_END -->

<details>
<summary>What changed between the headline proof and the live table?</summary>

The headline LangGraph benchmark measures a narrower, first-turn filesystem task and highlights the raw routing advantage. The auto-updating evaluator is broader: it runs multiple tasks, performs correctness checks, bootstraps and tears down the sandbox, and averages end-to-end inference time across the suite. The exact percentages shift, but the systems conclusion remains the same: routing a tiny tool subset is materially cheaper and safer than context stuffing.

</details>

## 🛡️ Features & Protections

ToolFinder hardens both selection and execution.

- Semantic routing narrows the prompt to the top-k MCP tools before inference.
- Strict schema enforcement injects `additionalProperties: false` into object schemas to reject speculative keys.
- AST recovery parsing salvages Python-style dicts and malformed local-model outputs when strict JSON fails.
- ReAct execution loops let the agent observe failures, retry, and continue rather than crash on the first malformed response.
- Idempotency guards and bounded scratchpads prevent repeated actions and runaway context growth.

## ⚡ Quickstart & Integration

CRITICAL: Install PyTorch with your required hardware acceleration (e.g., CUDA) before installing this package to avoid defaulting to slow CPU inference. Example: pip install torch --index-url https://download.pytorch.org/whl/cu121

Then install ToolFinder:

```bash
# 1. Install PyTorch with your specific hardware acceleration (e.g., CUDA)
pip install torch --index-url https://download.pytorch.org/whl/cu121

# 2. Install ToolFinder
pip install toolfinder-mcp
```

Minimal integration with LangChain or LangGraph:

```python
from toolfinder.dynamic_faiss_router import UniversalMCPRouter
from langchain_ollama import ChatOllama

# Initialize and ingest tools
router = UniversalMCPRouter()
for tool in mcp_server_tools:
    router.add_tool(tool)
router.build_index()

# Route and bind
llm = ChatOllama(model="llama3.2")
top_tools = router.route_top_k("Write a summary to output.txt", k=2)

# ToolFinder

ToolFinder is a semantic routing and guarded execution layer for MCP tool ecosystems. It narrows a natural-language intent to a small set
of relevant tools (via semantic retrieval) and executes selected calls with schema validation, policy enforcement, and telemetry.

## Short status (accurate and agnostic)

- The code provides a local HTTP runtime that can be started from the repository root.
- The project includes a pair of workspace-level validation scripts (`start_server.py` and `run_e2e_client.py`) that exercise the end-to-end API.
- This repository implements a runtime and examples; it does not package an opinionated multi-node distributed deployment.

If you need a production, multi-node deployment you will need to add an operational layer (service discovery, secrets, auth, load balancing, etc.).

## What is in this repo

- `toolfinder/` — core semantic router and FAISS-backed retrieval utilities.
- `Enterprise/` — hybrid runtime, API, policy, executor, planner, telemetry, and OpenClaw bridge.
- `examples/` — scripted demos and benchmarks.
- `academic_research/` — notebooks and datasets used for evaluation.
- Root-level validation helpers: `start_server.py` and `run_e2e_client.py`.

## Installation (minimal)

Requires Python 3.10+. From the repository root:

```bash
python -m pip install -e .
```

Optional extras (examples and dev tools):

```bash
python -m pip install -e '.[dev]'
python -m pip install -e '.[langgraph]'
python -m pip install -e '.[enterprise]'
```

## Local validation (agnostic instructions)

From a shell opened at the repository root:

1. Start the local runtime:

```bash
python start_server.py
```

2. In a second shell (same working directory) run the client:

```bash
python run_e2e_client.py
```

The client sends a JSON request with a single required field, `intent`, to `POST /execute` and prints the server response for inspection.

Notes:

- Run the commands from your repository root so relative imports and workspace checks work correctly.
- The validation scripts are convenience helpers; they are not intended to be production entry points.

## HTTP API (concise)

The API is implemented by a factory function in `Enterprise/runtime/api.py` and should be launched via the factory rather than assuming a module-level `app` object. See `create_app(...)` in [Enterprise/runtime/api.py](Enterprise/runtime/api.py#L44) and the request model `ExecuteIntentRequest` in [Enterprise/runtime/api.py](Enterprise/runtime/api.py#L22).

Endpoint

```http
POST /execute
```

Request JSON (required):

```json
{ "intent": "<user natural language intent>" }
```

Successful response (abridged):

```json
{
    "session_id": "api-...",
    "execution_output": {
        "status": "complete|failed|partial",
        "answer": "...",
        "tool_calls": [ ... ],
        "telemetry": { ... }
    }
}
```

If the request fails schema validation or policy enforcement the server will return an error payload (HTTP 4xx/5xx with an `error` field).

## How the runtime is organized (short)

- `HybridToolRegistry` — catalog + retrieval (FAISS/keyword fallback).
- `PolicyEngine` — path and argument guardrails.
- `HybridToolExecutor` — executes validated tool calls against MCP clients (or mocks).
- `OpenClawHybridPipeline` — end-to-end orchestration: routing → agent → tool execution → fallback → telemetry.

These are designed to be composable; the repo contains examples that wire them together for local validation.

## Distribution note (explicit)

This repository provides the runtime components and examples. A production distributed deployment requires additional operational components (deployment manifests, service discovery, secrets, authentication, logging/monitoring, scaling strategies) which are outside the scope of this codebase.

## Examples and demos

See `examples/` and `Enterprise/examples/` for many runnable demos. Typical commands (from repo root):

```bash
python examples/eval_toolfinder.py
python Enterprise/examples/run_e2e_hybrid.py --max-cycles 1
```

## Further reading

- `ARCHITECTURE_REPORT.md`
- `SYSTEM_REALITY_REPORT.md`
- `ENTERPRISE_SYSTEM_REPORT.md`

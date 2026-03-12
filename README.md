# Neural MCP Router & Autonomous Execution Engine

Semantic middleware for Model Context Protocol systems that replaces context stuffing with retrieval, schema enforcement, and runtime tool orchestration.

This project demonstrates that MCP-native agents do not need to bind every tool schema into the LLM prompt. Instead, ToolFinder narrows the tool surface semantically, enforces strict execution boundaries, and enables autonomous multi-step behavior across heterogeneous servers.

## The Core Problem

LLMs degrade when every available MCP tool is stuffed into the prompt at once.

That failure mode has three predictable consequences:

1. Prompt windows balloon as tool catalogs grow.
2. Inference latency increases because the model must repeatedly reason over irrelevant schemas.
3. Tool hallucinations rise because the model sees too many near-matching APIs at the same time.

In practical MCP environments, this means a simple task like listing a directory and writing one file can require the model to sift through an entire filesystem API, database API, fetch API, and memory API on every step. ToolFinder treats that as a retrieval problem first and an LLM problem second.

## The A/B Benchmark

The repository includes a direct LangGraph comparison between two agents that solve the same filesystem task inside a sandboxed MCP filesystem server.

Task:

`List the files in the sandbox directory. Then write a new file.`

### Naive Baseline

- Binds all 14 filesystem tools into `ChatOllama` at once.
- First-turn prompt payload: `9110` chars.
- Second-turn prompt payload: `9967` chars.
- First inference latency: `85.52s`.
- Second inference latency: `10.30s`.
- Result: completed, but only after forcing the full filesystem API into the model context.

### ToolFinder Enabled

- Uses FAISS semantic routing to expose only the top-2 tools.
- First-turn prompt payload: `485` chars.
- Second-turn prompt payload: `1148` chars.
- Routing latency: `18.93ms` then `67.02ms`.
- Inference latency: `13.71s` then `5.97s`.
- Context window saved: `7961` chars per routed turn.
- Result: completed with dynamic top-k tool binding and verified sandbox write execution.

### Why The Benchmark Matters

The delta is not cosmetic. It is architectural.

ToolFinder removes roughly $7961$ characters of irrelevant schema context per turn in this benchmark while preserving correct execution. That is the difference between an LLM reasoning over a targeted tool surface and an LLM drowning in an entire API catalog.

## Architecture Overview

### Phase 1: Academic Prototype

- 109M parameter bi-encoder built on `sentence-transformers/all-mpnet-base-v2`.
- Contrastive training with synthetic query-to-tool alignment data.
- `94.13%` zero-shot `Recall@1` on held-out tools.
- Established the thesis: semantic retrieval can select the correct MCP tool without exposing the entire tool inventory.

### Phase 2: Scalable Middleware

- Dynamic stdio ingestion of live MCP servers at runtime.
- Recursive schema hardening with `additionalProperties: false`.
- JSON extraction and recovery logic for brittle local-model outputs.
- FAISS indexing for scalable retrieval across multi-server tool inventories.
- Autonomous ReAct execution engine capable of fetch, memory, and sqlite workflows.
- LangGraph bridge that converts routed MCP schemas into dynamically bound LangChain tools.

## Usage & Quickstart

Install dependencies from the repository root.

```bash
cd ToolFinder
pip install -r requirements.txt
```

### AutonomousMCPAgent

```python
import asyncio

from toolfinder import AutonomousMCPAgent, DynamicMCPClient


async def main() -> None:
	async with AutonomousMCPAgent(
		model_name="sentence-transformers/all-mpnet-base-v2",
		ollama_model="llama3.2",
		max_iterations=7,
	) as agent:
		client = DynamicMCPClient(
			server_name="filesystem",
			command="npx",
			args=["-y", "@modelcontextprotocol/server-filesystem", "./sandbox"],
		)
		await agent.register_server("filesystem", client)
		result = await agent.execute_task("List files and write hello.txt in the sandbox.")
		print(result.status)
		print(result.answer)


asyncio.run(main())
```

### LangGraph Integration

ToolFinder-routed LangGraph benchmark:

```bash
cd ToolFinder
python -u examples/langgraph_integration/benchmark_agent.py
```

Naive all-tools baseline:

```bash
cd ToolFinder
python -u examples/langgraph_integration/baseline_agent.py
```

Minimal dynamic-binding pattern:

```python
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph

from toolfinder import UniversalMCPRouter

router = UniversalMCPRouter(model_name="sentence-transformers/all-mpnet-base-v2")
candidates = router.route_top_k("write a file in the sandbox", k=2)

llm = ChatOllama(model="llama3.2", temperature=0)
bound_llm = llm.bind_tools(active_tools)  # active_tools built from routed MCP schemas

graph = StateGraph(State)
```

## Repository Highlights

- `toolfinder/`: runtime MCP ingestion, FAISS routing, schema validation, autonomous execution.
- `examples/langgraph_integration/benchmark_agent.py`: ToolFinder-enabled LangGraph benchmark.
- `examples/langgraph_integration/baseline_agent.py`: naive control-group benchmark with full tool binding.
- `examples/prove_scalability.py`: tri-server autonomous execution proof across fetch, sqlite, and memory MCP servers.

## Engineering Takeaway

The project demonstrates a repeatable pattern for MCP-native agents:

1. Discover tools dynamically.
2. Normalize and harden schemas at the middleware boundary.
3. Retrieve only the semantically relevant tool subset.
4. Bind that subset into the LLM at execution time.
5. Keep the orchestration layer observable with explicit latency and hallucination telemetry.

That pattern scales better than prompt stuffing, produces cleaner execution traces, and turns semantic routing into an operational advantage instead of a research artifact.

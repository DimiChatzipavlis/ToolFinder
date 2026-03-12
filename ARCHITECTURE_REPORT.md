# ToolFinder Architecture Report

## Executive Summary

Model Context Protocol ecosystems create a predictable systems bottleneck: tool catalogs grow faster than local models can reason over them. When an agent binds every MCP schema into the prompt, the LLM is forced to solve two separate problems inside the same context window: first, identify the right tool from a large API surface; second, produce valid execution arguments for that tool. On smaller local models, that coupling causes context bloat, slower inference, and a sharp increase in malformed tool calls.

ToolFinder addresses that bottleneck by separating retrieval from generation. The router reduces the candidate tool surface before inference, then the middleware hardens execution after inference. This architecture keeps tool selection tractable, makes local-model behavior more reliable, and creates a path from research-grade semantic routing to production-grade MCP orchestration.

## The Mathematical Solution

The retrieval layer is built around a bi-encoder and a FAISS-backed similarity index.

- Query encoder: `sentence-transformers/all-mpnet-base-v2`
- Representation strategy: canonicalized MCP tool schemas embedded into the same semantic space as user queries
- Retrieval engine: FAISS inner-product similarity search over normalized dense vectors

The research basis lives in [academic_research](academic_research). The zero-shot evaluation pipeline in [academic_research/evaluate_zero_shot.py](academic_research/evaluate_zero_shot.py) validates generalization over unseen tool schemas, while the datasets in [academic_research/mcp_routing_dataset.csv](academic_research/mcp_routing_dataset.csv) and [academic_research/mcp_routing_dataset_v2.csv](academic_research/mcp_routing_dataset_v2.csv) capture contrastive query-to-tool supervision.

In the current package, the runtime index uses FAISS `IndexFlatIP`, which provides exact dense retrieval with very low practical latency for the observed tool counts. Architecturally, the same separation of concerns is compatible with approximate nearest-neighbor indexes when the tool graph grows larger and stricter sublinear scaling becomes necessary.

## The Engineering Solution

ToolFinder does not stop at routing. The package layers execution reliability on top of semantic retrieval.

### 1. Retrieval Before Generation

The router in [toolfinder/dynamic_faiss_router.py](toolfinder/dynamic_faiss_router.py) ingests tool schemas, strips low-signal nested descriptions for embeddings, and returns only the top-k semantically relevant tools for a query. This removes the need to bind an entire tool universe into every LLM prompt.

### 2. Strict Schema Validation

The MCP client in [toolfinder/mcp_adapter.py](toolfinder/mcp_adapter.py) recursively injects `additionalProperties: false` into object schemas so the execution layer rejects speculative argument keys from smaller models. The compatibility ingestion path in the router now applies the same hardening to manually added tool schemas.

### 3. ReAct Execution Loops

The autonomous execution engine in [toolfinder/autonomous_agent.py](toolfinder/autonomous_agent.py) uses a ReAct loop to route, reason, execute, observe, and retry. This is the crucial step that turns routing into usable middleware instead of a static retrieval demo.

### 4. AST and JSON Recovery

The parser in [toolfinder/utils.py](toolfinder/utils.py) first attempts strict JSON extraction, then falls back to `ast.literal_eval` for Python-style dicts and malformed local-model outputs. This is an engineering concession to real local-model behavior, not an academic convenience.

### 5. Idempotency and Operational Safeguards

The execution layer adds idempotency guards, observation truncation, and constrained scratchpad history so repeated actions and context blowups do not silently degrade runtime behavior.

## The Empirical Proof

ToolFinder has two complementary proof layers.

### Research Proof

The academic evaluation stack demonstrates that semantic retrieval can identify the correct MCP tool without exposing the entire tool inventory. The zero-shot evaluation code in [academic_research/evaluate_zero_shot.py](academic_research/evaluate_zero_shot.py) exists to validate the retrieval premise independently from downstream orchestration.

### Systems Proof

The LangGraph integration examples in [examples/langgraph_integration/benchmark_agent.py](examples/langgraph_integration/benchmark_agent.py) and [examples/langgraph_integration/baseline_agent.py](examples/langgraph_integration/baseline_agent.py) compare two execution strategies over the same filesystem task:

- Naive baseline: bind the full tool catalog into the model
- ToolFinder: route the top-2 tools, then bind only that subset

The original A/B proof point showed:

- Prompt payload reduction from roughly `9110` chars to `485` chars, or about `94.7%`
- First-turn latency reduction from about `85.52s` to `13.71s`, or about `84.0%`
- Routing latency in the `18ms` to `67ms` range

The current automated evaluator in [examples/eval_toolfinder.py](examples/eval_toolfinder.py) now runs a broader multi-task suite and auto-updates the README benchmark block. That broader suite reports a more conservative average because it measures multiple tasks end to end rather than a single first-turn filesystem interaction, but it still preserves the central result: routing a tiny candidate set is materially cheaper than stuffing the entire tool surface into the model context.

## Known Limitations

### Local-model reasoning ceilings still matter

Routing can dramatically improve tool selection, but it does not make a weak model strong at long-horizon reasoning. ToolFinder reduces the burden on the model; it does not eliminate model-quality ceilings.

### Exact FAISS search is low-latency, not magically sublinear

The current implementation uses exact similarity search through `IndexFlatIP`. It is fast enough for the demonstrated scale, but the strongest long-term scaling claim depends on swapping to ANN indexes when tool inventories become substantially larger.

### Execution hardening reduces, but does not erase, malformed outputs

Strict schemas, AST fallback parsing, and ReAct retries materially improve robustness. They do not guarantee perfect downstream execution on every task, particularly on underpowered local models or ill-specified prompts.

### Benchmark numbers vary by task mix and hardware

The repository now contains both focused proof scripts and a broader evaluator. The exact latency delta will move with model size, local hardware, and benchmark composition. The architectural conclusion is stable even when the raw timings shift.

## Conclusion

ToolFinder is best understood as semantic middleware for MCP-native agents. The core contribution is not simply that it retrieves tools quickly. The contribution is that it turns retrieval, schema hardening, parsing recovery, and ReAct orchestration into one coherent control plane for local-model tool use.

That makes the system useful to a senior engineering team for two reasons:

1. It changes the scaling constraint from prompt width to retrieval quality.
2. It converts brittle local-model tool invocation into an observable, defensible middleware pipeline.
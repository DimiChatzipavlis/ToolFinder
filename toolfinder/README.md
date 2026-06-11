# toolfinder/ — Core Library

The deployable routing layer. Four modules, no framework lock-in.

## Modules

| Module | Responsibility |
| --- | --- |
| `dynamic_faiss_router.py` | `UniversalMCPRouter`: embeds tool schemas (sentence-transformers) and routes queries via FAISS. Exact `IndexFlatIP` by default — measured faster than HNSW at every catalog size up to 100k vectors (see `experiments/results/scaling_bench.json`); HNSW is opt-in via `RouterHyperparameters(index_type="hnsw" \| "auto")`. Threshold-based abstention (`min_cosine_similarity`) plus top1-top2 ambiguity-margin logging. |
| `mcp_adapter.py` | `DynamicMCPClient`: real stdio MCP client — process spawn (no shell), initialize/tools-list handshake, request/response correlation with timeouts, bounded `stdin.drain()`, pending-request draining on shutdown. |
| `autonomous_agent.py` | `AutonomousMCPAgent`: ReAct loop over routed candidates with JSON-Schema argument validation, canonicalized duplicate-action guard (sorted-key SHA256), bounded scratchpad, and Ollama-backed reasoning. |
| `utils.py` | LLM-output recovery parsing: strict JSON first, then `ast.literal_eval` (literals only) with a nesting-depth guard. |

## API in 20 lines

```python
from toolfinder import UniversalMCPRouter, RouteNotFoundError, to_openai_tools

router = UniversalMCPRouter()                  # exact flat index, MPNet embeddings
for tool in tools_from_mcp_server:             # raw MCP payloads
    router.add_tool(tool, server_name="github")
router.build_index()

results = router.route_top_k("open a PR for the auth fix", k=3)
# -> list[RouteResult]: .server_name, .tool_name, .schema, .score (cosine)

try:
    best = router.route("what's the weather?")  # raises below threshold
except RouteNotFoundError:
    ...                                         # abstain instead of force-routing

llm_tools = to_openai_tools(results)            # OpenAI-style bindable schemas
```

## Design decisions (and where they're justified)

- **Exact search by default.** Query *encoding* (~tens of ms) dominates
  retrieval (<1 ms below 10⁵ vectors); approximate indexes buy nothing at MCP
  catalog sizes and cost exactness. Evidence: `experiments/benchmarks/scaling_bench.py`.
- **Type-stable routing API.** `route_top_k` always returns `RouteResult`;
  format conversion is an explicit function, not hidden state.
- **Abstention over force-routing.** The threshold's operating points are
  measured (risk-coverage / AUROC) in `experiments/evaluation/ood.py`, not
  guessed. Schemas get `additionalProperties: false` injected at ingest.
- **Security posture:** see [SECURITY.md](../SECURITY.md).

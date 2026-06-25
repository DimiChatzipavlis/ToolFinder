# toolfinder/ — Core Library

The deployable routing layer behind the MCP bridge. Five modules, no framework lock-in.

> **Status:** part of the OSS **v0.1** tool. See the [repo README](../README.md) *Status* section for the v0.1-vs-production split (what's ready vs. what's still needed for a production release).

## Modules

| Module | Responsibility |
| --- | --- |
| `dynamic_faiss_router.py` | `UniversalMCPRouter`: embeds tool schemas with any open sentence-transformers bi-encoder (configurable via `model_name` / `TOOLFINDER_MODEL`) and routes queries via FAISS. Exact `IndexFlatIP` by default — query *encoding* dominates latency, so approximate search buys nothing at MCP catalog sizes; HNSW is opt-in via `RouterHyperparameters(index_type="hnsw" \| "auto")`. Threshold-based abstention (`min_cosine_similarity`) plus top1-top2 ambiguity-margin logging. |
| `mcp_adapter.py` | `DynamicMCPClient`: real stdio MCP client — process spawn (no shell), initialize/tools-list handshake, request/response correlation with timeouts, bounded `stdin.drain()`, pending-request draining on shutdown. |
| `mcp_server.py` | `FastMCP` bridge server (`toolfinder-mcp`): fronts one or more downstream MCP servers / OpenAPI APIs, exposes `find_tools` / `call_tool` / `route_and_call` / `catalog_size` / `get_stats` / `refresh`, dispatches to the owning server, reconnects on failure. See [docs/MCP_SERVER.md](../docs/MCP_SERVER.md). |
| `openapi_adapter.py` | `OpenAPIClient`: fronts a REST API from an OpenAPI 3.x spec — each operation becomes a routable tool; maps a chosen tool + args back to an HTTP request with env-resolved auth. Same `initialize_and_get_tools()` / `call_tool()` interface as the MCP client, so the bridge is transport-agnostic. |
| `reranker.py` | `CrossEncoderReranker` (opt-in): re-scores the bi-encoder's top-k shortlist with a cross-encoder (`(query, tool)` scored jointly). Lazy-loaded, zero-shot — sharpens selection on confusable catalogs without retraining. Enabled via `RouterHyperparameters(rerank=True)`. |

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
  catalog sizes and cost exactness. Benchmarked in the archived study under
  `legacy/experiments/`.
- **Type-stable routing API.** `route_top_k` always returns `RouteResult`;
  format conversion is an explicit function, not hidden state.
- **Abstention over force-routing.** The threshold's operating points are
  measured (risk-coverage / AUROC) in the archived study under
  `legacy/experiments/`, not guessed. Schemas get `additionalProperties: false`
  injected at ingest.
- **Security posture:** see [SECURITY.md](../SECURITY.md).

# tests/ — Test Suite

Run everything: `pytest` (CI runs this on every push; see
`.github/workflows/ci.yml`). No network and no model downloads — encoders are
replaced by deterministic dummy embedders and downstream MCP servers by
recording stubs.

| File | Covers |
| --- | --- |
| `test_dynamic_faiss_router.py` | Routing behavior with a deterministic dummy embedder: best-match routing, threshold rejection, **flat-by-default index**, explicit HNSW opt-in, type-stable `route_top_k`, `to_openai_tools` formatting. |
| `test_mcp_server.py` | The FastMCP bridge end to end with a fake embedder + fake downstream clients: union catalog across servers, cross-server routing, dispatch to the owning server, unknown-tool handling, `get_stats`. |

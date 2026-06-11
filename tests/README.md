# tests/ — Test Suite

Run everything: `pytest` (CI runs this on every push; see
`.github/workflows/ci.yml`).

| File | Covers |
| --- | --- |
| `test_dynamic_faiss_router.py` | Routing behavior with a deterministic dummy embedder: best-match routing, threshold rejection, **flat-by-default index**, explicit HNSW opt-in, type-stable `route_top_k`, `to_openai_tools` formatting. |
| `test_split_hygiene.py` | **The leakage guards.** Asserts every row is annotated, regime-1 buckets partition v1 with no scenario crossing buckets, regime-2 test tools never appear in training, and both regimes rank the full merged corpus. If these fail, the benchmark numbers are invalid — fix the data, never the test. |
| `test_metrics.py` | Hand-computed checks for Recall@k / MRR / NDCG / bootstrap CI and a BM25 sanity test. |
| `test_autonomous_agent.py` | ReAct agent behavior with stub router/clients. |
| `test_hybrid_pipeline.py` / `test_enterprise_runtime.py` / `test_enterprise_backend.py` | Hybrid pipeline orchestration and fallbacks, policy enforcement (path traversal raises `SecurityPolicyViolation`), registry routing, telemetry, workspace change tracking (content-hash based — mtime+size alone proved flaky for same-size writes). |

## Conventions

- No network and no model downloads in tests: encoders are replaced by dummy
  embedders; MCP servers by recording stubs.
- Data-dependent tests (`test_split_hygiene`, `test_metrics`) skip cleanly when
  `experiments/data/` hasn't been generated or pandas/sklearn aren't installed,
  so a minimal library install still passes.

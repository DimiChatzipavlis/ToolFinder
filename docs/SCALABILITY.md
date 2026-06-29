# ToolFinder — Scalability: status & plan

**Honest TL;DR.** ToolFinder already scales on the axis that matters most for "many
tools under one gateway": **feasibility** (it's the only way to expose more tools
than a host can bind) and **cost** (flat context regardless of catalog size). The
real bottleneck we *measured* is **selection quality on large, confusable
catalogs** — not raw performance. This doc states what scales today, what doesn't,
and the prioritized plan, with each item tied to evidence in
[EVALUATION.md](EVALUATION.md).

## What already scales (today)

| Axis | Status | Evidence |
| --- | --- | --- |
| **Feasibility** — expose 100s–1000s of tools | ✅ gateway binds ~3 tools regardless of N (host bind limit is ~128) | `eval_selection_at_scale.py` |
| **Context / cost** — flat in N | ✅ ~10× cheaper at 120 tools (uncached); cache-aware ~6–10× | `eval_gateway_vs_baseline.py`, `bridge_cache_*` |
| **Vector index** | ✅ exact flat <1 ms below 10⁵; HNSW via `TOOLFINDER_INDEX=auto\|hnsw` (auto switches above 50k) | `benchmarks/scaling_bench.py` |
| **Startup with many servers** | ✅ downstream servers now spawn + handshake **concurrently** (`asyncio.gather` in `_build`) | — |
| **Robustness** | ✅ per-server fault isolation — one bad server is skipped, gateway stays up | `tests/test_mcp_server.py::test_failed_downstream_is_isolated` |

## What does NOT scale yet (with the bottleneck named)

**Track A — selection quality at scale (the *value* bottleneck):**
- The stock zero-shot encoder degrades on large confusable catalogs — router
  recall@1 **0.88 → 0.46** going 60 → 574 held-out tools. The gateway's accuracy
  ceiling *is* the router's recall, so at scale routing gets unreliable unless the
  encoder is better. (Evidence: `eval_selection_at_scale.py`.)

**Track B — operational / performance:**
- **Cold start re-embeds the entire catalog on every launch** (no persistence).
- **`refresh()` rebuilds everything** — no incremental per-tool/per-server update,
  and no push-based `tools/list_changed` subscription (E2).
- **Single process, per-query encode** — no batching / concurrency limits under
  high QPS; query encoding (tens of ms) dominates per-request latency.
- **Memory** — the router retains the embedding matrix *and* the FAISS index
  (~2× vector memory) to support hierarchical routing.
- **No exported metrics** (only in-process `get_stats()`).

## The plan (prioritized by impact × effort)

**P0 — selection quality at scale (highest value, attacks the measured bottleneck):**
- **Per-deployment fine-tuned / domain encoder.** Measured to lift recall@1 to
  ~0.99 in-domain (`eval_rerank_finetuned.py`). Ship a fine-tune recipe + the
  `TOOLFINDER_MODEL` hook (already there). *Effort: M · Impact: high.*
- **Auto-enable rerank + hierarchical above a tool-count threshold.** Both exist
  opt-in today (`TOOLFINDER_RERANK`, `TOOLFINDER_HIERARCHICAL`); make them kick in
  automatically for large catalogs. *Effort: S · Impact: medium-high.*

**P1 — startup & persistence:**
- **Persistent embedding cache**, keyed by `(model, schema-hash)` — only re-embed
  *new/changed* tools across restarts and `refresh()`. Design: a cache dir
  (`TOOLFINDER_CACHE_DIR`), an on-disk `hash → vector` store loaded at router init
  and consulted in `ingest_server` before encoding. *Effort: M · Impact: big at 10⁴+ tools.*
- **Batch-encode across all servers** (one encode pass) instead of per-server. *Effort: S.*

**P2 — live updates (E2):**
- Subscribe to downstream `tools/list_changed`; **incrementally** add/remove tools
  in the router (it already supports per-server ingest) instead of a full rebuild.
  *Effort: M · Impact: needed for long-running, mutating deployments.*

**P3 — throughput & ops:**
- Query-embedding cache + encode batching; optionally a separate embedding service;
  concurrency caps. *Effort: M.*
- **E4 — exported/structured metrics** (beyond `get_stats()`). *Effort: S.*
- Memory: drop the retained matrix when hierarchical is off; quantized / mmap index
  for very large catalogs. *Effort: M.*

## How we'll prove each scales (validation plan)

Each lands with a results JSON and honest scope (the [EVALUATION.md](EVALUATION.md) discipline):
- **Quality:** re-run `eval_selection_at_scale.py` with a fine-tuned encoder + rerank → target a higher router R@1 at 574 (currently 0.46).
- **Startup:** cold-start time with vs without the embedding cache at 10³ / 10⁴ tools.
- **Throughput:** latency vs QPS with and without encode batching.

## Honest verdict

**Today ToolFinder is scalable for the common gateway use case** — dozens to a few
hundred tools across several servers: feasible (the only option past the host bind
limit), cheap (flat context), fault-isolated, and now starting servers
concurrently. **It is not yet tuned for 10⁴+ tools at high QPS with strong
accuracy** — that needs **P0** (a better encoder, the measured bottleneck), **P1**
(embedding persistence), and **P2** (incremental live updates). The path is
concrete and the bottleneck is identified, not hand-waved.

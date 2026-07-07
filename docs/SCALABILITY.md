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
- ~~Cold start re-embeds the entire catalog on every launch~~ → **solved (opt-in)**
  by the persistent embedding cache (`TOOLFINDER_CACHE_DIR`, see P1 below).
- **`refresh()` still rebuilds the index structure** (though with the cache it no
  longer re-*encodes* unchanged tools) — no push-based `tools/list_changed`
  subscription (E2).
- **Single process, per-query encode** — no batching / concurrency limits under
  high QPS; query encoding (tens of ms) dominates per-request latency.
- **Memory** — the router retains the embedding matrix *and* the FAISS index
  (~2× vector memory) to support hierarchical routing.
- **No exported metrics** (only in-process `get_stats()`).

## The plan (prioritized by impact × effort)

**P0 — selection quality at scale (highest value, attacks the measured bottleneck):**
🟡 **Measured** (`eval_encoder_at_scale.py`, 60 held-out-server queries, $0 local):
- **Fine-tuned encoder + rerank OFF is the best config at every scale** — R@1 at
  574 tools: stock 0.40 → stock+rerank 0.50 → **fine-tuned(multi-server) 0.583**,
  with **recall@5 holding at 0.933** (so the `find_tools` top-5 pattern stays
  reliable at scale even where exact top-1 degrades).
- **Data-backed rule (holds at scale, matches the in-domain 2×2):** if you can
  fine-tune → `TOOLFINDER_MODEL`=your encoder, rerank **off**; if you can't →
  stock + `TOOLFINDER_RERANK=1` (rerank rescues stock, 0.40→0.50, but *hurts*
  every fine-tuned base, 0.583→0.500).
- **Honest limit:** no tested config fully restores R@1 at 574 (best 0.583) —
  the remaining fix is **per-deployment fine-tuning on your own catalog**
  (measured 0.99 in-domain), so ship the fine-tune recipe. *Effort: M · Impact: high.*
- ✅ *Shipped (data-narrowed):* rerank now **auto-enables** above
  `TOOLFINDER_SCALE_THRESHOLD` — **only** when the encoder is the stock default
  (it degrades fine-tuned bases) and only if the user made no explicit
  `TOOLFINDER_RERANK` choice; hierarchical remains opt-in (recall trade-off).
  Covered by `tests/test_mcp_server.py`. *Default lowered 100 → 25 after the R3
  live study:* a 36-tool catalog already produced a confusable-family routing
  miss (`create_entities` absent from the "store an entity" shortlist) that
  rerank fixes — recovering the live memory task from 2/5 to 5/5.

**P1 — startup & persistence:**
- ✅ **Persistent embedding cache shipped** (opt-in `TOOLFINDER_CACHE_DIR` /
  `RouterHyperparameters.cache_dir`): an on-disk `(model, schema-hash) → vector`
  store (`.npz`, atomic replace) loaded at router init and consulted in
  `ingest_server` — restarts and `refresh()` re-encode **only new/changed tools**
  (a corrupt cache falls back to fresh, never blocks startup). Covered by
  `tests/test_dynamic_faiss_router.py` (zero re-encodes warm, only-new-tools on
  growth). *Remaining:* batch-encode across all servers in one pass (*Effort: S*),
  and a cold-vs-warm startup benchmark at 10³/10⁴ tools.

**P2 — live updates (E2):**
- ✅ **Shipped.** Stdio downstreams that emit `notifications/tools/list_changed`
  now trigger a **debounced incremental refresh** of just that server (the stdio
  client surfaces notifications via an `on_notification` hook; previously they
  were silently dropped). `refresh(server=...)` does the same on demand, and the
  router's `reingest_server` replaces one server's vectors **without re-encoding
  the others** (plus the P1 cache: only changed tools hit the encoder). Covered
  by CI tests (incremental replace, per-server refresh, notification path).
  *Honest scope:* validated with fakes in CI; many reference MCP servers don't
  emit `list_changed` at all, so the manual per-server refresh is the guaranteed
  path. OpenAPI downstreams have no push channel (poll/refresh explicitly).

**P3 — throughput & ops:**
- ✅ **Query-embedding LRU cache shipped** (`RouterHyperparameters.query_cache_size`,
  default 256): identical repeated queries (agent retries, common intents) skip
  the encoder — and encoding dominates route latency. Deterministic, so results
  are byte-identical; CI-tested.
- ✅ **E4 shipped (v0.1 form):** `get_stats()` now reports `uptime_s` and
  `route_latency_ms` (n/p50/p95/max, 500-sample window), every recent route
  carries its latency, and `TOOLFINDER_METRICS_FILE` appends **structured JSONL
  events** (`route`, `refresh`) for external collection. CI-tested. *Deferred:*
  a native Prometheus/OTel exporter.
- *Open:* single-pass encode batching across servers at startup; concurrency
  caps under high QPS.
- *Memory (revised):* the retained embedding matrix is now **load-bearing** for
  P2 incremental updates (reingest slices it), so "drop it when hierarchical is
  off" is retired; very-large-catalog memory work means mmap/quantized indexes.
  *Effort: M.*

## How we'll prove each scales (validation plan)

Each lands with a results JSON and honest scope (the [EVALUATION.md](EVALUATION.md) discipline):
- **Quality:** ✅ done — `eval_encoder_at_scale.py` (grid: 3 encoders × rerank × N).
  Result: fine-tuned+rerank-off lifts R@1 at 574 from 0.40/0.50 to **0.583** with
  R@5 **0.933**; full restoration needs per-deployment fine-tuning.
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

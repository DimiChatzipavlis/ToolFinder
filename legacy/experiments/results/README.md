# experiments/results/ — Generated Evidence

Every number in `reports/report.md` and every figure is generated **from the
files in this folder**, which are in turn produced by the scripts in
`experiments/`. They are committed on purpose: the report's claims must be
verifiable by a grader *without* re-running ~1.5 hours of GPU training, and
`artifact_manifest.json` pins their hashes so a stale report can be detected.
If a number looks wrong, fix the experiment and regenerate — never edit these
files by hand.

## Files and who reads them

| File | Size | What it holds | Read by |
| --- | --- | --- | --- |
| `main_eval.json` | ~50 KB | Per-system metric summaries (R@k, MRR, NDCG, accuracy/P/R/F1, latency, CIs) for all 3 regimes — the human-readable scoreboard | `build_report.py`, `figures.py`, live notebook |
| `diagnostics/main_eval_per_query.json` | ~1.3 MB | Per-query rank + top-1 prediction for every system × regime (~1,000 queries × 16 systems). **Machine data — not meant to be read**, but required for the confusion matrix and any error analysis without re-running models | `figures.py` (confusion matrix) |
| `ood_eval.json` | ~90 KB | OOD scores: AUROC/FPR per subset + the full 101-point threshold sweep (the curve data behind the risk-coverage figure) | `build_report.py`, `figures.py` |
| `biencoder_training.json` | ~20 KB | All 9 training runs: hyperparameters, duration, loss history, per-epoch val MRR (the loss curves) | `build_report.py`, `figures.py` |
| `crossencoder_training.json` | ~3 KB | Same for the 3 cross-encoder runs | `build_report.py` |
| `ablation_representation.json` | ~4 KB | R@1/R@3/MRR per schema representation × system × regime | `build_report.py` |
| `scaling_bench.json` | ~10 KB | Flat-vs-HNSW latency/recall/build/size per tier + encoder latency | `build_report.py`, `figures.py` |
| `poisoning.json` | ~3 KB | Hijack rates per attack strength × system × mitigation | `build_report.py` |
| `calibration.json` | ~3 KB | ECE + reliability bins, raw vs temperature-scaled | `build_report.py` |
| `bridge_scaling_free.json` | ~2 KB | MCP-bridge study, free axis: per-turn schema-token weight + router recall@1/@3 vs catalog size N (14→400) | `bridge_figures.py`, cookbook |
| `bridge_scaling_gpt.json` | ~3 KB | MCP-bridge study, API axis: total tokens + task success for baseline / find+call / route_and_call at N∈{14,60,120} (GPT-5.4) | `bridge_figures.py`, cookbook |
| `bridge_cache_aware.json` | ~2 KB | Cache-aware re-scoring of `bridge_scaling_gpt.json`: billable input tokens + baseline/bridge ratios under prompt caching (uncached / 0.5× / 0.25× / 0.1×). Modeled from per-turn structure | `bridge_cache_aware.py` |
| `bridge_cache_measured.json` | ~6 KB | **Measured** (gpt-5.4) prompt-cache behavior: per-turn `cached_tokens` + cached fraction for the baseline arm (N=14/60/120) and the top-5 `find5` arm (N=60/120), with end-task success. Validates `bridge_cache_aware.json` | `bridge_cache_measured.py` |
| `bridge_selection_accuracy.json` | ~5 KB | Forced single-tool selection accuracy over 8 probes vs catalog size (N=14/60/120), weak (`gpt-4.1-mini`) vs strong (`gpt-5.4`) vs router recall@1 — the context→accuracy test | `bridge_selection_accuracy.py` |
| `rerank_eval.json` | ~1 KB | **Local ($0 API)** retrieval eval of the cross-encoder reranker on the confusable GitHub-MCP catalog (144 unseen queries): recall@1/@3/@5 + MRR, rerank OFF vs ON (recall@1 0.56→0.85) | `eval_rerank.py` |
| `rerank_finetuned_eval.json` | ~1 KB | 2×2 {stock, fine-tuned MiniLM} × rerank {off,on} on the same eval — fine-tune vs rerank vs both (recall@1: stock 0.56, +rerank 0.85, fine-tuned **0.99**; they're alternatives, stacking hurts) | `eval_rerank_finetuned.py` |
| `eval_gateway_vs_baseline.json` | ~1 KB | Strict A/B (gpt-4.1-mini, 3 trials/size): gateway vs binding all N tools, on the real filesystem task. **Success 3/3 both ways at N=15/60/120; gateway tokens ~1.3×/5×/10× cheaper** (uncached) — cost win that grows with N, no accuracy gap on this easy task | `eval_gateway_vs_baseline.py` |
| `eval_selection_at_scale.json` | ~1 KB | Scaled confusable selection (gpt-4.1-mini, 24 held-out queries, 574-tool pool). **Binding all tools is rejected by the API past ~128** (`tools array too long`) → baseline is **N/A at N=250/574**, gateway still runs. Where binding works (N=60) baseline 0.96 **>** gateway 0.75 (gateway is router-bound). Value at scale = **feasibility + cost, not accuracy** | `eval_selection_at_scale.py` |
| `eval_encoder_at_scale.json` | ~2 KB | **P0 grid** ($0, deterministic): {stock, ft-github, ft-multiserver} × rerank × N∈{60,250,574} on 60 held-out queries. Winner: **fine-tuned + rerank OFF** (R@1 at 574: 0.40→**0.583**, R@5 **0.933**); rerank rescues stock but hurts fine-tuned bases | `eval_encoder_at_scale.py` |
| `bridge_ab.json` | ~1 KB | single-N baseline-vs-bridge A/B (superseded by the scaling study) | reference |
| `multiserver_eval.json` | ~1 KB | R1 unseen-server comparison: BM25 / frozen / GitHub-trained / multi-server-trained on 7 held-out servers | README roadmap |
| `biencoder_multiserver_training.json` | ~7 KB | training record for the multi-server bi-encoder (R1) | reference |
| `artifact_manifest.json` | ~7 KB | SHA256 of datasets, results, and (uncommitted) model artifacts | release verification |
| `figures/*.png` | — | Every figure referenced by the report | report, notebooks |

## Why JSON, and why committed

- **JSON because the report is generated, not written.** `build_report.py`
  injects these values into the report; hand-typed numbers were the original
  project's core failure.
- **Committed because results are evidence.** Model weights are *not*
  committed (regenerable from seeds; see `artifact_manifest.json` for their
  hashes) — results are kept because regenerating them requires the weights.
- **Big files are quarantined.** Anything humans shouldn't read lives under
  `diagnostics/`. If a results file ever feels unreadable, the fix is moving
  detail into `diagnostics/`, not deleting evidence.

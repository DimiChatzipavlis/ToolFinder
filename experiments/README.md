# ToolFinder Experiments

The reproducible research pipeline behind `reports/report.md`. Every number in
the report is written by a script here into `results/*.json`; nothing is
hand-typed. The earlier exploratory `academic_research/` folder is kept for
provenance only — this package supersedes it.

## Pipeline

```bash
pip install -e ".[experiments]"
python experiments/run_all.py          # data -> train -> eval -> report
```

Stages (each runnable standalone):

| Stage | Script | Output |
| --- | --- | --- |
| Scenario recovery | `dataset/annotate_scenarios.py` | `data/queries_with_scenarios.csv`, `data/corpus.json` |
| Splits | `dataset/make_splits.py` | `data/splits/regime{1,1b,2}_*.json` |
| OOD sets | `dataset/make_ood.py` | `data/ood/*.csv` (eval-only) |
| Multi-server catalog | `dataset/build_multiserver_catalog.py` | `data/catalogs/multiserver_catalog.json` (544 real tools, 23 providers, apis.guru) |
| Unseen-server regime | `dataset/make_multiserver_queries.py` | `data/queries_multiserver.csv`, `data/corpus_multiserver.json` (574 tools), `data/splits/regime3_unseen_servers.json` |
| Bi-encoder training | `models/biencoder.py` | `artifacts/biencoder/*`, `results/biencoder_training.json` |
| Hard negatives + cross-encoder | `models/hard_negatives.py`, `models/crossencoder.py` | `artifacts/crossencoder/*`, `results/crossencoder_training.json` |
| Main evaluation (3 regimes) | `evaluation/evaluate.py` | `results/main_eval.json` |
| Template-disjoint control (regime 1b) | `evaluation/eval_template_disjoint.py` (after the r1b training run) | `results/template_disjoint_eval.json` |
| Paired significance vs BM25 | `evaluation/significance.py` | `results/significance.json` |
| Open-set rejection | `evaluation/ood.py` | `results/ood_eval.json` |
| Representation ablation | `ablation_representation.py` | `results/ablation_representation.json` |
| Flat-vs-HNSW scaling | `benchmarks/scaling_bench.py` | `results/scaling_bench.json` |
| Description poisoning | `attacks/poisoning.py` | `results/poisoning.json` |
| CE calibration | `evaluation/calibration.py` | `results/calibration.json`, `figures/fig_calibration.png` |
| LLM-in-context baseline | `evaluation/llm_incontext.py` | `results/llm_incontext.json` (requires a local Ollama service; not run in the authoring environment) |
| MCP-bridge scaling study | `bridge_scaling.py` (`--free-only` = no API) | `results/bridge_scaling_{free,gpt}.json` |
| MCP-bridge figures | `bridge_figures.py` | `results/figures/fig_bridge_scaling_{free,api}.png` |
| Figures | `figures.py` | `results/figures/*.png` |
| EDA notebook | `build_eda_notebook.py` | `notebooks/01_eda.ipynb` (executed, outputs committed) |
| Report + manifest | `build_report.py`, `build_manifest.py` | `reports/report.md`, `results/artifact_manifest.json` |

## Why the splits look the way they do

The query datasets follow a `scenario × template` generation grammar
(~5 scenarios × ~10 paraphrase templates per tool). A random row split puts
paraphrases of every test scenario into training: a 1-NN lookup over training
anchors scores ~96% Recall@1 on such a split without reading a single schema.
Scenario grouping fixes the row-level leak but still shares surface templates
across the split, so **regime 1b** additionally holds out templates *and*
scenarios jointly (the template-disjoint control; regime-1 numbers are
in-grammar upper bounds). Splits here are therefore **scenario-grouped**
(regime 1, unseen queries), **doubly-disjoint** (regime 1b), and
**tool-disjoint** (regime 2, unseen tools), both ranked against the full
30-tool corpus. `tests/test_split_hygiene.py` enforces this in CI — if those
tests fail, fix the data, never the test.

## Conventions

- Model weights live under `experiments/artifacts/` (gitignored, regenerable
  from seeds); datasets, splits, results JSON, and figures are committed.
  Derived training data that requires the artifacts to regenerate
  (`data/crossencoder_train_pairs.csv`) is gitignored too.
- `results/*.json` are human-readable summaries; bulky per-query machine data
  is quarantined in `results/diagnostics/`. See `results/README.md` for what
  each file holds, who reads it, and why results are committed at all.
- Trained systems run with seeds {13, 42, 1337}; tables report mean ± std over
  seeds, and per-system 95% bootstrap CIs over queries.
- All retrieval systems implement `rank(query) -> ordered tool names` and are
  evaluated through the same code path (`evaluation/evaluate.py`).

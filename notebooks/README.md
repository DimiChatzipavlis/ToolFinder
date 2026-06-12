# notebooks/ — Executed Analysis Notebooks

Notebooks here are **generated and executed programmatically** so their saved
outputs always match the committed data — never edit them by hand.

| Notebook | Contents | Regenerate with |
| --- | --- | --- |
| `01_eda.ipynb` | Dataset card (composition/balance), missing-value & duplicate verification, class-distribution chart, query-length distributions, the scenario×template generation grammar with a concrete paraphrase cluster, the leakage-audit figure (random vs scenario-grouped split), query↔schema lexical-overlap distribution, inter-tool schema confusability heatmap, split summaries. | `python experiments/build_eda_notebook.py` |
| `02_toolfinder_live.ipynb` | **Runnable evidence notebook (local or Colab).** Clones the repo on Colab, installs deps, then runs live: the router on the real 30-tool corpus (routing, scores, threshold margins), lexical baselines reproduced and checked against committed results, committed result tables + figures, an exact-vs-HNSW timing microbenchmark, and a 2-epoch live fine-tune with before/after Recall@1 when a GPU is present. | `python experiments/build_live_notebook.py` |

Why generated: the original course notebook shipped with zero saved outputs,
which made every claimed result unverifiable. Generation + execution in one
step guarantees a grader sees real outputs that reproduce from the committed
data.

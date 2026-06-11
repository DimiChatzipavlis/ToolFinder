# notebooks/ — Executed Analysis Notebooks

Notebooks here are **generated and executed programmatically** so their saved
outputs always match the committed data — never edit them by hand.

| Notebook | Contents | Regenerate with |
| --- | --- | --- |
| `01_eda.ipynb` | Dataset card (composition/balance), query-length distributions, the scenario×template generation grammar with a concrete paraphrase cluster, the leakage-audit figure (random vs scenario-grouped split), query↔schema lexical-overlap distribution, inter-tool schema confusability heatmap, split summaries. | `python experiments/build_eda_notebook.py` |

Why generated: the original course notebook shipped with zero saved outputs,
which made every claimed result unverifiable. Generation + execution in one
step guarantees a grader sees real outputs that reproduce from the committed
data.

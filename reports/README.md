# reports/ — Research Report

| File | What it is | Regenerate with |
| --- | --- | --- |
| `report.md` | **The course report.** Structured to the course rubric: Abstract (~150 words) → Introduction → Related Work → Methodology (data/EDA/preprocessing + the two deep models) → Experiments & Results (Recall@k/MRR/NDCG, Accuracy/Precision/Recall/F1, ROC-AUC for open-set rejection, loss curves, scaling, ablation, poisoning, calibration) → Discussion & Limitations (why the winner won, difficulties encountered) → Conclusions → References. **Every number is injected from `experiments/results/*.json`** — nothing is hand-typed. | `python experiments/build_report.py` |

Workflow: run experiments (`python experiments/run_all.py`) → results land in
`experiments/results/` → `build_report.py` renders this folder. If a number in
the report looks wrong, fix the experiment, never the prose.

The runnable companion that reproduces the evidence live (local or Colab) is
[notebooks/02_toolfinder_live.ipynb](../notebooks/02_toolfinder_live.ipynb).
For a presentation-ready figure set, everything referenced by the report lives
in `experiments/results/figures/`.

# reports/ — Research Report and Paper Draft

| File | What it is | Regenerate with |
| --- | --- | --- |
| `report.md` | The full research report. **Every number is injected from `experiments/results/*.json`** — abstract, main tables (3 regimes), training summary, OOD/threshold analysis, scaling study, representation ablation, poisoning attack, calibration, related work, limitations, conclusions. Nothing is hand-typed. | `python experiments/build_report.py` |
| `paper.md` | Condensed workshop-style draft distilled from the report. | hand-maintained, numbers from `report.md` |

Workflow: run experiments (`python experiments/run_all.py`) → results land in
`experiments/results/` → `build_report.py` renders this folder. If a number in
the report looks wrong, fix the experiment, never the prose.

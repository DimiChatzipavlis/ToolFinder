# academic_research/ — Raw Source Data

This folder now contains **only the raw query datasets** that everything else
derives from. The exploratory code and notebooks that used to live here were
superseded by the reproducible pipeline in [experiments/](../experiments/) and
removed in the 2026-06 cleanup (recoverable from git history).

| File | What it is |
| --- | --- |
| `mcp_routing_dataset.csv` | 750 author-templated queries over 15 GitHub MCP tools (50/tool). Training-eligible after scenario-grouped splitting. |
| `mcp_routing_dataset_v2.csv` | 750 queries over 15 *disjoint* GitHub MCP tools. Evaluation-only (unseen-tool regime). |
| `MODEL_ARTIFACTS.md` | Policy: model weights are never committed; artifacts are pinned by SHA256 manifest instead. |

Do not edit the CSVs by hand — they are the provenance anchor. Derived,
annotated, and split versions live in `experiments/data/` and are produced by
`experiments/dataset/annotate_scenarios.py` and `make_splits.py`. Dataset
statistics, known biases, and the leakage analysis are documented in
[experiments/data/DATASET_CARD.md](../experiments/data/DATASET_CARD.md).

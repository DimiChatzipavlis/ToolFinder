# research/ — Archived Material (not part of the MCP server)

This folder holds everything that is **not** needed to run or develop the
ToolFinder MCP server. It is kept for provenance and reproducibility, not as a
maintained part of the tool — nothing here is imported by `toolfinder/` or the
bridge server.

| Path | What it is |
| --- | --- |
| `experiments/` | The original research pipeline: datasets, leakage-controlled splits, bi-/cross-encoder training, baselines, the evaluation regimes, OOD/calibration, the MCP-bridge scaling study, and the report generator. The committed `experiments/results/*.json` are the evidence behind the numbers quoted in the top-level README; figures and the report regenerate from them via `build_report.py`. (Trained weights under `experiments/artifacts/` and the live `.env` are gitignored.) |
| `academic_research/` | The raw source datasets the study was built from (provenance anchor). |
| `examples/` | Earlier demos and the executed evidence notebooks (EDA + live routing), including the legacy LangGraph integration. |
| `audit_logs/` | Build/run logs from the research phases — safe to delete. |

The deployable tool lives at the repository root: `toolfinder/`,
`ToolFinder_mcp_server.py`, `docs/MCP_SERVER.md`, and `tests/`.

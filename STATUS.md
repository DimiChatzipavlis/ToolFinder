# Project Status

_Last updated: 2026-06-12_

Single source of truth for the repository's current state. Three dated audit
documents (`ARCHITECTURE_REPORT.md`, `SYSTEM_REALITY_REPORT.md`,
`ENTERPRISE_SYSTEM_REPORT.md`) were removed in the 2026-06 cleanup; they are
preserved in git history, and every open finding they recorded has been
addressed as below.

## Research pipeline (course deliverable)

- Benchmark, training, evaluation, and report live in [experiments/](experiments/)
  (see its README for the stage map). `academic_research/` is kept for
  provenance only.
- Leakage-controlled splits (scenario-grouped / unseen-tool / unseen-server)
  are enforced in CI by `tests/test_split_hygiene.py`.
- Results are generated artifacts: `experiments/results/*.json`,
  `experiments/results/figures/`, `reports/report.md`,
  `notebooks/01_eda.ipynb` (executed, outputs committed).
- Known environment limitation: the LLM-in-context baseline
  (`experiments/evaluation/llm_incontext.py`) requires a local Ollama service
  and has not been run on the authoring machine.

## Audit findings closed since the April reports

| Finding (SYSTEM_REALITY_REPORT) | Resolution |
| --- | --- |
| Duplicate-action signature not canonical (key-order evasion) | Arguments canonicalized via sorted-key JSON before hashing in `toolfinder/autonomous_agent.py` |
| Hybrid pipeline executed repeated tool calls | Canonicalized per-response dedup guard in `Enterprise/runtime/openclaw_hybrid_pipeline.py` |
| Path traversal enforcement not invariant across configs | Path arguments resolve against `workspace_root` (never process cwd) with `realpath` before containment checks in `Enterprise/runtime/policy.py` |
| `_send_message` could block indefinitely on stdio backpressure | Bounded `drain()` with `request_timeout_s` in `toolfinder/mcp_adapter.py` |

## Known mismatch: shipped default vs evaluated best

`UniversalMCPRouter` defaults to a zero-shot checkpoint; the benchmark's best
numbers come from fine-tuned artifacts that are regenerable but not committed.
Zero-shot dense retrieval underperforms BM25 on this benchmark. This is
disclosed in the README ("Performance note"), the router docstring, and the
report's limitations; loading a fine-tuned artifact is a one-line change.

## Runtime changes worth knowing

- `UniversalMCPRouter` defaults to exact `IndexFlatIP`; HNSW is opt-in
  (`RouterHyperparameters.index_type`), justified by the measured scaling
  benchmark (`experiments/benchmarks/scaling_bench.py`).
- `route_top_k` is type-stable (`list[RouteResult]` always); use
  `toolfinder.to_openai_tools()` for bindable schemas.
- `WorkspaceChangeTracker` content-hashes files ≤1MB (mtime+size alone missed
  same-size writes within one timestamp tick).

## Honest claim boundary

Quality numbers hold for the tested catalogs (30-tool GitHub corpus; 574-tool
multi-server corpus) and the documented query provenance (synthetic,
author/LLM-written). Latency numbers hold for the documented hardware and
single-thread FAISS configuration. Nothing in this repository demonstrates
behavior on production traffic.

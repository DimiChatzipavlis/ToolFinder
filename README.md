# ToolFinder: Semantic Tool Routing for MCP

ToolFinder is a retrieval-based routing layer for Model Context Protocol (MCP) tool ecosystems. Instead of binding every available tool schema into an LLM's context window, it embeds tool schemas and user intents into a shared vector space and retrieves only the top-k relevant tools before inference. This keeps prompts small, reduces tool-selection errors in small local models, and separates tool *selection* from tool *execution*.

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![MCP](https://img.shields.io/badge/MCP-compatible-black)](https://modelcontextprotocol.io/)
[![FAISS](https://img.shields.io/badge/retrieval-FAISS-orange)](https://github.com/facebookresearch/faiss)
[![LangGraph](https://img.shields.io/badge/integration-LangGraph-green)](https://github.com/langchain-ai/langgraph)

## The Problem: Context Bloat

Binding dozens of MCP tool schemas to a small local model (e.g. `llama3.2`) fills the prompt with irrelevant structure before reasoning begins. Similar APIs collide in-context, tool-selection errors rise, and smaller models emit malformed calls under long-prompt pressure — the "lost in the middle" failure mode applied to tool orchestration.

## The Approach

- A sentence-transformer bi-encoder embeds queries and MCP schemas into the same vector space.
- A FAISS index retrieves the top-k candidate tools for each query. The default index is **exact flat inner-product search** (`IndexFlatIP`): at realistic MCP catalog sizes, exact search is faster than approximate HNSW graph traversal and fully deterministic. HNSW remains available via `RouterHyperparameters(index_type="hnsw")` for very large catalogs (see the scaling benchmark in `experiments/`).
- A similarity threshold rejects queries that match no tool well enough, instead of force-routing them.
- The model then reasons over a small, relevant tool surface instead of the entire ecosystem.

The datasets, trained models, evaluation protocol, and benchmark results live in [experiments/](experiments/) (the maintained research pipeline; `academic_research/` contains earlier iterations kept for provenance).

## Quickstart

> Install PyTorch with your required hardware acceleration first (e.g. `pip install torch --index-url https://download.pytorch.org/whl/cu121`), otherwise pip will default to CPU inference.

```bash
python -m pip install -e .
```

Minimal integration:

```python
from toolfinder import UniversalMCPRouter, to_openai_tools

router = UniversalMCPRouter()          # exact flat index by default
for tool in mcp_server_tools:          # raw MCP tool payloads
    router.add_tool(tool)
router.build_index()

results = router.route_top_k("Write a summary to output.txt", k=2)
# results: list[RouteResult] with .server_name, .tool_name, .schema, .score

llm_tools = to_openai_tools(results)   # bindable function-calling schemas
```

Optional extras:

```bash
python -m pip install -e ".[dev]"          # pytest
python -m pip install -e ".[langgraph]"    # LangChain/LangGraph integration
python -m pip install -e ".[experiments]"  # research pipeline (pandas, sklearn, matplotlib)
```

## Empirical Results

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/DimiChatzipavlis/ToolFinder/blob/main/notebooks/02_toolfinder_live.ipynb) — runnable evidence notebook: clones this repo, routes live on the real corpus, reproduces baseline rows, renders the committed results, and (on GPU) fine-tunes live.

Two kinds of evidence, with their scope stated plainly:

**Retrieval quality** is evaluated in [experiments/](experiments/) under leakage-controlled splits (unseen queries and unseen tools), against lexical baselines (BM25, TF-IDF) and frozen-encoder baselines, with seed-averaged metrics and confidence intervals. See `experiments/results/` and `reports/report.md` for current numbers.

**End-to-end system effect** is measured by a small A/B harness that runs identical filesystem tasks against a local model with (a) all tools bound vs (b) top-k routed tools. It is a smoke-scale benchmark (n=3 tasks), not a statistical claim:

<!-- EVAL_TABLE_START -->
_Last auto-updated: 2026-03-16 17:01:57_

| Metric | Naive Baseline | ToolFinder Enabled |
| --- | --- | --- |
| Tasks Run | 3 | 3 |
| Average Tools In Context | 14 | 2 |
| Average Context Payload (Chars) | 9106 | 1450 |
| Average Total Latency (s) | 57.51 | 14.47 |
| Average Inference Latency (s) | 57.47 | 14.39 |
| Successful Tool Calls | 3/3 | 3/3 |
| Expected Tool Matches | 3/3 | 3/3 |
| State Verified | 3/3 | 3/3 |

Task outcomes:
- T1_READ: naive=`read_text_file` verified=`True`, toolfinder=`read_text_file` verified=`True`
- T2_WRITE: naive=`write_file` verified=`True`, toolfinder=`write_file` verified=`True`
- T3_LIST: naive=`list_directory` verified=`True`, toolfinder=`list_directory` verified=`True`
<!-- EVAL_TABLE_END -->

Regenerate with `python examples/eval_toolfinder.py --update-readme` (preserve the markers).

## Hardening Features

- Strict schema enforcement injects `additionalProperties: false` into object schemas to reject speculative keys.
- AST recovery parsing salvages Python-style dicts when strict JSON parsing of local-model output fails (literals only — no code execution).
- ReAct execution loops let the agent observe failures and retry rather than crash on the first malformed response.
- Idempotency guards and bounded scratchpads limit repeated actions and runaway context growth.
- Threshold-based rejection abstains on out-of-scope queries rather than force-routing them. Routing is similarity-based and can still select a wrong tool for ambiguous queries; treat destructive tools accordingly (see `reports/report.md`, Limitations).

## Repository Layout

Every folder has its own README with details.

- [`toolfinder/`](toolfinder/README.md) — core library: router, MCP stdio client, autonomous ReAct agent, recovery parsing.
- [`experiments/`](experiments/README.md) — the research pipeline: datasets, leakage-controlled splits, training (bi- and cross-encoders), baselines, three evaluation regimes, OOD/threshold analysis, Flat-vs-HNSW scaling benchmark, poisoning attack, calibration, figures, report generation.
- [`examples/`](examples/README.md) — runnable demos: A/B harness, multi-server agent demo, LangGraph integration (require local Ollama + Node).
- [`Enterprise/`](Enterprise/README.md) — optional hybrid runtime (HTTP API, policy engine, executor, telemetry). Out of scope for the research evaluation.
- [`tests/`](tests/README.md) — unit tests incl. CI-enforced split-hygiene guards (run `pytest`).
- [`notebooks/`](notebooks/README.md) — executed EDA notebook with committed outputs.
- [`reports/`](reports/README.md) — research report (auto-rendered from results) and paper draft.
- [`academic_research/`](academic_research/README.md) — raw source datasets only (provenance anchor).
- [STATUS.md](STATUS.md) — current state, closed audit findings. [SECURITY.md](SECURITY.md) — threat model and mitigations.

## Scope Notes

This repository provides a runtime library, a research pipeline, and examples. It does not package a production multi-node deployment (service discovery, secrets, auth, load balancing are out of scope; see SECURITY.md for the full residual-risk list). Latency and quality numbers are measured on the configurations documented in `experiments/`; claims do not extend beyond the tested catalog sizes.

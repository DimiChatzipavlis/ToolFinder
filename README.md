# ToolFinder: Semantic Tool Routing for MCP

ToolFinder is a retrieval-based routing layer for Model Context Protocol (MCP) tool ecosystems. Instead of binding every available tool schema into an LLM's context window, it embeds tool schemas and user intents into a shared vector space and retrieves only the top-k relevant tools before inference. This keeps prompts small, reduces tool-selection errors in small local models, and separates tool *selection* from tool *execution*.

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![MCP](https://img.shields.io/badge/MCP-compatible-black)](https://modelcontextprotocol.io/)
[![FAISS](https://img.shields.io/badge/retrieval-FAISS-orange)](https://github.com/facebookresearch/faiss)

> This repository is the **MCP server tool**: the routing library and the bridge server. The original empirical study — datasets, training, leakage-controlled evaluation, the bridge scaling experiments, and the report generator — is archived under [`legacy/`](legacy/README.md) for provenance; it is not needed to run or develop the tool.

## Status — OSS v0.1 (early-stage), not yet production-grade

ToolFinder ships as an **early-stage, research-backed v0.1**: clean, unit-tested, packaged, and every performance claim has a reproducible script. It is **not** production-grade yet — validation is deliberately scoped (one task family, n=1, filesystem-only live execution) and some resilience/observability features are partial. Good fit **today**: large or multi-server tool catalogs and weak/local models. For production, bring your own validation and see the gaps below.

| Area | ✅ Ready in v0.1 | 🔜 Needed for production |
| --- | --- | --- |
| Routing | exact FAISS flat (default), opt-in HNSW, opt-in server-aware hierarchical, threshold abstention | — |
| Bridge | multi-server union + dispatch (**MCP and OpenAPI**), env-based downstream auth, one-shot reconnect, `get_stats` | push-based `tools/list_changed`; exported metrics |
| Evidence | cost modeled **and** measured; top-5 + selection accuracy measured | live multi-server at scale; repeats/error bars |
| Quality | any open encoder; honest defaults | stronger/fine-tuned default; poisoning mitigations |
| Packaging | `pip install`, `toolfinder-mcp`, MIT, CI (lint+tests) | PyPI; CHANGELOG/CONTRIBUTING; integration tests |

See [Roadmap & release readiness](#roadmap--release-readiness) for the full checklist and the next immediate steps.

## The Problem: Context Bloat

Binding dozens of MCP tool schemas to a small local model (e.g. `llama3.2`) fills the prompt with irrelevant structure before reasoning begins. Similar APIs collide in-context, tool-selection errors rise, and smaller models emit malformed calls under long-prompt pressure — the "lost in the middle" failure mode applied to tool orchestration.

## The Approach

- A sentence-transformer bi-encoder embeds queries and MCP schemas into the same vector space.
- A FAISS index retrieves the top-k candidate tools for each query. The default index is **exact flat inner-product search** (`IndexFlatIP`): at realistic MCP catalog sizes, exact search is faster than approximate HNSW graph traversal and fully deterministic. HNSW remains available via `RouterHyperparameters(index_type="hnsw")` for very large catalogs.
- A similarity threshold rejects queries that match no tool well enough, instead of force-routing them.
- The model then reasons over a small, relevant tool surface instead of the entire ecosystem.

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

> **Shipped default vs evaluated-best.** Out of the box the router loads a
> *zero-shot* sentence-transformer checkpoint (`all-MiniLM-L6-v2`, auto-downloaded).
> The benchmark's headline retrieval numbers come from **fine-tuned** weights,
> produced by the archived pipeline under `legacy/experiments/` and intentionally
> not committed. Zero-shot dense retrieval is measurably weaker — on the
> benchmark, frozen MiniLM scores *below BM25*. To match the reported quality,
> fine-tune via `legacy/experiments/` and point the router at the result:
> `UniversalMCPRouter(model_name="path/to/your/fine-tuned/encoder")`.

Optional extras:

```bash
python -m pip install -e ".[dev]"          # pytest
```

## Choosing the embedding model

Routing uses **any open [sentence-transformers](https://www.sbert.net/) bi-encoder you choose** — nothing is hard-wired or trained into the tool. Override the default:

- **Server:** set `TOOLFINDER_MODEL` (e.g. `TOOLFINDER_MODEL=BAAI/bge-small-en-v1.5`).
- **Library:** `UniversalMCPRouter(model_name="...")` — a HuggingFace id or a local path to fine-tuned weights.

What to expect (measured on the archived GitHub-MCP study — direction, not a guarantee):

- **On small catalogs of distinct tools the model barely matters** — stock `all-MiniLM-L6-v2` already routes at recall@1 = 1.0, so a heavier encoder mostly buys latency.
- **On confusable or out-of-domain catalogs it matters a lot** — there, *frozen* MiniLM scored *below* BM25, a fine-tuned encoder beat it, and bigger stock encoders (MPNet, BGE) helped modestly.
- **Trade-off:** MiniLM (384-d) is fastest/smallest; MPNet/BGE are more accurate but slower to encode — and *encoding*, not search, is the latency bottleneck. Fine-tune only when your catalog has near-duplicate tools or you observe wrong routes.

## ToolFinder as an MCP Server (routing bridge)

`ToolFinder_mcp_server.py` runs ToolFinder as a **Model Context Protocol server that sits between an LLM agent and one or more downstream MCP servers — or OpenAPI REST APIs** (filesystem, git, memory, a Swagger/OpenAPI service, …). Rather than exposing every downstream catalog to the agent — which grows the prompt with every tool — the bridge embeds the **union** of all downstream tools once and exposes a few routing tools, dispatching execution to whichever server owns the chosen tool. It is drop‑in for any MCP host (Claude Desktop, Cursor, …) with no host code changes.

Register it in an MCP host with a config listing the servers to bridge (`mcp_servers.example.json`). **Use absolute paths to your Python interpreter and the script** — MCP hosts don't inherit your shell `PATH`, so a bare `"python"` won't launch:

```jsonc
"mcpServers": {
  "toolfinder": {
    "command": "C:\\Users\\you\\AppData\\Local\\Programs\\Python\\Python312\\python.exe",
    "args": ["C:\\path\\to\\ToolFinder\\ToolFinder_mcp_server.py"],
    "env": { "TOOLFINDER_CONFIG": "C:\\path\\to\\mcp_servers.json" }
  }
}
```

(Find your interpreter with `python -c "import sys; print(sys.executable)"`.) After `pip install -e .` the server is also available as the `toolfinder-mcp` console command and `python -m toolfinder.mcp_server`. Tools exposed: `find_tools(query)` (discover top‑k relevant tools), `call_tool(name, args)` (execute one), `route_and_call(intent, args)` (route + execute in one hop), `catalog_size()`, `get_stats()` (routing observability), `refresh()` (re-index after downstream tool changes).

**Two patterns — prefer `find_tools`+`call_tool`:** the agent sees the *retrieved* tools' schemas (top‑k, not the whole catalog), so it fills arguments correctly while the prompt stays small. `route_and_call` is the lowest‑token option but is **argument‑blind** (the agent supplies `arguments` without seeing the tool's schema) — reliable only for simple‑argument tools.

**Downstream entries** can be MCP servers (`command`/`args`) **or OpenAPI REST APIs** (`type: "openapi"` with `spec_url`/`spec_file`, optional `auth` resolved from environment variables — never inlined in config or logged). See `mcp_servers.example.json`.

**Measured (GPT‑5.4, create→edit→read task, 100% success in every configuration):** as the catalog grows, the baseline that binds all tools scales ≈ linearly (6.4k → 47.9k total tokens for N = 14 → 120), while the bridge stays flat — `route_and_call` is **~15× cheaper at 120 tools** and wins at every size; `find_tools`+`call_tool` is constant (~11.8k) and wins beyond ~30 tools. The router keeps **recall@1 = 1.0 even with the target tool buried among 386 distractors.** The bridge's value is **cost/context that scales with catalog size**, not selection accuracy for already‑capable models. (These are *uncached* token counts. A modeled cache-aware re-scoring — [`legacy/experiments/bridge_cache_aware.py`](legacy/experiments/bridge_cache_aware.py) — shrinks the N=120 `route_and_call` advantage from ~16× to **~6–10×** depending on the cache rate: the baseline's large tool block becomes a cached **read**, while the bridge's ~330‑token prefix sits below the ~1024‑token cache floor and gets no discount. The bridge still wins at scale but can be **break‑even at small N**. A *measured* version logging the API's `cached_tokens` is the remaining step.) Full study and figures: [`legacy/experiments/`](legacy/README.md).

Full cookbook (install, host registration, tool API, config, results, security): [docs/MCP_SERVER.md](docs/MCP_SERVER.md).

## Routing Safety

The protections below are what the tool actually enforces (full threat model and residual risks in [SECURITY.md](SECURITY.md)):

- **Strict schema enforcement** injects `additionalProperties: false` into object schemas at ingest, so the agent can't pass speculative keys a downstream tool would reject.
- **Threshold-based abstention** (`min_cosine_similarity`) rejects out-of-scope queries instead of force-routing them, with top1–top2 ambiguity-margin logging. Routing is similarity-based and can still pick a wrong tool for genuinely ambiguous queries; treat destructive downstream tools accordingly.
- **Safe downstream spawning** — `DynamicMCPClient` spawns servers with argument lists (never `shell=True`), correlates requests with per-request timeouts, and drains pending requests on shutdown.
- **No unsafe deserialization** — encoder weights load via safetensors; the FAISS index is built in-process, never loaded from disk.

## Repository Layout

- [`toolfinder/`](toolfinder/README.md) — core library: the FAISS router, the MCP stdio client, the OpenAPI adapter, and the FastMCP bridge server.
- [`ToolFinder_mcp_server.py`](ToolFinder_mcp_server.py) — entry-point shim for the MCP routing-bridge server; cookbook in [`docs/MCP_SERVER.md`](docs/MCP_SERVER.md).
- [`tests/`](tests/README.md) — unit tests for the router and the bridge (`pytest`).
- [`legacy/`](legacy/README.md) — archived research pipeline, datasets, notebooks, and demos (not part of the tool).
- [SECURITY.md](SECURITY.md) — threat model and mitigations.

## Roadmap & release readiness

### Shipped in v0.1 (done)

- **E1 — Multi-server bridge.** ✅ One ToolFinder fronts several downstream MCP servers via `TOOLFINDER_CONFIG`, routing across the union and dispatching to the owning server.
- **E3 — Package + entry point.** 🟡 Mostly done — `pip install` exposes `toolfinder-mcp` and `python -m toolfinder.mcp_server` (PyPI publish pending).
- **H1 — Hierarchical, server-aware routing.** ✅ Opt-in. `UniversalMCPRouter.route_top_k_hierarchical` + `TOOLFINDER_HIERARCHICAL` / `TOOLFINDER_ROUTE_SERVERS`: stage 1 ranks servers by tool-embedding centroid, stage 2 picks within the top `n_servers`. A **precision/scale** win, not latency (encoding dominates; flat search is already <1 ms). Honest recall trade-off — `n_servers` is tunable (covered by tests). Learned semantic categories remain a later extension.
- **M1 — Cache-aware + quality measurement.** ✅ Modeled *and* measured (gpt-5.4): live `cached_tokens` confirm the cache model within ~1–2% at N≥60 (cached share 18%→70%→73% for N=14→120); top-5 selection completes the task 100%; selection accuracy is router **100%** vs weak `gpt-4.1-mini` **62–75%** vs strong `gpt-5.4` **88–100%** (routing helps weak models; for strong models the value is cost). Scripts: [`bridge_cache_aware.py`](legacy/experiments/bridge_cache_aware.py), [`bridge_cache_measured.py`](legacy/experiments/bridge_cache_measured.py), [`bridge_selection_accuracy.py`](legacy/experiments/bridge_selection_accuracy.py).
- **G1 — OpenAPI gateway + downstream auth.** ✅ Downstream entries can be REST APIs described by an OpenAPI 3.x spec (`type: "openapi"`), routed identically to MCP servers; auth (bearer / API-key header or query) is resolved from environment variables. [`toolfinder/openapi_adapter.py`](toolfinder/openapi_adapter.py).

### Required for a production release (not yet)

- **E2 — Live tool-change.** Push-based `tools/list_changed` subscription with auto re-index (today: one-shot reconnect + manual `refresh()`).
- **R3 — Live multi-server validation.** Real downstream servers + real tasks, with repeats and error bars (today: one task family, filesystem-only live execution, n=1, synthetic distractor catalogs). **The biggest credibility gap.**
- **E4 — Exported metrics.** Structured/exported observability (today: `get_stats()` + logs).
- **E5 — Stronger default encoder.** Ship or recommend a fine-tuned/stronger encoder (today: zero-shot MiniLM, modest on confusable catalogs).
- **Poisoning mitigations.** Length cap / embedding-anomaly score / rerank were studied (in `legacy/`) but **not implemented** — needed before fronting untrusted downstream servers.

### Next immediate steps

1. **Release hygiene (before any public push):** rotate the API key that has sat in your local `.env`, confirm `.env` is untracked in every branch, add `CHANGELOG.md` + `CONTRIBUTING.md`, tag `v0.1.0`, and decide GitHub-only vs PyPI.
2. **R3 — live multi-server study** with repeats/error bars — the single highest-value step toward "trustworthy in production."
3. **E2 — push-based tool-change** (auto re-index).
4. *Optional / research:* **E5** stronger default, **E4** exported metrics, learned categories (H1 extension), and the broader study (human-written queries, benchmark release) in [`legacy/`](legacy/README.md).

## Scope Notes

This repository provides the routing library and the MCP bridge server. It does not package a production multi-node deployment (service discovery, secrets, auth, load balancing are out of scope; see [SECURITY.md](SECURITY.md) for the full residual-risk list). Latency and quality numbers are measured on the configurations documented under `legacy/experiments/`; claims do not extend beyond the tested catalog sizes.

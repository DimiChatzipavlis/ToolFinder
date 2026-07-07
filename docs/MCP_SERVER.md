# ToolFinder MCP Server — Cookbook

`ToolFinder_mcp_server.py` turns ToolFinder into a **Model Context Protocol (MCP)
server that bridges an LLM agent to one or more downstream MCP servers — or
OpenAPI REST APIs** (filesystem, git, memory, a Swagger/OpenAPI service, …).
Instead of exposing every downstream catalog to the agent — which grows the
prompt linearly and pressures the context window — the bridge exposes a tiny,
fixed set of routing tools and selects the relevant downstream tool(s) with the
dense retriever.

It is a thin wrapper over tested components: `UniversalMCPRouter` (selection),
`DynamicMCPClient` (stdio MCP execution), and `OpenAPIClient` (REST execution).

> **Want the "ToolFinder is my only MCP server" setup** (memory + SQL + filesystem
> + REST APIs all under one gateway, with copy-paste host configs)? See
> **[USE_AS_GATEWAY.md](USE_AS_GATEWAY.md)**. This cookbook is the deeper reference.

---

## When it helps — and when it doesn't (measured, not asserted)

Measured on the filesystem server (create → edit → read a file), GPT‑5.4 as the
agent, catalog padded with real distractor tools from a 574‑tool multi‑server
pool. **Every configuration completed the task (100% success).** Numbers are
total task tokens. The full study (code, data, figures) is archived under
`research/experiments/`.

| Catalog size N | Baseline (all tools bound) | `find_tools`+`call_tool` | `route_and_call` (single) |
|---|---|---|---|
| 14 | 6,430 | 11,753 | **4,261** |
| 60 | 22,512 | **11,779** | **4,674** |
| 120 | 47,872 | **11,779** | **3,200** |

- **Baseline cost grows with the catalog** (≈ linear): every turn re‑sends all N
  schemas. Per‑turn schema weight rises 2,092 → 61,384 tokens for N = 14 → 400.
- **`route_and_call` (single tool) wins at every size**, stays flat (~3–5k), and
  is **~15× cheaper than baseline at N = 120**.
- **`find_tools`+`call_tool` is constant in N (~11.8k)** — it never binds the
  catalog — so it *loses* below ~30 tools (extra round‑trips) but **wins beyond**.
- **Selection accuracy is not the win.** A capable model already selects
  correctly among 120 distinct tools; the bridge's value is **token/context
  cost**, and it **scales with catalog size**. The router itself stays at
  **recall@1 = 1.0 even with the correct tool buried among 386 distractors.**

> **Cost caveat (honest — modeled + measured).** The table is *uncached* token totals.
> A cache-aware re-scoring (`research/experiments/bridge_cache_aware.py` →
> `results/bridge_cache_aware.json`) models prompt caching from the per-turn
> structure: the baseline's tool block is a cached **read** on later turns, but
> the bridge arms' ~330-token prefix is below the ~1024-token cache floor and
> gets no discount. Net at N=120, `route_and_call` goes from ~16× cheaper
> (uncached) to **~6–10× cheaper** (cached, depending on rate); at N≤14 the
> multi-call `find_tools`+`call_tool` arm can actually cost **more** than the
> cached baseline. The bridge's cost win is real but emerges **at scale**. A live
> measurement (`bridge_cache_measured.py`, gpt-5.4) now confirms the model within
> ~1–2% at N≥60 — the baseline's API-cached input share rises 18%→70%→73% for
> N=14→120 — and **top-5 selection works**: GPT-5.4 completed the task 100% from a
> 5-tool shortlist. See `results/bridge_cache_measured.json`.

**Rule of thumb:** use the bridge when the agent faces **many** tools (dozens+)
or **multiple** MCP servers, or runs a **small/local** model that struggles to
select in‑context. For a handful of distinct tools and a strong model, bind them
directly — the bridge adds overhead for no accuracy gain (though `route_and_call`
still saves tokens even there).

*Measured (`research/experiments/bridge_selection_accuracy.py`):* on 8 forced
single-tool probes the router selects correctly at **100%** at every catalog
size, while a weak model (`gpt-4.1-mini`) manages only **62–75%** and strong
`gpt-5.4` **88–100%** — the accuracy benefit of routing is real for weak models
and marginal for strong ones (whose bridge value is cost). A clean
catalog-size-distraction curve is not established at these sizes (small sample).

---

## Install

```bash
pip install -e .          # installs the bridge + the `toolfinder-mcp` command
# Node is required to launch downstream npx-based MCP servers (e.g. filesystem)
```

## Run (stdio transport)

```bash
# multi-server (recommended): front several downstream servers from one config
TOOLFINDER_CONFIG=mcp_servers.json toolfinder-mcp

# or zero-config: a single filesystem server rooted at ./sandbox
TOOLFINDER_FS_ROOT=./sandbox toolfinder-mcp

# equivalents: `python -m toolfinder.mcp_server`  |  `python ToolFinder_mcp_server.py` (repo shim)
```

A config (`mcp_servers.json`, see `mcp_servers.example.json`) lists the servers to
bridge — the agent then sees the routing tools over their **combined** catalog:

```json
{
  "servers": [
    {"name": "filesystem", "command": "npx", "args": ["-y", "@modelcontextprotocol/server-filesystem", "./sandbox"]},
    {"name": "memory",     "command": "npx", "args": ["-y", "@modelcontextprotocol/server-memory"]},
    {"name": "git",        "command": "npx", "args": ["-y", "@modelcontextprotocol/server-git", "--repository", "."]}
  ]
}
```

## Register in an MCP host (Claude Desktop, Cursor, …)

`claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "toolfinder": {
      "command": "C:\\Users\\you\\AppData\\Local\\Programs\\Python\\Python312\\python.exe",
      "args": ["C:\\path\\to\\ToolFinder\\ToolFinder_mcp_server.py"],
      "env": { "TOOLFINDER_CONFIG": "C:\\path\\to\\mcp_servers.json" }
    }
  }
}
```

> Use the **absolute path to your Python interpreter** and to the script — MCP hosts (Claude Desktop, Claude Code, Cursor) don't inherit your shell `PATH`, so a bare `"python"` usually fails to launch. Find your interpreter with `python -c "import sys; print(sys.executable)"`. On macOS/Linux the same applies (e.g. `/usr/bin/python3` or your venv's `bin/python`).

The host then sees **4 small tools** instead of the downstream server's full
catalog. No host code changes — that interoperability is the point.

---

## Tools (API reference)

| Tool | Signature | Returns | Use |
|---|---|---|---|
| `find_tools` | `find_tools(query: str, k: int = 3)` | top‑k downstream tool schemas | discovery — agent then calls `call_tool` |
| `call_tool` | `call_tool(tool_name: str, arguments: dict)` | downstream result | execute a chosen tool |
| `route_and_call` | `route_and_call(intent: str, arguments: dict)` | downstream result | **one‑hop**: route by intent and execute (lowest token cost, but **argument‑blind** — no schema shown; simple‑argument tools only) |
| `catalog_size` | `catalog_size()` | `{"total_tools": N, "by_server": {…}}` | diagnostic |
| `get_stats` | `get_stats()` | recent routing decisions + per-tool counts | observability |
| `refresh` | `refresh(server: str \| None = None)` | with `server`: incremental re-index of that server only; without: full re-spawn + rebuild | after a downstream's tools change (stdio servers emitting `tools/list_changed` refresh **automatically**) |

`find_tools` and `route_and_call` route across the **union** of all configured
servers; `call_tool` and `route_and_call` dispatch execution to the server that
owns the chosen tool. (If two servers expose the same tool name, the first wins
for `call_tool`; `route_and_call` is always unambiguous because it dispatches by
the routed server.)

### Two usage patterns

**Prefer pattern 1 whenever tools have non-trivial arguments** — it shows the
agent the chosen tool's schema, so arguments are correct. Pattern 2 is cheaper
but the agent fills `arguments` blind.

1. **Agent‑controlled selection** — bind `find_tools` + `call_tool`. The agent
   discovers candidates (and sees their schemas), then chooses. Cost is constant
   in catalog size; the robust default.
2. **Delegated selection** — bind only `route_and_call`. The agent describes the
   action; the router picks and executes. Cheapest and flat in N, but
   **argument‑blind** (no schema shown) — best for large catalogs of
   simple‑argument tools.

---

## Downstream servers — MCP and OpenAPI

A config entry's `type` selects the transport:

- **`"mcp"` (default):** spawn a stdio MCP server (`command` + `args`, optional `env`).
- **`"openapi"`:** front a REST API from an OpenAPI 3.x spec — every operation becomes a routable tool.

```json
{
  "servers": [
    {"name": "git", "command": "npx", "args": ["-y", "@modelcontextprotocol/server-git", "--repository", "."]},
    {
      "name": "petstore",
      "type": "openapi",
      "spec_url": "https://petstore3.swagger.io/api/v3/openapi.json",
      "base_url": "https://petstore3.swagger.io/api/v3",
      "auth": { "type": "bearer", "token_env": "PETSTORE_TOKEN" }
    }
  ]
}
```

**Auth is resolved from environment variables — never inlined in config or logged.** Supported `auth.type`: `bearer` (`token_env`), `header` (`name` + `value_env`), `query` (`name` + `value_env`). The OpenAPI adapter is **v0.1**: OpenAPI 3.x, JSON bodies, local `$ref` resolution; no OAuth flows or multipart. It makes **outbound HTTP calls** — point it only at specs/endpoints you trust.

## How the token numbers were measured

The cost figures are **API-reported**, not estimated. `research/experiments/bridge_scaling.py` runs the agent loop and sums `usage.prompt_tokens` / `completion_tokens` per turn for each arm (bind-all-N / `find_tools`+`call_tool` / `route_and_call`) at several catalog sizes N; the deterministic per-turn schema weight is counted with `tiktoken`. `bridge_cache_measured.py` then logs the API's `cached_tokens` to re-score under prompt caching (see the *Cost caveat* above). Honest scope: **n=1, one task family, filesystem-only live execution, synthetic distractor catalogs, GPT‑5.4** — the *shapes* are robust, the statistics are not yet publication-grade (repeats/error bars are the R3 step).

The full set of evaluation scripts (cost, cache, selection accuracy, rerank, fine-tune, live gateway, and the at-scale feasibility test), what each does inside, the source code they exercise, and an honest production-grade-rigor assessment are in **[EVALUATION.md](EVALUATION.md)**.

## Configuration (environment variables)

| Variable | Default | Meaning |
|---|---|---|
| `TOOLFINDER_CONFIG` | — | path to a multi-server JSON config (see `mcp_servers.example.json`); overrides the single-server vars below |
| `TOOLFINDER_FS_ROOT` | cwd | directory the default single filesystem server may access |
| `TOOLFINDER_DOWNSTREAM_CMD` | `npx` | command for the default single server |
| `TOOLFINDER_DOWNSTREAM_ARGS` | filesystem server on `FS_ROOT` | JSON args for the default single server |
| `TOOLFINDER_MODEL` | `all-MiniLM-L6-v2` | embedding model for routing |
| `TOOLFINDER_TOPK` | `3` | default shortlist size for `find_tools` |
| `TOOLFINDER_HIERARCHICAL` | off | set (`1`/`true`) to enable two-stage **server-aware** routing |
| `TOOLFINDER_ROUTE_SERVERS` | `2` | servers kept in stage 1 when hierarchical routing is on (higher = more recall, less precision) |
| `TOOLFINDER_RERANK` | off | set (`1`/`true`) to re-rank the bi-encoder shortlist with a cross-encoder — helps confusable catalogs, adds latency |
| `TOOLFINDER_RERANK_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | CrossEncoder checkpoint used when reranking is on |
| `TOOLFINDER_INDEX` | `flat` | vector index: `flat` (exact) / `hnsw` / `auto` (switches to HNSW above ~50k tools) — for very large catalogs |
| `TOOLFINDER_CACHE_DIR` | off | persistent embedding cache dir — restarts/`refresh()` re-encode only new/changed tools (big cold-start win on large catalogs) |
| `TOOLFINDER_SCALE_THRESHOLD` | `100` | catalog size at which rerank **auto-enables** (stock encoder only — it degrades fine-tuned ones; set `TOOLFINDER_RERANK=0` to opt out) |
| `TOOLFINDER_METRICS_FILE` | off | append structured JSONL events (`route`, `refresh`) for external metrics collection; `get_stats()` also reports uptime + route-latency p50/p95 |

For more than one downstream server, use `TOOLFINDER_CONFIG` (a JSON file listing
servers) rather than the single-server env vars — see the Run section above.

`TOOLFINDER_MODEL` accepts any open sentence-transformers bi-encoder (a HuggingFace
id or a local path to fine-tuned weights). On small catalogs of distinct tools the
choice barely moves routing quality; it matters on confusable/out-of-domain
catalogs. See the README's "Choosing the embedding model" for the trade-offs.

---

## Figures

The full study (code, data, and figures) is archived under `research/experiments/`.
Reproduce the bridge figures from there: `python research/experiments/bridge_scaling.py --free-only`
(no API), then `python research/experiments/bridge_scaling.py --api-sizes 14 60 120`
(needs an OpenAI key in `research/experiments/.env` as `API_KEY` + `AGENT_MODEL`),
then `python research/experiments/bridge_figures.py`.

---

## Security notes

- **No credentials in the server.** The bridge needs no API key — routing is a
  local embedding model. Agent‑side LLM keys live only in the host / a local
  `.env`, which is git‑ignored and never published.
- **Downstream is sandboxed** by the downstream server's own allow‑list (e.g.
  the filesystem server only touches `TOOLFINDER_FS_ROOT`).
- **Schema hygiene:** tool schemas are normalized (`additionalProperties:false`,
  deduplicated `required`) before use, so a malformed downstream schema cannot
  break the agent's tool binding.
- ToolFinder's broader threat model (description poisoning, OOD rejection) is in
  [`SECURITY.md`](../SECURITY.md).

## Roadmap & release readiness (server)

**Shipped in v0.1:** multi-server union routing (`TOOLFINDER_CONFIG`); one-shot reconnect on a failed downstream call + on-demand `refresh()`; `pip install` + `toolfinder-mcp` entry point; `get_stats()` observability; opt-in server-aware hierarchical routing (`TOOLFINDER_HIERARCHICAL`); cost modeled **and** measured, plus top-k and selection-accuracy measured.

**Needed for a production release (not yet):**

- ~~push-based `tools/list_changed` subscription~~ → **shipped**: notifications trigger a debounced incremental per-server re-index (CI-tested with fakes; downstreams that never emit it still need `refresh(server=...)`);
- live multi-server validation at scale, with repeats/error bars (today: one task family, filesystem-only, n=1);
- exported/structured metrics (today: logs + `get_stats()`);
- stronger/fine-tuned default encoder (today: zero-shot MiniLM);
- description-poisoning mitigations (studied in `research/`, not implemented);
- PyPI publish.

See the repo [README "Roadmap & release readiness"](../README.md#roadmap--release-readiness) for the full checklist and next immediate steps.

## Limitations & scope

This is an **early-stage v0.1** tool — see the repo README's *Status* section for
the full v0.1-vs-production split.

- Results are for one downstream server (filesystem), one task family, and a
  schema‑padded catalog (distractor tools are real schemas but not executed).
  Live multi‑server execution at scale is not yet measured.
- The bridge improves **cost/context**, not selection **accuracy**, for strong
  models; its accuracy value appears with weak/local models or very large
  catalogs.
- Single hardware, English queries, single trial per cell (the *shape* is
  robust; add `--repeats` for error bars).

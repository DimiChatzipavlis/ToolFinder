# ToolFinder MCP Server — Cookbook

`ToolFinder_mcp_server.py` turns ToolFinder into a **Model Context Protocol (MCP)
server that bridges an LLM agent to a downstream MCP server** (filesystem, git,
memory, …). Instead of exposing the downstream server's whole tool catalog to the
agent — which grows the prompt linearly and pressures the context window — the
bridge exposes a tiny, fixed set of routing tools and selects the relevant
downstream tool(s) with the dense retriever.

It is a thin wrapper over two already-tested components: `UniversalMCPRouter`
(selection) and `DynamicMCPClient` (downstream execution).

---

## When it helps — and when it doesn't (measured, not asserted)

Measured on the filesystem server (create → edit → read a file), GPT‑5.4 as the
agent, catalog padded with real distractor tools from a 574‑tool multi‑server
pool. **Every configuration completed the task (100% success).** Numbers are
total task tokens; figures in `experiments/results/figures/`.

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

**Rule of thumb:** use the bridge when the agent faces **many** tools (dozens+)
or **multiple** MCP servers, or runs a **small/local** model that struggles to
select in‑context. For a handful of distinct tools and a strong model, bind them
directly — the bridge adds overhead for no accuracy gain (though `route_and_call`
still saves tokens even there).

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
| `route_and_call` | `route_and_call(intent: str, arguments: dict)` | downstream result | **one‑hop**: route by intent and execute (most token‑efficient) |
| `catalog_size` | `catalog_size()` | `{"total_tools": N, "by_server": {…}}` | diagnostic |
| `get_stats` | `get_stats()` | recent routing decisions + per-tool counts | observability |
| `refresh` | `refresh()` | re-spawns downstream servers and re-indexes | after a downstream's tools change |

`find_tools` and `route_and_call` route across the **union** of all configured
servers; `call_tool` and `route_and_call` dispatch execution to the server that
owns the chosen tool. (If two servers expose the same tool name, the first wins
for `call_tool`; `route_and_call` is always unambiguous because it dispatches by
the routed server.)

### Two usage patterns

1. **Agent‑controlled selection** — bind `find_tools` + `call_tool`. The agent
   discovers candidates and chooses, keeping it "in the loop." Cost is constant
   in catalog size; best when you want the agent to make the final call.
2. **Delegated selection** — bind only `route_and_call`. The agent describes the
   action; the router picks and executes. Cheapest and flat in N; best for large
   catalogs and cost‑sensitive deployments.

---

## Configuration (environment variables)

| Variable | Default | Meaning |
|---|---|---|
| `TOOLFINDER_CONFIG` | — | path to a multi-server JSON config (see `mcp_servers.example.json`); overrides the single-server vars below |
| `TOOLFINDER_FS_ROOT` | cwd | directory the default single filesystem server may access |
| `TOOLFINDER_DOWNSTREAM_CMD` | `npx` | command for the default single server |
| `TOOLFINDER_DOWNSTREAM_ARGS` | filesystem server on `FS_ROOT` | JSON args for the default single server |
| `TOOLFINDER_MODEL` | `all-MiniLM-L6-v2` | embedding model for routing |
| `TOOLFINDER_TOPK` | `3` | default shortlist size for `find_tools` |

For more than one downstream server, use `TOOLFINDER_CONFIG` (a JSON file listing
servers) rather than the single-server env vars — see the Run section above.

---

## Figures

- `experiments/results/figures/fig_bridge_scaling_free.png` — per‑turn schema
  weight (flat bridge vs linear baseline) and router recall@1 vs catalog size.
- `experiments/results/figures/fig_bridge_scaling_api.png` — end‑to‑end token
  cost vs catalog size for the three configurations (the crossover).

Reproduce: `python experiments/bridge_scaling.py --free-only` (no API), then
`python experiments/bridge_scaling.py --api-sizes 14 60 120` (needs an OpenAI
key in `experiments/.env` as `API_KEY` + `AGENT_MODEL`), then
`python experiments/bridge_figures.py`.

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
  `SECURITY.md` and `reports/report.md`.

## Roadmap (server)

Progress so far:

1. ~~**Multi-server**~~ — **done**: fronts several downstream servers via `TOOLFINDER_CONFIG`, routing across their union.
2. **Resilience** — *partial*: a failed downstream call triggers a one-shot reconnect+retry, and `refresh()` re-indexes on demand. Still TODO: push-based `tools/list_changed` subscription (auto re-index).
3. ~~**Package**~~ — **done**: `pip install` exposes the `toolfinder-mcp` console command (and `python -m toolfinder.mcp_server`).
4. **Observability** — *partial*: `get_stats()` exposes routing decisions and per-tool counts, and every route is logged. Still TODO: structured/exported metrics.
5. **Publish** — push to PyPI so users add the bridge without cloning.

See the repo README "Roadmap" for the matching research track (multi-server training, human queries, benchmark release).

## Limitations & scope

- Results are for one downstream server (filesystem), one task family, and a
  schema‑padded catalog (distractor tools are real schemas but not executed).
  Live multi‑server execution at scale is not yet measured.
- The bridge improves **cost/context**, not selection **accuracy**, for strong
  models; its accuracy value appears with weak/local models or very large
  catalogs.
- Single hardware, English queries, single trial per cell (the *shape* is
  robust; add `--repeats` for error bars).

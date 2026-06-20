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
pip install -e ".[experiments]"   # pulls fastmcp, sentence-transformers, faiss
# Node is required to launch downstream npx-based MCP servers (e.g. filesystem)
```

## Run (stdio transport)

```bash
# bridge to a filesystem server rooted at ./sandbox
TOOLFINDER_FS_ROOT=./sandbox python ToolFinder_mcp_server.py
```

## Register in an MCP host (Claude Desktop, Cursor, …)

`claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "toolfinder": {
      "command": "C:\\Users\\you\\AppData\\Local\\Programs\\Python\\Python312\\python.exe",
      "args": ["C:\\path\\to\\ToolFinder\\ToolFinder_mcp_server.py"],
      "env": { "TOOLFINDER_FS_ROOT": "C:\\path\\to\\sandbox" }
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
| `catalog_size` | `catalog_size()` | `{"downstream_tools": N}` | diagnostic |

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
| `TOOLFINDER_FS_ROOT` | cwd | directory the downstream filesystem server may access |
| `TOOLFINDER_DOWNSTREAM_CMD` | `npx` | command to launch the downstream MCP server |
| `TOOLFINDER_DOWNSTREAM_ARGS` | filesystem server on `FS_ROOT` | JSON list of args (override to bridge a different server) |
| `TOOLFINDER_MODEL` | `all-MiniLM-L6-v2` | embedding model for routing |
| `TOOLFINDER_TOPK` | `3` | default shortlist size for `find_tools` |

Bridge a different downstream server, e.g. git:

```bash
TOOLFINDER_DOWNSTREAM_ARGS='["-y","@modelcontextprotocol/server-git","--repository","."]' \
  python ToolFinder_mcp_server.py
```

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

Today the bridge fronts **one** downstream server with a static catalog. The path to production:

1. **Multi-server** — front several downstream MCP servers at once via a config file; route across their union. This is the real use case (one bridge for your whole MCP fleet) and where the token savings compound.
2. **Live tool changes & resilience** — handle `tools/list_changed` (re-index incrementally) and reconnect a downstream server that crashes or times out.
3. **Package** — a PyPI/`uvx` entry point so users add `toolfinder` to any host without cloning.
4. **Tests & observability** — in-memory FastMCP client tests in CI; log each routing decision for traceability.

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

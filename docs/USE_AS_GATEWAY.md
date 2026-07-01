# Use ToolFinder as your *only* MCP server

Goal: your LLM host (Claude Code, Claude Desktop, Cursor, …) binds **one** MCP
server — ToolFinder — and every other tool (memory, SQL, filesystem, git, and
even plain REST APIs via OpenAPI) is configured **under** ToolFinder. The agent
then sees ToolFinder's **6 routing tools** instead of every downstream tool, so
its context stays small no matter how many tools you add. (Verified live: a
filesystem + memory config gives `catalog_size` → **23 tools from 2 servers**,
while the host still binds only 6.)

There are **two config files**, and keeping them straight is the whole trick:

| File | Who reads it | What it does |
| --- | --- | --- |
| **host config** (`.mcp.json` / `claude_desktop_config.json` / `.cursor/mcp.json`) | your LLM host | registers **ToolFinder** as the host's single MCP server |
| **`mcp_servers.json`** (via `TOOLFINDER_CONFIG`, or auto-discovered at the repo root) | ToolFinder | lists every downstream tool/server **under** ToolFinder |

## Quick start (the MVP, with the bundled files)

The repo ships **both** copy-paste files at the root:

1. Copy **[`mcp_servers.example.json`](../mcp_servers.example.json)** → `mcp_servers.json`, and edit it to list your tools (memory, SQL, filesystem, REST APIs …).
2. Copy **[`mcp.host.example.json`](../mcp.host.example.json)** into your host config (`.mcp.json` for Claude Code, `claude_desktop_config.json` for Desktop, `.cursor/mcp.json` for Cursor) and replace the three `/ABSOLUTE/PATH/TO/...` placeholders — your Python interpreter, the `ToolFinder_mcp_server.py` script, and your `mcp_servers.json`.
3. **Fully restart the host** (quit and reopen — not just a refresh; see *Restart semantics* below).

That's the whole MVP: **one** host entry (`toolfinder`) + one downstream config.

> **Config auto-discovery.** If `TOOLFINDER_CONFIG` is not set, ToolFinder now
> looks for a `mcp_servers.json` **next to its install (the repo root)** before
> falling back to a single zero-config filesystem server. Setting the env var
> explicitly is still recommended — it always takes priority — but a forgotten
> env var no longer silently strands you on filesystem-only.

---

## 1. Install

```bash
python -m pip install -e .     # exposes the `toolfinder-mcp` command + the server
```

Downstream servers run with their own launchers (e.g. Node/`npx`, `uvx`, `python`).

## 2. List your tools *under* ToolFinder — `mcp_servers.json`

MCP servers use `command`/`args`/`env`; REST APIs use `type: "openapi"`. Example
with **memory + SQL + filesystem + git + a REST API**, all under one gateway:

```json
{
  "servers": [
    { "name": "memory",     "command": "npx", "args": ["-y", "@modelcontextprotocol/server-memory"] },
    { "name": "filesystem", "command": "npx", "args": ["-y", "@modelcontextprotocol/server-filesystem", "/abs/path/to/workspace"] },
    { "name": "git",        "command": "npx", "args": ["-y", "@modelcontextprotocol/server-git", "--repository", "/abs/path/to/repo"] },

    { "name": "sql", "command": "npx", "args": ["-y", "@modelcontextprotocol/server-postgres", "postgresql://localhost:5432/mydb"] },

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

> The launch commands above are **examples** — substitute the actual command for
> the MCP server you use (the MCP ecosystem moves fast), and start with the
> servers you can actually run (a `sql` entry with no Postgres running will
> simply be skipped and reported in `failed_servers`). The *structure* is what
> matters.
>
> **Secrets never go inline.** Note the `sql` entry's connection string has **no
> credentials** in it. The downstream process **inherits your shell environment**
> (merged with any per-server `"env"`), so export DB credentials like `PGUSER` /
> `PGPASSWORD` in the shell that launches ToolFinder — don't write them into the
> config. OpenAPI auth likewise uses env vars (`auth.*_env`). Your real
> `mcp_servers.json` and host configs (incl. `.mcp.json`) are **gitignored**;
> only the committed `*.example.json` templates (placeholders only) are tracked.

## 3. Register ToolFinder as the host's ONE server

Point the host at ToolFinder, passing `TOOLFINDER_CONFIG` → your `mcp_servers.json`.
**Use absolute paths** to your interpreter and the script — hosts don't inherit
your shell `PATH`. Find your interpreter with `python -c "import sys; print(sys.executable)"`.

**Claude Code** — `.mcp.json` (project root) or your user MCP settings:

```json
{
  "mcpServers": {
    "toolfinder": {
      "command": "/abs/path/to/python",
      "args": ["/abs/path/to/ToolFinder/ToolFinder_mcp_server.py"],
      "env": { "TOOLFINDER_CONFIG": "/abs/path/to/mcp_servers.json" }
    }
  }
}
```

**Claude Desktop** — `claude_desktop_config.json`, and **Cursor** — `.cursor/mcp.json`
use the **same shape**. In every case there is exactly **one** entry: `toolfinder`
— **remove any other/older toolfinder registrations** (a duplicate started without
the env var is the #1 cause of "only filesystem shows up").

That's it. The host now exposes only ToolFinder; the agent reaches everything
through `find_tools` → `call_tool`, or the one-hop `route_and_call`.

## 4. Verify it worked

In a fresh conversation, ask the agent to call **`catalog_size`**. Expected shape
(for the filesystem + memory example):

```
total_tools: 23   by_server: { filesystem: 14, memory: 9 }   failed_servers: {}
```

- **All configured servers listed** → you're done. The host's own tool list should
  show only ToolFinder's 6 tools (`find_tools`, `call_tool`, `route_and_call`,
  `catalog_size`, `get_stats`, `refresh`) — that gap *is* the value.
- **A server missing?** Check `failed_servers` — it holds the startup error for
  every downstream that couldn't launch (bad command, service not running, no
  network). The gateway stays up on the healthy ones.
- **First call is slow (~10–30 s):** on first use ToolFinder downloads the
  embedding model (~90 MB) and builds the index; afterwards it's fast. Downstream
  servers are spawned **concurrently**, so startup scales with the slowest
  server, not the sum.

## 5. Restart semantics (read this — it's the #1 gotcha)

- The `refresh()` tool **re-reads `mcp_servers.json`** and re-spawns/re-indexes —
  use it after *editing the downstream list*.
- But the **host config** (`.mcp.json` env vars, interpreter paths) is fixed when
  the host launches the ToolFinder *process*. After changing the host entry —
  or if refreshes keep showing stale results — **fully quit and reopen the host**
  so it kills the old process and starts one with the new environment. An
  in-place "refresh"/reload of the host window is often not enough.

## 6. Options (environment variables, set in the host entry's `env`)

| Variable | Effect |
| --- | --- |
| `TOOLFINDER_CONFIG` | path to `mcp_servers.json` (optional if the file sits at the repo root — auto-discovered) |
| `TOOLFINDER_RERANK=1` | cross-encoder re-rank — better selection on confusable catalogs (adds latency; measured recall@1 0.56→0.85) |
| `TOOLFINDER_RERANK_MODEL` | CrossEncoder checkpoint used when reranking is on |
| `TOOLFINDER_MODEL` | swap the embedding model (any sentence-transformers bi-encoder, or a fine-tuned path) |
| `TOOLFINDER_HIERARCHICAL=1` / `TOOLFINDER_ROUTE_SERVERS` | two-stage server-aware routing for very large multi-server setups |
| `TOOLFINDER_TOPK` | shortlist size for `find_tools` |
| `TOOLFINDER_INDEX` | vector index for very large catalogs: `flat` (default, exact) / `hnsw` / `auto` |

## 7. Resilience & diagnostics

- **One bad server won't take down the gateway.** A downstream that fails to
  start is logged and skipped; the rest come up. Check `catalog_size()` /
  `get_stats()` → `failed_servers` to see what didn't load and why (covered by a
  CI test: `tests/test_mcp_server.py::test_failed_downstream_is_isolated`).
- **Concurrent startup:** all downstream servers spawn and handshake in parallel;
  ingest order stays deterministic (first-wins on tool-name collisions).
- After a downstream's tool list changes, call `refresh()` to re-index.

## Troubleshooting

| Symptom | Cause → fix |
| --- | --- |
| `catalog_size` shows **only filesystem** | ToolFinder started without a config (env var missing **and** no `mcp_servers.json` at the repo root) → set `TOOLFINDER_CONFIG` in the host entry, or place the file at the repo root, then **fully restart the host** |
| Edits to `mcp_servers.json` don't appear | call `refresh()`; if still stale, fully quit + reopen the host (old process) |
| A server is missing from `by_server` | it failed to start — read its error in `failed_servers` (service down, bad command, no network) |
| Two toolfinder entries in `/mcp` | remove the older registration; duplicates launched without the env var mask the configured one |
| First call very slow | one-time embedding-model download (~90 MB) + index build |

## Honest caveats (v0.1)

- Validated **live**: filesystem + memory under one gateway (23 tools → host binds
  6), and a real filesystem MCP server **+** an OpenAPI API simultaneously
  (`legacy/experiments/gateway_heterogeneous_demo.py`). The SQL/git entries above
  are configuration **examples** — confirm them with your actual servers.
- Routing quality depends on the embedding model; for confusable/domain catalogs
  enable `TOOLFINDER_RERANK` or point `TOOLFINDER_MODEL` at a fine-tuned encoder.
- OpenAPI execution makes **outbound HTTP** — only point it at specs/endpoints you
  trust (see [SECURITY.md](../SECURITY.md)).
- Not yet production-grade (no live `tools/list_changed`, single process). See the
  README *Status* / *Roadmap* and [SCALABILITY.md](SCALABILITY.md).

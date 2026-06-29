# Use ToolFinder as your *only* MCP server

Goal: your LLM host (Claude Code, Claude Desktop, Cursor, …) binds **one** MCP
server — ToolFinder — and every other tool (memory, SQL, filesystem, git, and
even plain REST APIs via OpenAPI) is configured **under** ToolFinder. The agent
then sees a few routing tools instead of every downstream tool, so its context
stays small no matter how many tools you add.

There are **two config files**, and keeping them straight is the whole trick:

| File | Who reads it | What it does |
| --- | --- | --- |
| **host config** (`.mcp.json` / `claude_desktop_config.json` / `.cursor/mcp.json`) | your LLM host | registers **ToolFinder** as the host's single MCP server |
| **`mcp_servers.json`** (pointed to by `TOOLFINDER_CONFIG`) | ToolFinder | lists every downstream tool/server **under** ToolFinder |

## Quick start (the MVP, with the bundled files)

The repo ships **both** copy-paste files at the root:

1. Copy **[`mcp_servers.example.json`](../mcp_servers.example.json)** → `mcp_servers.json`, and edit it to list your tools (memory, SQL, filesystem, REST APIs …).
2. Copy **[`mcp.host.example.json`](../mcp.host.example.json)** into your host config (`.mcp.json` for Claude Code, `claude_desktop_config.json` for Desktop, `.cursor/mcp.json` for Cursor) and replace the three `/ABSOLUTE/PATH/TO/...` placeholders — your Python interpreter, the `ToolFinder_mcp_server.py` script, and your `mcp_servers.json`.

That's the whole MVP: **one** host entry (`toolfinder`) + one downstream config. The steps below explain each piece and the options.

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
> the MCP server you use (the MCP ecosystem moves fast). The *structure* is what
> matters.
>
> **Secrets never go inline.** Note the `sql` entry's connection string has **no
> credentials** in it. The downstream process **inherits your shell environment**
> (merged with any per-server `"env"`), so export DB credentials like `PGUSER` /
> `PGPASSWORD` in the shell that launches ToolFinder — don't write them into the
> config. OpenAPI auth likewise uses env vars (`auth.*_env`). Your real
> `mcp_servers.json` and host config are **gitignored**; only the committed
> `*.example.json` templates (placeholders only) are tracked.

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
use the **same shape**. In every case there is exactly **one** entry: `toolfinder`.

That's it. The host now exposes only ToolFinder; the agent reaches everything
through `find_tools` → `call_tool`, or the one-hop `route_and_call`.

## 4. Options (environment variables, set in the host entry's `env`)

| Variable | Effect |
| --- | --- |
| `TOOLFINDER_RERANK=1` | cross-encoder re-rank — better selection on confusable catalogs (adds latency) |
| `TOOLFINDER_MODEL` | swap the embedding model (any sentence-transformers bi-encoder, or a fine-tuned path) |
| `TOOLFINDER_HIERARCHICAL=1` / `TOOLFINDER_ROUTE_SERVERS` | two-stage server-aware routing for very large multi-server setups |
| `TOOLFINDER_TOPK` | shortlist size for `find_tools` |
| `TOOLFINDER_INDEX` | vector index for very large catalogs: `flat` (default, exact) / `hnsw` / `auto` |

## 5. Resilience & diagnostics

- **One bad server won't take down the gateway.** A downstream that fails to
  start is logged and skipped; the rest come up. Check `catalog_size()` /
  `get_stats()` → `failed_servers` to see what didn't load and why.
- After a downstream's tool list changes, call `refresh()` to re-index.

## Honest caveats (v0.1)

- Validated **live** end-to-end with a real filesystem MCP server **+** an OpenAPI
  API under one gateway (see `legacy/experiments/gateway_heterogeneous_demo.py`).
  The memory/SQL entries above are configuration **examples** — confirm them with
  your actual servers.
- Routing quality depends on the embedding model; for confusable/domain catalogs
  enable `TOOLFINDER_RERANK` or point `TOOLFINDER_MODEL` at a fine-tuned encoder.
- OpenAPI execution makes **outbound HTTP** — only point it at specs/endpoints you
  trust (see [SECURITY.md](../SECURITY.md)).
- Not yet production-grade (no live `tools/list_changed`, single process). See the
  README *Status* / *Roadmap*.
```

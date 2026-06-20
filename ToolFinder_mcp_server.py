"""ToolFinder as an MCP server — a routing bridge in front of one or more
downstream MCP servers (filesystem, git, memory, …).

Instead of exposing every downstream server's full catalog to the agent (which
bloats the context window and grows with every tool), ToolFinder embeds the
*union* of all downstream tools once at startup and exposes a small, fixed set
of routing tools:

  find_tools(query)            -> the top-k relevant tools across all servers
  call_tool(tool_name, args)   -> execute a tool (dispatched to its server)
  route_and_call(intent, args) -> route + execute in one hop (most efficient)
  catalog_size()               -> tool counts per downstream server

Selection is dense retrieval; execution is dispatched to the correct downstream
`DynamicMCPClient`. It is a thin wrapper over `UniversalMCPRouter` (selection)
and `DynamicMCPClient` (downstream execution), both already tested.

Configure downstream servers one of two ways:

  1. Multi-server (recommended) — a JSON config listing servers:
        TOOLFINDER_CONFIG=mcp_servers.json python ToolFinder_mcp_server.py
     where mcp_servers.json looks like mcp_servers.example.json.

  2. Single filesystem server (zero-config) — via env vars:
        TOOLFINDER_FS_ROOT=./sandbox python ToolFinder_mcp_server.py

Register in an MCP host (Claude Desktop / Claude Code / Cursor) with ABSOLUTE
paths to your interpreter and this script — hosts don't inherit your shell PATH:

    {
      "mcpServers": {
        "toolfinder": {
          "command": "C:\\\\Users\\\\you\\\\...\\\\python.exe",
          "args": ["C:\\\\path\\\\to\\\\ToolFinder_mcp_server.py"],
          "env": {"TOOLFINDER_CONFIG": "C:\\\\path\\\\to\\\\mcp_servers.json"}
        }
      }
    }

Environment variables:
    TOOLFINDER_CONFIG          path to a multi-server JSON config (overrides the single-server vars)
    TOOLFINDER_FS_ROOT         directory the default filesystem server may access (default: cwd)
    TOOLFINDER_DOWNSTREAM_CMD  command for the default single server (default: npx)
    TOOLFINDER_DOWNSTREAM_ARGS JSON list of args for the default single server
    TOOLFINDER_MODEL           embedding model (default: all-MiniLM-L6-v2)
    TOOLFINDER_TOPK            default shortlist size for find_tools (default: 3)
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Any

from fastmcp import FastMCP

from toolfinder import UniversalMCPRouter, to_openai_tools
from toolfinder.mcp_adapter import DynamicMCPClient

logger = logging.getLogger("toolfinder.mcp_server")

mcp = FastMCP("ToolFinder")

_state: dict[str, Any] = {"clients": {}, "router": None, "tool_to_server": {}, "counts": {}}
_init_lock = asyncio.Lock()


def _server_configs() -> list[dict]:
    """List of downstream server configs: [{name, command, args, env}]."""
    config_path = os.getenv("TOOLFINDER_CONFIG")
    if config_path and Path(config_path).exists():
        data = json.loads(Path(config_path).read_text(encoding="utf-8"))
        servers = data.get("servers", data if isinstance(data, list) else [])
        if not servers:
            raise ValueError(f"{config_path} contains no servers")
        return servers
    # Zero-config fallback: a single filesystem server from env vars.
    command = os.getenv("TOOLFINDER_DOWNSTREAM_CMD", "npx")
    raw_args = os.getenv("TOOLFINDER_DOWNSTREAM_ARGS")
    if raw_args:
        args = list(json.loads(raw_args))
    else:
        fs_root = os.getenv("TOOLFINDER_FS_ROOT", os.getcwd())
        args = ["-y", "@modelcontextprotocol/server-filesystem", fs_root]
    return [{"name": "filesystem", "command": command, "args": args}]


async def _ensure_ready() -> None:
    """Spawn every downstream server and build the union router (once)."""
    if _state["router"] is not None:
        return
    async with _init_lock:
        if _state["router"] is not None:
            return
        router = UniversalMCPRouter(
            model_name=os.getenv("TOOLFINDER_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
        )
        clients: dict[str, DynamicMCPClient] = {}
        tool_to_server: dict[str, str] = {}
        counts: dict[str, int] = {}
        for cfg in _server_configs():
            name = str(cfg["name"])
            client = DynamicMCPClient(
                server_name=name, command=str(cfg["command"]), args=[str(a) for a in cfg.get("args", [])],
                env=cfg.get("env"), startup_timeout_s=90.0, request_timeout_s=45.0,
            )
            tools = await client.initialize_and_get_tools()
            router.ingest_server(name, tools)
            clients[name] = client
            counts[name] = len(tools)
            for tool in tools:
                # First server wins on a name collision; route_and_call stays
                # unambiguous because it dispatches by the routed server_name.
                tool_to_server.setdefault(tool["tool_name"], name)
        _state.update(clients=clients, router=router, tool_to_server=tool_to_server, counts=counts)
        logger.info("ToolFinder bridge ready: %s servers, %s tools", len(clients), sum(counts.values()))


async def _dispatch(server: str, tool_name: str, arguments: dict[str, Any]) -> Any:
    client = _state["clients"].get(server)
    if client is None:
        return {"error": f"no downstream server '{server}'"}
    return await client.call_tool(tool_name, arguments or {})


@mcp.tool
async def find_tools(query: str, k: int | None = None) -> list[dict]:
    """Return the downstream tools most relevant to `query`, as bindable schemas
    (each tagged with its `server_name`). Call this first, then `call_tool`."""
    await _ensure_ready()
    top_k = k or int(os.getenv("TOOLFINDER_TOPK", "3"))
    return to_openai_tools(_state["router"].route_top_k(query, k=top_k))


@mcp.tool
async def call_tool(tool_name: str, arguments: dict[str, Any]) -> Any:
    """Execute one downstream tool by name, dispatched to the server that owns it."""
    await _ensure_ready()
    server = _state["tool_to_server"].get(tool_name)
    if server is None:
        return {"error": f"unknown tool '{tool_name}'. Use find_tools to discover available tools."}
    return await _dispatch(server, tool_name, arguments)


@mcp.tool
async def route_and_call(intent: str, arguments: dict[str, Any]) -> Any:
    """Single-step bridge: route `intent` to the best downstream tool across all
    servers and execute it with `arguments` in one hop (no discovery round-trip).

    The most token-efficient pattern — the agent binds just this one tool no
    matter how large the combined catalog is. Use find_tools + call_tool instead
    when the agent should see and choose among candidates itself.
    """
    await _ensure_ready()
    matches = _state["router"].route_top_k(intent, k=1)
    if not matches:
        return {"error": f"no downstream tool matched intent: {intent!r}"}
    chosen = matches[0]
    return await _dispatch(chosen.server_name, chosen.tool_name, arguments or {})


@mcp.tool
async def catalog_size() -> dict[str, Any]:
    """Report the downstream catalog behind the bridge (diagnostic)."""
    await _ensure_ready()
    return {"total_tools": sum(_state["counts"].values()), "by_server": dict(_state["counts"])}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    mcp.run()

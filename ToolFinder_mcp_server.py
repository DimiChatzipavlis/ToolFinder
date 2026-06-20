"""ToolFinder as an MCP server — a routing bridge between an LLM agent and a
downstream MCP server (e.g. filesystem).

Instead of exposing the downstream server's full tool catalog to the agent
(which bloats the context window), ToolFinder exposes just two tools:

  find_tools(query)            -> the top-k downstream tools relevant to a request
  call_tool(tool_name, args)   -> execute one downstream tool and return its result

The agent calls find_tools to discover the right tool from a narrowed shortlist,
then call_tool to run it. The downstream catalog is embedded once at startup with
the dense router; selection is retrieval, not prompt-stuffing.

This is a thin wrapper over components that already exist and are tested:
`UniversalMCPRouter` (selection) and `DynamicMCPClient` (downstream execution).

Run (stdio transport, the MCP default):
    # point it at a filesystem server rooted at a sandbox dir
    TOOLFINDER_FS_ROOT=./sandbox python ToolFinder_mcp_server.py

Register in an MCP host (e.g. Claude Desktop `claude_desktop_config.json`):
    {
      "mcpServers": {
        "toolfinder": {
          "command": "python",
          "args": ["ToolFinder_mcp_server.py"],
          "env": {"TOOLFINDER_FS_ROOT": "C:/path/to/sandbox"}
        }
      }
    }

Configuration (environment variables):
    TOOLFINDER_FS_ROOT        directory the downstream filesystem server may access (default: cwd)
    TOOLFINDER_DOWNSTREAM_CMD command to launch the downstream MCP server (default: npx)
    TOOLFINDER_DOWNSTREAM_ARGS  JSON list of args (default: filesystem server on FS_ROOT)
    TOOLFINDER_MODEL          embedding model (default: all-MiniLM-L6-v2)
    TOOLFINDER_TOPK           default shortlist size (default: 3)
"""

from __future__ import annotations

import asyncio
import json
import os
from typing import Any

from fastmcp import FastMCP

from toolfinder import UniversalMCPRouter, to_openai_tools
from toolfinder.mcp_adapter import DynamicMCPClient

mcp = FastMCP("ToolFinder")

_state: dict[str, Any] = {"client": None, "router": None, "tool_count": 0}
_init_lock = asyncio.Lock()


def _downstream_config() -> tuple[str, list[str]]:
    command = os.getenv("TOOLFINDER_DOWNSTREAM_CMD", "npx")
    raw_args = os.getenv("TOOLFINDER_DOWNSTREAM_ARGS")
    if raw_args:
        return command, list(json.loads(raw_args))
    fs_root = os.getenv("TOOLFINDER_FS_ROOT", os.getcwd())
    return command, ["-y", "@modelcontextprotocol/server-filesystem", fs_root]


async def _ensure_ready() -> None:
    """Lazily spawn the downstream server and build the router (once)."""
    if _state["router"] is not None:
        return
    async with _init_lock:
        if _state["router"] is not None:
            return
        command, args = _downstream_config()
        client = DynamicMCPClient(
            server_name="downstream", command=command, args=args,
            startup_timeout_s=90.0, request_timeout_s=45.0,
        )
        tools = await client.initialize_and_get_tools()
        router = UniversalMCPRouter(
            model_name=os.getenv("TOOLFINDER_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
        )
        router.ingest_server("downstream", tools)
        _state.update(client=client, router=router, tool_count=len(tools))


@mcp.tool
async def find_tools(query: str, k: int | None = None) -> list[dict]:
    """Return the downstream tools most relevant to `query`, as bindable schemas.

    Call this first with a short description of what you want to do; then call
    `call_tool` with one of the returned tool names.
    """
    await _ensure_ready()
    top_k = k or int(os.getenv("TOOLFINDER_TOPK", "3"))
    results = _state["router"].route_top_k(query, k=top_k)
    return to_openai_tools(results)


@mcp.tool
async def call_tool(tool_name: str, arguments: dict[str, Any]) -> Any:
    """Execute one downstream tool by name and return its result."""
    await _ensure_ready()
    return await _state["client"].call_tool(tool_name, arguments or {})


@mcp.tool
async def route_and_call(intent: str, arguments: dict[str, Any]) -> Any:
    """Single-step bridge: route `intent` to the best downstream tool and execute
    it with `arguments` in one hop (no separate discovery round-trip).

    This is the most token-efficient pattern — the agent binds just this one
    tool regardless of how large the downstream catalog is. Use it when the
    agent should delegate selection entirely to the router; use find_tools +
    call_tool when the agent should see and choose among candidates itself.
    """
    await _ensure_ready()
    matches = _state["router"].route_top_k(intent, k=1)
    if not matches:
        return {"error": f"no downstream tool matched intent: {intent!r}"}
    return await _state["client"].call_tool(matches[0].tool_name, arguments or {})


@mcp.tool
async def catalog_size() -> dict[str, int]:
    """Report how many downstream tools are behind the bridge (diagnostic)."""
    await _ensure_ready()
    return {"downstream_tools": _state["tool_count"]}


if __name__ == "__main__":
    mcp.run()

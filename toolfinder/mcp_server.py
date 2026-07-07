"""ToolFinder as an MCP server — a routing bridge in front of one or more
downstream MCP servers (filesystem, git, memory, …).

Instead of exposing every downstream server's full catalog to the agent (which
bloats the context window and grows with every tool), ToolFinder embeds the
*union* of all downstream tools once and exposes a small, fixed set of routing
tools, dispatching execution to whichever server owns the chosen tool:

  find_tools(query)            -> the top-k relevant tools across all servers
  call_tool(tool_name, args)   -> execute a tool (dispatched to its server)
  route_and_call(intent, args) -> route + execute in one hop (most efficient)
  catalog_size()               -> tool counts per downstream server
  get_stats()                  -> recent routing decisions (observability)
  refresh()                    -> re-spawn downstream servers and re-index

Resilience: a failed downstream call triggers a one-shot reconnect+retry of that
server, so a downstream crash doesn't permanently break the bridge.

Run it (after `pip install -e .`):
    toolfinder-mcp                       # console entry point
    python -m toolfinder.mcp_server      # module form
    python ToolFinder_mcp_server.py      # repo shim (back-compat)

Configure downstream servers either with a multi-server JSON config
(`TOOLFINDER_CONFIG`, see mcp_servers.example.json) or, zero-config, a single
filesystem server (`TOOLFINDER_FS_ROOT`). See docs/MCP_SERVER.md.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from collections import Counter, deque
from pathlib import Path
from typing import Any

from fastmcp import FastMCP

from toolfinder import UniversalMCPRouter, to_openai_tools
from toolfinder.dynamic_faiss_router import RouterHyperparameters
from toolfinder.mcp_adapter import DynamicMCPClient

logger = logging.getLogger("toolfinder.mcp_server")

mcp = FastMCP("ToolFinder")

_state: dict[str, Any] = {
    "clients": {},          # server_name -> downstream client
    "specs": {},            # server_name -> {command, args, env} | openapi spec
    "router": None,
    "tool_to_server": {},   # tool_name -> server_name
    "counts": {},           # server_name -> tool count
    "failed": {},           # server_name -> startup error (skipped, gateway stays up)
    "routes": 0,            # total routing decisions
    "by_tool": Counter(),   # chosen tool -> count
    "recent": deque(maxlen=50),
}
_init_lock = asyncio.Lock()


def _server_configs() -> list[dict]:
    """Downstream server configs: [{name, command, args, env}].

    Resolved in order: (1) the `TOOLFINDER_CONFIG` env var, then (2) a
    `mcp_servers.json` sitting next to the install (repo root) — so the gateway
    finds its multi-server config even when the host registration forgets to pass
    the env var. Only if neither exists does it fall back to a single filesystem
    server (`TOOLFINDER_FS_ROOT`).
    """
    candidates: list[Path] = []
    env_path = os.getenv("TOOLFINDER_CONFIG")
    if env_path:
        candidates.append(Path(env_path))
    candidates.append(Path(__file__).resolve().parent.parent / "mcp_servers.json")
    for path in candidates:
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            servers = data.get("servers", data if isinstance(data, list) else [])
            if not servers:
                raise ValueError(f"{path} contains no servers")
            logger.info("ToolFinder downstream config: %s (%d servers)", path, len(servers))
            return servers
    command = os.getenv("TOOLFINDER_DOWNSTREAM_CMD", "npx")
    raw_args = os.getenv("TOOLFINDER_DOWNSTREAM_ARGS")
    if raw_args:
        args = list(json.loads(raw_args))
    else:
        fs_root = os.getenv("TOOLFINDER_FS_ROOT", os.getcwd())
        args = ["-y", "@modelcontextprotocol/server-filesystem", fs_root]
    return [{"name": "filesystem", "command": command, "args": args}]


def _new_client(name: str, spec: dict):
    """Build the right downstream client for a config entry.

    `type` selects the transport: "mcp" (default) spawns a stdio MCP server;
    "openapi" fronts a REST API described by an OpenAPI spec. Both expose the
    same `initialize_and_get_tools()` / `call_tool()` / `close()` interface, so
    the rest of the bridge is transport-agnostic.
    """
    if str(spec.get("type", "mcp")).lower() == "openapi":
        from toolfinder.openapi_adapter import OpenAPIClient

        return OpenAPIClient(
            server_name=name,
            spec=spec.get("spec") or spec.get("spec_url") or spec.get("spec_file"),
            base_url=spec.get("base_url"),
            auth=spec.get("auth"),
            request_timeout_s=45.0,
        )
    return DynamicMCPClient(
        server_name=name, command=str(spec["command"]),
        args=[str(a) for a in spec.get("args", [])], env=spec.get("env"),
        startup_timeout_s=90.0, request_timeout_s=45.0,
    )


async def _spawn_one(cfg: dict) -> tuple[str, dict, Any, list | None, str | None]:
    """Construct + initialize one downstream client. Returns (name, cfg, client,
    tools, error) — errors are captured, not raised, for per-server isolation."""
    name = str(cfg["name"])
    try:
        client = _new_client(name, cfg)
        tools = await client.initialize_and_get_tools()
        return name, cfg, client, tools, None
    except Exception as exc:  # noqa: BLE001 - isolate a bad downstream, keep the gateway up
        return name, cfg, None, None, str(exc)[:300]


async def _build() -> None:
    """Spawn every downstream server (concurrently) and build the union router."""
    rerank = os.getenv("TOOLFINDER_RERANK", "").strip().lower() in {"1", "true", "yes", "on"}
    index_type = os.getenv("TOOLFINDER_INDEX", "flat").strip().lower()
    if index_type not in {"flat", "hnsw", "auto"}:
        index_type = "flat"
    stock_model = "sentence-transformers/all-MiniLM-L6-v2"
    model_name = os.getenv("TOOLFINDER_MODEL") or stock_model
    router = UniversalMCPRouter(
        model_name=model_name,
        config=RouterHyperparameters(
            index_type=index_type,
            rerank=rerank,
            rerank_model=os.getenv("TOOLFINDER_RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"),
            cache_dir=os.getenv("TOOLFINDER_CACHE_DIR") or None,
        ),
    )
    clients: dict[str, Any] = {}
    specs: dict[str, dict] = {}
    tool_to_server: dict[str, str] = {}
    counts: dict[str, int] = {}
    failed: dict[str, str] = {}

    # Spawn + handshake all servers concurrently (the slow, IO-bound part); then
    # ingest sequentially in config order (index build is not concurrency-safe,
    # and order keeps tool-name collisions deterministic / first-wins).
    spawned = await asyncio.gather(*[_spawn_one(cfg) for cfg in _server_configs()])
    for name, cfg, client, tools, err in spawned:
        specs[name] = cfg
        if err is not None:
            failed[name] = err
            logger.error("downstream server %r failed to start (skipped): %s", name, err)
            continue
        router.ingest_server(name, tools)
        clients[name] = client
        counts[name] = len(tools)
        for tool in tools:
            tool_to_server.setdefault(tool["tool_name"], name)
    # P0 auto-scale (data-backed, eval_encoder_at_scale.py): rerank rescues the
    # STOCK encoder on large confusable catalogs (R@1 0.40->0.50 at 574) but
    # DEGRADES fine-tuned encoders (0.583->0.500) — so it auto-enables only when
    # the user made no explicit choice, the encoder is the stock default, and
    # the union catalog is large.
    total_tools = sum(counts.values())
    threshold = int(os.getenv("TOOLFINDER_SCALE_THRESHOLD", "100"))
    if (os.getenv("TOOLFINDER_RERANK") is None and model_name == stock_model
            and total_tools >= threshold):
        router.enable_rerank()
        logger.info(
            "auto-enabled cross-encoder rerank (%d tools >= threshold %d, stock encoder); "
            "set TOOLFINDER_RERANK=0 to opt out", total_tools, threshold,
        )
    _state.update(clients=clients, specs=specs, router=router, tool_to_server=tool_to_server,
                  counts=counts, failed=failed)
    # E2: subscribe to push-based tool changes. Stdio MCP clients surface
    # server-initiated notifications via `on_notification`; a tools/list_changed
    # triggers a debounced incremental refresh of just that server. (OpenAPI
    # downstreams have no push channel — refresh them explicitly.)
    for name, client in clients.items():
        if hasattr(client, "on_notification"):
            client.on_notification = _make_notification_handler(name)
    logger.info(
        "ToolFinder bridge ready: %s/%s servers, %s tools%s",
        len(clients), len(clients) + len(failed), sum(counts.values()),
        f" (failed: {', '.join(failed)})" if failed else "",
    )


async def _ensure_ready() -> None:
    if _state["router"] is not None:
        return
    async with _init_lock:
        if _state["router"] is None:
            await _build()


def _rebuild_tool_map() -> None:
    """Recompute tool_name -> server from the router's live metadata
    (insertion order preserved, so name collisions stay first-wins)."""
    router = _state["router"]
    mapping: dict[str, str] = {}
    for index_id in sorted(router.metadata):
        owner, tool_name, _ = router.metadata[index_id]
        mapping.setdefault(tool_name, owner)
    _state["tool_to_server"] = mapping


async def _refresh_server(server: str) -> dict[str, Any]:
    """Incrementally refresh ONE downstream server: re-list its tools and
    re-index just that server (others untouched; with the embedding cache only
    new/changed tools are re-encoded)."""
    client = _state["clients"].get(server)
    if client is None:
        return {"error": f"unknown server '{server}'. Configured: {sorted(_state['clients'])}"}
    try:
        if hasattr(client, "refresh_tools"):
            tools = await client.refresh_tools()
        else:  # e.g. OpenAPI downstream — re-fetch the spec
            tools = await client.initialize_and_get_tools()
    except Exception as exc:  # noqa: BLE001 - downstream may have died; try a respawn once
        logger.warning("refresh of %r failed (%s); attempting reconnect", server, exc)
        retry = await _reconnect(server)
        if retry is None:
            return {"error": f"could not refresh '{server}': {exc}"}
        tools = await retry.initialize_and_get_tools()
    _state["router"].reingest_server(server, tools)
    _state["counts"][server] = len(tools)
    _rebuild_tool_map()
    logger.info("incrementally refreshed %r: %d tools", server, len(tools))
    return {"refreshed": server, "tools": len(tools), "total_tools": sum(_state["counts"].values())}


def _make_notification_handler(server: str):
    """Debounced handler for downstream `tools/list_changed` notifications:
    schedules one incremental refresh per server at a time."""

    def handler(method: str, params: dict[str, Any]) -> None:
        del params
        if method != "notifications/tools/list_changed":
            return
        tasks: dict[str, asyncio.Task] = _state.setdefault("refresh_tasks", {})
        existing = tasks.get(server)
        if existing is not None and not existing.done():
            return  # a refresh for this server is already in flight
        logger.info("tools/list_changed from %r — scheduling incremental refresh", server)
        tasks[server] = asyncio.create_task(_refresh_server(server))

    return handler


async def _reconnect(server: str) -> DynamicMCPClient | None:
    """Re-spawn one downstream server after a failed call (E2 resilience)."""
    spec = _state["specs"].get(server)
    if spec is None:
        return None
    old = _state["clients"].get(server)
    if old is not None:
        try:
            await old.close()
        except Exception:  # noqa: BLE001
            pass
    try:
        client = _new_client(server, spec)
        await client.initialize_and_get_tools()
        _state["clients"][server] = client
        logger.warning("reconnected downstream server %r", server)
        return client
    except Exception as exc:  # noqa: BLE001
        logger.error("failed to reconnect %r: %s", server, exc)
        return None


async def _dispatch(server: str, tool_name: str, arguments: dict[str, Any]) -> Any:
    client = _state["clients"].get(server)
    if client is None:
        return {"error": f"no downstream server '{server}'"}
    try:
        return await client.call_tool(tool_name, arguments or {})
    except Exception as exc:  # noqa: BLE001 - downstream may have died; try once more
        retry = await _reconnect(server)
        if retry is None:
            return {"error": f"downstream '{server}' unavailable: {exc}"}
        try:
            return await retry.call_tool(tool_name, arguments or {})
        except Exception as exc2:  # noqa: BLE001
            return {"error": f"downstream '{server}' failed after reconnect: {exc2}"}


def _record(query: str, chosen: str | None, server: str | None, score: float | None) -> None:
    _state["routes"] += 1
    if chosen:
        _state["by_tool"][chosen] += 1
    _state["recent"].append({"query": query[:120], "tool": chosen, "server": server, "score": score})
    logger.info("route %r -> %s@%s (score=%s)", query[:80], chosen, server, score)


def _route(query: str, k: int) -> list:
    """Route a query to the top-k tools.

    Flat search by default. When `TOOLFINDER_HIERARCHICAL` is set, use two-stage
    server-aware routing — pick the top `TOOLFINDER_ROUTE_SERVERS` servers (by
    centroid), then the best tools within them. Helps precision on confusable
    multi-server catalogs and bounds the search at large scale; not a latency win.
    """
    router = _state["router"]
    if os.getenv("TOOLFINDER_HIERARCHICAL", "").strip().lower() in {"1", "true", "yes", "on"}:
        n_servers = max(1, int(os.getenv("TOOLFINDER_ROUTE_SERVERS", "2")))
        return router.route_top_k_hierarchical(query, k=k, n_servers=n_servers)
    return router.route_top_k(query, k=k)


@mcp.tool
async def find_tools(query: str, k: int | None = None) -> list[dict]:
    """Return the downstream tools most relevant to `query`, as bindable schemas
    (each tagged with its `server_name`). **Recommended pattern:** call this
    first, then `call_tool` — the agent sees the chosen tools' *schemas* (top-k,
    not the whole catalog), so it fills arguments correctly while the prompt
    stays small."""
    await _ensure_ready()
    top_k = k or int(os.getenv("TOOLFINDER_TOPK", "3"))
    matches = _route(query, top_k)
    if matches:
        _record(query, matches[0].tool_name, matches[0].server_name, round(matches[0].score, 4))
    return to_openai_tools(matches)


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
    servers and execute it in one hop (no discovery round-trip). Lowest token
    cost — the agent binds just this one tool regardless of catalog size.

    Caveat: the agent never sees the chosen tool's schema, so it must supply
    `arguments` **blind**. Reliable only for tools with simple/obvious arguments;
    for schema-heavy tools prefer `find_tools` + `call_tool`, which surfaces the
    schema before execution."""
    await _ensure_ready()
    matches = _route(intent, 1)
    if not matches:
        _record(intent, None, None, None)
        return {"error": f"no downstream tool matched intent: {intent!r}"}
    chosen = matches[0]
    _record(intent, chosen.tool_name, chosen.server_name, round(chosen.score, 4))
    return await _dispatch(chosen.server_name, chosen.tool_name, arguments or {})


@mcp.tool
async def catalog_size() -> dict[str, Any]:
    """Report the downstream catalog behind the bridge (diagnostic)."""
    await _ensure_ready()
    return {
        "total_tools": sum(_state["counts"].values()),
        "by_server": dict(_state["counts"]),
        "failed_servers": dict(_state.get("failed", {})),
    }


@mcp.tool
async def get_stats() -> dict[str, Any]:
    """Routing observability: total routes, most-selected tools, recent decisions."""
    await _ensure_ready()
    return {
        "total_routes": _state["routes"],
        "top_tools": dict(_state["by_tool"].most_common(10)),
        "recent": list(_state["recent"])[-10:],
        "failed_servers": dict(_state.get("failed", {})),
    }


@mcp.tool
async def refresh(server: str | None = None) -> dict[str, Any]:
    """Re-index after downstream tool changes. With `server`, refresh just that
    one incrementally (fast — other servers stay live and are not re-encoded);
    without arguments, re-spawn everything and rebuild. Note: stdio servers that
    emit `tools/list_changed` are refreshed automatically."""
    await _ensure_ready()
    if server:
        return await _refresh_server(server)
    for client in _state["clients"].values():
        try:
            await client.close()
        except Exception:  # noqa: BLE001
            pass
    _state["router"] = None
    await _ensure_ready()
    return {"refreshed": True, "total_tools": sum(_state["counts"].values())}


def main() -> None:
    """Console entry point (`toolfinder-mcp`) — run the bridge over stdio."""
    logging.basicConfig(level=os.getenv("TOOLFINDER_LOG", "INFO"))
    mcp.run()


if __name__ == "__main__":
    main()

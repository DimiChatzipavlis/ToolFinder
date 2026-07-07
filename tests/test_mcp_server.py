"""Fast, CI-safe tests for the multi-server MCP bridge.

Downstream MCP servers are faked (no npx/network) by monkeypatching
DynamicMCPClient, so these test the bridge's routing/dispatch logic — union
catalog, cross-server selection, and call dispatch — in milliseconds.
"""

from __future__ import annotations

import importlib

import numpy as np
import pytest

pytest.importorskip("fastmcp")


def _fn(tool):
    """The underlying async function of an ``@mcp.tool``, robust across fastmcp
    versions — some wrap it as a FunctionTool exposing ``.fn``, others return the
    plain function itself."""
    return getattr(tool, "fn", tool)


class FakeEmbedder:
    """Deterministic 4-d embedder: routes by keyword so tests are stable."""

    def __init__(self, model_name, device=None):
        del model_name, device

    def get_sentence_embedding_dimension(self):
        return 4

    def encode(self, texts, batch_size=None, convert_to_numpy=True, **kwargs):
        del batch_size, convert_to_numpy, kwargs
        out = []
        for text in texts:
            t = text.lower()
            if "write" in t or "file" in t or "save" in t:
                out.append([1.0, 0.0, 0.0, 0.0])
            elif "remember" in t or "note" in t or "memory" in t or "entity" in t:
                out.append([0.0, 1.0, 0.0, 0.0])
            else:
                out.append([0.0, 0.0, 1.0, 0.0])
        return np.asarray(out, dtype=np.float32)


class FakeClient:
    """Stands in for a downstream MCP server."""

    # initialize_and_get_tools returns NORMALIZED tools (tool_name, not name).
    INVENTORY = {
        "filesystem": [{"server_name": "filesystem", "tool_name": "write_file", "description": "write a file", "inputSchema": {"type": "object", "properties": {}}}],
        "memory": [{"server_name": "memory", "tool_name": "create_entity", "description": "remember an entity in memory", "inputSchema": {"type": "object", "properties": {}}}],
    }

    def __init__(self, server_name, command, args, env=None, **kwargs):
        del command, args, env, kwargs
        self.server_name = server_name
        self.calls = []

    async def initialize_and_get_tools(self):
        return [dict(t) for t in self.INVENTORY[self.server_name]]

    async def call_tool(self, tool_name, arguments):
        self.calls.append((tool_name, arguments))
        return {"content": [{"type": "text", "text": f"{self.server_name}:{tool_name} ok"}]}

    async def close(self):
        pass


class FailingClient(FakeClient):
    """Like FakeClient, but the server named 'broken' fails to initialize."""

    async def initialize_and_get_tools(self):
        if self.server_name == "broken":
            raise RuntimeError("boom: cannot start")
        return [dict(t) for t in self.INVENTORY[self.server_name]]


@pytest.fixture
def server(monkeypatch, tmp_path):
    import toolfinder.dynamic_faiss_router as router_module

    monkeypatch.setattr(router_module, "SentenceTransformer", FakeEmbedder)
    import toolfinder.mcp_adapter as adapter

    monkeypatch.setattr(adapter, "DynamicMCPClient", FakeClient)

    config = tmp_path / "servers.json"
    config.write_text(
        '{"servers":[{"name":"filesystem","command":"x","args":[]},'
        '{"name":"memory","command":"x","args":[]}]}',
        encoding="utf-8",
    )
    monkeypatch.setenv("TOOLFINDER_CONFIG", str(config))

    srv = importlib.reload(importlib.import_module("toolfinder.mcp_server"))
    monkeypatch.setattr(srv, "DynamicMCPClient", FakeClient)
    return srv


@pytest.mark.asyncio
async def test_catalog_unions_all_servers(server):
    size = await _fn(server.catalog_size)()
    assert size["total_tools"] == 2
    assert size["by_server"] == {"filesystem": 1, "memory": 1}


@pytest.mark.asyncio
async def test_find_tools_routes_across_servers(server):
    fs = await _fn(server.find_tools)("save some text to a file", k=1)
    assert fs[0]["function"]["name"] == "write_file"
    mem = await _fn(server.find_tools)("remember this entity", k=1)
    assert mem[0]["function"]["name"] == "create_entity"


@pytest.mark.asyncio
async def test_route_and_call_dispatches_to_correct_server(server):
    result = await _fn(server.route_and_call)("write a file", {"path": "x"})
    text = result["content"][0]["text"]
    assert text.startswith("filesystem:write_file")


@pytest.mark.asyncio
async def test_call_tool_unknown_name_is_handled(server):
    result = await _fn(server.call_tool)("does_not_exist", {})
    assert "error" in result


@pytest.mark.asyncio
async def test_get_stats_records_routes(server):
    await _fn(server.route_and_call)("write a file", {"path": "x"})
    stats = await _fn(server.get_stats)()
    assert stats["total_routes"] >= 1
    assert "write_file" in stats["top_tools"]
    assert stats["recent"][-1]["tool"] == "write_file"


@pytest.mark.asyncio
async def test_failed_downstream_is_isolated(monkeypatch, tmp_path):
    """One bad server in the config must not take down the whole gateway."""
    import toolfinder.dynamic_faiss_router as router_module

    monkeypatch.setattr(router_module, "SentenceTransformer", FakeEmbedder)
    import toolfinder.mcp_adapter as adapter

    monkeypatch.setattr(adapter, "DynamicMCPClient", FailingClient)
    config = tmp_path / "servers.json"
    config.write_text(
        '{"servers":[{"name":"filesystem","command":"x","args":[]},'
        '{"name":"broken","command":"x","args":[]}]}',
        encoding="utf-8",
    )
    monkeypatch.setenv("TOOLFINDER_CONFIG", str(config))

    srv = importlib.reload(importlib.import_module("toolfinder.mcp_server"))
    monkeypatch.setattr(srv, "DynamicMCPClient", FailingClient)

    size = await _fn(srv.catalog_size)()
    assert size["by_server"] == {"filesystem": 1}      # healthy server still loaded
    assert "broken" in size["failed_servers"]          # bad server isolated + recorded
    fs = await _fn(srv.find_tools)("save some text to a file", k=1)
    assert fs[0]["function"]["name"] == "write_file"


@pytest.mark.asyncio
async def test_per_server_refresh_is_incremental(server, monkeypatch):
    """P2: refresh(server=...) re-lists one server and re-indexes it while the
    other server stays untouched."""
    changed = [{"server_name": "memory", "tool_name": "store_note",
                "description": "remember a note in memory",
                "inputSchema": {"type": "object", "properties": {}}}]
    monkeypatch.setitem(FakeClient.INVENTORY, "memory", changed)

    result = await _fn(server.refresh)("memory")

    assert result == {"refreshed": "memory", "tools": 1, "total_tools": 2}
    hit = await _fn(server.find_tools)("remember this note", k=1)
    assert hit[0]["function"]["name"] == "store_note"          # new tool routes
    missing = await _fn(server.call_tool)("create_entity", {})
    assert "error" in missing                                  # old tool unmapped
    fs = await _fn(server.find_tools)("save some text to a file", k=1)
    assert fs[0]["function"]["name"] == "write_file"           # filesystem untouched


@pytest.mark.asyncio
async def test_list_changed_notification_triggers_refresh(server, monkeypatch):
    """P2 push path: a tools/list_changed notification from a downstream schedules
    a debounced incremental refresh of that server."""
    await _fn(server.catalog_size)()  # ensure built + handlers wired
    client = server._state["clients"]["memory"]
    assert client.on_notification is not None  # bridge wired the hook

    changed = [{"server_name": "memory", "tool_name": "store_note",
                "description": "remember a note in memory",
                "inputSchema": {"type": "object", "properties": {}}}]
    monkeypatch.setitem(FakeClient.INVENTORY, "memory", changed)

    client.on_notification("notifications/tools/list_changed", {})
    await server._state["refresh_tasks"]["memory"]

    hit = await _fn(server.find_tools)("remember this note", k=1)
    assert hit[0]["function"]["name"] == "store_note"


@pytest.mark.asyncio
async def test_auto_rerank_enables_for_stock_encoder_at_threshold(monkeypatch, tmp_path):
    """P0 auto-scale: rerank turns on automatically only when the user made no
    explicit choice, the encoder is the stock default, and the catalog is large."""
    import toolfinder.dynamic_faiss_router as router_module
    import toolfinder.reranker as reranker_module

    monkeypatch.setattr(router_module, "SentenceTransformer", FakeEmbedder)
    import toolfinder.mcp_adapter as adapter

    monkeypatch.setattr(adapter, "DynamicMCPClient", FakeClient)

    class FakeCE:
        def __init__(self, *args, **kwargs):
            pass

        def rank(self, query, documents):
            return [(i, 1.0) for i in range(len(documents))]

    monkeypatch.setattr(reranker_module, "CrossEncoderReranker", FakeCE)

    config = tmp_path / "servers.json"
    config.write_text(
        '{"servers":[{"name":"filesystem","command":"x","args":[]},'
        '{"name":"memory","command":"x","args":[]}]}',
        encoding="utf-8",
    )
    monkeypatch.setenv("TOOLFINDER_CONFIG", str(config))
    monkeypatch.delenv("TOOLFINDER_RERANK", raising=False)
    monkeypatch.delenv("TOOLFINDER_MODEL", raising=False)
    monkeypatch.setenv("TOOLFINDER_SCALE_THRESHOLD", "2")  # the 2 fake tools cross it

    srv = importlib.reload(importlib.import_module("toolfinder.mcp_server"))
    monkeypatch.setattr(srv, "DynamicMCPClient", FakeClient)

    await _fn(srv.catalog_size)()  # triggers _build
    assert isinstance(srv._state["router"]._reranker, FakeCE)

    # An explicit opt-out must win over auto-enable.
    monkeypatch.setenv("TOOLFINDER_RERANK", "0")
    srv2 = importlib.reload(importlib.import_module("toolfinder.mcp_server"))
    monkeypatch.setattr(srv2, "DynamicMCPClient", FakeClient)
    await _fn(srv2.catalog_size)()
    assert srv2._state["router"]._reranker is None

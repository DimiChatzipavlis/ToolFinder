"""LIVE integration test — no fakes, no mocks, no synthetic parts.

Spawns the real `@modelcontextprotocol/server-filesystem` via npx, loads the real
embedding model, routes real intents, executes a real write+read verified on
disk, and exercises the live tool re-list (the P2 refresh path).

Opt-in (it needs Node/npx, network for the first model download, and ~30s):

    TOOLFINDER_LIVE=1 pytest tests/test_integration_live.py -v

It is skipped by default so CI stays fast and hermetic — the fakes in the unit
suite cover logic; THIS covers reality.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("TOOLFINDER_LIVE") != "1" or shutil.which("npx") is None,
    reason="live integration test: set TOOLFINDER_LIVE=1 (requires Node/npx; downloads the embedding model)",
)


def test_live_filesystem_route_execute_and_refresh(tmp_path):
    async def run() -> None:
        from toolfinder import UniversalMCPRouter
        from toolfinder.dynamic_faiss_router import RouterHyperparameters
        from toolfinder.mcp_adapter import DynamicMCPClient

        client = DynamicMCPClient(
            server_name="filesystem", command="npx",
            args=["-y", "@modelcontextprotocol/server-filesystem", str(tmp_path)],
            startup_timeout_s=120.0, request_timeout_s=60.0,
        )
        router = None
        try:
            # 1. Real server, real handshake, real tool list.
            tools = await client.initialize_and_get_tools()
            assert len(tools) >= 10, f"filesystem server exposed only {len(tools)} tools"

            # 2. Real embedding model + index (with the persistent cache exercised).
            router = UniversalMCPRouter(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                config=RouterHyperparameters(cache_dir=str(tmp_path / "embed-cache")),
            )
            assert router.ingest_server("filesystem", tools) == len(tools)

            # 3. Real routing on real schemas.
            names = [r.tool_name for r in router.route_top_k("create a text file with given contents", k=3)]
            assert "write_file" in names, f"router shortlist missed write_file: {names}"

            # 4. Real execution, verified on disk.
            target = (tmp_path / "live.txt").as_posix()
            await client.call_tool("write_file", {"path": target, "content": "live ok"})
            read_back = await client.call_tool("read_text_file", {"path": target})
            assert "live ok" in json.dumps(read_back)
            assert (tmp_path / "live.txt").read_text(encoding="utf-8") == "live ok"

            # 5. Live re-list on the running process (P2 refresh path) + re-index.
            refreshed = await client.refresh_tools()
            assert len(refreshed) == len(tools)
            assert router.reingest_server("filesystem", refreshed) == len(tools)
            assert router.route_top_k("read the contents of a file", k=1)[0].server_name == "filesystem"
        finally:
            if router is not None:
                router.teardown()
            await client.close()

    asyncio.run(run())

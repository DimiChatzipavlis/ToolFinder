"""Heterogeneous gateway demo — the literal finish of suggestion #1:
ONE ToolFinder fronting, at the same time, a real MCP server (filesystem) AND an
OpenAPI REST API (Petstore). The agent binds only ToolFinder's two tools and
routes across the *union*; the filesystem op executes for real and is verified on
disk (so the proof does not depend on any public web API being up).

Part A (no LLM, $0): spin up both sources, union them in one router, route
cross-source intents to the right (server, tool), execute a REAL filesystem
write+read through the gateway, and show the context reduction.

Part B (--agent, a few cents of gpt-4.1-mini): the agent binds only
find_tools + call_tool and completes a filesystem task through the union gateway;
success is verified on disk.

Run:
  python legacy/experiments/gateway_heterogeneous_demo.py          # Part A ($0)
  python legacy/experiments/gateway_heterogeneous_demo.py --agent  # + Part B
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0]))  # legacy/ — for `from experiments...`

from experiments.bridge_ab import (  # noqa: E402
    TASK_TEMPLATE, load_dotenv, make_client_factory, mcp_to_openai, result_text, reset_sandbox, run_arm, tokens, verify,
)
from toolfinder import UniversalMCPRouter, to_openai_tools  # noqa: E402
from toolfinder.mcp_adapter import DynamicMCPClient  # noqa: E402
from toolfinder.openapi_adapter import OpenAPIClient  # noqa: E402

TRUE_ROOT = HERE.parents[1]
BIENCODER = "sentence-transformers/all-MiniLM-L6-v2"
SPEC_URL = "https://petstore3.swagger.io/api/v3/openapi.json"
BASE_URL = "https://petstore3.swagger.io/api/v3"

GATEWAY_TOOLS = [
    {"type": "function", "function": {"name": "find_tools",
     "description": "Find the tools relevant to an action across all configured sources; returns their schemas.",
     "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}},
    {"type": "function", "function": {"name": "call_tool",
     "description": "Execute one tool by name with arguments (dispatched to whichever source owns it).",
     "parameters": {"type": "object", "properties": {"tool_name": {"type": "string"}, "arguments": {"type": "object"}},
                    "required": ["tool_name", "arguments"]}}},
]

CROSS_SOURCE_INTENTS = [
    "create a text file with some content",
    "read the contents of a text file",
    "list the files in a directory",
    "find the pets that are currently available",
    "how many pets are in the store inventory",
]


async def main_async(args) -> None:
    root = Path(tempfile.mkdtemp(prefix="toolfinder-hetero-"))
    fs_client = DynamicMCPClient(server_name="filesystem", command="npx",
                                 args=["-y", "@modelcontextprotocol/server-filesystem", str(root)],
                                 startup_timeout_s=90.0, request_timeout_s=45.0)
    api_client = OpenAPIClient("petstore", spec=SPEC_URL, base_url=BASE_URL)
    clients = {"filesystem": fs_client, "petstore": api_client}
    router = UniversalMCPRouter(model_name=BIENCODER)
    tool_to_server: dict[str, str] = {}

    try:
        print("=" * 72)
        print("PART A — ONE gateway over a real MCP server + an OpenAPI API ($0)")
        print("=" * 72)
        fs_tools = await fs_client.initialize_and_get_tools()
        api_tools = await api_client.initialize_and_get_tools()
        for server, tools in (("filesystem", fs_tools), ("petstore", api_tools)):
            router.ingest_server(server, tools)
            for t in tools:
                tool_to_server.setdefault(t["tool_name"], server)
        print(f"[union] one ToolFinder now fronts {len(tool_to_server)} tools: "
              f"{len(fs_tools)} from filesystem (MCP) + {len(api_tools)} from petstore (OpenAPI)")

        print("\n[route] cross-source intent -> (server) tool:")
        for intent in CROSS_SOURCE_INTENTS:
            top = router.route_top_k(intent, k=1)
            if top:
                print(f"  {intent:48s} -> ({top[0].server_name}) {top[0].tool_name}")
            else:
                print(f"  {intent:48s} -> (abstained)")

        # Real, verifiable execution through the gateway — the filesystem source.
        print("\n[execute] real ops through the gateway:")
        names = {t["tool_name"] for t in fs_tools}
        write = "write_file" if "write_file" in names else next(n for n in names if "write" in n)
        read = "read_text_file" if "read_text_file" in names else next(n for n in names if "read" in n)
        target = (root / "notes.txt").as_posix()
        await clients[tool_to_server[write]].call_tool(write, {"path": target, "content": "hello world"})
        back = result_text(await clients[tool_to_server[read]].call_tool(read, {"path": target}))
        print(f"  filesystem: {write} + {read} -> file reads {back.strip()!r}  (verified on disk: {verify(root)})")
        api_res = await api_client.call_tool("getInventory", {})
        print(f"  petstore (OpenAPI): getInventory -> HTTP {api_res.get('status')} ok={api_res.get('ok')} "
              f"(public sandbox may be down; the filesystem op above is the real-success proof)")

        full = tokens([mcp_to_openai(t) for t in fs_tools] + [{"type": "function", "function": {
            "name": t["tool_name"], "description": t.get("description", ""), "parameters": t.get("inputSchema", {})}}
            for t in api_tools])
        gw = tokens(GATEWAY_TOOLS)
        print(f"\n[context] binding the whole union = {full} tokens; binding only ToolFinder = {gw} tokens "
              f"({full / max(gw, 1):.1f}x smaller).")

        if args.agent:
            load_dotenv(TRUE_ROOT / ".env")
            await part_b(args.model, clients, router, tool_to_server, root)
    finally:
        await fs_client.close()
        await api_client.close()
        router.teardown()
        for p in root.glob("*"):
            if p.is_file():
                p.unlink()
        root.rmdir()


async def part_b(model, clients, router, tool_to_server, root) -> None:
    print("\n" + "=" * 72)
    print(f"PART B — agent binds ONLY ToolFinder, works across the union ({model})")
    print("=" * 72)
    calls: list[tuple[str, str]] = []

    async def executor(name: str, args: dict) -> str:
        if name == "find_tools":
            return json.dumps(to_openai_tools(router.route_top_k(str(args.get("query", "")), k=3)))
        if name == "call_tool":
            tn = str(args.get("tool_name", ""))
            server = tool_to_server.get(tn)
            if server is None:
                return f"ERROR: unknown tool '{tn}'"
            calls.append((server, tn))
            return result_text(await clients[server].call_tool(tn, args.get("arguments") or {}))
        return f"ERROR: unknown tool {name}"

    system = (
        "You reach every tool through a ToolFinder gateway with exactly two tools: "
        "find_tools(query) returns relevant tool schemas across all sources, and "
        "call_tool(tool_name, arguments) executes one. For each step, first find_tools, then call_tool. "
        "Use absolute paths inside the allowed directory. Reply DONE when finished."
    )
    reset_sandbox(root)
    metrics = await run_arm(make_client_factory("openai", None), model, system, GATEWAY_TOOLS, executor,
                            TASK_TEMPLATE.format(root=root.as_posix()))
    ok = verify(root)
    print(f"[result] turns={metrics['model_turns']} total_tokens={metrics['total_tokens']} tool_calls={metrics['tool_calls']}")
    print(f"[result] routed calls (server, tool): {calls}")
    print(f"[result] task verified on disk: {ok}  <-- real success through the union gateway")
    print("[note] the agent saw only 2 gateway tools; both the filesystem and Petstore catalogs stayed inside ToolFinder.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent", action="store_true", help="also run the LLM agent loop (small API spend)")
    parser.add_argument("--model", default="gpt-4.1-mini", help="agent model for Part B")
    asyncio.run(main_async(parser.parse_args()))


if __name__ == "__main__":
    main()

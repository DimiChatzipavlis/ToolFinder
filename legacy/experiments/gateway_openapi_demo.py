"""Live end-to-end demo of the professor's suggestion #1: ToolFinder as the ONE
MCP server, with (OpenAPI-powered) tools configured *under* it — and #2 (the
reranker) shown improving routing on that same live gateway.

We point ToolFinder at the public Swagger Petstore OpenAPI spec, so every Petstore
operation becomes a tool *inside* ToolFinder; an agent only ever sees ToolFinder's
two routing tools, never the API's full catalog.

Part A (no LLM, $0 OpenAI):
  - fetch the spec live -> ToolFinder ingests all operations,
  - route intents to operations with rerank OFF vs ON (does #2 fix #1's routing?),
  - execute a REAL HTTP call through the gateway,
  - show the context reduction (2 gateway tools vs the whole API catalog).

Part B (--agent, a few cents of gpt-4.1-mini):
  - an LLM binds ONLY find_tools + call_tool and completes a Petstore task,
    routing + executing through ToolFinder (rerank on). Reports tokens/turns.

Run:
  python legacy/experiments/gateway_openapi_demo.py            # Part A only ($0)
  python legacy/experiments/gateway_openapi_demo.py --agent    # + Part B (small API)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0]))  # legacy/ — for `from experiments...`

from experiments.bridge_ab import load_dotenv, make_client_factory, run_arm, tokens  # noqa: E402
from toolfinder import UniversalMCPRouter, to_openai_tools  # noqa: E402
from toolfinder.openapi_adapter import OpenAPIClient  # noqa: E402
from toolfinder.reranker import CrossEncoderReranker  # noqa: E402

TRUE_ROOT = HERE.parents[1]
SPEC_URL = "https://petstore3.swagger.io/api/v3/openapi.json"
BASE_URL = "https://petstore3.swagger.io/api/v3"
BIENCODER = "sentence-transformers/all-MiniLM-L6-v2"

GATEWAY_TOOLS = [
    {"type": "function", "function": {"name": "find_tools",
     "description": "Find the tools relevant to an action; returns their schemas.",
     "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}},
    {"type": "function", "function": {"name": "call_tool",
     "description": "Execute one tool by name with arguments.",
     "parameters": {"type": "object", "properties": {"tool_name": {"type": "string"}, "arguments": {"type": "object"}},
                    "required": ["tool_name", "arguments"]}}},
]

# intent -> the operationId a correct router should pick (for a quick scorecard)
INTENTS = {
    "find the pets that are currently available": "findPetsByStatus",
    "get a single pet by its id": "getPetById",
    "place an order to buy a pet": "placeOrder",
    "remove a pet from the store": "deletePet",
    "how many pets are in the store inventory": "getInventory",
}


async def part_a() -> tuple[OpenAPIClient, UniversalMCPRouter]:
    print("=" * 72)
    print("PART A — live OpenAPI gateway (no LLM, $0)")
    print("=" * 72)
    client = OpenAPIClient("petstore", spec=SPEC_URL, base_url=BASE_URL)
    tools = await client.initialize_and_get_tools()
    print(f"[ingest] ToolFinder loaded {len(tools)} Petstore operations under ONE gateway, e.g.: "
          f"{', '.join(t['tool_name'] for t in tools[:6])} ...")

    router = UniversalMCPRouter(model_name=BIENCODER)
    router.ingest_server("petstore", tools)

    def route_one(intent: str) -> str:
        top = router.route_top_k(intent, k=1)
        return top[0].tool_name if top else "(abstained)"

    off = {intent: route_one(intent) for intent in INTENTS}
    router._reranker = CrossEncoderReranker()  # enable #2 on the live gateway
    on = {intent: route_one(intent) for intent in INTENTS}

    off_hits = sum(off[i] == gold for i, gold in INTENTS.items())
    on_hits = sum(on[i] == gold for i, gold in INTENTS.items())
    print(f"\n[route] intent -> operation   (rerank OFF {off_hits}/{len(INTENTS)} vs ON {on_hits}/{len(INTENTS)} correct)")
    print(f"  {'intent':50s} {'OFF':18s} {'ON':18s} gold")
    for intent, gold in INTENTS.items():
        mark_off = "ok" if off[intent] == gold else "  "
        mark_on = "ok" if on[intent] == gold else "  "
        print(f"  {intent:50s} {off[intent]:15s}{mark_off}  {on[intent]:15s}{mark_on}  {gold}")

    print("\n[execute] real HTTP calls through the gateway:")
    for op, args in (("getInventory", {}), ("findPetsByStatus", {"status": "available"})):
        r = await client.call_tool(op, args)
        data = r.get("data")
        shape = f"{len(data)} items" if isinstance(data, list) else (f"keys={list(data)[:4]}" if isinstance(data, dict) else "")
        print(f"  call_tool({op!r}, {args}) -> HTTP {r.get('status')} ok={r.get('ok')} {shape}")

    full = tokens([{"type": "function", "function": {"name": t["tool_name"],
                    "description": t.get("description", ""), "parameters": t.get("inputSchema", {})}} for t in tools])
    gw = tokens(GATEWAY_TOOLS)
    print(f"\n[context] binding the whole API = {full} tokens; binding only ToolFinder = {gw} tokens "
          f"({full / max(gw, 1):.1f}x smaller — the agent never sees the API catalog).")
    return client, router


async def part_b(client: OpenAPIClient, router: UniversalMCPRouter, model: str) -> None:
    print("\n" + "=" * 72)
    print(f"PART B — agent binds ONLY ToolFinder, completes a Petstore task ({model})")
    print("=" * 72)
    calls: list[str] = []

    async def executor(name: str, args: dict) -> str:
        if name == "find_tools":
            return json.dumps(to_openai_tools(router.route_top_k(str(args.get("query", "")), k=3)))
        if name == "call_tool":
            tool_name = str(args.get("tool_name", ""))
            calls.append(tool_name)
            return json.dumps(await client.call_tool(tool_name, args.get("arguments") or {}))[:2000]
        return f"ERROR: unknown tool {name}"

    system = (
        "You reach all tools through a ToolFinder gateway with exactly two tools: "
        "find_tools(query) returns the relevant tool schemas, and call_tool(tool_name, arguments) "
        "executes one. For any action, first call find_tools, then call_tool. When done, reply DONE."
    )
    task = "Report how many pets are available in the store inventory, then reply DONE."

    make_client = make_client_factory("openai", None)
    metrics = await run_arm(make_client, model, system, GATEWAY_TOOLS, executor, task)
    print(f"[result] turns={metrics['model_turns']} total_tokens={metrics['total_tokens']} tool_calls={metrics['tool_calls']}")
    print(f"[result] downstream operations routed through the gateway: {calls}")
    print(f"[result] reached the inventory operation via routing: {'getInventory' in calls}")
    print("[note] the agent's bound context was just the 2 gateway tools; the 19-op Petstore "
          "catalog stayed inside ToolFinder.")


async def main_async(args) -> None:
    client, router = await part_a()
    try:
        if args.agent:
            load_dotenv(TRUE_ROOT / ".env")
            await part_b(client, router, args.model)
    finally:
        await client.close()
        router.teardown()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent", action="store_true", help="also run the LLM agent loop (small API spend)")
    parser.add_argument("--model", default="gpt-4.1-mini", help="agent model for Part B")
    asyncio.run(main_async(parser.parse_args()))


if __name__ == "__main__":
    main()

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Enterprise.runtime.config import EnterpriseConfig
from Enterprise.runtime.event_bus import EnterpriseEventBus
from Enterprise.runtime.executor import HybridToolExecutor
from Enterprise.runtime.orchestrator import HybridEnterpriseOrchestrator
from Enterprise.runtime.planner import HeuristicPlanner
from Enterprise.runtime.policy import PolicyEngine, ToolPolicy
from Enterprise.runtime.registry import HybridToolRegistry


class MockClient:
    def __init__(self, server_name: str) -> None:
        self.server_name = server_name

    async def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if tool_name == "list_directory":
            return {"content": [{"type": "text", "text": "files: input.txt, output.txt"}]}
        if tool_name == "write_file":
            return {
                "content": [
                    {
                        "type": "text",
                        "text": f"wrote {arguments.get('content', '')} to {arguments.get('path', '')}",
                    }
                ]
            }
        if tool_name == "create_entities":
            return {"content": [{"type": "text", "text": "memory entities created"}]}
        return {"content": [{"type": "text", "text": f"{self.server_name}/{tool_name} executed"}]}


async def main() -> None:
    config = EnterpriseConfig(top_k=2, min_score=0.05, max_turns=4)
    registry = HybridToolRegistry(
        model_name=config.model_name,
        allow_low_confidence_keyword_fallback=config.allow_keyword_low_confidence_fallback,
    )

    filesystem_tools = [
        {
            "tool_name": "list_directory",
            "description": "List files in a directory.",
            "inputSchema": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        },
        {
            "tool_name": "write_file",
            "description": "Write text to a file.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["path", "content"],
            },
        },
    ]

    memory_tools = [
        {
            "tool_name": "create_entities",
            "description": "Create entities in the memory graph.",
            "inputSchema": {
                "type": "object",
                "properties": {"entities": {"type": "array"}},
                "required": ["entities"],
            },
        }
    ]

    await registry.upsert_server_tools("filesystem", filesystem_tools)
    await registry.upsert_server_tools("memory", memory_tools)

    clients = {
        "filesystem": MockClient("filesystem"),
        "memory": MockClient("memory"),
    }
    planner = HeuristicPlanner()
    policy = PolicyEngine(ToolPolicy(allowed_servers={"filesystem", "memory"}))
    executor = HybridToolExecutor(clients)

    bus = EnterpriseEventBus(max_handler_errors=config.event_bus_max_errors)

    async def print_event(event: dict[str, Any]) -> None:
        compact = {
            "type": event.get("type"),
            "turn": event.get("turn"),
            "candidate_count": event.get("candidate_count"),
            "tool": event.get("tool"),
            "error": event.get("error"),
        }
        print("[event]", json.dumps(compact, ensure_ascii=True))

    await bus.subscribe(print_event)

    orchestrator = HybridEnterpriseOrchestrator(
        registry=registry,
        planner=planner,
        executor=executor,
        policy_engine=policy,
        config=config,
        event_bus=bus,
    )

    result = await orchestrator.run(
        session_id="demo-session",
        user_query="List the directory contents in the sandbox.",
    )

    print("\n=== SESSION RESULT ===")
    print("status:", result.status)
    print("turns:", result.turns)
    print("answer:", result.answer)
    print("tool_calls:")
    for call in result.tool_calls:
        print(
            f"- turn={call.turn_index} tool={call.server_name}/{call.tool_name} "
            f"args={json.dumps(call.arguments, ensure_ascii=True)}"
        )
    print("telemetry:", json.dumps(result.telemetry, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    asyncio.run(main())

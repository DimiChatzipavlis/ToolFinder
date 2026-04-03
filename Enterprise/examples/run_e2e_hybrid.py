from __future__ import annotations

"""End-to-end hybrid pipeline example: ToolFinder retrieval + OpenClaw agent.

Demonstrates the full hybrid flow:
1. Build tool registry with filesystem + memory MCP tools.
2. Construct the OpenClawHybridPipeline with an HTTP/CLI backend.
3. Run a user query through the pipeline (routing → openclaw agent → fallback).
4. Print rich phase-by-phase output, tool calls, and telemetry.
"""

import argparse
import asyncio
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Enterprise.runtime.config import EnterpriseConfig
from Enterprise.runtime.event_bus import EnterpriseEventBus
from Enterprise.runtime.executor import HybridToolExecutor
from Enterprise.runtime.openclaw_backend import build_openclaw_backend
from Enterprise.runtime.openclaw_hybrid_pipeline import (
    FallbackStrategy,
    OpenClawHybridPipeline,
    OpenClawSessionDriver,
)
from Enterprise.runtime.orchestrator import HybridEnterpriseOrchestrator
from Enterprise.runtime.planner import HeuristicPlanner
from Enterprise.runtime.policy import PolicyEngine, ToolPolicy
from Enterprise.runtime.registry import HybridToolRegistry
from Enterprise.runtime.telemetry import TelemetryCollector
from toolfinder.mcp_adapter import DynamicMCPClient


# ---------------------------------------------------------------------------
# Mock MCP client for offline demos
# ---------------------------------------------------------------------------


class MockPipelineClient:
    """Lightweight mock that simulates MCP tool responses."""

    def __init__(self, server_name: str) -> None:
        self.server_name = server_name

    async def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if tool_name == "list_directory":
            path = arguments.get("path", ".")
            return {
                "content": [
                    {
                        "type": "text",
                        "text": f"Files in {path}: README.md, main.py, config.yaml, tests/",
                    }
                ]
            }
        if tool_name == "read_file":
            path = arguments.get("path", "unknown")
            return {
                "content": [
                    {"type": "text", "text": f"# Contents of {path}\nSample file content."}
                ]
            }
        if tool_name == "write_file":
            path = arguments.get("path", "report.txt")
            content = arguments.get("content", "")
            return {
                "content": [
                    {"type": "text", "text": f"Successfully wrote {len(content)} chars to {path}"}
                ]
            }
        if tool_name == "create_entities":
            entities = arguments.get("entities", [])
            return {
                "content": [
                    {"type": "text", "text": f"Created {len(entities)} entities in memory graph"}
                ]
            }
        if tool_name == "search_nodes":
            query = arguments.get("query", "")
            return {
                "content": [
                    {"type": "text", "text": f"Found 3 nodes matching '{query}'"}
                ]
            }
        return {"content": [{"type": "text", "text": f"{self.server_name}/{tool_name} executed"}]}


def can_launch_npx() -> bool:
    return shutil.which("npx") is not None or shutil.which("npx.cmd") is not None


def strict_mode_env_enabled() -> bool:
    return os.getenv("STRICT_MODE", "0").strip().lower() in {"1", "true", "yes", "on"}


# ---------------------------------------------------------------------------
# Tool catalog
# ---------------------------------------------------------------------------


FILESYSTEM_TOOLS = [
    {
        "tool_name": "list_directory",
        "description": "List all files and subdirectories at a given path.",
        "inputSchema": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
    },
    {
        "tool_name": "read_file",
        "description": "Read the contents of a file at the specified path.",
        "inputSchema": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
    },
    {
        "tool_name": "write_file",
        "description": "Write text content to a file, creating it if needed.",
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

MEMORY_TOOLS = [
    {
        "tool_name": "create_entities",
        "description": "Create named entities in the knowledge memory graph.",
        "inputSchema": {
            "type": "object",
            "properties": {"entities": {"type": "array"}},
            "required": ["entities"],
        },
    },
    {
        "tool_name": "search_nodes",
        "description": "Search for nodes in the knowledge graph by query.",
        "inputSchema": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    },
]


# ---------------------------------------------------------------------------
# Pretty printer
# ---------------------------------------------------------------------------


def print_header(title: str) -> None:
    width = 60
    print(f"\n{'─' * width}")
    print(f"  {title}")
    print(f"{'─' * width}")


def print_result(result: Any) -> None:
    print_header("PIPELINE RESULT")
    print(f"  status:          {result.status}")
    print(f"  execution_path:  {result.execution_path}")
    print(f"  turns:           {result.turns}")
    print(f"  phase_trace:     {' → '.join(result.phase_trace)}")
    print(f"  answer:          {result.answer[:200]}")

    if result.tool_calls:
        print_header("TOOL CALLS")
        for call in result.tool_calls:
            print(
                f"  [{call.turn_index}] {call.server_name}/{call.tool_name} "
                f"args={json.dumps(call.arguments, ensure_ascii=True)}"
            )
            if call.observation:
                print(f"       → {call.observation[:120]}")

    if result.openclaw_response:
        print_header("OPENCLAW AGENT RESPONSE")
        resp = result.openclaw_response
        print(f"  success:     {resp.success}")
        if resp.error:
            print(f"  error:       {resp.error}")
        if resp.metadata:
            print(f"  metadata:    {json.dumps(resp.metadata, ensure_ascii=True)}")
        if resp.tool_calls:
            print(f"  agent_calls: {len(resp.tool_calls)}")

    print_header("TELEMETRY")
    print(json.dumps(result.telemetry, ensure_ascii=True, indent=2))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run end-to-end hybrid pipeline: ToolFinder retrieval + OpenClaw agent."
    )
    parser.add_argument(
        "--backend-kind",
        default=os.getenv("OPENCLAW_BACKEND_KIND", "http"),
        choices=["http", "cli"],
        help="OpenClaw backend type.",
    )
    parser.add_argument(
        "--endpoint",
        default=os.getenv("OPENCLAW_ENDPOINT", "http://127.0.0.1:11434/api/generate"),
    )
    parser.add_argument("--model", default=os.getenv("OPENCLAW_MODEL", "llama3.2"))
    parser.add_argument(
        "--api-mode",
        default=os.getenv("OPENCLAW_API_MODE", "ollama-generate"),
        choices=["ollama-generate", "openai-chat"],
    )
    parser.add_argument(
        "--fallback-strategy",
        default=FallbackStrategy.HEURISTIC_PLANNER,
        choices=list(FallbackStrategy.VALID),
        help="What to do if the OpenClaw agent fails.",
    )
    parser.add_argument(
        "--max-agent-steps",
        type=int,
        default=10,
        help="Maximum steps the OpenClaw agent can take.",
    )
    parser.add_argument(
        "--query",
        default="List the files in the project root directory and summarize the structure.",
        help="Query to execute through the hybrid pipeline.",
    )
    parser.add_argument(
        "--max-cycles",
        type=int,
        default=1,
        help="Number of pipeline runs (for stress/smoke testing).",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        default=strict_mode_env_enabled(),
        help="Disable mocks and require real MCP servers. Fails hard if live servers are unavailable.",
    )
    parser.add_argument(
        "--live-filesystem-root",
        default=os.getenv("LIVE_FILESYSTEM_ROOT", ""),
        help="Filesystem root for live MCP filesystem server (used in strict mode).",
    )
    args = parser.parse_args()
    strict_mode = bool(args.strict)

    # ── Build config & registry ────────────────────────────────────────
    config = EnterpriseConfig.from_env()
    registry = HybridToolRegistry(
        model_name=config.model_name,
        allow_low_confidence_keyword_fallback=config.allow_keyword_low_confidence_fallback,
    )
    live_clients: list[DynamicMCPClient] = []

    if strict_mode:
        if not can_launch_npx():
            raise RuntimeError("STRICT_MODE requires npx to launch real MCP servers.")
        live_root = args.live_filesystem_root or str(REPO_ROOT)

        filesystem_client = DynamicMCPClient(
            server_name="filesystem",
            command="npx",
            args=["-y", "@modelcontextprotocol/server-filesystem", live_root],
            cwd=str(REPO_ROOT),
            startup_timeout_s=90.0,
            request_timeout_s=45.0,
        )
        filesystem_tools = await filesystem_client.initialize_and_get_tools()
        await registry.upsert_server_tools("filesystem", filesystem_tools)

        memory_client = DynamicMCPClient(
            server_name="memory",
            command="npx",
            args=["-y", "@modelcontextprotocol/server-memory"],
            cwd=str(REPO_ROOT),
            startup_timeout_s=90.0,
            request_timeout_s=45.0,
        )
        memory_tools = await memory_client.initialize_and_get_tools()
        await registry.upsert_server_tools("memory", memory_tools)

        live_clients.extend([filesystem_client, memory_client])
        clients: dict[str, Any] = {
            "filesystem": filesystem_client,
            "memory": memory_client,
        }
    else:
        await registry.upsert_server_tools("filesystem", FILESYSTEM_TOOLS)
        await registry.upsert_server_tools("memory", MEMORY_TOOLS)

        # ── Build mock clients & executor ──────────────────────────────
        clients = {
            "filesystem": MockPipelineClient("filesystem"),
            "memory": MockPipelineClient("memory"),
        }

    executor = HybridToolExecutor(clients)

    # ── Build OpenClaw backend & session driver ────────────────────────
    backend = build_openclaw_backend(
        backend_kind=args.backend_kind,
        endpoint=args.endpoint,
        model=args.model,
        api_mode=args.api_mode,
        api_key=os.getenv("OPENCLAW_API_KEY"),
        timeout_s=config.planner_timeout_s,
        cli_binary=os.getenv("OPENCLAW_CLI_BIN", "openclaw"),
    )
    session_driver = OpenClawSessionDriver(backend=backend, timeout_s=config.planner_timeout_s)

    # ── Build fallback orchestrator ────────────────────────────────────
    planner = HeuristicPlanner()
    policy_engine = PolicyEngine(ToolPolicy(allowed_servers={"filesystem", "memory"}))
    fallback_orchestrator = HybridEnterpriseOrchestrator(
        registry=registry,
        planner=planner,
        executor=executor,
        policy_engine=policy_engine,
        config=config,
    )

    # ── Build event bus with logging ───────────────────────────────────
    event_bus = EnterpriseEventBus(max_handler_errors=config.event_bus_max_errors)

    async def log_event(event: dict[str, Any]) -> None:
        event_type = event.get("type", "unknown")
        compact = {k: v for k, v in event.items() if k not in ("timestamp",) and v is not None}
        print(f"  [event] {event_type}: {json.dumps(compact, ensure_ascii=True)}")

    await event_bus.subscribe(log_event)

    # ── Build pipeline ─────────────────────────────────────────────────
    pipeline = OpenClawHybridPipeline(
        registry=registry,
        session_driver=session_driver,
        fallback_orchestrator=fallback_orchestrator,
        executor=executor,
        config=config,
        event_bus=event_bus,
        telemetry=TelemetryCollector(
            max_latency_samples_per_metric=config.telemetry_max_latency_samples,
            allowed_root=config.telemetry_allowed_root or None,
        ),
        fallback_strategy=args.fallback_strategy,
        max_agent_steps=args.max_agent_steps,
        model=args.model,
    )

    # ── Run pipeline ───────────────────────────────────────────────────
    print_header("END-TO-END HYBRID PIPELINE")
    print(f"  backend:   {args.backend_kind}")
    print(f"  model:     {args.model}")
    print(f"  fallback:  {args.fallback_strategy}")
    print(f"  strict:    {strict_mode}")
    print(f"  query:     {args.query[:80]}")

    try:
        for cycle in range(1, args.max_cycles + 1):
            if args.max_cycles > 1:
                print(f"\n  ── Cycle {cycle}/{args.max_cycles} ──")

            result = await pipeline.run(
                session_id=f"e2e-demo-{cycle}",
                user_query=args.query,
            )
            print_result(result)
    finally:
        for client in reversed(live_clients):
            await client.close()


if __name__ == "__main__":
    asyncio.run(main())

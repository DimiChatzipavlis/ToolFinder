from __future__ import annotations

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
from Enterprise.runtime.orchestrator import HybridEnterpriseOrchestrator
from Enterprise.runtime.planner import OpenClawPlanner
from Enterprise.runtime.policy import PolicyEngine, ToolPolicy
from Enterprise.runtime.realtime_service import RealTimeHybridService, WorkspaceChangeTracker
from Enterprise.runtime.registry import HybridToolRegistry
from toolfinder.mcp_adapter import DynamicMCPClient


class MockRealtimeClient:
    def __init__(self, server_name: str) -> None:
        self.server_name = server_name

    async def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if tool_name == "list_directory":
            path = arguments.get("path", ".")
            return {"content": [{"type": "text", "text": f"directory snapshot requested for {path}"}]}
        if tool_name == "write_file":
            path = arguments.get("path", "report.txt")
            content = arguments.get("content", "")
            return {"content": [{"type": "text", "text": f"mock write to {path}: {content}"}]}
        if tool_name == "create_entities":
            return {"content": [{"type": "text", "text": "memory graph updated"}]}
        return {"content": [{"type": "text", "text": f"{self.server_name}/{tool_name} executed"}]}


def build_query(changed_paths: list[str]) -> str:
    if not changed_paths:
        return "List repository files and summarize relevant changes."
    joined = ", ".join(changed_paths[:5])
    return (
        "Analyze recent repository changes and choose a tool action. "
        f"Changed files: {joined}."
    )


def can_launch_npx() -> bool:
    return shutil.which("npx") is not None or shutil.which("npx.cmd") is not None


def strict_mode_env_enabled() -> bool:
    return os.getenv("STRICT_MODE", "0").strip().lower() in {"1", "true", "yes", "on"}


async def main() -> None:
    parser = argparse.ArgumentParser(description="Run real-time OpenClaw hybrid orchestration loop.")
    parser.add_argument("--workspace", default=str(REPO_ROOT), help="Workspace path to watch for changes.")
    parser.add_argument(
        "--backend-kind",
        default=os.getenv("OPENCLAW_BACKEND_KIND", "http"),
        choices=["http", "cli"],
        help="Planner backend type: http endpoint or local openclaw CLI.",
    )
    parser.add_argument("--endpoint", default=os.getenv("OPENCLAW_ENDPOINT", "http://127.0.0.1:11434/api/generate"))
    parser.add_argument("--model", default=os.getenv("OPENCLAW_MODEL", "llama3.2"))
    parser.add_argument(
        "--api-mode",
        default=os.getenv("OPENCLAW_API_MODE", "ollama-generate"),
        choices=["ollama-generate", "openai-chat"],
    )
    parser.add_argument("--poll-interval", type=float, default=1.5)
    parser.add_argument(
        "--tool-runtime",
        default="auto",
        choices=["auto", "mock", "live"],
        help="Tool execution mode for filesystem server.",
    )
    parser.add_argument(
        "--live-filesystem-root",
        default="",
        help="If set, runs a real MCP filesystem server for this root via npx.",
    )
    parser.add_argument(
        "--openclaw-cli-bin",
        default=os.getenv("OPENCLAW_CLI_BIN", "openclaw"),
        help="Binary path/name for backend-kind=cli.",
    )
    parser.add_argument(
        "--max-cycles",
        type=int,
        default=0,
        help="Run finite polling cycles for smoke tests; 0 means run forever.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        default=strict_mode_env_enabled(),
        help="Disable all mock clients; require real MCP servers and fail hard if unavailable.",
    )
    args = parser.parse_args()
    strict_mode = bool(args.strict)

    config = EnterpriseConfig.from_env()
    registry = HybridToolRegistry(
        model_name=config.model_name,
        allow_low_confidence_keyword_fallback=config.allow_keyword_low_confidence_fallback,
    )

    filesystem_tools = [
        {
            "tool_name": "list_directory",
            "description": "List files and directories for a target path.",
            "inputSchema": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        },
        {
            "tool_name": "write_file",
            "description": "Write a textual summary report.",
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
    if not strict_mode:
        await registry.upsert_server_tools(
            "memory",
            [
                {
                    "tool_name": "create_entities",
                    "description": "Persist summary entities in memory graph.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {"entities": {"type": "array"}},
                        "required": ["entities"],
                    },
                }
            ],
        )

    backend = build_openclaw_backend(
        backend_kind=args.backend_kind,
        endpoint=args.endpoint,
        model=args.model,
        api_mode=args.api_mode,
        api_key=os.getenv("OPENCLAW_API_KEY"),
        timeout_s=config.planner_timeout_s,
        cli_binary=args.openclaw_cli_bin,
    )
    planner = OpenClawPlanner(
        backend=backend,
        timeout_s=config.planner_timeout_s,
        fallback_on_parse_error=config.planner_fallback_on_parse_error,
        fallback_allows_tool_retry=config.planner_fallback_allows_tool_retry,
    )

    clients: dict[str, Any] = {}
    live_clients: list[DynamicMCPClient] = []

    effective_live_root = args.live_filesystem_root
    if strict_mode and args.tool_runtime == "mock":
        raise RuntimeError("STRICT_MODE prohibits --tool-runtime mock; use live/auto with real MCP servers.")
    if strict_mode and not can_launch_npx():
        raise RuntimeError("STRICT_MODE requires npx to launch real MCP servers.")

    if not effective_live_root and args.tool_runtime == "live":
        effective_live_root = args.workspace
    if not effective_live_root and args.tool_runtime == "auto" and can_launch_npx():
        effective_live_root = args.workspace
    if strict_mode and not effective_live_root:
        effective_live_root = args.workspace

    if effective_live_root:
        filesystem_client = DynamicMCPClient(
            server_name="filesystem",
            command="npx",
            args=["-y", "@modelcontextprotocol/server-filesystem", effective_live_root],
            cwd=str(REPO_ROOT),
            startup_timeout_s=90.0,
            request_timeout_s=45.0,
        )
        discovered_tools = await filesystem_client.initialize_and_get_tools()
        await registry.upsert_server_tools("filesystem", discovered_tools)
        clients["filesystem"] = filesystem_client
        live_clients.append(filesystem_client)
    else:
        if strict_mode:
            raise RuntimeError("STRICT_MODE requires a live filesystem MCP server.")
        await registry.upsert_server_tools("filesystem", filesystem_tools)
        clients["filesystem"] = MockRealtimeClient("filesystem")

    if strict_mode:
        memory_client = DynamicMCPClient(
            server_name="memory",
            command="npx",
            args=["-y", "@modelcontextprotocol/server-memory"],
            cwd=str(REPO_ROOT),
            startup_timeout_s=90.0,
            request_timeout_s=45.0,
        )
        discovered_memory_tools = await memory_client.initialize_and_get_tools()
        await registry.upsert_server_tools("memory", discovered_memory_tools)
        clients["memory"] = memory_client
        live_clients.append(memory_client)
    else:
        clients["memory"] = MockRealtimeClient("memory")

    executor = HybridToolExecutor(clients)
    policy_engine = PolicyEngine(ToolPolicy(allowed_servers={"filesystem", "memory"}))

    event_bus = EnterpriseEventBus(max_handler_errors=config.event_bus_max_errors)

    async def print_event(event: dict[str, Any]) -> None:
        filtered = {
            "type": event.get("type"),
            "turn": event.get("turn"),
            "candidate_count": event.get("candidate_count"),
            "decision": event.get("decision"),
            "tool": event.get("tool"),
            "error": event.get("error"),
        }
        print("[event]", json.dumps(filtered, ensure_ascii=True))

    event_bus.subscribe(print_event)

    orchestrator = HybridEnterpriseOrchestrator(
        registry=registry,
        planner=planner,
        executor=executor,
        policy_engine=policy_engine,
        config=config,
        event_bus=event_bus,
    )

    tracker = WorkspaceChangeTracker(root=Path(args.workspace))
    service = RealTimeHybridService(
        orchestrator=orchestrator,
        tracker=tracker,
        query_builder=build_query,
        poll_interval_s=args.poll_interval,
        run_on_startup=config.realtime_run_on_startup,
        error_backoff_s=config.realtime_error_backoff_s,
    )

    print("Starting real-time OpenClaw hybrid service")
    print("workspace:", args.workspace)
    print("backend_kind:", args.backend_kind)
    print("strict_mode:", strict_mode)
    if args.backend_kind == "http":
        print("endpoint:", args.endpoint)
    print("api_mode:", args.api_mode)
    print("live_filesystem_mcp:", bool(effective_live_root))

    try:
        if args.max_cycles > 0:
            await service.run_for_cycles(args.max_cycles)
            return
        await service.run_forever()
    finally:
        for client in reversed(live_clients):
            await client.close()


if __name__ == "__main__":
    asyncio.run(main())

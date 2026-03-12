from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import logging
import tempfile
from pathlib import Path
from typing import Any

from toolfinder import AutonomousMCPAgent, DynamicMCPClient
from toolfinder.mcp_adapter import MCPClientError, ServerProcessConfig


GOAL = "Fetch the JSON from 'https://raw.githubusercontent.com/modelcontextprotocol/specification/main/package.json'. Create a table in sqlite named 'metadata' with a 'description' text column. Insert the description found in the fetched JSON into the table. Finally, create a memory note stating the metadata extraction is complete."


def build_server_configs(temp_dir: Path) -> list[ServerProcessConfig]:
    memory_path = temp_dir / "memory.jsonl"
    sqlite_path = temp_dir / "tri_server.db"
    sqlite_path.touch(exist_ok=True)

    return [
        ServerProcessConfig(
            server_name="memory",
            command="npx",
            args=("-y", "@modelcontextprotocol/server-memory"),
            env={"MEMORY_FILE_PATH": str(memory_path)},
        ),
        ServerProcessConfig(
            server_name="sqlite",
            command="npx",
            args=("-y", "@modelcontextprotocol/server-sqlite", "--db-path", str(sqlite_path)),
        ),
        ServerProcessConfig(
            server_name="fetch",
            command="npx",
            args=("-y", "@modelcontextprotocol/server-fetch"),
        ),
    ]


def build_sqlite_fallback_config(temp_dir: Path) -> ServerProcessConfig:
    sqlite_path = temp_dir / "tri_server.db"
    sqlite_path.touch(exist_ok=True)
    return ServerProcessConfig(
        server_name="sqlite",
        command="npx",
        args=("-y", "mcp-server-sqlite"),
        env={"SQLITE_DB_PATH": str(sqlite_path)},
    )


def build_fetch_fallback_config() -> ServerProcessConfig:
    return ServerProcessConfig(
        server_name="fetch",
        command="npx",
        args=("-y", "mcp-server-fetch"),
    )


def build_fetch_second_fallback_config() -> ServerProcessConfig:
    return ServerProcessConfig(
        server_name="fetch",
        command="npx",
        args=("-y", "mcp-fetch-server"),
    )


async def connect_client(config: ServerProcessConfig) -> tuple[DynamicMCPClient, list[dict[str, Any]]]:
    client = DynamicMCPClient(
        server_name=config.server_name,
        command=config.command,
        args=list(config.args),
        env=config.env,
        cwd=config.cwd,
        startup_timeout_s=90.0,
        request_timeout_s=45.0,
    )
    tools = await client.initialize_and_get_tools()
    return client, tools


async def connect_with_fallback(
    primary: ServerProcessConfig,
    fallback: ServerProcessConfig | None = None,
    second_fallback: ServerProcessConfig | None = None,
) -> tuple[DynamicMCPClient, list[dict[str, Any]]]:
    try:
        return await connect_client(primary)
    except (MCPClientError, Exception) as primary_exc:
        if fallback is None:
            raise
        logging.warning(
            "Failed to initialize %s via %s %s: %s. Falling back to %s %s",
            primary.server_name,
            primary.command,
            " ".join(primary.args),
            primary_exc,
            fallback.command,
            " ".join(fallback.args),
        )
        try:
            return await connect_client(fallback)
        except (MCPClientError, Exception) as fallback_exc:
            if second_fallback is None:
                raise
            logging.warning(
                "Failed to initialize %s via %s %s: %s. Falling back to %s %s",
                fallback.server_name,
                fallback.command,
                " ".join(fallback.args),
                fallback_exc,
                second_fallback.command,
                " ".join(second_fallback.args),
            )
            return await connect_client(second_fallback)


def print_inventory(server_name: str, tools: list[dict[str, Any]]) -> None:
    tool_names = ", ".join(tool["tool_name"] for tool in tools)
    print(f"[{server_name}] tools: {tool_names}")


def print_iteration_metrics(result: Any) -> None:
    print("=== Iteration Metrics ===")
    for step in result.steps:
        active_context_size = len(json.dumps(step.available_tools, ensure_ascii=True))
        selected_server = step.server_name if step.server_name is not None else "none"
        print(f"Iteration {step.iteration}")
        print(f"Routing Latency: {step.routing_latency_ms:.2f} ms")
        print(f"Active Context Size: {active_context_size} chars")
        print(f"Selected Server: {selected_server}")
        print(f"Action: {step.action}")
        print(f"Thought: {step.thought}")
        print(f"Observation: {step.observation}")
        print()


async def run_proof(model_name: str, ollama_model: str) -> None:
    temp_dir = Path(tempfile.mkdtemp(prefix="tri-server-proof-"))
    clients: list[DynamicMCPClient] = []

    try:
        configs = build_server_configs(temp_dir)
        inventories: list[tuple[str, list[dict[str, Any]]]] = []
        for config in configs:
            fallback_config: ServerProcessConfig | None = None
            second_fallback_config: ServerProcessConfig | None = None
            if config.server_name == "sqlite":
                fallback_config = build_sqlite_fallback_config(temp_dir)
            elif config.server_name == "fetch":
                fallback_config = build_fetch_fallback_config()
                second_fallback_config = build_fetch_second_fallback_config()

            client, tools = await connect_with_fallback(config, fallback_config, second_fallback_config)
            clients.append(client)
            inventories.append((config.server_name, tools))

        print("=== Server Inventory ===")
        for server_name, tools in inventories:
            print_inventory(server_name, tools)
        print()

        async with AutonomousMCPAgent(model_name=model_name, ollama_model=ollama_model, max_iterations=7) as agent:
            for client in clients:
                await agent.register_server(client.server_name, client)

            result = await agent.execute_task(GOAL)

        print_iteration_metrics(result)

        print("=== Final Result ===")
        print(f"Status: {result.status}")
        print(f"Answer: {result.answer}")
        print("Scratchpad:")
        print(json.dumps(result.scratchpad, ensure_ascii=True, indent=2, sort_keys=True))
    finally:
        for client in reversed(clients):
            with contextlib.suppress(Exception):
                await client.close()
        with contextlib.suppress(Exception):
            for path in temp_dir.iterdir():
                path.unlink()
            temp_dir.rmdir()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a tri-server scalability proof for the autonomous MCP agent.")
    parser.add_argument(
        "--model-name",
        default="sentence-transformers/all-mpnet-base-v2",
        help="SentenceTransformer model name or local path.",
    )
    parser.add_argument(
        "--ollama-model",
        default="llama3.2",
        help="Local Ollama model name.",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    args = parse_args()
    asyncio.run(run_proof(model_name=args.model_name, ollama_model=args.ollama_model))


if __name__ == "__main__":
    main()
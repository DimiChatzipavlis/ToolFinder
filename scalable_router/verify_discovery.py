from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import logging
import sqlite3
import tempfile
from pathlib import Path

if __package__ in {None, ""}:
    from autonomous_agent import AutonomousMCPAgent
    from mcp_adapter import DynamicMCPClient, ServerProcessConfig
else:
    from .autonomous_agent import AutonomousMCPAgent
    from .mcp_adapter import DynamicMCPClient, ServerProcessConfig


DISCOVERY_QUERY = (
    "Find out what tables exist in the sqlite database. If there is a table, look at its schema. "
    "Then, write a memory note summarizing the database structure."
)


def seed_sqlite_database(db_path: Path) -> None:
    connection = sqlite3.connect(db_path)
    try:
        cursor = connection.cursor()
        cursor.executescript(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY,
                username TEXT NOT NULL,
                preferred_language TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS projects (
                id INTEGER PRIMARY KEY,
                owner_id INTEGER NOT NULL,
                project_name TEXT NOT NULL,
                FOREIGN KEY(owner_id) REFERENCES users(id)
            );

            DELETE FROM users;
            DELETE FROM projects;

            INSERT INTO users (id, username, preferred_language) VALUES
                (1, 'alice', 'Python'),
                (2, 'bob', 'Rust');

            INSERT INTO projects (id, owner_id, project_name) VALUES
                (1, 1, 'router-audit'),
                (2, 2, 'schema-check');
            """
        )
        connection.commit()
    finally:
        connection.close()


async def connect_client(config: ServerProcessConfig) -> tuple[DynamicMCPClient, list[dict[str, object]]]:
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


async def connect_sqlite_client(sqlite_path: Path) -> tuple[DynamicMCPClient, list[dict[str, object]], str]:
    candidates = [
        ServerProcessConfig(
            server_name="sqlite",
            command="npx",
            args=("-y", "mcp-server-sqlite"),
            env={"SQLITE_DB_PATH": str(sqlite_path)},
        ),
        ServerProcessConfig(
            server_name="sqlite",
            command="npx",
            args=("-y", "@modelcontextprotocol/server-sqlite", "--db-path", str(sqlite_path)),
        ),
    ]

    failures: list[str] = []
    for candidate in candidates:
        client: DynamicMCPClient | None = None
        try:
            client, tools = await connect_client(candidate)
            return client, tools, f"{candidate.command} {' '.join(candidate.args)}"
        except Exception as exc:
            failures.append(f"{candidate.command} {' '.join(candidate.args)} -> {exc}")
            if client is not None:
                with contextlib.suppress(Exception):
                    await client.close()

    raise RuntimeError("No sqlite server candidate could be started:\n" + "\n".join(failures))


def read_memory_file(memory_path: Path) -> str:
    if not memory_path.exists():
        return ""
    return memory_path.read_text(encoding="utf-8")


def print_trace(result: object) -> None:
    steps = getattr(result, "steps")
    for step in steps:
        print(f"Iteration {step.iteration}")
        print(f"Routing Latency Ms: {step.routing_latency_ms:.2f}")
        print(f"Available Tools: {json.dumps(step.available_tools, ensure_ascii=True)}")
        print(f"Thought: {step.thought}")
        if step.action == "tool_call":
            print(
                "Action: "
                f"{step.server_name}/{step.tool_name} {json.dumps(step.arguments or {}, ensure_ascii=True, sort_keys=True)}"
            )
        else:
            print(f"Action: {step.action}")
        print(f"Observation: {step.observation}")
        print(f"Raw Model Output: {step.raw_model_output}")
        print()


async def run_verification(model_name: str, ollama_model: str) -> None:
    temp_dir = Path(tempfile.mkdtemp(prefix="scalable-router-discovery-"))
    memory_path = temp_dir / "memory.jsonl"
    sqlite_path = temp_dir / "discovery.db"
    seed_sqlite_database(sqlite_path)

    memory_config = ServerProcessConfig(
        server_name="memory",
        command="npx",
        args=("-y", "@modelcontextprotocol/server-memory"),
        env={"MEMORY_FILE_PATH": str(memory_path)},
    )

    memory_client: DynamicMCPClient | None = None
    sqlite_client: DynamicMCPClient | None = None
    try:
        memory_client, memory_tools = await connect_client(memory_config)
        sqlite_client, sqlite_tools, sqlite_command = await connect_sqlite_client(sqlite_path)

        print("=== Server Inventory ===")
        print(f"memory tools: {json.dumps(memory_tools, ensure_ascii=True)}")
        print(f"sqlite command: {sqlite_command}")
        print(f"sqlite tools: {json.dumps(sqlite_tools, ensure_ascii=True)}")
        print()

        async with AutonomousMCPAgent(model_name=model_name, ollama_model=ollama_model, max_iterations=7) as agent:
            await agent.register_server("memory", memory_client)
            await agent.register_server("sqlite", sqlite_client)
            result = await agent.execute_task(DISCOVERY_QUERY)

        print("=== ReAct Trace ===")
        print_trace(result)

        print("=== Final Result ===")
        print(f"Status: {result.status}")
        print(f"Answer: {result.answer}")
        print("Scratchpad:")
        print(json.dumps(result.scratchpad, ensure_ascii=True, indent=2, sort_keys=True))
        print()

        memory_text = read_memory_file(memory_path)
        print("=== Memory File ===")
        print(memory_text)

        if result.status != "complete":
            raise AssertionError(f"Discovery benchmark did not complete successfully: {result.answer}")

        for expected_fragment in ("users", "projects"):
            if expected_fragment not in memory_text.lower():
                raise AssertionError(
                    f"Expected memory output to mention '{expected_fragment}', memory file was: {memory_text}"
                )
    finally:
        for client in (sqlite_client, memory_client):
            if client is not None:
                with contextlib.suppress(Exception):
                    await client.close()
        with contextlib.suppress(Exception):
            for path in temp_dir.iterdir():
                path.unlink()
            temp_dir.rmdir()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify autonomous discovery without hardcoded tool selection.")
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
    asyncio.run(run_verification(model_name=args.model_name, ollama_model=args.ollama_model))


if __name__ == "__main__":
    main()
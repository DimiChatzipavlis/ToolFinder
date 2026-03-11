from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import sqlite3
import tempfile
from pathlib import Path

if __package__ in {None, ""}:
    from autonomous_agent import AutonomousMCPAgent, extract_text_from_tool_result
    from mcp_adapter import DynamicMCPClient, ServerProcessConfig
else:
    from .autonomous_agent import AutonomousMCPAgent, extract_text_from_tool_result
    from .mcp_adapter import DynamicMCPClient, ServerProcessConfig


def seed_sqlite_database(db_path: Path) -> None:
    connection = sqlite3.connect(db_path)
    try:
        cursor = connection.cursor()
        cursor.executescript(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                user_name TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS user_data (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                user_name TEXT NOT NULL
            );

            DELETE FROM users;
            DELETE FROM user_data;

            INSERT INTO users (id, name, user_name) VALUES (1, 'Dimit', 'Dimit');
            INSERT INTO user_data (id, name, user_name) VALUES (1, 'Dimit', 'Dimit');
            """
        )
        connection.commit()
    finally:
        connection.close()


def collect_database_summary(db_path: Path) -> tuple[list[str], list[str]]:
    connection = sqlite3.connect(db_path)
    try:
        cursor = connection.cursor()
        table_rows = cursor.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
        ).fetchall()
        table_names = [str(row[0]) for row in table_rows]

        content_fragments: list[str] = []
        for table_name in table_names:
            rows = cursor.execute(f"SELECT * FROM {table_name} ORDER BY 1").fetchall()
            for row in rows:
                content_fragments.extend(str(value) for value in row)
        return table_names, content_fragments
    finally:
        connection.close()


async def connect_client(config: ServerProcessConfig) -> DynamicMCPClient:
    client = DynamicMCPClient(
        server_name=config.server_name,
        command=config.command,
        args=list(config.args),
        env=config.env,
        cwd=config.cwd,
        startup_timeout_s=90.0,
        request_timeout_s=45.0,
    )
    await client.initialize_and_get_tools()
    return client


async def connect_sqlite_client(sqlite_path: Path) -> DynamicMCPClient:
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
            client = await connect_client(candidate)
            return client
        except Exception as exc:
            failures.append(f"{candidate.command} {' '.join(candidate.args)} -> {exc}")
            if client is not None:
                with contextlib.suppress(Exception):
                    await client.close()

    raise RuntimeError("No sqlite server candidate could be started:\n" + "\n".join(failures))


async def run_verification(model_name: str, ollama_model: str) -> None:
    temp_dir = Path(tempfile.mkdtemp(prefix="react-agent-"))
    memory_path = temp_dir / "memory.jsonl"
    sqlite_path = temp_dir / "react.db"
    seed_sqlite_database(sqlite_path)
    expected_tables, expected_fragments = collect_database_summary(sqlite_path)

    memory_config = ServerProcessConfig(
        server_name="memory",
        command="npx",
        args=("-y", "@modelcontextprotocol/server-memory"),
        env={"MEMORY_FILE_PATH": str(memory_path)},
    )

    memory_client: DynamicMCPClient | None = None
    sqlite_client: DynamicMCPClient | None = None
    try:
        memory_client = await connect_client(memory_config)
        sqlite_client = await connect_sqlite_client(sqlite_path)

        async with AutonomousMCPAgent(model_name=model_name, ollama_model=ollama_model, max_iterations=8) as agent:
            await agent.register_server("memory", memory_client)
            await agent.register_server("sqlite", sqlite_client)

            result = await agent.execute_task(
                "List all tables in the sqlite database. If there is a table, read its contents. Then, create a memory note summarizing what you found."
            )

            print("=== ReAct Trace ===")
            for step in result.steps:
                print(f"Iteration {step.iteration}")
                print(f"Thought: {step.thought}")
                if step.action == "tool_call":
                    print(
                        "Tool Call: "
                        f"{step.server_name}/{step.tool_name} {json.dumps(step.arguments or {}, ensure_ascii=True, sort_keys=True)}"
                    )
                else:
                    print("Tool Call: none")
                print(f"Observation: {step.observation}")
                print()

            print("=== Final Answer ===")
            print(result.answer)

            memory_graph = await memory_client.call_tool("read_graph", {})
            memory_text = extract_text_from_tool_result(memory_graph)
            missing_tables = [table_name for table_name in expected_tables if table_name not in memory_text]
            if missing_tables:
                raise AssertionError(
                    "Expected memory note to mention discovered tables. "
                    f"Missing tables={missing_tables}, memory_text={memory_text}"
                )

            representative_fragments = [fragment for fragment in expected_fragments if fragment and not fragment.isdigit()]
            if representative_fragments:
                if not any(fragment in memory_text for fragment in representative_fragments):
                    raise AssertionError(
                        "Expected memory note to mention at least one value from the discovered table contents. "
                        f"Candidates={representative_fragments}, memory_text={memory_text}"
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
    parser = argparse.ArgumentParser(description="Verify the multi-server ReAct autonomous MCP agent.")
    parser.add_argument(
        "--model-name",
        default="sentence-transformers/all-mpnet-base-v2",
        help="SentenceTransformer model name or local path.",
    )
    parser.add_argument(
        "--ollama-model",
        default="llama3.2:latest",
        help="Local Ollama model name.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    asyncio.run(run_verification(model_name=args.model_name, ollama_model=args.ollama_model))


if __name__ == "__main__":
    main()
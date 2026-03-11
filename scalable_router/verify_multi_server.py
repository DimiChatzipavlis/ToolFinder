from __future__ import annotations

import argparse
import asyncio
import contextlib
import sqlite3
import statistics
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    from dynamic_faiss_router import RouteResult, UniversalMCPRouter
    from mcp_adapter import DynamicMCPClient
else:
    from .dynamic_faiss_router import RouteResult, UniversalMCPRouter
    from .mcp_adapter import DynamicMCPClient


@dataclass(frozen=True)
class ServerCandidate:
    server_name: str
    command: str
    args: tuple[str, ...]
    env: dict[str, str] | None = None


@dataclass(frozen=True)
class BenchmarkCase:
    query: str
    expected_server: str
    expected_tool: str


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

            CREATE TABLE IF NOT EXISTS orders (
                id INTEGER PRIMARY KEY,
                user_id INTEGER NOT NULL,
                total_amount REAL NOT NULL,
                FOREIGN KEY(user_id) REFERENCES users(id)
            );

            DELETE FROM users;
            DELETE FROM orders;

            INSERT INTO users (id, username, preferred_language) VALUES
                (1, 'alice', 'Python'),
                (2, 'bob', 'SQL');

            INSERT INTO orders (id, user_id, total_amount) VALUES
                (1, 1, 99.50),
                (2, 2, 15.25);
            """
        )
        connection.commit()
    finally:
        connection.close()


def choose_available_tool(available_tools: set[str], preferred_names: list[str]) -> str:
    for name in preferred_names:
        if name in available_tools:
            return name
    raise RuntimeError(f"Expected one of {preferred_names}, but only found: {sorted(available_tools)}")


def build_benchmark_cases(memory_tools: list[dict[str, Any]], sqlite_tools: list[dict[str, Any]]) -> list[BenchmarkCase]:
    memory_tool_names = {tool["tool_name"] for tool in memory_tools}
    sqlite_tool_names = {tool["tool_name"] for tool in sqlite_tools}

    memory_add = choose_available_tool(memory_tool_names, ["add_observations", "create_entities"])
    memory_search = choose_available_tool(memory_tool_names, ["search_nodes", "open_nodes"])
    memory_read = choose_available_tool(memory_tool_names, ["read_graph"])

    sqlite_select = choose_available_tool(sqlite_tool_names, ["query", "read_query"])
    sqlite_list_tables = choose_available_tool(sqlite_tool_names, ["list-tables", "list_tables"])
    sqlite_describe = choose_available_tool(sqlite_tool_names, ["describe-table", "describe_table"])

    return [
        BenchmarkCase(
            query="Add a new observation to the knowledge graph that the user prefers Python.",
            expected_server="memory",
            expected_tool=memory_add,
        ),
        BenchmarkCase(
            query="Search the knowledge graph for nodes related to Python preferences.",
            expected_server="memory",
            expected_tool=memory_search,
        ),
        BenchmarkCase(
            query="Show the entire knowledge graph you currently remember.",
            expected_server="memory",
            expected_tool=memory_read,
        ),
        BenchmarkCase(
            query="Execute a SELECT * FROM users query and return the rows.",
            expected_server="sqlite",
            expected_tool=sqlite_select,
        ),
        BenchmarkCase(
            query="List every table in the SQLite database.",
            expected_server="sqlite",
            expected_tool=sqlite_list_tables,
        ),
        BenchmarkCase(
            query="Show the existing columns and data types for the users table.",
            expected_server="sqlite",
            expected_tool=sqlite_describe,
        ),
    ]


async def connect_and_list_tools(candidate: ServerCandidate) -> tuple[DynamicMCPClient, list[dict[str, Any]]]:
    client = DynamicMCPClient(
        server_name=candidate.server_name,
        command=candidate.command,
        args=list(candidate.args),
        env=candidate.env,
        startup_timeout_s=90.0,
        request_timeout_s=45.0,
    )
    tools = await client.initialize_and_get_tools()
    if not tools:
        raise RuntimeError(f"{candidate.server_name}: no tools returned by server")
    return client, tools


async def connect_first_available(candidates: list[ServerCandidate]) -> tuple[DynamicMCPClient, list[dict[str, Any]], ServerCandidate]:
    failures: list[str] = []

    for candidate in candidates:
        client: DynamicMCPClient | None = None
        try:
            client, tools = await connect_and_list_tools(candidate)
            return client, tools, candidate
        except Exception as exc:
            failures.append(f"{candidate.command} {' '.join(candidate.args)} -> {exc}")
            if client is not None:
                with contextlib.suppress(Exception):
                    await client.close()

    raise RuntimeError("No server candidate could be started:\n" + "\n".join(failures))


def print_tool_inventory(server_name: str, tools: list[dict[str, Any]], candidate: ServerCandidate) -> None:
    tool_names = ", ".join(sorted(tool["tool_name"] for tool in tools))
    print(f"[{server_name}] command: {candidate.command} {' '.join(candidate.args)}")
    print(f"[{server_name}] tools ({len(tools)}): {tool_names}")


def print_case_result(case: BenchmarkCase, result: RouteResult, latency_ms: float, passed: bool) -> None:
    status = "PASS" if passed else "FAIL"
    print(
        f"[{status}] {case.query}\n"
        f"  expected: {case.expected_server}/{case.expected_tool}\n"
        f"  actual:   {result.server_name}/{result.tool_name}\n"
        f"  score:    {result.score:.4f}\n"
        f"  latency:  {latency_ms:.2f} ms"
    )


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = (len(ordered) - 1) * p
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    weight = rank - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


async def run_benchmark(model_name: str) -> None:
    workspace_root = Path(__file__).resolve().parent
    temp_dir = Path(tempfile.mkdtemp(prefix="scalable-router-"))
    memory_path = temp_dir / "memory.jsonl"
    sqlite_path = temp_dir / "benchmark.db"
    seed_sqlite_database(sqlite_path)

    memory_candidates = [
        ServerCandidate(
            server_name="memory",
            command="npx",
            args=("-y", "@modelcontextprotocol/server-memory"),
            env={"MEMORY_FILE_PATH": str(memory_path)},
        )
    ]
    sqlite_candidates = [
        ServerCandidate(
            server_name="sqlite",
            command="npx",
            args=("-y", "@modelcontextprotocol/server-sqlite"),
        ),
        ServerCandidate(
            server_name="sqlite",
            command="npx",
            args=("-y", "mcp-server-sqlite", "--db", str(sqlite_path)),
        ),
    ]

    clients: list[DynamicMCPClient] = []
    try:
        memory_client, memory_tools, memory_candidate = await connect_first_available(memory_candidates)
        clients.append(memory_client)
        print_tool_inventory("memory", memory_tools, memory_candidate)

        sqlite_client, sqlite_tools, sqlite_candidate = await connect_first_available(sqlite_candidates)
        clients.append(sqlite_client)
        print_tool_inventory("sqlite", sqlite_tools, sqlite_candidate)

        router = UniversalMCPRouter(model_name=model_name)
        router.ingest_server("memory", memory_tools)
        router.ingest_server("sqlite", sqlite_tools)

        cases = build_benchmark_cases(memory_tools, sqlite_tools)
        latencies_ms: list[float] = []
        failures: list[str] = []

        print("\n=== Zero-Shot Multi-Server Routing Benchmark ===")
        for case in cases:
            t0 = time.perf_counter()
            result = router.route(case.query)
            latency_ms = (time.perf_counter() - t0) * 1000.0
            latencies_ms.append(latency_ms)

            passed = result.server_name == case.expected_server and result.tool_name == case.expected_tool
            print_case_result(case, result, latency_ms, passed)
            if not passed:
                failures.append(
                    f"{case.query} -> expected {case.expected_server}/{case.expected_tool}, got {result.server_name}/{result.tool_name}"
                )

        passed_count = len(cases) - len(failures)
        accuracy = (passed_count / len(cases)) * 100.0
        avg_latency = statistics.fmean(latencies_ms)
        median_latency = statistics.median(latencies_ms)
        p95_latency = percentile(latencies_ms, 0.95)

        print("\n=== Strict Report ===")
        print(f"Queries:        {len(cases)}")
        print(f"Correct:        {passed_count}")
        print(f"Accuracy:       {accuracy:.2f}%")
        print(f"Avg latency:    {avg_latency:.2f} ms")
        print(f"Median latency: {median_latency:.2f} ms")
        print(f"P95 latency:    {p95_latency:.2f} ms")

        if failures:
            raise AssertionError("Benchmark assertions failed:\n" + "\n".join(failures))
    finally:
        for client in reversed(clients):
            with contextlib.suppress(Exception):
                await client.close()
        with contextlib.suppress(Exception):
            for path in temp_dir.iterdir():
                path.unlink()
            temp_dir.rmdir()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify multi-server dynamic MCP routing with live stdio servers.")
    parser.add_argument(
        "--model-name",
        default="sentence-transformers/all-mpnet-base-v2",
        help="SentenceTransformer model name or local path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    asyncio.run(run_benchmark(model_name=args.model_name))


if __name__ == "__main__":
    main()
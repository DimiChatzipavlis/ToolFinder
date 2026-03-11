from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import tempfile
from pathlib import Path

if __package__ in {None, ""}:
    from autonomous_agent import AutonomousMCPAgent, extract_text_from_tool_result
    from mcp_adapter import ServerProcessConfig
else:
    from .autonomous_agent import AutonomousMCPAgent, extract_text_from_tool_result
    from .mcp_adapter import ServerProcessConfig


async def run_verification(model_name: str, ollama_model: str) -> None:
    temp_dir = Path(tempfile.mkdtemp(prefix="scalable-router-exec-"))
    memory_path = temp_dir / "memory.jsonl"
    memory_config = ServerProcessConfig(
        server_name="memory",
        command="npx",
        args=("-y", "@modelcontextprotocol/server-memory"),
        env={"MEMORY_FILE_PATH": str(memory_path)},
    )

    async with AutonomousMCPAgent(model_name=model_name, ollama_model=ollama_model) as agent:
        try:
            tools = await agent.add_server(memory_config)
            print(f"[memory] ingested {len(tools)} tools")

            write_result = await agent.execute_task(
                "Create a memory note that the user's favorite language is Rust."
            )
            print(
                json.dumps(
                    {
                        "phase": "write",
                        "server_name": write_result.server_name,
                        "tool_name": write_result.tool_name,
                        "arguments": write_result.arguments,
                        "result": write_result.result,
                    },
                    indent=2,
                )
            )

            await asyncio.sleep(0.5)

            read_result = await agent.execute_task("What is the user's favorite language?")
            final_text = extract_text_from_tool_result(read_result.result)
            if "Rust" not in final_text:
                raise AssertionError(f"Expected Rust in retrieval output, got: {final_text}")

            print(
                json.dumps(
                    {
                        "phase": "read",
                        "server_name": read_result.server_name,
                        "tool_name": read_result.tool_name,
                        "arguments": read_result.arguments,
                        "result": read_result.result,
                        "extracted_text": final_text,
                    },
                    indent=2,
                )
            )
        finally:
            with contextlib.suppress(Exception):
                for path in temp_dir.iterdir():
                    path.unlink()
                temp_dir.rmdir()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify end-to-end autonomous MCP execution with a live memory server.")
    parser.add_argument(
        "--model-name",
        default="sentence-transformers/all-mpnet-base-v2",
        help="SentenceTransformer model name or local path.",
    )
    parser.add_argument(
        "--ollama-model",
        default="llama3.2",
        help="Local Ollama model name to use for argument generation.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    asyncio.run(run_verification(model_name=args.model_name, ollama_model=args.ollama_model))


if __name__ == "__main__":
    main()
from __future__ import annotations

import asyncio
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage
from langchain_core.tools import BaseTool, StructuredTool
from langchain_ollama import ChatOllama
from pydantic import Field, create_model

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from toolfinder.mcp_adapter import DynamicMCPClient  # noqa: E402
from toolfinder.dynamic_faiss_router import RouteResult, UniversalMCPRouter  # noqa: E402


MODEL_NAME = "llama3.2"
SANDBOX_DIR = (REPO_ROOT / "examples" / "langgraph_integration" / "sandbox").resolve()
MCP_CMD = f"npx -y @modelcontextprotocol/server-filesystem {SANDBOX_DIR}"
TASK_QUERY = (
    "Read the contents of 'input.txt', summarize it, and write the summary to "
    "a new file named 'output.txt'."
)


def schema_type_to_annotation(schema: dict[str, Any]) -> Any:
    schema_type = schema.get("type")
    if schema_type == "string":
        return str
    if schema_type == "integer":
        return int
    if schema_type == "number":
        return float
    if schema_type == "boolean":
        return bool
    if schema_type == "array":
        return list[Any]
    if schema_type == "object":
        return dict[str, Any]
    return Any


def sanitize_identifier(value: str) -> str:
    sanitized = re.sub(r"\W+", "_", value).strip("_")
    return sanitized or "DynamicSchema"


def build_args_schema(server_name: str, tool_name: str, input_schema: dict[str, Any]) -> type:
    properties = input_schema.get("properties", {}) if isinstance(input_schema, dict) else {}
    required = set(input_schema.get("required", [])) if isinstance(input_schema, dict) else set()
    fields: dict[str, tuple[Any, Any]] = {}

    for property_name, property_schema in properties.items():
        property_definition = property_schema if isinstance(property_schema, dict) else {}
        annotation = schema_type_to_annotation(property_definition)
        description = property_definition.get("description", "")
        if property_name in required:
            fields[property_name] = (annotation, Field(description=description))
        else:
            fields[property_name] = (annotation, Field(default=None, description=description))

    model_name = f"{sanitize_identifier(server_name)}_{sanitize_identifier(tool_name)}_Args"
    return create_model(model_name, **fields)


def build_langchain_tool(tool_schema: dict[str, Any], client: DynamicMCPClient) -> StructuredTool:
    args_schema = build_args_schema(
        str(tool_schema["server_name"]),
        str(tool_schema["tool_name"]),
        tool_schema.get("inputSchema", {}),
    )

    async def invoke_tool(**kwargs: Any) -> dict[str, Any]:
        clean_kwargs = {key: value for key, value in kwargs.items() if value is not None}
        return await client.call_tool(str(tool_schema["tool_name"]), clean_kwargs)

    description = tool_schema.get("description") or (
        f"{tool_schema['server_name']}/{tool_schema['tool_name']}"
    )
    return StructuredTool.from_function(
        coroutine=invoke_tool,
        name=str(tool_schema["tool_name"]),
        description=str(description),
        args_schema=args_schema,
    )


def route_result_to_schema(result: RouteResult) -> dict[str, Any]:
    return {
        "server_name": result.server_name,
        "tool_name": result.tool_name,
        "description": result.schema.get("description", ""),
        "inputSchema": result.schema.get("inputSchema", {}),
        "score": round(result.score, 6),
    }


def make_cli_table(naive_metrics: dict[str, Any], tf_metrics: dict[str, Any]) -> str:
    lines = [
        "==================================================",
        "FINAL BENCHMARK TELEMETRY REPORT",
        "==================================================",
        f"{'Metric':<25} | {'Naive Baseline':<20} | {'ToolFinder Enabled':<20}",
        "-" * 74,
    ]
    for key in naive_metrics:
        if key == "Architecture":
            continue
        lines.append(
            f"{key:<25} | {str(naive_metrics[key]):<20} | {str(tf_metrics[key]):<20}"
        )
    lines.append("=" * 50)
    return "\n".join(lines)


def make_markdown_table(naive_metrics: dict[str, Any], tf_metrics: dict[str, Any]) -> str:
    rows = [
        "| Metric | Naive Baseline | ToolFinder Enabled |",
        "| --- | --- | --- |",
    ]
    for key in naive_metrics:
        if key == "Architecture":
            continue
        rows.append(f"| {key} | {naive_metrics[key]} | {tf_metrics[key]} |")
    return "\n".join(rows)


def measure_execution(
    name: str,
    tool_binding_func: Any,
    llm: ChatOllama,
    query: str,
) -> dict[str, Any]:
    print(f"\n--- [STARTING RUN: {name}] ---")
    start_time = time.time()

    tools, context_size = tool_binding_func(query)
    llm_with_tools = llm.bind_tools(tools)

    print(
        f"[SYSTEM] Prompting {MODEL_NAME} with {len(tools)} tools "
        f"(Context size: {context_size} chars)..."
    )
    inference_start = time.time()

    try:
        response = llm_with_tools.invoke([HumanMessage(content=query)])
        inference_time = time.time() - inference_start
        success = hasattr(response, "tool_calls") and len(response.tool_calls) > 0
        tool_called = response.tool_calls[0]["name"] if success else "None (Hallucination/Text)"
    except Exception as exc:
        inference_time = time.time() - inference_start
        success = False
        tool_called = f"Error: {exc}"

    total_time = time.time() - start_time

    return {
        "Architecture": name,
        "Tools In Context": len(tools),
        "Context Payload (Chars)": context_size,
        "Total Latency (s)": round(total_time, 2),
        "Inference Latency (s)": round(inference_time, 2),
        "Successful Tool Call": success,
        "Action Attempted": tool_called,
    }


async def run_benchmark() -> None:
    print("==================================================")
    print("TOOLFINDER vs. NAIVE SLM BENCHMARK")
    print(f"Task: {TASK_QUERY}")
    print("==================================================")

    print("[SYSTEM] Booting Filesystem MCP Server...")
    client = DynamicMCPClient(
        server_name="filesystem",
        command="npx",
        args=["-y", "@modelcontextprotocol/server-filesystem", str(SANDBOX_DIR)],
        cwd=str(REPO_ROOT),
        startup_timeout_s=90.0,
        request_timeout_s=45.0,
    )
    all_tools = await client.initialize_and_get_tools()

    print("[SYSTEM] Indexing tools in ToolFinder (FAISS)...")
    router = UniversalMCPRouter(model_name="sentence-transformers/all-mpnet-base-v2")
    router.ingest_server(client.server_name, all_tools)

    all_langchain_tools = [build_langchain_tool(tool_schema, client) for tool_schema in all_tools]
    llm = ChatOllama(model=MODEL_NAME, temperature=0)

    def bind_naive(query: str) -> tuple[list[BaseTool], int]:
        del query
        context_size = len(json.dumps(all_tools, ensure_ascii=True, sort_keys=True))
        return all_langchain_tools, context_size

    def bind_toolfinder(query: str) -> tuple[list[BaseTool], int]:
        route_start = time.time()
        top_k = router.route_top_k(query, k=2)
        route_time = time.time() - route_start
        print(f"[ROUTER] FAISS Search completed in {route_time * 1000:.2f} ms")
        top_k_payload = [route_result_to_schema(result) for result in top_k]
        context_size = len(json.dumps(top_k_payload, ensure_ascii=True, sort_keys=True))
        tools = [build_langchain_tool(schema, client) for schema in top_k_payload]
        return tools, context_size

    naive_metrics = measure_execution("Naive (Context Stuffing)", bind_naive, llm, TASK_QUERY)
    tf_metrics = measure_execution("ToolFinder (Semantic Routing)", bind_toolfinder, llm, TASK_QUERY)

    cli_table = make_cli_table(naive_metrics, tf_metrics)
    markdown_table = make_markdown_table(naive_metrics, tf_metrics)

    print(f"\n{cli_table}")
    print("\nMarkdown Table:\n")
    print(markdown_table)

    await client.close()


if __name__ == "__main__":
    asyncio.run(run_benchmark())
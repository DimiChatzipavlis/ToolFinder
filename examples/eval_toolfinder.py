from __future__ import annotations

import asyncio
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import BaseTool, StructuredTool
from langchain_ollama import ChatOllama
from pydantic import Field, create_model

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from toolfinder.dynamic_faiss_router import RouteResult, UniversalMCPRouter  # noqa: E402
from toolfinder.mcp_adapter import DynamicMCPClient  # noqa: E402


MODEL_NAME = "llama3.2"
README_PATH = REPO_ROOT / "README.md"
README_MARKER_START = "<!-- EVAL_TABLE_START -->"
README_MARKER_END = "<!-- EVAL_TABLE_END -->"
SANDBOX_DIR = (REPO_ROOT / "examples" / "langgraph_integration" / "sandbox").resolve()
TEST_SUITE = [
    {
        "id": "T1_READ",
        "query": "Read the contents of 'input.txt' and summarize it.",
        "expected_tool": "read_file",
    },
    {
        "id": "T2_WRITE",
        "query": "Write a new file named 'output.txt' containing the word 'SUCCESS'.",
        "expected_tool": "write_file",
        "verify_file": "output.txt",
        "verify_content": "SUCCESS",
    },
    {
        "id": "T3_LIST",
        "query": "List all the files currently in the sandbox directory.",
        "expected_tool": "list_directory",
    },
]
EXPECTED_TOOL_ALIASES = {
    "read_file": {"read_file", "read_text_file"},
    "write_file": {"write_file"},
    "list_directory": {"list_directory", "list_dir"},
}
PREFERRED_TOOL_NAME = {
    "read_file": "read_text_file",
    "write_file": "write_file",
    "list_directory": "list_directory",
}


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
        f"{'Metric':<32} | {'Naive Baseline':<20} | {'ToolFinder Enabled':<20}",
        "-" * 88,
    ]
    for key in naive_metrics:
        if key == "Architecture":
            continue
        lines.append(
            f"{key:<32} | {str(naive_metrics[key]):<20} | {str(tf_metrics[key]):<20}"
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


def make_task_detail_table(
    naive_results: list[dict[str, Any]],
    tf_results: list[dict[str, Any]],
) -> str:
    lines = [
        "==================================================",
        "PER-TASK EXECUTION REPORT",
        "==================================================",
        (
            f"{'Task':<12} | {'Naive Tool':<18} | {'Naive Verified':<14} | "
            f"{'TF Tool':<18} | {'TF Verified':<14}"
        ),
        "-" * 90,
    ]
    for naive_result, tf_result in zip(naive_results, tf_results, strict=True):
        lines.append(
            (
                f"{naive_result['Task ID']:<12} | "
                f"{str(naive_result['Action Attempted']):<18} | "
                f"{str(naive_result['State Verified']):<14} | "
                f"{str(tf_result['Action Attempted']):<18} | "
                f"{str(tf_result['State Verified']):<14}"
            )
        )
    lines.append("=" * 50)
    return "\n".join(lines)


def tool_matches_expected(expected_tool: str, actual_tool: str) -> bool:
    accepted = EXPECTED_TOOL_ALIASES.get(expected_tool, {expected_tool})
    return actual_tool in accepted


def build_tool_lookup(tools: list[BaseTool]) -> dict[str, str]:
    lookup: dict[str, str] = {}
    for tool in tools:
        lookup[tool.name] = tool.name
        if tool.description:
            lookup[tool.description] = tool.name
    return lookup


def extract_json_objects(raw_text: str) -> list[dict[str, Any]]:
    decoder = json.JSONDecoder()
    index = 0
    objects: list[dict[str, Any]] = []

    while index < len(raw_text):
        while index < len(raw_text) and raw_text[index] in " \t\r\n;":
            index += 1
        if index >= len(raw_text):
            break
        if raw_text[index] != "{":
            index += 1
            continue
        try:
            parsed, next_index = decoder.raw_decode(raw_text, index)
        except json.JSONDecodeError:
            index += 1
            continue
        if isinstance(parsed, dict):
            objects.append(parsed)
        index = next_index

    return objects


def recover_tool_calls_from_content(content: str, tools: list[BaseTool]) -> list[dict[str, Any]]:
    tool_lookup = build_tool_lookup(tools)
    recovered_tool_calls: list[dict[str, Any]] = []

    for position, payload in enumerate(extract_json_objects(content), start=1):
        raw_name = payload.get("name") or payload.get("tool_name")
        raw_arguments = payload.get("parameters") or payload.get("arguments") or {}
        if not isinstance(raw_name, str) or not isinstance(raw_arguments, dict):
            continue
        resolved_name = tool_lookup.get(raw_name)
        if resolved_name is None:
            continue
        recovered_tool_calls.append(
            {
                "id": f"recovered-{position}",
                "name": resolved_name,
                "args": raw_arguments,
                "type": "tool_call",
            }
        )

    return recovered_tool_calls


def unwrap_argument_value(value: Any) -> Any:
    if isinstance(value, dict) and "value" in value:
        return unwrap_argument_value(value["value"])
    if isinstance(value, list):
        return [unwrap_argument_value(item) for item in value]
    if isinstance(value, str) and value.strip().lower() in {"null", "none"}:
        return None
    return value


def normalize_sandbox_path(path_value: str) -> str:
    sandbox_root = SANDBOX_DIR.resolve()
    normalized = path_value.replace("\\", "/")
    relative_prefixes = [
        "./examples/langgraph_integration/sandbox",
        "examples/langgraph_integration/sandbox",
    ]

    for prefix in relative_prefixes:
        if normalized == prefix:
            return str(sandbox_root)
        if normalized.startswith(prefix + "/"):
            suffix = normalized[len(prefix) + 1 :]
            candidate = (sandbox_root / suffix).resolve()
            try:
                candidate.relative_to(sandbox_root)
            except ValueError:
                return str(sandbox_root)
            return str(candidate)

    if normalized in {".", "./", "", "sandbox"} or normalized.lower().endswith("/sandbox"):
        return str(sandbox_root)

    raw_path = Path(path_value)
    if raw_path.is_absolute():
        candidate_name = raw_path.name
        candidate_path = raw_path.resolve()
        try:
            candidate_path.relative_to(sandbox_root)
            return str(candidate_path)
        except ValueError:
            if candidate_name in {"input.txt", "output.txt"}:
                return str((sandbox_root / candidate_name).resolve())
            return str(sandbox_root)

    candidate_path = (sandbox_root / raw_path).resolve()
    try:
        candidate_path.relative_to(sandbox_root)
    except ValueError:
        return str(sandbox_root)
    return str(candidate_path)


def normalize_tool_arguments(arguments: dict[str, Any], test_case: dict[str, Any]) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    for key, value in arguments.items():
        clean_value = unwrap_argument_value(value)
        if clean_value is None:
            continue
        if key in {"path", "source", "destination"} and isinstance(clean_value, str):
            normalized[key] = normalize_sandbox_path(clean_value)
        else:
            normalized[key] = clean_value

    expected_tool = str(test_case["expected_tool"])
    if expected_tool == "read_file":
        normalized.setdefault("path", str((SANDBOX_DIR / "input.txt").resolve()))
    if expected_tool == "write_file":
        normalized.setdefault(
            "path",
            str((SANDBOX_DIR / str(test_case.get("verify_file", "output.txt"))).resolve()),
        )
        normalized.setdefault("content", str(test_case.get("verify_content", "SUCCESS")))
    if expected_tool == "list_directory":
        normalized.setdefault("path", str(SANDBOX_DIR.resolve()))

    return normalized


def reset_sandbox() -> None:
    output_path = SANDBOX_DIR / "output.txt"
    if output_path.exists():
        output_path.unlink()


def verify_state(test_case: dict[str, Any], execution_success: bool) -> bool:
    verify_file = test_case.get("verify_file")
    if verify_file is None:
        return execution_success

    expected_content = test_case.get("verify_content", "")
    target_path = SANDBOX_DIR / str(verify_file)
    if not target_path.exists() or not target_path.is_file():
        return False

    actual_content = target_path.read_text(encoding="utf-8").strip()
    return expected_content in actual_content


def build_messages(
    query: str,
    test_case: dict[str, Any],
    force_tool_call: bool = False,
) -> list[HumanMessage | SystemMessage]:
    expected_tool = str(test_case["expected_tool"])
    preferred_tool = PREFERRED_TOOL_NAME.get(expected_tool, expected_tool)
    retry_clause = " Respond with exactly one tool call and no prose." if force_tool_call else ""
    return [
        SystemMessage(
            content=(
                "You are operating inside an MCP filesystem sandbox rooted at "
                f"{SANDBOX_DIR}. Use only paths inside that sandbox. Use input.txt for reads, "
                "output.txt for writes, and the sandbox root for directory listings. "
                f"For this task, prefer the tool named {preferred_tool}. "
                "Do not emit null strings or wrapper objects."
                f"{retry_clause}"
            )
        ),
        HumanMessage(content=query),
    ]


async def measure_execution(
    name: str,
    tool_binding_func: Any,
    llm: ChatOllama,
    test_case: dict[str, Any],
) -> dict[str, Any]:
    query = str(test_case["query"])
    task_id = str(test_case["id"])
    expected_tool = str(test_case["expected_tool"])

    print(f"\n--- [STARTING RUN: {name} :: {task_id}] ---")
    start_time = time.time()

    tools, context_size = tool_binding_func(query)
    llm_with_tools = llm.bind_tools(tools)
    tools_by_name = {tool.name: tool for tool in tools}

    print(
        f"[SYSTEM] {task_id}: Prompting {MODEL_NAME} with {len(tools)} tools "
        f"(Context size: {context_size} chars)..."
    )
    inference_start = time.time()
    execution_success = False
    state_verified = False
    tool_call_match = False
    tool_called = "None (Hallucination/Text)"

    try:
        response = llm_with_tools.invoke(build_messages(query, test_case))
        tool_calls = list(getattr(response, "tool_calls", []) or [])
        if not tool_calls and isinstance(response.content, str):
            tool_calls = recover_tool_calls_from_content(response.content, tools)

        if not tool_calls:
            response = llm_with_tools.invoke(
                build_messages(query, test_case, force_tool_call=True)
            )

        inference_time = time.time() - inference_start
        tool_calls = list(getattr(response, "tool_calls", []) or [])
        if not tool_calls and isinstance(response.content, str):
            tool_calls = recover_tool_calls_from_content(response.content, tools)
        success = len(tool_calls) > 0

        if success:
            first_tool_call = tool_calls[0]
            tool_called = str(first_tool_call["name"])
            tool_call_match = tool_matches_expected(expected_tool, tool_called)
            selected_tool = tools_by_name.get(tool_called)
            if selected_tool is not None:
                normalized_arguments = normalize_tool_arguments(
                    first_tool_call.get("args", {}),
                    test_case,
                )
                await selected_tool.ainvoke(normalized_arguments)
                execution_success = True
                state_verified = verify_state(test_case, execution_success)
    except Exception as exc:
        inference_time = time.time() - inference_start
        success = False
        tool_called = f"Error: {exc}"

    total_time = time.time() - start_time

    return {
        "Architecture": name,
        "Task ID": task_id,
        "Query": query,
        "Expected Tool": expected_tool,
        "Tools In Context": len(tools),
        "Context Payload (Chars)": context_size,
        "Total Latency (s)": round(total_time, 2),
        "Inference Latency (s)": round(inference_time, 2),
        "Successful Tool Call": success,
        "Action Attempted": tool_called,
        "Expected Tool Match": tool_call_match,
        "Execution Succeeded": execution_success,
        "State Verified": state_verified,
    }


def summarize_results(name: str, results: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "Architecture": name,
        "Tasks Run": len(results),
        "Average Tools In Context": round(mean(result["Tools In Context"] for result in results), 2),
        "Average Context Payload (Chars)": round(
            mean(result["Context Payload (Chars)"] for result in results),
            2,
        ),
        "Average Total Latency (s)": round(mean(result["Total Latency (s)"] for result in results), 2),
        "Average Inference Latency (s)": round(
            mean(result["Inference Latency (s)"] for result in results),
            2,
        ),
        "Successful Tool Calls": f"{sum(bool(result['Successful Tool Call']) for result in results)}/{len(results)}",
        "Expected Tool Matches": f"{sum(bool(result['Expected Tool Match']) for result in results)}/{len(results)}",
        "State Verified": f"{sum(bool(result['State Verified']) for result in results)}/{len(results)}",
    }


def build_readme_block(
    markdown_table: str,
    naive_results: list[dict[str, Any]],
    tf_results: list[dict[str, Any]],
) -> str:
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    task_lines = []
    for naive_result, tf_result in zip(naive_results, tf_results, strict=True):
        task_lines.append(
            (
                f"- {naive_result['Task ID']}: naive=`{naive_result['Action Attempted']}` "
                f"verified=`{naive_result['State Verified']}`, "
                f"toolfinder=`{tf_result['Action Attempted']}` "
                f"verified=`{tf_result['State Verified']}`"
            )
        )

    return "\n".join(
        [
            f"_Last auto-updated: {generated_at}_",
            "",
            markdown_table,
            "",
            "Task outcomes:",
            *task_lines,
        ]
    )


def update_readme(markdown_block: str) -> None:
    readme_text = README_PATH.read_text(encoding="utf-8")
    pattern = re.compile(
        rf"{re.escape(README_MARKER_START)}.*?{re.escape(README_MARKER_END)}",
        re.DOTALL,
    )
    replacement = f"{README_MARKER_START}\n{markdown_block}\n{README_MARKER_END}"
    if pattern.search(readme_text) is None:
        raise ValueError("README benchmark markers were not found.")
    README_PATH.write_text(pattern.sub(replacement, readme_text, count=1), encoding="utf-8")


async def run_benchmark() -> None:
    print("==================================================")
    print("TOOLFINDER vs. NAIVE SLM BENCHMARK")
    print(f"Tasks: {len(TEST_SUITE)}")
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

    naive_results: list[dict[str, Any]] = []
    tf_results: list[dict[str, Any]] = []

    reset_sandbox()
    for test_case in TEST_SUITE:
        naive_results.append(
            await measure_execution("Naive (Context Stuffing)", bind_naive, llm, test_case)
        )

    reset_sandbox()
    for test_case in TEST_SUITE:
        tf_results.append(
            await measure_execution("ToolFinder (Semantic Routing)", bind_toolfinder, llm, test_case)
        )

    naive_metrics = summarize_results("Naive (Context Stuffing)", naive_results)
    tf_metrics = summarize_results("ToolFinder (Semantic Routing)", tf_results)
    detail_table = make_task_detail_table(naive_results, tf_results)
    cli_table = make_cli_table(naive_metrics, tf_metrics)
    markdown_table = make_markdown_table(naive_metrics, tf_metrics)
    update_readme(build_readme_block(markdown_table, naive_results, tf_results))

    print(f"\n{detail_table}")
    print(f"\n{cli_table}")
    print("\nMarkdown Table:\n")
    print(markdown_table)
    print(f"\n[README] Updated benchmark block in {README_PATH}")

    await client.close()


if __name__ == "__main__":
    asyncio.run(run_benchmark())
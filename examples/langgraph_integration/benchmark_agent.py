from __future__ import annotations

import asyncio
import json
import re
import sys
import time
from pathlib import Path
from typing import Annotated, Any, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool, StructuredTool
from langchain_ollama import ChatOllama
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from pydantic import Field, create_model

INTEGRATION_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from toolfinder import DynamicMCPClient, UniversalMCPRouter  # noqa: E402
from toolfinder.mcp_adapter import MCPClientError  # noqa: E402


USER_QUERY = (
    "List the files in the sandbox directory. Then write a new file named "
    "'hello.txt' containing the word 'success'."
)

ToolLookup = dict[str, str]


class GraphState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    active_tools: list[BaseTool]
    telemetry: dict[str, Any]


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


def build_langchain_tool(candidate: Any, client: DynamicMCPClient) -> StructuredTool:
    args_schema = build_args_schema(
        candidate.server_name,
        candidate.tool_name,
        candidate.schema.get("inputSchema", {}),
    )

    async def invoke_tool(**kwargs: Any) -> dict[str, Any]:
        clean_kwargs = {key: value for key, value in kwargs.items() if value is not None}
        return await client.call_tool(candidate.tool_name, clean_kwargs)

    description = (
        candidate.schema.get("description")
        or f"{candidate.server_name}/{candidate.tool_name}"
    )
    return StructuredTool.from_function(
        coroutine=invoke_tool,
        name=candidate.tool_name,
        description=description,
        args_schema=args_schema,
    )


def init_telemetry() -> dict[str, Any]:
    return {
        "routing_latencies_ms": [],
        "llm_latencies_ms": [],
        "context_window_saved_chars": [],
        "prompt_payload_chars": [],
        "hallucination_events": [],
        "routed_tools": [],
    }


def append_telemetry(telemetry: dict[str, Any], **values: Any) -> dict[str, Any]:
    updated = {
        "routing_latencies_ms": list(telemetry.get("routing_latencies_ms", [])),
        "llm_latencies_ms": list(telemetry.get("llm_latencies_ms", [])),
        "context_window_saved_chars": list(telemetry.get("context_window_saved_chars", [])),
        "prompt_payload_chars": list(telemetry.get("prompt_payload_chars", [])),
        "hallucination_events": list(telemetry.get("hallucination_events", [])),
        "routed_tools": list(telemetry.get("routed_tools", [])),
    }
    for key, value in values.items():
        updated.setdefault(key, [])
        updated[key].append(value)
    return updated


def safe_json_size(payload: Any) -> int:
    return len(json.dumps(payload, ensure_ascii=True, default=str, sort_keys=True))


def extract_last_user_message(messages: list[BaseMessage]) -> str:
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            if isinstance(message.content, str):
                return message.content
            return json.dumps(message.content, ensure_ascii=True, default=str)
    raise ValueError("Graph state does not contain a human message.")


def estimate_prompt_payload_chars(messages: list[BaseMessage]) -> int:
    serialized_messages = []
    for message in messages:
        serialized_messages.append(
            {
                "type": message.type,
                "content": message.content,
                "tool_calls": getattr(message, "tool_calls", None),
            }
        )
    return safe_json_size(serialized_messages)


def build_tool_lookup(active_tools: list[BaseTool]) -> ToolLookup:
    tool_lookup: ToolLookup = {}
    for tool in active_tools:
        tool_lookup[tool.name] = tool.name
        if tool.description:
            tool_lookup[tool.description] = tool.name
    return tool_lookup


def resolve_tool_name(raw_name: str, tool_lookup: ToolLookup) -> str | None:
    return tool_lookup.get(raw_name)


def normalize_sandbox_path(path_value: str) -> str:
    sandbox_root = (INTEGRATION_DIR / "sandbox").resolve()
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

    if normalized in {".", "./", ""}:
        return str(sandbox_root)

    try:
        raw_path = Path(path_value)
    except (OSError, RuntimeError):
        return path_value

    if raw_path.is_absolute():
        candidate_path = raw_path.resolve()
        try:
            candidate_path.relative_to(sandbox_root)
        except ValueError:
            return path_value
        return str(candidate_path)

    candidate_path = (sandbox_root / raw_path).resolve()
    try:
        candidate_path.relative_to(sandbox_root)
    except ValueError:
        return str(sandbox_root)
    return str(candidate_path)


def normalize_tool_arguments(arguments: dict[str, Any]) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    for key, value in arguments.items():
        if key in {"path", "source", "destination"} and isinstance(value, str):
            normalized[key] = normalize_sandbox_path(value)
        else:
            normalized[key] = value
    return normalized


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


def recover_tool_calls_from_content(
    content: str,
    active_tools: list[BaseTool],
) -> list[dict[str, Any]]:
    recovered_tool_calls: list[dict[str, Any]] = []
    tool_lookup = build_tool_lookup(active_tools)
    for position, payload in enumerate(extract_json_objects(content), start=1):
        raw_name = payload.get("name") or payload.get("tool_name")
        raw_arguments = payload.get("parameters") or payload.get("arguments") or {}
        if not isinstance(raw_name, str) or not isinstance(raw_arguments, dict):
            continue
        resolved_name = resolve_tool_name(raw_name, tool_lookup)
        if resolved_name is None:
            continue
        recovered_tool_calls.append(
            {
                "name": resolved_name,
                "args": normalize_tool_arguments(raw_arguments),
                "id": f"recovered_tool_call_{position}",
                "type": "tool_call",
            }
        )
    return recovered_tool_calls


def should_execute_tools(state: GraphState) -> str:
    last_message = state["messages"][-1]
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "tools"
    return END


async def main() -> None:
    sandbox_dir = INTEGRATION_DIR / "sandbox"
    sandbox_dir.mkdir(parents=True, exist_ok=True)

    client = DynamicMCPClient(
        server_name="filesystem",
        command="npx",
        args=["-y", "@modelcontextprotocol/server-filesystem", "./examples/langgraph_integration/sandbox"],
        cwd=str(REPO_ROOT),
        startup_timeout_s=90.0,
        request_timeout_s=45.0,
    )

    try:
        all_tools = await client.initialize_and_get_tools()
        router = UniversalMCPRouter(model_name="sentence-transformers/all-mpnet-base-v2")
        router.ingest_server(client.server_name, all_tools)
        all_schema_chars = safe_json_size(all_tools)

        async def semantic_router_node(state: GraphState) -> GraphState:
            query = extract_last_user_message(state["messages"])
            started = time.perf_counter()
            candidates = router.route_top_k(query, k=2)
            routing_latency_ms = (time.perf_counter() - started) * 1000.0
            active_tools = [build_langchain_tool(candidate, client) for candidate in candidates]
            top_k_payload = [
                {
                    "server_name": candidate.server_name,
                    "tool_name": candidate.tool_name,
                    "description": candidate.schema.get("description", ""),
                    "inputSchema": candidate.schema.get("inputSchema", {}),
                    "score": round(candidate.score, 6),
                }
                for candidate in candidates
            ]
            routed_tool_names = [tool.name for tool in active_tools]
            top_k_chars = safe_json_size(top_k_payload)
            context_saved = all_schema_chars - top_k_chars
            telemetry = append_telemetry(
                state["telemetry"],
                routing_latencies_ms=round(routing_latency_ms, 2),
                context_window_saved_chars=context_saved,
                routed_tools=routed_tool_names,
            )
            print(
                f"[ROUTER] Query='{query}' | Top-2={routed_tool_names} | "
                f"Context Window Saved={context_saved} chars | "
                f"FAISS Latency={routing_latency_ms:.2f} ms"
            )
            return {"active_tools": active_tools, "telemetry": telemetry}

        async def llm_reasoning_node(state: GraphState) -> GraphState:
            llm = ChatOllama(model="llama3.2", temperature=0)
            prompt_chars = estimate_prompt_payload_chars(state["messages"])
            started = time.perf_counter()
            response = await llm.bind_tools(state["active_tools"]).ainvoke(state["messages"])
            llm_latency_ms = (time.perf_counter() - started) * 1000.0
            if not response.tool_calls and isinstance(response.content, str):
                recovered_tool_calls = recover_tool_calls_from_content(
                    response.content,
                    state["active_tools"],
                )
                if recovered_tool_calls:
                    response = AIMessage(content="", tool_calls=recovered_tool_calls)
                    print(
                        f"[LLM] Recovered {len(recovered_tool_calls)} tool call(s) "
                        "from plain JSON output."
                    )
            telemetry = append_telemetry(
                state["telemetry"],
                prompt_payload_chars=prompt_chars,
                llm_latencies_ms=round(llm_latency_ms, 2),
            )
            print(
                f"[LLM] Prompt Payload={prompt_chars} chars | "
                f"Inference Latency={llm_latency_ms:.2f} ms"
            )
            return {"messages": [response], "telemetry": telemetry}

        async def tool_execution_node(state: GraphState) -> GraphState:
            tool_node = ToolNode(state["active_tools"])
            try:
                result = await tool_node.ainvoke({"messages": state["messages"]})
                return {"messages": result["messages"], "telemetry": state["telemetry"]}
            except Exception as exc:
                telemetry = append_telemetry(state["telemetry"], hallucination_events=str(exc))
                print(f"[TOOLS] Hallucination Event: {exc}")
                last_ai_message = next(
                    (
                        message
                        for message in reversed(state["messages"])
                        if isinstance(message, AIMessage)
                    ),
                    None,
                )
                tool_call_id = None
                if last_ai_message is not None and last_ai_message.tool_calls:
                    tool_call_id = last_ai_message.tool_calls[0].get("id")

                if tool_call_id is None:
                    return {
                        "messages": [HumanMessage(content=f"Hallucination Event: {exc}")],
                        "telemetry": telemetry,
                    }

                return {
                    "messages": [
                        ToolMessage(
                            content=f"Hallucination Event: {exc}",
                            tool_call_id=tool_call_id,
                        )
                    ],
                    "telemetry": telemetry,
                }

        workflow = StateGraph(GraphState)
        workflow.add_node("Semantic_Router_Node", semantic_router_node)
        workflow.add_node("LLM_Reasoning_Node", llm_reasoning_node)
        workflow.add_node("Tool_Execution_Node", tool_execution_node)
        workflow.set_entry_point("Semantic_Router_Node")
        workflow.add_edge("Semantic_Router_Node", "LLM_Reasoning_Node")
        workflow.add_conditional_edges(
            "LLM_Reasoning_Node",
            should_execute_tools,
            {
                "tools": "Tool_Execution_Node",
                END: END,
            },
        )
        workflow.add_edge("Tool_Execution_Node", "Semantic_Router_Node")

        graph = workflow.compile()
        final_state = await graph.ainvoke(
            {
                "messages": [
                    SystemMessage(
                        content=(
                            "You are operating inside an MCP filesystem sandbox rooted at "
                            "./examples/langgraph_integration/sandbox. Use only relative paths inside "
                            "that sandbox. Use '.' to list the sandbox contents and 'hello.txt' "
                            "to write the file. "
                            "Do not claim success unless a tool result confirms it."
                        )
                    ),
                    HumanMessage(content=USER_QUERY),
                ],
                "active_tools": [],
                "telemetry": init_telemetry(),
            },
            config={"recursion_limit": 8},
        )

        print("\n=== Final Messages ===")
        for message in final_state["messages"]:
            if isinstance(message.content, str):
                print(f"[{message.type}] {message.content}")
            else:
                print(
                    f"[{message.type}] "
                    f"{json.dumps(message.content, ensure_ascii=True, default=str)}"
                )

        telemetry = final_state["telemetry"]
        print("\n=== Telemetry Report ===")
        print(
            "Routing Latencies (ms):",
            json.dumps(telemetry.get("routing_latencies_ms", []), ensure_ascii=True),
        )
        print(
            "LLM Latencies (ms):",
            json.dumps(telemetry.get("llm_latencies_ms", []), ensure_ascii=True),
        )
        print(
            "Context Window Saved (chars):",
            json.dumps(telemetry.get("context_window_saved_chars", []), ensure_ascii=True),
        )
        print(
            "Prompt Payload Chars:",
            json.dumps(telemetry.get("prompt_payload_chars", []), ensure_ascii=True),
        )
        print(
            "Hallucination Events:",
            json.dumps(telemetry.get("hallucination_events", []), ensure_ascii=True),
        )
        print(
            "Routed Tools:",
            json.dumps(telemetry.get("routed_tools", []), ensure_ascii=True),
        )
    finally:
        try:
            await client.close()
        except MCPClientError:
            pass


if __name__ == "__main__":
    asyncio.run(main())
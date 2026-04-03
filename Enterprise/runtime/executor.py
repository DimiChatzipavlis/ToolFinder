from __future__ import annotations

import json
from typing import Protocol

from jsonschema import validate

from toolfinder.mcp_adapter import DynamicMCPClient

from .contracts import ToolCandidate


JsonPrimitive = str | int | float | bool | None
JsonValue = JsonPrimitive | list["JsonValue"] | dict[str, "JsonValue"]
JsonObject = dict[str, JsonValue]


class ToolClientProtocol(Protocol):
    async def call_tool(self, tool_name: str, arguments: JsonObject) -> JsonObject: ...


class HybridToolExecutor:
    """Executes validated tool calls against registered MCP clients."""

    def __init__(self, clients: dict[str, DynamicMCPClient | ToolClientProtocol]) -> None:
        self.clients = clients

    async def execute(self, candidate: ToolCandidate, arguments: JsonObject) -> tuple[JsonObject, str]:
        client = self.clients.get(candidate.server_name)
        if client is None:
            raise RuntimeError(f"no client registered for server {candidate.server_name}")

        schema = candidate.input_schema if isinstance(candidate.input_schema, dict) else {}
        if schema:
            validate(instance=arguments, schema=schema)

        raw_result = await client.call_tool(candidate.tool_name, arguments)
        if not isinstance(raw_result, dict):
            raise RuntimeError("tool result must be a JSON object")

        observation = self._extract_text(raw_result)
        if not observation:
            observation = json.dumps(raw_result, ensure_ascii=True, sort_keys=True)

        return raw_result, observation

    def _extract_text(self, payload: object) -> str:
        fragments: list[str] = []

        def walk(node: object) -> None:
            if isinstance(node, str):
                fragments.append(node)
                return
            if isinstance(node, dict):
                for value in node.values():
                    walk(value)
                return
            if isinstance(node, list):
                for item in node:
                    walk(item)

        walk(payload)
        return "\n".join(fragment for fragment in fragments if fragment)

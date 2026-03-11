from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import socket
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any

from jsonschema import ValidationError, validate

if __package__ in {None, ""}:
    from dynamic_faiss_router import RouteResult, UniversalMCPRouter
    from mcp_adapter import DynamicMCPClient, MCPClientError, ServerProcessConfig
    from utils import LLMOutputParsingError, extract_and_parse_json
else:
    from .dynamic_faiss_router import RouteResult, UniversalMCPRouter
    from .mcp_adapter import DynamicMCPClient, MCPClientError, ServerProcessConfig
    from .utils import LLMOutputParsingError, extract_and_parse_json


logger = logging.getLogger(__name__)

JsonDict = dict[str, Any]


@dataclass(frozen=True)
class ReActStep:
    iteration: int
    thought: str
    action: str
    server_name: str | None
    tool_name: str | None
    arguments: JsonDict | None
    observation: str
    raw_model_output: str
    routing_latency_ms: float
    available_tools: list[JsonDict]


@dataclass(frozen=True)
class AgentExecutionResult:
    status: str
    answer: str
    steps: list[ReActStep]
    scratchpad: list[JsonDict] = field(default_factory=list)


class Scratchpad:
    def __init__(self, user_query: str) -> None:
        self._entries: list[JsonDict] = [
            {"role": "system", "content": "Structured ReAct scratchpad."},
            {"role": "user", "content": user_query},
        ]

    def add(self, role: str, content: Any, *, iteration: int | None = None) -> None:
        entry: JsonDict = {"role": role, "content": content}
        if iteration is not None:
            entry["iteration"] = iteration
        self._entries.append(entry)

    def render(self) -> str:
        return json.dumps(self._entries, ensure_ascii=True, sort_keys=True)

    def entries(self) -> list[JsonDict]:
        return list(self._entries)

    def recent_observations(self, limit: int = 2) -> list[str]:
        observations: list[str] = []
        for entry in reversed(self._entries):
            if entry.get("role") != "observation":
                continue
            observations.append(str(entry.get("content", "")))
            if len(observations) >= limit:
                break
        observations.reverse()
        return observations

    def last_observation_text(self) -> str:
        for entry in reversed(self._entries):
            if entry.get("role") != "observation":
                continue
            content = entry.get("content")
            if isinstance(content, str):
                return content
            if isinstance(content, dict):
                text = content.get("text")
                if isinstance(text, str):
                    return text
                return json.dumps(content, ensure_ascii=True, sort_keys=True)
            return str(content)
        return ""


def extract_text_from_tool_result(payload: Any) -> str:
    fragments: list[str] = []

    def walk(node: Any) -> None:
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


class AutonomousMCPAgent:
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-mpnet-base-v2",
        ollama_model: str = "llama3.2",
        ollama_url: str = "http://127.0.0.1:11434/api/generate",
        max_iterations: int = 7,
    ) -> None:
        self.router = UniversalMCPRouter(model_name=model_name)
        self.ollama_model = ollama_model
        self.ollama_url = ollama_url
        self.max_iterations = min(max_iterations, 7)
        self.clients: dict[str, DynamicMCPClient] = {}
        self._owned_clients: list[DynamicMCPClient] = []

    async def __aenter__(self) -> AutonomousMCPAgent:
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        await self.close()

    async def add_server(self, config: ServerProcessConfig) -> list[JsonDict]:
        client = DynamicMCPClient(
            server_name=config.server_name,
            command=config.command,
            args=list(config.args),
            env=config.env,
            cwd=config.cwd,
            startup_timeout_s=90.0,
            request_timeout_s=45.0,
        )

        try:
            tools = await self.register_server(config.server_name, client)
        except Exception:
            await client.close()
            raise

        self._owned_clients.append(client)
        return tools

    async def register_server(self, server_name: str, client: DynamicMCPClient) -> list[JsonDict]:
        if server_name != client.server_name:
            raise ValueError(f"server name mismatch: registry={server_name}, client={client.server_name}")
        if server_name in self.clients:
            raise ValueError(f"server already registered: {server_name}")

        tools_list = await client.initialize_and_get_tools()
        self.clients[server_name] = client
        self.router.ingest_server(server_name, tools_list)
        return tools_list

    async def execute_task(self, user_query: str) -> AgentExecutionResult:
        scratchpad = Scratchpad(user_query)
        steps: list[ReActStep] = []
        iteration = 1

        while iteration <= self.max_iterations:
            routing_query = self._build_routing_query(user_query, scratchpad)
            routing_started = time.perf_counter()
            candidates = self.router.route_top_k(routing_query, k=5)
            routing_latency_ms = (time.perf_counter() - routing_started) * 1000.0
            available_tools = self._serialize_candidates(candidates)
            logger.info(
                "Iteration %s routed tools in %.2f ms: %s",
                iteration,
                routing_latency_ms,
                json.dumps(available_tools, ensure_ascii=True, sort_keys=True),
            )

            prompt = self._build_prompt(user_query, available_tools, scratchpad)
            try:
                raw_model_output = await self._call_ollama_async(prompt)
            except (TimeoutError, urllib.error.URLError, socket.timeout) as exc:
                timeout_observation = "Observation: The LLM API timed out. Please try your thought again."
                scratchpad.add("system", timeout_observation)
                steps.append(
                    ReActStep(
                        iteration=iteration,
                        thought="",
                        action="llm_timeout",
                        server_name=None,
                        tool_name=None,
                        arguments=None,
                        observation=f"{timeout_observation} Error: {exc}",
                        raw_model_output="",
                        routing_latency_ms=routing_latency_ms,
                        available_tools=available_tools,
                    )
                )
                iteration += 1
                continue
            logger.debug("Iteration %s raw model output: %s", iteration, raw_model_output)

            try:
                decision = extract_and_parse_json(raw_model_output)
                status = decision.get("status")
                if status == "complete":
                    answer = decision.get("answer")
                    if not isinstance(answer, str) or not answer.strip():
                        raise ValidationError("Completion payload must include a non-empty answer.")

                    thought = str(decision.get("thought", "")).strip()
                    scratchpad.add("assistant", {"status": "complete", "answer": answer}, iteration=iteration)
                    steps.append(
                        ReActStep(
                            iteration=iteration,
                            thought=thought,
                            action="complete",
                            server_name=None,
                            tool_name=None,
                            arguments=None,
                            observation=answer,
                            raw_model_output=raw_model_output,
                            routing_latency_ms=routing_latency_ms,
                            available_tools=available_tools,
                        )
                    )
                    return AgentExecutionResult(
                        status="complete",
                        answer=answer,
                        steps=steps,
                        scratchpad=scratchpad.entries(),
                    )

                if status != "tool_call":
                    raise ValidationError(
                        "Model response must be either {'status':'tool_call', ...} or {'status':'complete', ...}."
                    )

                selected_candidate = self._select_candidate(decision, candidates)
                arguments = decision.get("arguments")
                if not isinstance(arguments, dict):
                    raise ValidationError("Tool call payload must include an object 'arguments' field.")

                validate(instance=arguments, schema=selected_candidate.schema.get("inputSchema", {}))

                tool_result = await self._execute_tool(selected_candidate, arguments)
                observation_text = extract_text_from_tool_result(tool_result) or json.dumps(
                    tool_result,
                    ensure_ascii=True,
                    sort_keys=True,
                )
                thought = str(decision.get("thought", "")).strip()

                scratchpad.add(
                    "assistant",
                    {
                        "status": "tool_call",
                        "server_name": selected_candidate.server_name,
                        "tool_name": selected_candidate.tool_name,
                        "arguments": arguments,
                    },
                    iteration=iteration,
                )
                scratchpad.add(
                    "observation",
                    {
                        "server_name": selected_candidate.server_name,
                        "tool_name": selected_candidate.tool_name,
                        "text": observation_text,
                        "raw": tool_result,
                    },
                    iteration=iteration,
                )
                steps.append(
                    ReActStep(
                        iteration=iteration,
                        thought=thought,
                        action="tool_call",
                        server_name=selected_candidate.server_name,
                        tool_name=selected_candidate.tool_name,
                        arguments=arguments,
                        observation=observation_text,
                        raw_model_output=raw_model_output,
                        routing_latency_ms=routing_latency_ms,
                        available_tools=available_tools,
                    )
                )
            except (ValidationError, LLMOutputParsingError) as exc:
                error_text = self._format_recovery_observation(exc)
                scratchpad.add("observation", error_text, iteration=iteration)
                steps.append(
                    ReActStep(
                        iteration=iteration,
                        thought="",
                        action="recovery",
                        server_name=None,
                        tool_name=None,
                        arguments=None,
                        observation=error_text,
                        raw_model_output=raw_model_output,
                        routing_latency_ms=routing_latency_ms,
                        available_tools=available_tools,
                    )
                )
                iteration += 1
                continue
            except MCPClientError as exc:
                error_text = f"Observation: Tool execution failed. {exc}"
                scratchpad.add("observation", error_text, iteration=iteration)
                steps.append(
                    ReActStep(
                        iteration=iteration,
                        thought="",
                        action="tool_error",
                        server_name=None,
                        tool_name=None,
                        arguments=None,
                        observation=error_text,
                        raw_model_output=raw_model_output,
                        routing_latency_ms=routing_latency_ms,
                        available_tools=available_tools,
                    )
                )
                iteration += 1
                continue

            iteration += 1

        return AgentExecutionResult(
            status="failed",
            answer="Iteration limit exceeded before the agent could complete the task.",
            steps=steps,
            scratchpad=scratchpad.entries(),
        )

    async def close(self) -> None:
        while self._owned_clients:
            client = self._owned_clients.pop()
            with contextlib.suppress(Exception):
                await client.close()

    def _build_routing_query(self, user_query: str, scratchpad: Scratchpad) -> str:
        last_observation_text = scratchpad.last_observation_text()
        return f"Original Goal: {user_query}. Recent Observation: {last_observation_text}"

    def _serialize_candidates(self, candidates: list[RouteResult]) -> list[JsonDict]:
        return [
            {
                "server_name": candidate.server_name,
                "tool_name": candidate.tool_name,
                "description": candidate.schema.get("description", ""),
                "inputSchema": candidate.schema.get("inputSchema", {}),
                "score": round(candidate.score, 6),
            }
            for candidate in candidates
        ]

    def _build_prompt(self, user_query: str, faiss_top_k_schemas: list[JsonDict], scratchpad: Scratchpad) -> str:
        return (
            f"Goal: {user_query}\n"
            f"Available Tools: {json.dumps(faiss_top_k_schemas, ensure_ascii=True)}\n"
            f"History: {scratchpad.render()}\n"
            "Output strictly a JSON object to call a tool, or return {'status': 'complete', 'answer': '...'}\n"
            "When calling a tool, return {'status': 'tool_call', 'server_name': '...', 'tool_name': '...', 'arguments': {...}, 'thought': '...'}\n"
            "Choose only from Available Tools. Use exactly one tool call per iteration."
        )

    async def _call_ollama_async(self, prompt: str) -> str:
        return await asyncio.to_thread(self._call_ollama_blocking, prompt)

    def _call_ollama_blocking(self, prompt: str) -> str:
        payload = {
            "model": self.ollama_model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0,
            },
        }
        request = urllib.request.Request(
            self.ollama_url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=45) as response:
                raw_response = response.read().decode("utf-8")
        except TimeoutError:
            raise
        except socket.timeout as exc:
            raise TimeoutError("Ollama request timed out.") from exc
        except urllib.error.URLError:
            raise

        parsed_response = json.loads(raw_response)
        response_text = parsed_response.get("response")
        if not isinstance(response_text, str):
            raise RuntimeError(f"Unexpected Ollama response payload: {parsed_response!r}")
        return response_text

    def _select_candidate(self, decision: JsonDict, candidates: list[RouteResult]) -> RouteResult:
        server_name = decision.get("server_name")
        tool_name = decision.get("tool_name")
        if not isinstance(server_name, str) or not isinstance(tool_name, str):
            raise ValidationError("Tool call payload must include string server_name and tool_name fields.")

        for candidate in candidates:
            if candidate.server_name == server_name and candidate.tool_name == tool_name:
                return candidate

        raise ValidationError(
            f"Selected tool {server_name}/{tool_name} was not present in the routed top-k tool set."
        )

    async def _execute_tool(self, candidate: RouteResult, arguments: JsonDict) -> JsonDict:
        client = self.clients.get(candidate.server_name)
        if client is None:
            raise MCPClientError(f"No MCP client registered for server {candidate.server_name}")
        return await client.call_tool(candidate.tool_name, arguments)

    @staticmethod
    def _format_recovery_observation(exc: ValidationError | LLMOutputParsingError) -> str:
        if isinstance(exc, LLMOutputParsingError):
            return f"Observation: Your JSON was malformed. Fix it. Raw output: {exc.raw_text}"
        return f"Observation: Your tool call was invalid. Fix it. Error: {exc.message}"
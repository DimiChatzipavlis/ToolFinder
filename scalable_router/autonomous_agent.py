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
        max_iterations: int = 15,
    ) -> None:
        self.router = UniversalMCPRouter(model_name=model_name)
        self.ollama_model = ollama_model
        self.ollama_url = ollama_url
        self.max_iterations = max(15, max_iterations)
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
        last_thought: str | None = None
        executed_actions: set[str] = set()

        while iteration <= self.max_iterations:
            routing_query = last_thought if last_thought else user_query
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

            history_entries = scratchpad.entries()
            # Keep only the latest turns to cap context growth per iteration.
            recent_history = history_entries[-4:] if len(history_entries) > 4 else history_entries
            history_text = "\n".join(
                f"{msg.get('role', 'unknown')}: "
                f"{json.dumps(msg.get('content'), ensure_ascii=True, sort_keys=True) if isinstance(msg.get('content'), (dict, list)) else msg.get('content')}"
                for msg in recent_history
            )
            prompt = self._build_prompt(user_query, available_tools, history_text)
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
                parsed_json = decision
                thought = parsed_json.get("thought")
                if not isinstance(thought, str) or not thought.strip():
                    raise ValidationError("Model response must include a non-empty thought field.")
                last_thought = thought.strip()

                status = parsed_json.get("status")
                if status == "complete":
                    answer = parsed_json.get("answer")
                    if not isinstance(answer, str) or not answer.strip():
                        raise ValidationError("Completion payload must include a non-empty answer.")

                    scratchpad.add("assistant", {"status": "complete", "answer": answer}, iteration=iteration)
                    steps.append(
                        ReActStep(
                            iteration=iteration,
                            thought=last_thought,
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

                action = parsed_json.get("action")
                if action != "call_tool":
                    raise ValidationError(
                        "Model response must be either {'thought': '...', 'action': 'call_tool', ...} or {'thought': '...', 'status': 'complete', ...}."
                    )

                selected_candidate = self._select_candidate(parsed_json, candidates)
                arguments = parsed_json.get("arguments")
                if not isinstance(arguments, dict):
                    raise ValidationError("Tool call payload must include an object 'arguments' field.")

                action_signature = (
                    f"{parsed_json.get('server_name')}:"
                    f"{parsed_json.get('tool_name')}:"
                    f"{str(parsed_json.get('arguments'))}"
                )
                if action_signature in executed_actions:
                    scratchpad.add(
                        "system",
                        (
                            f"Observation: ERROR. You already executed {action_signature}. "
                            "DO NOT REPEAT IT. Look at the data and move to the next logical step."
                        ),
                        iteration=iteration,
                    )
                    iteration += 1
                    continue
                executed_actions.add(action_signature)

                validate(instance=arguments, schema=selected_candidate.schema.get("inputSchema", {}))

                execution_result = await self._execute_tool(selected_candidate, arguments)
                observation_text = extract_text_from_tool_result(execution_result) or json.dumps(
                    execution_result,
                    ensure_ascii=True,
                    sort_keys=True,
                )
                raw_result = str(execution_result)
                if len(raw_result) > 800:
                    safe_result = raw_result[:800] + "\n...[TRUNCATED TO PROTECT VRAM]"
                else:
                    safe_result = raw_result
                scratchpad.add(
                    "assistant",
                    {
                        "action": "call_tool",
                        "server_name": selected_candidate.server_name,
                        "tool_name": selected_candidate.tool_name,
                        "arguments": arguments,
                        "thought": last_thought,
                    },
                    iteration=iteration,
                )
                scratchpad.add("system", f"Observation: {safe_result}", iteration=iteration)
                steps.append(
                    ReActStep(
                        iteration=iteration,
                        thought=last_thought,
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

    def _build_prompt(self, user_query: str, faiss_top_k: list[JsonDict], history_text: str) -> str:
        return (
            f"GOAL: {user_query}\n\n"
            "RECENT HISTORY:\n"
            f"{history_text}\n\n"
            "AVAILABLE TOOLS:\n"
            f"{json.dumps(faiss_top_k, ensure_ascii=True)}\n\n"
            "INSTRUCTIONS:\n"
            "You are an autonomous agent. You must output EXACTLY ONE valid JSON object and nothing else. Do not use markdown formatting.\n\n"
            "EXAMPLE VALID OUTPUT (To take an action):\n"
            "{\"thought\": \"I need to fetch the data first.\", \"action\": \"call_tool\", \"server_name\": \"fetch\", \"tool_name\": \"fetch_json\", \"arguments\": {\"url\": \"https://example.com/data.json\"}}\n\n"
            "EXAMPLE VALID OUTPUT (To finish the goal):\n"
            "{\"thought\": \"I have inserted the data and created the note.\", \"status\": \"complete\", \"answer\": \"The pipeline is finished.\"}\n\n"
            "YOUR TURN. Output only JSON:"
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
        prompt_bytes = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            self.ollama_url,
            data=prompt_bytes,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        print(f"\n[SYSTEM] Sending prompt to local Ollama (Context Size: {len(prompt_bytes)} bytes)...")
        start_time = time.time()
        try:
            with urllib.request.urlopen(request, timeout=300) as response:
                raw_response = response.read().decode("utf-8")
            elapsed = time.time() - start_time
            print(f"[SYSTEM] Ollama responded in {elapsed:.2f} seconds.")
        except TimeoutError:
            elapsed = time.time() - start_time
            print(f"[SYSTEM] Ollama request failed after {elapsed:.2f} seconds.")
            raise
        except socket.timeout as exc:
            elapsed = time.time() - start_time
            print(f"[SYSTEM] Ollama request failed after {elapsed:.2f} seconds.")
            raise TimeoutError("Ollama request timed out.") from exc
        except urllib.error.URLError:
            elapsed = time.time() - start_time
            print(f"[SYSTEM] Ollama request failed after {elapsed:.2f} seconds.")
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
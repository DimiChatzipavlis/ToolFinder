from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable
from typing import Any, Protocol

from toolfinder.utils import LLMOutputParsingError, extract_and_parse_json

from .contracts import PlannerDecision, PlannerTurnInput, ToolCandidate


class PlannerBackend(Protocol):
    async def complete(self, prompt: str) -> str:
        ...


class OpenClawPlanner:
    """OpenClaw-style planner over retrieved tool candidates."""

    def __init__(
        self,
        backend: PlannerBackend,
        timeout_s: float = 45.0,
        fallback_on_parse_error: bool = True,
    ) -> None:
        self._backend = backend
        self._timeout_s = timeout_s
        self._fallback_on_parse_error = fallback_on_parse_error

    async def plan(self, turn_input: PlannerTurnInput) -> PlannerDecision:
        prompt = self._build_prompt(turn_input)
        try:
            raw = await asyncio.wait_for(self._backend.complete(prompt), timeout=self._timeout_s)
            payload = extract_and_parse_json(raw)
            return self._decision_from_payload(payload)
        except (TimeoutError, LLMOutputParsingError, ValueError, KeyError, RuntimeError):
            if not self._fallback_on_parse_error:
                raise
            return self._fallback_decision(turn_input)

    def _decision_from_payload(self, payload: dict[str, Any]) -> PlannerDecision:
        thought = payload.get("thought")
        if not isinstance(thought, str) or not thought.strip():
            raise ValueError("planner response missing non-empty thought")

        status = payload.get("status")
        if status == "complete":
            answer = payload.get("answer")
            if not isinstance(answer, str) or not answer.strip():
                raise ValueError("complete response missing answer")
            return PlannerDecision(action="complete", thought=thought.strip(), answer=answer.strip())

        action = payload.get("action")
        if action != "call_tool":
            raise ValueError("planner response must be complete or call_tool")

        server_name = payload.get("server_name")
        tool_name = payload.get("tool_name")
        arguments = payload.get("arguments")
        if not isinstance(server_name, str) or not isinstance(tool_name, str):
            raise ValueError("call_tool response missing server_name/tool_name")
        if not isinstance(arguments, dict):
            raise ValueError("call_tool response missing object arguments")

        return PlannerDecision(
            action="call_tool",
            thought=thought.strip(),
            server_name=server_name,
            tool_name=tool_name,
            arguments=arguments,
        )

    def _build_prompt(self, turn_input: PlannerTurnInput) -> str:
        candidates_payload = [
            {
                "server_name": c.server_name,
                "tool_name": c.tool_name,
                "description": c.description,
                "inputSchema": c.input_schema,
                "score": round(c.score, 6),
            }
            for c in turn_input.candidates
        ]
        return (
            f"SESSION: {turn_input.session_id}\\n"
            f"TURN: {turn_input.turn_index}\\n"
            f"GOAL: {turn_input.user_query}\\n\\n"
            "HISTORY (json):\\n"
            f"{json.dumps(turn_input.history, ensure_ascii=True)}\\n\\n"
            "ROUTED_TOOLS (json):\\n"
            f"{json.dumps(candidates_payload, ensure_ascii=True)}\\n\\n"
            "Return exactly one JSON object and nothing else.\\n"
            "Never include markdown code fences.\\n"
            "Do not call tools outside ROUTED_TOOLS.\\n"
            "Valid action format:\\n"
            "{\"thought\":\"...\",\"action\":\"call_tool\",\"server_name\":\"...\",\"tool_name\":\"...\",\"arguments\":{...}}\\n"
            "Valid completion format:\\n"
            "{\"thought\":\"...\",\"status\":\"complete\",\"answer\":\"...\"}"
        )

    def _fallback_decision(self, turn_input: PlannerTurnInput) -> PlannerDecision:
        observations = [entry for entry in turn_input.history if entry.get("role") == "observation"]
        if observations:
            last_observation = str(observations[-1].get("content", ""))
            return PlannerDecision(
                action="complete",
                thought="Planner backend unavailable; completing from latest observation.",
                answer=f"Completed from tool observation: {last_observation}",
            )

        if not turn_input.candidates:
            return PlannerDecision(
                action="complete",
                thought="No routed tools available.",
                answer="No candidate tools matched the query.",
            )

        selected = turn_input.candidates[0]
        return PlannerDecision(
            action="call_tool",
            thought="Fallback planner selected top-ranked routed tool.",
            server_name=selected.server_name,
            tool_name=selected.tool_name,
            arguments=self._minimal_arguments(selected),
        )

    @staticmethod
    def _minimal_arguments(candidate: ToolCandidate) -> dict[str, Any]:
        schema = candidate.input_schema if isinstance(candidate.input_schema, dict) else {}
        required = schema.get("required", []) if isinstance(schema.get("required"), list) else []
        properties = schema.get("properties", {}) if isinstance(schema.get("properties"), dict) else {}

        defaults: dict[str, Any] = {}
        for key in required:
            property_schema = properties.get(key, {}) if isinstance(properties.get(key), dict) else {}
            value_type = property_schema.get("type")
            if value_type == "string":
                key_lower = str(key).lower()
                if "path" in key_lower or "file" in key_lower or "dir" in key_lower:
                    defaults[key] = "."
                elif "content" in key_lower or "text" in key_lower:
                    defaults[key] = "autogenerated summary"
                else:
                    defaults[key] = ""
            elif value_type == "integer":
                defaults[key] = 0
            elif value_type == "number":
                defaults[key] = 0.0
            elif value_type == "boolean":
                defaults[key] = False
            elif value_type == "array":
                defaults[key] = []
            elif value_type == "object":
                defaults[key] = {}
            else:
                defaults[key] = None

        return defaults


class HeuristicPlanner:
    """Deterministic local planner for smoke tests and offline demos."""

    def __init__(self, complete_on_empty: bool = True) -> None:
        self.complete_on_empty = complete_on_empty

    async def plan(self, turn_input: PlannerTurnInput) -> PlannerDecision:
        observations = [entry for entry in turn_input.history if entry.get("role") == "observation"]
        if observations:
            last_observation = str(observations[-1].get("content", ""))
            return PlannerDecision(
                action="complete",
                thought="A tool observation is available; finishing deterministically.",
                answer=f"Completed using routed tools. Last observation: {last_observation}",
            )

        if not turn_input.candidates:
            if self.complete_on_empty:
                return PlannerDecision(
                    action="complete",
                    thought="No tools available.",
                    answer="No tools were routed for this request.",
                )
            raise RuntimeError("no routed tools")

        query_lower = turn_input.user_query.lower()
        ranked = list(turn_input.candidates)
        ranked.sort(key=lambda c: c.score, reverse=True)

        if "list" in query_lower:
            for candidate in ranked:
                if "list" in candidate.tool_name:
                    return PlannerDecision(
                        action="call_tool",
                        thought="Listing intent detected; selecting list-like tool.",
                        server_name=candidate.server_name,
                        tool_name=candidate.tool_name,
                        arguments=OpenClawPlanner._minimal_arguments(candidate),
                    )

        selected = ranked[0]
        return PlannerDecision(
            action="call_tool",
            thought="Selecting top ranked routed tool.",
            server_name=selected.server_name,
            tool_name=selected.tool_name,
            arguments=OpenClawPlanner._minimal_arguments(selected),
        )


class CallableBackendAdapter:
    """Adapter that wraps an async callable as a planner backend."""

    def __init__(self, fn: Callable[[str], Awaitable[str]]) -> None:
        self._fn = fn

    async def complete(self, prompt: str) -> str:
        return await self._fn(prompt)

from __future__ import annotations

"""End-to-end hybrid pipeline: ToolFinder retrieval gate + OpenClaw agent execution.

This module provides a unified pipeline that:
1. Uses ToolFinder's semantic retrieval to pre-filter tools for a query.
2. Converts the filtered candidates into an OpenClaw tool manifest.
3. Launches an OpenClaw agent session with the constrained tool set.
4. Falls back to the existing heuristic orchestrator if openclaw fails.
"""

import asyncio
import copy
import json
import time
from typing import Any

from .config import EnterpriseConfig
from .contracts import (
    HybridPipelineResult,
    OpenClawAgentRequest,
    OpenClawAgentResponse,
    PlannerDecision,
    PipelinePhase,
    ToolCallRecord,
    ToolCandidate,
)
from .event_bus import EnterpriseEventBus
from .executor import HybridToolExecutor
from .openclaw_backend import OpenClawCliBackend, OpenClawHttpBackend
from .orchestrator import HybridEnterpriseOrchestrator
from .policy import PolicyEngine, PolicyViolation
from .registry import HybridToolRegistry
from .telemetry import TelemetryCollector


# ---------------------------------------------------------------------------
# Tool manifest builder
# ---------------------------------------------------------------------------


class OpenClawToolManifest:
    """Converts ToolCandidate objects into the OpenClaw tool definition format.

    OpenClaw expects a tool manifest as a JSON array, where each tool has:
        - name: Fully qualified tool name (server/tool).
        - description: Human-readable purpose.
        - parameters: JSON Schema for inputs.
        - metadata: Extra routing info (server_name, score).
    """

    @staticmethod
    def from_candidates(candidates: list[ToolCandidate]) -> list[dict[str, Any]]:
        """Build an OpenClaw-compatible tool manifest from routed candidates."""
        manifest: list[dict[str, Any]] = []
        for candidate in candidates:
            manifest.append(
                {
                    "name": f"{candidate.server_name}/{candidate.tool_name}",
                    "description": candidate.description,
                    "parameters": copy.deepcopy(candidate.input_schema),
                    "metadata": {
                        "server_name": candidate.server_name,
                        "tool_name": candidate.tool_name,
                        "routing_score": round(candidate.score, 6),
                    },
                }
            )
        return manifest

    @staticmethod
    def render_json(manifest: list[dict[str, Any]]) -> str:
        """Serialize manifest to a compact JSON string for CLI/HTTP payloads."""
        return json.dumps(manifest, ensure_ascii=True, indent=None,  sort_keys=True)


# ---------------------------------------------------------------------------
# OpenClaw session driver
# ---------------------------------------------------------------------------


class StructuredParsingError(RuntimeError):
    """Raised when OpenClaw output is not a structured JSON action/completion payload."""

    pass


class OpenClawSessionDriver:
    """Manages an OpenClaw agent session over HTTP or CLI backends.

    Unlike the existing plan-only usage, this driver sends a full agent
    request with embedded tool manifest so OpenClaw can handle multi-step
    reasoning and tool calls internally.
    """

    def __init__(
        self,
        backend: OpenClawHttpBackend | OpenClawCliBackend,
        timeout_s: float = 90.0,
    ) -> None:
        self._backend = backend
        self._timeout_s = timeout_s

    async def run_agent(self, request: OpenClawAgentRequest) -> OpenClawAgentResponse:
        """Execute a full OpenClaw agent session and parse the response."""
        prompt = self._build_agent_prompt(request)
        try:
            raw_output = await asyncio.wait_for(
                self._backend.complete(prompt),
                timeout=self._timeout_s,
            )
            return self._parse_agent_output(raw_output)
        except TimeoutError:
            return OpenClawAgentResponse(
                answer="",
                tool_calls=[],
                raw_output="",
                success=False,
                error="OpenClaw agent session timed out",
            )
        except StructuredParsingError as exc:
            return OpenClawAgentResponse(
                answer="",
                tool_calls=[],
                raw_output="",
                success=False,
                error=f"OpenClaw agent structured parsing failed: {exc}",
            )
        except Exception as exc:
            return OpenClawAgentResponse(
                answer="",
                tool_calls=[],
                raw_output="",
                success=False,
                error=f"OpenClaw agent session failed: {exc}",
            )

    def _build_agent_prompt(self, request: OpenClawAgentRequest) -> str:
        """Build an agent-mode prompt with embedded tool manifest."""
        manifest_json = OpenClawToolManifest.render_json(request.tool_manifest)
        return (
            f"SESSION: {request.session_id}\n"
            f"MODE: agent\n"
            f"MAX_STEPS: {request.max_steps}\n\n"
            "AVAILABLE_TOOLS (json):\n"
            f"{manifest_json}\n\n"
            f"GOAL: {request.query}\n\n"
            "You are an agent that can use the tools listed above.\n"
            "Execute the user's goal step by step.\n"
            "For each step, if you need a tool, output a JSON action block:\n"
            '{"thought":"...","action":"call_tool","tool":"server/tool","arguments":{...}}\n'
            "When done, output a final JSON completion block:\n"
            '{"thought":"...","status":"complete","answer":"..."}\n\n'
            "If you can answer directly without tools, provide the completion block immediately.\n"
            "Return all steps as a JSON array. Never include markdown code fences.\n"
        )

    def _parse_agent_output(self, raw_output: str) -> OpenClawAgentResponse:
        """Parse OpenClaw agent output into a structured response."""
        tool_calls: list[dict[str, Any]] = []
        answer = ""

        # Attempt to parse as a JSON array of steps.
        try:
            steps = json.loads(raw_output)
            if isinstance(steps, list):
                return self._extract_from_steps(steps, raw_output)
        except (json.JSONDecodeError, ValueError):
            pass

        # Attempt to parse as a single JSON object.
        try:
            payload = json.loads(raw_output)
            if isinstance(payload, dict):
                return self._extract_from_single(payload, raw_output)
        except (json.JSONDecodeError, ValueError):
            pass

        # Attempt line-by-line JSON extraction (mixed output).
        for line in raw_output.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if not isinstance(obj, dict):
                    continue
                if obj.get("status") == "complete":
                    answer = str(obj.get("answer", ""))
                elif obj.get("action") == "call_tool":
                    tool_calls.append(obj)
            except (json.JSONDecodeError, ValueError):
                continue

        if answer or tool_calls:
            return OpenClawAgentResponse(
                answer=answer,
                tool_calls=tool_calls,
                raw_output=raw_output,
                success=True,
            )

        stripped = raw_output.strip()
        if stripped:
            raise StructuredParsingError("backend returned unstructured plain text")
        raise StructuredParsingError("backend returned empty output")

    def _extract_from_steps(
        self, steps: list[Any], raw_output: str
    ) -> OpenClawAgentResponse:
        """Extract answer and tool calls from a JSON array of agent steps."""
        tool_calls: list[dict[str, Any]] = []
        answer = ""
        for step in steps:
            if not isinstance(step, dict):
                continue
            if step.get("status") == "complete":
                answer = str(step.get("answer", ""))
            elif step.get("action") == "call_tool":
                tool_calls.append(step)
        if not answer and not tool_calls:
            raise StructuredParsingError("json array did not contain completion or tool calls")
        return OpenClawAgentResponse(
            answer=answer,
            tool_calls=tool_calls,
            raw_output=raw_output,
            success=True,
            metadata={"parse_mode": "json_array", "step_count": len(steps)},
        )

    def _extract_from_single(
        self, payload: dict[str, Any], raw_output: str
    ) -> OpenClawAgentResponse:
        """Extract answer from a single JSON completion object."""
        if payload.get("status") == "complete":
            return OpenClawAgentResponse(
                answer=str(payload.get("answer", "")),
                tool_calls=[],
                raw_output=raw_output,
                success=True,
                metadata={"parse_mode": "json_single"},
            )
        if payload.get("action") == "call_tool":
            return OpenClawAgentResponse(
                answer="",
                tool_calls=[payload],
                raw_output=raw_output,
                success=True,
                metadata={"parse_mode": "json_single_action"},
            )
        # Unrecognized shape — try extracting common keys.
        for key in ("answer", "response", "content", "output"):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                raise StructuredParsingError(
                    "json object used unsupported answer field; expected structured completion payload"
                )
        raise StructuredParsingError("unrecognized OpenClaw agent response shape")


# ---------------------------------------------------------------------------
# Hybrid fallback strategy
# ---------------------------------------------------------------------------


class FallbackStrategy:
    """Controls what happens when the OpenClaw agent path fails."""

    HEURISTIC_PLANNER = "heuristic_planner"
    ERROR = "error"
    BEST_EFFORT = "best_effort"

    VALID = {HEURISTIC_PLANNER, ERROR, BEST_EFFORT}

    @classmethod
    def validate(cls, value: str) -> str:
        if value not in cls.VALID:
            raise ValueError(
                f"invalid fallback strategy: {value!r}, must be one of {cls.VALID}"
            )
        return value


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


class OpenClawHybridPipeline:
    """End-to-end hybrid runtime: retrieval gate → OpenClaw agent → fallback.

    This pipeline orchestrates the full hybrid flow:
    1. Route user query through ToolFinder's semantic retrieval to get top-k tools.
    2. Convert candidates into an OpenClaw tool manifest.
    3. Launch an OpenClaw agent session with the constrained tool set.
    4. If OpenClaw succeeds, map its tool calls back to MCP-server-specific
       records and return a unified result.
    5. If OpenClaw fails, fall back to the existing HybridEnterpriseOrchestrator
       (retrieval → heuristic/planner → executor).
    """

    def __init__(
        self,
        registry: HybridToolRegistry,
        session_driver: OpenClawSessionDriver,
        fallback_orchestrator: HybridEnterpriseOrchestrator | None = None,
        executor: HybridToolExecutor | None = None,
        policy_engine: PolicyEngine | None = None,
        config: EnterpriseConfig | None = None,
        event_bus: EnterpriseEventBus | None = None,
        telemetry: TelemetryCollector | None = None,
        fallback_strategy: str = FallbackStrategy.HEURISTIC_PLANNER,
        max_agent_steps: int = 10,
        model: str = "",
        require_all_tool_calls_success: bool = True,
    ) -> None:
        self.registry = registry
        self.session_driver = session_driver
        self.fallback_orchestrator = fallback_orchestrator
        self.executor = executor
        self.config = config or EnterpriseConfig()
        if policy_engine is not None:
            self.policy_engine = policy_engine
        elif fallback_orchestrator is not None:
            self.policy_engine = fallback_orchestrator.policy_engine
        else:
            self.policy_engine = PolicyEngine()
        self.event_bus = event_bus or EnterpriseEventBus(max_handler_errors=self.config.event_bus_max_errors)
        self.telemetry = telemetry or TelemetryCollector(
            max_latency_samples_per_metric=self.config.telemetry_max_latency_samples
        )
        if not self.telemetry.sink_path and self.config.telemetry_sink_path:
            self.telemetry.sink_path = self.config.telemetry_sink_path
        self.fallback_strategy = FallbackStrategy.validate(fallback_strategy)
        self.max_agent_steps = max_agent_steps
        self.model = model
        self.require_all_tool_calls_success = require_all_tool_calls_success

    async def run(self, session_id: str, user_query: str) -> HybridPipelineResult:
        """Execute the full hybrid pipeline for a user query."""
        phase_trace: list[str] = []
        history: list[dict[str, Any]] = [{"role": "user", "content": user_query}]

        # ── Phase 1: Semantic routing ──────────────────────────────────
        phase_trace.append(PipelinePhase.ROUTING.value)
        routing_started = time.perf_counter()
        candidates = await self.registry.route(
            user_query,
            k=self.config.top_k,
            min_score=self.config.min_score,
        )
        routing_ms = (time.perf_counter() - routing_started) * 1000.0
        self.telemetry.record_latency("pipeline_routing", routing_ms)
        await self._publish(
            {
                "type": "pipeline_routing",
                "session_id": session_id,
                "latency_ms": round(routing_ms, 3),
                "candidate_count": len(candidates),
            }
        )

        if not candidates:
            self.telemetry.increment("pipeline_no_route")
            phase_trace.append(PipelinePhase.COMPLETE.value)
            return self._finalize_result(
                session_id,
                HybridPipelineResult(
                status="failed",
                answer="No tools met routing threshold.",
                turns=0,
                tool_calls=[],
                telemetry=self.telemetry.to_dict(),
                final_history=history,
                phase_trace=phase_trace,
                execution_path="direct",
                fallback_triggered=False,
                ),
            )

        # ── Phase 2: OpenClaw agent execution ─────────────────────────
        phase_trace.append(PipelinePhase.OPENCLAW_AGENT.value)
        manifest = OpenClawToolManifest.from_candidates(candidates)
        agent_request = OpenClawAgentRequest(
            query=user_query,
            tool_manifest=manifest,
            session_id=session_id,
            model=self.model,
            max_steps=self.max_agent_steps,
        )

        agent_started = time.perf_counter()
        agent_response = await self.session_driver.run_agent(agent_request)
        agent_ms = (time.perf_counter() - agent_started) * 1000.0
        self.telemetry.record_latency("pipeline_openclaw_agent", agent_ms)
        await self._publish(
            {
                "type": "pipeline_openclaw_agent",
                "session_id": session_id,
                "latency_ms": round(agent_ms, 3),
                "success": agent_response.success,
                "tool_call_count": len(agent_response.tool_calls),
            }
        )

        if agent_response.success and agent_response.tool_calls:
            if self.executor is None:
                agent_response = OpenClawAgentResponse(
                    answer=agent_response.answer,
                    tool_calls=agent_response.tool_calls,
                    raw_output=agent_response.raw_output,
                    success=False,
                    error="OpenClaw returned tool calls but no executor is configured.",
                    metadata=dict(agent_response.metadata),
                )
            else:
                self.telemetry.increment("pipeline_openclaw_tool_execution")
                tool_execution_result = await self._execute_openclaw_tool_calls(
                    session_id=session_id,
                    user_query=user_query,
                    agent_response=agent_response,
                    candidates=candidates,
                    history=history,
                    phase_trace=phase_trace,
                )
                if tool_execution_result.status == "complete":
                    return self._finalize_result(session_id, tool_execution_result)

                agent_response = OpenClawAgentResponse(
                    answer=agent_response.answer,
                    tool_calls=agent_response.tool_calls,
                    raw_output=agent_response.raw_output,
                    success=False,
                    error=tool_execution_result.answer,
                    metadata=dict(agent_response.metadata),
                )

        if agent_response.success and agent_response.answer:
            self.telemetry.increment("pipeline_openclaw_success")
            tool_records = self._map_tool_calls(agent_response.tool_calls, candidates)
            history.append(
                {
                    "role": "assistant",
                    "content": {
                        "source": "openclaw_agent",
                        "answer": agent_response.answer,
                        "tool_calls": agent_response.tool_calls,
                    },
                }
            )
            phase_trace.append(PipelinePhase.COMPLETE.value)
            return self._finalize_result(
                session_id,
                HybridPipelineResult(
                    status="complete",
                    answer=agent_response.answer,
                    turns=max(1, len(agent_response.tool_calls)),
                    tool_calls=tool_records,
                    telemetry=self.telemetry.to_dict(),
                    final_history=history,
                    phase_trace=phase_trace,
                    execution_path="openclaw",
                    openclaw_response=agent_response,
                    fallback_triggered=False,
                ),
            )

        # ── Phase 3: Fallback ─────────────────────────────────────────
        self.telemetry.increment("pipeline_openclaw_failed")
        await self._publish(
            {
                "type": "pipeline_openclaw_failed",
                "session_id": session_id,
                "error": agent_response.error,
            }
        )

        if self.fallback_strategy == FallbackStrategy.ERROR:
            phase_trace.append(PipelinePhase.COMPLETE.value)
            return self._finalize_result(
                session_id,
                HybridPipelineResult(
                status="failed",
                answer=f"OpenClaw agent failed: {agent_response.error}",
                turns=0,
                tool_calls=[],
                telemetry=self.telemetry.to_dict(),
                final_history=history,
                phase_trace=phase_trace,
                execution_path="openclaw",
                openclaw_response=agent_response,
                fallback_triggered=False,
                ),
            )

        if self.fallback_strategy == FallbackStrategy.BEST_EFFORT:
            # Return whatever we got, even partial.
            partial_answer = agent_response.answer or agent_response.raw_output or ""
            tool_records = self._map_tool_calls(agent_response.tool_calls, candidates)
            phase_trace.append(PipelinePhase.COMPLETE.value)
            return self._finalize_result(
                session_id,
                HybridPipelineResult(
                status="partial",
                answer=partial_answer[:self.config.max_observation_chars],
                turns=max(1, len(agent_response.tool_calls)),
                tool_calls=tool_records,
                telemetry=self.telemetry.to_dict(),
                final_history=history,
                phase_trace=phase_trace,
                execution_path="openclaw",
                openclaw_response=agent_response,
                fallback_triggered=False,
                ),
            )

        # ── HEURISTIC_PLANNER fallback ────────────────────────────────
        phase_trace.append(PipelinePhase.FALLBACK.value)
        if self.fallback_orchestrator is None:
            phase_trace.append(PipelinePhase.COMPLETE.value)
            return self._finalize_result(
                session_id,
                HybridPipelineResult(
                status="failed",
                answer="OpenClaw agent failed and no fallback orchestrator is configured.",
                turns=0,
                tool_calls=[],
                telemetry=self.telemetry.to_dict(),
                final_history=history,
                phase_trace=phase_trace,
                execution_path="fallback",
                openclaw_response=agent_response,
                fallback_triggered=True,
                ),
            )

        self.telemetry.increment("pipeline_fallback")
        await self._publish(
            {
                "type": "pipeline_fallback",
                "session_id": session_id,
                "strategy": self.fallback_strategy,
            }
        )

        fallback_started = time.perf_counter()
        fallback_result = await self.fallback_orchestrator.run(
            session_id=f"{session_id}-fallback",
            user_query=user_query,
        )
        fallback_ms = (time.perf_counter() - fallback_started) * 1000.0
        self.telemetry.record_latency("pipeline_fallback", fallback_ms)
        self.telemetry.merge_snapshot(fallback_result.telemetry)
        self.telemetry.increment("pipeline_fallback_triggered")

        phase_trace.append(PipelinePhase.COMPLETE.value)
        return self._finalize_result(
            session_id,
            HybridPipelineResult(
            status="degraded_fallback",
            answer=fallback_result.answer,
            turns=fallback_result.turns,
            tool_calls=fallback_result.tool_calls,
            telemetry=self.telemetry.to_dict(),
            final_history=fallback_result.final_history,
            phase_trace=phase_trace,
            execution_path="fallback",
            openclaw_response=agent_response,
            fallback_triggered=True,
            ),
        )

    # -------------------------------------------------------------------
    # Execute openclaw tool calls through the MCP executor
    # -------------------------------------------------------------------

    async def _execute_openclaw_tool_calls(
        self,
        *,
        session_id: str,
        user_query: str,
        agent_response: OpenClawAgentResponse,
        candidates: list[ToolCandidate],
        history: list[dict[str, Any]],
        phase_trace: list[str],
    ) -> HybridPipelineResult:
        """Execute tool calls from OpenClaw's plan through the MCP executor."""
        del user_query
        assert self.executor is not None

        tool_records: list[ToolCallRecord] = []
        candidate_lookup = {
            (c.server_name, c.tool_name): c for c in candidates
        }
        observations: list[str] = []
        failed_messages: list[str] = []
        successful_calls = 0
        expected_calls = len(agent_response.tool_calls)

        for idx, call in enumerate(agent_response.tool_calls, 1):
            tool_ref = str(call.get("tool", ""))
            arguments = call.get("arguments", {})
            if not isinstance(arguments, dict):
                arguments = {}

            # Parse "server/tool" reference.
            server_name, tool_name = self._parse_tool_ref(tool_ref)
            candidate = candidate_lookup.get((server_name, tool_name))
            if candidate is None:
                self.telemetry.increment("pipeline_tool_errors")
                message = f"Tool {tool_ref} not found in routed candidates"
                observations.append(message)
                failed_messages.append(message)
                await self._publish(
                    {
                        "type": "pipeline_tool_error",
                        "session_id": session_id,
                        "turn": idx,
                        "tool": tool_ref,
                        "error": message,
                    }
                )
                continue

            try:
                policy_decision = self._build_policy_decision(server_name, tool_name, arguments)
                self.policy_engine.enforce_call(
                    policy_decision,
                    candidate_lookup,
                    allow_unrouted_tool_calls=self.config.allow_unrouted_tool_calls,
                )

                exec_started = time.perf_counter()
                _, observation = await self.executor.execute(candidate, arguments)
                exec_ms = (time.perf_counter() - exec_started) * 1000.0
                self.telemetry.record_latency("pipeline_tool_execution", exec_ms)
                self.telemetry.increment("pipeline_tool_calls")
                successful_calls += 1

                tool_records.append(
                    ToolCallRecord(
                        turn_index=idx,
                        server_name=server_name,
                        tool_name=tool_name,
                        arguments=dict(arguments),
                        observation=observation,
                    )
                )
                observations.append(observation)
                await self._publish(
                    {
                        "type": "pipeline_tool_execution",
                        "session_id": session_id,
                        "turn": idx,
                        "tool": tool_ref,
                        "latency_ms": round(exec_ms, 3),
                    }
                )
            except PolicyViolation as exc:
                self.telemetry.increment("pipeline_policy_violations")
                message = f"Policy violation for {tool_ref}: {exc}"
                observations.append(message)
                failed_messages.append(message)
                await self._publish(
                    {
                        "type": "pipeline_tool_error",
                        "session_id": session_id,
                        "turn": idx,
                        "tool": tool_ref,
                        "error": message,
                    }
                )
            except Exception as exc:
                self.telemetry.increment("pipeline_tool_errors")
                message = f"Execution error for {tool_ref}: {exc}"
                observations.append(message)
                failed_messages.append(message)
                await self._publish(
                    {
                        "type": "pipeline_tool_error",
                        "session_id": session_id,
                        "turn": idx,
                        "tool": tool_ref,
                        "error": message,
                    }
                )

        failures = len(failed_messages)
        all_calls_succeeded = successful_calls == expected_calls

        if failures > 0 and self.require_all_tool_calls_success:
            status = "failed"
            answer = (
                "OpenClaw tool execution failed: "
                f"{failures} of {expected_calls} calls failed. "
                f"Last error: {failed_messages[-1]}"
            )
        elif failures > 0 and not tool_records and not agent_response.answer:
            status = "failed"
            answer = failed_messages[-1]
        else:
            status = "complete"
            answer = agent_response.answer
            if not answer and observations:
                answer = observations[-1]
            if not answer:
                answer = "Tool execution completed."

        if failures > 0 and not all_calls_succeeded:
            self.telemetry.increment("pipeline_openclaw_execution_failed")
        elif status == "complete":
            self.telemetry.increment("pipeline_openclaw_success")

        history.append(
            {
                "role": "assistant",
                "content": {
                    "source": "openclaw_agent_executed",
                    "status": status,
                    "tool_calls": [r.__dict__ for r in tool_records] if tool_records else [],
                    "observations": observations,
                    "errors": failed_messages,
                },
            }
        )
        phase_trace.append(PipelinePhase.COMPLETE.value)

        return HybridPipelineResult(
            status=status,
            answer=answer,
            turns=max(1, len(tool_records) or expected_calls),
            tool_calls=tool_records,
            telemetry=self.telemetry.to_dict(),
            final_history=history,
            phase_trace=phase_trace,
            execution_path="openclaw",
            openclaw_response=agent_response,
            fallback_triggered=False,
        )

    # -------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------

    def _map_tool_calls(
        self,
        tool_calls: list[dict[str, Any]],
        candidates: list[ToolCandidate],
    ) -> list[ToolCallRecord]:
        """Map OpenClaw tool call dicts back to typed ToolCallRecord."""
        records: list[ToolCallRecord] = []
        for idx, call in enumerate(tool_calls, 1):
            tool_ref = str(call.get("tool", call.get("tool_name", "")))
            server_name, tool_name = self._parse_tool_ref(tool_ref)
            arguments = call.get("arguments", {})
            if not isinstance(arguments, dict):
                arguments = {}

            records.append(
                ToolCallRecord(
                    turn_index=idx,
                    server_name=server_name,
                    tool_name=tool_name,
                    arguments=dict(arguments),
                    observation=str(call.get("observation", "")),
                )
            )
        return records

    @staticmethod
    def _parse_tool_ref(tool_ref: str) -> tuple[str, str]:
        """Parse 'server_name/tool_name' into a (server, tool) tuple."""
        if "/" in tool_ref:
            parts = tool_ref.split("/", 1)
            return parts[0], parts[1]
        return "", tool_ref

    @staticmethod
    def _build_policy_decision(
        server_name: str,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> PlannerDecision:
        return PlannerDecision(
            action="call_tool",
            thought="openclaw tool call",
            server_name=server_name,
            tool_name=tool_name,
            arguments=dict(arguments),
        )

    def _finalize_result(self, session_id: str, result: HybridPipelineResult) -> HybridPipelineResult:
        try:
            self.telemetry.persist_snapshot(
                {
                    "session_id": session_id,
                    "status": result.status,
                    "execution_path": result.execution_path,
                }
            )
        except Exception:
            # Telemetry persistence must never break request completion.
            self.telemetry.increment("pipeline_telemetry_persist_errors")
        return result

    async def _publish(self, event: dict[str, Any]) -> None:
        event["timestamp"] = time.time()
        await self.event_bus.publish(event)

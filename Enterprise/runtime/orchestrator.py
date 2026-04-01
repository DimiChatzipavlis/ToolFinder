from __future__ import annotations

import json
import time
from typing import Any

from .config import EnterpriseConfig
from .contracts import PlannerTurnInput, SessionResult, ToolCallRecord, ToolCandidate
from .event_bus import EnterpriseEventBus
from .executor import HybridToolExecutor
from .planner import HeuristicPlanner, OpenClawPlanner
from .policy import PolicyEngine
from .registry import HybridToolRegistry
from .telemetry import TelemetryCollector


class HybridEnterpriseOrchestrator:
    """Enterprise hybrid runtime: retrieval gate + planner + guarded execution."""

    def __init__(
        self,
        registry: HybridToolRegistry,
        planner: OpenClawPlanner | HeuristicPlanner,
        executor: HybridToolExecutor,
        policy_engine: PolicyEngine,
        config: EnterpriseConfig | None = None,
        event_bus: EnterpriseEventBus | None = None,
        telemetry: TelemetryCollector | None = None,
    ) -> None:
        self.registry = registry
        self.planner = planner
        self.executor = executor
        self.policy_engine = policy_engine
        self.config = config or EnterpriseConfig()
        self.event_bus = event_bus or EnterpriseEventBus(max_handler_errors=self.config.event_bus_max_errors)
        self.telemetry = telemetry or TelemetryCollector(
            max_latency_samples_per_metric=self.config.telemetry_max_latency_samples
        )
        if not self.telemetry.sink_path and self.config.telemetry_sink_path:
            self.telemetry.sink_path = self.config.telemetry_sink_path

    async def run(self, session_id: str, user_query: str) -> SessionResult:
        history: list[dict[str, Any]] = [{"role": "user", "content": user_query}]
        tool_calls: list[ToolCallRecord] = []
        retries = 0
        repeated_call_streak = 0
        last_call_signature: str | None = None

        for turn_index in range(1, self.config.max_turns + 1):
            routing_query = self._build_routing_query(user_query, history)
            routing_started = time.perf_counter()
            candidates = await self.registry.route(
                routing_query,
                k=self.config.top_k,
                min_score=self.config.min_score,
            )
            routing_ms = (time.perf_counter() - routing_started) * 1000.0
            self.telemetry.record_latency("routing", routing_ms)
            await self._publish(
                {
                    "type": "routing",
                    "session_id": session_id,
                    "turn": turn_index,
                    "latency_ms": round(routing_ms, 3),
                    "candidate_count": len(candidates),
                }
            )

            if not candidates:
                self.telemetry.increment("no_route")
                return self._finalize_result(session_id, SessionResult(
                    status="failed",
                    answer="No tools met routing threshold.",
                    turns=turn_index,
                    tool_calls=tool_calls,
                    telemetry=self.telemetry.to_dict(),
                    final_history=history,
                ))

            planner_input = PlannerTurnInput(
                session_id=session_id,
                user_query=user_query,
                history=history,
                candidates=candidates,
                turn_index=turn_index,
            )
            planning_started = time.perf_counter()
            decision = await self.planner.plan(planner_input)
            planning_ms = (time.perf_counter() - planning_started) * 1000.0
            self.telemetry.record_latency("planning", planning_ms)
            await self._publish(
                {
                    "type": "planning",
                    "session_id": session_id,
                    "turn": turn_index,
                    "latency_ms": round(planning_ms, 3),
                    "decision": {
                        "action": decision.action,
                        "server_name": decision.server_name,
                        "tool_name": decision.tool_name,
                    },
                }
            )

            if decision.action == "complete":
                self.telemetry.increment("complete")
                return self._finalize_result(session_id, SessionResult(
                    status="complete",
                    answer=decision.answer or "",
                    turns=turn_index,
                    tool_calls=tool_calls,
                    telemetry=self.telemetry.to_dict(),
                    final_history=history,
                ))

            if decision.action != "call_tool":
                retries += 1
                self.telemetry.increment("invalid_planner_action")
                history.append({
                    "role": "observation",
                    "content": f"Invalid planner action: {decision.action}",
                })
                if retries > self.config.max_retries:
                    return self._finalize_result(session_id, SessionResult(
                        status="failed",
                        answer="Planner produced invalid actions repeatedly.",
                        turns=turn_index,
                        tool_calls=tool_calls,
                        telemetry=self.telemetry.to_dict(),
                        final_history=history,
                    ))
                continue

            candidate_lookup = self._build_candidate_lookup(candidates)
            try:
                self.policy_engine.enforce_call(
                    decision,
                    candidate_lookup,
                    allow_unrouted_tool_calls=self.config.allow_unrouted_tool_calls,
                )
                selected = candidate_lookup.get((decision.server_name or "", decision.tool_name or ""))
                if selected is None:
                    selected = ToolCandidate(
                        server_name=decision.server_name or "",
                        tool_name=decision.tool_name or "",
                        description="unrouted-tool-call",
                        input_schema={},
                        score=0.0,
                    )

                execution_started = time.perf_counter()
                _, observation = await self.executor.execute(selected, decision.arguments)
                execution_ms = (time.perf_counter() - execution_started) * 1000.0
                self.telemetry.record_latency("execution", execution_ms)
                self.telemetry.increment("tool_calls")

                clipped_observation = self._truncate_observation(observation)
                history.append(
                    {
                        "role": "assistant",
                        "content": {
                            "thought": decision.thought,
                            "action": decision.action,
                            "server_name": selected.server_name,
                            "tool_name": selected.tool_name,
                            "arguments": decision.arguments,
                        },
                    }
                )
                history.append({"role": "observation", "content": clipped_observation})
                tool_calls.append(
                    ToolCallRecord(
                        turn_index=turn_index,
                        server_name=selected.server_name,
                        tool_name=selected.tool_name,
                        arguments=dict(decision.arguments),
                        observation=clipped_observation,
                    )
                )
                await self._publish(
                    {
                        "type": "tool_execution",
                        "session_id": session_id,
                        "turn": turn_index,
                        "tool": f"{selected.server_name}/{selected.tool_name}",
                        "latency_ms": round(execution_ms, 3),
                    }
                )

                call_signature = json.dumps(
                    {
                        "server": selected.server_name,
                        "tool": selected.tool_name,
                        "arguments": decision.arguments,
                    },
                    ensure_ascii=True,
                    sort_keys=True,
                )
                if call_signature == last_call_signature:
                    repeated_call_streak += 1
                else:
                    repeated_call_streak = 1
                    last_call_signature = call_signature

                if repeated_call_streak >= self.config.loop_guard_repetition_limit:
                    counter_key = "loop_guard_complete" if self.config.loop_guard_marks_complete else "loop_guard_failed"
                    self.telemetry.increment(counter_key)
                    await self._publish(
                        {
                            "type": "loop_guard",
                            "session_id": session_id,
                            "turn": turn_index,
                            "tool": f"{selected.server_name}/{selected.tool_name}",
                        }
                    )
                    status = "complete" if self.config.loop_guard_marks_complete else "failed"
                    answer = (
                        f"Completed from repeated observation: {clipped_observation}"
                        if self.config.loop_guard_marks_complete
                        else "Loop guard triggered: planner repeated identical tool calls without progress."
                    )
                    return self._finalize_result(session_id, SessionResult(
                        status=status,
                        answer=answer,
                        turns=turn_index,
                        tool_calls=tool_calls,
                        telemetry=self.telemetry.to_dict(),
                        final_history=history,
                    ))
                retries = 0
            except Exception as exc:
                retries += 1
                self.telemetry.increment("execution_errors")
                history.append({"role": "observation", "content": f"Execution error: {exc}"})
                await self._publish(
                    {
                        "type": "execution_error",
                        "session_id": session_id,
                        "turn": turn_index,
                        "error": str(exc),
                    }
                )
                if retries > self.config.max_retries:
                    return self._finalize_result(session_id, SessionResult(
                        status="failed",
                        answer=f"Execution failed after retries: {exc}",
                        turns=turn_index,
                        tool_calls=tool_calls,
                        telemetry=self.telemetry.to_dict(),
                        final_history=history,
                    ))

        self.telemetry.increment("iteration_limit")
        return self._finalize_result(session_id, SessionResult(
            status="failed",
            answer="Iteration limit reached before completion.",
            turns=self.config.max_turns,
            tool_calls=tool_calls,
            telemetry=self.telemetry.to_dict(),
            final_history=history,
        ))

    def _build_candidate_lookup(
        self,
        candidates: list[ToolCandidate],
    ) -> dict[tuple[str, str], ToolCandidate]:
        lookup: dict[tuple[str, str], ToolCandidate] = {}
        for candidate in candidates:
            lookup[(candidate.server_name, candidate.tool_name)] = candidate
        return lookup

    def _truncate_observation(self, observation: str) -> str:
        if len(observation) <= self.config.max_observation_chars:
            return observation
        return observation[: self.config.max_observation_chars] + "\n...[TRUNCATED]"

    def _build_routing_query(self, user_query: str, history: list[dict[str, Any]]) -> str:
        if not self.config.route_with_history_context:
            return user_query

        latest_observation = ""
        latest_thought = ""
        for entry in reversed(history):
            if not latest_observation and entry.get("role") == "observation":
                latest_observation = str(entry.get("content", ""))[:320]
            if latest_thought:
                continue
            if entry.get("role") != "assistant":
                continue
            content = entry.get("content")
            if isinstance(content, dict):
                thought = content.get("thought")
                if isinstance(thought, str):
                    latest_thought = thought[:200]
            elif isinstance(content, str):
                latest_thought = content[:200]

            if latest_observation and latest_thought:
                break

        if not latest_observation and not latest_thought:
            return user_query

        parts = [f"GOAL: {user_query}"]
        if latest_thought:
            parts.append(f"LAST_THOUGHT: {latest_thought}")
        if latest_observation:
            parts.append(f"LAST_OBSERVATION: {latest_observation}")
        return "\n".join(parts)

    async def _publish(self, event: dict[str, Any]) -> None:
        event["timestamp"] = time.time()
        await self.event_bus.publish(event)

    def _finalize_result(self, session_id: str, result: SessionResult) -> SessionResult:
        try:
            self.telemetry.persist_snapshot(
                {
                    "session_id": session_id,
                    "status": result.status,
                    "turns": result.turns,
                }
            )
        except Exception:
            # Persistence failures are non-fatal for runtime execution.
            self.telemetry.increment("telemetry_persist_errors")
        return result

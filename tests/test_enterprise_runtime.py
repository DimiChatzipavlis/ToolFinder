from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from Enterprise.runtime.contracts import PlannerDecision, PlannerTurnInput, ToolCandidate
from Enterprise.runtime.event_bus import EnterpriseEventBus
from Enterprise.runtime.orchestrator import HybridEnterpriseOrchestrator
from Enterprise.runtime.planner import HeuristicPlanner, OpenClawPlanner
from Enterprise.runtime.policy import PolicyEngine, PolicyViolation, SecurityPolicyViolation, ToolPolicy
from Enterprise.runtime.realtime_service import RealTimeHybridService, WorkspaceChangeTracker
from Enterprise.runtime.config import EnterpriseConfig
from Enterprise.runtime.telemetry import TelemetryCollector


class BrokenBackend:
    async def complete(self, prompt: str) -> str:
        del prompt
        return "not-json"


def _candidate() -> ToolCandidate:
    return ToolCandidate(
        server_name="filesystem",
        tool_name="list_directory",
        description="list files",
        input_schema={
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
        score=0.9,
    )


def test_policy_rejects_unrouted_tool() -> None:
    policy = PolicyEngine(ToolPolicy(allowed_servers={"filesystem"}))
    decision = PlannerDecision(
        action="call_tool",
        thought="test",
        server_name="filesystem",
        tool_name="write_file",
        arguments={"path": "x", "content": "y"},
    )

    with pytest.raises(PolicyViolation):
        policy.enforce_call(decision, candidate_lookup={("filesystem", "list_directory"): _candidate()})


def test_policy_raises_security_violation_on_path_traversal() -> None:
    policy = PolicyEngine(ToolPolicy(allowed_servers={"filesystem"}, allowed_path_roots=("C:/workspace",)))
    decision = PlannerDecision(
        action="call_tool",
        thought="test",
        server_name="filesystem",
        tool_name="list_directory",
        arguments={"path": "../secrets"},
    )

    with pytest.raises(SecurityPolicyViolation):
        policy.enforce_call(decision, candidate_lookup={("filesystem", "list_directory"): _candidate()})


def test_heuristic_planner_completes_after_observation() -> None:
    planner = HeuristicPlanner()
    turn_input = PlannerTurnInput(
        session_id="s1",
        user_query="list files",
        history=[{"role": "observation", "content": "files: a.txt"}],
        candidates=[_candidate()],
        turn_index=2,
    )

    decision = asyncio.run(planner.plan(turn_input))

    assert decision.action == "complete"
    assert "files:" in (decision.answer or "")


def test_openclaw_planner_falls_back_on_parse_error() -> None:
    planner = OpenClawPlanner(backend=BrokenBackend())
    turn_input = PlannerTurnInput(
        session_id="s2",
        user_query="list files",
        history=[],
        candidates=[_candidate()],
        turn_index=1,
    )

    decision = asyncio.run(planner.plan(turn_input))

    assert decision.action == "complete"
    assert "refused speculative tool execution" in (decision.answer or "")


def test_openclaw_planner_fallback_completes_with_observation() -> None:
    planner = OpenClawPlanner(backend=BrokenBackend())
    turn_input = PlannerTurnInput(
        session_id="s3",
        user_query="list files",
        history=[{"role": "observation", "content": "files: a.txt"}],
        candidates=[_candidate()],
        turn_index=2,
    )

    decision = asyncio.run(planner.plan(turn_input))

    assert decision.action == "complete"
    assert "files:" in (decision.answer or "")


def test_openclaw_planner_fallback_retries_on_error_observation() -> None:
    planner = OpenClawPlanner(backend=BrokenBackend())
    turn_input = PlannerTurnInput(
        session_id="s4",
        user_query="list files",
        history=[{"role": "observation", "content": "Execution error: filesystem unavailable"}],
        candidates=[_candidate()],
        turn_index=2,
    )

    decision = asyncio.run(planner.plan(turn_input))

    assert decision.action == "complete"
    assert "fail-closed mode" in (decision.answer or "")


def test_openclaw_planner_fallback_retry_requires_explicit_opt_in() -> None:
    planner = OpenClawPlanner(backend=BrokenBackend(), fallback_allows_tool_retry=True)
    turn_input = PlannerTurnInput(
        session_id="s4-opt-in",
        user_query="list files",
        history=[{"role": "observation", "content": "Execution error: filesystem unavailable"}],
        candidates=[_candidate()],
        turn_index=2,
    )

    decision = asyncio.run(planner.plan(turn_input))

    assert decision.action == "call_tool"
    assert decision.server_name == "filesystem"
    assert decision.tool_name == "list_directory"


class _Result:
    def __init__(self) -> None:
        self.status = "complete"
        self.answer = "ok"


class _OrchestratorStub:
    def __init__(self) -> None:
        self.calls = 0

    async def run(self, session_id: str, user_query: str) -> _Result:
        del session_id, user_query
        self.calls += 1
        return _Result()


def test_realtime_service_runs_on_startup(tmp_path: Path) -> None:
    tracker = WorkspaceChangeTracker(root=tmp_path)
    (tmp_path / "main.py").write_text("print('x')", encoding="utf-8")

    orchestrator = _OrchestratorStub()
    service = RealTimeHybridService(
        orchestrator=orchestrator,
        tracker=tracker,
        query_builder=lambda touched: f"changed={len(touched)}",
        poll_interval_s=0.01,
        run_on_startup=True,
    )

    asyncio.run(service.run_for_cycles(1))
    assert orchestrator.calls == 1


def test_event_bus_isolates_failing_handlers() -> None:
    bus = EnterpriseEventBus()
    events: list[str] = []

    def _failing_handler(event: dict[str, object]) -> None:
        del event
        raise RuntimeError("subscriber broke")

    def _recording_handler(event: dict[str, object]) -> None:
        events.append(str(event.get("type", "")))

    asyncio.run(bus.subscribe(_failing_handler))
    asyncio.run(bus.subscribe(_recording_handler))

    asyncio.run(bus.publish({"type": "runtime_event"}))

    assert "runtime_event" in events
    assert bus.recent_errors()


class _StaticRegistry:
    async def route(self, query: str, k: int, min_score: float) -> list[ToolCandidate]:
        del query, k, min_score
        return [_candidate()]


class _RepeatingPlanner:
    async def plan(self, turn_input: PlannerTurnInput) -> PlannerDecision:
        del turn_input
        return PlannerDecision(
            action="call_tool",
            thought="repeat",
            server_name="filesystem",
            tool_name="list_directory",
            arguments={"path": "."},
        )


class _ExecutorStub:
    async def execute(self, candidate: ToolCandidate, arguments: dict[str, object]) -> tuple[dict[str, object], str]:
        del candidate, arguments
        return {}, "files: a.txt"


def test_orchestrator_loop_guard_fails_on_repeated_calls() -> None:
    orchestrator = HybridEnterpriseOrchestrator(
        registry=_StaticRegistry(),
        planner=_RepeatingPlanner(),
        executor=_ExecutorStub(),
        policy_engine=PolicyEngine(ToolPolicy(allowed_servers={"filesystem"})),
        config=EnterpriseConfig(max_turns=8),
    )

    result = asyncio.run(orchestrator.run(session_id="loop-guard", user_query="list files"))

    assert result.status == "failed"
    assert result.turns == 2
    counters = result.telemetry.get("counters", {})
    assert counters.get("loop_guard_failed") == 1


def test_telemetry_merge_snapshot() -> None:
    telemetry = TelemetryCollector()
    telemetry.increment("pipeline_fallback")
    telemetry.record_latency("pipeline_routing", 10.0)

    telemetry.merge_snapshot(
        {
            "counters": {"complete": 1},
            "latencies_ms": {
                "pipeline_routing": {"count": 2, "min": 5.0, "max": 15.0, "mean": 10.0}
            },
        }
    )

    snapshot = telemetry.to_dict()
    counters = snapshot.get("counters", {})
    latencies = snapshot.get("latencies_ms", {})

    assert counters.get("pipeline_fallback") == 1
    assert counters.get("complete") == 1
    assert isinstance(latencies.get("pipeline_routing"), dict)
    assert latencies["pipeline_routing"]["count"] == 3


def test_telemetry_persist_snapshot(tmp_path: Path) -> None:
    sink = tmp_path / "telemetry.jsonl"
    telemetry = TelemetryCollector(sink_path=str(sink))
    telemetry.increment("complete")

    telemetry.persist_snapshot({"session_id": "s1", "status": "complete"})

    persisted = sink.read_text(encoding="utf-8")
    assert "session_id" in persisted
    assert "telemetry" in persisted

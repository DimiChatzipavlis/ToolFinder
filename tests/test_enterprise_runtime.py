from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from Enterprise.runtime.contracts import PlannerDecision, PlannerTurnInput, ToolCandidate
from Enterprise.runtime.orchestrator import HybridEnterpriseOrchestrator
from Enterprise.runtime.planner import HeuristicPlanner, OpenClawPlanner
from Enterprise.runtime.policy import PolicyEngine, PolicyViolation, ToolPolicy
from Enterprise.runtime.realtime_service import RealTimeHybridService, WorkspaceChangeTracker
from Enterprise.runtime.config import EnterpriseConfig


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

    assert decision.action == "call_tool"
    assert decision.server_name == "filesystem"
    assert decision.tool_name == "list_directory"


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


def test_orchestrator_loop_guard_completes_on_repeated_calls() -> None:
    orchestrator = HybridEnterpriseOrchestrator(
        registry=_StaticRegistry(),
        planner=_RepeatingPlanner(),
        executor=_ExecutorStub(),
        policy_engine=PolicyEngine(ToolPolicy(allowed_servers={"filesystem"})),
        config=EnterpriseConfig(max_turns=8),
    )

    result = asyncio.run(orchestrator.run(session_id="loop-guard", user_query="list files"))

    assert result.status == "complete"
    assert result.turns == 2
    counters = result.telemetry.get("counters", {})
    assert counters.get("loop_guard_complete") == 1

from __future__ import annotations

import asyncio
import json

import pytest

from Enterprise.runtime.contracts import (
    HybridPipelineResult,
    OpenClawAgentRequest,
    OpenClawAgentResponse,
    PipelinePhase,
    ToolCallRecord,
    ToolCandidate,
)
from Enterprise.runtime.config import EnterpriseConfig
from Enterprise.runtime.event_bus import EnterpriseEventBus
from Enterprise.runtime.executor import HybridToolExecutor
from Enterprise.runtime.openclaw_hybrid_pipeline import (
    FallbackStrategy,
    OpenClawHybridPipeline,
    OpenClawSessionDriver,
    OpenClawToolManifest,
)
from Enterprise.runtime.orchestrator import HybridEnterpriseOrchestrator
from Enterprise.runtime.planner import HeuristicPlanner
from Enterprise.runtime.policy import PolicyEngine, ToolPolicy
from Enterprise.runtime.registry import HybridToolRegistry


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _candidate(
    server: str = "filesystem",
    tool: str = "list_directory",
    desc: str = "List files",
    score: float = 0.9,
) -> ToolCandidate:
    return ToolCandidate(
        server_name=server,
        tool_name=tool,
        description=desc,
        input_schema={
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
        score=score,
    )


class _MockExecutorClient:
    async def call_tool(self, tool_name: str, arguments: dict) -> dict:
        return {"content": [{"type": "text", "text": f"{tool_name} result"}]}


class _SuccessBackend:
    """Backend that returns a valid OpenClaw agent completion."""

    def __init__(self, answer: str = "Done.") -> None:
        self._answer = answer

    async def complete(self, prompt: str) -> str:
        return json.dumps({"thought": "completed", "status": "complete", "answer": self._answer})


class _ToolCallBackend:
    """Backend that returns tool calls then completion."""

    async def complete(self, prompt: str) -> str:
        return json.dumps([
            {
                "thought": "need to list files",
                "action": "call_tool",
                "tool": "filesystem/list_directory",
                "arguments": {"path": "."},
            },
            {
                "thought": "got result",
                "status": "complete",
                "answer": "Found files: README.md",
            },
        ])


class _MemoryToolCallBackend:
    async def complete(self, prompt: str) -> str:
        return json.dumps([
            {
                "thought": "search memory",
                "action": "call_tool",
                "tool": "memory/search_nodes",
                "arguments": {"query": "files"},
            },
            {
                "thought": "done",
                "status": "complete",
                "answer": "Memory searched.",
            },
        ])


class _FailingBackend:
    """Backend that always raises an error."""

    async def complete(self, prompt: str) -> str:
        raise RuntimeError("backend unavailable")


class _EmptyBackend:
    """Backend that returns empty output."""

    async def complete(self, prompt: str) -> str:
        return ""


class _StaticRegistry:
    """Registry that always returns fixed candidates."""

    def __init__(self, candidates: list[ToolCandidate] | None = None) -> None:
        self._candidates = candidates or [_candidate()]

    async def route(self, query: str, k: int, min_score: float) -> list[ToolCandidate]:
        return self._candidates


class _EmptyRegistry:
    """Registry that returns no candidates."""

    async def route(self, query: str, k: int, min_score: float) -> list[ToolCandidate]:
        return []


# ---------------------------------------------------------------------------
# Tests: OpenClawToolManifest
# ---------------------------------------------------------------------------


def test_manifest_from_candidates() -> None:
    candidates = [
        _candidate("fs", "list_dir", "List files", 0.95),
        _candidate("memory", "search", "Search nodes", 0.80),
    ]
    manifest = OpenClawToolManifest.from_candidates(candidates)

    assert len(manifest) == 2
    assert manifest[0]["name"] == "fs/list_dir"
    assert manifest[0]["description"] == "List files"
    assert manifest[0]["metadata"]["routing_score"] == 0.95
    assert manifest[1]["name"] == "memory/search"


def test_manifest_render_json_is_valid() -> None:
    candidates = [_candidate()]
    manifest = OpenClawToolManifest.from_candidates(candidates)
    rendered = OpenClawToolManifest.render_json(manifest)

    parsed = json.loads(rendered)
    assert isinstance(parsed, list)
    assert len(parsed) == 1


def test_manifest_empty_candidates() -> None:
    manifest = OpenClawToolManifest.from_candidates([])
    assert manifest == []
    assert OpenClawToolManifest.render_json(manifest) == "[]"


# ---------------------------------------------------------------------------
# Tests: OpenClawSessionDriver parsing
# ---------------------------------------------------------------------------


def test_session_driver_parse_completion() -> None:
    driver = OpenClawSessionDriver(backend=_SuccessBackend())
    result = asyncio.run(
        driver.run_agent(
            OpenClawAgentRequest(query="test", tool_manifest=[], session_id="s1")
        )
    )
    assert result.success is True
    assert result.answer == "Done."
    assert result.tool_calls == []


def test_session_driver_parse_tool_calls_array() -> None:
    driver = OpenClawSessionDriver(backend=_ToolCallBackend())
    result = asyncio.run(
        driver.run_agent(
            OpenClawAgentRequest(query="list files", tool_manifest=[], session_id="s2")
        )
    )
    assert result.success is True
    assert result.answer == "Found files: README.md"
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0]["tool"] == "filesystem/list_directory"


def test_session_driver_handles_backend_error() -> None:
    driver = OpenClawSessionDriver(backend=_FailingBackend())
    result = asyncio.run(
        driver.run_agent(
            OpenClawAgentRequest(query="test", tool_manifest=[], session_id="s3")
        )
    )
    assert result.success is False
    assert "failed" in result.error.lower()


def test_session_driver_handles_empty_output() -> None:
    driver = OpenClawSessionDriver(backend=_EmptyBackend())
    result = asyncio.run(
        driver.run_agent(
            OpenClawAgentRequest(query="test", tool_manifest=[], session_id="s4")
        )
    )
    assert result.success is False
    assert "empty" in result.error.lower()


def test_session_driver_raw_text_is_rejected() -> None:
    class _TextBackend:
        async def complete(self, prompt: str) -> str:
            return "The answer is 42."

    driver = OpenClawSessionDriver(backend=_TextBackend())
    result = asyncio.run(
        driver.run_agent(
            OpenClawAgentRequest(query="test", tool_manifest=[], session_id="s5")
        )
    )
    assert result.success is False
    assert "structured parsing failed" in (result.error or "")


# ---------------------------------------------------------------------------
# Tests: FallbackStrategy
# ---------------------------------------------------------------------------


def test_fallback_strategy_validates() -> None:
    assert FallbackStrategy.validate("error") == "error"
    assert FallbackStrategy.validate("heuristic_planner") == "heuristic_planner"
    assert FallbackStrategy.validate("best_effort") == "best_effort"

    with pytest.raises(ValueError):
        FallbackStrategy.validate("invalid")


# ---------------------------------------------------------------------------
# Tests: OpenClawHybridPipeline
# ---------------------------------------------------------------------------


def test_pipeline_no_route_returns_failed() -> None:
    driver = OpenClawSessionDriver(backend=_SuccessBackend())
    pipeline = OpenClawHybridPipeline(
        registry=_EmptyRegistry(),
        session_driver=driver,
    )
    result = asyncio.run(pipeline.run("session-1", "unknown query"))

    assert result.status == "failed"
    assert "No tools" in result.answer
    assert result.execution_path == "direct"
    assert PipelinePhase.ROUTING.value in result.phase_trace


def test_pipeline_openclaw_success_path() -> None:
    driver = OpenClawSessionDriver(backend=_SuccessBackend("All files listed."))
    pipeline = OpenClawHybridPipeline(
        registry=_StaticRegistry(),
        session_driver=driver,
    )
    result = asyncio.run(pipeline.run("session-2", "list files"))

    assert result.status == "complete"
    assert result.answer == "All files listed."
    assert result.execution_path == "openclaw"
    assert PipelinePhase.OPENCLAW_AGENT.value in result.phase_trace
    assert result.openclaw_response is not None
    assert result.openclaw_response.success is True


def test_pipeline_openclaw_with_tool_execution() -> None:
    driver = OpenClawSessionDriver(backend=_ToolCallBackend())
    clients = {"filesystem": _MockExecutorClient()}
    executor = HybridToolExecutor(clients)

    pipeline = OpenClawHybridPipeline(
        registry=_StaticRegistry(),
        session_driver=driver,
        executor=executor,
    )
    result = asyncio.run(pipeline.run("session-3", "list files"))

    assert result.status == "complete"
    assert result.execution_path == "openclaw"
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].server_name == "filesystem"
    assert result.tool_calls[0].tool_name == "list_directory"
    assert "list_directory result" in result.tool_calls[0].observation


def test_pipeline_policy_enforced_during_openclaw_tool_execution() -> None:
    driver = OpenClawSessionDriver(backend=_MemoryToolCallBackend())
    clients = {"memory": _MockExecutorClient()}
    executor = HybridToolExecutor(clients)

    pipeline = OpenClawHybridPipeline(
        registry=_StaticRegistry([_candidate("memory", "search_nodes")]),
        session_driver=driver,
        executor=executor,
        policy_engine=PolicyEngine(ToolPolicy(allowed_servers={"filesystem"})),
        fallback_strategy=FallbackStrategy.ERROR,
    )
    result = asyncio.run(pipeline.run("session-policy", "search memory"))

    assert result.status == "failed"
    assert result.execution_path == "openclaw"
    assert "policy violation" in result.answer.lower()


def test_pipeline_fallback_on_openclaw_failure() -> None:
    driver = OpenClawSessionDriver(backend=_FailingBackend())
    clients = {"filesystem": _MockExecutorClient()}
    executor = HybridToolExecutor(clients)

    registry = HybridToolRegistry()
    asyncio.run(
        registry.upsert_server_tools(
            "filesystem",
            [
                {
                    "tool_name": "list_directory",
                    "description": "List files in a directory",
                    "inputSchema": {
                        "type": "object",
                        "properties": {"path": {"type": "string"}},
                        "required": ["path"],
                    },
                }
            ],
        )
    )

    fallback_orchestrator = HybridEnterpriseOrchestrator(
        registry=registry,
        planner=HeuristicPlanner(),
        executor=executor,
        policy_engine=PolicyEngine(ToolPolicy(allowed_servers={"filesystem"})),
        config=EnterpriseConfig(top_k=2, min_score=0.01, max_turns=4),
    )

    pipeline = OpenClawHybridPipeline(
        registry=registry,
        session_driver=driver,
        fallback_orchestrator=fallback_orchestrator,
        executor=executor,
        config=EnterpriseConfig(top_k=2, min_score=0.01),
        fallback_strategy=FallbackStrategy.HEURISTIC_PLANNER,
    )
    result = asyncio.run(pipeline.run("session-4", "list files"))

    assert result.execution_path == "fallback"
    assert result.status == "degraded_fallback"
    assert result.fallback_triggered is True
    assert PipelinePhase.FALLBACK.value in result.phase_trace
    assert result.openclaw_response is not None
    assert result.openclaw_response.success is False
    telemetry_counters = result.telemetry.get("counters", {})
    assert telemetry_counters.get("pipeline_fallback_triggered", 0) >= 1


def test_pipeline_error_strategy_returns_failed() -> None:
    driver = OpenClawSessionDriver(backend=_FailingBackend())
    pipeline = OpenClawHybridPipeline(
        registry=_StaticRegistry(),
        session_driver=driver,
        fallback_strategy=FallbackStrategy.ERROR,
    )
    result = asyncio.run(pipeline.run("session-5", "list files"))

    assert result.status == "failed"
    assert result.execution_path == "openclaw"
    assert "failed" in result.answer.lower()


def test_pipeline_best_effort_returns_partial() -> None:
    driver = OpenClawSessionDriver(backend=_EmptyBackend())
    pipeline = OpenClawHybridPipeline(
        registry=_StaticRegistry(),
        session_driver=driver,
        fallback_strategy=FallbackStrategy.BEST_EFFORT,
    )
    result = asyncio.run(pipeline.run("session-6", "list files"))

    assert result.status == "partial"
    assert result.execution_path == "openclaw"


def test_pipeline_phase_trace_complete() -> None:
    driver = OpenClawSessionDriver(backend=_SuccessBackend("ok"))
    pipeline = OpenClawHybridPipeline(
        registry=_StaticRegistry(),
        session_driver=driver,
    )
    result = asyncio.run(pipeline.run("session-7", "test"))

    assert result.phase_trace == [
        PipelinePhase.ROUTING.value,
        PipelinePhase.OPENCLAW_AGENT.value,
        PipelinePhase.COMPLETE.value,
    ]


def test_pipeline_telemetry_records_latencies() -> None:
    driver = OpenClawSessionDriver(backend=_SuccessBackend("ok"))
    pipeline = OpenClawHybridPipeline(
        registry=_StaticRegistry(),
        session_driver=driver,
    )
    result = asyncio.run(pipeline.run("session-8", "test"))

    latencies = result.telemetry.get("latencies_ms", {})
    assert "pipeline_routing" in latencies
    assert "pipeline_openclaw_agent" in latencies


def test_pipeline_event_bus_fires_events() -> None:
    driver = OpenClawSessionDriver(backend=_SuccessBackend("ok"))
    events: list[dict] = []

    bus = EnterpriseEventBus()
    bus.subscribe(lambda e: events.append(e))

    pipeline = OpenClawHybridPipeline(
        registry=_StaticRegistry(),
        session_driver=driver,
        event_bus=bus,
    )
    asyncio.run(pipeline.run("session-9", "test"))

    event_types = [e["type"] for e in events]
    assert "pipeline_routing" in event_types
    assert "pipeline_openclaw_agent" in event_types

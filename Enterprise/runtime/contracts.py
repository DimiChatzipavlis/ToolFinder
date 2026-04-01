from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any


JsonDict = dict[str, Any]


@dataclass(frozen=True)
class ToolCandidate:
    server_name: str
    tool_name: str
    description: str
    input_schema: JsonDict
    score: float


@dataclass(frozen=True)
class PlannerTurnInput:
    session_id: str
    user_query: str
    history: list[JsonDict]
    candidates: list[ToolCandidate]
    turn_index: int


@dataclass(frozen=True)
class PlannerDecision:
    action: str
    thought: str
    server_name: str | None = None
    tool_name: str | None = None
    arguments: JsonDict = field(default_factory=dict)
    answer: str | None = None


@dataclass(frozen=True)
class ToolCallRecord:
    turn_index: int
    server_name: str
    tool_name: str
    arguments: JsonDict
    observation: str


@dataclass(frozen=True)
class SessionResult:
    status: str
    answer: str
    turns: int
    tool_calls: list[ToolCallRecord]
    telemetry: JsonDict
    final_history: list[JsonDict]


# ---------------------------------------------------------------------------
# End-to-end hybrid pipeline contracts
# ---------------------------------------------------------------------------


class PipelinePhase(enum.Enum):
    """Tracks the current execution phase of the hybrid pipeline."""

    ROUTING = "routing"
    OPENCLAW_AGENT = "openclaw_agent"
    FALLBACK = "fallback"
    COMPLETE = "complete"


@dataclass(frozen=True)
class OpenClawAgentRequest:
    """Input payload sent to the openclaw agent session."""

    query: str
    tool_manifest: list[JsonDict]
    session_id: str
    model: str = ""
    max_steps: int = 10
    extra: JsonDict = field(default_factory=dict)


@dataclass(frozen=True)
class OpenClawAgentResponse:
    """Parsed output coming back from an openclaw agent session."""

    answer: str
    tool_calls: list[JsonDict]
    raw_output: str
    success: bool
    error: str | None = None
    metadata: JsonDict = field(default_factory=dict)


@dataclass(frozen=True)
class HybridPipelineResult:
    """Unified result from the end-to-end hybrid pipeline."""

    status: str
    answer: str
    turns: int
    tool_calls: list[ToolCallRecord]
    telemetry: JsonDict
    final_history: list[JsonDict]
    phase_trace: list[str]
    execution_path: str  # "openclaw" | "fallback" | "direct"
    openclaw_response: OpenClawAgentResponse | None = None
    fallback_triggered: bool = False


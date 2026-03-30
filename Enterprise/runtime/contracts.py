from __future__ import annotations

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

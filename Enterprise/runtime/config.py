from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class EnterpriseConfig:
    model_name: str = "sentence-transformers/all-mpnet-base-v2"
    top_k: int = 3
    min_score: float = 0.15
    max_turns: int = 8
    planner_timeout_s: float = 45.0
    planner_fallback_on_parse_error: bool = True
    planner_fallback_allows_tool_retry: bool = False
    max_observation_chars: int = 1200
    max_retries: int = 2
    route_with_history_context: bool = True
    loop_guard_repetition_limit: int = 2
    loop_guard_marks_complete: bool = False
    allow_unrouted_tool_calls: bool = False
    allow_keyword_low_confidence_fallback: bool = False
    telemetry_max_latency_samples: int = 500
    event_bus_max_errors: int = 200
    realtime_run_on_startup: bool = True
    realtime_error_backoff_s: float = 3.0
    telemetry_sink_path: str = ""
    telemetry_allowed_root: str = ""

    def __post_init__(self) -> None:
        if self.top_k < 1:
            raise ValueError("top_k must be >= 1")
        if self.max_turns < 1:
            raise ValueError("max_turns must be >= 1")
        if not (0.0 <= self.min_score <= 1.0):
            raise ValueError("min_score must be between 0.0 and 1.0")
        if self.planner_timeout_s <= 0:
            raise ValueError("planner_timeout_s must be > 0")
        if self.loop_guard_repetition_limit < 2:
            raise ValueError("loop_guard_repetition_limit must be >= 2")
        if self.max_observation_chars < 100:
            raise ValueError("max_observation_chars must be >= 100")
        if self.max_retries < 0:
            raise ValueError("max_retries must be >= 0")
        if self.telemetry_max_latency_samples < 1:
            raise ValueError("telemetry_max_latency_samples must be >= 1")
        if self.event_bus_max_errors < 1:
            raise ValueError("event_bus_max_errors must be >= 1")
        if self.realtime_error_backoff_s < 0:
            raise ValueError("realtime_error_backoff_s must be >= 0")

    @staticmethod
    def from_env() -> "EnterpriseConfig":
        return EnterpriseConfig(
            model_name=os.getenv("ENTERPRISE_EMBEDDING_MODEL", "sentence-transformers/all-mpnet-base-v2"),
            top_k=int(os.getenv("ENTERPRISE_TOP_K", "3")),
            min_score=float(os.getenv("ENTERPRISE_MIN_SCORE", "0.15")),
            max_turns=int(os.getenv("ENTERPRISE_MAX_TURNS", "8")),
            planner_timeout_s=float(os.getenv("ENTERPRISE_PLANNER_TIMEOUT_S", "45")),
            planner_fallback_on_parse_error=os.getenv("ENTERPRISE_PLANNER_FALLBACK_ON_PARSE_ERROR", "1") == "1",
            planner_fallback_allows_tool_retry=os.getenv("ENTERPRISE_PLANNER_FALLBACK_TOOL_RETRY", "0") == "1",
            max_observation_chars=int(os.getenv("ENTERPRISE_MAX_OBS_CHARS", "1200")),
            max_retries=int(os.getenv("ENTERPRISE_MAX_RETRIES", "2")),
            route_with_history_context=os.getenv("ENTERPRISE_ROUTE_WITH_HISTORY_CONTEXT", "1") == "1",
            loop_guard_repetition_limit=int(os.getenv("ENTERPRISE_LOOP_GUARD_REPETITION_LIMIT", "2")),
            loop_guard_marks_complete=os.getenv("ENTERPRISE_LOOP_GUARD_MARKS_COMPLETE", "0") == "1",
            allow_unrouted_tool_calls=os.getenv("ENTERPRISE_ALLOW_UNROUTED", "0") == "1",
            allow_keyword_low_confidence_fallback=os.getenv("ENTERPRISE_ALLOW_KEYWORD_LOW_CONFIDENCE_FALLBACK", "0") == "1",
            telemetry_max_latency_samples=int(os.getenv("ENTERPRISE_TELEMETRY_MAX_LATENCY_SAMPLES", "500")),
            event_bus_max_errors=int(os.getenv("ENTERPRISE_EVENT_BUS_MAX_ERRORS", "200")),
            realtime_run_on_startup=os.getenv("ENTERPRISE_REALTIME_RUN_ON_STARTUP", "1") == "1",
            realtime_error_backoff_s=float(os.getenv("ENTERPRISE_REALTIME_ERROR_BACKOFF_S", "3")),
            telemetry_sink_path=os.getenv("ENTERPRISE_TELEMETRY_SINK", ""),
            telemetry_allowed_root=os.getenv("ENTERPRISE_TELEMETRY_ALLOWED_ROOT", ""),
        )

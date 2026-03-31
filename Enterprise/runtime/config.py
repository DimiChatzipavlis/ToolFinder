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
    max_observation_chars: int = 1200
    max_retries: int = 2
    allow_unrouted_tool_calls: bool = False
    realtime_run_on_startup: bool = True
    realtime_error_backoff_s: float = 3.0
    telemetry_sink_path: str = ""

    def __post_init__(self) -> None:
        if self.top_k < 1:
            raise ValueError("top_k must be >= 1")
        if self.max_turns < 1:
            raise ValueError("max_turns must be >= 1")
        if self.max_observation_chars < 100:
            raise ValueError("max_observation_chars must be >= 100")
        if self.max_retries < 0:
            raise ValueError("max_retries must be >= 0")
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
            max_observation_chars=int(os.getenv("ENTERPRISE_MAX_OBS_CHARS", "1200")),
            max_retries=int(os.getenv("ENTERPRISE_MAX_RETRIES", "2")),
            allow_unrouted_tool_calls=os.getenv("ENTERPRISE_ALLOW_UNROUTED", "0") == "1",
            realtime_run_on_startup=os.getenv("ENTERPRISE_REALTIME_RUN_ON_STARTUP", "1") == "1",
            realtime_error_backoff_s=float(os.getenv("ENTERPRISE_REALTIME_ERROR_BACKOFF_S", "3")),
            telemetry_sink_path=os.getenv("ENTERPRISE_TELEMETRY_SINK", ""),
        )

from __future__ import annotations

import json
from dataclasses import dataclass, field

from .contracts import PlannerDecision, ToolCandidate


class PolicyViolation(RuntimeError):
    pass


@dataclass(frozen=True)
class ToolPolicy:
    allowed_servers: set[str] | None = None
    denied_tool_pairs: set[tuple[str, str]] = field(default_factory=set)
    max_argument_bytes: int = 8192


class PolicyEngine:
    def __init__(self, policy: ToolPolicy | None = None) -> None:
        self.policy = policy or ToolPolicy()

    def enforce_call(
        self,
        decision: PlannerDecision,
        candidate_lookup: dict[tuple[str, str], ToolCandidate],
        allow_unrouted_tool_calls: bool = False,
    ) -> None:
        if decision.server_name is None or decision.tool_name is None:
            raise PolicyViolation("planner decision missing server_name/tool_name")

        key = (decision.server_name, decision.tool_name)
        if self.policy.allowed_servers is not None and decision.server_name not in self.policy.allowed_servers:
            raise PolicyViolation(f"server not allowed by policy: {decision.server_name}")

        if key in self.policy.denied_tool_pairs:
            raise PolicyViolation(f"tool denied by policy: {decision.server_name}/{decision.tool_name}")

        if not allow_unrouted_tool_calls and key not in candidate_lookup:
            raise PolicyViolation(
                "planner selected a tool outside routed candidates: "
                f"{decision.server_name}/{decision.tool_name}"
            )

        encoded = json.dumps(decision.arguments or {}, ensure_ascii=True, sort_keys=True)
        if len(encoded.encode("utf-8")) > self.policy.max_argument_bytes:
            raise PolicyViolation(
                f"argument payload exceeds policy limit ({self.policy.max_argument_bytes} bytes)"
            )

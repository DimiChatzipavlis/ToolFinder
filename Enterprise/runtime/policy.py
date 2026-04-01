from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

from .contracts import PlannerDecision, ToolCandidate


class PolicyViolation(RuntimeError):
    pass


class SecurityPolicyViolation(Exception):
    pass


@dataclass(frozen=True)
class ToolPolicy:
    allowed_servers: set[str] | None = None
    denied_tool_pairs: set[tuple[str, str]] = field(default_factory=set)
    max_argument_bytes: int = 8192
    max_string_argument_chars: int = 4096
    max_collection_items: int = 256
    deny_parent_path_segments: bool = True
    allowed_path_roots: tuple[str, ...] = ()
    path_argument_keys: set[str] = field(
        default_factory=lambda: {
            "path",
            "paths",
            "file",
            "filepath",
            "source",
            "destination",
            "dir",
            "directory",
            "target_path",
        }
    )


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

        self._enforce_argument_guardrails(decision.arguments or {})

    def _enforce_argument_guardrails(self, arguments: dict[str, object]) -> None:
        def walk(node: object, parent_key: str = "") -> None:
            if isinstance(node, dict):
                if len(node) > self.policy.max_collection_items:
                    raise PolicyViolation(
                        f"argument object exceeds item limit ({self.policy.max_collection_items})"
                    )
                for key, value in node.items():
                    walk(value, str(key))
                return

            if isinstance(node, list):
                if len(node) > self.policy.max_collection_items:
                    raise PolicyViolation(
                        f"argument list exceeds item limit ({self.policy.max_collection_items})"
                    )
                for item in node:
                    walk(item, parent_key)
                return

            if isinstance(node, str):
                if len(node) > self.policy.max_string_argument_chars:
                    raise PolicyViolation(
                        "string argument exceeds policy limit "
                        f"({self.policy.max_string_argument_chars} chars)"
                    )
                if self._looks_like_path(parent_key):
                    self._enforce_path_safety(node, parent_key)

        walk(arguments)

    def _looks_like_path(self, key: str) -> bool:
        normalized = key.strip().lower()
        if not normalized:
            return False
        if normalized in self.policy.path_argument_keys:
            return True
        return normalized.endswith("_path") or normalized.endswith("_file")

    def _enforce_path_safety(self, value: str, key: str) -> None:
        if "\x00" in value:
            raise SecurityPolicyViolation(f"path-like argument contains null byte: {key}")

        if not self.policy.deny_parent_path_segments:
            self._enforce_allowed_path_roots(value, key)
            return

        normalized = value.replace("\\", "/")
        segments = [segment for segment in normalized.split("/") if segment and segment != "."]
        if any(segment == ".." for segment in segments):
            raise SecurityPolicyViolation(
                f"path-like argument traverses outside allowed roots via parent segments: {key}"
            )

        self._enforce_allowed_path_roots(value, key)

    def _enforce_allowed_path_roots(self, value: str, key: str) -> None:
        if not self.policy.allowed_path_roots:
            return

        if not self._is_absolute_path(value):
            return

        candidate = os.path.normpath(os.path.abspath(value))
        allowed_roots = [
            os.path.normpath(os.path.abspath(root))
            for root in self.policy.allowed_path_roots
        ]

        for root in allowed_roots:
            if self._is_within_root(candidate, root):
                return

        raise SecurityPolicyViolation(
            f"path-like argument traverses outside allowed roots: {key}={value}"
        )

    @staticmethod
    def _is_absolute_path(value: str) -> bool:
        normalized = value.replace("\\", "/")
        if len(normalized) >= 2 and normalized[1] == ":":
            return True
        return normalized.startswith("/") or normalized.startswith("\\")

    @staticmethod
    def _is_within_root(candidate: str, root: str) -> bool:
        try:
            return os.path.commonpath([candidate, root]) == root
        except ValueError:
            return False

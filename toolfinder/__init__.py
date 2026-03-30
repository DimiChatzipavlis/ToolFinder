from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .autonomous_agent import AutonomousMCPAgent
    from .dynamic_faiss_router import RouteNotFoundError, UniversalMCPRouter
    from .mcp_adapter import DynamicMCPClient

__all__ = [
    "AutonomousMCPAgent",
    "DynamicMCPClient",
    "RouteNotFoundError",
    "UniversalMCPRouter",
]


def __getattr__(name: str) -> Any:
    if name == "AutonomousMCPAgent":
        from .autonomous_agent import AutonomousMCPAgent as _AutonomousMCPAgent

        return _AutonomousMCPAgent
    if name == "DynamicMCPClient":
        from .mcp_adapter import DynamicMCPClient as _DynamicMCPClient

        return _DynamicMCPClient
    if name == "RouteNotFoundError":
        from .dynamic_faiss_router import RouteNotFoundError as _RouteNotFoundError

        return _RouteNotFoundError
    if name == "UniversalMCPRouter":
        from .dynamic_faiss_router import UniversalMCPRouter as _UniversalMCPRouter

        return _UniversalMCPRouter
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
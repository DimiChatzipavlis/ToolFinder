from __future__ import annotations

import pytest

from toolfinder import autonomous_agent as agent_module


class EmptyRouter:
    def __init__(self, model_name: str = "dummy") -> None:
        self.model_name = model_name

    def route_top_k(self, query: str, k: int = 5):
        del query, k
        return []


@pytest.mark.asyncio
async def test_execute_task_handles_empty_routing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(agent_module, "UniversalMCPRouter", EmptyRouter)

    agent = agent_module.AutonomousMCPAgent(model_name="dummy", max_iterations=1)
    result = await agent.execute_task("do something")

    assert result.status == "failed"
    assert len(result.steps) == 1
    assert result.steps[0].action == "no_route"


def test_max_iterations_clamps_to_one(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(agent_module, "UniversalMCPRouter", EmptyRouter)

    agent = agent_module.AutonomousMCPAgent(model_name="dummy", max_iterations=0)

    assert agent.max_iterations == 1

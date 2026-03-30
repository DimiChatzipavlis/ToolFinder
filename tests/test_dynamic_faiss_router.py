from __future__ import annotations

import numpy as np
import pytest

from toolfinder import dynamic_faiss_router as router_module


class DummySentenceTransformer:
    def __init__(self, model_name: str, device: str | None = None) -> None:
        self.model_name = model_name
        self.device = device

    def get_sentence_embedding_dimension(self) -> int:
        return 4

    def encode(self, texts, batch_size: int | None = None, convert_to_numpy: bool = True):
        del batch_size, convert_to_numpy
        embeddings = []
        for text in texts:
            if '"tool_name":"alpha_tool"' in text:
                embeddings.append([1.0, 0.0, 0.0, 0.0])
            elif '"tool_name":"beta_tool"' in text:
                embeddings.append([0.0, 1.0, 0.0, 0.0])
            elif "alpha query" in text:
                embeddings.append([1.0, 0.0, 0.0, 0.0])
            else:
                embeddings.append([0.0, 0.0, 1.0, 0.0])
        return np.asarray(embeddings, dtype=np.float32)


def build_router(monkeypatch: pytest.MonkeyPatch) -> router_module.UniversalMCPRouter:
    monkeypatch.setattr(router_module, "SentenceTransformer", DummySentenceTransformer)
    router = router_module.UniversalMCPRouter(model_name="dummy")
    router.ingest_server(
        "test-server",
        [
            {
                "tool_name": "alpha_tool",
                "description": "Tool for alpha query",
                "inputSchema": {"type": "object", "properties": {"x": {"type": "string"}}},
            },
            {
                "tool_name": "beta_tool",
                "description": "Tool for beta query",
                "inputSchema": {"type": "object", "properties": {"y": {"type": "string"}}},
            },
        ],
    )
    return router


def test_route_returns_best_match(monkeypatch: pytest.MonkeyPatch) -> None:
    router = build_router(monkeypatch)

    result = router.route("alpha query")

    assert result.tool_name == "alpha_tool"
    assert result.server_name == "test-server"


def test_route_raises_when_no_match_above_threshold(monkeypatch: pytest.MonkeyPatch) -> None:
    router = build_router(monkeypatch)

    with pytest.raises(router_module.RouteNotFoundError):
        router.route("unrelated request")

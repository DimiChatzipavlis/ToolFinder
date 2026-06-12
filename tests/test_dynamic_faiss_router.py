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


def test_default_index_is_exact_flat(monkeypatch: pytest.MonkeyPatch) -> None:
    router = build_router(monkeypatch)

    assert isinstance(router.faiss_index, router_module.faiss.IndexFlatIP)


def test_explicit_hnsw_index_is_honored(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(router_module, "SentenceTransformer", DummySentenceTransformer)
    config = router_module.RouterHyperparameters(index_type="hnsw")
    router = router_module.UniversalMCPRouter(model_name="dummy", config=config)

    assert isinstance(router.faiss_index, router_module.faiss.IndexHNSWFlat)


def test_route_top_k_is_type_stable_after_build_index(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(router_module, "SentenceTransformer", DummySentenceTransformer)
    router = router_module.UniversalMCPRouter(model_name="dummy")
    router.add_tool(
        {
            "name": "alpha_tool",
            "description": "Tool for alpha query",
            "inputSchema": {"type": "object", "properties": {"x": {"type": "string"}}},
        },
        server_name="test-server",
    )
    router.build_index()

    results = router.route_top_k("alpha query", k=1)

    assert all(isinstance(item, router_module.RouteResult) for item in results)
    assert results[0].tool_name == "alpha_tool"


def test_to_openai_tools_formats_bindable_schema(monkeypatch: pytest.MonkeyPatch) -> None:
    router = build_router(monkeypatch)

    bindable = router_module.to_openai_tools(router.route_top_k("alpha query", k=1))

    assert bindable[0]["type"] == "function"
    assert bindable[0]["function"]["name"] == "alpha_tool"
    assert "parameters" in bindable[0]["function"]


def test_set_catalog_accepts_raw_mcp_payloads(monkeypatch: pytest.MonkeyPatch) -> None:
    """Raw MCP payloads use 'name', not 'tool_name'; set_catalog must normalize
    them through the same path as add_tool instead of crashing in build_index."""
    monkeypatch.setattr(router_module, "SentenceTransformer", DummySentenceTransformer)
    router = router_module.UniversalMCPRouter(model_name="dummy")

    count = router.set_catalog(
        {
            "test-server": [
                {
                    "name": "alpha_tool",
                    "description": "Tool for alpha query",
                    "inputSchema": {"type": "object", "properties": {"x": {"type": "string"}}},
                }
            ]
        }
    )

    assert count == 1
    result = router.route("alpha query")
    assert result.tool_name == "alpha_tool"
    assert result.schema["inputSchema"]["additionalProperties"] is False


def test_singleton_release_is_reference_counted(monkeypatch: pytest.MonkeyPatch) -> None:
    """Tearing down one router must not evict a model another live router shares."""
    monkeypatch.setattr(router_module, "SentenceTransformer", DummySentenceTransformer)
    first = router_module.UniversalMCPRouter(model_name="dummy-shared")
    second = router_module.UniversalMCPRouter(model_name="dummy-shared")
    key = ("dummy-shared", first.device)

    first.teardown()
    assert key in router_module._EmbeddingModelSingleton._models

    second.teardown()
    assert key not in router_module._EmbeddingModelSingleton._models

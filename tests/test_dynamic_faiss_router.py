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


class MultiServerEmbedder:
    """Two servers with a deliberately adversarial geometry for the query
    "cross query": its nearest *tool* is in `beta` (b_one), but `beta`'s centroid
    is pulled toward b_two, so `alpha`'s centroid is the nearer *server*.
    """

    def __init__(self, model_name: str, device: str | None = None) -> None:
        self.model_name = model_name
        self.device = device

    def get_sentence_embedding_dimension(self) -> int:
        return 4

    def encode(self, texts, batch_size: int | None = None, convert_to_numpy: bool = True):
        del batch_size, convert_to_numpy
        out = []
        for text in texts:
            if '"tool_name":"a_one"' in text or '"tool_name":"a_two"' in text:
                out.append([1.0, 0.0, 0.0, 0.0])
            elif '"tool_name":"b_one"' in text:
                out.append([0.0, 0.0, 1.0, 0.0])
            elif '"tool_name":"b_two"' in text:
                out.append([0.0, 1.0, 0.0, 0.0])
            elif "alpha thing" in text:
                out.append([1.0, 0.0, 0.0, 0.0])
            elif "cross query" in text:
                out.append([0.6, 0.0, 0.8, 0.0])
            else:
                out.append([0.0, 0.0, 0.0, 1.0])
        return np.asarray(out, dtype=np.float32)


def build_multi_server_router(monkeypatch: pytest.MonkeyPatch) -> router_module.UniversalMCPRouter:
    monkeypatch.setattr(router_module, "SentenceTransformer", MultiServerEmbedder)
    router = router_module.UniversalMCPRouter(model_name="dummy-multi")
    schema = {"type": "object", "properties": {"x": {"type": "string"}}}
    router.ingest_server("alpha", [
        {"tool_name": "a_one", "description": "alpha one", "inputSchema": schema},
        {"tool_name": "a_two", "description": "alpha two", "inputSchema": schema},
    ])
    router.ingest_server("beta", [
        {"tool_name": "b_one", "description": "beta one", "inputSchema": schema},
        {"tool_name": "b_two", "description": "beta two", "inputSchema": schema},
    ])
    return router


def test_hierarchical_gates_to_top_server(monkeypatch: pytest.MonkeyPatch) -> None:
    router = build_multi_server_router(monkeypatch)

    results = router.route_top_k_hierarchical("alpha thing", k=3, n_servers=1)

    assert results, "expected a match"
    assert results[0].tool_name == "a_one"
    # Only the selected server's tools are candidates.
    assert {r.server_name for r in results} == {"alpha"}


def test_hierarchical_recall_tradeoff_widens_with_n_servers(monkeypatch: pytest.MonkeyPatch) -> None:
    router = build_multi_server_router(monkeypatch)

    # Flat search would pick beta/b_one (the globally nearest tool)...
    assert router.route_top_k("cross query", k=1)[0].tool_name == "b_one"

    # ...but gating to a single server picks the nearer *centroid* (alpha) and
    # therefore MISSES b_one — the documented recall trade-off.
    tight = router.route_top_k_hierarchical("cross query", k=1, n_servers=1)
    assert tight[0].server_name == "alpha"
    assert tight[0].tool_name != "b_one"

    # Widening the gate recovers the flat result.
    wide = router.route_top_k_hierarchical("cross query", k=1, n_servers=2)
    assert wide[0].tool_name == "b_one"


def test_hierarchical_equivalent_to_flat_when_all_servers_selected(monkeypatch: pytest.MonkeyPatch) -> None:
    router = build_multi_server_router(monkeypatch)

    flat = router.route_top_k("cross query", k=4)
    hier = router.route_top_k_hierarchical("cross query", k=4, n_servers=5)  # >= server count

    # Same top result and same candidate set (tie order between equal-score tools
    # is unspecified, so compare as sets rather than exact order).
    assert hier[0].tool_name == flat[0].tool_name == "b_one"
    assert {r.tool_name for r in hier} == {r.tool_name for r in flat}


def test_hierarchical_rejects_invalid_n_servers(monkeypatch: pytest.MonkeyPatch) -> None:
    router = build_multi_server_router(monkeypatch)

    with pytest.raises(ValueError):
        router.route_top_k_hierarchical("alpha thing", k=1, n_servers=0)


def test_reingest_server_replaces_only_that_server(monkeypatch: pytest.MonkeyPatch) -> None:
    """P2: reingest one server incrementally — its removed tool disappears, its
    new tool routes, and the other server's tools are untouched."""
    router = build_multi_server_router(monkeypatch)
    schema = {"type": "object", "properties": {"x": {"type": "string"}}}
    assert router.route_top_k("cross query", k=1)[0].tool_name == "b_one"

    count = router.reingest_server("beta", [
        {"tool_name": "b_two", "description": "beta two", "inputSchema": schema},
    ])

    assert count == 1
    assert len(router.metadata) == 3  # 2 alpha + 1 beta
    top = router.route_top_k("cross query", k=1)[0]
    assert top.tool_name != "b_one"          # removed tool is gone from routing
    assert top.server_name == "alpha"        # nearest remaining candidate
    assert router.route_top_k("alpha thing", k=1)[0].tool_name == "a_one"  # alpha intact


def test_remove_server_unknown_is_noop(monkeypatch: pytest.MonkeyPatch) -> None:
    router = build_multi_server_router(monkeypatch)
    assert router.remove_server("nope") == 0
    assert len(router.metadata) == 4


class FakeReranker:
    """Reverses the bi-encoder order, to prove the router actually applies it."""

    def rank(self, query, documents):
        n = len(documents)
        return [(i, float(n - pos)) for pos, i in enumerate(reversed(range(n)))]


def test_rerank_reorders_the_shortlist(monkeypatch: pytest.MonkeyPatch) -> None:
    router = build_multi_server_router(monkeypatch)
    base = [r.tool_name for r in router.route_top_k("cross query", k=4)]
    assert len(base) >= 2, "need a multi-candidate shortlist to exercise reranking"

    router._reranker = FakeReranker()
    reranked = [r.tool_name for r in router.route_top_k("cross query", k=4)]

    assert reranked == list(reversed(base))  # the reranker reordered the candidates


def test_rerank_disabled_leaves_order_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    router = build_multi_server_router(monkeypatch)
    assert router._reranker is None  # opt-in: off by default
    first = [r.tool_name for r in router.route_top_k("cross query", k=4)]
    second = [r.tool_name for r in router.route_top_k("cross query", k=4)]
    assert first == second


def test_rerank_config_constructs_reranker(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(router_module, "SentenceTransformer", DummySentenceTransformer)
    config = router_module.RouterHyperparameters(rerank=True)
    router = router_module.UniversalMCPRouter(model_name="dummy", config=config)
    assert router._reranker is not None  # constructed, but the cross-encoder loads lazily


class CountingEmbedder(DummySentenceTransformer):
    """Counts how many TOOL texts (not queries) were actually encoded."""

    tool_texts_encoded = 0

    def encode(self, texts, batch_size=None, convert_to_numpy=True):
        CountingEmbedder.tool_texts_encoded += sum('"tool_name"' in t for t in texts)
        return super().encode(texts, batch_size, convert_to_numpy)


def test_embedding_cache_persists_across_router_instances(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    """P1: with cache_dir set, a second router (fresh process in real life) must
    not re-encode unchanged tools — and must still route correctly from cache."""
    monkeypatch.setattr(router_module, "SentenceTransformer", CountingEmbedder)
    CountingEmbedder.tool_texts_encoded = 0
    schema = {"type": "object", "properties": {"x": {"type": "string"}}}
    tools = [
        {"tool_name": "alpha_tool", "description": "Tool for alpha query", "inputSchema": schema},
        {"tool_name": "beta_tool", "description": "Tool for beta query", "inputSchema": schema},
    ]

    first = router_module.UniversalMCPRouter(
        model_name="dummy-cache", config=router_module.RouterHyperparameters(cache_dir=str(tmp_path)))
    first.ingest_server("test-server", tools)
    assert CountingEmbedder.tool_texts_encoded == 2  # cold start encodes both
    first.teardown()

    second = router_module.UniversalMCPRouter(
        model_name="dummy-cache", config=router_module.RouterHyperparameters(cache_dir=str(tmp_path)))
    second.ingest_server("test-server", tools)
    assert CountingEmbedder.tool_texts_encoded == 2  # warm start: zero re-encodes
    assert second.route("alpha query").tool_name == "alpha_tool"  # cached vectors route correctly
    second.teardown()


def test_embedding_cache_encodes_only_new_tools(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    monkeypatch.setattr(router_module, "SentenceTransformer", CountingEmbedder)
    CountingEmbedder.tool_texts_encoded = 0
    schema = {"type": "object", "properties": {"x": {"type": "string"}}}
    base = [{"tool_name": "alpha_tool", "description": "Tool for alpha query", "inputSchema": schema}]

    first = router_module.UniversalMCPRouter(
        model_name="dummy-cache2", config=router_module.RouterHyperparameters(cache_dir=str(tmp_path)))
    first.ingest_server("test-server", base)
    first.teardown()
    assert CountingEmbedder.tool_texts_encoded == 1

    grown = base + [{"tool_name": "beta_tool", "description": "Tool for beta query", "inputSchema": schema}]
    second = router_module.UniversalMCPRouter(
        model_name="dummy-cache2", config=router_module.RouterHyperparameters(cache_dir=str(tmp_path)))
    second.ingest_server("test-server", grown)
    second.teardown()
    assert CountingEmbedder.tool_texts_encoded == 2  # only the NEW tool was encoded


def test_query_embedding_lru_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    """P3: identical repeated queries must not re-hit the encoder (results unchanged)."""

    class QueryCountingEmbedder(DummySentenceTransformer):
        query_encodes = 0

        def encode(self, texts, batch_size=None, convert_to_numpy=True):
            QueryCountingEmbedder.query_encodes += sum('"tool_name"' not in t for t in texts)
            return super().encode(texts, batch_size, convert_to_numpy)

    monkeypatch.setattr(router_module, "SentenceTransformer", QueryCountingEmbedder)
    QueryCountingEmbedder.query_encodes = 0
    router = router_module.UniversalMCPRouter(model_name="dummy-qcache")
    router.ingest_server("test-server", [
        {"tool_name": "alpha_tool", "description": "Tool for alpha query",
         "inputSchema": {"type": "object", "properties": {}}},
    ])

    first = router.route_top_k("alpha query", k=1)
    second = router.route_top_k("alpha query", k=1)

    assert QueryCountingEmbedder.query_encodes == 1  # second call served from LRU
    assert [r.tool_name for r in first] == [r.tool_name for r in second]
    router.teardown()


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

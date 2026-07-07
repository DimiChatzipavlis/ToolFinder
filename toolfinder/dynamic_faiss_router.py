"""Semantic routing for MCP tools backed by FAISS similarity search."""

from __future__ import annotations

import copy
import gc
import hashlib
import json
import logging
import os
import threading
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import faiss
import numpy as np
import torch
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer


logger = logging.getLogger(__name__)


ToolSchema = dict[str, Any]


class RouterHyperparameters(BaseModel):
    index_type: Literal["flat", "hnsw", "auto"] = Field(
        default="flat",
        description=(
            "Vector index backend. 'flat' is exact and fastest below ~5e4 tools; "
            "'hnsw' is approximate and only pays off on very large catalogs; "
            "'auto' picks hnsw when the staged catalog exceeds hnsw_auto_threshold."
        ),
    )
    hnsw_auto_threshold: int = Field(
        default=50_000,
        description="Catalog size at which index_type='auto' switches from flat to HNSW",
    )
    hnsw_m: int = Field(default=32, description="Number of bi-directional links for HNSW graph")
    hnsw_ef_search: int = Field(default=64, description="Depth of search exploration at query time")
    min_cosine_similarity: float = Field(
        default=0.15,
        description="Absolute fallback threshold for cosine similarity",
    )
    top_k_candidates: int = Field(default=3, description="Default number of route candidates to evaluate")
    rerank: bool = Field(
        default=False,
        description=(
            "Opt-in cross-encoder re-ranking of the bi-encoder shortlist. Helps confusable / "
            "out-of-domain catalogs and weak agent models; adds latency, so it is off by default."
        ),
    )
    rerank_model: str = Field(
        default="cross-encoder/ms-marco-MiniLM-L-6-v2",
        description="CrossEncoder checkpoint used when rerank=True (stock, zero-shot — no retraining).",
    )
    rerank_pool: int = Field(
        default=20,
        description="How many bi-encoder candidates to re-rank before truncating to k.",
    )
    cache_dir: str | None = Field(
        default=None,
        description=(
            "Opt-in persistent embedding cache directory. Tool embeddings are stored "
            "keyed by (model, schema-hash), so restarts and refresh() only re-encode "
            "new or changed tools instead of the whole catalog."
        ),
    )


@dataclass(frozen=True)
class RouteResult:
    """Represents a routed tool candidate with its similarity score."""

    server_name: str
    tool_name: str
    schema: ToolSchema
    score: float


class RouteNotFoundError(LookupError):
    """Raised when no tool satisfies routing constraints for a query."""

    pass


class _EmbeddingModelSingleton:
    """Shared, reference-counted cache of loaded SentenceTransformer models.

    Multiple routers using the same (model, device) share one instance; a
    router's teardown releases its reference, and weights are evicted only when
    the last holder releases — so one router cannot evict a model another live
    router still depends on.
    """

    _models: dict[tuple[str, str], SentenceTransformer] = {}
    _refcounts: dict[tuple[str, str], int] = {}
    _lock = threading.Lock()

    @classmethod
    def acquire(cls, model_name: str, device: str) -> SentenceTransformer:
        key = (model_name, device)
        with cls._lock:
            model = cls._models.get(key)
            if model is None:
                model = SentenceTransformer(model_name, device=device)
                cls._models[key] = model
            cls._refcounts[key] = cls._refcounts.get(key, 0) + 1
            return model

    @classmethod
    def release(cls, model_name: str, device: str) -> None:
        key = (model_name, device)
        evicted = False
        with cls._lock:
            remaining = cls._refcounts.get(key, 0) - 1
            if remaining > 0:
                cls._refcounts[key] = remaining
            else:
                cls._refcounts.pop(key, None)
                model = cls._models.pop(key, None)
                if model is not None:
                    del model
                    evicted = True

        if evicted:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


def to_openai_tools(results: Iterable[RouteResult]) -> list[dict[str, Any]]:
    """Convert routed candidates into OpenAI-compatible bindable tool schemas."""
    return [
        {
            "server_name": result.server_name,
            "tool_name": result.tool_name,
            "type": "function",
            "function": {
                "name": result.tool_name,
                "description": result.schema.get("description", ""),
                "parameters": copy.deepcopy(result.schema.get("inputSchema", {})),
            },
        }
        for result in results
    ]


class UniversalMCPRouter:
    """Route natural-language queries to MCP tools using dense retrieval.

    The router ingests tool schemas, builds embeddings, and queries a FAISS
    index (exact `IndexFlatIP` by default; see `RouterHyperparameters.index_type`).
    Routing methods always return `RouteResult` objects; use `to_openai_tools()`
    to convert candidates into bindable function-calling schemas.

    Quality note: `model_name` defaults to a zero-shot checkpoint. The
    benchmark's best results come from fine-tuned weights (produced by the
    archived pipeline under `research/experiments/`); pass a local artifact path
    to `model_name` to load them. Zero-shot dense retrieval can underperform
    lexical search on in-domain catalogs.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-mpnet-base-v2",
        device: str | None = None,
        batch_size: int = 32,
        config: RouterHyperparameters | None = None,
    ) -> None:
        """Initialize the router and allocate the FAISS index.

        Args:
            model_name: SentenceTransformer model used for embeddings.
            device: Explicit device override. Uses CUDA when available, else CPU.
            batch_size: Batch size used for embedding generation.
            config: Optional router hyperparameters. Defaults to RouterHyperparameters().
        """
        self.model_name: str = model_name
        self.device: str = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size: int = batch_size
        self.config: RouterHyperparameters = config or RouterHyperparameters()
        self.model: SentenceTransformer | None = _EmbeddingModelSingleton.acquire(self.model_name, self.device)
        embedding_dim = int(self.model.get_sentence_embedding_dimension())

        self._embedding_dim: int = embedding_dim
        self.faiss_index: faiss.Index = self._create_index()
        self.metadata: dict[int, tuple[str, str, ToolSchema]] = {}
        self._staged_tools: list[tuple[str, ToolSchema]] = []
        # Server-aware (hierarchical) routing state, populated during ingest.
        self._embeddings: np.ndarray = np.zeros((0, embedding_dim), dtype=np.float32)
        self._server_to_ids: dict[str, list[int]] = {}
        self._server_centroids: dict[str, np.ndarray] | None = None
        # Optional cross-encoder reranker (opt-in; its model loads lazily on first use).
        self._reranker = None
        if self.config.rerank:
            from toolfinder.reranker import CrossEncoderReranker

            self._reranker = CrossEncoderReranker(self.config.rerank_model, device=self.device)
        # Optional persistent embedding cache (opt-in via config.cache_dir).
        self._embedding_cache: dict[str, np.ndarray] | None = None
        self._cache_path: Path | None = None
        self._load_embedding_cache()

    def enable_rerank(self, model_name: str | None = None) -> None:
        """Turn on cross-encoder re-ranking after construction.

        Used by the bridge's auto-scale logic (the catalog size is only known
        after ingest). The cross-encoder itself still loads lazily on first use.
        """
        from toolfinder.reranker import CrossEncoderReranker

        self.config.rerank = True
        if model_name:
            self.config.rerank_model = model_name
        self._reranker = CrossEncoderReranker(self.config.rerank_model, device=self.device)

    # --- persistent embedding cache -----------------------------------------

    def _load_embedding_cache(self) -> None:
        if not self.config.cache_dir:
            return
        slug = hashlib.sha256(self.model_name.encode("utf-8")).hexdigest()[:16]
        self._cache_path = Path(self.config.cache_dir) / f"embeddings_{slug}.npz"
        cache: dict[str, np.ndarray] = {}
        if self._cache_path.exists():
            try:
                with np.load(self._cache_path) as archive:
                    for key in archive.files:
                        vector = np.asarray(archive[key], dtype=np.float32)
                        if vector.shape == (self._embedding_dim,):
                            cache[key] = vector
            except Exception as exc:  # noqa: BLE001 - a corrupt cache must never block startup
                logger.warning("embedding cache unreadable (%s); starting fresh: %s", self._cache_path, exc)
                cache = {}
        self._embedding_cache = cache
        logger.info("embedding cache: %d vectors loaded from %s", len(cache), self._cache_path)

    def _save_embedding_cache(self) -> None:
        if self._cache_path is None or self._embedding_cache is None:
            return
        self._cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._cache_path.with_name(self._cache_path.stem + ".tmp.npz")
        np.savez(tmp, **self._embedding_cache)
        os.replace(tmp, self._cache_path)

    def _encode_tool_texts(self, texts: list[str]) -> np.ndarray:
        """L2-normalized embeddings for tool texts, consulting the persistent
        cache when configured — only new/changed tools hit the encoder."""
        out = np.zeros((len(texts), self._embedding_dim), dtype=np.float32)
        cache = self._embedding_cache
        hashes: list[str] | None = None
        miss_idx = list(range(len(texts)))
        if cache is not None:
            hashes = [hashlib.sha256(t.encode("utf-8")).hexdigest() for t in texts]
            miss_idx = []
            for i, h in enumerate(hashes):
                cached = cache.get(h)
                if cached is not None:
                    out[i] = cached
                else:
                    miss_idx.append(i)
        if miss_idx:
            model = self.model
            if model is None:
                raise RuntimeError("embedding model has been torn down")
            with torch.inference_mode():
                encoded = model.encode(
                    [texts[i] for i in miss_idx],
                    batch_size=self.batch_size,
                    convert_to_numpy=True,
                )
            encoded = np.asarray(encoded, dtype=np.float32)
            faiss.normalize_L2(encoded)
            for j, i in enumerate(miss_idx):
                out[i] = encoded[j]
            if cache is not None and hashes is not None:
                for i in miss_idx:
                    cache[hashes[i]] = out[i].copy()
                self._save_embedding_cache()
        return out

    def _create_index(self, expected_size: int = 0) -> faiss.Index:
        """Allocate the FAISS index configured by `index_type`.

        Exact flat search is the default: at MCP-realistic catalog sizes it is
        faster than HNSW graph traversal, exact, and deterministic. HNSW is used
        only when explicitly requested, or in 'auto' mode once the staged
        catalog exceeds `hnsw_auto_threshold`.
        """
        use_hnsw = self.config.index_type == "hnsw" or (
            self.config.index_type == "auto" and expected_size >= self.config.hnsw_auto_threshold
        )
        if use_hnsw:
            index = faiss.IndexHNSWFlat(
                self._embedding_dim,
                self.config.hnsw_m,
                faiss.METRIC_INNER_PRODUCT,
            )
            index.hnsw.efSearch = self.config.hnsw_ef_search
            return index
        return faiss.IndexFlatIP(self._embedding_dim)

    def set_catalog(self, catalog: dict[str, list[ToolSchema]]) -> int:
        """Replace the staged tool catalog and rebuild the FAISS index.

        Accepts raw MCP payloads (`name`/`inputSchema`) or pre-normalized tools
        (`tool_name`): both ingestion paths share `add_tool`'s normalization, so
        a payload shape that works in one path cannot crash the other.
        """
        self._staged_tools = []
        for server_name, tools in catalog.items():
            for tool in tools:
                self.add_tool(tool, server_name=server_name)
        return self.build_index()

    def teardown(self) -> None:
        """Release router state and aggressively free embedding memory."""
        self.faiss_index = self._create_index()
        self.metadata.clear()
        self._staged_tools.clear()
        self._reset_hierarchical_state()

        if self.model is not None:
            del self.model
            self.model = None

        _EmbeddingModelSingleton.release(self.model_name, self.device)

    def _reset_hierarchical_state(self) -> None:
        """Clear retained embeddings / per-server index ids / cached centroids."""
        self._embeddings = np.zeros((0, self._embedding_dim), dtype=np.float32)
        self._server_to_ids = {}
        self._server_centroids = None

    @staticmethod
    def canonicalize_schema(schema: ToolSchema) -> str:
        """Serialize a schema into a canonical JSON string."""
        return json.dumps(schema, sort_keys=True, separators=(",", ":"))

    def add_tool(self, tool: ToolSchema, server_name: str = "external") -> None:
        """Stage a tool for index construction.

        Args:
            tool: Raw MCP-like tool payload containing name/description/schema fields.
            server_name: Fallback server name when the tool payload omits one.

        Raises:
            ValueError: If no usable tool name is present.
        """
        resolved_server_name = str(tool.get("server_name", server_name))
        tool_name = str(tool.get("tool_name") or tool.get("name") or "")
        if not tool_name:
            raise ValueError("tool must include tool_name or name")

        strict_schema = self._inject_additional_properties_false(
            copy.deepcopy(tool.get("inputSchema") or tool.get("parameters") or {})
        )
        normalized_tool = {
            "tool_name": tool_name,
            "description": str(tool.get("description", "")),
            "inputSchema": strict_schema,
        }
        self._staged_tools.append((resolved_server_name, normalized_tool))

    def build_index(self) -> int:
        """Build the FAISS index from staged tools.

        Returns:
            Number of tools ingested into the index.

        Edge cases:
            Returns `0` when no tools have been staged.
        """
        if not self._staged_tools:
            return 0

        self.faiss_index = self._create_index(expected_size=len(self._staged_tools))
        self.metadata.clear()
        self._reset_hierarchical_state()

        # Staged tools were normalized (and strict-schema injected) exactly once
        # by add_tool; here they are only grouped per server and ingested.
        grouped_tools: dict[str, list[ToolSchema]] = {}
        for server_name, normalized_tool in self._staged_tools:
            grouped_tools.setdefault(server_name, []).append(normalized_tool)

        ingested_count = 0
        for grouped_server_name, grouped_list in grouped_tools.items():
            ingested_count += self.ingest_server(grouped_server_name, grouped_list)

        return ingested_count

    def ingest_server(self, server_name: str, tools_list: list[ToolSchema]) -> int:
        """Ingest all tools from one server directly into the FAISS index.

        Args:
            server_name: Source MCP server identifier.
            tools_list: List of normalized tool payloads.

        Returns:
            Number of tools added for the server.

        Edge cases:
            Returns `0` when `tools_list` is empty.
        """
        if not tools_list:
            return 0

        normalized_tools: list[ToolSchema] = []
        embedding_corpus: list[str] = []

        for raw_tool in tools_list:
            normalized_tool = {
                "server_name": server_name,
                "tool_name": str(raw_tool["tool_name"]),
                "description": str(raw_tool.get("description", "")),
                "inputSchema": copy.deepcopy(raw_tool.get("inputSchema", {})),
            }
            normalized_tools.append(normalized_tool)
            embedding_corpus.append(self._minify_schema_for_embedding(normalized_tool))

        # Cache-aware encode: normalized vectors, only new/changed tools hit the model.
        embeddings = self._encode_tool_texts(embedding_corpus)

        start_index = int(self.faiss_index.ntotal)
        self.faiss_index.add(embeddings)

        for offset, tool in enumerate(normalized_tools):
            index_id = start_index + offset
            self.metadata[index_id] = (server_name, tool["tool_name"], tool)

        # Retain normalized embeddings + per-server index ids for hierarchical
        # (server-aware) routing; invalidate any cached server centroids.
        self._embeddings = np.vstack([self._embeddings, embeddings])
        self._server_to_ids.setdefault(server_name, []).extend(
            range(start_index, start_index + len(normalized_tools))
        )
        self._server_centroids = None

        return len(normalized_tools)

    def route_top_k(
        self,
        query: str,
        k: int | None = None,
    ) -> list[RouteResult]:
        """Route a query to the top-k matching tools.

        Args:
            query: The natural language query to route.
            k: Maximum number of results to return. Defaults to the configured candidate budget.

        Returns:
            List of matching `RouteResult` objects, best first. Returns an empty
            list if no tools meet the configured similarity threshold. Use
            `to_openai_tools()` to convert results into bindable schemas.
        """
        k = self.config.top_k_candidates if k is None else k
        if k < 1:
            raise ValueError("k must be at least 1")
        if self.faiss_index.ntotal == 0:
            raise ValueError("router index is empty; ingest at least one server first")

        model = self.model
        if model is None:
            raise RuntimeError("embedding model has been torn down")

        with torch.inference_mode():
            query_embedding = model.encode([query], convert_to_numpy=True)

        query_embedding = np.asarray(query_embedding, dtype=np.float32)
        faiss.normalize_L2(query_embedding)

        # When reranking, retrieve a larger bi-encoder pool to re-score; otherwise just k.
        pool = max(k, self.config.rerank_pool) if self._reranker is not None else k
        scores, indices = self.faiss_index.search(
            query_embedding,
            k=min(pool, int(self.faiss_index.ntotal)),
        )

        top_score = float(scores[0][0])
        if scores.shape[1] >= 2 and float(indices[0][1]) >= 0:
            margin = float(scores[0][0]) - float(scores[0][1])
            if margin < 0.02:
                logger.warning(
                    "Ambiguous semantic routing for query %r (score1=%.4f, score2=%.4f, margin=%.4f)",
                    query,
                    float(scores[0][0]),
                    float(scores[0][1]),
                    margin,
                )

        if top_score < self.config.min_cosine_similarity:
            logger.debug(
                "Rejected query %r (top_score=%.4f < min_cosine_similarity=%.2f)",
                query,
                top_score,
                self.config.min_cosine_similarity,
            )
            return []

        scored: list[tuple[float, str, str, ToolSchema]] = []
        for score, index_id in zip(scores[0], indices[0], strict=True):
            if index_id < 0 or float(score) < self.config.min_cosine_similarity:
                continue
            server_name, tool_name, schema = self.metadata[int(index_id)]
            scored.append((float(score), server_name, tool_name, schema))
        return self._finalize_candidates(query, scored, k)

    def _finalize_candidates(
        self, query: str, scored: list[tuple[float, str, str, ToolSchema]], k: int
    ) -> list[RouteResult]:
        """Optionally cross-encoder re-rank the threshold-filtered, bi-encoder-ordered
        candidates, then truncate to k and build RouteResults. With no reranker
        configured this is a plain truncation, so default behavior is unchanged."""
        if self._reranker is not None and len(scored) > 1:
            documents = [self._rerank_text(tool_name, schema) for _, _, tool_name, schema in scored]
            ranked = self._reranker.rank(query, documents)
            scored = [(ce_score, scored[i][1], scored[i][2], scored[i][3]) for i, ce_score in ranked]
        return [
            RouteResult(server_name=server_name, tool_name=tool_name, schema=schema, score=score)
            for score, server_name, tool_name, schema in scored[:k]
        ]

    def _rerank_text(self, tool_name: str, schema: ToolSchema) -> str:
        """Compact natural-language document for the cross-encoder."""
        description = schema.get("description", "") if isinstance(schema, dict) else ""
        return f"{tool_name}: {description}".strip()[:512]

    def route(self, query: str) -> RouteResult:
        """Return the single highest-ranked tool candidate for a query.

        Args:
            query: Natural language query to route.

        Returns:
            The best matching tool as `RouteResult`.

        Edge cases:
            Raises `RouteNotFoundError` if no result survives threshold filtering.
        """
        matches = self.route_top_k(query, k=1)
        if not matches:
            raise RouteNotFoundError(
                "no route candidates met the similarity threshold; try lowering the router "
                "threshold or rephrasing the query"
            )
        return matches[0]

    def route_top_k_hierarchical(
        self,
        query: str,
        k: int | None = None,
        n_servers: int = 2,
    ) -> list[RouteResult]:
        """Two-stage, server-aware routing.

        Stage 1 ranks the configured MCP servers by similarity to a per-server
        **centroid** (the mean of that server's tool embeddings) and keeps the
        top `n_servers`. Stage 2 returns the top-k tools drawn only from those
        servers. Narrowing to the most relevant server(s) before tool selection
        improves precision when different servers expose confusable tools, and
        bounds the candidate set for very large multi-server catalogs.

        This is **not** a latency optimization — query encoding dominates and
        flat search is already sub-millisecond at MCP catalog sizes. The win is
        precision and scale.

        Recall trade-off: gating to the top server(s) excludes the correct tool
        if its server is mis-ranked in stage 1; raise `n_servers` to widen the
        gate (more recall, less precision). With one configured server, or
        `n_servers` >= the number of servers, results match `route_top_k`.

        Returns an empty list if nothing clears `min_cosine_similarity`.
        """
        k = self.config.top_k_candidates if k is None else k
        if k < 1:
            raise ValueError("k must be at least 1")
        if n_servers < 1:
            raise ValueError("n_servers must be at least 1")
        if self.faiss_index.ntotal == 0:
            raise ValueError("router index is empty; ingest at least one server first")

        model = self.model
        if model is None:
            raise RuntimeError("embedding model has been torn down")

        with torch.inference_mode():
            query_embedding = model.encode([query], convert_to_numpy=True)
        query_embedding = np.asarray(query_embedding, dtype=np.float32)
        faiss.normalize_L2(query_embedding)
        query_vector = query_embedding[0]

        candidate_ids = self._select_server_candidate_ids(query_vector, n_servers)
        if not candidate_ids:
            return []

        candidate_scores = self._embeddings[candidate_ids] @ query_vector
        pool = max(k, self.config.rerank_pool) if self._reranker is not None else k
        order = np.argsort(-candidate_scores)[:pool]

        top_score = float(candidate_scores[order[0]])
        if top_score < self.config.min_cosine_similarity:
            logger.debug(
                "Rejected query %r hierarchically (top_score=%.4f < min_cosine_similarity=%.2f)",
                query,
                top_score,
                self.config.min_cosine_similarity,
            )
            return []

        scored: list[tuple[float, str, str, ToolSchema]] = []
        for pos in order:
            score = float(candidate_scores[int(pos)])
            if score < self.config.min_cosine_similarity:
                continue
            server_name, tool_name, schema = self.metadata[candidate_ids[int(pos)]]
            scored.append((score, server_name, tool_name, schema))
        return self._finalize_candidates(query, scored, k)

    def _select_server_candidate_ids(self, query_vector: np.ndarray, n_servers: int) -> list[int]:
        """Stage 1: rank servers by centroid similarity, return the tool index
        ids belonging to the top `n_servers`."""
        centroids = self._server_centroids_cached()
        if not centroids:
            return []
        ranked = sorted(centroids, key=lambda name: float(centroids[name] @ query_vector), reverse=True)
        selected = ranked[: max(1, n_servers)]
        logger.info("Hierarchical stage-1 selected %s of %d server(s)", selected, len(centroids))
        candidate_ids: list[int] = []
        for server_name in selected:
            candidate_ids.extend(self._server_to_ids.get(server_name, []))
        return candidate_ids

    def _server_centroids_cached(self) -> dict[str, np.ndarray]:
        """Compute (and cache) the L2-normalized centroid of each server's tool
        embeddings. Invalidated whenever the index is rebuilt."""
        if self._server_centroids is None:
            centroids: dict[str, np.ndarray] = {}
            for server_name, ids in self._server_to_ids.items():
                if not ids:
                    continue
                centroid = self._embeddings[ids].mean(axis=0)
                norm = float(np.linalg.norm(centroid))
                if norm > 0.0:
                    centroid = centroid / norm
                centroids[server_name] = centroid.astype(np.float32)
            self._server_centroids = centroids
        return self._server_centroids

    def _inject_additional_properties_false(self, node: Any, depth: int = 0, max_depth: int = 100) -> Any:
        # ALGY-4 FIX: Prevent stack overflow on deeply nested / circular schemas
        if depth > max_depth:
            logger.warning(
                "Schema depth exceeded %d; returning node unmodified to prevent stack overflow",
                max_depth,
            )
            return node

        if isinstance(node, dict):
            normalized_node: dict[str, Any] = {}
            for key, value in node.items():
                normalized_node[key] = self._inject_additional_properties_false(value, depth + 1, max_depth)

            if normalized_node.get("type") == "object":
                normalized_node["additionalProperties"] = False

            return normalized_node

        if isinstance(node, list):
            return [self._inject_additional_properties_false(item, depth + 1, max_depth) for item in node]

        return node

    def _minify_schema_for_embedding(self, schema: ToolSchema) -> str:
        minified = {
            "server_name": schema["server_name"],
            "tool_name": schema["tool_name"],
            "description": schema.get("description", ""),
            "inputSchema": self._strip_nested_descriptions(
                copy.deepcopy(schema.get("inputSchema", {})),
                prune_nested_descriptions=False,
            ),
        }
        return json.dumps(minified, sort_keys=True, separators=(",", ":"))

    def _strip_nested_descriptions(
        self,
        node: Any,
        prune_nested_descriptions: bool,
    ) -> Any:
        if isinstance(node, dict):
            cleaned: dict[str, Any] = {}
            for key, value in node.items():
                if key == "description" and prune_nested_descriptions:
                    continue

                if key in {
                    "properties",
                    "patternProperties",
                    "$defs",
                    "definitions",
                    "dependentSchemas",
                } and isinstance(value, dict):
                    cleaned[key] = {
                        child_key: self._strip_nested_descriptions(
                            child_value,
                            prune_nested_descriptions=True,
                        )
                        for child_key, child_value in value.items()
                    }
                    continue

                if key in {
                    "items",
                    "additionalProperties",
                    "contains",
                    "if",
                    "then",
                    "else",
                    "not",
                }:
                    cleaned[key] = self._strip_nested_descriptions(
                        value,
                        prune_nested_descriptions=True,
                    )
                    continue

                if key in {"allOf", "anyOf", "oneOf", "prefixItems"} and isinstance(value, list):
                    cleaned[key] = [
                        self._strip_nested_descriptions(item, prune_nested_descriptions=True)
                        for item in value
                    ]
                    continue

                cleaned[key] = self._strip_nested_descriptions(
                    value,
                    prune_nested_descriptions=prune_nested_descriptions,
                )

            return cleaned

        if isinstance(node, list):
            return [
                self._strip_nested_descriptions(
                    item,
                    prune_nested_descriptions=prune_nested_descriptions,
                )
                for item in node
            ]

        return node
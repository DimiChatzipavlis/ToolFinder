"""Semantic routing for MCP tools backed by FAISS similarity search."""

from __future__ import annotations

import copy
import gc
import json
import logging
import threading
from collections.abc import Iterable
from dataclasses import dataclass
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

        if self.model is not None:
            del self.model
            self.model = None

        _EmbeddingModelSingleton.release(self.model_name, self.device)

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

        model = self.model
        if model is None:
            raise RuntimeError("embedding model has been torn down")

        for raw_tool in tools_list:
            normalized_tool = {
                "server_name": server_name,
                "tool_name": str(raw_tool["tool_name"]),
                "description": str(raw_tool.get("description", "")),
                "inputSchema": copy.deepcopy(raw_tool.get("inputSchema", {})),
            }
            normalized_tools.append(normalized_tool)
            embedding_corpus.append(self._minify_schema_for_embedding(normalized_tool))

        with torch.inference_mode():
            embeddings = model.encode(
                embedding_corpus,
                batch_size=self.batch_size,
                convert_to_numpy=True,
            )

        embeddings = np.asarray(embeddings, dtype=np.float32)
        faiss.normalize_L2(embeddings)

        start_index = int(self.faiss_index.ntotal)
        self.faiss_index.add(embeddings)

        for offset, tool in enumerate(normalized_tools):
            index_id = start_index + offset
            self.metadata[index_id] = (server_name, tool["tool_name"], tool)

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

        scores, indices = self.faiss_index.search(
            query_embedding,
            k=min(k, int(self.faiss_index.ntotal)),
        )

        top_score = float(scores[0][0])
        if k > 1 and scores.shape[1] >= 2 and float(indices[0][1]) >= 0:
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

        matches: list[RouteResult] = []
        for score, index_id in zip(scores[0], indices[0], strict=True):
            if index_id < 0:
                continue
            server_name, tool_name, schema = self.metadata[int(index_id)]
            if float(score) < self.config.min_cosine_similarity:
                logger.debug(
                    "Filtered tool %s/%s (score=%.4f < min_cosine_similarity=%.2f)",
                    server_name,
                    tool_name,
                    float(score),
                    self.config.min_cosine_similarity,
                )
                continue
            logger.info(
                "Matched tool %s/%s with score=%.4f (threshold=%.2f)",
                server_name,
                tool_name,
                float(score),
                self.config.min_cosine_similarity,
            )
            matches.append(
                RouteResult(
                    server_name=server_name,
                    tool_name=tool_name,
                    schema=schema,
                    score=float(score),
                )
            )
        return matches

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
from __future__ import annotations

"""Semantic routing for MCP tools backed by FAISS similarity search."""

import copy
import gc
import json
import logging
import threading
from dataclasses import dataclass
from typing import Any

import faiss
import numpy as np
import torch
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer


logger = logging.getLogger(__name__)


ToolSchema = dict[str, Any]


class RouterHyperparameters(BaseModel):
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
    _models: dict[tuple[str, str], SentenceTransformer] = {}
    _lock = threading.Lock()

    @classmethod
    def acquire(cls, model_name: str, device: str) -> SentenceTransformer:
        key = (model_name, device)
        with cls._lock:
            model = cls._models.get(key)
            if model is None:
                model = SentenceTransformer(model_name, device=device)
                cls._models[key] = model
            return model

    @classmethod
    def teardown(cls, model_name: str | None = None, device: str | None = None) -> None:
        with cls._lock:
            if model_name is None and device is None:
                keys = list(cls._models.keys())
            else:
                keys = [
                    key
                    for key in cls._models
                    if (model_name is None or key[0] == model_name)
                    and (device is None or key[1] == device)
                ]

            for key in keys:
                model = cls._models.pop(key, None)
                if model is not None:
                    del model

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class UniversalMCPRouter:
    """Route natural-language queries to MCP tools using dense retrieval.

    The router ingests tool schemas, builds embeddings, and queries a FAISS index.
    It can return either `RouteResult` objects or OpenAI-compatible bindable tool
    schemas when compatibility mode is enabled through `build_index()`.
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
        self.faiss_index: faiss.IndexHNSWFlat = faiss.IndexHNSWFlat(
            self._embedding_dim,
            self.config.hnsw_m,
            faiss.METRIC_INNER_PRODUCT,
        )
        self.faiss_index.hnsw.efSearch = self.config.hnsw_ef_search
        self.metadata: dict[int, tuple[str, str, ToolSchema]] = {}
        self._staged_tools: list[tuple[str, ToolSchema]] = []
        self._compat_mode: bool = False

    def set_catalog(self, catalog: dict[str, list[ToolSchema]]) -> int:
        """Replace the staged tool catalog and rebuild only the FAISS index."""
        self._staged_tools = []
        for server_name, tools in catalog.items():
            for tool in tools:
                self._staged_tools.append((server_name, copy.deepcopy(tool)))
        return self.build_index()

    def teardown(self) -> None:
        """Release router state and aggressively free embedding memory."""
        self.faiss_index = faiss.IndexHNSWFlat(
            self._embedding_dim,
            self.config.hnsw_m,
            faiss.METRIC_INNER_PRODUCT,
        )
        self.faiss_index.hnsw.efSearch = self.config.hnsw_ef_search
        self.metadata.clear()
        self._staged_tools.clear()
        self._compat_mode = False

        if self.model is not None:
            del self.model
            self.model = None

        _EmbeddingModelSingleton.teardown(self.model_name, self.device)

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

        self.faiss_index = faiss.IndexHNSWFlat(
            self._embedding_dim,
            self.config.hnsw_m,
            faiss.METRIC_INNER_PRODUCT,
        )
        self.faiss_index.hnsw.efSearch = self.config.hnsw_ef_search
        self.metadata.clear()

        grouped_tools: dict[str, list[ToolSchema]] = {}
        for default_server_name, raw_tool in self._staged_tools:
            resolved_server_name = str(raw_tool.get("server_name", default_server_name))
            normalized_tool = {
                "tool_name": str(raw_tool["tool_name"]),
                "description": str(raw_tool.get("description", "")),
                "inputSchema": self._inject_additional_properties_false(
                    copy.deepcopy(raw_tool.get("inputSchema", {}))
                ),
            }
            grouped_tools.setdefault(resolved_server_name, []).append(normalized_tool)

        ingested_count = 0
        for grouped_server_name, grouped_list in grouped_tools.items():
            ingested_count += self.ingest_server(grouped_server_name, grouped_list)

        self._compat_mode = True
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
    ) -> list[RouteResult] | list[dict[str, Any]]:
        """Route a query to the top-k matching tools.

        Args:
            query: The natural language query to route.
            k: Maximum number of results to return. Defaults to the configured candidate budget.

        Returns:
            List of matching tools (RouteResult or dict depending on compat_mode).
            Returns empty list if no tools meet the configured similarity threshold.
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
        if self._compat_mode:
            return [self._format_bindable_tool_schema(match) for match in matches]
        return matches

    def route(self, query: str) -> RouteResult | dict[str, Any]:
        """Return the single highest-ranked tool candidate for a query.

        Args:
            query: Natural language query to route.

        Returns:
            The best matching tool as `RouteResult` or bindable schema.

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

    @staticmethod
    def _format_bindable_tool_schema(result: RouteResult) -> dict[str, Any]:
        return {
            "server_name": result.server_name,
            "tool_name": result.tool_name,
            "type": "function",
            "function": {
                "name": result.tool_name,
                "description": result.schema.get("description", ""),
                "parameters": copy.deepcopy(result.schema.get("inputSchema", {})),
            },
        }

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
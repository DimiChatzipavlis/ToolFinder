from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from typing import Any

import faiss
import numpy as np
import torch
from sentence_transformers import SentenceTransformer


ToolSchema = dict[str, Any]


@dataclass(frozen=True)
class RouteResult:
    server_name: str
    tool_name: str
    schema: ToolSchema
    score: float


class UniversalMCPRouter:
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-mpnet-base-v2",
        device: str | None = None,
        batch_size: int = 32,
    ) -> None:
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.model = SentenceTransformer(self.model_name, device=self.device)
        embedding_dim = int(self.model.get_sentence_embedding_dimension())

        self._embedding_dim = embedding_dim
        self.faiss_index = faiss.IndexFlatIP(self._embedding_dim)
        self.metadata: dict[int, tuple[str, str, ToolSchema]] = {}
        self._staged_tools: list[tuple[str, ToolSchema]] = []
        self._compat_mode = False

    @staticmethod
    def canonicalize_schema(schema: ToolSchema) -> str:
        return json.dumps(schema, sort_keys=True, separators=(",", ":"))

    def add_tool(self, tool: ToolSchema, server_name: str = "external") -> None:
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
        if not self._staged_tools:
            return 0

        self.faiss_index = faiss.IndexFlatIP(self._embedding_dim)
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

        with torch.inference_mode():
            embeddings = self.model.encode(
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

    def route_top_k(self, query: str, k: int = 3) -> list[RouteResult] | list[dict[str, Any]]:
        if k < 1:
            raise ValueError("k must be at least 1")
        if self.faiss_index.ntotal == 0:
            raise ValueError("router index is empty; ingest at least one server first")

        with torch.inference_mode():
            query_embedding = self.model.encode([query], convert_to_numpy=True)

        query_embedding = np.asarray(query_embedding, dtype=np.float32)
        faiss.normalize_L2(query_embedding)

        scores, indices = self.faiss_index.search(
            query_embedding,
            k=min(k, int(self.faiss_index.ntotal)),
        )

        matches: list[RouteResult] = []
        for score, index_id in zip(scores[0], indices[0], strict=True):
            if index_id < 0:
                continue
            server_name, tool_name, schema = self.metadata[int(index_id)]
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
        return self.route_top_k(query, k=1)[0]

    @staticmethod
    def _format_bindable_tool_schema(result: RouteResult) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": result.tool_name,
                "description": result.schema.get("description", ""),
                "parameters": copy.deepcopy(result.schema.get("inputSchema", {})),
            },
        }

    def _inject_additional_properties_false(self, node: Any) -> Any:
        if isinstance(node, dict):
            normalized_node: dict[str, Any] = {}
            for key, value in node.items():
                normalized_node[key] = self._inject_additional_properties_false(value)

            if normalized_node.get("type") == "object":
                normalized_node["additionalProperties"] = False

            return normalized_node

        if isinstance(node, list):
            return [self._inject_additional_properties_false(item) for item in node]

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
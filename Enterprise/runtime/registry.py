from __future__ import annotations

import asyncio
import copy
import logging
import os
import re
from typing import Any

from toolfinder.dynamic_faiss_router import RouteResult, UniversalMCPRouter

from .contracts import ToolCandidate


logger = logging.getLogger(__name__)


class HybridToolRegistry:
    """Maintains a live tool catalog and semantic retrieval router."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-mpnet-base-v2",
        allow_low_confidence_keyword_fallback: bool | None = None,
    ) -> None:
        self.model_name = model_name
        self._allow_low_confidence_keyword_fallback = (
            os.getenv("ENTERPRISE_ALLOW_KEYWORD_LOW_CONFIDENCE_FALLBACK", "0") == "1"
            if allow_low_confidence_keyword_fallback is None
            else allow_low_confidence_keyword_fallback
        )
        self._lock = asyncio.Lock()
        self._server_tools: dict[str, list[dict[str, Any]]] = {}
        self._router: UniversalMCPRouter | None = None
        self._router_error: str | None = (
            "semantic router disabled by ENTERPRISE_DISABLE_SEMANTIC_ROUTER"
            if os.getenv("ENTERPRISE_DISABLE_SEMANTIC_ROUTER", "0") == "1"
            else None
        )
        self._router = self._build_router()

    async def upsert_server_tools(self, server_name: str, tools: list[dict[str, Any]]) -> None:
        normalized_tools: list[dict[str, Any]] = []
        for raw_tool in tools:
            tool_name = str(raw_tool.get("tool_name") or raw_tool.get("name") or "").strip()
            if not tool_name:
                continue

            input_schema = raw_tool.get("inputSchema") or raw_tool.get("parameters") or {}
            if not isinstance(input_schema, dict):
                input_schema = {}

            normalized_tools.append(
                {
                    "tool_name": tool_name,
                    "description": str(raw_tool.get("description", "")),
                    "inputSchema": copy.deepcopy(input_schema),
                }
            )

        async with self._lock:
            self._server_tools[server_name] = normalized_tools
            self._router = self._build_router()

    async def route(self, query: str, k: int, min_score: float) -> list[ToolCandidate]:
        async with self._lock:
            if self._router is not None:
                previous_min_cosine_similarity = self._router.config.min_cosine_similarity
                self._router.config.min_cosine_similarity = min_score
                try:
                    routed = await asyncio.to_thread(self._router.route_top_k, query, k=k)
                finally:
                    self._router.config.min_cosine_similarity = previous_min_cosine_similarity
            else:
                routed = self._keyword_route_top_k(query, k=k, min_score=min_score)

        candidates: list[ToolCandidate] = []
        for item in routed:
            if isinstance(item, RouteResult):
                candidates.append(
                    ToolCandidate(
                        server_name=item.server_name,
                        tool_name=item.tool_name,
                        description=str(item.schema.get("description", "")),
                        input_schema=copy.deepcopy(item.schema.get("inputSchema", {})),
                        score=float(item.score),
                    )
                )
                continue

            function_payload = item.get("function", {}) if isinstance(item, dict) else {}
            candidates.append(
                ToolCandidate(
                    server_name=str(item.get("server_name") or "external") if isinstance(item, dict) else "external",
                    tool_name=str(function_payload.get("name", "")),
                    description=str(function_payload.get("description", "")),
                    input_schema=copy.deepcopy(function_payload.get("parameters", {})),
                    score=float(item.get("score", 0.0)) if isinstance(item, dict) else 0.0,
                )
            )

        return candidates

    async def snapshot(self) -> dict[str, list[dict[str, Any]]]:
        async with self._lock:
            return copy.deepcopy(self._server_tools)

    def _build_router(self) -> UniversalMCPRouter | None:
        if self._router_error is not None:
            return None

        try:
            router = self._router or UniversalMCPRouter(model_name=self.model_name)
            router.set_catalog(self._server_tools)
            self._router_error = None
            return router
        except Exception as exc:
            logger.exception("Runtime error encountered")
            self._router_error = str(exc)
            return None

    def teardown(self) -> None:
        if self._router is not None:
            self._router.teardown()
            self._router = None
        self._server_tools.clear()

    def _keyword_route_top_k(self, query: str, k: int, min_score: float) -> list[RouteResult]:
        query_tokens = self._tokenize(query)
        scored: list[tuple[float, RouteResult]] = []

        for server_name, tools in self._server_tools.items():
            for tool in tools:
                tool_name = str(tool.get("tool_name", ""))
                description = str(tool.get("description", ""))
                input_schema = tool.get("inputSchema", {}) if isinstance(tool.get("inputSchema"), dict) else {}

                signature_parts = [server_name, tool_name, description]
                properties = input_schema.get("properties", {}) if isinstance(input_schema, dict) else {}
                if isinstance(properties, dict):
                    signature_parts.extend(str(key) for key in properties.keys())

                tool_signature = " ".join(signature_parts)
                tool_tokens = self._tokenize(tool_signature)
                overlap = len(query_tokens & tool_tokens)
                denominator = max(1, min(len(query_tokens), len(tool_tokens)))
                score = overlap / denominator

                scored.append(
                    (
                        score,
                        RouteResult(
                            server_name=server_name,
                            tool_name=tool_name,
                            score=score,
                            schema={
                                "description": description,
                                "inputSchema": copy.deepcopy(input_schema),
                            },
                        ),
                    )
                )

        scored.sort(key=lambda item: item[0], reverse=True)
        filtered = [result for score, result in scored if score >= min_score]
        if filtered:
            return filtered[:k]

        if not self._allow_low_confidence_keyword_fallback:
            return []

        # Optional degraded mode fallback for environments that prioritize liveness.
        fallback = [result for _, result in scored[:k]]
        for result in fallback:
            if result.score <= 0:
                result.score = 0.01
        return fallback

    @staticmethod
    def _tokenize(text: str) -> set[str]:
        return {token for token in re.findall(r"[a-z0-9_]+", text.lower()) if token}

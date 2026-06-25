"""OpenAPI ingestion adapter — front any REST API described by an OpenAPI 3.x
spec as if it were an MCP server, so ToolFinder routes over its operations with
the same `initialize_and_get_tools()` / `call_tool()` interface as the stdio
MCP client.

Each operation (path + method) becomes one tool:
  tool_name   = operationId (sanitized) or "{method}_{path}"
  description = summary + description
  inputSchema = object: the operation's path/query/header parameters, plus a
                `body` property for the JSON request body when present.

`call_tool` maps the chosen tool + arguments back to an HTTP request
(path-param substitution, query params, headers, JSON body) and applies optional
auth resolved from **environment variables** — credentials are never inlined in
config and never logged.

Scope (v0.1, stated honestly): OpenAPI 3.x; JSON request/response; local `$ref`
resolution within the document (cycle-guarded). No OAuth flows, no multipart, no
response-schema validation. Auth: bearer token / API-key header / API-key query.
"""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any

import httpx

logger = logging.getLogger(__name__)

_HTTP_METHODS = ("get", "post", "put", "patch", "delete")


class OpenAPIClient:
    """Routing/execution client for an OpenAPI-described REST API."""

    def __init__(
        self,
        server_name: str,
        spec: Any,
        base_url: str | None = None,
        auth: dict | None = None,
        request_timeout_s: float = 45.0,
    ) -> None:
        # `spec` may be an already-parsed dict, an http(s) URL, or a file path.
        self.server_name = server_name
        self._spec_src = spec
        self._base_url_override = base_url
        self._auth = auth or {}
        self._timeout = request_timeout_s
        self._spec: dict = {}
        self._base_url: str = ""
        self._ops: dict[str, dict] = {}
        self._client: httpx.AsyncClient | None = None

    async def initialize_and_get_tools(self) -> list[dict]:
        self._spec = await self._load_spec()
        self._base_url = (self._base_url_override or self._spec_base_url()).rstrip("/")
        self._client = httpx.AsyncClient(timeout=self._timeout, base_url=self._base_url)

        tools: list[dict] = []
        used: set[str] = set()
        self._ops = {}
        for path, item in (self._spec.get("paths") or {}).items():
            if not isinstance(item, dict):
                continue
            shared = item.get("parameters", []) if isinstance(item.get("parameters"), list) else []
            for method in _HTTP_METHODS:
                op = item.get(method)
                if not isinstance(op, dict):
                    continue
                name = self._operation_name(op, method, str(path), used)
                params = self._deref(shared + (op.get("parameters") or []))
                props: dict[str, Any] = {}
                required: list[str] = []
                param_loc: dict[str, str] = {}
                for p in params:
                    if not isinstance(p, dict) or "name" not in p or p.get("in") not in ("path", "query", "header"):
                        continue
                    schema = self._deref(p.get("schema") or {"type": "string"})
                    if p.get("description") and isinstance(schema, dict):
                        schema = {**schema, "description": str(p["description"])[:300]}
                    props[p["name"]] = schema
                    param_loc[p["name"]] = p["in"]
                    if p.get("required"):
                        required.append(p["name"])
                has_body = False
                rb = op.get("requestBody")
                if isinstance(rb, dict):
                    content = (rb.get("content") or {}).get("application/json")
                    if isinstance(content, dict) and "schema" in content:
                        props["body"] = self._deref(content["schema"])
                        has_body = True
                        if rb.get("required"):
                            required.append("body")
                input_schema: dict[str, Any] = {"type": "object", "properties": props}
                if required:
                    input_schema["required"] = required
                description = " — ".join(
                    s for s in (op.get("summary"), op.get("description")) if isinstance(s, str) and s
                )[:1024]
                self._ops[name] = {"method": method, "path": str(path), "param_loc": param_loc, "has_body": has_body}
                tools.append({"tool_name": name, "description": description or name, "inputSchema": input_schema})

        logger.info("OpenAPI %r: %d operations from %s", self.server_name, len(tools), self._base_url or "<no base_url>")
        return tools

    async def call_tool(self, tool_name: str, arguments: dict[str, Any] | None) -> Any:
        if tool_name not in self._ops:
            return {"error": f"unknown OpenAPI operation '{tool_name}'"}
        if self._client is None:
            return {"error": "OpenAPI client not initialized"}
        method, url_path, params, headers, body = self._build_request(tool_name, arguments)
        try:
            resp = await self._client.request(
                method, url_path, params=params or None, headers=headers or None, json=body
            )
            ctype = resp.headers.get("content-type", "")
            data = resp.json() if "application/json" in ctype else resp.text[:4000]
            return {"ok": resp.is_success, "status": resp.status_code, "data": data}
        except Exception as exc:  # noqa: BLE001 - surface transport errors to the agent, don't crash the bridge
            return {"error": f"OpenAPI request to '{tool_name}' failed: {exc}"}

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    # --- internals -------------------------------------------------------

    def _build_request(
        self, tool_name: str, arguments: dict[str, Any] | None
    ) -> tuple[str, str, dict[str, Any], dict[str, str], Any]:
        """Pure request construction (no network) — unit-testable."""
        op = self._ops[tool_name]
        args = dict(arguments or {})
        headers, params = self._resolve_auth()
        body = args.pop("body", None) if op["has_body"] else None
        url_path = op["path"]
        for pname, loc in op["param_loc"].items():
            if pname not in args:
                continue
            value = args[pname]
            if loc == "path":
                url_path = url_path.replace("{" + pname + "}", str(value))
            elif loc == "query":
                params[pname] = value
            elif loc == "header":
                headers[pname] = str(value)
        return op["method"].upper(), url_path, params, headers, body

    def _resolve_auth(self) -> tuple[dict[str, str], dict[str, Any]]:
        """Build auth headers/query params from env vars. Never logs secrets."""
        headers: dict[str, str] = {}
        params: dict[str, Any] = {}
        kind = str(self._auth.get("type", "")).lower()
        if kind == "bearer":
            token = os.environ.get(str(self._auth.get("token_env", "")))
            if token:
                headers["Authorization"] = f"Bearer {token}"
        elif kind == "header":
            value = os.environ.get(str(self._auth.get("value_env", "")))
            if value and self._auth.get("name"):
                headers[str(self._auth["name"])] = value
        elif kind == "query":
            value = os.environ.get(str(self._auth.get("value_env", "")))
            if value and self._auth.get("name"):
                params[str(self._auth["name"])] = value
        return headers, params

    async def _load_spec(self) -> dict:
        src = self._spec_src
        if isinstance(src, dict):
            return src
        if not isinstance(src, str) or not src:
            raise ValueError(f"OpenAPI server {self.server_name!r}: no spec/spec_url/spec_file given")
        if src.startswith(("http://", "https://")):
            async with httpx.AsyncClient(timeout=self._timeout) as fetch:
                resp = await fetch.get(src)
                resp.raise_for_status()
                text = resp.text
        else:
            text = Path(src).read_text(encoding="utf-8")
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            try:
                import yaml  # optional dependency, only needed for YAML specs
            except ImportError as imp:
                raise ValueError(
                    f"OpenAPI server {self.server_name!r}: spec is not JSON and PyYAML is not installed"
                ) from imp
            loaded = yaml.safe_load(text)
            if not isinstance(loaded, dict):
                raise ValueError(f"OpenAPI server {self.server_name!r}: spec did not parse to an object") from exc
            return loaded

    def _spec_base_url(self) -> str:
        servers = self._spec.get("servers")
        if isinstance(servers, list) and servers and isinstance(servers[0], dict):
            return str(servers[0].get("url", ""))
        return ""

    def _operation_name(self, op: dict, method: str, path: str, used: set[str]) -> str:
        raw = op.get("operationId") or f"{method}_{path}"
        name = re.sub(r"[^a-zA-Z0-9_-]", "_", str(raw))[:60] or "op"
        candidate, i = name, 1
        while candidate in used:
            candidate = f"{name}_{i}"[:64]
            i += 1
        used.add(candidate)
        return candidate

    def _deref(self, node: Any, stack: tuple[str, ...] = ()) -> Any:
        """Resolve local `$ref` pointers (`#/...`) within the spec, cycle-guarded."""
        if isinstance(node, dict):
            ref = node.get("$ref")
            if isinstance(ref, str) and ref.startswith("#/"):
                if ref in stack:
                    return {}
                return self._deref(self._lookup_pointer(ref), stack + (ref,))
            return {k: self._deref(v, stack) for k, v in node.items()}
        if isinstance(node, list):
            return [self._deref(x, stack) for x in node]
        return node

    def _lookup_pointer(self, ref: str) -> dict:
        cur: Any = self._spec
        for part in ref[2:].split("/"):
            key = part.replace("~1", "/").replace("~0", "~")
            cur = cur.get(key) if isinstance(cur, dict) else None
            if cur is None:
                return {}
        return cur if isinstance(cur, dict) else {}

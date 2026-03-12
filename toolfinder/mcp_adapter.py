from __future__ import annotations

import asyncio
import contextlib
import copy
import json
import logging
import os
import shutil
import uuid
from collections import deque
from dataclasses import dataclass
from typing import Any


logger = logging.getLogger(__name__)

JsonDict = dict[str, Any]
NormalizedTool = dict[str, Any]
RequestId = int | str


class MCPClientError(RuntimeError):
    pass


class MCPResponseError(MCPClientError):
    def __init__(self, message: str, error: JsonDict) -> None:
        super().__init__(message)
        self.error = error


@dataclass(frozen=True)
class ServerProcessConfig:
    server_name: str
    command: str
    args: tuple[str, ...] = ()
    env: dict[str, str] | None = None
    cwd: str | None = None


class DynamicMCPClient:
    def __init__(
        self,
        server_name: str,
        command: str,
        args: list[str] | tuple[str, ...] | None = None,
        env: dict[str, str] | None = None,
        cwd: str | None = None,
        protocol_version: str = "2025-06-18",
        startup_timeout_s: float = 45.0,
        request_timeout_s: float = 30.0,
    ) -> None:
        self.server_name = server_name
        self.command = command
        self.args = tuple(args or ())
        self.env = dict(env) if env is not None else None
        self.cwd = cwd
        self.protocol_version = protocol_version
        self.startup_timeout_s = startup_timeout_s
        self.request_timeout_s = request_timeout_s

        self.process: asyncio.subprocess.Process | None = None
        self._stdout_task: asyncio.Task[None] | None = None
        self._stderr_task: asyncio.Task[None] | None = None
        self._pending_requests: dict[RequestId, asyncio.Future[JsonDict]] = {}
        self._stderr_lines: deque[str] = deque(maxlen=200)
        self._server_info: JsonDict | None = None
        self._server_capabilities: JsonDict | None = None
        self._tools_cache: list[NormalizedTool] | None = None
        self._closed = False
        self._started = False

    @property
    def stderr_tail(self) -> list[str]:
        return list(self._stderr_lines)

    @property
    def server_info(self) -> JsonDict | None:
        return self._server_info

    @property
    def server_capabilities(self) -> JsonDict | None:
        return self._server_capabilities

    async def __aenter__(self) -> DynamicMCPClient:
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        await self.close()

    async def initialize_and_get_tools(self) -> list[NormalizedTool]:
        if self._tools_cache is not None:
            return copy.deepcopy(self._tools_cache)

        if not self._started:
            await self._start_process()
            try:
                await asyncio.wait_for(self._initialize(), timeout=self.startup_timeout_s)
            except Exception:
                await self.close()
                raise
            self._started = True

        self._tools_cache = await self._list_tools()
        logger.debug(
            "Initialized MCP server %s with %s tools",
            self.server_name,
            len(self._tools_cache),
        )
        return copy.deepcopy(self._tools_cache)

    async def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if not self._started:
            await self.initialize_and_get_tools()

        response = await self._request(
            "tools/call",
            params={
                "name": tool_name,
                "arguments": arguments,
            },
        )
        result = response.get("result")
        if not isinstance(result, dict):
            raise MCPClientError(f"{self.server_name}: tools/call response missing result object")

        if result.get("isError") is True:
            raise MCPClientError(
                f"{self.server_name}: tool {tool_name} returned an execution error: "
                f"{json.dumps(result, ensure_ascii=True)}"
            )

        return result

    async def _start_process(self) -> None:
        if self._started:
            return

        merged_env = os.environ.copy()
        if self.env is not None:
            merged_env.update(self.env)

        spawn_command = self._build_spawn_command()
        logger.debug("Starting MCP server %s with command %s", self.server_name, spawn_command)

        self.process = await asyncio.create_subprocess_exec(
            *spawn_command,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=self.cwd,
            env=merged_env,
        )

        if self.process.stdout is None or self.process.stdin is None or self.process.stderr is None:
            raise MCPClientError(f"{self.server_name}: failed to open stdio pipes")

        self._stdout_task = asyncio.create_task(
            self._stdout_loop(),
            name=f"{self.server_name}-stdout",
        )
        self._stderr_task = asyncio.create_task(
            self._stderr_loop(),
            name=f"{self.server_name}-stderr",
        )

    def _build_spawn_command(self) -> list[str]:
        resolved_command = shutil.which(self.command) or self.command
        command_suffix = os.path.splitext(resolved_command)[1].lower()

        if os.name == "nt":
            direct_node_launch = self._build_windows_node_shim_command(resolved_command)
            if direct_node_launch is not None:
                return direct_node_launch

        if os.name == "nt" and command_suffix in {".cmd", ".bat"}:
            comspec = os.environ.get("COMSPEC", "cmd.exe")
            return [comspec, "/d", "/c", resolved_command, *self.args]

        return [resolved_command, *self.args]

    def _build_windows_node_shim_command(self, resolved_command: str) -> list[str] | None:
        command_name = os.path.basename(resolved_command).lower()
        if command_name not in {"npx.cmd", "npx", "npm.cmd", "npm"}:
            return None

        install_dir = os.path.dirname(resolved_command)
        node_executable = os.path.join(install_dir, "node.exe")
        cli_name = "npx-cli.js" if command_name.startswith("npx") else "npm-cli.js"
        cli_entrypoint = os.path.join(install_dir, "node_modules", "npm", "bin", cli_name)

        if not (os.path.exists(node_executable) and os.path.exists(cli_entrypoint)):
            return None

        return [node_executable, cli_entrypoint, *self.args]

    async def _initialize(self) -> None:
        response = await self._request(
            "initialize",
            {
                "protocolVersion": self.protocol_version,
                "capabilities": {
                    "tools": {
                        "listChanged": False,
                    },
                },
                "clientInfo": {
                    "name": "scalable-router-client",
                    "version": "1.0.0",
                },
            },
            timeout_s=self.startup_timeout_s,
        )
        result = response.get("result")
        if not isinstance(result, dict):
            raise MCPClientError(f"{self.server_name}: initialize response missing result")

        self._server_info = (
            result.get("serverInfo")
            if isinstance(result.get("serverInfo"), dict)
            else None
        )
        self._server_capabilities = (
            result.get("capabilities")
            if isinstance(result.get("capabilities"), dict)
            else None
        )
        await self._notify("notifications/initialized")

    async def _request(
        self,
        method: str,
        params: JsonDict | None = None,
        timeout_s: float | None = None,
    ) -> JsonDict:
        if self.process is None or self.process.stdin is None:
            raise MCPClientError(f"{self.server_name}: process is not running")

        request_id = uuid.uuid4().hex
        message: JsonDict = {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": method,
        }
        if params is not None:
            message["params"] = params

        loop = asyncio.get_running_loop()
        future: asyncio.Future[JsonDict] = loop.create_future()
        self._pending_requests[request_id] = future

        try:
            await self._send_message(message)
            response = await asyncio.wait_for(future, timeout=timeout_s or self.request_timeout_s)
        except Exception:
            pending = self._pending_requests.pop(request_id, None)
            if pending is not None and not pending.done():
                pending.cancel()
            raise

        if "error" in response:
            error = response["error"]
            if isinstance(error, dict):
                raise MCPResponseError(
                    f"{self.server_name}: {error.get('message', 'unknown MCP error')}",
                    error,
                )
            raise MCPClientError(f"{self.server_name}: malformed error response for {method}")

        return response

    async def _notify(self, method: str, params: JsonDict | None = None) -> None:
        message: JsonDict = {
            "jsonrpc": "2.0",
            "method": method,
        }
        if params is not None:
            message["params"] = params
        await self._send_message(message)

    async def _list_tools(self) -> list[NormalizedTool]:
        tools: list[NormalizedTool] = []
        cursor: str | None = None
        seen_cursors: set[str] = set()

        while True:
            params: JsonDict | None = {"cursor": cursor} if cursor is not None else None
            response = await self._request("tools/list", params=params)
            result = response.get("result")
            if not isinstance(result, dict):
                raise MCPClientError(f"{self.server_name}: tools/list response missing result")

            raw_tools = result.get("tools")
            if not isinstance(raw_tools, list):
                raise MCPClientError(f"{self.server_name}: tools/list response missing tools array")

            tools.extend(self._normalize_tools(raw_tools))

            next_cursor = result.get("nextCursor")
            if not isinstance(next_cursor, str) or not next_cursor:
                break
            if next_cursor in seen_cursors:
                raise MCPClientError(f"{self.server_name}: duplicate pagination cursor received")
            seen_cursors.add(next_cursor)
            cursor = next_cursor

        return tools

    def _normalize_tools(self, raw_tools: list[Any]) -> list[NormalizedTool]:
        normalized: list[NormalizedTool] = []
        for raw_tool in raw_tools:
            if not isinstance(raw_tool, dict):
                continue

            tool_name = raw_tool.get("name")
            if not isinstance(tool_name, str) or not tool_name:
                continue

            description = raw_tool.get("description")
            input_schema = raw_tool.get("inputSchema")
            strict_schema = self._inject_additional_properties_false(
                input_schema if isinstance(input_schema, dict) else {}
            )

            normalized.append(
                {
                    "server_name": self.server_name,
                    "tool_name": tool_name,
                    "description": description if isinstance(description, str) else "",
                    "inputSchema": strict_schema,
                }
            )

        return normalized

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

    async def _send_message(self, message: JsonDict) -> None:
        if self.process is None or self.process.stdin is None:
            raise MCPClientError(f"{self.server_name}: process is not running")

        payload = json.dumps(message, separators=(",", ":")) + "\n"
        self.process.stdin.write(payload.encode("utf-8"))
        await self.process.stdin.drain()

    async def _stdout_loop(self) -> None:
        assert self.process is not None
        assert self.process.stdout is not None

        try:
            while True:
                line = await self.process.stdout.readline()
                if not line:
                    break

                raw_line = line.decode("utf-8", errors="replace").strip()
                if not raw_line:
                    continue

                try:
                    message = json.loads(raw_line)
                except json.JSONDecodeError:
                    self._stderr_lines.append(f"[stdout] {raw_line}")
                    continue

                if not isinstance(message, dict):
                    self._stderr_lines.append(f"[stdout-json] {raw_line}")
                    continue

                request_id = message.get("id")
                if isinstance(request_id, (int, str)):
                    future = self._pending_requests.pop(request_id, None)
                    if future is not None and not future.done():
                        future.set_result(message)
        except Exception as exc:
            self._fail_pending_requests(exc)
            raise
        finally:
            if self.process.returncode is None:
                await self.process.wait()
            self._fail_pending_requests(
                MCPClientError(
                    f"{self.server_name}: server exited with code {self.process.returncode}; "
                    f"stderr tail: {' | '.join(self.stderr_tail[-10:])}"
                )
            )

    async def _stderr_loop(self) -> None:
        assert self.process is not None
        assert self.process.stderr is not None

        while True:
            line = await self.process.stderr.readline()
            if not line:
                break
            decoded = line.decode("utf-8", errors="replace").rstrip()
            if decoded:
                self._stderr_lines.append(decoded)

    def _fail_pending_requests(self, exc: BaseException) -> None:
        for request_id, future in list(self._pending_requests.items()):
            if not future.done():
                future.set_exception(exc)
            self._pending_requests.pop(request_id, None)

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True

        process = self.process
        if process is not None and process.stdin is not None:
            with contextlib.suppress(Exception):
                process.stdin.close()

        if process is not None:
            try:
                await asyncio.wait_for(process.wait(), timeout=3.0)
            except asyncio.TimeoutError:
                with contextlib.suppress(ProcessLookupError):
                    process.terminate()
                try:
                    await asyncio.wait_for(process.wait(), timeout=3.0)
                except asyncio.TimeoutError:
                    with contextlib.suppress(ProcessLookupError):
                        process.kill()
                    await process.wait()

        for task in (self._stdout_task, self._stderr_task):
            if task is not None and not task.done():
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task

        self._fail_pending_requests(MCPClientError(f"{self.server_name}: client closed"))
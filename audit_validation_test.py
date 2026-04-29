from __future__ import annotations

import asyncio
import json
import os
import socket
import sys
import threading
import time
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import uvicorn

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.chdir(REPO_ROOT)

import toolfinder.dynamic_faiss_router as router_module
from Enterprise.runtime.api import create_app
from Enterprise.runtime.config import EnterpriseConfig
from Enterprise.runtime.executor import HybridToolExecutor
from Enterprise.runtime.openclaw_hybrid_pipeline import FallbackStrategy, OpenClawHybridPipeline, OpenClawSessionDriver
from Enterprise.runtime.policy import PolicyEngine, ToolPolicy
from Enterprise.runtime.registry import HybridToolRegistry


class _DummySentenceTransformer:
    def __init__(self, model_name: str, device: str | None = None) -> None:
        self.model_name = model_name
        self.device = device

    def get_sentence_embedding_dimension(self) -> int:
        return 4

    def encode(self, texts: list[str], batch_size: int | None = None, convert_to_numpy: bool = True):
        del batch_size, convert_to_numpy
        embeddings = []
        for text in texts:
            normalized = text.lower()
            if "read" in normalized or "file" in normalized or "windows/system32" in normalized:
                embeddings.append([1.0, 0.0, 0.0, 0.0])
            else:
                embeddings.append([0.0, 1.0, 0.0, 0.0])
        return np.asarray(embeddings, dtype=np.float32)


class _RecordingClient:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        self.calls.append({"tool_name": tool_name, "arguments": dict(arguments)})
        return {
            "content": [
                {
                    "type": "text",
                    "text": f"{tool_name}::{arguments.get('path', '')}",
                }
            ]
        }


class _PromptAwareBackend:
    async def complete(self, prompt: str) -> str:
        path = "../../Windows/System32" if "../../Windows/System32" in prompt else "examples/sample.txt"
        return json.dumps(
            [
                {
                    "thought": "normalize tool_name",
                    "action": "call_tool",
                    "tool_name": "read_file",
                    "arguments": {"path": path},
                },
                {
                    "thought": "done",
                    "status": "complete",
                    "answer": "completed",
                },
            ]
        )


def _pick_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


async def _build_registry() -> HybridToolRegistry:
    router_module.SentenceTransformer = _DummySentenceTransformer

    registry = HybridToolRegistry(model_name="audit-dummy-model")
    await registry.upsert_server_tools(
        "server",
        [
            {
                "tool_name": "read_file",
                "description": "Read a file from disk",
                "inputSchema": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"],
                },
            }
        ],
    )
    return registry


def _start_server(app) -> tuple[uvicorn.Server, threading.Thread, int]:
    port = _pick_port()
    config = uvicorn.Config(
        app,
        host="127.0.0.1",
        port=port,
        log_level="warning",
        access_log=False,
        lifespan="on",
    )
    server = uvicorn.Server(config)
    server.install_signal_handlers = lambda: None

    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    deadline = time.time() + 30.0
    while not server.started:
        if not thread.is_alive():
            raise RuntimeError("Uvicorn server exited before startup completed")
        if time.time() > deadline:
            raise TimeoutError("Timed out waiting for Uvicorn startup")
        time.sleep(0.05)

    return server, thread, port


def _build_app():
    registry = asyncio.run(_build_registry())
    recording_client = _RecordingClient()

    executor = HybridToolExecutor({"server": recording_client})
    session_driver = OpenClawSessionDriver(backend=_PromptAwareBackend())
    pipeline = OpenClawHybridPipeline(
        registry=registry,
        session_driver=session_driver,
        executor=executor,
        policy_engine=PolicyEngine(
            ToolPolicy(allowed_servers={"server"}),
            workspace_root=str(REPO_ROOT),
        ),
        config=EnterpriseConfig(top_k=1, min_score=0.01, max_turns=2),
        fallback_strategy=FallbackStrategy.ERROR,
    )

    app = create_app(
        workspace_root=str(REPO_ROOT),
        pipeline=pipeline,
        registry=registry,
    )
    return app, recording_client


def _assert_path_traversal_rejected(client: httpx.Client) -> None:
    response = client.post("/execute", json={"intent": "read file ../../Windows/System32"})
    assert response.status_code == 403, response.text
    payload = response.json()
    assert "error" in payload
    assert "Path Traversal error" in payload["error"]


def _assert_schema_validation_rejected(client: httpx.Client) -> None:
    response = client.post("/execute", json={})
    assert response.status_code == 422, response.text
    payload = response.json()
    assert "detail" in payload


def _assert_tool_name_normalizes_to_executor(client: httpx.Client, recording_client: _RecordingClient) -> None:
    response = client.post("/execute", json={"intent": "read file"})
    assert response.status_code == 200, response.text

    payload = response.json()
    assert "execution_output" in payload
    assert payload["execution_output"]["status"] == "complete"
    assert recording_client.calls, "executor was never called"
    assert recording_client.calls[0]["tool_name"] == "read_file"
    assert recording_client.calls[0]["arguments"]["path"] == "examples/sample.txt"
    assert payload["execution_output"]["tool_calls"][0]["server_name"] == "server"
    assert payload["execution_output"]["tool_calls"][0]["tool_name"] == "read_file"


def main() -> int:
    app, recording_client = _build_app()
    server, thread, port = _start_server(app)

    try:
        with httpx.Client(base_url=f"http://127.0.0.1:{port}", timeout=30.0) as client:
            _assert_path_traversal_rejected(client)
            _assert_schema_validation_rejected(client)
            _assert_tool_name_normalizes_to_executor(client, recording_client)
    finally:
        server.should_exit = True
        thread.join(timeout=30.0)

    print("audit_validation_test: passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
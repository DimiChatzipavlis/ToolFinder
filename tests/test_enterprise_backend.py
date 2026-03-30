from __future__ import annotations

import asyncio
from pathlib import Path

from Enterprise.runtime.openclaw_backend import OpenClawCliBackend, OpenClawHttpBackend, build_openclaw_backend
from Enterprise.runtime.registry import HybridToolRegistry
from Enterprise.runtime.realtime_service import WorkspaceChangeTracker


def test_openclaw_backend_payload_modes() -> None:
    ollama_backend = OpenClawHttpBackend(
        endpoint="http://localhost:11434/api/generate",
        model="llama3.2",
        api_mode="ollama-generate",
    )
    openai_backend = OpenClawHttpBackend(
        endpoint="http://localhost:8000/v1/chat/completions",
        model="oss-model",
        api_mode="openai-chat",
    )

    ollama_payload = ollama_backend._build_payload("hello")
    openai_payload = openai_backend._build_payload("hello")

    assert ollama_payload["stream"] is False
    assert "messages" not in ollama_payload
    assert isinstance(openai_payload.get("messages"), list)


def test_openclaw_backend_extract_text_variants() -> None:
    backend = OpenClawHttpBackend(
        endpoint="http://localhost:11434/api/generate",
        model="llama3.2",
    )

    assert backend._extract_text({"response": "ok"}) == "ok"
    assert backend._extract_text({"message": {"content": "chat"}}) == "chat"
    assert backend._extract_text({"choices": [{"text": "legacy"}]}) == "legacy"


def test_workspace_change_tracker_detects_modification(tmp_path: Path) -> None:
    file_path = tmp_path / "sample.py"
    file_path.write_text("print('a')", encoding="utf-8")

    tracker = WorkspaceChangeTracker(root=tmp_path)
    changed, _ = tracker.detect_changes()
    assert changed is False

    file_path.write_text("print('b')", encoding="utf-8")
    changed, touched = tracker.detect_changes()

    assert changed is True
    assert "sample.py" in touched


def test_build_openclaw_backend_http_mode() -> None:
    backend = build_openclaw_backend(
        backend_kind="http",
        endpoint="http://localhost:11434/api/generate",
        model="llama3.2",
        api_mode="ollama-generate",
    )
    assert isinstance(backend, OpenClawHttpBackend)


def test_openclaw_cli_extract_output() -> None:
    stdout = """\
noise line
{"event":"progress"}
{"answer":"final answer"}
"""
    extracted = OpenClawCliBackend._extract_cli_output(stdout)
    assert extracted == "final answer"


def test_registry_keyword_fallback_when_router_init_fails(monkeypatch) -> None:
    import Enterprise.runtime.registry as registry_module

    class BrokenRouter:
        def __init__(self, *args: object, **kwargs: object) -> None:
            del args, kwargs
            raise RuntimeError("router init failed")

    monkeypatch.setattr(registry_module, "UniversalMCPRouter", BrokenRouter)

    registry = HybridToolRegistry(model_name="broken-model")
    asyncio.run(
        registry.upsert_server_tools(
            "filesystem",
            [
                {
                    "tool_name": "list_directory",
                    "description": "List files and directories",
                    "inputSchema": {
                        "type": "object",
                        "properties": {"path": {"type": "string"}},
                    },
                }
            ],
        )
    )

    candidates = asyncio.run(registry.route("list files", k=1, min_score=0.15))
    assert candidates
    assert candidates[0].tool_name == "list_directory"

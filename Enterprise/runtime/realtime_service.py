from __future__ import annotations

import asyncio
import hashlib
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .orchestrator import HybridEnterpriseOrchestrator


class WorkspaceChangeTracker:
    """Poll-based filesystem tracker for enterprise-safe real-time loops."""

    def __init__(
        self,
        root: Path,
        include_suffixes: set[str] | None = None,
        exclude_dirs: set[str] | None = None,
    ) -> None:
        self.root = root.resolve()
        self.include_suffixes = include_suffixes or {".py", ".md", ".json", ".yaml", ".yml", ".toml"}
        self.exclude_dirs = exclude_dirs or {".git", "__pycache__", ".pytest_cache", ".venv", "venv"}
        self._last_fingerprint: str | None = None

    def detect_changes(self) -> tuple[bool, list[str]]:
        paths = self._collect_paths()
        fingerprint = self._fingerprint(paths)
        if self._last_fingerprint is None:
            self._last_fingerprint = fingerprint
            return False, []

        if fingerprint == self._last_fingerprint:
            return False, []

        self._last_fingerprint = fingerprint
        return True, [str(path.relative_to(self.root)) for path in paths[:10]]

    def _collect_paths(self) -> list[Path]:
        collected: list[Path] = []
        for path in self.root.rglob("*"):
            if not path.is_file():
                continue
            if any(part in self.exclude_dirs for part in path.parts):
                continue
            if path.suffix and path.suffix.lower() not in self.include_suffixes:
                continue
            collected.append(path)
        collected.sort()
        return collected

    def _fingerprint(self, paths: list[Path]) -> str:
        digest = hashlib.sha256()
        for path in paths:
            stat = path.stat()
            digest.update(str(path.relative_to(self.root)).encode("utf-8"))
            digest.update(str(stat.st_mtime_ns).encode("utf-8"))
            digest.update(str(stat.st_size).encode("utf-8"))
        return digest.hexdigest()


class RealTimeHybridService:
    def __init__(
        self,
        orchestrator: HybridEnterpriseOrchestrator,
        tracker: WorkspaceChangeTracker,
        query_builder: Callable[[list[str]], str],
        poll_interval_s: float = 1.5,
        run_on_startup: bool = True,
        error_backoff_s: float = 3.0,
    ) -> None:
        if poll_interval_s <= 0:
            raise ValueError("poll_interval_s must be > 0")
        if error_backoff_s < 0:
            raise ValueError("error_backoff_s must be >= 0")

        self.orchestrator = orchestrator
        self.tracker = tracker
        self.query_builder = query_builder
        self.poll_interval_s = poll_interval_s
        self.run_on_startup = run_on_startup
        self.error_backoff_s = error_backoff_s
        self._started = False

    async def run_forever(self) -> None:
        session_seq = 1
        while True:
            try:
                changed, touched = self.tracker.detect_changes()
                should_run = changed or (self.run_on_startup and not self._started)
                self._started = True
                if should_run:
                    query = self.query_builder(touched)
                    session_id = f"realtime-{session_seq}"
                    session_seq += 1
                    started = time.perf_counter()
                    result = await self.orchestrator.run(session_id=session_id, user_query=query)
                    elapsed_ms = (time.perf_counter() - started) * 1000.0
                    print(
                        "[realtime]",
                        f"session={session_id}",
                        f"status={result.status}",
                        f"turns={result.turns}",
                        f"elapsed_ms={elapsed_ms:.2f}",
                    )
                    print("[realtime] answer:", result.answer)
            except Exception as exc:
                print("[realtime] loop error:", str(exc))
                if self.error_backoff_s > 0:
                    await asyncio.sleep(self.error_backoff_s)
                continue

            await asyncio.sleep(self.poll_interval_s)

    async def run_for_cycles(self, cycles: int) -> None:
        if cycles < 1:
            raise ValueError("cycles must be >= 1")
        for _ in range(cycles):
            changed, touched = self.tracker.detect_changes()
            should_run = changed or (self.run_on_startup and not self._started)
            self._started = True
            if should_run:
                query = self.query_builder(touched)
                result = await self.orchestrator.run(session_id="realtime-test", user_query=query)
                print("[realtime-test]", result.status, result.answer)
            await asyncio.sleep(self.poll_interval_s)

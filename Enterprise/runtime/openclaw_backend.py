from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
from typing import Any
from urllib.parse import urlparse

import httpx


logger = logging.getLogger(__name__)


class OpenClawHttpBackend:
    """OSS planner backend with OpenClaw-compatible prompt contracts.

    Supports:
    - `ollama-generate`: POST /api/generate
    - `openai-chat`: OpenAI-compatible /v1/chat/completions
    """

    def __init__(
        self,
        endpoint: str,
        model: str,
        api_mode: str = "ollama-generate",
        api_key: str | None = None,
        extra_headers: dict[str, str] | None = None,
        timeout_s: float = 60.0,
        max_retries: int = 2,
        retry_backoff_s: float = 0.75,
    ) -> None:
        parsed = urlparse(endpoint)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("endpoint must be a valid http(s) URL")
        if max_retries < 0:
            raise ValueError("max_retries must be >= 0")

        self.endpoint = endpoint
        self.model = model
        self.api_mode = api_mode
        self.api_key = api_key
        self.extra_headers = extra_headers or {}
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self.retry_backoff_s = retry_backoff_s

    async def complete(self, prompt: str) -> str:
        headers = {"Content-Type": "application/json", **self.extra_headers}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload = self._build_payload(prompt)

        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                async with httpx.AsyncClient(timeout=self.timeout_s) as client:
                    response = await client.post(self.endpoint, json=payload, headers=headers)
                    response.raise_for_status()
                    body = response.json()
                text = self._extract_text(body)
                if text:
                    return text
                raise RuntimeError(f"unrecognized planner backend response shape: {body}")
            except (httpx.HTTPError, ValueError, RuntimeError) as exc:
                last_error = exc
                if attempt >= self.max_retries:
                    break
                logger.warning(
                    "Backend probe failed on attempt %s: %s. Retrying...",
                    attempt + 1,
                    exc,
                )
                await asyncio.sleep(self.retry_backoff_s * (attempt + 1))

        raise RuntimeError(f"planner backend request failed after retries: {last_error}")

    async def probe(self) -> bool:
        """Fast readiness probe for deployment checks."""
        try:
            async with httpx.AsyncClient(timeout=min(self.timeout_s, 10.0)) as client:
                response = await client.head(self.endpoint)
                if response.status_code == 200:
                    return True
        except httpx.HTTPError:
            pass

        # Fallback: many local OSS gateways do not implement HEAD.
        try:
            async with httpx.AsyncClient(timeout=min(self.timeout_s, 10.0)) as client:
                response = await client.get(self.endpoint)
            return response.status_code == 200
        except httpx.HTTPError:
            return False

    def _build_payload(self, prompt: str) -> dict[str, Any]:
        if self.api_mode == "ollama-generate":
            return {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0},
            }

        if self.api_mode == "openai-chat":
            return {
                "model": self.model,
                "temperature": 0,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
            }

        raise ValueError(f"unsupported api_mode: {self.api_mode}")

    def _extract_text(self, body: dict[str, Any]) -> str | None:
        if isinstance(body.get("response"), str):
            return str(body["response"])

        message = body.get("message")
        if isinstance(message, dict) and isinstance(message.get("content"), str):
            return str(message["content"])

        choices = body.get("choices")
        if isinstance(choices, list) and choices:
            first = choices[0]
            message = first.get("message", {}) if isinstance(first, dict) else {}
            content = message.get("content")
            if isinstance(content, str):
                return content

            text = first.get("text") if isinstance(first, dict) else None
            if isinstance(text, str):
                return text

        output = body.get("output")
        if isinstance(output, str):
            return output

        if isinstance(output, list):
            content_fragments: list[str] = []
            for item in output:
                if isinstance(item, dict) and isinstance(item.get("content"), str):
                    content_fragments.append(item["content"])
            if content_fragments:
                return "\n".join(content_fragments)

        return None


class OpenClawCliBackend:
    """OpenClaw CLI backend for local/edge deployments.

    This backend calls the `openclaw` executable directly and extracts a
    textual response from stdout.
    """

    def __init__(
        self,
        binary: str = "openclaw",
        timeout_s: float = 90.0,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        extra_args: list[str] | None = None,
    ) -> None:
        self.binary = binary
        self.timeout_s = timeout_s
        self.cwd = cwd
        self.env = dict(env) if env is not None else None
        self.extra_args = list(extra_args or [])

    async def complete(self, prompt: str) -> str:
        resolved_binary = self._resolve_binary()
        if resolved_binary is None:
            raise RuntimeError(
                "OpenClaw CLI binary not found. Install with `npm install -g openclaw` "
                "or provide OPENCLAW_CLI_BIN."
            )

        command = [
            resolved_binary,
            "agent",
            "--message",
            prompt,
            "--json",
            *self.extra_args,
        ]
        process = await asyncio.create_subprocess_exec(
            *command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=self.cwd,
            env=self._build_env(),
        )
        try:
            stdout_bytes, stderr_bytes = await asyncio.wait_for(process.communicate(), timeout=self.timeout_s)
        except TimeoutError:
            process.kill()
            await process.wait()
            raise RuntimeError("OpenClaw CLI backend timed out")

        stdout_text = stdout_bytes.decode("utf-8", errors="replace")
        stderr_text = stderr_bytes.decode("utf-8", errors="replace")
        if process.returncode != 0:
            raise RuntimeError(
                "OpenClaw CLI backend failed "
                f"(exit={process.returncode}): {stderr_text or stdout_text}"
            )

        extracted = self._extract_cli_output(stdout_text)
        if extracted:
            return extracted
        raise RuntimeError("OpenClaw CLI backend returned no parseable response")

    def _build_env(self) -> dict[str, str]:
        merged = os.environ.copy()
        if self.env is not None:
            merged.update(self.env)
        return merged

    def _resolve_binary(self) -> str | None:
        if shutil.which(self.binary):
            return self.binary
        if os.name == "nt" and not self.binary.lower().endswith(".cmd"):
            candidate = f"{self.binary}.cmd"
            if shutil.which(candidate):
                return candidate
        return None

    @staticmethod
    def _extract_cli_output(stdout_text: str) -> str | None:
        lines = [line.strip() for line in stdout_text.splitlines() if line.strip()]
        if not lines:
            return None

        # OpenClaw CLI often prints JSON per line in --json mode; parse last JSON line first.
        for line in reversed(lines):
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue

            if isinstance(payload, dict):
                for key in ("answer", "response", "content", "message"):
                    value = payload.get(key)
                    if isinstance(value, str) and value.strip():
                        return value.strip()
                    if isinstance(value, dict) and isinstance(value.get("content"), str):
                        return value["content"].strip()

        # Fallback to raw text if no JSON line parsed.
        return "\n".join(lines)


def build_openclaw_backend(
    backend_kind: str = "http",
    *,
    endpoint: str = "http://127.0.0.1:11434/api/generate",
    model: str = "llama3.2",
    api_mode: str = "ollama-generate",
    api_key: str | None = None,
    timeout_s: float = 60.0,
    cli_binary: str = "openclaw",
    cli_extra_args: list[str] | None = None,
) -> OpenClawHttpBackend | OpenClawCliBackend:
    if backend_kind == "http":
        return OpenClawHttpBackend(
            endpoint=endpoint,
            model=model,
            api_mode=api_mode,
            api_key=api_key,
            timeout_s=timeout_s,
        )
    if backend_kind == "cli":
        return OpenClawCliBackend(
            binary=cli_binary,
            timeout_s=timeout_s,
            extra_args=cli_extra_args,
        )
    raise ValueError(f"unsupported backend_kind: {backend_kind}")

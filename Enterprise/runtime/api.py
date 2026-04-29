from __future__ import annotations

import asyncio
import os
from contextlib import asynccontextmanager
from dataclasses import asdict
from typing import Any
from uuid import uuid4

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, model_validator

from .config import EnterpriseConfig
from .contracts import HybridPipelineResult
from .executor import HybridToolExecutor
from .openclaw_hybrid_pipeline import OpenClawHybridPipeline, OpenClawSessionDriver
from .policy import PolicyEngine, SecurityViolation
from .registry import HybridToolRegistry


class ExecuteIntentRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True, strict=True)

    intent: str = Field(min_length=1, description="Natural language intent to execute.")


class ExecuteIntentResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    session_id: str
    execution_output: dict[str, Any] | None = None
    error: str | None = None

    @model_validator(mode="after")
    def _validate_xor(self) -> "ExecuteIntentResponse":
        has_output = self.execution_output is not None
        has_error = self.error is not None
        if has_output == has_error:
            raise ValueError("exactly one of execution_output or error must be set")
        return self


def create_app(
    *,
    workspace_root: str,
    pipeline: OpenClawHybridPipeline | None = None,
    registry: HybridToolRegistry | None = None,
    executor: HybridToolExecutor | None = None,
    session_driver: OpenClawSessionDriver | None = None,
    config: EnterpriseConfig | None = None,
) -> FastAPI:
    normalized_root = os.path.abspath(workspace_root)
    if not normalized_root:
        raise ValueError("workspace_root is required")
    if not os.path.isdir(normalized_root):
        raise ValueError(f"workspace_root must be an existing directory: {workspace_root}")

    runtime_config = config or EnterpriseConfig()
    runtime_pipeline = pipeline
    runtime_registry = registry

    if runtime_pipeline is None:
        if runtime_registry is None or executor is None or session_driver is None:
            raise ValueError(
                "pipeline or registry/executor/session_driver must be supplied to create_app"
            )

        runtime_pipeline = OpenClawHybridPipeline(
            registry=runtime_registry,
            session_driver=session_driver,
            executor=executor,
            policy_engine=PolicyEngine(workspace_root=normalized_root),
            config=runtime_config,
        )
    elif runtime_registry is None:
        runtime_registry = getattr(runtime_pipeline, "registry", None)

    def _teardown_runtime() -> None:
        seen_ids: set[int] = set()
        for candidate in (runtime_registry, getattr(runtime_pipeline, "registry", None)):
            if candidate is None:
                continue
            candidate_id = id(candidate)
            if candidate_id in seen_ids:
                continue
            seen_ids.add(candidate_id)

            teardown = getattr(candidate, "teardown", None)
            if callable(teardown):
                teardown()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.pipeline = runtime_pipeline
        app.state.registry = runtime_registry
        app.state.workspace_root = normalized_root
        try:
            yield
        finally:
            await asyncio.to_thread(_teardown_runtime)

    app = FastAPI(
        title="ToolFinder Execution API",
        version="1.0",
        lifespan=lifespan,
    )

    @app.post("/execute", response_model=ExecuteIntentResponse, response_model_exclude_none=True)
    async def execute(payload: ExecuteIntentRequest) -> ExecuteIntentResponse | JSONResponse:
        session_id = f"api-{uuid4().hex}"

        try:
            result = await runtime_pipeline.run(session_id=session_id, user_query=payload.intent)
        except SecurityViolation as exc:
            return JSONResponse(status_code=403, content={"error": f"Path Traversal error: {exc}"})
        except Exception as exc:
            return ExecuteIntentResponse(session_id=session_id, error=f"execution failed: {exc}")

        if _is_security_fault(result):
            return JSONResponse(status_code=403, content={"error": _security_error_message(result)})

        return ExecuteIntentResponse(session_id=session_id, execution_output=asdict(result))

    return app


def _is_security_fault(result: HybridPipelineResult) -> bool:
    if result.status == "degraded_security_fault":
        return True

    answer = (result.answer or "").lower()
    if "workspace_root" in answer:
        return True
    if "path traversal" in answer:
        return True
    return False


def _security_error_message(result: HybridPipelineResult) -> str:
    if result.answer:
        return result.answer

    response_error = getattr(result.openclaw_response, "error", None)
    if isinstance(response_error, str) and response_error.strip():
        return response_error

    return "Path Traversal error"
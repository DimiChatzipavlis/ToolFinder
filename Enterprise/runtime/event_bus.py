from __future__ import annotations

import asyncio
import inspect
from collections.abc import Awaitable, Callable
from typing import Any


EventHandler = Callable[[dict[str, Any]], None | Awaitable[None]]


class EnterpriseEventBus:
    def __init__(
        self,
        *,
        handler_timeout_s: float | None = None,
        continue_on_error: bool = True,
        max_handler_errors: int = 200,
    ) -> None:
        if max_handler_errors < 1:
            raise ValueError("max_handler_errors must be >= 1")
        self._handlers: list[EventHandler] = []
        self._handler_timeout_s = handler_timeout_s
        self._continue_on_error = continue_on_error
        self._max_handler_errors = max_handler_errors
        self._handler_errors: list[str] = []

    def subscribe(self, handler: EventHandler) -> Callable[[], None]:
        self._handlers.append(handler)

        def unsubscribe() -> None:
            if handler in self._handlers:
                self._handlers.remove(handler)

        return unsubscribe

    async def publish(self, event: dict[str, Any]) -> None:
        for handler in list(self._handlers):
            try:
                result = handler(event)
                if inspect.isawaitable(result):
                    if self._handler_timeout_s is not None:
                        await asyncio.wait_for(result, timeout=self._handler_timeout_s)
                    else:
                        await result
            except Exception as exc:
                self._handler_errors.append(str(exc))
                overflow = len(self._handler_errors) - self._max_handler_errors
                if overflow > 0:
                    del self._handler_errors[:overflow]
                if not self._continue_on_error:
                    raise

    def recent_errors(self) -> list[str]:
        return list(self._handler_errors)

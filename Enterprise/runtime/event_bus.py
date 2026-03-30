from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable
from typing import Any


EventHandler = Callable[[dict[str, Any]], None | Awaitable[None]]


class EnterpriseEventBus:
    def __init__(self) -> None:
        self._handlers: list[EventHandler] = []

    def subscribe(self, handler: EventHandler) -> Callable[[], None]:
        self._handlers.append(handler)

        def unsubscribe() -> None:
            if handler in self._handlers:
                self._handlers.remove(handler)

        return unsubscribe

    async def publish(self, event: dict[str, Any]) -> None:
        for handler in list(self._handlers):
            result = handler(event)
            if inspect.isawaitable(result):
                await result

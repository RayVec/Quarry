from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import TypeVar


T = TypeVar("T")


async def with_retries(
    operation: Callable[[], Awaitable[T]],
    *,
    attempts: int = 3,
    backoffs: tuple[float, ...] = (1.0, 3.0),
) -> T:
    last_error: Exception | None = None
    for index in range(attempts):
        try:
            return await operation()
        except Exception as exc:  # pragma: no cover - caller asserts behavior
            last_error = exc
            if index >= attempts - 1:
                break
            delay = backoffs[min(index, len(backoffs) - 1)]
            await asyncio.sleep(delay)
    if last_error is None:  # pragma: no cover
        raise RuntimeError("Retry loop exited without result or error.")
    raise last_error

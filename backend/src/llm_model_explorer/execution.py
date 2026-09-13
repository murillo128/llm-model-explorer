"""App-owned off-event-loop seam, not the long-operation or GPU scheduler."""

import asyncio
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
from functools import partial
from typing import ParamSpec, TypeVar

P = ParamSpec("P")
T = TypeVar("T")


class BlockingWork:
    def __init__(self) -> None:
        self._executor = ThreadPoolExecutor(thread_name_prefix="model-explorer")
        self._closed = False

    async def run(self, function: Callable[P, T], *args: P.args, **kwargs: P.kwargs) -> T:
        if self._closed:
            raise RuntimeError("blocking work service is closed")
        call = partial(function, *args, **kwargs)
        return await asyncio.get_running_loop().run_in_executor(
            self._executor, copy_context().run, call
        )

    async def aclose(self) -> None:
        self._closed = True
        await asyncio.to_thread(self._executor.shutdown, wait=True, cancel_futures=True)

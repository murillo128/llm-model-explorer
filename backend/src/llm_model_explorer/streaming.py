"""Reusable ASGI delivery of one runtime consumer; endpoints own preflight checks."""

import asyncio
import logging
from collections.abc import Awaitable, Callable

from starlette.responses import Response
from starlette.types import Receive, Scope, Send

from .lmex import DEFAULT_DATA_BYTES, Frame, LMEXWriter
from .model_files import ModelError
from .operations import Consumer, OperationCancelled

logger = logging.getLogger(__name__)
MetadataFactory = Callable[[], Awaitable[object]]


class _SendFailed(Exception):
    """Do not append an error frame after a potentially partial transport write."""


class LMEXResponse(Response):
    media_type = "application/vnd.llm-model-explorer.stream"

    def __init__(
        self,
        consumer: Consumer,
        metadata: MetadataFactory,
        *,
        max_data_bytes: int = DEFAULT_DATA_BYTES,
    ) -> None:
        if consumer.operation_id is None:
            raise ValueError("HTTP delivery requires a public consumer")
        self.consumer = consumer
        self.metadata_factory = metadata
        self.writer = LMEXWriter(max_data_bytes)
        self.status_code = 200
        self.background = None
        # No body attribute: Response must not infer Content-Length.
        self.init_headers(
            {
                "X-Operation-Id": str(consumer.operation_id),
                "Cache-Control": "no-store",
            }
        )

    async def _metadata(self) -> object:
        self.consumer.cancellation.check()
        task = asyncio.ensure_future(self.metadata_factory())
        done = asyncio.Event()
        task.add_done_callback(lambda _: done.set())
        try:
            await self.consumer.cancellation.wait(done)
            return task.result()
        finally:
            if not task.done():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    async def _deliver(self, send: Send) -> None:
        async def write(frame: Frame) -> None:
            for part in frame:
                if part:
                    try:
                        await send({"type": "http.response.body", "body": part, "more_body": True})
                    except Exception as exc:
                        raise _SendFailed() from exc

        await send({"type": "http.response.start", "status": 200, "headers": self.raw_headers})
        try:
            await write(self.writer.metadata(await self._metadata()))
            # Readers can expose arbitrary producer append boundaries. Keep at
            # most three carry bytes, never a whole-result accumulation buffer.
            carry = b""
            while chunk := await self.consumer.read(self.writer.max_data_bytes):
                view = memoryview(chunk)
                if carry:
                    needed = min(4 - len(carry), len(view))
                    carry += view[:needed]
                    view = view[needed:]
                    if len(carry) == 4:
                        await write(self.writer.data(carry))
                        carry = b""
                aligned = len(view) - len(view) % 4
                if aligned:
                    await write(self.writer.data(view[:aligned]))
                carry += view[aligned:]
            if carry:
                raise ValueError("unaligned producer EOF")
            # Consumer EOF is successful only after producer + artifact commit.
            await write(self.writer.complete())
        except _SendFailed:
            raise
        except OperationCancelled:
            await write(self.writer.cancelled())
        except Exception as exc:
            if isinstance(exc, ModelError):
                error = {"code": exc.code, "message": str(exc)}
            elif isinstance(exc, MemoryError):
                error = {"code": "resource_exhausted", "message": "Insufficient runtime memory."}
            else:
                logger.exception("LMEX production or delivery failed")
                error = {"code": "internal_error", "message": "Unable to complete stream."}
            await write(self.writer.error(error))
        await send({"type": "http.response.body", "body": b"", "more_body": False})

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        async def disconnected() -> None:
            while (await receive())["type"] != "http.disconnect":
                pass

        delivery = asyncio.create_task(self._deliver(send))
        disconnect = asyncio.create_task(disconnected())
        try:
            done, _ = await asyncio.wait(
                [delivery, disconnect], return_when=asyncio.FIRST_COMPLETED
            )
            for task in done:
                task.result()
        finally:
            for task in (delivery, disconnect):
                task.cancel()
            try:
                await asyncio.gather(delivery, disconnect, return_exceptions=True)
            finally:
                await self.consumer.aclose()

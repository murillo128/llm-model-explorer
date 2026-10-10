"""The catalogue HTTP adapter; blocking discovery never runs on the event loop."""

import asyncio
import logging
from collections.abc import AsyncGenerator
from typing import Annotated

import anyio
from fastapi import APIRouter, Depends, Query
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.types import Receive, Scope, Send

from .dependencies import get_blocking_work, get_catalogue, get_model_observer
from .execution import BlockingWork
from .model_events import HEARTBEAT_SECONDS, ModelObserver, ModelSubscription
from .model_files import ModelError
from .models import ModelCatalogue
from .session_routes import LifecycleRoute

router = APIRouter(route_class=LifecycleRoute)
logger = logging.getLogger(__name__)


@router.get("/models")
async def list_models(
    catalogue: Annotated[ModelCatalogue, Depends(get_catalogue)],
    work: Annotated[BlockingWork, Depends(get_blocking_work)],
) -> JSONResponse:
    try:
        listing = await work.run(catalogue.list_catalogue)
        return JSONResponse(
            {
                "models": [
                    model.model_dump(mode="json", exclude_none=True) for model in listing.models
                ],
                "diagnostics": [
                    diagnostic.model_dump(mode="json", exclude_none=True)
                    for diagnostic in listing.diagnostics
                ],
            },
            headers={"Cache-Control": "no-store"},
        )
    except ModelError as exc:
        # /models has no session and therefore no model_content_changed response.
        code = "validation_error" if exc.code == "model_content_changed" else exc.code
        status = 422 if exc.code == "model_content_changed" else exc.status
        return JSONResponse(
            {"code": code, "message": str(exc)},
            status_code=status,
            headers={"Cache-Control": "no-store"},
        )
    except MemoryError:
        return JSONResponse(
            {"code": "resource_exhausted", "message": "Insufficient memory to list models."},
            status_code=503,
            headers={"Cache-Control": "no-store"},
        )
    except Exception:
        logger.exception("Local model discovery failed")
        return JSONResponse(
            {"code": "internal_error", "message": "Unable to discover local models."},
            status_code=500,
            headers={"Cache-Control": "no-store"},
        )


class ModelEventResponse(StreamingResponse):
    def __init__(self, observer: ModelObserver, subscription: ModelSubscription) -> None:
        self.observer = observer
        self.subscription = subscription
        super().__init__(
            self.frames(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"},
        )

    async def frames(self) -> AsyncGenerator[bytes, None]:
        yield b"retry: 2000\n\n"
        loop = asyncio.get_running_loop()
        heartbeat = loop.time() + HEARTBEAT_SECONDS
        while True:
            try:
                frame = await asyncio.wait_for(
                    self.subscription.queue.get(), max(0, heartbeat - loop.time())
                )
            except TimeoutError:
                heartbeat = loop.time() + HEARTBEAT_SECONDS
                yield b": heartbeat\n\n"
                continue
            if frame is None:
                return
            yield frame

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        try:
            await super().__call__(scope, receive, send)
        finally:
            # Starlette cancels its streaming task group on disconnect. Cleanup
            # must outlive that cancellation, including a header-send failure.
            with anyio.CancelScope(shield=True):
                await self.observer.unsubscribe(self.subscription)


@router.get("/models/events")
async def watch_model(
    model_id: Annotated[str, Query(min_length=1)],
    observer: Annotated[ModelObserver, Depends(get_model_observer)],
) -> StreamingResponse:
    return ModelEventResponse(observer, await observer.subscribe(model_id))

"""Reusable application factory. The normative API lives in docs/spec/api/."""

import asyncio
import threading
from collections.abc import AsyncIterator, Callable, Sequence
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from types import FrameType

import uvicorn
from fastapi import APIRouter, FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .architecture_routes import router as architecture_router
from .embedding_routes import router as embedding_router
from .model_routes import router as model_router
from .services import Services, open_services
from .session_routes import router as session_router
from .settings import Settings
from .tensor_routes import router as tensor_router
from .tokenizer_routes import router as tokenizer_router


def create_app(
    settings: Settings,
    *,
    routers: Sequence[APIRouter] = (),
    service_lifespan: Callable[[Settings], AbstractAsyncContextManager[Services]] = open_services,
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.startup_task = asyncio.current_task()
        app.state.startup_stop = threading.Event()
        try:
            context = (
                open_services(settings, startup_stop=app.state.startup_stop)
                if service_lifespan is open_services
                else service_lifespan(settings)
            )
            async with context as services:
                del app.state.startup_task
                app.state.services = services
                try:
                    yield
                finally:
                    del app.state.services
        finally:
            if hasattr(app.state, "startup_task"):
                del app.state.startup_task

    # Do not serve a generated, incomplete alternative to the authoritative OpenAPI.
    app = FastAPI(lifespan=lifespan, openapi_url=None, docs_url=None, redoc_url=None)
    app.state.settings = settings
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(settings.cors_origins),
        allow_credentials=False,
        allow_methods=["GET", "POST", "DELETE"],
        allow_headers=["Content-Type"],
        expose_headers=["X-Operation-Id"],
    )
    for router in (
        model_router,
        architecture_router,
        session_router,
        tokenizer_router,
        tensor_router,
        embedding_router,
        *routers,
    ):
        app.include_router(router)
    return app


class ApplicationServer(uvicorn.Server):
    """Deliver CLI shutdown to a pending startup lifespan before server readiness.

    Uvicorn's normal handler sets should_exit but waits for startup to finish;
    cancelling the app-owned lifespan lets preparation settle and abort safely.
    """

    def __init__(self, app: FastAPI, *, host: str, port: int) -> None:
        super().__init__(uvicorn.Config(app, host=host, port=port, lifespan="on"))
        self.application = app

    def handle_exit(self, sig: int, frame: FrameType | None) -> None:
        super().handle_exit(sig, frame)
        task = getattr(self.application.state, "startup_task", None)
        if task is not None and not self.started:
            # The worker may reach publication before the loop delivers cancellation.
            self.application.state.startup_stop.set()
            task.cancel()

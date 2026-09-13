"""Reusable application factory. The normative API lives in docs/spec/api/."""

from collections.abc import AsyncIterator, Callable, Sequence
from contextlib import AbstractAsyncContextManager, asynccontextmanager

from fastapi import APIRouter, FastAPI
from fastapi.middleware.cors import CORSMiddleware

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
        async with service_lifespan(settings) as services:
            app.state.services = services
            try:
                yield
            finally:
                del app.state.services

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
    for router in (model_router, session_router, tokenizer_router, tensor_router, *routers):
        app.include_router(router)
    return app

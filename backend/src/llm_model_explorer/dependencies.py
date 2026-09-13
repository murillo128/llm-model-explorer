"""FastAPI dependencies shared by explicit domain routers."""

from typing import cast

from fastapi import Request

from .artifacts import ArtifactStore
from .execution import BlockingWork
from .models import ModelCatalogue
from .operations import OperationRuntime
from .services import Services
from .sessions import SessionRegistry
from .settings import Settings


def get_settings(request: Request) -> Settings:
    return cast(Settings, request.app.state.settings)


def get_services(request: Request) -> Services:
    services = getattr(request.app.state, "services", None)
    if not isinstance(services, Services):
        raise RuntimeError("application services are not running; enter the application lifespan")
    return services


def get_blocking_work(request: Request) -> BlockingWork:
    return get_services(request).blocking_work


def get_catalogue(request: Request) -> ModelCatalogue:
    service = get_services(request).catalogue
    if service is None:
        raise RuntimeError("catalogue service is not configured")
    return service


def get_sessions(request: Request) -> SessionRegistry:
    service = get_services(request).sessions
    if service is None:
        raise RuntimeError("session service is not configured")
    return service


def get_artifacts(request: Request) -> ArtifactStore:
    service = get_services(request).artifacts
    if service is None:
        raise RuntimeError("artifact service is not configured")
    return service


def get_operation_delivery(request: Request) -> OperationRuntime:
    service = get_services(request).operation_delivery
    if service is None:
        raise RuntimeError("operation delivery service is not configured")
    return service

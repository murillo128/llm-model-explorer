import asyncio
import threading
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from contextvars import ContextVar
from pathlib import Path
from unittest.mock import Mock

import pytest
import torch
from fastapi import APIRouter, Depends, Request, Response
from fastapi.testclient import TestClient

from llm_model_explorer.app import create_app
from llm_model_explorer.dependencies import (
    get_artifacts,
    get_catalogue,
    get_operation_delivery,
    get_services,
    get_sessions,
    get_settings,
)
from llm_model_explorer.execution import BlockingWork
from llm_model_explorer.services import Services, open_services
from llm_model_explorer.settings import Settings


def test_foundation_has_no_product_or_generated_contract_endpoints(settings: Settings) -> None:
    app = create_app(settings)
    with TestClient(app) as client:
        for path in ["/models", "/sessions", "/openapi.json", "/docs", "/redoc"]:
            assert client.get(path).status_code == 404
        assert isinstance(app.state.services, Services)
    assert not hasattr(app.state, "services")


def test_app_creation_and_lifespan_do_not_touch_models_or_tensors(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Settings have already validated paths. No file/model I/O belongs in the app factory.
    forbidden = Mock(side_effect=AssertionError("unexpected I/O or allocation"))
    monkeypatch.setattr("builtins.open", forbidden)
    monkeypatch.setattr(Path, "open", forbidden)
    monkeypatch.setattr("urllib.request.urlopen", forbidden)
    for name in ["load", "tensor", "empty", "zeros", "ones"]:
        monkeypatch.setattr(torch, name, forbidden)
    monkeypatch.setattr(torch.cuda, "init", forbidden)
    with TestClient(create_app(settings)) as client:
        assert client.get("/models").status_code == 404
    forbidden.assert_not_called()


def test_lifecycle_and_injected_domain_services(settings: Settings) -> None:
    events: list[str] = []
    sentinel = object()
    work = BlockingWork()

    @asynccontextmanager
    async def services(config: Settings) -> AsyncIterator[Services]:
        assert config is settings
        events.append("start")
        try:
            yield Services(work, sentinel, sentinel, sentinel, sentinel)
        finally:
            await work.aclose()
            events.append("stop")

    router = APIRouter()

    @router.get("/test-only")
    def inspect(request: Request) -> dict[str, bool]:
        assert get_settings(request) is settings
        for dependency in [get_catalogue, get_sessions, get_artifacts, get_operation_delivery]:
            assert dependency(request) is sentinel
        return {"injected": True}

    app = create_app(settings, routers=[router], service_lifespan=services)
    assert events == []
    with TestClient(app) as client:
        assert client.get("/test-only").json() == {"injected": True}
        assert events == ["start"]
    assert events == ["start", "stop"]
    assert not hasattr(app.state, "services")


@pytest.mark.parametrize(
    "dependency", [get_catalogue, get_sessions, get_artifacts, get_operation_delivery]
)
def test_unconfigured_domain_fails_and_supports_override(
    settings: Settings, dependency: Callable[[Request], object]
) -> None:
    router = APIRouter()

    @router.get("/test-only")
    def inspect(service: object = Depends(dependency)) -> dict[str, bool]:
        return {"injected": service is sentinel}

    sentinel = object()
    app = create_app(settings, routers=[router])
    with TestClient(app) as client:
        with pytest.raises(RuntimeError, match="not configured"):
            client.get("/test-only")
        app.dependency_overrides[dependency] = lambda: sentinel
        assert client.get("/test-only").json() == {"injected": True}


def test_dependencies_outside_lifespan_fail(settings: Settings) -> None:
    app = create_app(settings)
    request = Request({"type": "http", "app": app})
    with pytest.raises(RuntimeError, match="not running"):
        get_services(request)


def test_lifespan_releases_worker_on_exception(settings: Settings) -> None:
    async def scenario() -> None:
        with pytest.raises(ValueError, match="synthetic failure"):
            async with open_services(settings) as services:
                assert await services.blocking_work.run(lambda: 42) == 42
                raise ValueError("synthetic failure")
        with pytest.raises(RuntimeError, match="closed"):
            await services.blocking_work.run(lambda: 42)

    asyncio.run(scenario())


def test_blocking_work_keeps_loop_responsive_and_drains_on_close() -> None:
    async def scenario() -> None:
        work = BlockingWork()
        release = threading.Event()
        started = threading.Event()
        context: ContextVar[str] = ContextVar("test-context", default="absent")
        context.set("request")
        main_thread = threading.get_ident()

        def blocking(*, expected: str) -> str:
            assert threading.get_ident() != main_thread
            assert context.get() == expected
            started.set()
            assert release.wait(timeout=5)
            return "complete"

        task = asyncio.create_task(work.run(blocking, expected="request"))
        closing: asyncio.Task[None] | None = None
        try:
            assert await asyncio.to_thread(started.wait, 2)
            # A worker is blocked, but the loop can still run this task and begin shutdown.
            closing = asyncio.create_task(work.aclose())
            await asyncio.sleep(0.01)
            assert not closing.done()
            release.set()
            assert await asyncio.wait_for(task, 2) == "complete"
            await asyncio.wait_for(closing, 2)
        finally:
            release.set()
            await work.aclose()
        with pytest.raises(RuntimeError, match="closed"):
            await work.run(lambda: None)

    asyncio.run(scenario())


def test_cors_preflight_and_operation_header(model_root: Path, tmp_path: Path) -> None:
    origins = ("http://localhost:5173", "https://remote.example", "http://[::1]:5173")
    settings = Settings(model_root=model_root, cache_dir=tmp_path / "cache", cors_origins=origins)
    router = APIRouter()

    @router.get("/test-stream")
    def stream() -> Response:
        return Response(b"test", headers={"X-Operation-Id": "test-operation"})

    with TestClient(create_app(settings, routers=[router])) as client:
        for origin in origins:
            response = client.options(
                "/test-stream",
                headers={
                    "Origin": origin,
                    "Access-Control-Request-Method": "POST",
                    "Access-Control-Request-Headers": "Content-Type",
                },
            )
            assert response.status_code == 200
            assert response.headers["access-control-allow-origin"] == origin
            assert "access-control-allow-credentials" not in response.headers
            response = client.get("/test-stream", headers={"Origin": origin})
            assert response.headers["access-control-expose-headers"] == "X-Operation-Id"
            assert response.headers["x-operation-id"] == "test-operation"
        denied = client.options(
            "/test-stream",
            headers={
                "Origin": "https://unconfigured.example",
                "Access-Control-Request-Method": "GET",
            },
        )
        assert denied.status_code == 400
        assert "access-control-allow-origin" not in denied.headers


def test_cors_disabled_by_default(settings: Settings) -> None:
    with TestClient(create_app(settings)) as client:
        response = client.options(
            "/models",
            headers={"Origin": "http://localhost:5173", "Access-Control-Request-Method": "GET"},
        )
        assert response.status_code == 400
        assert "access-control-allow-origin" not in response.headers

"""Startup ownership, snapshot-safe HTTP retrieval, and real TCP readiness."""

import asyncio
import json
import logging
import shutil
import socket
import threading
from pathlib import Path
from typing import Any, cast
from unittest.mock import Mock
from uuid import uuid4

import httpx
import pytest
import torch
import uvicorn
from dense_fixtures import local_fixture
from fastapi.testclient import TestClient
from test_quantized_models import write_storage

from llm_model_explorer.app import create_app
from llm_model_explorer.architecture_analysis import (
    AnalysisInput,
    DescriptionRegistry,
    GraphBuilder,
)
from llm_model_explorer.architecture_analysis.core import AnalysisResult
from llm_model_explorer.architecture_service import ArchitectureService
from llm_model_explorer.artifacts import ArchitectureArtifactWriter, ArtifactStore
from llm_model_explorer.services import open_services
from llm_model_explorer.settings import Settings
from llm_model_explorer.tensor_source import ModelSource


def session(client: TestClient, model_id: str = "dense") -> str:
    response = client.post("/sessions", json={"model_id": model_id})
    assert response.status_code == 201, response.text
    return str(response.json()["id"])


def architecture(client: TestClient, sid: str) -> httpx.Response:
    response = client.get(f"/sessions/{sid}/architecture")
    assert response.headers["cache-control"] == "no-store"
    assert "x-operation-id" not in response.headers
    return cast(httpx.Response, response)


def metadata_fixture(root: Path, name: str) -> None:
    fixture = json.loads((Path(__file__).parent / "fixtures" / f"{name}-tiny.json").read_text())
    directory = root / name
    directory.mkdir()
    (directory / "config.json").write_text(json.dumps(fixture["configuration"]))
    write_storage(
        directory / "model.safetensors",
        [(key, value["dtype"], value["shape"]) for key, value in fixture["storage"].items()],
    )


@pytest.mark.parametrize("family", ["dense", "qwen3", "qwen35", "vjepa2"])
def test_registered_descriptions_and_no_tokenizer_session(settings: Settings, family: str) -> None:
    if family in {"dense", "qwen3"}:
        local_fixture(settings.model_root, family == "qwen3")
        model_id = "dense"
    else:
        metadata_fixture(settings.model_root, family)
        model_id = family
    with TestClient(create_app(settings)) as client:
        models = client.get("/models").json()["models"]
        assert not models[0]["tokenizer_available"]
        sid = session(client, model_id)
        response = architecture(client, sid)
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["status"] == "available", body
        assert body["graph"]["coverage"] == "complete"
        assert body["model_id"] == model_id
        assert str(settings.model_root) not in response.text
        assert len(response.content) <= 33_554_432
        inventory = client.get(f"/sessions/{sid}/tensors").json()
        assert inventory["coverage"] == "complete"
        assert inventory["diagnostics"] == []
        if family in {"qwen3", "qwen35"}:
            numeric = {tensor["id"]: tensor for tensor in inventory["tensors"]}
            packed = [p for p in body["graph"]["parameters"] if p["binding"] == "quantized"]
            assert packed
            for parameter in packed:
                assert parameter["inspection"]["status"] == "available"
                tensor = numeric[parameter["inspection"]["tensor_id"]]
                assert tensor["name"] == parameter["name"]
                assert tensor["shape"] == [d["value"] for d in parameter["logical_shape"]]
        sid2 = session(client, model_id)
        assert client.delete(f"/sessions/{sid}").status_code == 204
        assert architecture(client, sid2).content == response.content


def test_warm_start_and_get_never_generate(
    settings: Settings, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    local_fixture(settings.model_root, False)
    caplog.set_level(logging.INFO, logger="llm_model_explorer.architecture_service")
    with TestClient(create_app(settings)) as client:
        cold = architecture(client, session(client)).content
    forbidden = Mock(side_effect=AssertionError("warm lookup generated graph"))
    monkeypatch.setattr(DescriptionRegistry, "analyze", forbidden)
    monkeypatch.setattr(GraphBuilder, "__init__", forbidden)
    with TestClient(create_app(settings)) as client:
        assert architecture(client, session(client)).content == cold
    forbidden.assert_not_called()
    assert "mode=cache outcome=available" in caplog.text
    for field in [
        "hashing_seconds",
        "analysis_seconds",
        "cache_read_seconds",
        "preparation_seconds",
    ]:
        assert field in caplog.text


@pytest.mark.parametrize("damage", ["deleted", "corrupt", "whole-cache"])
def test_post_readiness_cache_loss_never_regenerates(
    settings: Settings, monkeypatch: pytest.MonkeyPatch, damage: str
) -> None:
    local_fixture(settings.model_root, False)
    with TestClient(create_app(settings)) as client:
        sid = session(client)
        assert architecture(client, sid).json()["status"] == "available"
        payload = next(settings.cache_dir.glob("*/payload.bin"))
        if damage == "deleted":
            payload.unlink()
        elif damage == "corrupt":
            payload.write_bytes(b"invalid")
        else:
            shutil.rmtree(settings.cache_dir)
        forbidden = Mock(side_effect=AssertionError("GET regenerated"))
        monkeypatch.setattr(DescriptionRegistry, "analyze", forbidden)
        response = architecture(client, sid)
        assert response.status_code == 200
        assert response.json()["reason"] == "cache_unavailable"
        assert response.json()["requires_restart"] is True
        assert "graph" not in response.json()
        forbidden.assert_not_called()


def test_session_errors_and_new_or_changed_content(settings: Settings) -> None:
    directory = local_fixture(settings.model_root, False)
    with TestClient(create_app(settings)) as client:
        old = session(client)
        assert architecture(client, "invalid").status_code == 422
        assert architecture(client, str(uuid4())).status_code == 404
        config = directory / "config.json"
        config.write_text(config.read_text() + " ")
        response = architecture(client, old)
        assert response.status_code == 409
        assert response.json()["code"] == "model_content_changed"
        assert architecture(client, session(client)).json()["reason"] == "restart_required"
        shutil.copytree(directory, settings.model_root / "new")
        assert architecture(client, session(client, "new")).json()["reason"] == "restart_required"


@pytest.mark.parametrize("failure", ["unsupported", "analysis", "partial", "size"])
def test_terminal_results_do_not_skip_later_models(
    settings: Settings, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    directory = local_fixture(settings.model_root, False)
    shutil.copytree(directory, settings.model_root / "later")
    config = directory / "config.json"
    config.write_text(config.read_text() + " ")  # Distinct snapshots for the failure spy.
    if failure == "unsupported":
        data = json.loads(config.read_text())
        data["model_type"] = "not_supported"
        config.write_text(json.dumps(data))
    original = DescriptionRegistry.analyze
    calls = 0

    def analyze(self: DescriptionRegistry, inputs: AnalysisInput, **kwargs: Any) -> AnalysisResult:
        nonlocal calls
        calls += 1
        if calls == 1 and failure == "analysis":
            raise RuntimeError(f"secret {settings.model_root}")
        if calls == 1 and failure == "size":
            kwargs["byte_limit"] = 100
        result = original(self, inputs, **kwargs)
        if calls == 1 and failure == "partial":
            assert result.graph is not None
            result = AnalysisResult(result.graph.model_copy(update={"coverage": "partial"}), None)
        return result

    monkeypatch.setattr(DescriptionRegistry, "analyze", analyze)
    with TestClient(create_app(settings)) as client:
        body = architecture(client, session(client)).json()
        if failure == "partial":
            assert body["graph"]["coverage"] == "partial"
        else:
            assert (
                body["reason"]
                == {
                    "unsupported": "unsupported_architecture",
                    "analysis": "analysis_failed",
                    "size": "unsupported_size",
                }[failure]
            )
            assert str(settings.model_root) not in json.dumps(body)
        assert architecture(client, session(client, "later")).json()["status"] == "available"


@pytest.mark.parametrize("global_failure", [False, True])
def test_model_local_cache_failure_versus_global_storage_failure(
    settings: Settings, monkeypatch: pytest.MonkeyPatch, global_failure: bool
) -> None:
    directory = local_fixture(settings.model_root, False)
    shutil.copytree(directory, settings.model_root / "later")
    original = ArtifactStore.begin_graph_write
    calls = 0

    def write(self: ArtifactStore, *args: Any, **kwargs: Any) -> ArchitectureArtifactWriter:
        nonlocal calls
        calls += 1
        if calls == 1:
            if global_failure:
                shutil.rmtree(settings.cache_dir)
            raise PermissionError("private storage details")
        return original(self, *args, **kwargs)

    monkeypatch.setattr(ArtifactStore, "begin_graph_write", write)
    if global_failure:
        with pytest.raises(OSError), TestClient(create_app(settings)):
            pytest.fail("globally unusable storage became ready")
    else:
        with TestClient(create_app(settings)) as client:
            assert architecture(client, session(client)).json()["reason"] == "analysis_failed"
            assert architecture(client, session(client, "later")).json()["status"] == "available"


@pytest.mark.parametrize("error, status", [(PermissionError, 500), (MemoryError, 503)])
def test_request_failures_are_http_errors(
    settings: Settings, monkeypatch: pytest.MonkeyPatch, error: type[Exception], status: int
) -> None:
    local_fixture(settings.model_root, False)
    with TestClient(create_app(settings)) as client:
        sid = session(client)
        monkeypatch.setattr(
            ArtifactStore, "lookup_graph", Mock(side_effect=error("private details"))
        )
        response = architecture(client, sid)
        assert response.status_code == status
        assert "private details" not in response.text


@pytest.mark.parametrize("boundary", ["analysis", "publication"])
def test_cancellation_settles_worker_aborts_unfinished_graph_and_never_yields(
    settings: Settings, monkeypatch: pytest.MonkeyPatch, boundary: str
) -> None:
    local_fixture(settings.model_root, False)
    entered, release = threading.Event(), threading.Event()
    target: Any = DescriptionRegistry if boundary == "analysis" else ArchitectureArtifactWriter
    method = "analyze" if boundary == "analysis" else "_check_publication"
    original = getattr(target, method)

    def barrier(*args: Any, **kwargs: Any) -> Any:
        entered.set()
        assert release.wait(5)
        return original(*args, **kwargs)

    monkeypatch.setattr(target, method, barrier)

    async def scenario() -> None:
        async def start() -> None:
            async with open_services(settings):
                pytest.fail("cancelled preparation became ready")

        task = asyncio.create_task(start())
        try:
            assert await asyncio.to_thread(entered.wait, 5)
            task.cancel()
            await asyncio.sleep(0.02)
            assert not task.done()  # Writer/analysis is still owned until its boundary.
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(task, 5)
        finally:
            release.set()
        assert not list(settings.cache_dir.iterdir())

    asyncio.run(scenario())


def test_startup_is_metadata_only(settings: Settings, monkeypatch: pytest.MonkeyPatch) -> None:
    local_fixture(settings.model_root, False)
    forbidden = Mock(side_effect=AssertionError("numeric/network work during preparation"))
    for name in ["load", "tensor", "empty", "zeros", "ones"]:
        monkeypatch.setattr(torch, name, forbidden)
    monkeypatch.setattr(torch.cuda, "init", forbidden)
    monkeypatch.setattr(torch.nn.Module, "__call__", forbidden)
    monkeypatch.setattr("llm_model_explorer.operations.DeviceScheduler.run", forbidden)
    monkeypatch.setattr(ModelSource, "iter_tensor", forbidden)
    monkeypatch.setattr("urllib.request.urlopen", forbidden)
    monkeypatch.setattr("socket.create_connection", forbidden)
    with TestClient(create_app(settings)) as client:
        assert architecture(client, session(client)).json()["status"] == "available"
    forbidden.assert_not_called()


def test_real_tcp_waits_for_all_terminal_preparation(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    directory = local_fixture(settings.model_root, False)
    shutil.copytree(directory, settings.model_root / "later")
    entered, release = threading.Event(), threading.Event()
    original = ArchitectureService._prepare_one
    calls = 0

    def prepare(self: ArchitectureService, entry: Any) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            entered.set()
            assert release.wait(5)
        original(self, entry)

    monkeypatch.setattr(ArchitectureService, "_prepare_one", prepare)

    async def scenario() -> None:
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
            server = uvicorn.Server(uvicorn.Config(create_app(settings), log_level="error"))
            task = asyncio.create_task(server.serve(sockets=[sock]))
            try:
                assert await asyncio.to_thread(entered.wait, 5)
                assert not server.started
                async with httpx.AsyncClient(
                    base_url=f"http://127.0.0.1:{port}", trust_env=False
                ) as c:
                    with pytest.raises(httpx.ConnectError):
                        await c.get("/models")
                    release.set()
                    async with asyncio.timeout(5):
                        while not server.started:
                            await asyncio.sleep(0.01)
                    response = await c.post("/sessions", json={"model_id": "later"})
                    sid = response.json()["id"]
                    graph = await c.get(f"/sessions/{sid}/architecture")
                    assert graph.status_code == 200
                    assert graph.json()["status"] == "available"
            finally:
                release.set()
                server.should_exit = True
                await asyncio.wait_for(task, 5)

    asyncio.run(scenario())


def test_source_change_during_get_is_rejected(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    directory = local_fixture(settings.model_root, False)
    with TestClient(create_app(settings)) as client:
        sid = session(client)
        original = ArtifactStore.lookup_graph

        def lookup(self: ArtifactStore, *args: Any, **kwargs: Any) -> Any:
            reader = original(self, *args, **kwargs)
            config = directory / "config.json"
            config.write_text(config.read_text() + " ")
            return reader

        monkeypatch.setattr(ArtifactStore, "lookup_graph", lookup)
        assert architecture(client, sid).status_code == 409


def test_response_budget_includes_envelope_on_cold_and_warm_start(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    local_fixture(settings.model_root, False)
    with TestClient(create_app(settings)) as client:
        body = architecture(client, session(client)).json()
        # This gate concerns the mandatory graph and HTTP envelope. Optional
        # annotations may be omitted during a cold rebuild to fit its budget.
        body["graph"].pop("templates", None)
        response_length = len(json.dumps(body, separators=(",", ":"), ensure_ascii=False).encode())
    monkeypatch.setattr("llm_model_explorer.architecture_service.MAX_BYTES", response_length - 1)
    for cold in [False, True]:
        if cold:
            shutil.rmtree(settings.cache_dir)
        with TestClient(create_app(settings)) as client:
            response = architecture(client, session(client))
            assert response.json()["reason"] == "unsupported_size"
            assert len(response.content) < response_length
        if cold:
            assert not list(settings.cache_dir.iterdir())


def test_optional_template_budget_rebuild_preserves_source_graph(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    local_fixture(settings.model_root, False)
    with TestClient(create_app(settings)) as client:
        response = architecture(client, session(client))
        original = response.json()["graph"]
        limit = len(response.content) - 1
    monkeypatch.setattr("llm_model_explorer.architecture_service.MAX_BYTES", limit)
    # Oversized immutable cached artifacts are rejected, never edited on GET.
    with TestClient(create_app(settings)) as client:
        assert architecture(client, session(client)).json()["reason"] == "unsupported_size"
    shutil.rmtree(settings.cache_dir)
    for _ in range(2):  # New publication and subsequent immutable cache retrieval.
        with TestClient(create_app(settings)) as client:
            response = architecture(client, session(client))
            body = response.json()
            assert body["status"] == "available"
            assert len(response.content) <= limit
            graph = body["graph"]
            assert len(graph.get("templates", [])) < len(original["templates"])
            for field in ["graph_id", "coverage", "nodes", "edges", "parameters", "repetitions"]:
                assert graph[field] == original[field]


def test_failed_source_publication_aborts_and_continues(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    directory = local_fixture(settings.model_root, False)
    shutil.copytree(directory, settings.model_root / "later")
    original = ArchitectureArtifactWriter._check_publication
    calls = 0

    def changed(self: ArchitectureArtifactWriter) -> None:
        nonlocal calls
        calls += 1
        if calls == 1:
            config = directory / "config.json"
            config.write_text(config.read_text() + " ")
        original(self)

    monkeypatch.setattr(ArchitectureArtifactWriter, "_check_publication", changed)
    with TestClient(create_app(settings)) as client:
        assert architecture(client, session(client)).json()["reason"] == "restart_required"
        assert architecture(client, session(client, "later")).json()["status"] == "available"
    assert len(list(settings.cache_dir.iterdir())) == 1
    assert not list(settings.cache_dir.glob(".tmp-*"))


def test_cli_sigterm_during_preparation_aborts_before_readiness(
    settings: Settings, tmp_path: Path
) -> None:
    import subprocess
    import sys
    import time

    local_fixture(settings.model_root, False)
    # A separate process exercises the CLI's real signal handler, lifespan, thread
    # settlement and writer. Files are barriers, never timing-based work completion.
    script = tmp_path / "startup_signal.py"
    script.write_text("""
import pathlib
import sys
import time
from llm_model_explorer.app import ApplicationServer
from llm_model_explorer.artifacts import ArchitectureArtifactWriter
from llm_model_explorer.cli import main
root = pathlib.Path(sys.argv[1])
original_check = ArchitectureArtifactWriter._check_publication
original_exit = ApplicationServer.handle_exit
def check(self):
    (root / "entered").touch()
    deadline = time.monotonic() + 15
    while not (root / "release").exists():
        if time.monotonic() > deadline:
            raise RuntimeError("barrier timed out")
        time.sleep(0.01)
    original_check(self)
def handle_exit(self, sig, frame):
    original_exit(self, sig, frame)
    (root / "signalled").touch()
ArchitectureArtifactWriter._check_publication = check
ApplicationServer.handle_exit = handle_exit
main(sys.argv[2:])
""")
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    with (tmp_path / "startup.log").open("w+") as log:
        process = subprocess.Popen(
            [
                sys.executable,
                str(script),
                str(tmp_path),
                "--model-root",
                str(settings.model_root),
                "--cache-dir",
                str(settings.cache_dir),
                "--port",
                str(port),
            ],
            stdout=log,
            stderr=log,
        )
        try:
            for marker in ["entered", "signalled"]:
                deadline = time.monotonic() + 10
                while not (tmp_path / marker).exists():
                    if process.poll() is not None or time.monotonic() > deadline:
                        log.seek(0)
                        pytest.fail(log.read())
                    time.sleep(0.01)
                if marker == "entered":
                    process.terminate()
            assert process.poll() is None
            (tmp_path / "release").touch()
            process.wait(timeout=10)
        finally:
            (tmp_path / "release").touch()
            if process.poll() is None:
                process.kill()
                process.wait(timeout=5)
        log.seek(0)
        output = log.read()
    assert "Application startup complete" not in output
    assert "outcome=cancelled" in output
    assert not list(settings.cache_dir.iterdir())

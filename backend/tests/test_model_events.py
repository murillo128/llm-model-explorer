"""Snapshot identity and subscription behavior, using explicit tick/worker barriers."""

import asyncio
import json
import os
import shutil
from dataclasses import replace
from pathlib import Path
from threading import Event
from unittest.mock import Mock

import pytest
from test_models import make_model, mutate_last_byte
from test_peft_adapters import composite_id, make_adapter, make_bases

from llm_model_explorer import model_files, models
from llm_model_explorer.catalogue_generation import CatalogueGeneration
from llm_model_explorer.execution import BlockingWork
from llm_model_explorer.model_events import ModelObserver, ModelSubscription
from llm_model_explorer.model_files import ModelError
from llm_model_explorer.models import ModelCatalogue
from llm_model_explorer.services import open_services
from llm_model_explorer.settings import Settings
from llm_model_explorer.tensor_source import ModelSource


class Ticks:
    def __init__(self, observer: ModelObserver, monkeypatch: pytest.MonkeyPatch) -> None:
        self.observer = observer
        self.gates: asyncio.Queue[asyncio.Future[None]] = asyncio.Queue()
        self.gate: asyncio.Future[None] | None = None
        monkeypatch.setattr(observer, "_wait_tick", self.wait)

    async def wait(self) -> None:
        gate = asyncio.get_running_loop().create_future()
        self.gates.put_nowait(gate)
        stop = asyncio.create_task(self.observer._stop.wait())
        try:
            await asyncio.wait((gate, stop), return_when=asyncio.FIRST_COMPLETED)
        finally:
            gate.cancel()
            stop.cancel()
            await asyncio.gather(stop, return_exceptions=True)

    async def ready(self) -> None:
        self.gate = await asyncio.wait_for(self.gates.get(), 5)

    async def tick(self) -> None:
        assert self.gate is not None
        self.gate.set_result(None)
        await self.ready()


def event(subscription: ModelSubscription) -> dict[str, object]:
    raw = subscription.queue.get_nowait()
    assert raw is not None and len(raw) <= 65536
    lines = raw.decode().splitlines()
    value: dict[str, object] = json.loads(lines[2].removeprefix("data: "))
    assert lines[0] == f"id: {value['epoch']}:{value['sequence']}"
    assert lines[1] == ("event: observation-error" if "code" in value else "event: model-state")
    return value


@pytest.mark.parametrize(
    "change",
    [
        "architecture",
        "config",
        "tokenizer",
        "weight",
        "nested",
        "replacement",
        "retarget",
        "adapter",
        "base",
    ],
)
def test_revision_uses_exact_dependency_snapshot(model_root: Path, change: str) -> None:
    if change in {"adapter", "base"}:
        bases = make_bases(model_root)
        adapter = make_adapter(model_root)
        directory = adapter if change == "adapter" else model_root / "base"
        # Fixture base directory names are owned by make_bases.
        if change == "base":
            directory = bases[0]
        identity = composite_id()
        target = directory / (
            "adapter_model.safetensors" if change == "adapter" else "model.safetensors"
        )
    else:
        directory = make_model(model_root)
        identity = "test/tiny"
        target = directory / "model.safetensors"
        if change == "nested":
            nested = directory / "weights/part.safetensors"
            nested.parent.mkdir()
            target.rename(nested)
            (directory / "model.safetensors.index.json").write_text(
                json.dumps({"weight_map": {"layer.weight": "weights/part.safetensors"}})
            )
            target = nested
        if change in {"architecture", "tokenizer"}:
            target = directory / f"{change}.json"
            target.write_text("{}")
        if change == "config":
            target = directory / "config.json"
    make_model(model_root, "unrelated", identity="other/model")
    catalogue = ModelCatalogue(model_root)
    old = catalogue.pin(identity)
    old_revision = old.model_revision
    unrelated = catalogue.pin("other/model").model_revision
    assert catalogue.pin(identity).model_revision == old_revision
    before = target.stat()
    if change in {"architecture", "tokenizer", "config"}:
        target.write_text(target.read_text() + " ")
    elif change == "replacement":
        replacement = directory / "replacement.tmp"
        replacement.write_bytes(target.read_bytes())
        os.utime(replacement, ns=(before.st_atime_ns, before.st_mtime_ns))
        replacement.replace(target)
    elif change == "retarget":
        replacement = model_root / "weights-copy"
        replacement.write_bytes(target.read_bytes())
        target.unlink()
        target.symlink_to(replacement)
    else:
        mutate_last_byte(target)
        os.utime(target, ns=(before.st_atime_ns, before.st_mtime_ns))
    fresh = catalogue.pin(identity)
    assert fresh.model_revision != old_revision
    assert old.model_revision == old_revision
    assert catalogue.pin("other/model").model_revision == unrelated
    assert (
        next(e for e in catalogue.discover() if e.summary.id == identity).model_revision
        == fresh.model_revision
    )
    with pytest.raises(ModelError) as failure:
        old.check_unchanged()
    assert failure.value.status == 409
    assert str(model_root) not in fresh.model_revision


def test_shared_observer_quiet_ticks_latest_queue_and_disconnects(
    model_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    directory = make_model(model_root)
    catalogue = ModelCatalogue(model_root)
    source = catalogue.pin("test/tiny")
    discover = Mock(wraps=catalogue._discover)
    monkeypatch.setattr(catalogue, "_discover", discover)

    async def scenario() -> None:
        work = BlockingWork()
        observer = ModelObserver(catalogue, work)
        ticks = Ticks(observer, monkeypatch)
        try:
            a = await observer.subscribe("test/tiny")
            b = await observer.subscribe("test/tiny")
            task = observer._task
            await ticks.ready()
            assert a.queue.empty()
            await ticks.tick()
            first = event(a)
            assert first["model_revision"] == source.model_revision
            assert event(b)["model_revision"] == first["model_revision"]
            forbidden = Mock(side_effect=AssertionError("idle observation read model bytes"))
            with monkeypatch.context() as patch:
                patch.setattr(model_files, "open_local", forbidden)
                for name in ("read_json", "parse_header", "logical_locations", "bind_adapter"):
                    patch.setattr(models, name, forbidden)
                for _ in range(3):
                    await ticks.tick()
                forbidden.assert_not_called()
            assert discover.call_count == 0 and a.queue.empty()
            # The one pending state must converge even if the client never drains it.
            for i in range(3):
                (directory / "architecture.json").write_text(f"invalid architecture {i}")
                await ticks.tick()
                await ticks.tick()
                assert a.queue.qsize() == b.queue.qsize() == 1
            latest = event(a)
            assert latest["status"] == "present"  # architecture is an independent capability
            assert latest["model_revision"] == catalogue.pin("test/tiny").model_revision
            assert latest["sequence"] > first["sequence"]  # type: ignore[operator]
            assert discover.call_count == 3
            await observer.unsubscribe(a)
            assert observer._task is task
            make_model(model_root, "unrelated", identity="other/model")
            event(b)
            await ticks.tick()
            await ticks.tick()
            assert b.queue.empty()
            await observer.unsubscribe(b)
            assert observer._task is None and task is not None and task.done()
        finally:
            await observer.aclose()
            await work.aclose()

    asyncio.run(scenario())


def test_reconnect_repair_removal_errors_and_backoff(
    model_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    directory = make_model(model_root)
    catalogue = ModelCatalogue(model_root)

    async def scenario() -> None:
        work = BlockingWork()
        observer = ModelObserver(catalogue, work)
        ticks = Ticks(observer, monkeypatch)
        try:
            a = await observer.subscribe("test/tiny")
            await ticks.ready()
            await ticks.tick()
            first = event(a)
            weights = directory / "model.safetensors"
            original = weights.read_bytes()
            weights.write_bytes(b"partial")
            await ticks.tick()
            await ticks.tick()
            assert event(a)["status"] == "unavailable"
            weights.write_bytes(original)
            await ticks.tick()
            await ticks.tick()
            repaired = event(a)
            assert (
                repaired["status"] == "present"
                and repaired["model_revision"] != first["model_revision"]
            )
            shutil.rmtree(directory)
            await ticks.tick()
            await ticks.tick()
            assert event(a)["model_revision"] is None
            make_model(model_root)
            await ticks.tick()
            await ticks.tick()
            assert event(a)["status"] == "present"
            # Ambiguous catalogue is not proof of deletion; expensive discovery backs off.
            duplicate = make_model(model_root, "duplicate")
            mutate_last_byte(duplicate / "model.safetensors")
            discover = Mock(wraps=catalogue._discover)
            monkeypatch.setattr(catalogue, "_discover", discover)
            await ticks.tick()
            await ticks.tick()
            failure = event(a)
            assert failure["code"] == "observation_failed" and "status" not in failure
            assert str(model_root) not in json.dumps(failure)
            for _ in range(12):
                await ticks.tick()
            assert discover.call_count == 4
            b = await observer.subscribe("test/tiny")
            await ticks.tick()
            assert event(b)["code"] == "observation_failed"
            shutil.rmtree(duplicate)
            await ticks.tick()
            await ticks.tick()
            assert event(a)["status"] == event(b)["status"] == "present"
            # Root failure also preserves the distinction, and recovery sends state.
            renamed = model_root.with_name("unavailable-root")
            model_root.rename(renamed)
            await ticks.tick()
            assert event(a)["code"] == "observation_failed"
            renamed.rename(model_root)
            await ticks.tick()
            await ticks.tick()
            assert event(a)["status"] == "present"
            await observer.unsubscribe(a)
            await observer.unsubscribe(b)
            mutate_last_byte(model_root / "tiny/model.safetensors")
            c = await observer.subscribe("test/tiny")
            await ticks.ready()
            await ticks.tick()
            assert event(c)["model_revision"] != repaired["model_revision"]
            assert observer.epoch != ModelObserver(catalogue, work).epoch
        finally:
            await observer.aclose()
            await work.aclose()

    asyncio.run(scenario())


def test_initial_observation_race_and_session_pin_race(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    directory = make_model(settings.model_root)

    async def scenario() -> None:
        async with open_services(settings) as services:
            assert services.sessions and services.catalogue and services.model_observer
            catalogue = services.catalogue
            observer = services.model_observer
            ticks = Ticks(observer, monkeypatch)
            old_revision = catalogue.pin("test/tiny").model_revision
            original_pin = catalogue.pin
            entered, release = Event(), Event()

            def pinned(model_id: str) -> ModelSource:
                source = original_pin(model_id)
                entered.set()
                assert release.wait(5)
                return source

            monkeypatch.setattr(catalogue, "pin", pinned)
            pending = asyncio.create_task(services.sessions.create("test/tiny"))
            try:
                assert await asyncio.to_thread(entered.wait, 5)
                a = await observer.subscribe("test/tiny")
                await ticks.ready()  # first metadata capture, before initial state
                mutate_last_byte(directory / "model.safetensors")
                await ticks.tick()
                assert a.queue.empty()  # changed generation must settle again
                await ticks.tick()
                fresh = event(a)
                release.set()
                session = await pending
                assert session.descriptor()["model_revision"] == old_revision
                assert fresh["model_revision"] != old_revision
                with pytest.raises(ModelError) as error:
                    await services.sessions.get(session.id)
                assert error.value.status == 409
            finally:
                release.set()
                await asyncio.gather(pending, return_exceptions=True)

    asyncio.run(scenario())


@pytest.mark.parametrize("cancel, fail", [(False, False), (True, False), (False, True)])
def test_capacity_and_shutdown_settles_blocking_worker(
    model_root: Path, monkeypatch: pytest.MonkeyPatch, cancel: bool, fail: bool
) -> None:
    make_model(model_root)
    catalogue = ModelCatalogue(model_root)
    entered, release = Event(), Event()
    original = catalogue.metadata_generation

    def blocked() -> CatalogueGeneration:
        entered.set()
        assert release.wait(5)
        if fail:
            raise OSError("private root unavailable during shutdown")
        return original()

    monkeypatch.setattr(catalogue, "metadata_generation", blocked)

    async def scenario() -> None:
        work = BlockingWork()
        observer = ModelObserver(catalogue, work)
        subscriptions = [await observer.subscribe("test/tiny") for _ in range(256)]
        try:
            with pytest.raises(ModelError) as failure:
                await observer.subscribe("test/tiny")
            assert failure.value.status == 503
            assert await asyncio.to_thread(entered.wait, 5)
            closing = asyncio.create_task(observer.aclose())
            await asyncio.sleep(0)
            assert not closing.done()
            if cancel:
                closing.cancel()
                await asyncio.sleep(0)
                assert not closing.done()
            release.set()
            if cancel:
                with pytest.raises(asyncio.CancelledError):
                    await closing
            else:
                await closing
            assert all(s.queue.get_nowait() is None for s in subscriptions)
            assert observer._task is None
        finally:
            release.set()
            await observer.aclose()
            await work.aclose()

    asyncio.run(scenario())


def test_real_tcp_incremental_sse_reconnect_cors_and_disconnect(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    import socket
    from collections.abc import AsyncIterator

    import httpx

    from llm_model_explorer.app import ApplicationServer, create_app

    directory = make_model(settings.model_root)
    app = create_app(replace(settings, cors_origins=("http://ui.example",)))

    async def scenario() -> None:
        ready = asyncio.Event()

        class Server(ApplicationServer):
            async def startup(self, sockets: list[socket.socket] | None = None) -> None:
                await super().startup(sockets)
                ready.set()

        server = Server(app, host="127.0.0.1", port=0)
        server.config.timeout_graceful_shutdown = 1
        sock = socket.socket()
        sock.bind(("127.0.0.1", 0))
        sock.listen()
        server_task = asyncio.create_task(server.serve(sockets=[sock]))
        try:
            await asyncio.wait_for(ready.wait(), 10)
            observer: ModelObserver = app.state.services.model_observer
            ticks = Ticks(observer, monkeypatch)
            disconnected = asyncio.Event()
            original = observer.unsubscribe

            async def unsubscribe(subscription: ModelSubscription) -> None:
                await original(subscription)
                disconnected.set()

            monkeypatch.setattr(observer, "unsubscribe", unsubscribe)

            async def read_frame(lines: AsyncIterator[str]) -> list[str]:
                result: list[str] = []
                async with asyncio.timeout(5):
                    async for line in lines:
                        if not line:
                            return result
                        result.append(line)
                raise AssertionError("SSE ended before a complete frame")

            base = f"http://127.0.0.1:{sock.getsockname()[1]}"
            async with httpx.AsyncClient(base_url=base, timeout=10) as client:
                response = await client.options(
                    "/models/events",
                    headers={
                        "Origin": "http://ui.example",
                        "Access-Control-Request-Method": "GET",
                        "Access-Control-Request-Headers": "Last-Event-ID",
                    },
                )
                assert response.status_code == 200
                invalid = await client.get("/models/events", params={"model_id": ""})
                assert invalid.status_code == 422 and invalid.json()["code"] == "validation_error"
                async with client.stream(
                    "GET",
                    "/models/events",
                    params={"model_id": "test/tiny"},
                    headers={"Origin": "http://ui.example"},
                ) as response:
                    assert response.status_code == 200
                    assert response.headers["content-type"].startswith("text/event-stream")
                    assert response.headers["cache-control"] == "no-store"
                    assert response.headers["x-accel-buffering"] == "no"
                    assert response.headers["access-control-allow-origin"] == "http://ui.example"
                    lines = response.aiter_lines()
                    assert await read_frame(lines) == ["retry: 2000"]
                    await ticks.ready()
                    await ticks.tick()
                    first = await read_frame(lines)
                    first_state = json.loads(first[2].removeprefix("data: "))
                    assert first_state["status"] == "present"
                    mutate_last_byte(directory / "model.safetensors")
                    await ticks.tick()
                    await ticks.tick()
                    second = await read_frame(lines)
                    second_state = json.loads(second[2].removeprefix("data: "))
                    assert second_state["model_revision"] != first_state["model_revision"]
                await asyncio.wait_for(disconnected.wait(), 5)
                assert observer._task is None
                disconnected.clear()
                # Reconnect ignores unknown IDs, including missed updates during idle.
                mutate_last_byte(directory / "model.safetensors")
                async with client.stream(
                    "GET",
                    "/models/events",
                    params={"model_id": "test/tiny"},
                    headers={"Last-Event-ID": "unknown:900"},
                ) as response:
                    lines = response.aiter_lines()
                    assert await read_frame(lines) == ["retry: 2000"]
                    await ticks.ready()
                    await ticks.tick()
                    current = await read_frame(lines)
                    state = json.loads(current[2].removeprefix("data: "))
                    assert state["model_revision"] not in {
                        first_state["model_revision"],
                        second_state["model_revision"],
                    }
                    session = await client.post("/sessions", json={"model_id": "test/tiny"})
                    assert session.json()["model_revision"] == state["model_revision"]
                await asyncio.wait_for(disconnected.wait(), 5)
                disconnected.clear()
                identity = "org/model@rev+peft-lora:org/adapter"
                async with client.stream(
                    "GET", "/models/events", params={"model_id": identity}
                ) as response:
                    lines = response.aiter_lines()
                    await read_frame(lines)
                    await ticks.ready()
                    await ticks.tick()
                    frame = await read_frame(lines)
                    absent = json.loads(frame[2].removeprefix("data: "))
                    assert absent["model_id"] == identity and absent["status"] == "unavailable"
                    server.should_exit = True
                    assert await anext(lines, None) is None
                await asyncio.wait_for(disconnected.wait(), 5)
        finally:
            server.should_exit = True
            await asyncio.wait_for(server_task, 10)
            sock.close()

    asyncio.run(scenario())


def test_heartbeat_is_a_comment_not_a_state(
    monkeypatch: pytest.MonkeyPatch, model_root: Path
) -> None:
    from llm_model_explorer import model_routes

    async def scenario() -> None:
        work = BlockingWork()
        observer = ModelObserver(ModelCatalogue(model_root), work)
        subscription = ModelSubscription("absent/model")
        frames = model_routes.ModelEventResponse(observer, subscription).frames()
        try:
            assert await anext(frames) == b"retry: 2000\n\n"
            # A zero timeout forces the idle timeout deterministically, with no sleep.
            monkeypatch.setattr(model_routes, "HEARTBEAT_SECONDS", 0)
            assert await anext(frames) == b": heartbeat\n\n"
            assert subscription.state is None and subscription.queue.empty()
        finally:
            await frames.aclose()
            await observer.aclose()
            await work.aclose()

    asyncio.run(scenario())


@pytest.mark.parametrize("model_id", ["local model", "local\nmodel", "local😀model"])
def test_exact_fallback_identity_and_event_byte_bound(
    model_root: Path, monkeypatch: pytest.MonkeyPatch, model_id: str
) -> None:
    make_model(model_root, model_id, identity=None)

    async def scenario() -> None:
        work = BlockingWork()
        observer = ModelObserver(ModelCatalogue(model_root), work)
        ticks = Ticks(observer, monkeypatch)
        try:
            a = await observer.subscribe(model_id)
            # Escaped astral code points take twelve bytes each; reject before registration.
            with pytest.raises(ModelError) as error:
                await observer.subscribe("😀" * 6000)
            assert error.value.code == "unsupported_size"
            assert len(observer._subscribers) == 1
            await ticks.ready()
            await ticks.tick()
            state = event(a)
            assert state["model_id"] == model_id and state["status"] == "present"
        finally:
            await observer.aclose()
            await work.aclose()

    asyncio.run(scenario())


def test_notification_disconnect_preserves_numeric_consumer(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    from llm_model_explorer.artifacts import ArtifactSpec
    from llm_model_explorer.operations import ProducerContext

    make_model(settings.model_root)

    async def scenario() -> None:
        async with open_services(settings) as services:
            assert services.sessions and services.model_observer and services.architectures
            session = await services.sessions.create("test/tiny")
            finish = asyncio.Event()

            async def produce(context: ProducerContext) -> None:
                await context.append(b"abcd")
                await context.cancellation.wait(finish)
                await context.append(b"efgh")

            spec = ArtifactSpec(
                model_fingerprint=session.source.fingerprint,
                source="weight",
                operation="test",
                parameters={},
                dtype="float32",
                layout="C",
                shape=(2,),
                expected_bytes=8,
                producer="test",
            )
            consumer = await services.sessions.subscribe(session.id, spec, produce)
            try:
                assert await consumer.read(4) == b"abcd"
                observer = services.model_observer
                ticks = Ticks(observer, monkeypatch)
                subscription = await observer.subscribe("test/tiny")
                await ticks.ready()
                await observer.unsubscribe(subscription)
                assert observer._task is None
                assert not services.architectures._stop.is_set()
                finish.set()
                assert await consumer.read(4) == b"efgh"
                assert await consumer.read() == b""
                assert consumer.terminal == "complete"
            finally:
                finish.set()
                await consumer.aclose()

    asyncio.run(scenario())

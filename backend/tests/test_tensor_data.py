"""Real endpoint bytes, progressive reuse, and bounded failure/cancellation."""

import asyncio
import errno
import json
import struct
from collections.abc import Generator
from dataclasses import replace
from typing import Any
from uuid import UUID, uuid4

import pytest
import torch
from cache_helpers import numeric_cache_entries, numeric_manifests
from fastapi.testclient import TestClient
from test_models import make_model, mutate_last_byte, replace_header, write_weights
from test_operations import run, spec
from test_streaming import Frames, forgotten, server

from llm_model_explorer.app import create_app
from llm_model_explorer.artifacts import ArtifactStore, ArtifactWriter
from llm_model_explorer.materialization import CHUNK_ELEMENTS, LogicalTensor
from llm_model_explorer.model_files import ModelError
from llm_model_explorer.models import ModelCatalogue
from llm_model_explorer.operations import MAX_READ_BYTES, OperationCancelled, ProducerContext
from llm_model_explorer.services import Services
from llm_model_explorer.settings import Settings
from llm_model_explorer.tensor_source import ModelSource


def frames(raw: bytes) -> list[tuple[int, bytes]]:
    result = []
    while raw:
        magic, kind, flags, reserved, size = struct.unpack("<4sBBHI", raw[:12])
        assert (magic, flags, reserved) == (b"LMEX", 0, 0)
        assert len(raw) >= 12 + size
        result.append((kind, raw[12 : 12 + size]))
        raw = raw[12 + size :]
    return result


def address(client: TestClient) -> str:
    session = client.post("/sessions", json={"model_id": "test/tiny"}).json()["id"]
    tensor = client.get(f"/sessions/{session}/tensors").json()["tensors"][0]["id"]
    return f"/sessions/{session}/tensors/{tensor}/data"


@pytest.mark.parametrize("dtype", ["F32", "F16", "BF16"])
@pytest.mark.parametrize("shape", [[2, 3], [6], [], [2, 0, 3], [1, 2, 3]])
def test_exact_shapes_and_native_source_unchanged(
    settings: Settings, dtype: str, shape: list[int]
) -> None:
    directory = make_model(settings.model_root)
    values = [] if 0 in shape else ([1.5] if not shape else [0.0, -0.0, 1.5, -2.25, 3.75, 0.125])
    path = directory / "model.safetensors"
    write_weights(path, [("weight", dtype, shape, values)])
    before = path.read_bytes()
    with TestClient(create_app(settings)) as client:
        url = address(client)
        for _ in range(2):
            response = client.get(url)
            assert response.status_code == 200
            UUID(response.headers["x-operation-id"])
            assert response.headers["cache-control"] == "no-store"
            result = frames(response.content)
            assert json.loads(result[0][1]) == dict(
                kind="tensor",
                tensor_id=url.split("/")[-2],
                name="weight",
                shape=shape,
                dtype="float32",
                byte_order="little",
                layout="c",
                byte_length=len(values) * 4,
            )
            assert result[-1] == (4, b"")
            assert all(len(data) % 4 == 0 for kind, data in result if kind == 2)
            assert b"".join(data for kind, data in result if kind == 2) == struct.pack(
                "<" + "f" * len(values), *values
            )
    assert path.read_bytes() == before
    assert len(numeric_manifests(settings.cache_dir)) == (0 if dtype == "F32" else 1)


@pytest.mark.parametrize("dtype,width", [("F32", 4), ("F16", 2), ("BF16", 2)])
def test_all_half_finite_widening_and_nonfinite_bits(
    settings: Settings, dtype: str, width: int
) -> None:
    directory = make_model(settings.model_root)
    if dtype == "F32":
        words = [0, 0x80000000, 0x7F800000, 0xFF800000, 0x7FC01234, 0xFFA00001, 1, 0x7F7FFFFF]
        expected = struct.pack("<8I", *words)
    else:
        mask = 0x7C00 if dtype == "F16" else 0x7F80
        words = [word for word in range(65536) if (word & mask) != mask]
        expected = b"".join(
            struct.pack("<f", struct.unpack("<e", struct.pack("<H", word))[0])
            if dtype == "F16"
            else struct.pack("<I", word << 16)
            for word in words
        )
    payload = struct.pack("<" + ("I" if width == 4 else "H") * len(words), *words)
    replace_header(
        directory / "model.safetensors",
        {"weight": {"dtype": dtype, "shape": [len(words)], "data_offsets": [0, len(payload)]}},
        payload,
    )
    with TestClient(create_app(settings)) as client:
        result = frames(client.get(address(client)).content)
    assert result[-1] == (4, b"")
    assert b"".join(data for kind, data in result if kind == 2) == expected


def test_preflight_failures(settings: Settings, monkeypatch: pytest.MonkeyPatch) -> None:
    directory = make_model(settings.model_root)
    app = create_app(settings)
    with TestClient(app) as client:
        url = address(client)
        for path, status, code in [
            (url.replace(url.split("/")[2], str(uuid4())), 404, "session_not_found"),
            (url.replace(url.split("/")[-2], "missing"), 404, "tensor_not_found"),
            (url.replace(url.split("/")[2], "invalid"), 422, "validation_error"),
        ]:
            response = client.get(path)
            assert response.status_code == status
            assert response.json()["code"] == code
            assert "x-operation-id" not in response.headers
        service: Services = app.state.services
        assert service.sessions is not None
        session = service.sessions.require(UUID(url.split("/")[2]))
        location = session.source._locations[0]
        cases: list[tuple[dict[str, object], str]] = [
            ({"storage_dtype": "I8"}, "unsupported_representation"),
            ({"numel": 10}, "validation_error"),
            ({"shape": (2**53,)}, "unsupported_size"),
        ]
        for updates, code in cases:
            altered = replace(
                session.source,
                _locations=(
                    replace(location, descriptor=location.descriptor.model_copy(update=updates)),
                ),
            )
            with monkeypatch.context() as patch:
                patch.setattr(
                    ModelSource,
                    "tensors",
                    lambda self, source=altered: tuple(loc.descriptor for loc in source._locations),
                )
                response = client.get(url)
                assert response.status_code == 422
                assert response.json()["code"] == code
        mutate_last_byte(directory / "model.safetensors")
        response = client.get(url)
        assert response.status_code == 409
        assert response.json()["code"] == "model_content_changed"
        assert str(directory) not in response.text


async def setup_stream(client: Any) -> str:
    session = (await client.post("/sessions", json={"model_id": "test/tiny"})).json()["id"]
    tensor = (await client.get(f"/sessions/{session}/tensors")).json()["tensors"][0]["id"]
    return f"/sessions/{session}/tensors/{tensor}/data"


async def payload(reader: Frames) -> bytes:
    data = bytearray()
    while True:
        kind, block = await reader.next()
        if kind == 4:
            await reader.end()
            return bytes(data)
        assert kind == 2
        assert len(block) <= MAX_READ_BYTES and len(block) % 4 == 0
        data.extend(block)


@pytest.mark.parametrize("dtype", ["F32", "F16", "BF16"])
def test_tall_progressive_reuse_and_independent_cancellation(
    settings: Settings, monkeypatch: pytest.MonkeyPatch, dtype: str
) -> None:
    directory = make_model(settings.model_root)
    count = CHUNK_ELEMENTS * 2 + 3
    write_weights(directory / "model.safetensors", [("weight", dtype, [count, 1], [1.5] * count)])
    calls, bounds = [], []
    original = ModelSource.iter_tensor

    def instrument(
        self: ModelSource, tensor_id: str, *, chunk_elements: int = CHUNK_ELEMENTS
    ) -> Generator[torch.Tensor, None, None]:
        calls.append(tensor_id)
        for block in original(self, tensor_id, chunk_elements=chunk_elements):
            bounds.append(block.numel())
            yield block

    monkeypatch.setattr(ModelSource, "iter_tensor", instrument)

    async def scenario() -> None:
        app = create_app(settings)
        async with app.router.lifespan_context(app):
            services: Services = app.state.services
            runtime = services.operation_delivery
            assert runtime is not None
            runtime.device = "cuda:0"
            gate = asyncio.Event()
            original_produce = LogicalTensor.produce

            async def held(self: LogicalTensor, context: ProducerContext) -> None:
                original_append = context.append
                first = True

                async def append(data: bytes) -> None:
                    nonlocal first
                    await original_append(data)
                    if first:
                        first = False
                        await context.cancellation.wait(gate)

                context.append = append  # type: ignore[method-assign]
                await original_produce(self, context)

            monkeypatch.setattr(LogicalTensor, "produce", held)
            async with server(app) as client:
                url = await setup_stream(client)
                async with client.stream("GET", url) as a:
                    fa = Frames(a)
                    assert (await fa.next())[0] == 1
                    kind, first = await fa.next()
                    assert kind == 2 and first and len(first) < count * 4
                    if dtype != "F32":
                        assert not numeric_manifests(settings.cache_dir)
                    other_url = await setup_stream(client)
                    async with client.stream("GET", other_url) as b:
                        fb = Frames(b)
                        assert (await fb.next())[0] == 1
                        assert a.headers["x-operation-id"] != b.headers["x-operation-id"]
                        assert (
                            await client.delete("/operations/" + a.headers["x-operation-id"])
                        ).status_code == 204
                        gate.set()
                        assert await payload(fb) == struct.pack("<f", 1.5) * count
                    if dtype != "F32":
                        assert (await fa.next()) == (6, b"")
                await forgotten(runtime)
                previous = len(calls)
                async with client.stream("GET", url) as warm:
                    fw = Frames(warm)
                    assert (await fw.next())[0] == 1
                    assert await payload(fw) == struct.pack("<f", 1.5) * count
                assert len(calls) == previous + (1 if dtype == "F32" else 0)
                assert len(calls) == (3 if dtype == "F32" else 1)
                assert not runtime.scheduler._devices
        assert max(bounds) <= CHUNK_ELEMENTS
        assert len(numeric_manifests(settings.cache_dir)) == (0 if dtype == "F32" else 1)

    run(scenario())


@pytest.mark.parametrize("failure", ["cancel", "memory", "disk", "change"])
def test_partial_production_never_commits(
    settings: Settings, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    directory = make_model(settings.model_root)
    write_weights(directory / "model.safetensors", [("weight", "F16", [4], [1, 2, 3, 4])])

    async def scenario() -> None:
        gate = asyncio.Event()

        async def held(self: LogicalTensor, ctx: ProducerContext) -> None:
            await ctx.append(struct.pack("<f", 1))
            await ctx.cancellation.wait(gate)
            if failure == "memory":
                raise MemoryError("/private/model")
            if failure == "disk":
                raise OSError(errno.ENOSPC, "/private/cache")
            if failure == "change":
                mutate_last_byte(directory / "model.safetensors")
                self.source.check_unchanged()

        monkeypatch.setattr(LogicalTensor, "produce", held)
        app = create_app(settings)
        async with app.router.lifespan_context(app):
            async with server(app) as client:
                url = await setup_stream(client)
                async with client.stream("GET", url) as response:
                    reader = Frames(response)
                    assert (await reader.next())[0] == 1
                    assert (await reader.next()) == (2, struct.pack("<f", 1))
                    if failure == "cancel":
                        await client.delete("/operations/" + response.headers["x-operation-id"])
                    else:
                        gate.set()
                    kind, data = await reader.next()
                    if failure == "cancel":
                        assert (kind, data) == (6, b"")
                    else:
                        assert kind == 5
                        assert json.loads(data)["code"] == (
                            "model_content_changed" if failure == "change" else "resource_exhausted"
                        )
                        assert b"/private" not in data
                    await reader.end()
        assert not numeric_cache_entries(settings.cache_dir)

    run(scenario())


@pytest.mark.parametrize("dtype", ["F32", "F16"])
def test_internal_dependency_reuses_logical_materialization(settings: Settings, dtype: str) -> None:
    directory = make_model(settings.model_root)
    write_weights(directory / "model.safetensors", [("weight", dtype, [2], [1.5, -2.25])])

    async def scenario() -> None:
        app = create_app(settings)
        async with app.router.lifespan_context(app):
            services: Services = app.state.services
            logical = services.logical_tensors
            runtime = services.operation_delivery
            assert logical is not None and runtime is not None
            source = ModelCatalogue(settings.model_root).pin("test/tiny")
            tensor = logical.resolve(source, source.tensors()[0].id)

            async def produce(ctx: ProducerContext) -> None:
                async with logical.dependency(ctx, tensor) as dependency:
                    assert dependency.operation_id is None
                    while data := await dependency.read():
                        await ctx.append(data)

            consumer = await runtime.subscribe(uuid4(), spec("derived-copy"), produce)
            try:
                assert await consumer.read() == struct.pack("<2f", 1.5, -2.25)
                assert await consumer.read() == b""
            finally:
                await consumer.aclose()

    run(scenario())


@pytest.mark.parametrize(
    "fault",
    ["allocation", "native_allocation", "begin", "append", "commit", "truncated", "source_change"],
)
def test_real_producer_faults(
    settings: Settings, monkeypatch: pytest.MonkeyPatch, fault: str
) -> None:
    directory = make_model(settings.model_root)
    path = directory / "model.safetensors"
    write_weights(path, [("weight", "F16", [2], [1.5, -2.25])])
    with TestClient(create_app(settings)) as client:
        url = address(client)
        if fault == "native_allocation":

            def out_of_memory(*args: object, **kwargs: object) -> None:
                raise torch.OutOfMemoryError("/private/allocation")

            monkeypatch.setattr(torch, "frombuffer", out_of_memory)
        elif fault in {"begin", "append", "commit"}:

            def fail(*args: object, **kwargs: object) -> None:
                raise OSError(errno.ENOSPC, "/private/cache")

            owner = ArtifactStore if fault == "begin" else ArtifactWriter
            monkeypatch.setattr(owner, "begin_write" if fault == "begin" else fault, fail)
        else:
            original = ModelSource.iter_tensor

            def broken(
                self: ModelSource, tensor_id: str, *, chunk_elements: int = CHUNK_ELEMENTS
            ) -> Generator[torch.Tensor, None, None]:
                if fault == "allocation":
                    raise MemoryError("/private/allocation")
                for block in original(self, tensor_id, chunk_elements=chunk_elements):
                    if fault == "truncated":
                        yield block[:1]
                        return
                    yield block
                    mutate_last_byte(path)

            monkeypatch.setattr(ModelSource, "iter_tensor", broken)
        response = client.get(url)
        result = frames(response.content)
        assert result[-1][0] == 5
        expected = (
            "model_content_changed"
            if fault == "source_change"
            else "internal_error"
            if fault == "truncated"
            else "resource_exhausted"
        )
        assert json.loads(result[-1][1])["code"] == expected
        assert b"/private" not in response.content
        assert all(kind != 4 for kind, _ in result)
    assert not numeric_cache_entries(settings.cache_dir)


@pytest.mark.parametrize("dtype", ["F32", "BF16"])
def test_active_reader_rejects_changed_snapshot(settings: Settings, dtype: str) -> None:
    directory = make_model(settings.model_root)
    path = directory / "model.safetensors"
    write_weights(path, [("weight", dtype, [3], [1.5, 2, 3])])

    async def scenario() -> None:
        app = create_app(settings)
        async with app.router.lifespan_context(app):
            services: Services = app.state.services
            logical, sessions = services.logical_tensors, services.sessions
            assert logical is not None and sessions is not None
            session = await sessions.create("test/tiny")
            tensor_id = (await sessions.tensors(session.id))[0].id
            tensor, cold = await logical.subscribe(sessions, session.id, tensor_id)
            try:
                while await cold.read():
                    pass
            finally:
                await cold.aclose()
            _, warm = await logical.subscribe(sessions, session.id, tensor_id)
            try:
                assert await warm.read(4) == struct.pack("<f", 1.5)
                mutate_last_byte(path)
                with pytest.raises(ModelError, match="Model content changed"):
                    await warm.read()
            finally:
                await warm.aclose()
            new_session = await sessions.create("test/tiny")
            fresh, consumer = await logical.subscribe(sessions, new_session.id, tensor_id)
            assert fresh.spec.key != tensor.spec.key
            await consumer.aclose()

    run(scenario())


def test_direct_session_deletion_closes_source(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    make_model(settings.model_root)
    original = ModelSource.iter_tensor
    closed = []

    def observed(
        self: ModelSource, tensor_id: str, *, chunk_elements: int = CHUNK_ELEMENTS
    ) -> Generator[torch.Tensor, None, None]:
        try:
            yield from original(self, tensor_id, chunk_elements=chunk_elements)
        finally:
            closed.append(True)

    monkeypatch.setattr(ModelSource, "iter_tensor", observed)

    async def scenario() -> None:
        app = create_app(settings)
        async with app.router.lifespan_context(app):
            services: Services = app.state.services
            logical, sessions = services.logical_tensors, services.sessions
            assert logical is not None and sessions is not None
            session = await sessions.create("test/tiny")
            tensor_id = (await sessions.tensors(session.id))[0].id
            _, consumer = await logical.subscribe(sessions, session.id, tensor_id)
            assert await consumer.read(4)
            assert not closed
            await sessions.delete(session.id)
            assert closed == [True]
            with pytest.raises(OperationCancelled):
                await consumer.read()
        assert not numeric_cache_entries(settings.cache_dir)

    run(scenario())

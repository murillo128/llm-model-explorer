"""Exact gathered-matrix oracles, real HTTP streams and ephemeral ownership."""

import asyncio
import gc
import json
import struct
import weakref
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path
from threading import Event
from typing import Any
from uuid import UUID

import pytest
import torch
from cache_helpers import numeric_cache_entries
from fastapi.testclient import TestClient
from starlette.types import Message
from test_embeddings import KEY, address, embedding_model
from test_models import mutate_last_byte, write_weights
from test_operations import run
from test_streaming import Frames, forgotten, server
from test_tensor_data import frames, payload

from llm_model_explorer import embedding_analysis, tensor_analysis
from llm_model_explorer.app import create_app
from llm_model_explorer.embedding_analysis import EmbeddingAnalysis, subscribe_embedding_analysis
from llm_model_explorer.materialization import CHUNK_ELEMENTS, _TensorReader
from llm_model_explorer.model_files import ModelError
from llm_model_explorer.models import ModelCatalogue
from llm_model_explorer.operations import Cancellation, OperationRuntime, SourceReader
from llm_model_explorer.services import Services
from llm_model_explorer.settings import Settings
from llm_model_explorer.streaming import LMEXResponse
from llm_model_explorer.tensor_source import ModelSource

FIXTURES: dict[str, Any] = json.loads(
    (Path(__file__).resolve().parents[2] / "api/fixtures/embedding-analysis.json").read_text()
)
KINDS = ("input_embeddings_statistics", "input_embeddings_distributions")


def make_case(settings: Settings, case: dict[str, Any], dtype: str = "F32") -> Path:
    directory = embedding_model(settings.model_root, vocab=3)
    values = struct.unpack("<9f", bytes.fromhex(case["table_data_hex"]))
    write_weights(directory / "model.safetensors", [(KEY, dtype, [3, 3], list(values))])
    return directory


def assert_metadata(actual: dict[str, Any], expected: dict[str, Any]) -> None:
    if actual["kind"] == "input_embeddings_statistics" and actual["finite_count"]:
        # Only derived floating reductions have a tolerance; identities, shape,
        # counts, min/max and histogram payloads remain exact.
        for field in ("mean", "stddev"):
            assert actual.pop(field) == pytest.approx(expected[field], rel=1e-6, abs=1e-7)
        assert actual.pop("percentiles") == pytest.approx(
            expected["percentiles"], rel=1e-6, abs=1e-7
        )
        expected = {k: v for k, v in expected.items() if k not in ("mean", "stddev", "percentiles")}
    assert actual == expected


@pytest.mark.parametrize("case", FIXTURES["numerical_cases"], ids=lambda c: c["name"])
def test_http_numerical_oracles(settings: Settings, case: dict[str, Any]) -> None:
    directory = make_case(settings, case)
    before = (directory / "model.safetensors").read_bytes()
    with TestClient(
        create_app(replace(settings, cors_origins=("http://localhost:5173",)))
    ) as client:
        url = address(client)
        for _ in range(2):
            for suffix in ("statistics", "distributions"):
                response = client.post(
                    url + "/" + suffix,
                    json={"token_ids": case["token_ids"]},
                    headers={"Origin": "http://localhost:5173"},
                )
                assert response.status_code == 200
                UUID(response.headers["x-operation-id"])
                assert response.headers["cache-control"] == "no-store"
                assert "X-Operation-Id" in response.headers["access-control-expose-headers"]
                result = frames(response.content)
                assert result[0][0] == 1 and result[-1] == (4, b"")
                assert_metadata(json.loads(result[0][1]), case[suffix])
                data = b"".join(raw for kind, raw in result if kind == 2)
                if suffix == "statistics":
                    assert all(kind != 2 for kind, _ in result)
                else:
                    assert data.hex() == case["counts_hex"]
                    counts = struct.unpack(f"<{len(data) // 4}I", data)
                    assert (
                        sum(counts[: len(case["token_ids"]) * 100])
                        == sum(counts[len(case["token_ids"]) * 100 :])
                        == case["statistics"]["finite_count"]
                    )
                assert KEY.encode() not in response.content
            values = client.post(url, json={"token_ids": case["token_ids"]})
            assert (
                b"".join(raw for kind, raw in frames(values.content) if kind == 2).hex()
                == case["values_hex"]
            )
    assert (directory / "model.safetensors").read_bytes() == before
    assert not numeric_cache_entries(settings.cache_dir)
    assert not list(settings.cache_dir.glob(".tmp-*"))


@pytest.mark.parametrize("dtype", ["F16", "BF16"])
def test_logical_float32_conversion(settings: Settings, dtype: str) -> None:
    case = FIXTURES["numerical_cases"][0]
    make_case(settings, case, dtype)
    with TestClient(create_app(settings)) as client:
        url = address(client)
        for suffix in ("statistics", "distributions"):
            response = client.post(url + "/" + suffix, json={"token_ids": case["token_ids"]})
            result = frames(response.content)
            assert_metadata(json.loads(result[0][1]), case[suffix])
            assert result[-1] == (4, b"")


@pytest.mark.parametrize("suffix", ["statistics", "distributions"])
def test_preflight_invalid_unsupported_and_size(
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
    suffix: str,
) -> None:
    directory = embedding_model(settings.model_root)

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        pytest.fail("preflight must not read any requested values")

    monkeypatch.setattr(ModelSource, "iter_rows", forbidden)
    with TestClient(create_app(settings)) as client:
        url = address(client) + "/" + suffix
        for ids in ([0, 4], [0, -1], [0, 1.5], [0, True], [0, "2"]):
            response = client.post(url, json={"token_ids": ids})
            assert response.status_code == 422 and response.json()["code"] == "validation_error"
            assert "x-operation-id" not in response.headers
        response = client.post(url, content="{", headers={"Content-Type": "application/json"})
        assert response.status_code == 400 and response.json()["code"] == "malformed_json"
        response = client.post(url, json={"token_ids": [0] * 524288})
        assert response.status_code == 422 and response.json()["code"] == "unsupported_size"
        assert "x-operation-id" not in response.headers
        mutate_last_byte(directory / "model.safetensors")
        response = client.post(url, json={"token_ids": []})
        assert response.status_code == 409 and response.json()["code"] == "model_content_changed"
    # A missing family mapping remains a capability error even for empty input.
    path = directory / "config.json"
    config = json.loads(path.read_text())
    config["architectures"] = []
    path.write_text(json.dumps(config))
    with TestClient(create_app(settings)) as client:
        response = client.post(address(client) + "/" + suffix, json={"token_ids": []})
        assert (
            response.status_code == 422 and response.json()["code"] == "unsupported_representation"
        )


@pytest.mark.parametrize("kind", KINDS)
def test_shape_preflight_without_allocation(
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
    kind: embedding_analysis.Kind,
) -> None:
    embedding_model(settings.model_root)
    source = ModelCatalogue(settings.model_root).pin("test/tiny")
    from llm_model_explorer.embeddings import resolve_embeddings

    source_rows = resolve_embeddings(source, (0,))
    hidden = (2**53 - 1) // 4
    giant = replace(source_rows, table=source_rows.table.model_copy(update={"shape": (4, hidden)}))
    with pytest.raises(ModelError) as caught:
        EmbeddingAnalysis(giant, kind).preflight()
    assert caught.value.code == "unsupported_size"


@pytest.mark.parametrize("fault", ["memory", "native_memory", "overflow", "source_change", "short"])
@pytest.mark.parametrize("suffix", ["statistics", "distributions"])
def test_auxiliary_error_does_not_poison_values(
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
    fault: str,
    suffix: str,
) -> None:
    case = FIXTURES["numerical_cases"][0]
    directory = make_case(settings, case)
    original = tensor_analysis.calculate

    def fail(*args: Any, **kwargs: Any) -> Any:
        if fault == "memory":
            raise MemoryError("private allocation")
        if fault == "native_memory":
            raise RuntimeError("DefaultCPUAllocator: can't allocate memory: private Error code 12")
        if fault == "overflow":
            return tensor_analysis.checked_uint32(torch.tensor([2**32], dtype=torch.int64))
        result = original(*args, **kwargs)
        if fault == "source_change":
            mutate_last_byte(directory / "model.safetensors")
        return result

    monkeypatch.setattr(embedding_analysis, "calculate", fail)
    if fault == "short":
        original_rows = ModelSource.iter_rows

        def short_rows(
            self: ModelSource, tensor: str, ids: tuple[int, ...], *, chunk_elements: int
        ) -> Iterator[torch.Tensor]:
            yield from original_rows(self, tensor, ids[:1], chunk_elements=chunk_elements)

        monkeypatch.setattr(ModelSource, "iter_rows", short_rows)
    with TestClient(create_app(settings)) as client:
        url = address(client)
        response = client.post(url + "/" + suffix, json={"token_ids": case["token_ids"]})
        result = frames(response.content)
        assert [kind for kind, _ in result] == [5]
        assert (
            json.loads(result[-1][1])["code"]
            == {
                "memory": "resource_exhausted",
                "native_memory": "resource_exhausted",
                "overflow": "unsupported_size",
                "source_change": "model_content_changed",
                "short": "internal_error",
            }[fault]
        )
        assert b"private" not in response.content
        if fault != "source_change":
            if fault == "short":
                monkeypatch.setattr(ModelSource, "iter_rows", original_rows)
            values = client.post(url, json={"token_ids": case["token_ids"]})
            assert frames(values.content)[-1] == (4, b"")
            assert (
                b"".join(raw for kind, raw in frames(values.content) if kind == 2).hex()
                == case["values_hex"]
            )
    assert not numeric_cache_entries(settings.cache_dir)


def test_tcp_independent_consumers_cancel_before_meta_and_value_progress(
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = FIXTURES["numerical_cases"][0]
    make_case(settings, case)

    async def scenario() -> None:
        app = create_app(settings)
        async with app.router.lifespan_context(app):
            services: Services = app.state.services
            runtime = services.operation_delivery
            assert runtime is not None
            gate, entered = asyncio.Event(), asyncio.Event()
            original = EmbeddingAnalysis.prepare

            async def held(
                self: EmbeddingAnalysis, runtime: OperationRuntime, cancellation: Cancellation
            ) -> SourceReader:
                entered.set()
                await cancellation.wait(gate)
                return await original(self, runtime, cancellation)

            monkeypatch.setattr(EmbeddingAnalysis, "prepare", held)
            async with server(app) as client:
                a = (await client.post("/sessions", json={"model_id": "test/tiny"})).json()["id"]
                b = (await client.post("/sessions", json={"model_id": "test/tiny"})).json()["id"]
                url_a, url_b = f"/sessions/{a}/embeddings", f"/sessions/{b}/embeddings"
                body = {"token_ids": case["token_ids"]}
                async with (
                    client.stream("POST", url_a + "/statistics", json=body) as first,
                    client.stream("POST", url_b + "/statistics", json=body) as second,
                    client.stream("POST", url_b + "/distributions", json=body) as third,
                ):
                    await entered.wait()
                    assert len({r.headers["x-operation-id"] for r in (first, second, third)}) == 3
                    fa, fb, fc = Frames(first), Frames(second), Frames(third)
                    async with client.stream("POST", url_a, json=body) as values:
                        reader = Frames(values)
                        assert (await reader.next())[0] == 1
                        assert (await payload(reader)).hex() == case["values_hex"]
                    assert not gate.is_set()
                    response = await client.delete("/operations/" + first.headers["x-operation-id"])
                    assert response.status_code == 204
                    assert await fa.next() == (6, b"")
                    await fa.end()
                    gate.set()
                    assert_metadata(json.loads((await fb.next())[1]), case["statistics"])
                    assert await payload(fb) == b""
                    assert_metadata(json.loads((await fc.next())[1]), case["distributions"])
                    assert (await payload(fc)).hex() == case["counts_hex"]
                await forgotten(runtime)
            assert not numeric_cache_entries(settings.cache_dir)

    run(scenario())


@pytest.mark.parametrize("action", ["cancel", "delete_session", "disconnect", "source_change"])
def test_partial_data_releases_counts(
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
    action: str,
) -> None:
    directory = embedding_model(settings.model_root)
    buffers: list[weakref.ReferenceType[torch.Tensor]] = []
    original = embedding_analysis._AnalysisReader.__init__

    def observed(
        self: Any, source: ModelSource, header: bytes, counts: torch.Tensor | None
    ) -> None:
        assert counts is not None
        buffers.append(weakref.ref(counts))
        original(self, source, header, counts)

    monkeypatch.setattr(embedding_analysis._AnalysisReader, "__init__", observed)

    async def scenario() -> None:
        app = create_app(settings)
        async with app.router.lifespan_context(app):
            services: Services = app.state.services
            sessions, runtime = services.sessions, services.operation_delivery
            assert sessions is not None and runtime is not None
            session = await sessions.create("test/tiny")
            result, consumer = await subscribe_embedding_analysis(
                sessions, session.id, (2, 0, 2), "input_embeddings_distributions"
            )
            prefix, release, disconnect = asyncio.Event(), asyncio.Event(), asyncio.Event()
            sent: list[bytes] = []

            async def send(message: Message) -> None:
                if message["type"] == "http.response.body":
                    sent.append(bytes(message["body"]))
                    if isinstance(message["body"], memoryview) and not prefix.is_set():
                        prefix.set()
                        await release.wait()

            async def receive() -> Message:
                await disconnect.wait()
                return {"type": "http.disconnect"}

            task = asyncio.create_task(
                LMEXResponse(consumer, lambda: result.metadata(consumer), max_data_bytes=4)(
                    {}, receive, send
                )
            )
            await prefix.wait()
            assert buffers[0]() is not None
            if action == "cancel":
                assert consumer.operation_id is not None
                await runtime.cancel(consumer.operation_id)
            elif action == "delete_session":
                await sessions.delete(session.id)
            elif action == "source_change":
                mutate_last_byte(directory / "model.safetensors")
            else:
                disconnect.set()
                await task
            release.set()
            await task
            result_frames = frames(b"".join(sent))
            assert (
                result_frames[-1][0]
                == {"cancel": 6, "delete_session": 6, "disconnect": 2, "source_change": 5}[action]
            )
            if action == "source_change":
                assert json.loads(result_frames[-1][1])["code"] == "model_content_changed"
            assert consumer._source_reader is None and consumer._closed
            await forgotten(runtime)
            gc.collect()
            assert all(ref() is None for ref in buffers)
        assert not numeric_cache_entries(settings.cache_dir)

    run(scenario())


def test_requested_rows_bounded_reads_and_repeated_buffer_release(
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    embedding_model(settings.model_root, vocab=20, hidden=CHUNK_ELEMENTS + 1)
    ids_seen: list[tuple[int, ...]] = []
    values_refs: list[weakref.ReferenceType[torch.Tensor]] = []
    counts_refs: list[weakref.ReferenceType[torch.Tensor]] = []
    original_rows, original_calculate = ModelSource.iter_rows, tensor_analysis.calculate

    def rows(
        self: ModelSource, tensor: str, ids: tuple[int, ...], *, chunk_elements: int
    ) -> Iterator[torch.Tensor]:
        ids_seen.append(ids)
        assert chunk_elements <= CHUNK_ELEMENTS
        for chunk in original_rows(self, tensor, ids, chunk_elements=chunk_elements):
            assert chunk.numel() <= CHUNK_ELEMENTS
            yield chunk

    def calculate(*args: Any, **kwargs: Any) -> Any:
        values_refs.append(weakref.ref(args[0]))
        assert args[0].numel() == 3 * (CHUNK_ELEMENTS + 1)
        result = original_calculate(*args, **kwargs)
        if result[1] is not None:
            counts_refs.append(weakref.ref(result[1]))
        return result

    monkeypatch.setattr(ModelSource, "iter_rows", rows)
    monkeypatch.setattr(embedding_analysis, "calculate", calculate)

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        pytest.fail("derived analysis must never read the complete vocabulary")

    monkeypatch.setattr(ModelSource, "iter_tensor", forbidden)
    with TestClient(create_app(settings)) as client:
        url = address(client)
        for _ in range(2):
            for suffix in ("statistics", "distributions"):
                response = client.post(url + "/" + suffix, json={"token_ids": [2, 0, 2]})
                assert frames(response.content)[-1] == (4, b"")
                del response
                gc.collect()
                assert all(ref() is None for ref in values_refs + counts_refs)
    assert ids_seen == [(2, 0, 2)] * 4
    assert not numeric_cache_entries(settings.cache_dir)


def test_cancel_during_blocking_gather_drains_before_close(
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    embedding_model(settings.model_root)
    entered, release = Event(), Event()
    closed: list[bool] = []
    original_read, original_close = _TensorReader.read_available, _TensorReader.close

    def held(self: _TensorReader, size: int) -> bytes:
        entered.set()
        assert release.wait(5)
        return original_read(self, size)

    def close(self: _TensorReader) -> None:
        assert release.is_set()
        closed.append(True)
        original_close(self)

    monkeypatch.setattr(_TensorReader, "read_available", held)
    monkeypatch.setattr(_TensorReader, "close", close)

    async def scenario() -> None:
        app = create_app(settings)
        async with app.router.lifespan_context(app):
            services: Services = app.state.services
            sessions, runtime = services.sessions, services.operation_delivery
            assert sessions is not None and runtime is not None
            session = await sessions.create("test/tiny")
            result, consumer = await subscribe_embedding_analysis(
                sessions, session.id, (2, 0, 2), "input_embeddings_statistics"
            )
            task = asyncio.create_task(result.metadata(consumer))
            assert await asyncio.to_thread(entered.wait, 3)
            task.cancel()
            await asyncio.sleep(0)
            assert not task.done() and not closed
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await task
            await consumer.aclose()
            assert closed
            await forgotten(runtime)

    try:
        run(scenario())
    finally:
        release.set()


@pytest.mark.parametrize("kind", KINDS)
def test_gather_precedes_device_queue_and_queued_cancellation_releases_buffer(
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
    kind: embedding_analysis.Kind,
) -> None:
    embedding_model(settings.model_root)
    gathered = Event()
    input_buffers: list[weakref.ReferenceType[torch.Tensor]] = []
    closed = 0
    original_empty, original_close = torch.empty, _TensorReader.close

    def empty(*args: Any, **kwargs: Any) -> torch.Tensor:
        value = original_empty(*args, **kwargs)
        input_buffers.append(weakref.ref(value))
        return value

    def close(self: _TensorReader) -> None:
        nonlocal closed
        original_close(self)
        closed += 1
        if closed == 2:
            gathered.set()

    async def scenario() -> None:
        app = create_app(settings)
        async with app.router.lifespan_context(app):
            services: Services = app.state.services
            sessions, runtime = services.sessions, services.operation_delivery
            assert sessions is not None and runtime is not None
            session = await sessions.create("test/tiny")
            monkeypatch.setattr(torch, "empty", empty)
            monkeypatch.setattr(_TensorReader, "close", close)
            # Exercise the real per-device scheduler with CPU kernels as the
            # deterministic oracle; this is not evidence of actual CUDA execution.
            runtime.device = "cuda:7"
            lock = runtime.scheduler._devices.setdefault("cuda:7", asyncio.Lock())
            await lock.acquire()
            calls: list[bool] = []

            def calculate(
                values: torch.Tensor,
                kernel: tensor_analysis.Kind,
                shape: tuple[int, ...],
                device: str,
                cancellation: Cancellation,
            ) -> tuple[bytes, torch.Tensor | None]:
                assert lock.locked() and closed == 2 and device == "cuda:7"
                calls.append(True)
                return tensor_analysis.calculate(values, kernel, shape, "cpu", cancellation)

            monkeypatch.setattr(embedding_analysis, "calculate", calculate)
            first, a = await subscribe_embedding_analysis(sessions, session.id, (2, 0, 2), kind)
            second, b = await subscribe_embedding_analysis(sessions, session.id, (2, 0, 2), kind)
            ta, tb = asyncio.create_task(first.metadata(a)), asyncio.create_task(second.metadata(b))
            assert await asyncio.to_thread(gathered.wait, 3)
            assert len(input_buffers) == 2 and not calls
            assert not ta.done() and not tb.done()
            assert a.operation_id is not None
            await runtime.cancel(a.operation_id)
            from llm_model_explorer.operations import OperationCancelled

            with pytest.raises(OperationCancelled):
                await ta
            # Remove the test's exception-owning task before checking GC.
            del ta
            gc.collect()
            assert sum(ref() is not None for ref in input_buffers) == 1
            assert not tb.done() and not calls
            lock.release()
            await tb
            while await b.read():
                pass
            await b.aclose()
            assert calls == [True]
            del tb
            gc.collect()
            assert all(ref() is None for ref in input_buffers)
            await forgotten(runtime)
        assert not numeric_cache_entries(settings.cache_dir)

    run(scenario())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device unavailable")
def test_actual_cuda_embedding_analysis(settings: Settings) -> None:
    case = FIXTURES["numerical_cases"][0]
    make_case(settings, case)
    with TestClient(create_app(replace(settings, device="cuda:0"))) as client:
        url = address(client)
        for suffix in ("statistics", "distributions"):
            response = client.post(url + "/" + suffix, json={"token_ids": case["token_ids"]})
            result = frames(response.content)
            assert result[-1] == (4, b"")
            assert_metadata(json.loads(result[0][1]), case[suffix])
            if suffix == "distributions":
                assert (
                    b"".join(raw for kind, raw in result if kind == 2).hex() == case["counts_hex"]
                )


@pytest.mark.parametrize("kind", KINDS)
def test_analysis_control_limit_includes_its_own_metadata(
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
    kind: embedding_analysis.Kind,
) -> None:
    from llm_model_explorer import lmex
    from llm_model_explorer.embeddings import resolve_embeddings

    embedding_model(settings.model_root)
    source = ModelCatalogue(settings.model_root).pin("test/tiny")
    result = resolve_embeddings(source, (2, 0, 2))
    limit = len(lmex.LMEXWriter().metadata(result.metadata())[1])
    monkeypatch.setattr(lmex, "MAX_CONTROL_BYTES", limit)
    # Values metadata still fits; each richer analysis object must be checked
    # independently before a reader, buffer or consumer is allocated.
    lmex.LMEXWriter().metadata(result.metadata())
    with pytest.raises(ModelError) as caught:
        EmbeddingAnalysis(result, kind).preflight()
    assert caught.value.code == "unsupported_size"


def test_statistics_writer_rejects_even_empty_data() -> None:
    from llm_model_explorer.lmex import LMEXWriter

    writer = LMEXWriter()
    writer.metadata(FIXTURES["numerical_cases"][0]["statistics"])
    for raw in (b"", b"abcd"):
        with pytest.raises(ValueError, match="invalid DATA"):
            writer.data(raw)
    writer.complete()

"""Logical packed weights through inventory, row access and real operation flows."""

import asyncio
import hashlib
import json
import struct
from dataclasses import replace
from pathlib import Path

import httpx
import pytest
import torch
from cache_helpers import numeric_manifests
from fastapi.testclient import TestClient
from quantized_oracles import (
    PackedFixture,
    Stored,
    compressed_tensors_fixture,
    gptq_fixture,
    nvfp4_fixture,
)
from test_models import mutate_last_byte
from test_operations import run
from test_streaming import Frames, forgotten, server
from test_tensor_analysis import histogram_oracle, oracle
from test_tensor_data import frames, payload

from llm_model_explorer.app import create_app
from llm_model_explorer.materialization import LogicalTensor
from llm_model_explorer.model_files import ModelError
from llm_model_explorer.models import ModelCatalogue
from llm_model_explorer.operations import ProducerContext
from llm_model_explorer.services import Services
from llm_model_explorer.settings import Settings
from llm_model_explorer.tensor_analysis import TensorAnalysis


@pytest.fixture(params=["gptq-int4", "nvfp4", "compressed-tensors-w4a16-int4"])
def packed(request: pytest.FixtureRequest) -> PackedFixture:
    if request.param == "gptq-int4":
        return gptq_fixture()
    if request.param == "nvfp4":
        return nvfp4_fixture()
    return compressed_tensors_fixture()


def tensor_id(packed: PackedFixture) -> str:
    return hashlib.sha256(packed.name.encode()).hexdigest()


def model_id(packed: PackedFixture) -> str:
    return f"numeric@{packed.encoding}"


def address(client: TestClient, packed: PackedFixture) -> str:
    session = client.post("/sessions", json={"model_id": model_id(packed)})
    assert session.status_code == 201
    return f"/sessions/{session.json()['id']}/tensors/{tensor_id(packed)}/"


async def async_address(client: httpx.AsyncClient, packed: PackedFixture) -> str:
    session = await client.post("/sessions", json={"model_id": model_id(packed)})
    assert session.status_code == 201
    return f"/sessions/{session.json()['id']}/tensors/{tensor_id(packed)}/"


def raw(values: torch.Tensor) -> bytes:
    return bytes(values.view(torch.uint8).tolist())


def test_logical_identity_shape_coverage_and_shard_independence(
    tmp_path: Path, packed: PackedFixture
) -> None:
    descriptors = []
    expected = packed.expected()
    for split in [False, True]:
        root = tmp_path / str(split)
        packed.write(root, split=split)
        source = ModelCatalogue(root).pin(model_id(packed))
        (descriptor,) = source.tensors()
        descriptors.append(descriptor)
        assert descriptor.id == tensor_id(packed)
        assert descriptor.name == packed.name
        assert descriptor.shape == packed.shape
        assert descriptor.rank == 2 and descriptor.numel == len(expected) // 4
        assert descriptor.logical_dtype == "float32"
        assert descriptor.storage_dtype == ("U8" if packed.encoding == "nvfp4" else "I32")
        assert descriptor.storage_format == packed.encoding
        inventory = source.inventory()
        assert inventory["coverage"] == "complete" and inventory["diagnostics"] == []
        assert len(source.physical_tensors()) == len(packed.storage)
        chunks = list(source.iter_tensor(descriptor.id, chunk_elements=131))
        assert all(block.numel() <= 131 for block in chunks)
        assert raw(torch.cat(chunks)) == expected
    assert descriptors[0] == descriptors[1]


@pytest.mark.parametrize(
    "unknown_name,dtype",
    [("unknown.region", "F32"), ("unknown.packed", "I32"), ("orphan.g_idx", "I32")],
)
def test_unresolved_physical_records_keep_precise_partial_inventory(
    settings: Settings, packed: PackedFixture, unknown_name: str, dtype: str
) -> None:
    unknown = Stored(unknown_name, dtype, (2,), struct.pack("<2I", 0, 1))
    fixture = replace(packed, storage=(*packed.storage, unknown))
    fixture.write(settings.model_root, split=True)
    source = ModelCatalogue(settings.model_root).pin(model_id(packed))
    assert [descriptor.name for descriptor in source.tensors()] == [packed.name]
    inventory = source.inventory()
    assert inventory["coverage"] == "partial"
    assert unknown_name in str(inventory["diagnostics"])
    with TestClient(create_app(settings)) as client:
        url = address(client, packed)
        assert frames(client.get(url + "data").content)[-1] == (4, b"")
        unknown_id = hashlib.sha256(unknown_name.encode()).hexdigest()
        for endpoint in ["data", "statistics", "distributions"]:
            response = client.get(url.replace(tensor_id(packed), unknown_id) + endpoint)
            assert response.status_code == 404
            assert response.json()["code"] == "tensor_not_found"


def test_rows_keep_requested_order_duplicates_and_unaligned_chunks(
    tmp_path: Path, packed: PackedFixture
) -> None:
    packed.write(tmp_path, split=True)
    source = ModelCatalogue(tmp_path).pin(model_id(packed))
    rows = (packed.shape[0] - 1, 0, 1, 1, 1, 0)
    width = packed.shape[1] * 4
    expected = packed.expected()
    blocks = list(source.iter_rows(tensor_id(packed), rows, chunk_elements=7))
    assert all(block.numel() <= 7 for block in blocks)
    assert raw(torch.cat(blocks)) == b"".join(
        expected[row * width : (row + 1) * width] for row in rows
    )
    assert list(source.iter_rows(tensor_id(packed), (), chunk_elements=7)) == []
    for invalid_rows in [(-1,), (packed.shape[0],), (True,)]:
        with pytest.raises(ModelError) as raised:
            list(source.iter_rows(tensor_id(packed), invalid_rows, chunk_elements=7))
        assert raised.value.code == "validation_error"


def test_long_unknown_name_keeps_inventory_diagnostic_within_api_contract(
    settings: Settings, packed: PackedFixture
) -> None:
    name = "unknown." + "module." * 3000 + "region"
    unknown = Stored(name, "I32", (1,), struct.pack("<i", 1))
    replace(packed, storage=(*packed.storage, unknown)).write(settings.model_root, split=True)
    with TestClient(create_app(settings)) as client:
        session = client.post("/sessions", json={"model_id": model_id(packed)}).json()
        response = client.get(f"/sessions/{session['id']}/tensors")
        assert response.status_code == 200
        inventory = response.json()
        assert inventory["coverage"] == "partial"
        assert [tensor["name"] for tensor in inventory["tensors"]] == [packed.name]
        (diagnostic,) = inventory["diagnostics"]
        # InventoryDiagnostic.message uses ArchitectureText's published maxLength.
        assert 0 < len(diagnostic["message"]) <= 16_384
        assert "unknown." in diagnostic["message"] and "region" in diagnostic["message"]


def test_http_data_statistics_histograms_and_persistent_warm_reuse(
    settings: Settings, packed: PackedFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    directory = packed.write(settings.model_root, split=True)
    files = {path: path.read_bytes() for path in directory.glob("*.safetensors")}
    expected = packed.expected()
    values = list(struct.unpack(f"<{len(expected) // 4}f", expected))
    results: dict[str, bytes] = {}
    with TestClient(create_app(settings)) as client:
        url = address(client, packed)
        for endpoint in ["data", "statistics", "distributions"]:
            response = client.get(url + endpoint)
            assert response.status_code == 200
            assert response.headers["cache-control"] == "no-store"
            assert response.headers["x-operation-id"]
            results[endpoint] = response.content
            result = frames(response.content)
            assert result[0][0] == 1 and result[-1] == (4, b"")
            metadata = json.loads(result[0][1])
            data = b"".join(block for kind, block in result if kind == 2)
            if endpoint == "data":
                assert metadata["tensor_id"] == tensor_id(packed)
                assert metadata["name"] == packed.name
                assert metadata["shape"] == list(packed.shape)
                assert metadata["dtype"] == "float32" and metadata["layout"] == "c"
                assert metadata["byte_length"] == len(expected)
                assert data == expected
            elif endpoint == "statistics":
                assert [kind for kind, _ in result] == [1, 4]
                assert metadata["count"] == len(values) == metadata["finite_count"]
                assert metadata["non_finite_count"] == 0
                actual = [metadata[key] for key in ["minimum", "maximum", "mean", "stddev"]]
                actual.extend(
                    metadata["percentiles"][key] for key in ["p01", "p05", "p50", "p95", "p99"]
                )
                assert actual == pytest.approx(oracle(values), rel=1e-6, abs=1e-7)
            else:
                assert data == histogram_oracle(list(packed.shape), values)
                assert metadata["domain_minimum"] == min(values)
                assert metadata["domain_maximum"] == max(values)
                assert metadata["byte_length"] == len(data)
        assert len(numeric_manifests(settings.cache_dir)) == 3

    async def forbidden_tensor(self: LogicalTensor, context: ProducerContext) -> None:
        pytest.fail("warm logical tensor request decoded again")

    async def forbidden_analysis(self: TensorAnalysis, context: ProducerContext) -> None:
        pytest.fail("warm packed analysis request computed again")

    monkeypatch.setattr(LogicalTensor, "produce", forbidden_tensor)
    monkeypatch.setattr(TensorAnalysis, "produce", forbidden_analysis)
    with TestClient(create_app(settings)) as client:
        url = address(client, packed)
        for endpoint, expected_response in results.items():
            response = client.get(url + endpoint)
            assert response.status_code == 200
            assert response.content == expected_response
    assert {path: path.read_bytes() for path in files} == files


@pytest.mark.parametrize("cancel_all", [False, True])
def test_shared_decode_cancellation_preserves_other_session(
    settings: Settings, packed: PackedFixture, monkeypatch: pytest.MonkeyPatch, cancel_all: bool
) -> None:
    packed.write(settings.model_root, split=True)
    expected = packed.expected()

    async def scenario() -> None:
        app = create_app(settings)
        async with app.router.lifespan_context(app):
            services: Services = app.state.services
            runtime = services.operation_delivery
            assert runtime is not None
            started, finish = asyncio.Event(), asyncio.Event()
            conversions = 0
            original = LogicalTensor.produce

            async def held(self: LogicalTensor, context: ProducerContext) -> None:
                nonlocal conversions
                conversions += 1
                append = context.append

                async def hold_after_append(data: bytes) -> None:
                    await append(data)
                    started.set()
                    await context.cancellation.wait(finish)

                context.append = hold_after_append  # type: ignore[method-assign]
                await original(self, context)

            monkeypatch.setattr(LogicalTensor, "produce", held)
            async with server(app) as client:
                first_url = await async_address(client, packed)
                second_url = await async_address(client, packed)
                async with (
                    client.stream("GET", first_url + "data") as first,
                    client.stream("GET", second_url + "data") as second,
                ):
                    await asyncio.wait_for(started.wait(), timeout=3)
                    a, b = Frames(first), Frames(second)
                    assert first.headers["x-operation-id"] != second.headers["x-operation-id"]
                    for reader in [a, b]:
                        assert (await reader.next())[0] == 1
                        assert await reader.next() == (2, expected)
                    assert not numeric_manifests(settings.cache_dir)
                    first_session = first_url.split("/")[2]
                    assert (await client.delete(f"/sessions/{first_session}")).status_code == 204
                    assert await a.next() == (6, b"")
                    await a.end()
                    assert (await client.get(first_url + "data")).status_code == 404
                    assert (
                        await client.get(second_url.split("/tensors/")[0] + "/tensors")
                    ).status_code == 200
                    if cancel_all:
                        await client.delete("/operations/" + second.headers["x-operation-id"])
                        assert await b.next() == (6, b"")
                    else:
                        finish.set()
                        assert await b.next() == (4, b"")
                    await b.end()
                await forgotten(runtime)
                assert conversions == 1
                assert len(numeric_manifests(settings.cache_dir)) == (0 if cancel_all else 1)
                if not cancel_all:
                    async with client.stream("GET", second_url + "data") as warm:
                        reader = Frames(warm)
                        assert (await reader.next())[0] == 1
                        assert await payload(reader) == expected
                    assert conversions == 1
            await forgotten(runtime)
            assert not list(settings.cache_dir.glob(".tmp-*"))

    run(scenario())


@pytest.mark.parametrize("warm", [False, True])
def test_source_mutation_rejects_even_warm_decoded_artifacts(
    settings: Settings, packed: PackedFixture, warm: bool
) -> None:
    directory = packed.write(settings.model_root, split=True)
    with TestClient(create_app(settings)) as client:
        url = address(client, packed)
        if warm:
            for endpoint in ["data", "statistics", "distributions"]:
                assert frames(client.get(url + endpoint).content)[-1] == (4, b"")
        mutate_last_byte(directory / "part-1.safetensors")
        for endpoint in ["data", "statistics", "distributions"]:
            response = client.get(url + endpoint)
            assert response.status_code == 409
            assert response.json()["code"] == "model_content_changed"
            assert "x-operation-id" not in response.headers
            assert str(directory) not in response.text


def test_mutation_mid_decode_never_publishes_complete_artifact(
    settings: Settings, packed: PackedFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    directory = packed.write(settings.model_root, split=True)
    original = LogicalTensor.produce

    async def changed(self: LogicalTensor, context: ProducerContext) -> None:
        append = context.append

        async def mutate_after_append(data: bytes) -> None:
            await append(data)
            mutate_last_byte(directory / "part-1.safetensors")

        context.append = mutate_after_append  # type: ignore[method-assign]
        await original(self, context)

    monkeypatch.setattr(LogicalTensor, "produce", changed)
    with TestClient(create_app(settings)) as client:
        result = frames(client.get(address(client, packed) + "data").content)
        assert result[-1][0] == 5
        assert json.loads(result[-1][1])["code"] == "model_content_changed"
        assert all(kind != 4 for kind, _ in result)
    assert not numeric_manifests(settings.cache_dir)
    assert not list(settings.cache_dir.glob(".tmp-*"))


def test_invalid_gptq_group_fails_stream_without_cache(settings: Settings) -> None:
    packed = gptq_fixture()
    indices = packed.storage[-1]
    packed = replace(
        packed,
        storage=packed.storage[:-1]
        + (replace(indices, data=struct.pack("<i", -1) + indices.data[4:]),),
    )
    packed.write(settings.model_root, split=True)
    with TestClient(create_app(settings)) as client:
        result = frames(client.get(address(client, packed) + "data").content)
        assert result[-1][0] == 5
        assert json.loads(result[-1][1])["code"] in {
            "validation_error",
            "unsupported_representation",
        }
        assert all(kind not in {2, 4} for kind, _ in result)
    assert not numeric_manifests(settings.cache_dir)
    assert not list(settings.cache_dir.glob(".tmp-*"))

"""Independent float64 oracles, exact layout, and real derived-operation lifetimes."""

import asyncio
import json
import math
import os
import statistics as reference
import struct
import subprocess
import sys
from pathlib import Path
from typing import Any
from uuid import uuid4

import pytest
import torch
from cache_helpers import numeric_manifests
from fastapi.testclient import TestClient
from test_models import make_model, mutate_last_byte, write_weights
from test_operations import run
from test_streaming import Frames, forgotten, server
from test_tensor_data import address, frames, payload, setup_stream

import llm_model_explorer
from llm_model_explorer.app import create_app
from llm_model_explorer.materialization import LogicalTensor
from llm_model_explorer.model_files import ModelError
from llm_model_explorer.operations import Cancellation, ProducerContext
from llm_model_explorer.services import Services
from llm_model_explorer.settings import Settings
from llm_model_explorer.stream_metadata import METADATA
from llm_model_explorer.tensor_analysis import (
    DOMAIN,
    FRACTIONS,
    STATISTICS,
    TensorAnalysis,
    checked_uint32,
    distributions,
    statistics,
)

EXTREME = float(torch.finfo(torch.float32).max)
CASES = [
    ([2, 3], [-2.0, 0.0, 2.0, -1.0, 0.0, 1.0]),
    ([2, 3], [-2.0, 0.0, 6.0, 0.0, 2.0, 4.0]),
    ([2, 100], [-10000.0, 10000.0] + [-0.125, 0.0, 0.125] * 66),
    ([2, 3], [-7.0, 0.0, 3.0, -2.0, 1.0, 12.0]),
    ([2, 3], [3.5] * 6),
    ([2, 3], [-0.0, 0.0, -0.0, 0.0, 0.0, -0.0]),
    ([2, 3], [1.0, 1.0, 1.0, 1.0, 1.0, 10000.0]),
    ([2, 3], [math.nan, -math.inf, math.inf, 1.0, -3.0, 9.0]),
    ([2, 3], [math.nan] * 6),
    ([1, 2], [math.inf, -math.inf]),
    ([2, 3], [-EXTREME, EXTREME, 0.0, 1.0, -1.0, EXTREME]),
    ([1, 3], [EXTREME, EXTREME, EXTREME]),
    ([1, 3], [EXTREME, 1.0, -EXTREME]),
    ([1, 4], [1.0, EXTREME, -EXTREME, 1.0]),
    ([1, 4], [2**-149, -(2**-149), 2**-149, 0.0]),
    ([1, 3], [math.nan, 1.25, math.inf]),
    ([0, 3], []),
    ([2, 0], []),
    ([0, 0], []),
]


def oracle(values: list[float]) -> list[float]:
    finite = sorted(x for x in values if math.isfinite(x))
    if not finite:
        return [math.nan] * 9
    quantiles = []
    for q in FRACTIONS:
        h = (len(finite) - 1) * q
        fraction = h - math.floor(h)
        quantiles.append((1 - fraction) * finite[math.floor(h)] + fraction * finite[math.ceil(h)])
    return [min(finite), max(finite), reference.mean(finite), reference.pstdev(finite), *quantiles]


def histogram_oracle(shape: list[int], values: list[float]) -> bytes:
    rows, columns = shape
    row_counts = [0] * (rows * 100)
    column_counts = [0] * (columns * 100)
    finite = [x for x in values if math.isfinite(x)]
    if finite:
        lo, hi = min(finite), max(finite)
        for index, value in enumerate(values):
            if not math.isfinite(value):
                continue
            bin_index = (
                50 if lo == hi else min(99, max(0, math.floor((value - lo) / (hi - lo) * 100)))
            )
            row_counts[(index // columns) * 100 + bin_index] += 1
            column_counts[bin_index * columns + index % columns] += 1
    counts = row_counts + column_counts
    return struct.pack(f"<{len(counts)}I", *counts)


def assert_statistics(raw: bytes, values: list[float]) -> None:
    count, finite, nonfinite, *scalars = STATISTICS.unpack(raw)
    assert count == len(values)
    assert finite == sum(math.isfinite(x) for x in values)
    assert nonfinite == count - finite
    expected = oracle(values)
    if finite:
        assert scalars[:2] == expected[:2]
        assert scalars[2:] == pytest.approx(expected[2:], rel=1e-6, abs=1e-7)
    else:
        assert all(math.isnan(x) for x in scalars)


@pytest.mark.parametrize("shape,values", CASES)
def test_native_numerics(shape: list[int], values: list[float]) -> None:
    tensor = torch.tensor(values, dtype=torch.float32)
    assert_statistics(statistics(tensor, Cancellation()), values)
    domain, counts = distributions(tensor, shape[0], shape[1], Cancellation())
    expected_bytes = histogram_oracle(shape, values)
    assert counts.tolist() == list(struct.unpack(f"<{counts.numel()}I", expected_bytes))
    finite_count = sum(math.isfinite(x) for x in values)
    rows, columns = shape
    row_counts = counts[: rows * 100].to(torch.int64).reshape(rows, 100)
    column_counts = counts[rows * 100 :].to(torch.int64).reshape(100, columns)
    assert row_counts.sum().item() == column_counts.sum().item() == finite_count
    assert row_counts.sum(dim=1).tolist() == [
        sum(math.isfinite(x) for x in values[r * columns : (r + 1) * columns]) for r in range(rows)
    ]
    assert column_counts.sum(dim=0).tolist() == [
        sum(math.isfinite(values[r * columns + c]) for r in range(rows)) for c in range(columns)
    ]
    lo, hi = DOMAIN.unpack(domain)
    if finite_count:
        assert (lo, hi) == tuple(oracle(values)[:2])
    else:
        assert math.isnan(lo) and math.isnan(hi)


def test_exact_percentiles_above_quantile_library_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*args: Any, **kwargs: Any) -> None:
        pytest.fail("reference-scale input must not require torch.quantile")

    monkeypatch.setattr(torch, "quantile", forbidden)
    # Generated only in RAM. All integers through 2**24 are exact F32 values.
    n = 2**24 + 1
    values = torch.arange(n, dtype=torch.float64).to(torch.float32).flip(0)
    count, finite, nonfinite, lo, hi, mean, stddev, *quantiles = STATISTICS.unpack(
        statistics(values, Cancellation())
    )
    assert (count, finite, nonfinite, lo, hi) == (n, n, 0, 0, n - 1)
    assert mean == pytest.approx((n - 1) / 2, rel=1e-6, abs=1e-7)
    assert stddev == pytest.approx(math.sqrt((n * n - 1) / 12), rel=1e-6, abs=1e-7)
    assert quantiles == pytest.approx([(n - 1) * q for q in FRACTIONS], rel=1e-6, abs=1e-7)


def test_uint32_overflow_rejected_before_conversion() -> None:
    assert checked_uint32(torch.tensor([2**32 - 1], dtype=torch.int64)).item() == 2**32 - 1
    for count in [-1, 2**32]:
        with pytest.raises(ModelError, match="exceeds uint32") as caught:
            checked_uint32(torch.tensor([count], dtype=torch.int64))
        assert caught.value.code == "unsupported_size"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device unavailable")
@pytest.mark.parametrize("shape,values", CASES)
def test_actual_cuda_equivalence(shape: list[int], values: list[float]) -> None:
    tensor = torch.tensor(values, dtype=torch.float32, device="cuda")
    assert_statistics(statistics(tensor, Cancellation()), values)
    _, counts = distributions(tensor, shape[0], shape[1], Cancellation())
    assert counts.tolist() == list(
        struct.unpack(f"<{counts.numel()}I", histogram_oracle(shape, values))
    )
    torch.cuda.synchronize()


@pytest.mark.parametrize("shape,values", CASES)
def test_endpoints_and_disk_reuse(
    settings: Settings, monkeypatch: pytest.MonkeyPatch, shape: list[int], values: list[float]
) -> None:
    directory = make_model(settings.model_root)
    write_weights(directory / "model.safetensors", [("weight", "F32", shape, values)])
    with TestClient(create_app(settings)) as client:
        url = address(client).removesuffix("data")
        for endpoint in ["statistics", "distributions"]:
            response = client.get(url + endpoint)
            assert response.status_code == 200
            assert response.headers["cache-control"] == "no-store"
            assert response.headers["x-operation-id"]
            result = frames(response.content)
            assert result[0][0] == 1 and result[-1] == (4, b"")
            metadata = METADATA.validate_json(result[0][1]).model_dump()
            data = b"".join(block for kind, block in result if kind == 2)
            if endpoint == "statistics":
                assert [kind for kind, _ in result] == [1, 4]
                assert metadata["byte_length"] == 0
                assert metadata["count"] == len(values)
                assert metadata["finite_count"] == sum(math.isfinite(x) for x in values)
                actual = [metadata[key] for key in ["minimum", "maximum", "mean", "stddev"]]
                actual.extend(metadata["percentiles"].values())
                expected = oracle(values)
                if metadata["finite_count"]:
                    assert actual == pytest.approx(expected, rel=1e-6, abs=1e-7)
                else:
                    assert actual == [None] * 9
            else:
                assert data == histogram_oracle(shape, values)
                finite = [x for x in values if math.isfinite(x)]
                assert metadata["domain_minimum"] == (min(finite) if finite else None)
                assert metadata["domain_maximum"] == (max(finite) if finite else None)
                assert metadata["byte_length"] == len(data)
                assert metadata["sections"][1]["offset"] == shape[0] * 400
        assert len(numeric_manifests(settings.cache_dir)) == 2

    # New app/session proves persistent cache reuse without resident result state.
    async def forbidden(self: TensorAnalysis, context: ProducerContext) -> None:
        pytest.fail("warm derived request recomputed")

    monkeypatch.setattr(TensorAnalysis, "produce", forbidden)
    with TestClient(create_app(settings)) as client:
        url = address(client).removesuffix("data")
        for endpoint in ["statistics", "distributions"]:
            assert frames(client.get(url + endpoint).content)[-1] == (4, b"")


@pytest.mark.parametrize("dtype", ["F16", "BF16"])
def test_analysis_uses_logical_float32(settings: Settings, dtype: str) -> None:
    directory = make_model(settings.model_root)
    values = [-2.5, -0.0, 0.125, 1.0, 2.0, 10.0]
    write_weights(directory / "model.safetensors", [("weight", dtype, [2, 3], values)])
    with TestClient(create_app(settings)) as client:
        url = address(client).removesuffix("data")
        for endpoint in ["statistics", "distributions"]:
            assert frames(client.get(url + endpoint).content)[-1] == (4, b"")
    assert len(numeric_manifests(settings.cache_dir)) == 3


def test_preflight_and_content_invalidation(settings: Settings) -> None:
    directory = make_model(settings.model_root)
    app = create_app(settings)
    with TestClient(app) as client:
        url = address(client).removesuffix("data")
        for endpoint in ["statistics", "distributions"]:
            for bad, status in [
                (url.replace(url.split("/")[2], "bad"), 422),
                (url.replace(url.split("/")[2], str(uuid4())), 404),
                (url.replace(url.split("/")[-2], "missing"), 404),
            ]:
                response = client.get(bad + endpoint)
                assert response.status_code == status and "x-operation-id" not in response.headers
            assert frames(client.get(url + endpoint).content)[-1] == (4, b"")
        mutate_last_byte(directory / "model.safetensors")
        for endpoint in ["statistics", "distributions"]:
            response = client.get(url + endpoint)
            assert response.status_code == 409
            assert response.json()["code"] == "model_content_changed"
        new_url = address(client).removesuffix("data")
        for endpoint in ["statistics", "distributions"]:
            assert frames(client.get(new_url + endpoint).content)[-1] == (4, b"")
    assert len(numeric_manifests(settings.cache_dir)) == 4


@pytest.mark.parametrize("shape,values", [([], [1.0]), ([2], [1.0, 2.0]), ([1, 1, 1], [1.0])])
def test_rank_rejected_before_stream(
    settings: Settings, shape: list[int], values: list[float]
) -> None:
    directory = make_model(settings.model_root)
    write_weights(directory / "model.safetensors", [("weight", "F32", shape, values)])
    with TestClient(create_app(settings)) as client:
        url = address(client).removesuffix("data")
        response = client.get(url + "distributions")
        assert response.status_code == 422
        assert response.json()["code"] == "unsupported_rank"
        assert "x-operation-id" not in response.headers
        assert frames(client.get(url + "statistics").content)[-1] == (4, b"")


@pytest.mark.parametrize("endpoint", ["statistics", "distributions"])
@pytest.mark.parametrize("failure", [MemoryError, torch.OutOfMemoryError])
def test_allocation_failure_is_clean(
    settings: Settings, monkeypatch: pytest.MonkeyPatch, endpoint: str, failure: type[Exception]
) -> None:
    make_model(settings.model_root)

    def fail(*args: Any, **kwargs: Any) -> Any:
        raise failure("/private/model/path")

    monkeypatch.setattr(TensorAnalysis, "_calculate", fail)
    with TestClient(create_app(settings)) as client:
        response = client.get(address(client).removesuffix("data") + endpoint)
        result = frames(response.content)
        assert result[0][0] == 5
        assert json.loads(result[0][1])["code"] == "resource_exhausted"
        assert b"private" not in response.content
    assert not numeric_manifests(settings.cache_dir)
    assert not list(settings.cache_dir.glob(".tmp-*"))


@pytest.mark.skipif(
    sys.platform != "linux", reason="real allocator probe uses Linux RLIMIT_AS/proc"
)
def test_real_cpu_allocator_exhaustion(tmp_path: Path) -> None:
    environment = dict(os.environ)
    # Preserve the package actually under test, including isolated wheel runs.
    environment["PYTHONPATH"] = str(Path(llm_model_explorer.__file__).resolve().parents[1])
    environment["OMP_NUM_THREADS"] = "2"
    probe = subprocess.run(
        [sys.executable, str(Path(__file__).with_name("cpu_allocation_probe.py")), str(tmp_path)],
        env=environment,
        capture_output=True,
        text=True,
        timeout=60,
        check=True,
    )
    result = json.loads(probe.stdout)
    assert result["allocator_type"] == "RuntimeError"
    assert "DefaultCPUAllocator: can't allocate memory:" in result["allocator_message"]
    assert "Error code 12" in result["allocator_message"]
    assert result["status"] == 200 and result["frame_types"] == [5]
    assert result["error"] == {
        "code": "resource_exhausted",
        "message": "Insufficient memory for tensor analysis.",
    }
    assert result["complete_artifacts"] == 1  # Warmed statistics only.
    assert result["temporary_artifacts"] == 0


@pytest.mark.parametrize(
    "message",
    [
        "unrelated native calculation failure",
        "DefaultCPUAllocator: can't allocate memory: Error code 22 (Invalid argument)",
        "OtherAllocator: can't allocate memory: Error code 12 (Cannot allocate memory)",
    ],
)
def test_unrelated_runtime_errors_remain_internal(
    settings: Settings, monkeypatch: pytest.MonkeyPatch, message: str
) -> None:
    make_model(settings.model_root)

    def fail(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError(message)

    monkeypatch.setattr(TensorAnalysis, "_calculate", fail)
    with TestClient(create_app(settings)) as client:
        response = client.get(address(client).removesuffix("data") + "statistics")
        result = frames(response.content)
        assert [kind for kind, _ in result] == [5]
        assert json.loads(result[0][1])["code"] == "internal_error"
    assert not numeric_manifests(settings.cache_dir)


@pytest.mark.parametrize("cancel_all", [False, True])
def test_shared_cancellation_and_tensor_first_data(
    settings: Settings, monkeypatch: pytest.MonkeyPatch, cancel_all: bool
) -> None:
    directory = make_model(settings.model_root)
    write_weights(directory / "model.safetensors", [("weight", "F16", [2, 3], [1.0] * 6)])

    async def scenario() -> None:
        app = create_app(settings)
        async with app.router.lifespan_context(app):
            services: Services = app.state.services
            runtime = services.operation_delivery
            assert runtime is not None
            gate = asyncio.Event()
            entered = asyncio.Event()
            calls = 0
            conversions = 0
            original = TensorAnalysis.produce
            original_materialize = LogicalTensor.produce

            async def held(self: TensorAnalysis, context: ProducerContext) -> None:
                nonlocal calls
                calls += 1
                entered.set()
                await context.cancellation.wait(gate)
                await original(self, context)

            async def materialize(self: LogicalTensor, context: ProducerContext) -> None:
                nonlocal conversions
                conversions += 1
                await original_materialize(self, context)

            monkeypatch.setattr(TensorAnalysis, "produce", held)
            monkeypatch.setattr(LogicalTensor, "produce", materialize)
            async with server(app) as client:
                url1 = (await setup_stream(client)).removesuffix("data")
                url2 = (await setup_stream(client)).removesuffix("data")
                async with (
                    client.stream("GET", url1 + "statistics") as first,
                    client.stream("GET", url2 + "statistics") as second,
                    client.stream("GET", url2 + "distributions") as third,
                ):
                    await entered.wait()
                    a, b, c = Frames(first), Frames(second), Frames(third)
                    assert first.headers["x-operation-id"] != second.headers["x-operation-id"]
                    # Both derived producers are deliberately stalled. Tensor
                    # bytes still arrive, establishing absence of first-render gating.
                    async with client.stream("GET", url1 + "data") as data:
                        reader = Frames(data)
                        assert (await reader.next())[0] == 1
                        assert await payload(reader) == struct.pack("<6f", *([1.0] * 6))
                    await client.delete("/operations/" + first.headers["x-operation-id"])
                    assert await a.next() == (6, b"")
                    await a.end()
                    if cancel_all:
                        for response, reader in [(second, b), (third, c)]:
                            await client.delete("/operations/" + response.headers["x-operation-id"])
                            assert await reader.next() == (6, b"")
                            await reader.end()
                    else:
                        gate.set()
                        assert (await b.next())[0] == 1
                        assert await payload(b) == b""
                        assert (await c.next())[0] == 1
                        assert await payload(c) == histogram_oracle([2, 3], [1.0] * 6)
            await forgotten(runtime)
            assert calls == 2 and conversions == 1
            assert len(numeric_manifests(settings.cache_dir)) == (1 if cancel_all else 3)
            assert not list(settings.cache_dir.glob(".tmp-*"))

    run(scenario())


def test_binning_across_work_blocks() -> None:
    # Asymmetric dimensions cross both source blocks and uint32 output frames.
    rows, columns = 257, 509
    values = torch.arange(rows * columns, dtype=torch.float32).remainder(101) - 50
    values[::137] = math.nan
    _, counts = distributions(values, rows, columns, Cancellation())
    expected = histogram_oracle([rows, columns], values.tolist())
    assert counts.tolist() == list(struct.unpack(f"<{counts.numel()}I", expected))


def test_shared_golden_uint32_bytes() -> None:
    from test_lmex import FIXTURES

    from llm_model_explorer.tensor_analysis import _count_bytes

    case = FIXTURES["numeric_cases"]["uint32"]
    counts = checked_uint32(torch.tensor(case["values"], dtype=torch.int64))
    assert _count_bytes(counts).hex() == case["little_endian_hex"]
    overflow = FIXTURES["numeric_cases"]["histogram_overflow"]
    with pytest.raises(ModelError) as caught:
        checked_uint32(torch.tensor([overflow["count"]], dtype=torch.int64))
    assert caught.value.code == overflow["error_code"]


@pytest.mark.parametrize("cancel_all", [False, True])
def test_device_queue_dependency_and_mid_materialization_cancellation(
    settings: Settings, monkeypatch: pytest.MonkeyPatch, cancel_all: bool
) -> None:
    directory = make_model(settings.model_root)
    write_weights(directory / "model.safetensors", [("weight", "BF16", [2, 3], [1.0] * 6)])

    async def scenario() -> None:
        app = create_app(settings)
        async with app.router.lifespan_context(app):
            services: Services = app.state.services
            runtime = services.operation_delivery
            assert runtime is not None
            # Exercise the real per-device queue with CPU callbacks; actual GPU
            # numerics have a separate capability-gated test, never a fake pass.
            runtime.device = "cuda:0"
            started, finish = asyncio.Event(), asyncio.Event()
            conversions = 0
            original = LogicalTensor.produce

            async def held(self: LogicalTensor, context: ProducerContext) -> None:
                nonlocal conversions
                conversions += 1
                await context.compute(lambda: None)
                append = context.append

                async def hold_after_append(data: bytes) -> None:
                    await append(data)
                    started.set()
                    await context.cancellation.wait(finish)

                context.append = hold_after_append  # type: ignore[method-assign]
                await original(self, context)

            def calculate(
                self: TensorAnalysis, values: torch.Tensor, cancellation: Cancellation
            ) -> tuple[bytes, torch.Tensor | None]:
                if self.kind == "tensor_statistics":
                    return statistics(values, cancellation), None
                return distributions(values, 2, 3, cancellation)

            monkeypatch.setattr(LogicalTensor, "produce", held)
            monkeypatch.setattr(TensorAnalysis, "_calculate", calculate)
            async with server(app) as client:
                url = (await setup_stream(client)).removesuffix("data")
                async with (
                    client.stream("GET", url + "statistics") as first,
                    client.stream("GET", url + "distributions") as second,
                    client.stream("GET", url + "data") as third,
                ):
                    await asyncio.wait_for(started.wait(), timeout=3)
                    a, b, c = Frames(first), Frames(second), Frames(third)
                    assert (await c.next())[0] == 1
                    assert (await c.next()) == (2, struct.pack("<6f", *([1.0] * 6)))
                    assert not numeric_manifests(settings.cache_dir)
                    if cancel_all:
                        for response, reader in [(first, a), (second, b), (third, c)]:
                            await client.delete("/operations/" + response.headers["x-operation-id"])
                            assert await reader.next() == (6, b"")
                            await reader.end()
                    else:
                        # Cancelling derived interest must not kill the conversion
                        # while tensor/distribution consumers are still attached.
                        await client.delete("/operations/" + first.headers["x-operation-id"])
                        assert await a.next() == (6, b"")
                        await a.end()
                        finish.set()
                        assert (await b.next())[0] == 1
                        assert await payload(b) == histogram_oracle([2, 3], [1.0] * 6)
                        assert await c.next() == (4, b"")
                        await c.end()
            await forgotten(runtime)
            assert conversions == 1
            assert len(numeric_manifests(settings.cache_dir)) == (0 if cancel_all else 2)
            assert not list(settings.cache_dir.glob(".tmp-*"))

    run(scenario())


@pytest.mark.parametrize("endpoint", ["statistics", "distributions"])
def test_source_change_during_analysis_leaves_no_artifact(
    settings: Settings, monkeypatch: pytest.MonkeyPatch, endpoint: str
) -> None:
    directory = make_model(settings.model_root)
    original = TensorAnalysis._calculate

    def changed(
        self: TensorAnalysis, values: torch.Tensor, cancellation: Cancellation
    ) -> tuple[bytes, torch.Tensor | None]:
        result = original(self, values, cancellation)
        mutate_last_byte(directory / "model.safetensors")
        return result

    monkeypatch.setattr(TensorAnalysis, "_calculate", changed)
    with TestClient(create_app(settings)) as client:
        result = frames(client.get(address(client).removesuffix("data") + endpoint).content)
        assert result[0][0] == 5
        assert json.loads(result[0][1])["code"] == "model_content_changed"
    assert not numeric_manifests(settings.cache_dir)


def test_unsafe_distribution_output_rejected_before_allocation(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    from dataclasses import replace

    from llm_model_explorer.materialization import LogicalTensorService

    make_model(settings.model_root)
    original = LogicalTensorService.resolve

    def oversized(self: LogicalTensorService, *args: Any) -> LogicalTensor:
        tensor = original(self, *args)
        return replace(
            tensor, descriptor=tensor.descriptor.model_copy(update={"shape": (0, 2**52)})
        )

    monkeypatch.setattr(LogicalTensorService, "resolve", oversized)
    with TestClient(create_app(settings)) as client:
        response = client.get(address(client).removesuffix("data") + "distributions")
        assert response.status_code == 422
        assert response.json()["code"] == "unsupported_size"
        assert "x-operation-id" not in response.headers


@pytest.mark.parametrize("endpoint", ["statistics", "distributions"])
@pytest.mark.parametrize("cancel", [False, True])
def test_partial_derived_spool_never_publishes(
    settings: Settings, monkeypatch: pytest.MonkeyPatch, endpoint: str, cancel: bool
) -> None:
    make_model(settings.model_root)

    async def scenario() -> None:
        app = create_app(settings)
        async with app.router.lifespan_context(app):
            services: Services = app.state.services
            runtime = services.operation_delivery
            assert runtime is not None
            appended, hold = asyncio.Event(), asyncio.Event()
            original = TensorAnalysis.produce

            async def interrupted(self: TensorAnalysis, context: ProducerContext) -> None:
                append = context.append

                async def stop_after_prefix(data: bytes) -> None:
                    await append(data)
                    appended.set()
                    if cancel:
                        await context.cancellation.wait(hold)
                    else:
                        raise MemoryError("/private/spool")

                context.append = stop_after_prefix  # type: ignore[method-assign]
                await original(self, context)

            monkeypatch.setattr(TensorAnalysis, "produce", interrupted)
            async with server(app) as client:
                url = (await setup_stream(client)).removesuffix("data")
                async with client.stream("GET", url + endpoint) as response:
                    await appended.wait()
                    assert not numeric_manifests(settings.cache_dir)
                    if cancel:
                        await client.delete("/operations/" + response.headers["x-operation-id"])
                    reader = Frames(response)
                    kind, block = await reader.next()
                    if kind == 1:
                        kind, block = await reader.next()
                    assert kind == (6 if cancel else 5)
                    if not cancel:
                        assert json.loads(block)["code"] == "resource_exhausted"
                        assert b"private" not in block
                    await reader.end()
            await forgotten(runtime)
            assert not numeric_manifests(settings.cache_dir)
            assert not list(settings.cache_dir.glob(".tmp-*"))

    run(scenario())

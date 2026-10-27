"""Observation-specific numerics and transport; generic lifetime stays in runtime tests."""

import asyncio
import gc
import json
import struct
import weakref
from functools import partial
from pathlib import Path
from threading import Event
from typing import Any
from unittest.mock import Mock

import pytest
import torch
from fastapi.testclient import TestClient
from neuron_fixtures import KnownModel, direct, known_activations, reference_activations, source_for
from neuron_fixtures import tiny as tiny
from test_streaming import Frames, server

from llm_model_explorer import neuron_routes
from llm_model_explorer.app import create_app
from llm_model_explorer.execution import BlockingWork
from llm_model_explorer.model_files import ModelError
from llm_model_explorer.neuron_execution import (
    NeuronInput,
    NeuronLease,
    NeuronRuntime,
    admit,
    runtime,
)
from llm_model_explorer.neuron_execution.activations import (
    ActivationReader,
    Activations,
    subscribe_activations,
)
from llm_model_explorer.neuron_execution.capture import capture_activations
from llm_model_explorer.operations import (
    Cancellation,
    DeviceScheduler,
    OperationCancelled,
    SourceReader,
)
from llm_model_explorer.services import open_services
from llm_model_explorer.settings import Settings
from llm_model_explorer.streaming import LMEXResponse

TOKENS = (1, 0, 2, 1)  # Special EOS 0 and duplicate 1 remain in their original positions.
NEURONS = (15, 1, 7)


def selection(settings: Settings) -> NeuronInput:
    model = admit(source_for(settings), settings.model_root, "cpu")
    return NeuronInput(
        model, model.source.model_revision, model.modules[1]["module_id"], NEURONS, TOKENS
    )


@pytest.mark.parametrize("fixture", ["known", "transformers"])
def test_selected_capture_is_signed_exact_and_does_not_change_logits(
    tiny: tuple[Settings, Path, Any], monkeypatch: pytest.MonkeyPatch, fixture: str
) -> None:
    settings, _, reference = tiny
    request = selection(settings)
    if fixture == "known":
        reference = KnownModel()
        monkeypatch.setattr(runtime, "load_model", lambda *_: reference)
        expected = known_activations(TOKENS, NEURONS)
        assert (expected < 0).any() and (expected > 0).any()
        assert expected[1].tolist() == [0, 0, 0]
    else:
        expected = reference_activations(reference, TOKENS, 1)[0, :, list(NEURONS)]
    baseline = direct(reference, TOKENS)
    logits = []
    forward = NeuronLease.forward

    def observed_forward(lease: NeuronLease, token_ids: tuple[int, ...]) -> torch.Tensor:
        assert token_ids == TOKENS
        result = forward(lease, token_ids)
        logits.append(result)
        return result

    monkeypatch.setattr(NeuronLease, "forward", observed_forward)

    async def scenario() -> None:
        work = BlockingWork()
        execution = NeuronRuntime(DeviceScheduler(work), "cpu")
        try:
            actual = await execution.run(
                request, Cancellation(), lambda lease: capture_activations(lease, request)
            )
            assert actual.shape == (4, 3) and actual.dtype == torch.float32
            assert actual.device.type == "cpu" and not actual.requires_grad
            assert actual.is_contiguous() and actual.untyped_storage().nbytes() == 4 * 4 * 3
            # Float32 linear/SiLU kernels versus independent math; errors below
            # one part per million with a small absolute allowance near zero.
            torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)
            torch.testing.assert_close(logits[0], baseline, rtol=1e-6, atol=1e-7)
            # Returned storage is detached from runtime/projection storage.
            with torch.inference_mode():
                actual.fill_(100)
            again = await execution.run(
                request, Cancellation(), lambda lease: capture_activations(lease, request)
            )
            torch.testing.assert_close(again, expected, rtol=1e-6, atol=1e-7)
        finally:
            await execution.aclose()
            await work.aclose()

    asyncio.run(scenario())


@pytest.mark.parametrize("fault", ["shape", "missing", "twice", "weights"])
def test_incompatible_runtime_never_returns_approximate_selection(
    tiny: tuple[Settings, Path, Any], monkeypatch: pytest.MonkeyPatch, fault: str
) -> None:
    settings, _, _ = tiny
    request = selection(settings)
    model = KnownModel()
    if fault == "weights":
        model.layers[1].down_proj = torch.nn.Linear(15, 8, bias=False)
    monkeypatch.setattr(runtime, "load_model", lambda *_: model)
    forward = NeuronLease.forward

    def bad_forward(lease: NeuronLease, token_ids: tuple[int, ...]) -> torch.Tensor:
        if fault == "missing":
            return torch.zeros(1, 4, 8)
        if fault == "twice":
            forward(lease, token_ids)
        return forward(lease, token_ids[:1] if fault == "shape" else token_ids)

    monkeypatch.setattr(NeuronLease, "forward", bad_forward)

    async def scenario() -> None:
        work = BlockingWork()
        execution = NeuronRuntime(DeviceScheduler(work), "cpu")
        try:
            with pytest.raises(ModelError) as error:
                await execution.run(
                    request, Cancellation(), lambda lease: capture_activations(lease, request)
                )
            assert error.value.code == "unsupported_representation"
            assert not model.layers[1].down_proj._forward_pre_hooks
        finally:
            await execution.aclose()
            await work.aclose()

    asyncio.run(scenario())


@pytest.mark.parametrize("stop", ["failure", "cancel"])
def test_failure_after_capture_removes_hook_before_next_execution(
    tiny: tuple[Settings, Path, Any], monkeypatch: pytest.MonkeyPatch, stop: str
) -> None:
    settings, _, _ = tiny
    request = selection(settings)
    model = KnownModel()
    monkeypatch.setattr(runtime, "load_model", lambda *_: model)
    entered, release = Event(), Event()
    projection = model.layers[1].down_proj
    original = projection.forward

    def interrupted(values: torch.Tensor) -> torch.Tensor:
        # The actual capture pre-hook has already run before this boundary.
        entered.set()
        assert release.wait(10)
        if stop == "failure":
            raise ValueError("private activation diagnostic")
        return original(values)

    monkeypatch.setattr(projection, "forward", interrupted)

    async def scenario() -> None:
        work = BlockingWork()
        execution = NeuronRuntime(DeviceScheduler(work), "cpu")
        cancel = Cancellation()
        task = asyncio.create_task(
            execution.run(request, cancel, lambda lease: capture_activations(lease, request))
        )
        try:
            assert await asyncio.to_thread(entered.wait, 10)
            if stop == "cancel":
                cancel.cancel()
            assert not task.done()
            release.set()
            with pytest.raises(OperationCancelled if stop == "cancel" else ModelError) as error:
                await task
            assert "private" not in str(error.value)
            assert not projection._forward_pre_hooks
            monkeypatch.setattr(projection, "forward", original)
            actual = await execution.run(
                request, Cancellation(), lambda lease: capture_activations(lease, request)
            )
            torch.testing.assert_close(
                actual, known_activations(TOKENS, NEURONS), rtol=1e-6, atol=1e-7
            )
        finally:
            release.set()
            await execution.aclose()
            await work.aclose()

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "fields,code",
    [
        ({"token_ids": [True]}, "validation_error"),
        ({"neuron_indices": [1.0]}, "validation_error"),
        ({"token_ids": []}, "validation_error"),
        ({"neuron_indices": [1, 1]}, "validation_error"),
        ({"token_ids": [1] * 33}, "unsupported_size"),
        ({"model_revision": "old"}, "model_content_changed"),
        ({"module_id": "unknown"}, "validation_error"),
        ({"batch": 2}, "validation_error"),
    ],
)
def test_route_preflight_rejects_before_loading(
    tiny: tuple[Settings, Path, Any],
    monkeypatch: pytest.MonkeyPatch,
    fields: dict[str, Any],
    code: str,
) -> None:
    settings, _, _ = tiny
    fail = Mock(side_effect=AssertionError("invalid request loaded model"))
    monkeypatch.setattr(runtime, "load_model", fail)
    with TestClient(create_app(settings)) as client:
        model_id = client.get("/models").json()["models"][0]["id"]
        session = client.post("/sessions", json={"model_id": model_id}).json()
        modules = client.get(f"/sessions/{session['id']}/neuron-capabilities").json()["modules"]
        response = client.post(
            f"/sessions/{session['id']}/neuron-activations",
            json=dict(
                model_revision=session["model_revision"],
                module_id=modules[1]["module_id"],
                neuron_indices=list(NEURONS),
                token_ids=list(TOKENS),
            )
            | fields,
        )
        assert response.status_code == (409 if code == "model_content_changed" else 422)
        assert response.json()["code"] == code
        assert response.headers["cache-control"] == "no-store"
        assert "x-operation-id" not in response.headers
    fail.assert_not_called()


def test_real_tcp_metadata_selected_bytes_partiality_and_terminal(
    tiny: tuple[Settings, Path, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    settings, _, reference = tiny
    expected = reference_activations(reference, TOKENS, 1)[0, :, list(NEURONS)]
    monkeypatch.setattr(neuron_routes, "LMEXResponse", partial(LMEXResponse, max_data_bytes=8))
    app = create_app(settings)

    async def scenario() -> None:
        async with app.router.lifespan_context(app):
            before = {p: p.read_bytes() for p in settings.cache_dir.rglob("*") if p.is_file()}
            async with server(app) as client:
                model_id = (await client.get("/models")).json()["models"][0]["id"]
                session = (await client.post("/sessions", json={"model_id": model_id})).json()
                prefix = f"/sessions/{session['id']}"
                capabilities = (await client.get(prefix + "/neuron-capabilities")).json()
                module = capabilities["modules"][1]
                body = dict(
                    model_revision=session["model_revision"],
                    module_id=module["module_id"],
                    neuron_indices=list(NEURONS),
                    token_ids=list(TOKENS),
                )
                async with client.stream(
                    "POST", prefix + "/neuron-activations", json=body
                ) as response:
                    assert response.status_code == 200
                    assert response.headers["cache-control"] == "no-store"
                    assert response.headers["x-operation-id"]
                    frames = Frames(response)
                    kind, payload = await frames.next()
                    assert kind == 1
                    metadata = json.loads(payload)
                    assert metadata == dict(
                        kind="neuron_activations",
                        session_id=session["id"],
                        model_id=model_id,
                        model_revision=session["model_revision"],
                        execution=capabilities["execution"],
                        limits=capabilities["limits"],
                        module=module,
                        token_ids=list(TOKENS),
                        neuron_indices=list(NEURONS),
                        dtype="float32",
                        byte_order="little",
                        layout="c",
                        sections=[dict(name="activations", shape=[4, 3], offset=0, byte_length=48)],
                        byte_length=48,
                    )
                    data = bytearray()
                    for _ in range(6):
                        assert len(data) < 48  # Each prefix is incomplete, never success.
                        kind, payload = await frames.next()
                        assert kind == 2 and len(payload) == 8
                        data.extend(payload)
                    actual = torch.tensor(struct.unpack("<12f", data)).reshape(4, 3)
                    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)
                    assert await frames.next() == (4, b"")
                    await frames.end()
            assert {
                p: p.read_bytes() for p in settings.cache_dir.rglob("*") if p.is_file()
            } == before

    asyncio.run(scenario())


@pytest.mark.parametrize("when", ["eof", "after_compute"])
def test_source_guard_releases_result_even_with_retained_error(
    tiny: tuple[Settings, Path, Any], monkeypatch: pytest.MonkeyPatch, when: str
) -> None:
    settings, directory, _ = tiny
    readers: list[weakref.ReferenceType[ActivationReader]] = []
    original = ActivationReader.__init__

    def track(reader: ActivationReader, *args: Any) -> None:
        original(reader, *args)
        readers.append(weakref.ref(reader))

    monkeypatch.setattr(ActivationReader, "__init__", track)

    def mutate() -> None:
        path = directory / "config.json"
        path.write_text(path.read_text() + " ")

    compute = Activations.compute

    def mutate_after_compute(result: Activations, lease: NeuronLease) -> SourceReader:
        reader = compute(result, lease)
        mutate()
        return reader

    if when == "after_compute":
        monkeypatch.setattr(Activations, "compute", mutate_after_compute)

    async def scenario() -> None:
        async with open_services(settings) as services:
            sessions, execution = services.sessions, services.neurons
            assert sessions is not None and execution is not None
            model_id = source_for(settings).model_id
            session = await sessions.create(model_id)
            request = selection(settings)
            _, consumer = await subscribe_activations(
                sessions,
                execution,
                settings.model_root,
                session.id,
                session.source.model_revision,
                request.module_id,
                NEURONS,
                TOKENS,
            )
            try:
                if when == "eof":
                    assert len(await consumer.read()) == 48
                    mutate()
                with pytest.raises(ModelError) as error:
                    await consumer.read()
                assert error.value.code == "model_content_changed"
                assert consumer.terminal == "error"
            finally:
                await consumer.aclose()
            gc.collect()
            assert readers[0]() is None

    asyncio.run(scenario())

"""Real local Transformers forward plus independent binding/lifetime controls.

Tiny geometry is patched only in this fixture; production admission remains the
fixed reviewed checkpoint. These tests do not attest full reference acceptance.
"""

import asyncio
import gc
import json
import weakref
from dataclasses import replace
from pathlib import Path
from threading import Event
from typing import Any
from unittest.mock import Mock

import pytest
import torch
from fastapi.testclient import TestClient
from neuron_fixtures import TOKENS, direct, reference_activations, request, source_for
from neuron_fixtures import tiny as tiny

from llm_model_explorer.app import create_app
from llm_model_explorer.artifacts import ArtifactSpec
from llm_model_explorer.execution import BlockingWork
from llm_model_explorer.model_files import ModelError
from llm_model_explorer.neuron_execution import (
    AdmittedModel,
    NeuronLease,
    NeuronRuntime,
    Unavailable,
    admission,
    admit,
    runtime,
)
from llm_model_explorer.operations import Cancellation, DeviceScheduler, OperationCancelled
from llm_model_explorer.services import open_services
from llm_model_explorer.settings import Settings
from llm_model_explorer.tensor_source import ModelSource


def test_real_forward_exact_bindings_and_operation_local_hooks(
    tiny: tuple[Settings, Path, Any],
) -> None:
    settings, _, reference = tiny
    admitted = admit(source_for(settings), settings.model_root, "cpu")
    expected = direct(reference)
    activations = reference_activations(reference, TOKENS, 1)
    assert len(admitted.modules) == 2
    module = admitted.modules[1]
    assert module["module"] == {"kind": "module", "name": "model.layers.1.mlp"}
    assert module["weights"]["gate"] == dict(
        tensor_id=admitted.tensors["model.layers.1.mlp.gate_proj.weight"].id,
        shape=[16, 8],
        axis="row",
    )
    assert module["weights"]["down"] == dict(
        tensor_id=admitted.tensors["model.layers.1.mlp.down_proj.weight"].id,
        shape=[8, 16],
        axis="column",
    )

    async def scenario() -> None:
        work = BlockingWork()
        execution = NeuronRuntime(DeviceScheduler(work), "cpu")
        captured = []
        rng = torch.random.get_rng_state().clone()
        try:

            def observe(lease: NeuronLease) -> torch.Tensor:
                assert torch.is_inference_mode_enabled() and not torch.is_grad_enabled()
                with lease.hook(module["module_id"], lambda a: captured.append(a.clone())):
                    return lease.forward(TOKENS)

            actual = await execution.run(request(admitted), Cancellation(), observe)
            torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)
            torch.testing.assert_close(captured[0], activations, rtol=1e-6, atol=1e-7)
            assert not actual.requires_grad
            assert torch.equal(rng, torch.random.get_rng_state())

            def intervene(lease: NeuronLease) -> torch.Tensor:
                with lease.hook(module["module_id"], torch.zeros_like):
                    return lease.forward(TOKENS)

            changed = await execution.run(request(admitted), Cancellation(), intervene)
            assert not torch.allclose(changed, expected, atol=1e-7, rtol=1e-6)
            baseline = await execution.run(
                request(admitted), Cancellation(), lambda lease: lease.forward(TOKENS)
            )
            torch.testing.assert_close(baseline, expected, rtol=1e-6, atol=1e-7)
        finally:
            await execution.aclose()
            await work.aclose()

    asyncio.run(scenario())


def test_cuda_queue_is_shared_and_synchronization_failure_is_recoverable(
    tiny: tuple[Settings, Path, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Scheduling/OOM double, not evidence of real CUDA numerical execution."""
    settings, directory, _ = tiny
    cpu = admit(source_for(settings), settings.model_root, "cpu")
    admitted = replace(cpu, execution=dict(cpu.execution, device="cuda:0"))
    monkeypatch.setattr(runtime, "runtime_identity", lambda _: admitted.execution)
    loader = Mock(return_value=object())
    monkeypatch.setattr(runtime, "load_model", loader)
    synchronize = Mock(side_effect=[torch.OutOfMemoryError("private diagnostic"), None])
    monkeypatch.setattr(torch.cuda, "synchronize", synchronize)
    entered, release = Event(), Event()

    async def scenario() -> None:
        work = BlockingWork()
        scheduler = DeviceScheduler(work)
        execution = NeuronRuntime(scheduler, "cuda:0")

        def other_compute() -> None:
            entered.set()
            assert release.wait(10)

        occupying = asyncio.create_task(scheduler.run("cuda:0", Cancellation(), other_compute))
        try:
            assert await asyncio.to_thread(entered.wait, 10)
            cancellation = Cancellation()
            queued = asyncio.create_task(
                execution.run(request(admitted), cancellation, lambda _: 7)
            )
            # Let run register before cancellation; it cannot load behind the occupied queue.
            await asyncio.sleep(0)
            cancellation.cancel()
            with pytest.raises(OperationCancelled):
                await queued
            loader.assert_not_called()
            release.set()
            await occupying
            with pytest.raises(ModelError) as error:
                await execution.run(request(admitted), Cancellation(), lambda _: 7)
            assert error.value.code == "resource_exhausted"
            assert "private" not in str(error.value)
            assert await execution.run(request(admitted), Cancellation(), lambda _: 9) == 9
            assert loader.call_count == 2
            assert synchronize.call_count == 2

            def mutate_during_synchronization(_device: str) -> None:
                path = directory / "config.json"
                path.write_text(path.read_text() + " ")

            synchronize.side_effect = mutate_during_synchronization
            with pytest.raises(ModelError) as changed:
                await execution.run(request(admitted), Cancellation(), lambda _: 11)
            assert changed.value.code == "model_content_changed"
        finally:
            release.set()
            await occupying
            await execution.aclose()
            await work.aclose()

    asyncio.run(scenario())


def test_capability_unavailability_and_stale_session_use_contract_responses(
    tiny: tuple[Settings, Path, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings, directory, _ = tiny
    with TestClient(create_app(settings)) as client:
        model_id = client.get("/models").json()["models"][0]["id"]
        session = client.post("/sessions", json={"model_id": model_id}).json()
        url = f"/sessions/{session['id']}/neuron-capabilities"
        with monkeypatch.context() as patch:
            patch.setattr(
                "llm_model_explorer.neuron_execution.admission.transformers.__version__",
                "unreviewed",
            )
            response = client.get(url)
            assert response.status_code == 200
            unavailable = response.json()
            for feature in ("observation", "intervention", "generation"):
                assert unavailable[feature]["status"] == "unavailable"
                assert unavailable[feature]["reason"]["code"] == "runtime_unavailable"
            assert unavailable["execution"] is unavailable["limits"] is None
            assert unavailable["modules"] == []
        config_path = directory / "config.json"
        config_path.write_text(config_path.read_text() + " ")
        stale = client.get(url)
        assert stale.status_code == 409
        assert stale.json()["code"] == "model_content_changed"
        assert stale.headers["cache-control"] == "no-store"


def test_cancellation_during_load_and_before_first_load(
    tiny: tuple[Settings, Path, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings, _, _ = tiny
    admitted = admit(source_for(settings), settings.model_root, "cpu")
    entered, release = Event(), Event()
    original = runtime.load_model

    def load(model: AdmittedModel, cancellation: Cancellation) -> Any:
        entered.set()
        assert release.wait(10)
        return original(model, cancellation)

    loader = Mock(side_effect=load)
    monkeypatch.setattr(runtime, "load_model", loader)

    async def scenario() -> None:
        work = BlockingWork()
        execution = NeuronRuntime(DeviceScheduler(work), "cpu")
        operation = Mock(side_effect=AssertionError("cancelled forward started"))
        cancelled = Cancellation()
        cancelled.cancel()
        try:
            with pytest.raises(OperationCancelled):
                await execution.run(request(admitted), cancelled, operation)
            loader.assert_not_called()
            task = asyncio.create_task(execution.run(request(admitted), Cancellation(), operation))
            assert await asyncio.to_thread(entered.wait, 10)
            task.cancel()
            await asyncio.sleep(0)
            task.cancel()
            await asyncio.sleep(0)
            assert not task.done()
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await task
            operation.assert_not_called()
        finally:
            release.set()
            await execution.aclose()
            await work.aclose()

    asyncio.run(scenario())


def test_capabilities_and_static_inspection_never_load_weights(
    tiny: tuple[Settings, Path, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings, _, _ = tiny
    fail = Mock(side_effect=AssertionError("metadata must not load weights"))
    monkeypatch.setattr(runtime, "load_model", fail)
    monkeypatch.setattr(ModelSource, "iter_tensor", fail)
    with TestClient(create_app(settings)) as client:
        model = client.get("/models").json()["models"][0]
        created = client.post("/sessions", json={"model_id": model["id"]})
        assert created.status_code == 201
        session = created.json()
        prefix = f"/sessions/{session['id']}"
        response = client.get(prefix + "/neuron-capabilities")
        assert response.status_code == 200
        assert response.headers["cache-control"] == "no-store"
        value = response.json()
        assert value["model_revision"] == session["model_revision"]
        assert (
            value["observation"]
            == value["intervention"]
            == value["generation"]
            == {"status": "available"}
        )
        assert value["execution"]["dtype"] == "float32"
        assert value["execution"]["device"] == "cpu"
        assert value["limits"]["max_input_tokens"] == 32
        assert value["limits"]["eos_token_ids"] == [0]
        assert len(value["modules"]) == 2
        assert client.get(prefix + "/tensors").status_code == 200
        assert client.get(prefix + "/architecture").status_code == 200
        assert str(settings.model_root) not in response.text
        assert source_for(settings).fingerprint not in response.text
    fail.assert_not_called()


@pytest.mark.parametrize(
    "change,reason",
    [
        ({"_name_or_path": "HuggingFaceTB/SmolLM2-135M-Instruct"}, "unsupported_model"),
        ({"_commit_hash": "wrong"}, "unsupported_model"),
        ({"hidden_size": 12}, "unsupported_model"),
        ({"hidden_act": "relu"}, "unsupported_model"),
        ({"auto_map": {"AutoModel": "malicious.Model"}}, "unsupported_model"),
        ({"attention_dropout": 0.1}, "unsupported_model"),
    ],
)
def test_rejects_other_checkpoints_without_disabling_static_inspection(
    tiny: tuple[Settings, Path, Any],
    change: dict[str, Any],
    reason: str,
) -> None:
    settings, directory, _ = tiny
    path = directory / "config.json"
    path.write_text(json.dumps(json.loads(path.read_text()) | change))
    source = source_for(settings)
    with pytest.raises(Unavailable) as error:
        admit(source, settings.model_root, "cpu")
    assert error.value.feature["reason"]["code"] == reason
    assert len(source.tensors()) == 20


def test_missing_tokenizer_and_incomplete_inventory_are_honest(
    tiny: tuple[Settings, Path, Any],
) -> None:
    from safetensors.torch import load_file, save_file

    settings, directory, _ = tiny
    tokenizer = directory / "tokenizer.json"
    saved = tokenizer.read_bytes()
    tokenizer.unlink()
    with pytest.raises(Unavailable) as error:
        admit(source_for(settings), settings.model_root, "cpu")
    assert error.value.feature["reason"]["code"] == "tokenizer_unavailable"
    tokenizer.write_bytes(saved)
    weights = load_file(directory / "model.safetensors")
    del weights["model.layers.1.mlp.gate_proj.weight"]
    save_file(weights, directory / "model.safetensors")
    with pytest.raises(Unavailable) as error:
        admit(source_for(settings), settings.model_root, "cpu")
    assert error.value.feature["reason"]["code"] == "unsupported_representation"


def test_cancellation_queue_and_active_forward_settle_before_reuse(
    tiny: tuple[Settings, Path, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings, _, reference = tiny
    admitted = admit(source_for(settings), settings.model_root, "cpu")
    entered, release = Event(), Event()
    loader = Mock(wraps=runtime.load_model)
    monkeypatch.setattr(runtime, "load_model", loader)

    async def scenario() -> None:
        work = BlockingWork()
        execution = NeuronRuntime(DeviceScheduler(work), "cpu")
        cancel = Cancellation()

        def slow(lease: NeuronLease) -> torch.Tensor:
            def hook(values: torch.Tensor) -> None:
                entered.set()
                assert release.wait(10)

            with lease.hook(admitted.modules[0]["module_id"], hook):
                return lease.forward(TOKENS)

        running = asyncio.create_task(execution.run(request(admitted), cancel, slow))
        try:
            assert await asyncio.to_thread(entered.wait, 10)
            queued_cancel = Cancellation()
            never = Mock(side_effect=AssertionError("cancelled work started"))
            queued = asyncio.create_task(execution.run(request(admitted), queued_cancel, never))
            queued_cancel.cancel()
            with pytest.raises(OperationCancelled):
                await queued
            never.assert_not_called()
            assert loader.call_count == 1
            cancel.cancel()
            clean = asyncio.create_task(
                execution.run(
                    request(admitted), Cancellation(), lambda lease: lease.forward(TOKENS)
                )
            )
            assert not running.done() and not clean.done()
            release.set()
            with pytest.raises(OperationCancelled):
                await running
            torch.testing.assert_close(await clean, direct(reference), rtol=1e-6, atol=1e-7)
        finally:
            release.set()
            await execution.aclose()
            await work.aclose()

    asyncio.run(scenario())


@pytest.mark.parametrize("failure", ["exception", "oom", "nonfinite"])
def test_failure_removes_hooks_and_keeps_runtime_recoverable(
    tiny: tuple[Settings, Path, Any],
    failure: str,
) -> None:
    settings, _, reference = tiny
    admitted = admit(source_for(settings), settings.model_root, "cpu")

    async def scenario() -> None:
        work = BlockingWork()
        execution = NeuronRuntime(DeviceScheduler(work), "cpu")
        try:

            def operation(lease: NeuronLease) -> torch.Tensor:
                def hook(values: torch.Tensor) -> torch.Tensor:
                    if failure == "oom":
                        raise torch.OutOfMemoryError("private prompt must not escape")
                    if failure == "exception":
                        raise ValueError("private prompt must not escape")
                    return torch.full_like(values, float("nan"))

                with lease.hook(admitted.modules[0]["module_id"], hook):
                    return lease.forward(TOKENS)

            with pytest.raises(ModelError) as error:
                await execution.run(request(admitted), Cancellation(), operation)
            assert error.value.code == (
                "resource_exhausted" if failure == "oom" else "internal_error"
            )
            assert "private prompt" not in str(error.value)
            if failure == "nonfinite":
                assert error.value.details == {"reason": "numerical_failure"}
            actual = await execution.run(
                request(admitted), Cancellation(), lambda lease: lease.forward(TOKENS)
            )
            torch.testing.assert_close(actual, direct(reference), rtol=1e-6, atol=1e-7)
        finally:
            await execution.aclose()
            await work.aclose()

    asyncio.run(scenario())


def test_failed_load_does_not_keep_model_alive_in_error_traceback(
    tiny: tuple[Settings, Path, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings, _, _ = tiny
    admitted = admit(source_for(settings), settings.model_root, "cpu")
    models_alive: list[weakref.ReferenceType[Any]] = []
    original = runtime.load_model

    def fail(model: AdmittedModel, cancellation: Cancellation) -> Any:
        loaded = original(model, cancellation)
        models_alive.append(weakref.ref(loaded))
        raise torch.OutOfMemoryError("Failure after allocation")

    monkeypatch.setattr(runtime, "load_model", fail)

    async def scenario() -> None:
        work = BlockingWork()
        execution = NeuronRuntime(DeviceScheduler(work), "cpu")
        try:
            with pytest.raises(ModelError) as error:
                await execution.run(request(admitted), Cancellation(), lambda _: None)
            assert error.value.code == "resource_exhausted"
            gc.collect()
            # Retaining a public error must not retain the failed model or permit
            # a second resident instance when the next operation loads a model.
            assert models_alive[0]() is None
        finally:
            await execution.aclose()
            await work.aclose()

    asyncio.run(scenario())


def test_production_geometry_matches_independently_reviewed_metadata() -> None:
    reference = json.loads(
        (Path(__file__).parent / "fixtures/dense-reference-metadata.json").read_text()
    )["smollm2"]
    source = Mock(spec=ModelSource, model_id="SmolLM2-135M")
    source.configuration.return_value = reference["config"]
    config = admission.checked_config(source)
    shapes = {name: tuple(shape) for name, (_, shape) in reference["global_storage"].items()}
    for layer in range(30):
        shapes.update(
            {
                f"model.layers.{layer}.{name}": tuple(shape)
                for name, (_, shape) in reference["layer_storage"].items()
            }
        )
    assert len(shapes) == 272
    assert admission.expected_weights(config) == shapes
    source.configuration.return_value = dict(reference["config"], hidden_size=8)
    with pytest.raises(Unavailable):
        admission.checked_config(source)


@pytest.mark.parametrize(
    "fields,code",
    [
        ({"token_ids": ()}, "validation_error"),
        ({"token_ids": (32,)}, "validation_error"),
        ({"token_ids": (True,)}, "validation_error"),
        ({"token_ids": (1,) * 33}, "unsupported_size"),
        ({"neuron_indices": (0, 0)}, "validation_error"),
        ({"neuron_indices": (16,)}, "validation_error"),
        ({"neuron_indices": tuple(range(9))}, "unsupported_size"),
        ({"module_id": "unknown"}, "validation_error"),
        ({"model_revision": "old"}, "model_content_changed"),
    ],
)
def test_preflight_rejects_before_loading_or_running(
    tiny: tuple[Settings, Path, Any],
    monkeypatch: pytest.MonkeyPatch,
    fields: dict[str, Any],
    code: str,
) -> None:
    settings, _, _ = tiny
    admitted = admit(source_for(settings), settings.model_root, "cpu")
    invalid = replace(request(admitted), **fields)
    loader = Mock(side_effect=AssertionError("preflight must precede loading"))
    monkeypatch.setattr(runtime, "load_model", loader)

    async def scenario() -> None:
        work = BlockingWork()
        execution = NeuronRuntime(DeviceScheduler(work), "cpu")
        operation = Mock(side_effect=AssertionError("invalid operation ran"))
        try:
            with pytest.raises(ModelError) as error:
                await execution.run(invalid, Cancellation(), operation)
            assert error.value.code == code
            loader.assert_not_called()
            operation.assert_not_called()
        finally:
            await execution.aclose()
            await work.aclose()

    asyncio.run(scenario())


def test_reuse_is_content_bound_and_weights_stay_unchanged(
    tiny: tuple[Settings, Path, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from safetensors.torch import load_file, save_file

    settings, directory, reference = tiny
    weights_path = directory / "model.safetensors"
    original_bytes = weights_path.read_bytes()
    loader = Mock(wraps=runtime.load_model)
    monkeypatch.setattr(runtime, "load_model", loader)

    async def scenario() -> None:
        work = BlockingWork()
        execution = NeuronRuntime(DeviceScheduler(work), "cpu")
        try:
            initial = admit(source_for(settings), settings.model_root, "cpu")
            for _ in range(2):
                # Independent pins with the same content share immutable loaded weights.
                model = admit(source_for(settings), settings.model_root, "cpu")
                actual = await execution.run(
                    request(model), Cancellation(), lambda lease: lease.forward(TOKENS)
                )
                torch.testing.assert_close(actual, direct(reference), rtol=1e-6, atol=1e-7)
            assert loader.call_count == 1
            assert weights_path.read_bytes() == original_bytes
            replacement = weights_path.with_suffix(".replacement")
            replacement.write_bytes(original_bytes)
            replacement.replace(weights_path)
            replaced = admit(source_for(settings), settings.model_root, "cpu")
            assert replaced.source.fingerprint == initial.source.fingerprint
            assert replaced.source.model_revision != initial.source.model_revision
            await execution.run(
                request(replaced), Cancellation(), lambda lease: lease.forward(TOKENS)
            )
            assert loader.call_count == 2
            state = load_file(weights_path)
            state["model.layers.0.mlp.down_proj.weight"].add_(0.1)
            save_file(state, weights_path)
            with torch.no_grad():
                reference.model.layers[0].mlp.down_proj.weight.add_(0.1)
            fresh = admit(source_for(settings), settings.model_root, "cpu")
            assert fresh.source.model_id == initial.source.model_id
            assert fresh.source.fingerprint != initial.source.fingerprint
            with pytest.raises(ModelError, match="Model content changed"):
                await execution.run(
                    request(initial), Cancellation(), lambda lease: lease.forward(TOKENS)
                )
            actual = await execution.run(
                request(fresh), Cancellation(), lambda lease: lease.forward(TOKENS)
            )
            torch.testing.assert_close(actual, direct(reference), rtol=1e-6, atol=1e-7)
            assert loader.call_count == 3
        finally:
            await execution.aclose()
            await work.aclose()

    asyncio.run(scenario())


@pytest.mark.parametrize("when", ["before_load", "after_load", "during_forward"])
def test_pinned_mutation_never_publishes_a_result(
    tiny: tuple[Settings, Path, Any],
    when: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings, directory, _ = tiny
    admitted = admit(source_for(settings), settings.model_root, "cpu")

    def mutate() -> None:
        path = directory / "config.json"
        path.write_text(path.read_text() + " ")

    original = runtime.load_model

    def load(*args: Any) -> Any:
        result = original(*args)
        if when == "after_load":
            mutate()
        return result

    loader = Mock(side_effect=load)
    monkeypatch.setattr(runtime, "load_model", loader)

    async def scenario() -> None:
        work = BlockingWork()
        execution = NeuronRuntime(DeviceScheduler(work), "cpu")
        try:
            if when == "before_load":
                mutate()

            def run(lease: NeuronLease) -> torch.Tensor:
                with lease.hook(admitted.modules[0]["module_id"], lambda _: mutate()):
                    return lease.forward(TOKENS)

            with pytest.raises(ModelError) as error:
                await execution.run(request(admitted), Cancellation(), run)
            assert error.value.code == "model_content_changed"
            assert loader.call_count == (0 if when == "before_load" else 1)
        finally:
            await execution.aclose()
            await work.aclose()

    asyncio.run(scenario())


@pytest.mark.parametrize("stop", ["session", "disconnect", "shutdown"])
def test_consumer_lifecycle_drains_worker_and_other_session_stays_clean(
    tiny: tuple[Settings, Path, Any],
    stop: str,
) -> None:
    settings, _, reference = tiny
    entered, release = Event(), Event()

    async def scenario() -> None:
        async with open_services(settings) as services:
            original_cache = {
                p: p.read_bytes() for p in settings.cache_dir.rglob("*") if p.is_file()
            }
            sessions, execution = services.sessions, services.neurons
            assert sessions is not None and execution is not None
            first = await sessions.create(admission.REPOSITORY + "@" + admission.REVISION)
            second = await sessions.create(first.source.model_id)
            admitted = admit(first.source, settings.model_root, "cpu")

            def blocked(lease: NeuronLease) -> torch.Tensor:
                def hook(values: torch.Tensor) -> torch.Tensor:
                    entered.set()
                    assert release.wait(10)
                    return torch.zeros_like(values)

                with lease.hook(admitted.modules[0]["module_id"], hook):
                    return lease.forward(TOKENS)

            async def prepare(cancellation: Cancellation) -> Any:
                await execution.run(request(admitted), cancellation, blocked)
                raise AssertionError("cancelled result must not publish")

            spec = ArtifactSpec(
                model_fingerprint=first.source.fingerprint,
                source="neurons",
                operation="test",
                parameters={},
                dtype="float32",
                layout="c",
                shape=(1,),
                expected_bytes=4,
                producer="test",
            )
            consumer = sessions.operations.subscribe_source(
                spec, prepare=prepare, session_id=first.id
            )
            reading = asyncio.create_task(consumer.read())
            try:
                assert await asyncio.to_thread(entered.wait, 10)
                if stop == "session":
                    stopping = asyncio.create_task(sessions.delete(first.id))
                elif stop == "shutdown":
                    stopping = asyncio.create_task(execution.aclose())
                else:
                    reading.cancel()
                    stopping = asyncio.create_task(consumer.aclose())
                # A cancelled consumer still owns the blocked hook until it settles.
                await asyncio.sleep(0)
                assert not stopping.done()
                release.set()
                await stopping
                with pytest.raises((OperationCancelled, asyncio.CancelledError)):
                    await reading
                if stop != "shutdown":
                    other = admit(second.source, settings.model_root, "cpu")
                    clean = await execution.run(
                        request(other), Cancellation(), lambda lease: lease.forward(TOKENS)
                    )
                    torch.testing.assert_close(clean, direct(reference), rtol=1e-6, atol=1e-7)
                assert {
                    p: p.read_bytes() for p in settings.cache_dir.rglob("*") if p.is_file()
                } == original_cache
            finally:
                release.set()
                await consumer.aclose()

    asyncio.run(scenario())

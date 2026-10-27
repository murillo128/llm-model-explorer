"""Numerical comparison oracles; shared lifetime/protocol matrices stay with their owners."""

import asyncio
import gc
import json
import math
import struct
import weakref
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import replace
from functools import partial
from pathlib import Path
from threading import Event
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock
from uuid import UUID

import pytest
import torch
from fastapi.testclient import TestClient
from neuron_fixtures import KnownModel, direct, known_activations, source_for
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
from llm_model_explorer.neuron_execution.activations import ActivationReader
from llm_model_explorer.neuron_execution.comparison import compare_neurons
from llm_model_explorer.neuron_execution.comparisons import Comparisons, subscribe_comparisons
from llm_model_explorer.operations import Cancellation, DeviceScheduler, OperationCancelled
from llm_model_explorer.services import open_services
from llm_model_explorer.settings import Settings
from llm_model_explorer.stream_metadata import NeuronComparisonOptions
from llm_model_explorer.streaming import LMEXResponse
from llm_model_explorer.tokenization import load_tokenizer

BASELINE = {"id": "baseline", "mode": "baseline"}
TOKENS = (1, 0, 2, 1)  # Prompt EOS and duplicates are passed verbatim.


def selection(settings: Settings, tokens: tuple[int, ...] = TOKENS) -> NeuronInput:
    model = admit(source_for(settings), settings.model_root, "cpu")
    return NeuronInput(
        model, model.source.model_revision, model.modules[1]["module_id"], (7,), tokens
    )


def options(*conditions: dict[str, Any], maximum: int = 0, k: int = 2) -> NeuronComparisonOptions:
    return NeuronComparisonOptions.model_validate(
        dict(conditions=[BASELINE, *conditions], top_k=k, max_new_tokens=maximum, decoding="greedy")
    )


class ComparisonModel(KnownModel):
    """Reuse the explicit MLP, extending its logits to the fixture vocabulary."""

    def forward(self, input_ids: torch.Tensor, **kwargs: Any) -> Any:
        result = super().forward(input_ids, **kwargs)
        return SimpleNamespace(logits=torch.nn.functional.pad(result.logits, (0, 24)))


def scalar_distribution(activation: float) -> list[float]:
    # For final prompt input (1,-2), down_proj selects a[1] and a[7].
    # All other vocabulary logits are zero; retain all 32 denominator terms.
    logits = [0.0] * 32
    logits[1] = -2 / (1 + math.exp(-1))
    logits[7] = activation
    denominator = sum(math.exp(value) for value in logits)
    return [math.exp(value) / denominator for value in logits]


@pytest.mark.parametrize(
    "condition,expected",
    [
        ({"mode": "zero"}, 0.0),
        ({"mode": "scale", "factor": 0}, 0.0),
        ({"mode": "scale", "factor": 1e-50}, 0.0),
        ({"mode": "scale", "factor": 1}, 3 / (1 + math.exp(3))),
        ({"mode": "add", "delta": 0}, 3 / (1 + math.exp(3))),
        ({"mode": "add", "delta": 2}, 2 + 3 / (1 + math.exp(3))),
        ({"mode": "scale", "factor": -2}, -6 / (1 + math.exp(3))),
    ],
)
def test_operators_projection_and_full_vocabulary_union(
    tiny: tuple[Settings, Path, Any],
    monkeypatch: pytest.MonkeyPatch,
    condition: dict[str, Any],
    expected: float,
) -> None:
    settings, _, _ = tiny
    request = selection(settings)
    model = ComparisonModel()
    weights = {name: value.clone() for name, value in model.named_parameters()}
    monkeypatch.setattr(runtime, "load_model", lambda *_: model)
    observed = []
    down = model.layers[1].down_proj
    original = down.forward

    def projection(values: torch.Tensor) -> torch.Tensor:
        observed.append(values.clone())
        return original(values)

    monkeypatch.setattr(down, "forward", projection)
    opts = options(dict(id="changed", scope="all", **condition))

    async def scenario() -> None:
        work = BlockingWork()
        execution = NeuronRuntime(DeviceScheduler(work), "cpu")
        try:
            result = await execution.run(
                request, Cancellation(), lambda lease: compare_neurons(lease, request, opts, None)
            )
            distributions = [
                scalar_distribution(3 / (1 + math.exp(3))),
                scalar_distribution(expected),
            ]
            tops = [sorted(range(32), key=lambda i: (-row[i], i))[:2] for row in distributions]
            candidates = sorted({i for row in tops for i in row})
            assert result.top_token_ids == tops
            assert result.candidate_token_ids == candidates
            torch.testing.assert_close(
                result.probabilities,
                torch.tensor([[row[i] for i in candidates] for row in distributions]),
                rtol=1e-6,
                atol=1e-7,
            )
            assert (result.probabilities.sum(dim=1) < 1).all()
            if condition["mode"] == "zero" or condition.get("factor") == 0:
                assert tops == [[7, 0], [0, 2]] and candidates == [0, 2, 7]
                # Candidate 7 is outside the changed top-k but has its true
                # full-vocabulary probability, not zero or union-normalized mass.
                assert 0 < result.probabilities[1, 2] < 0.04
            baseline = known_activations(TOKENS, tuple(range(16))).unsqueeze(0)
            changed = baseline.clone()
            if condition["mode"] == "zero":
                changed[:, :, 7] = 0
            elif condition["mode"] == "scale":
                changed[:, :, 7] *= condition["factor"]
            else:
                changed[:, :, 7] += condition["delta"]
            torch.testing.assert_close(observed[0], baseline, rtol=1e-6, atol=1e-7)
            torch.testing.assert_close(observed[1], changed, rtol=1e-6, atol=1e-7)
            assert result.continuations == [
                dict(condition_id=i, token_ids=[], text="", finish_reason="not_requested")
                for i in ("baseline", "changed")
            ]
            assert not down._forward_pre_hooks
            for name, value in model.named_parameters():
                torch.testing.assert_close(value, weights[name], rtol=0, atol=0)
        finally:
            await execution.aclose()
            await work.aclose()

    asyncio.run(scenario())


@contextmanager
def reference_intervention(
    reference: Any, request: NeuronInput, condition: dict[str, Any]
) -> Iterator[None]:
    """Replace the entire MLP, independent of the production down-projection hook."""
    mlp = reference.model.layers[1].mlp
    original = mlp.forward

    def forward(x: torch.Tensor) -> torch.Tensor:
        gate = torch.nn.functional.linear(x, mlp.gate_proj.weight)
        up = torch.nn.functional.linear(x, mlp.up_proj.weight)
        values = gate * torch.sigmoid(gate) * up
        for position in range(values.shape[1]):
            applies = condition["scope"] == "all" or (
                (position < len(request.token_ids)) == (condition["scope"] == "prompt")
            )
            if applies:
                for neuron in request.neuron_indices:
                    if condition["mode"] == "add":
                        values[0, position, neuron] += condition["delta"]
                    elif condition["mode"] == "scale":
                        values[0, position, neuron] *= condition["factor"]
                    else:
                        values[0, position, neuron] = 0
        return torch.nn.functional.linear(values, mlp.down_proj.weight)

    mlp.forward = forward
    try:
        yield
    finally:
        mlp.forward = original


def test_greedy_uses_logits_when_wire_probabilities_round_to_a_tie(
    tiny: tuple[Settings, Path, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings, directory, _ = tiny
    request = selection(settings)
    model = ComparisonModel()
    monkeypatch.setattr(runtime, "load_model", lambda *_: model)
    tokenizer = load_tokenizer(settings.model_root, directory)
    opts = options(dict(id="small", mode="scale", scope="all", factor=1e-9), maximum=1)

    async def scenario() -> None:
        work = BlockingWork()
        execution = NeuronRuntime(DeviceScheduler(work), "cpu")
        try:
            result = await execution.run(
                request,
                Cancellation(),
                lambda lease: compare_neurons(lease, request, opts, tokenizer),
            )
            # logit[7] = 3e-9 / (1 + exp(3)) > 0; other maximum logits are 0.
            # Their probabilities round to equal F32, so table ordering and
            # greedy decoding legitimately have different first candidates.
            assert result.candidate_token_ids == [0, 2, 7]
            assert result.top_token_ids == [[7, 0], [0, 2]]
            assert result.probabilities[1, 0] == result.probabilities[1, 2]
            assert result.continuations[1] == dict(
                condition_id="small", token_ids=[7], text="token-7", finish_reason="length"
            )
        finally:
            await execution.aclose()
            await work.aclose()

    asyncio.run(scenario())


def test_transformers_greedy_scopes_fresh_baseline_and_exact_context(
    tiny: tuple[Settings, Path, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings, directory, reference = tiny
    request = replace(selection(settings), neuron_indices=(7, 1))
    tokenizer = load_tokenizer(settings.model_root, directory)
    conditions = [
        dict(id=scope, mode="add", delta=-3, scope=scope)
        for scope in ("prompt", "continuation", "all")
    ]
    opts = options(*conditions, maximum=4, k=3)
    reference_rows = []
    reference_ids = []
    reference_prefixes = []

    def expected_branch() -> None:
        prefix, generated, prefixes = request.token_ids, [], []
        for step in range(4):
            prefixes.append(prefix)
            logits = direct(reference, prefix)[0, -1]
            if step == 0:
                reference_rows.append(torch.softmax(logits.double(), dim=-1).float())
            token = int(logits.argmax())
            generated.append(token)
            if token == 0:
                break
            prefix = (*prefix, token)
        reference_ids.append(generated)
        reference_prefixes.extend(prefixes)

    expected_branch()
    for condition in conditions:
        with reference_intervention(reference, request, condition):
            expected_branch()
    # This fixture must distinguish prompt and continuation behavior; an inert
    # hook or all-prompt-as-continuation defect cannot pass by coincidence.
    assert not torch.allclose(reference_rows[0], reference_rows[1], rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(reference_rows[0], reference_rows[2], rtol=0, atol=0)
    assert reference_ids[0][0] == reference_ids[2][0]
    assert reference_ids[0] != reference_ids[2]
    prefixes = []
    forward = NeuronLease.forward

    def observe(lease: NeuronLease, ids: tuple[int, ...]) -> torch.Tensor:
        prefixes.append(ids)
        return forward(lease, ids)

    monkeypatch.setattr(NeuronLease, "forward", observe)

    async def scenario() -> None:
        work = BlockingWork()
        execution = NeuronRuntime(DeviceScheduler(work), "cpu")
        try:
            result = await execution.run(
                request,
                Cancellation(),
                lambda lease: compare_neurons(lease, request, opts, tokenizer),
            )
            assert prefixes == reference_prefixes
            assert [c["token_ids"] for c in result.continuations] == reference_ids
            assert all(c["finish_reason"] == "length" for c in result.continuations)
            for continuation, ids in zip(result.continuations, reference_ids, strict=True):
                assert continuation["text"] == tokenizer.decode(
                    ids, skip_special_tokens=False, clean_up_tokenization_spaces=False
                )
            candidates = result.candidate_token_ids
            torch.testing.assert_close(
                result.probabilities,
                torch.stack(reference_rows)[:, candidates],
                rtol=1e-6,
                atol=1e-7,
            )
            clean = await execution.run(
                request,
                Cancellation(),
                lambda lease: compare_neurons(lease, request, options(maximum=4, k=3), tokenizer),
            )
            assert clean.continuations[0] == result.continuations[0]
            torch.testing.assert_close(
                clean.probabilities[0],
                reference_rows[0][clean.candidate_token_ids],
                rtol=1e-6,
                atol=1e-7,
            )
        finally:
            await execution.aclose()
            await work.aclose()

    asyncio.run(scenario())


def test_real_transformers_eos_and_context_boundary(
    tiny: tuple[Settings, Path, Any],
) -> None:
    settings, directory, reference = tiny
    # Zero native weights make all logits equal, independently proving greedy
    # lowest-ID ties and EOS inclusion. Keep the real packaged forward class.
    with torch.no_grad():
        for parameter in reference.parameters():
            parameter.zero_()
    reference.save_pretrained(directory, safe_serialization=True)
    request = selection(settings, tokens=(0,) * 31)
    tokenizer = load_tokenizer(settings.model_root, directory)

    async def scenario() -> None:
        work = BlockingWork()
        execution = NeuronRuntime(DeviceScheduler(work), "cpu")
        try:
            result = await execution.run(
                request,
                Cancellation(),
                lambda lease: compare_neurons(lease, request, options(maximum=1), tokenizer),
            )
            assert result.top_token_ids == [[0, 1]]
            assert result.continuations == [
                dict(condition_id="baseline", token_ids=[0], text="token-0", finish_reason="eos")
            ]
            torch.testing.assert_close(
                result.probabilities, torch.full((1, 2), 1 / 32), rtol=0, atol=0
            )
        finally:
            await execution.aclose()
            await work.aclose()

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "fields,code",
    [
        ({"conditions": []}, "validation_error"),
        ({"conditions": [dict(id="x", mode="zero", scope="all"), BASELINE]}, "validation_error"),
        ({"conditions": [BASELINE, BASELINE]}, "validation_error"),
        (
            {
                "conditions": [
                    BASELINE,
                    dict(id="x", mode="zero", scope="all"),
                    dict(id="x", mode="zero", scope="all"),
                ]
            },
            "validation_error",
        ),
        ({"conditions": [dict(BASELINE, scope="all")]}, "validation_error"),
        (
            {"conditions": [BASELINE, dict(id="x", mode="scale", scope="all", factor=2, delta=1)]},
            "validation_error",
        ),
        (
            {"conditions": [BASELINE, dict(id="x", mode="scale", scope="all", factor=1e39)]},
            "validation_error",
        ),
        (
            {"conditions": [BASELINE, dict(id="x", mode="add", scope="all", delta=float("nan"))]},
            "validation_error",
        ),
        (
            {
                "conditions": [
                    BASELINE,
                    dict(id="x", mode="scale", scope="all", factor=float("inf")),
                ]
            },
            "validation_error",
        ),
        (
            {
                "conditions": [
                    BASELINE,
                    *[dict(id=f"x{i}", mode="zero", scope="all") for i in range(4)],
                ]
            },
            "unsupported_size",
        ),
        ({"top_k": 0}, "validation_error"),
        ({"top_k": 21}, "unsupported_size"),
        ({"max_new_tokens": -1}, "validation_error"),
        ({"max_new_tokens": True}, "validation_error"),
        ({"max_new_tokens": 29}, "unsupported_size"),
        ({"max_new_tokens": 32}, "unsupported_size"),
        ({"decoding": "sample"}, "validation_error"),
        ({"neuron_indices": [7, 7]}, "validation_error"),
        ({"neuron_indices": [16]}, "validation_error"),
        ({"token_ids": [32]}, "validation_error"),
    ],
)
def test_comparison_preflight_before_model_loading(
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
        prefix = f"/sessions/{session['id']}"
        module = client.get(prefix + "/neuron-capabilities").json()["modules"][1]
        body = (
            dict(
                model_revision=session["model_revision"],
                module_id=module["module_id"],
                neuron_indices=[7],
                token_ids=list(TOKENS),
                **options().model_dump(),
            )
            | fields
        )
        response = client.post(
            prefix + "/neuron-comparisons",
            content=json.dumps(body),
            headers={"Content-Type": "application/json"},
        )
        assert response.status_code == 422
        assert response.json()["code"] == code
        assert response.headers["cache-control"] == "no-store"
        assert "x-operation-id" not in response.headers
    fail.assert_not_called()


@pytest.mark.parametrize("stop", ["failure", "cancel", "numerical"])
def test_incomplete_conditions_release_hooks_and_buffers(
    tiny: tuple[Settings, Path, Any],
    monkeypatch: pytest.MonkeyPatch,
    stop: str,
) -> None:
    settings, _, _ = tiny
    request = replace(selection(settings), neuron_indices=(1,))
    model = ComparisonModel()
    monkeypatch.setattr(runtime, "load_model", lambda *_: model)
    condition = dict(
        id="changed",
        mode="scale",
        scope="all",
        factor=3.4028234663852886e38 if stop == "numerical" else 2,
    )
    opts = options(condition)
    down = model.layers[1].down_proj
    original = down.forward
    entered, release = Event(), Event()
    values: list[weakref.ReferenceType[torch.Tensor]] = []

    def interrupted(activation: torch.Tensor) -> torch.Tensor:
        values.append(weakref.ref(activation))
        if len(values) == 2:
            entered.set()
            assert release.wait(10)
            if stop == "failure":
                raise RuntimeError("private neuron output diagnostic")
        return original(activation)

    monkeypatch.setattr(down, "forward", interrupted)

    async def scenario() -> None:
        work = BlockingWork()
        execution = NeuronRuntime(DeviceScheduler(work), "cpu")
        cancellation = Cancellation()
        task = asyncio.create_task(
            execution.run(
                request, cancellation, lambda lease: compare_neurons(lease, request, opts, None)
            )
        )
        try:
            if stop != "numerical":
                assert await asyncio.to_thread(entered.wait, 10)
                if stop == "cancel":
                    cancellation.cancel()
                assert not task.done()
                release.set()
            with pytest.raises(OperationCancelled if stop == "cancel" else ModelError) as error:
                await task
            assert "private" not in str(error.value)
            if stop == "numerical":
                assert isinstance(error.value, ModelError)
                assert error.value.details == {"reason": "numerical_failure"}
            assert not down._forward_pre_hooks
            gc.collect()  # Retain the task/error while checking freed activation storage.
            assert all(value() is None for value in values)
            monkeypatch.setattr(down, "forward", original)
            clean = await execution.run(
                request,
                Cancellation(),
                lambda lease: compare_neurons(lease, request, options(), None),
            )
            expected = scalar_distribution(3 / (1 + math.exp(3)))
            torch.testing.assert_close(
                clean.probabilities,
                torch.tensor([[expected[i] for i in clean.candidate_token_ids]]),
                rtol=1e-6,
                atol=1e-7,
            )
        finally:
            release.set()
            await execution.aclose()
            await work.aclose()

    asyncio.run(scenario())


@pytest.mark.parametrize("stop", ["success", "failure", "cancel"])
def test_tcp_headers_precede_computation_and_complete_comparison_layout(
    tiny: tuple[Settings, Path, Any],
    monkeypatch: pytest.MonkeyPatch,
    stop: str,
) -> None:
    settings, _, _ = tiny
    model = ComparisonModel()
    monkeypatch.setattr(runtime, "load_model", lambda *_: model)
    monkeypatch.setattr(neuron_routes, "LMEXResponse", partial(LMEXResponse, max_data_bytes=8))
    down = model.layers[1].down_proj
    original = down.forward
    calls = 0
    entered, release = Event(), Event()

    def paused(values: torch.Tensor) -> torch.Tensor:
        nonlocal calls
        calls += 1
        if calls == 2:  # One complete condition must never escape as a complete comparison.
            entered.set()
            assert release.wait(10)
            if stop == "failure":
                raise RuntimeError("private comparison diagnostic")
        return original(values)

    monkeypatch.setattr(down, "forward", paused)
    app = create_app(settings)

    async def scenario() -> None:
        async with app.router.lifespan_context(app):
            before = {p: p.read_bytes() for p in settings.cache_dir.rglob("*") if p.is_file()}
            async with server(app) as client:
                model_id = (await client.get("/models")).json()["models"][0]["id"]
                session = (await client.post("/sessions", json={"model_id": model_id})).json()
                prefix = f"/sessions/{session['id']}"
                caps = (await client.get(prefix + "/neuron-capabilities")).json()
                module = caps["modules"][1]
                opts = options(dict(id="zero", mode="zero", scope="prompt"))
                body = dict(
                    model_revision=session["model_revision"],
                    module_id=module["module_id"],
                    neuron_indices=[7],
                    token_ids=list(TOKENS),
                    **opts.model_dump(),
                )
                try:
                    async with client.stream(
                        "POST", prefix + "/neuron-comparisons", json=body
                    ) as response:
                        assert response.status_code == 200
                        assert response.headers["cache-control"] == "no-store"
                        operation = response.headers["x-operation-id"]
                        assert await asyncio.to_thread(entered.wait, 10)
                        if stop == "cancel":
                            consumer = app.state.services.operation_delivery._operations[
                                UUID(operation)
                            ]
                            cancelled = asyncio.create_task(
                                client.delete(f"/operations/{operation}")
                            )
                            await asyncio.wait_for(consumer.cancellation._wake.wait(), 10)
                        release.set()
                        if stop == "cancel":
                            assert (await cancelled).status_code == 204
                        frames = Frames(response)
                        kind, payload = await frames.next()
                        if stop == "success":
                            assert kind == 1
                            metadata = json.loads(payload)
                            assert metadata == dict(
                                kind="neuron_comparison",
                                session_id=session["id"],
                                model_id=model_id,
                                model_revision=session["model_revision"],
                                execution=caps["execution"],
                                limits=caps["limits"],
                                module=module,
                                token_ids=list(TOKENS),
                                neuron_indices=[7],
                                dtype="float32",
                                byte_order="little",
                                layout="c",
                                **opts.model_dump(),
                                candidate_token_ids=[0, 2, 7],
                                top_token_ids=[[7, 0], [0, 2]],
                                continuations=[
                                    dict(
                                        condition_id=i,
                                        token_ids=[],
                                        text="",
                                        finish_reason="not_requested",
                                    )
                                    for i in ("baseline", "zero")
                                ],
                                sections=[
                                    dict(
                                        name="probabilities", shape=[2, 3], offset=0, byte_length=24
                                    )
                                ],
                                byte_length=24,
                            )
                            raw = bytearray()
                            for _ in range(3):
                                kind, payload = await frames.next()
                                assert kind == 2 and len(payload) == 8
                                raw.extend(payload)
                            expected = [scalar_distribution(a) for a in (3 / (1 + math.exp(3)), 0)]
                            torch.testing.assert_close(
                                torch.tensor(struct.unpack("<6f", raw)).reshape(2, 3),
                                torch.tensor([[row[i] for i in (0, 2, 7)] for row in expected]),
                                rtol=1e-6,
                                atol=1e-7,
                            )
                            assert await frames.next() == (4, b"")
                        elif stop == "failure":
                            assert kind == 5
                            assert json.loads(payload) == dict(
                                code="internal_error", message="Neuron execution failed."
                            )
                        else:
                            assert (kind, payload) == (6, b"")
                        await frames.end()
                finally:
                    release.set()
                assert not down._forward_pre_hooks
            assert {
                p: p.read_bytes() for p in settings.cache_dir.rglob("*") if p.is_file()
            } == before

    asyncio.run(scenario())


@pytest.mark.parametrize("when", ["after_compute", "eof"])
def test_comparison_reader_releases_output_on_content_change(
    tiny: tuple[Settings, Path, Any],
    monkeypatch: pytest.MonkeyPatch,
    when: str,
) -> None:
    settings, directory, _ = tiny
    readers: list[weakref.ReferenceType[ActivationReader]] = []
    compute = Comparisons.compute
    read_init = ActivationReader.__init__

    def track(reader: ActivationReader, *args: Any) -> None:
        read_init(reader, *args)
        readers.append(weakref.ref(reader))

    def mutate() -> None:
        path = directory / "config.json"
        path.write_text(path.read_text() + " ")

    def after_compute(result: Comparisons, lease: NeuronLease) -> Any:
        reader = compute(result, lease)
        if when == "after_compute":
            mutate()
        return reader

    monkeypatch.setattr(ActivationReader, "__init__", track)
    monkeypatch.setattr(Comparisons, "compute", after_compute)

    async def scenario() -> None:
        async with open_services(settings) as services:
            sessions, execution = services.sessions, services.neurons
            assert sessions is not None and execution is not None
            request = selection(settings)
            session = await sessions.create(request.model.source.model_id)
            result, consumer = await subscribe_comparisons(
                sessions,
                execution,
                settings.model_root,
                session.id,
                session.source.model_revision,
                request.module_id,
                request.neuron_indices,
                request.token_ids,
                options(dict(id="changed", mode="add", scope="all", delta=-3), maximum=4),
            )
            try:
                if when == "eof":
                    metadata = await result.metadata(consumer)
                    assert isinstance(metadata, dict)
                    assert len(await consumer.read()) == metadata["byte_length"]
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

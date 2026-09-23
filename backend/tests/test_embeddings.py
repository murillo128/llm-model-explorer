"""Input rows: independent contract bytes, source bounds and operation ownership."""

import asyncio
import json
import struct
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from threading import Event
from typing import Any, BinaryIO, cast
from unittest.mock import Mock
from uuid import UUID, uuid4

import pytest
import torch
from cache_helpers import numeric_cache_entries
from fastapi.testclient import TestClient
from starlette.types import Message
from test_models import make_model, mutate_last_byte, write_weights
from test_operations import run
from test_quantized_models import config_for, packed_group, write_storage
from test_streaming import Frames, forgotten, server
from test_tensor_data import frames, payload
from test_tokenization import make_tokenizer

from llm_model_explorer.app import create_app
from llm_model_explorer.embeddings import (
    resolve_embeddings,
    resolve_input_table,
    subscribe_embeddings,
)
from llm_model_explorer.materialization import CHUNK_ELEMENTS, _TensorReader
from llm_model_explorer.model_files import FileSnapshot, ModelError
from llm_model_explorer.models import ModelCatalogue
from llm_model_explorer.services import Services
from llm_model_explorer.settings import Settings
from llm_model_explorer.streaming import LMEXResponse, _SendFailed
from llm_model_explorer.tensor_source import DTYPES, ModelSource

CONTRACT: dict[str, Any] = json.loads(
    (Path(__file__).resolve().parents[2] / "api/fixtures/embeddings.json").read_text()
)
ROWS: list[list[float]] = CONTRACT["source"]["input_table"]
KEY = "model.embed_tokens.weight"
FAMILIES = {
    "llama": ("LlamaForCausalLM", KEY),
    "qwen3": ("Qwen3ForCausalLM", KEY),
    "qwen3_5": ("Qwen3_5ForConditionalGeneration", "model.language_model.embed_tokens.weight"),
}


@pytest.fixture(params=FAMILIES)
def family(request: pytest.FixtureRequest) -> str:
    return str(request.param)


def embedding_model(
    root: Path, dtype: str = "F32", *, vocab: int = 4, hidden: int = 3, family: str = "llama"
) -> Path:
    directory = make_model(root)
    config = json.loads((directory / "config.json").read_text())
    architecture, key = FAMILIES[family]
    config.update(model_type=family, architectures=[architecture])
    if family == "qwen3_5":
        # Conflicting top-level and visual dimensions must never control text rows.
        config.update(vocab_size=99, hidden_size=88, vision_config={"hidden_size": 77})
        config["text_config"] = {"vocab_size": vocab, "hidden_size": hidden}
    else:
        config.update(vocab_size=vocab, hidden_size=hidden)
    (directory / "config.json").write_text(json.dumps(config))
    values = (
        [v for row in ROWS for v in row] if (vocab, hidden) == (4, 3) else [1.5] * (vocab * hidden)
    )
    write_weights(
        directory / "model.safetensors",
        [(key, dtype, [vocab, hidden], values), ("lm_head.weight", dtype, [1], [-99])],
    )
    return directory


def address(client: TestClient, model_id: str = "test/tiny") -> str:
    response = client.post("/sessions", json={"model_id": model_id})
    assert response.status_code == 201
    return f"/sessions/{response.json()['id']}/embeddings"


@pytest.mark.parametrize("dtype", ["F32", "F16", "BF16"])
@pytest.mark.parametrize("case", CONTRACT["request_cases"], ids=lambda c: c["name"])
def test_contract_requests(
    settings: Settings, dtype: str, case: dict[str, Any], family: str
) -> None:
    directory = embedding_model(settings.model_root, dtype, family=family)
    before = (directory / "model.safetensors").read_bytes()
    with TestClient(create_app(settings)) as client:
        response = client.post(address(client), json=case["request"])
        assert response.headers["cache-control"] == "no-store"
        if not case["valid"]:
            assert response.status_code == 422
            assert response.json()["code"] == "validation_error"
            assert "x-operation-id" not in response.headers
        else:
            assert response.status_code == 200
            UUID(response.headers["x-operation-id"])
            result = frames(response.content)
            ids = case["request"]["token_ids"]
            values = [v for token in ids for v in ROWS[token]]
            assert json.loads(result[0][1]) == dict(
                kind="input_embeddings",
                token_ids=ids,
                shape=[len(ids), 3],
                dtype="float32",
                byte_order="little",
                layout="c",
                byte_length=4 * len(values),
            )
            assert b"".join(data for kind, data in result if kind == 2) == struct.pack(
                f"<{len(values)}f", *values
            )
            assert result[-1] == (4, b"")
            assert KEY.encode() not in response.content
    assert (directory / "model.safetensors").read_bytes() == before
    assert not numeric_cache_entries(settings.cache_dir)


@pytest.mark.parametrize(
    "updates,key,shape",
    [
        ({"model_type": "unknown"}, KEY, [4, 3]),
        ({"architectures": []}, KEY, [4, 3]),
        ({"architectures": ["LlamaForCausalLM", "Other"]}, KEY, [4, 3]),
        ({"auto_map": {"AutoModel": "private.Custom"}}, KEY, [4, 3]),
        ({"hidden_size": None}, KEY, [4, 3]),
        ({"hidden_size": True}, KEY, [4, 3]),
        ({"vocab_size": 5}, KEY, [4, 3]),
        ({}, "lm_head.weight", [4, 3]),
        ({}, "prefix.model.embed_tokens.weight", [4, 3]),
        ({}, KEY, [12]),
        ({"hidden_size": 0}, KEY, [4, 0]),
    ],
)
def test_unsupported_resolution(
    settings: Settings, updates: dict[str, object], key: str, shape: list[int]
) -> None:
    directory = embedding_model(settings.model_root)
    path = directory / "config.json"
    path.write_text(json.dumps(json.loads(path.read_text()) | updates))
    write_weights(
        directory / "model.safetensors", [(key, "F32", shape, [] if 0 in shape else [1] * 12)]
    )
    with TestClient(create_app(settings)) as client:
        url = address(client)
        for ids in ([], [0]):
            response = client.post(url, json={"token_ids": ids})
            assert response.status_code == 422
            assert response.json()["code"] == "unsupported_representation"
            assert "x-operation-id" not in response.headers
            assert str(directory) not in response.text and key not in response.text


def test_preflight_and_control_limits(settings: Settings, monkeypatch: pytest.MonkeyPatch) -> None:
    directory = embedding_model(settings.model_root)
    with TestClient(create_app(settings)) as client:
        url = address(client)
        for body in ({}, {"token_ids": [0], "extra": 1}, {"token_ids": None}):
            response = client.post(url, json=body)
            assert response.status_code == 422 and response.json()["code"] == "validation_error"
        response = client.post(url, content="{", headers={"Content-Type": "application/json"})
        assert response.status_code == 400 and response.json()["code"] == "malformed_json"
        for session, status in ((str(uuid4()), 404), ("bad", 422)):
            response = client.post(f"/sessions/{session}/embeddings", json={"token_ids": []})
            assert response.status_code == status
        original = ModelSource.iter_rows
        reads: list[bool] = []

        def observed(self: ModelSource, *args: Any, **kwargs: Any) -> Any:
            reads.append(True)
            return original(self, *args, **kwargs)

        monkeypatch.setattr(ModelSource, "iter_rows", observed)
        # Above artifact-manifest limits is legal: only the 1 MiB META limit applies.
        response = client.post(url, json={"token_ids": [0] * 20_000})
        assert frames(response.content)[-1] == (4, b"")
        reads.clear()
        for ids, code in (([0, 4], "validation_error"), ([0] * 524_288, "unsupported_size")):
            response = client.post(url, json={"token_ids": ids})
            assert response.status_code == 422 and response.json()["code"] == code
            assert "x-operation-id" not in response.headers
        assert not reads
        mutate_last_byte(directory / "model.safetensors")
        response = client.post(url, json={"token_ids": []})
        assert response.status_code == 409 and response.json()["code"] == "model_content_changed"


@pytest.mark.parametrize("dtype", ["F32", "F16", "BF16"])
@pytest.mark.parametrize("hidden", [3, CHUNK_ELEMENTS + 1])
def test_bounded_physical_reads_and_allocations(
    settings: Settings, monkeypatch: pytest.MonkeyPatch, dtype: str, hidden: int, family: str
) -> None:
    embedding_model(settings.model_root, dtype, vocab=20, hidden=hidden, family=family)
    source = ModelCatalogue(settings.model_root).pin("test/tiny")
    # Pinning hashes complete assets in bounded blocks by the existing snapshot contract.
    # Instrument after pinning to distinguish it from row lookup/materialization.
    original_open = FileSnapshot.open
    original_frombuffer = torch.frombuffer
    reads: list[int] = []
    allocations: list[int] = []

    @contextmanager
    def observed_open(self: FileSnapshot, name: str) -> Iterator[BinaryIO]:
        with original_open(self, name) as stream:
            if name.endswith(".safetensors"):
                proxy = Mock(wraps=stream)

                def read(count: int = -1) -> bytes:
                    reads.append(count)
                    assert 0 < count <= CHUNK_ELEMENTS * DTYPES[dtype][1]
                    return stream.read(count)

                proxy.read = read
                yield cast(BinaryIO, proxy)
            else:
                yield stream

    def observed_buffer(*args: Any, **kwargs: Any) -> torch.Tensor:
        result = original_frombuffer(*args, **kwargs)
        allocations.append(result.numel())
        assert 0 < result.numel() <= CHUNK_ELEMENTS
        return result

    monkeypatch.setattr(FileSnapshot, "open", observed_open)
    monkeypatch.setattr(torch, "frombuffer", observed_buffer)
    monkeypatch.setattr(ModelSource, "iter_tensor", Mock(side_effect=AssertionError("full table")))
    result = resolve_embeddings(source, (19, 0, 1, 1, 1))
    assert not reads and not allocations  # Resolution only inspects metadata.
    reader = result.reader()
    data = bytearray()
    try:
        while block := reader.read_available(256 * 1024):
            data.extend(block)
    finally:
        reader.close()
    assert data == struct.pack("<f", 1.5) * hidden * 5
    assert sum(reads) <= 5 * hidden * DTYPES[dtype][1]
    if hidden == 3:
        assert reads == [
            hidden * DTYPES[dtype][1],
            2 * hidden * DTYPES[dtype][1],
            hidden * DTYPES[dtype][1],
        ]
    reads.clear()
    allocations.clear()
    empty = resolve_embeddings(source, ()).reader()
    assert empty.read_available(4) == b""
    empty.close()
    assert not reads and not allocations


@pytest.mark.parametrize("fault", ["change", "memory", "native_memory", "short"])
def test_midstream_faults(
    settings: Settings, monkeypatch: pytest.MonkeyPatch, fault: str, family: str
) -> None:
    directory = embedding_model(settings.model_root, family=family)
    original = ModelSource.iter_rows

    def broken(self: ModelSource, *args: Any, **kwargs: Any) -> Any:
        iterator = original(self, *args, **kwargs)
        try:
            yield next(iterator)
            if fault == "change":
                mutate_last_byte(directory / "model.safetensors")
                yield from iterator
            elif fault == "memory":
                raise MemoryError("/private/weight")
            elif fault == "native_memory":
                monkeypatch.setattr(
                    torch, "frombuffer", Mock(side_effect=torch.OutOfMemoryError("private"))
                )
                yield from iterator
        finally:
            iterator.close()

    monkeypatch.setattr(ModelSource, "iter_rows", broken)
    with TestClient(create_app(settings)) as client:
        response = client.post(address(client), json={"token_ids": [3, 0]})
        result = frames(response.content)
        assert [kind for kind, _ in result] == [1, 2, 5]
        assert (
            json.loads(result[-1][1])["code"]
            == {
                "change": "model_content_changed",
                "memory": "resource_exhausted",
                "native_memory": "resource_exhausted",
                "short": "internal_error",
            }[fault]
        )
        assert b"private" not in response.content and KEY.encode() not in response.content
    assert not numeric_cache_entries(settings.cache_dir)


@pytest.mark.parametrize("action", ["cancel", "delete_session", "disconnect", "partial_send"])
def test_progressive_cleanup_and_independent_consumer(
    settings: Settings, action: str, family: str
) -> None:
    embedding_model(settings.model_root, family=family)

    async def scenario() -> None:
        app = create_app(settings)
        async with app.router.lifespan_context(app):
            services: Services = app.state.services
            sessions, runtime = services.sessions, services.operation_delivery
            assert sessions is not None and runtime is not None
            first = await sessions.create("test/tiny")
            second = await sessions.create("test/tiny")
            result, consumer = await subscribe_embeddings(sessions, first.id, (3, 0, 3))
            _, survivor = await subscribe_embeddings(sessions, second.id, (3, 0, 3))
            prefix, release, disconnect = asyncio.Event(), asyncio.Event(), asyncio.Event()
            sent: list[bytes] = []

            async def meta() -> object:
                return result.metadata()

            async def send(message: Message) -> None:
                if message["type"] == "http.response.body":
                    sent.append(bytes(message["body"]))
                    if isinstance(message["body"], memoryview) and not prefix.is_set():
                        prefix.set()
                        await release.wait()
                        if action == "partial_send":
                            raise OSError("partially written socket")

            async def receive() -> Message:
                await disconnect.wait()
                return {"type": "http.disconnect"}

            task = asyncio.create_task(
                LMEXResponse(consumer, meta, max_data_bytes=4)({}, receive, send)
            )
            await prefix.wait()
            assert consumer.terminal is None  # Partial data precedes source exhaustion.
            if action == "cancel":
                await runtime.cancel(consumer.operation_id)  # type: ignore[arg-type]
            elif action == "delete_session":
                await sessions.delete(first.id)
            elif action == "disconnect":
                disconnect.set()
                await task
            release.set()
            if action == "partial_send":
                with pytest.raises(_SendFailed):
                    await task
            else:
                await task
            kinds = [kind for kind, _ in frames(b"".join(sent))]
            if action in {"cancel", "delete_session"}:
                assert kinds[-1] == 6
            else:
                assert kinds == [1, 2]  # No frame appended after transport failure.
            assert consumer._source_reader is None and consumer._closed
            actual = bytearray()
            while block := await survivor.read():
                actual.extend(block)
            assert actual == struct.pack("<9f", *(ROWS[3] + ROWS[0] + ROWS[3]))
            await survivor.aclose()
            await forgotten(runtime)
        assert not numeric_cache_entries(settings.cache_dir)

    run(scenario())


def test_tcp_progress_and_responsive_io(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    embedding_model(settings.model_root)
    make_tokenizer(settings.model_root, name="tokenizer")
    entered, release = Event(), Event()
    original = _TensorReader.read_available
    calls = 0

    def held(self: _TensorReader, size: int) -> bytes:
        nonlocal calls
        calls += 1
        if calls == 2:
            entered.set()
            assert release.wait(5)
        return original(self, size)

    monkeypatch.setattr(_TensorReader, "read_available", held)

    async def scenario() -> None:
        app = create_app(settings)
        async with app.router.lifespan_context(app):
            async with server(app) as client:
                session = (await client.post("/sessions", json={"model_id": "test/tiny"})).json()[
                    "id"
                ]
                async with client.stream(
                    "POST", f"/sessions/{session}/embeddings", json={"token_ids": [3, 0]}
                ) as response:
                    reader = Frames(response)
                    assert (await reader.next())[0] == 1
                    assert await reader.next() == (2, struct.pack("<3f", *ROWS[3]))
                    assert await asyncio.to_thread(entered.wait, 3)
                    try:
                        assert (await client.get("/models")).status_code == 200
                        assert (await client.get(f"/sessions/{session}/tensors")).status_code == 200
                        token_session = await client.post(
                            "/sessions", json={"model_id": "test/tokenizer"}
                        )
                        tokenized = await client.post(
                            f"/sessions/{token_session.json()['id']}/tokenize",
                            json={"text": "hello"},
                        )
                        assert tokenized.status_code == 200 and tokenized.json()["tokens"]
                        assert not release.is_set()
                    finally:
                        release.set()
                    assert await payload(reader) == struct.pack("<3f", *ROWS[0])

    try:
        run(scenario())
    finally:
        release.set()


def test_indexed_shard_and_cors(settings: Settings, family: str) -> None:
    directory = embedding_model(settings.model_root, "BF16", family=family)
    shard = directory / "weights" / "input.safetensors"
    shard.parent.mkdir()
    (directory / "model.safetensors").rename(shard)
    (directory / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    FAMILIES[family][1]: "weights/input.safetensors",
                    "lm_head.weight": "weights/input.safetensors",
                }
            }
        )
    )
    settings = replace(settings, cors_origins=("http://localhost:5173",))
    with TestClient(create_app(settings)) as client:
        response = client.post(
            address(client),
            json={"token_ids": [3, 0, 3]},
            headers={"Origin": "http://localhost:5173"},
        )
        assert response.status_code == 200
        assert response.headers["access-control-allow-origin"] == "http://localhost:5173"
        assert "X-Operation-Id" in response.headers["access-control-expose-headers"]
        result = frames(response.content)
        assert result[-1] == (4, b"")
        assert b"".join(data for kind, data in result if kind == 2) == struct.pack(
            "<9f", *(ROWS[3] + ROWS[0] + ROWS[3])
        )


def test_change_during_read_rejects_block(
    settings: Settings, monkeypatch: pytest.MonkeyPatch, family: str
) -> None:
    directory = embedding_model(settings.model_root, family=family)
    source = ModelCatalogue(settings.model_root).pin("test/tiny")
    result = resolve_embeddings(source, (0,))
    original = FileSnapshot.open

    @contextmanager
    def changing(self: FileSnapshot, name: str) -> Iterator[BinaryIO]:
        with original(self, name) as stream:
            proxy = Mock(wraps=stream)

            def read(count: int) -> bytes:
                data = stream.read(count)
                mutate_last_byte(directory / "model.safetensors")
                return data

            proxy.read = read
            yield cast(BinaryIO, proxy)

    monkeypatch.setattr(FileSnapshot, "open", changing)
    reader = result.reader()
    try:
        with pytest.raises(ModelError) as exc:
            reader.read_available(256 * 1024)
        assert exc.value.code == "model_content_changed"
    finally:
        reader.close()


def test_unsafe_output_size_before_source_access(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    embedding_model(settings.model_root)
    source = ModelCatalogue(settings.model_root).pin("test/tiny")
    table = next(t for t in source.tensors() if t.name == KEY)
    hidden = (2**53 - 1) // 4
    oversized = table.model_copy(update={"shape": (1, hidden), "numel": hidden})
    monkeypatch.setattr(
        ModelSource,
        "configuration",
        lambda self: {
            "model_type": "llama",
            "architectures": ["LlamaForCausalLM"],
            "vocab_size": 1,
            "hidden_size": hidden,
        },
    )
    monkeypatch.setattr(ModelSource, "tensors", lambda self: (oversized,))
    with pytest.raises(ModelError) as exc:
        resolve_embeddings(source, (0, 0))
    assert exc.value.code == "unsupported_size"


@pytest.mark.parametrize("kind,family", [("JunHowie", "qwen3"), ("AxionML", "qwen3_5")])
@pytest.mark.parametrize("dtype", ["F32", "F16", "BF16"])
def test_quantized_checkpoint_native_input_rows(
    settings: Settings, kind: str, family: str, dtype: str
) -> None:
    directory = embedding_model(settings.model_root, dtype, family=family)
    path = directory / "config.json"
    config = json.loads(path.read_text())
    config["quantization_config"] = config_for(kind)["quantization_config"]
    path.write_text(json.dumps(config))
    write_storage(directory / "packed.safetensors", packed_group(kind))
    # Both native rows and packed linear groups live behind an exact shard index.
    mapping = {FAMILIES[family][1]: "model.safetensors", "lm_head.weight": "model.safetensors"}
    mapping.update({name: "packed.safetensors" for name, _, _ in packed_group(kind)})
    (directory / "model.safetensors.index.json").write_text(json.dumps({"weight_map": mapping}))
    variant = "gptq-int4" if kind == "JunHowie" else "nvfp4"
    with TestClient(create_app(settings)) as client:
        response = client.post(
            address(client, f"test/tiny@{variant}"), json={"token_ids": [3, 0, 3]}
        )
        assert response.status_code == 200
        result = frames(response.content)
        assert result[-1] == (4, b"")
        assert b"".join(data for kind, data in result if kind == 2) == struct.pack(
            "<9f", *(ROWS[3] + ROWS[0] + ROWS[3])
        )


@pytest.mark.parametrize(
    "fault", ["missing", "output", "vision", "vocab", "hidden", "ambiguous", "custom"]
)
def test_family_mapping_fails_closed(settings: Settings, family: str, fault: str) -> None:
    directory = embedding_model(settings.model_root, family=family)
    path = directory / "config.json"
    config = json.loads(path.read_text())
    dims = config["text_config"] if family == "qwen3_5" else config
    if fault in {"vocab", "hidden"}:
        dims["vocab_size" if fault == "vocab" else "hidden_size"] += 1
    elif fault == "ambiguous":
        config["architectures"].append(FAMILIES[family][0])
    elif fault == "custom":
        dims["auto_map"] = {"AutoModel": "private.Custom"}
    else:
        config["tie_word_embeddings"] = True
        name = {
            "missing": "prefix." + FAMILIES[family][1],
            "output": "lm_head.weight",
            "vision": "model.visual.embed_tokens.weight",
        }[fault]
        write_weights(directory / "model.safetensors", [(name, "F32", [4, 3], [9.0] * 12)])
    path.write_text(json.dumps(config))
    with TestClient(create_app(settings)) as client:
        url = address(client)
        for ids in ([], [0]):
            response = client.post(url, json={"token_ids": ids})
            assert response.status_code == 422
            assert response.json()["code"] == "unsupported_representation"


@pytest.mark.parametrize(
    "text_config", [None, [], {}, {"vocab_size": 4}, {"vocab_size": True, "hidden_size": 3}]
)
def test_nested_dimensions_required(settings: Settings, text_config: object) -> None:
    directory = embedding_model(settings.model_root, family="qwen3_5")
    path = directory / "config.json"
    config = json.loads(path.read_text())
    config.update(vocab_size=4, hidden_size=3, text_config=text_config)
    path.write_text(json.dumps(config))
    with pytest.raises(ModelError) as exc:
        resolve_embeddings(ModelCatalogue(settings.model_root).pin("test/tiny"), ())
    assert exc.value.code == "unsupported_representation"


def test_duplicate_logical_candidates(
    settings: Settings, family: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    embedding_model(settings.model_root, family=family)
    source = ModelCatalogue(settings.model_root).pin("test/tiny")
    table = resolve_input_table(source)
    monkeypatch.setattr(ModelSource, "tensors", lambda self: (table, table))
    with pytest.raises(ModelError) as exc:
        resolve_embeddings(source, ())
    assert exc.value.code == "unsupported_representation"


def test_duplicate_physical_candidates_rejected_at_admission(
    settings: Settings, family: str
) -> None:
    directory = embedding_model(settings.model_root, family=family)
    write_weights(
        directory / "duplicate.safetensors", [(FAMILIES[family][1], "F32", [4, 3], [1.0] * 12)]
    )
    assert ModelCatalogue(settings.model_root).discover() == ()


def test_non_text_model_has_no_input_capability(settings: Settings) -> None:
    directory = embedding_model(settings.model_root)
    path = directory / "config.json"
    config = json.loads(path.read_text())
    config.update(model_type="vjepa2", architectures=["VJEPA2Model"])
    path.write_text(json.dumps(config))
    with pytest.raises(ModelError) as exc:
        resolve_embeddings(ModelCatalogue(settings.model_root).pin("test/tiny"), (0,))
    assert exc.value.code == "unsupported_representation"


@pytest.mark.parametrize("family", ["qwen3", "qwen3_5"])
def test_actionable_packed_table_reuses_logical_rows(settings: Settings, family: str) -> None:
    from quantized_oracles import gptq_fixture, nvfp4_fixture

    fixture = gptq_fixture() if family == "qwen3" else nvfp4_fixture()
    prefix = FAMILIES[family][1].removesuffix(".weight")
    fixture = replace(
        fixture,
        storage=tuple(
            replace(item, name=prefix + "." + item.name.rsplit(".", 1)[1])
            for item in fixture.storage
        ),
    )
    directory = fixture.write(settings.model_root, split=True)
    path = directory / "config.json"
    config = json.loads(path.read_text())
    config.update(_name_or_path="test/tiny", architectures=[FAMILIES[family][0]])
    dims = config.setdefault("text_config", {}) if family == "qwen3_5" else config
    dims.update(vocab_size=fixture.shape[0], hidden_size=fixture.shape[1])
    path.write_text(json.dumps(config))
    ids = [fixture.shape[0] - 1, 0, 0]
    oracle = fixture.expected()
    width = fixture.shape[1] * 4
    expected = b"".join(oracle[token * width : (token + 1) * width] for token in ids)
    with TestClient(create_app(settings)) as client:
        url = address(client, f"test/tiny@{fixture.encoding}")
        response = client.post(url, json={"token_ids": ids})
        assert response.status_code == 200
        result = frames(response.content)
        assert result[-1] == (4, b"")
        assert b"".join(data for kind, data in result if kind == 2) == expected
        empty = frames(client.post(url, json={"token_ids": []}).content)
        assert json.loads(empty[0][1])["shape"] == [0, fixture.shape[1]]
        assert empty[-1] == (4, b"")
    assert not numeric_cache_entries(settings.cache_dir)

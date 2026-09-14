import json
import os
import re
import shutil
import struct
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from typing import BinaryIO, cast
from unittest.mock import Mock

import pytest
import torch
from fastapi.testclient import TestClient

from llm_model_explorer import model_files
from llm_model_explorer.app import create_app
from llm_model_explorer.dependencies import get_catalogue
from llm_model_explorer.model_files import FileSnapshot, ModelError
from llm_model_explorer.models import ModelCatalogue
from llm_model_explorer.settings import Settings

Tensor = tuple[str, str, list[int], list[float]]


def write_weights(path: Path, tensors: list[Tensor]) -> None:
    header: dict[str, object] = {"__metadata__": {"format": "pt"}}
    data = bytearray()
    for name, dtype, shape, values in tensors:
        start = len(data)
        for value in values:
            if dtype == "BF16":
                data.extend(struct.pack("<f", value)[2:])
            else:
                data.extend(struct.pack("<f" if dtype == "F32" else "<e", value))
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [start, len(data)]}
    raw = json.dumps(header).encode()
    raw += b" " * (-len(raw) % 8)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + data)


def make_model(root: Path, name: str = "tiny", *, identity: str | None = "test/tiny") -> Path:
    directory = root / name
    directory.mkdir()
    config: dict[str, object] = {"model_type": "llama", "architectures": ["LlamaForCausalLM"]}
    if identity is not None:
        config["_name_or_path"] = identity
    (directory / "config.json").write_text(json.dumps(config))
    write_weights(directory / "model.safetensors", [("layer.weight", "F32", [2, 2], [1, -2, 3, 4])])
    return directory


def replace_header(path: Path, header: object, payload: bytes = b"\0" * 16) -> None:
    raw = header if isinstance(header, bytes) else json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + payload)


def mutate_last_byte(path: Path) -> None:
    with path.open("r+b") as stream:
        stream.seek(-1, 2)
        value = stream.read(1)
        stream.seek(-1, 2)
        stream.write(bytes([value[0] ^ 1]))


def test_sharded_inventory_and_bounded_values(model_root: Path) -> None:
    directory = make_model(model_root)
    (directory / "model.safetensors").unlink()
    tensors: list[Tensor] = [
        ("a.vector", "F32", [3], [1.25, -2, 3]),
        ("b.matrix", "F16", [2, 2], [0.5, -1, 2, 4]),
        ("c.scalar", "BF16", [], [1.5]),
        ("d.empty", "F32", [2, 0, 3], []),
        ("e.cube", "BF16", [1, 2, 2], [1, 2, -3, 0.25]),
    ]
    write_weights(directory / "part1.safetensors", tensors[:2])
    write_weights(directory / "weights/part2.safetensors", tensors[2:])
    weight_map = {t[0]: "part1.safetensors" for t in tensors[:2]} | {
        t[0]: "weights/part2.safetensors" for t in tensors[2:]
    }
    (directory / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))
    source = ModelCatalogue(model_root).pin("test/tiny")
    for descriptor, (name, dtype, shape, expected) in zip(source.tensors(), tensors, strict=True):
        assert descriptor.name == name
        assert descriptor.shape == tuple(shape)
        assert descriptor.rank == len(shape)
        assert descriptor.numel == len(expected)
        assert descriptor.path == tuple(name.split("."))
        assert descriptor.storage_dtype == dtype
        assert descriptor.logical_dtype == "float32"
        assert re.fullmatch(r"[A-Za-z0-9_-]+", descriptor.id)
        chunks = list(source.iter_tensor(descriptor.id, chunk_elements=2))
        assert all(chunk.dtype == torch.float32 and chunk.device.type == "cpu" for chunk in chunks)
        assert all(len(chunk) <= 2 for chunk in chunks)
        assert [value for chunk in chunks for value in chunk.tolist()] == expected


def test_discovery_reads_only_headers_without_tensor_allocations(
    model_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    directory = make_model(model_root)
    weight_reads: list[tuple[int, int]] = []
    original = model_files.open_local

    @contextmanager
    def tracked(root: Path, path: Path) -> Iterator[BinaryIO]:
        with original(root, path) as stream:
            wrapper = Mock(wraps=stream)

            def read(size: int = -1) -> bytes:
                if path.suffix == ".safetensors":
                    weight_reads.append((stream.tell(), size))
                return stream.read(size)

            wrapper.read.side_effect = read
            yield cast(BinaryIO, wrapper)

    forbidden = Mock(side_effect=AssertionError("full-model read/allocation"))
    monkeypatch.setattr(model_files, "open_local", tracked)
    monkeypatch.setattr(FileSnapshot, "fingerprint", forbidden)
    for name in ["load", "tensor", "empty", "zeros", "frombuffer"]:
        monkeypatch.setattr(torch, name, forbidden)
    monkeypatch.setattr("urllib.request.urlopen", forbidden)
    entries = ModelCatalogue(model_root).discover()
    assert len(entries) == 1
    assert len(entries[0].tensors()) == 1
    length = struct.unpack("<Q", (directory / "model.safetensors").read_bytes()[:8])[0]
    assert weight_reads == [(0, 8), (8, length)]
    forbidden.assert_not_called()


@pytest.mark.parametrize(
    "identity",
    [
        "/private/model",
        "./local/model",
        "../model",
        "C:\\models\\x",
        "file:///x",
        "a/b/c",
        "~user/model",
    ],
)
def test_filesystem_identity_falls_back(model_root: Path, identity: str) -> None:
    make_model(model_root, identity=identity)
    summary = ModelCatalogue(model_root).list_models()[0]
    assert summary.id == "tiny"
    assert summary.display_name == "tiny"
    assert identity not in summary.model_dump_json()


def test_identity_revision_metadata_and_fallback(model_root: Path) -> None:
    directory = make_model(model_root, identity=None)
    assert ModelCatalogue(model_root).list_models()[0].id == "tiny"
    config = {
        "model_type": "/private/type",
        "architectures": ["/private/class"],
        "_name_or_path": "org/model",
        "_commit_hash": "abc123",
    }
    (directory / "config.json").write_text(json.dumps(config))
    summary = ModelCatalogue(model_root).list_models()[0]
    assert summary.id == "org/model@abc123"
    assert summary.architectures == ()
    assert summary.model_type is None
    assert "parameter_count" not in summary.model_dump(exclude_none=True)


def test_duplicate_identity_collapse_and_conflict(model_root: Path) -> None:
    directory = make_model(model_root)
    duplicate = model_root / "duplicate"
    shutil.copytree(directory, duplicate)
    assert len(ModelCatalogue(model_root).list_models()) == 1
    mutate_last_byte(duplicate / "model.safetensors")
    with pytest.raises(ModelError, match="Ambiguous") as error:
        ModelCatalogue(model_root).discover()
    assert error.value.code == "validation_error"


def test_relocation_and_same_length_content_change(model_root: Path) -> None:
    directory = make_model(model_root)
    catalogue = ModelCatalogue(model_root)
    source = catalogue.pin("test/tiny")
    ids = [d.id for d in source.tensors()]
    relocated = model_root / "relocated"
    directory.rename(relocated)
    moved = catalogue.pin("test/tiny")
    assert moved.fingerprint == source.fingerprint
    assert [d.id for d in moved.tensors()] == ids
    weights = relocated / "model.safetensors"
    before = weights.stat()
    mutate_last_byte(weights)
    os.utime(weights, ns=(before.st_atime_ns, before.st_mtime_ns))
    assert weights.stat().st_size == before.st_size
    with pytest.raises(ModelError) as error:
        moved.check_unchanged()
    assert error.value.code == "model_content_changed"
    assert catalogue.pin("test/tiny").fingerprint != moved.fingerprint


@pytest.mark.parametrize(
    "change", ["weights", "replace", "config", "tokenizer", "added", "deleted", "symlink"]
)
def test_snapshot_rejects_changes_between_chunks(model_root: Path, change: str) -> None:
    directory = make_model(model_root)
    tokenizer = directory / "tokenizer.json"
    tokenizer.write_text('{"version":"1.0"}')
    source = ModelCatalogue(model_root).pin("test/tiny")
    iterator = source.iter_tensor(source.tensors()[0].id, chunk_elements=1)
    assert next(iterator).tolist() == [1]
    weights = directory / "model.safetensors"
    if change == "weights":
        mutate_last_byte(weights)
    elif change == "replace":
        replacement = directory / "replacement"
        replacement.write_bytes(weights.read_bytes())
        replacement.replace(weights)
    elif change == "config":
        (directory / "config.json").write_text('{"model_type":"gpt2"}')
    elif change == "tokenizer":
        tokenizer.write_text('{"version":"2.0"}')
    elif change == "added":
        (directory / "added_tokens.json").write_text("{}")
    elif change == "deleted":
        tokenizer.unlink()
    else:
        target = model_root / "replacement-weights"
        target.write_bytes(weights.read_bytes())
        weights.unlink()
        weights.symlink_to(target)
    with pytest.raises(ModelError) as error:
        next(iterator)
    assert error.value.code == "model_content_changed"
    assert error.value.status == 409


def test_final_chunk_check_and_tokenizer_guard(model_root: Path) -> None:
    directory = make_model(model_root)
    source = ModelCatalogue(model_root).pin("test/tiny")
    iterator = source.iter_tensor(source.tensors()[0].id)
    assert next(iterator).tolist() == [1, -2, 3, 4]
    with pytest.raises(ModelError, match="changed"):
        with source.local_directory() as local:
            assert local == directory
            mutate_last_byte(local / "model.safetensors")
    with pytest.raises(ModelError, match="changed"):
        next(iterator)


def test_change_during_read_is_not_yielded(
    model_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    directory = make_model(model_root)
    source = ModelCatalogue(model_root).pin("test/tiny")
    original = torch.frombuffer

    def change(raw: bytearray, *, dtype: torch.dtype) -> torch.Tensor:
        result = original(raw, dtype=dtype)
        mutate_last_byte(directory / "model.safetensors")
        return result

    monkeypatch.setattr(torch, "frombuffer", change)
    with pytest.raises(ModelError, match="changed"):
        next(source.iter_tensor(source.tensors()[0].id))


@pytest.mark.parametrize(
    "header",
    [
        {"x": {"dtype": "F32", "shape": [4], "data_offsets": [1, 17]}},
        {"x": {"dtype": "F32", "shape": [4], "data_offsets": [0, 15]}},
        {"x": {"dtype": "F32", "shape": [-1], "data_offsets": [0, 16]}},
        {"x": {"dtype": "F32", "shape": [True], "data_offsets": [0, 16]}},
        {"x": {"dtype": "F32", "shape": [2**52, 4], "data_offsets": [0, 16]}},
        {"x": {"dtype": "I8", "shape": [16], "data_offsets": [0, 16]}},
        {"x": {"dtype": "NF4", "shape": [32], "data_offsets": [0, 16]}},
        {
            "x": {"dtype": "F32", "shape": [2], "data_offsets": [0, 8]},
            "y": {"dtype": "F32", "shape": [2], "data_offsets": [0, 8]},
        },
        {"x": {"dtype": "F32", "shape": [2], "data_offsets": [8, 16]}},
        {"__metadata__": {"wrong": 123}},
        b'{"x":{},"x":{}}',
        b"[]",
        b"{invalid",
    ],
)
def test_invalid_headers_are_reported_locally(
    model_root: Path, header: object, caplog: pytest.LogCaptureFixture
) -> None:
    directory = make_model(model_root)
    replace_header(directory / "model.safetensors", header)
    assert ModelCatalogue(model_root).discover() == ()
    assert "Skipping invalid local model candidate" in caplog.text


@pytest.mark.parametrize(
    "payload", [b"", b"1234", struct.pack("<Q", 2**63), struct.pack("<Q", 999)]
)
def test_truncated_or_huge_header(model_root: Path, payload: bytes) -> None:
    directory = make_model(model_root)
    (directory / "model.safetensors").write_bytes(payload)
    assert ModelCatalogue(model_root).discover() == ()


@pytest.mark.parametrize(
    "reference",
    [
        "missing.safetensors",
        "../outside.safetensors",
        "/outside.safetensors",
        "C:\\outside.safetensors",
        "a/../../outside.safetensors",
    ],
)
def test_missing_shard_and_traversal(model_root: Path, reference: str) -> None:
    directory = make_model(model_root)
    (directory / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"layer.weight": reference}})
    )
    assert ModelCatalogue(model_root).discover() == ()


def test_index_mismatch_duplicate_shards_and_duplicate_index_keys(model_root: Path) -> None:
    directory = make_model(model_root)
    index = directory / "model.safetensors.index.json"
    index.write_text('{"weight_map":{"other":"model.safetensors"}}')
    assert ModelCatalogue(model_root).discover() == ()
    index.write_text('{"weight_map":{"x":"model.safetensors","x":"model.safetensors"}}')
    assert ModelCatalogue(model_root).discover() == ()
    index.unlink()
    shutil.copyfile(directory / "model.safetensors", directory / "duplicate.safetensors")
    assert ModelCatalogue(model_root).discover() == ()


@pytest.mark.parametrize(
    "asset", ["directory", "config.json", "model.safetensors", "tokenizer.json"]
)
def test_outside_symlinks_are_refused(model_root: Path, tmp_path: Path, asset: str) -> None:
    outside = make_model(tmp_path, "outside")
    if asset == "directory":
        (model_root / "escape").symlink_to(outside, target_is_directory=True)
    else:
        directory = make_model(model_root)
        target = outside / asset
        if not target.exists():
            target.write_text("{}")
        (directory / asset).unlink(missing_ok=True)
        (directory / asset).symlink_to(target)
    assert ModelCatalogue(model_root).discover() == ()


def test_in_root_symlinks_and_unrelated_directories(model_root: Path) -> None:
    directory = make_model(model_root)
    original = directory / "model.safetensors"
    target = model_root / "weight-blob"
    original.rename(target)
    original.symlink_to(target)
    unrelated = model_root / "unrelated"
    unrelated.mkdir()
    make_model(unrelated, "deep")
    assert len(ModelCatalogue(model_root).discover()) == 1
    source = ModelCatalogue(model_root).pin("test/tiny")
    assert next(source.iter_tensor(source.tensors()[0].id)).tolist() == [1, -2, 3, 4]


@pytest.mark.parametrize(
    "config",
    [
        [],
        {},
        {"model_type": 3},
        {"model_type": "llama", "quantization_config": {"quant_method": "bitsandbytes"}},
    ],
)
def test_bad_or_unsupported_configuration(model_root: Path, config: object) -> None:
    directory = make_model(model_root)
    (directory / "config.json").write_text(json.dumps(config))
    assert ModelCatalogue(model_root).discover() == ()


def test_independent_concurrent_readers(model_root: Path) -> None:
    make_model(model_root)
    source = ModelCatalogue(model_root).pin("test/tiny")
    tensor_id = source.tensors()[0].id
    abandoned = source.iter_tensor(tensor_id, chunk_elements=1)
    assert next(abandoned).tolist() == [1]
    abandoned.close()

    def consume() -> list[float]:
        return [
            v for chunk in source.iter_tensor(tensor_id, chunk_elements=1) for v in chunk.tolist()
        ]

    with ThreadPoolExecutor(max_workers=2) as workers:
        assert list(workers.map(lambda _: consume(), range(2))) == [[1, -2, 3, 4]] * 2


def test_models_http_contract_privacy_and_errors(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    directory = make_model(settings.model_root, identity=str(settings.model_root / "private"))
    app = create_app(settings)
    with TestClient(app) as client:
        response = client.get("/models")
        assert response.status_code == 200
        assert response.headers["cache-control"] == "no-store"
        summary = response.json()["models"][0]
        assert summary["id"] == "tiny"
        assert set(summary) == {
            "id",
            "display_name",
            "architectures",
            "model_type",
            "size_bytes",
            "tokenizer_available",
        }
        assert str(settings.model_root) not in response.text
        assert "fingerprint" not in response.text
        assert summary["size_bytes"] == sum(p.stat().st_size for p in directory.iterdir())
        fake = Mock(spec=ModelCatalogue)
        app.dependency_overrides[get_catalogue] = lambda: fake
        for error, status, code in [
            (MemoryError("private/path"), 503, "resource_exhausted"),
            (OSError("private/path"), 500, "internal_error"),
            (ModelError("validation_error", "Ambiguous model identity."), 422, "validation_error"),
        ]:
            fake.list_models.side_effect = error
            response = client.get("/models")
            assert response.status_code == status
            assert response.json()["code"] == code
            assert "private/path" not in response.text
            assert response.headers["cache-control"] == "no-store"


def test_unknown_source_and_tensor(model_root: Path) -> None:
    make_model(model_root)
    catalogue = ModelCatalogue(model_root)
    with pytest.raises(ModelError) as error:
        catalogue.pin("missing")
    assert error.value.code == "model_not_found"
    source = catalogue.pin("test/tiny")
    with pytest.raises(ModelError) as error:
        next(source.iter_tensor("missing"))
    assert error.value.code == "tensor_not_found"
    for chunk_size in [0, -1, 2**53]:
        with pytest.raises(ValueError):
            next(source.iter_tensor(source.tensors()[0].id, chunk_elements=chunk_size))


def test_fingerprint_includes_tokenizer_and_config_across_roots(
    model_root: Path, tmp_path: Path
) -> None:
    directory = make_model(model_root)
    (directory / "vocab.txt").write_text("first\nsecond\n")
    source = ModelCatalogue(model_root).pin("test/tiny")
    second_root = tmp_path / "other-root"
    second_root.mkdir()
    copy = second_root / "different-directory"
    shutil.copytree(directory, copy)
    catalogue = ModelCatalogue(second_root)
    assert catalogue.pin("test/tiny").fingerprint == source.fingerprint
    assert catalogue.list_models()[0].tokenizer_available
    (copy / "vocab.txt").write_text("third\nsecond\n")
    changed_tokenizer = catalogue.pin("test/tiny")
    assert changed_tokenizer.fingerprint != source.fingerprint
    (copy / "config.json").write_text((copy / "config.json").read_text() + " ")
    assert catalogue.pin("test/tiny").fingerprint != changed_tokenizer.fingerprint


def test_hashing_is_bounded_and_rejects_mid_read_changes(
    model_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    directory = make_model(model_root)
    entry = ModelCatalogue(model_root).discover()[0]
    original = model_files.open_local
    read_sizes: list[int] = []

    @contextmanager
    def tracked(root: Path, path: Path) -> Iterator[BinaryIO]:
        with original(root, path) as stream:
            wrapper = Mock(wraps=stream)

            def read(size: int = -1) -> bytes:
                read_sizes.append(size)
                raw = stream.read(size)
                if path.suffix == ".safetensors" and raw:
                    mutate_last_byte(directory / "model.safetensors")
                return raw

            wrapper.read.side_effect = read
            yield cast(BinaryIO, wrapper)

    monkeypatch.setattr(model_files, "open_local", tracked)
    with pytest.raises(ModelError, match="changed"):
        entry.pin()
    assert read_sizes and all(size == model_files.HASH_CHUNK_BYTES for size in read_sizes)


@pytest.mark.skipif(os.open not in os.supports_dir_fd, reason="POSIX descriptor-relative opens")
def test_symlink_swapped_after_resolution_is_never_read(
    model_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    directory = make_model(model_root)
    target = directory / "model.safetensors"
    outside = tmp_path / "outside"
    outside.write_bytes(b"must never be read")
    original = model_files.confined

    def resolve_then_swap(root: Path, path: Path) -> Path:
        resolved = original(root, path)
        target.unlink()
        target.symlink_to(outside)
        return resolved

    monkeypatch.setattr(model_files, "confined", resolve_then_swap)
    with pytest.raises(OSError):
        with model_files.open_local(model_root, target):
            pytest.fail("replaced symlink was opened")


def test_real_identity_collision_has_path_free_http_error(settings: Settings) -> None:
    directory = make_model(settings.model_root)
    duplicate = settings.model_root / "copy"
    with TestClient(create_app(settings)) as client:
        shutil.copytree(directory, duplicate)
        mutate_last_byte(duplicate / "model.safetensors")
        response = client.get("/models")
    assert response.status_code == 422
    assert response.json()["code"] == "validation_error"
    assert str(settings.model_root) not in response.text
    assert response.headers["cache-control"] == "no-store"


def test_identity_collision_at_startup_prevents_readiness(settings: Settings) -> None:
    directory = make_model(settings.model_root)
    duplicate = settings.model_root / "copy"
    shutil.copytree(directory, duplicate)
    mutate_last_byte(duplicate / "model.safetensors")
    app = create_app(settings)
    with pytest.raises(ModelError, match="Ambiguous model identity"), TestClient(app):
        pytest.fail("ambiguous catalogue became ready")
    assert not hasattr(app.state, "services")

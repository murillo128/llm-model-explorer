"""Reduced encoding fixtures; provenance is in fixtures/quantized-configs.json.

Payloads are synthetic, not checkpoint acceptance or a quantization decoder oracle.
The GPTQ [input=128, output=8] and NVFP4 [input=16, output=2] groups
preserve the reviewed packing axes, dtypes and scale geometry.
"""

import hashlib
import json
import math
import shutil
import struct
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, BinaryIO, cast
from unittest.mock import Mock

import pytest
import torch
from fastapi.testclient import TestClient
from test_models import make_model, mutate_last_byte, replace_header, write_weights
from test_tensor_data import frames

from llm_model_explorer import model_files
from llm_model_explorer.app import create_app
from llm_model_explorer.embeddings import resolve_input_table
from llm_model_explorer.model_files import FileSnapshot, ModelError
from llm_model_explorer.models import ModelCatalogue
from llm_model_explorer.settings import Settings

KINDS = ["JunHowie", "AxionML"]
PREFIX = "model.layers.0.self_attn.q_proj"
# Deliberately independent byte-width oracle, not the production dtype registry.
WIDTHS = {"F32": 4, "F16": 2, "BF16": 2, "I32": 4, "U8": 1, "F8_E4M3": 1}
Storage = tuple[str, str, list[int]]


def config_for(kind: str) -> dict[str, Any]:
    fixtures = json.loads((Path(__file__).parent / "fixtures/quantized-configs.json").read_text())
    result: dict[str, Any] = fixtures[kind]["config"]
    return result


def packed_group(kind: str) -> list[Storage]:
    if kind == "JunHowie":
        entries = [
            ("qweight", "I32", [16, 8]),
            ("qzeros", "I32", [1, 1]),
            ("scales", "F16", [1, 8]),
            ("g_idx", "I32", [128]),
        ]
    else:
        entries = [
            ("weight", "U8", [2, 8]),
            ("weight_scale", "F8_E4M3", [2, 1]),
            ("weight_scale_2", "F32", []),
            ("input_scale", "F32", []),
        ]
    prefix = (
        PREFIX
        if kind == "JunHowie"
        else PREFIX.replace("model.layers", "model.language_model.layers")
    )
    return [(f"{prefix}.{name}", dtype, shape) for name, dtype, shape in entries]


def write_storage(path: Path, entries: list[Storage]) -> None:
    header: dict[str, object] = {}
    offset = 0
    for name, dtype, shape in entries:
        end = offset + math.prod(shape) * WIDTHS[dtype]
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [offset, end]}
        offset = end
    path.parent.mkdir(parents=True, exist_ok=True)
    replace_header(path, header, b"\0" * offset)


def quantized_model(root: Path, kind: str, *, native: bool = True, indexed: bool = False) -> Path:
    directory = root / "quantized"
    directory.mkdir()
    (directory / "config.json").write_text(json.dumps(config_for(kind)))
    group = packed_group(kind)
    if indexed:
        write_storage(directory / "parts/a.safetensors", group[:2])
        write_storage(directory / "parts/b.safetensors", group[2:])
        mapping = {name: "parts/a.safetensors" for name, _, _ in group[:2]} | {
            name: "parts/b.safetensors" for name, _, _ in group[2:]
        }
    else:
        write_storage(directory / "model.safetensors", group)
        mapping = {name: "model.safetensors" for name, _, _ in group}
    if native:
        weights = [
            ("model.norm.weight", "BF16", [2], [1.5, -2.25]),
            ("model.embed_tokens.weight", "F16", [2, 2], [1.0, -0.0, 0.5, -1.5]),
            ("model.conv.weight", "F32", [1, 1, 2], [0.0, -0.0]),
        ]
        write_weights(directory / "native.safetensors", weights)
        mapping.update({name: "native.safetensors" for name, *_ in weights})
    if indexed:
        (directory / "model.safetensors.index.json").write_text(json.dumps({"weight_map": mapping}))
    return directory


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("indexed", [False, True])
def test_physical_inventory_native_ids_and_exact_bytes(
    settings: Settings, kind: str, indexed: bool
) -> None:
    directory = quantized_model(settings.model_root, kind, indexed=indexed)
    catalogue = ModelCatalogue(settings.model_root)
    entry = catalogue.discover()[0]
    source = entry.pin()
    physical = source.physical_tensors()
    assert entry.physical_tensors() == physical
    assert len(physical) == 7
    assert [item.name for item in physical] == sorted(item.name for item in physical)
    for storage in physical:
        assert storage.byte_length == math.prod(storage.shape) * WIDTHS[storage.dtype]
        assert not hasattr(storage, "id") and not hasattr(storage, "logical_dtype")
        raw = (directory / storage.file).read_bytes()
        header_length = struct.unpack("<Q", raw[:8])[0]
        record = json.loads(raw[8 : 8 + header_length])[storage.name]
        assert storage.offset == 8 + header_length + record["data_offsets"][0]
    names = {t.name for t in source.tensors()}
    logical_name = packed_group(kind)[0][0].rsplit(".", 1)[0] + ".weight"
    assert names == {
        "model.norm.weight",
        "model.embed_tokens.weight",
        "model.conv.weight",
        logical_name,
    }
    expected = {
        "model.norm.weight": struct.pack("<2f", 1.5, -2.25),
        "model.embed_tokens.weight": struct.pack("<4f", 1, -0.0, 0.5, -1.5),
        "model.conv.weight": struct.pack("<2f", 0, -0.0),
        logical_name: struct.pack("<f", -0.0 if kind == "JunHowie" else 0.0)
        * (1024 if kind == "JunHowie" else 32),
    }
    with TestClient(create_app(settings)) as client:
        session = client.post("/sessions", json={"model_id": "quantized"})
        assert session.status_code == 201
        base = f"/sessions/{session.json()['id']}"
        response = client.get(base + "/tensors")
        assert response.headers["cache-control"] == "no-store"
        inventory = response.json()
        assert inventory == source.inventory()
        assert inventory["coverage"] == "complete" and inventory["diagnostics"] == []
        assert str(directory) not in response.text
        for tensor in inventory["tensors"]:
            assert tensor["id"] == hashlib.sha256(tensor["name"].encode()).hexdigest()
            for _ in range(2):  # Native direct reads and warm conversion-cache reads.
                result = frames(client.get(base + f"/tensors/{tensor['id']}/data").content)
                assert result[-1] == (4, b"")
                assert (
                    b"".join(data for kind, data in result if kind == 2) == expected[tensor["name"]]
                )
        for storage in physical:
            if storage.name in names:
                continue
            excluded_id = hashlib.sha256(storage.name.encode()).hexdigest()
            for suffix in ["data", "statistics", "distributions"]:
                response = client.get(base + f"/tensors/{excluded_id}/{suffix}")
                assert response.status_code == 404
                assert response.json()["code"] == "tensor_not_found"
                assert "x-operation-id" not in response.headers
            for read in [source.iter_tensor(excluded_id), source.iter_rows(excluded_id, (0,))]:
                with pytest.raises(ModelError) as error:
                    next(read)
                assert error.value.code == "tensor_not_found"
        response = client.post(base + "/embeddings", json={"token_ids": []})
        assert response.status_code == 422
        assert response.json()["code"] == "unsupported_representation"


@pytest.mark.parametrize("kind", KINDS)
def test_explicit_partial_for_unknown_native_metadata(model_root: Path, kind: str) -> None:
    directory = quantized_model(model_root, kind, native=False)
    # No claimed mathematical role for this native storage record.
    write_weights(directory / "unresolved.safetensors", [("unknown.region", "F32", [2], [1, 2])])
    source = ModelCatalogue(model_root).pin("quantized")
    assert [tensor.name for tensor in source.tensors()] == [
        packed_group(kind)[0][0].rsplit(".", 1)[0] + ".weight"
    ]
    assert len(source.physical_tensors()) == 5
    inventory = source.inventory()
    assert isinstance(inventory["tensors"], list)
    assert len(inventory["tensors"]) == 1 and inventory["coverage"] == "partial"
    assert inventory["diagnostics"]
    assert "unknown.region" in str(inventory["diagnostics"])
    with pytest.raises(ModelError, match="embedding source"):
        resolve_input_table(source)


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("index", range(4))
@pytest.mark.parametrize("defect", ["missing", "dtype", "shape"])
def test_rejects_incomplete_or_wrong_encoding_groups(
    model_root: Path, kind: str, index: int, defect: str
) -> None:
    directory = quantized_model(model_root, kind)
    group = packed_group(kind)
    name, dtype, shape = group[index]
    if defect == "missing":
        group.pop(index)
    elif defect == "dtype":
        group[index] = (name, "BF16", shape)
    else:
        group[index] = (name, dtype, [*shape, 1])
    write_storage(directory / "model.safetensors", group)
    assert ModelCatalogue(model_root).discover() == ()


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("defect", ["width", "gap", "overlap", "oversize", "unknown_dtype"])
def test_physical_offsets_remain_strict(model_root: Path, kind: str, defect: str) -> None:
    directory = quantized_model(model_root, kind)
    path = directory / "model.safetensors"
    raw = path.read_bytes()
    length = struct.unpack("<Q", raw[:8])[0]
    header = json.loads(raw[8 : 8 + length])
    record = header[packed_group(kind)[0][0]]
    if defect == "width":
        record["data_offsets"][1] -= 1
    elif defect == "gap":
        record["data_offsets"] = [n + 1 for n in record["data_offsets"]]
    elif defect == "overlap":
        other = header[packed_group(kind)[1][0]]
        other["data_offsets"] = [0, other["data_offsets"][1] - other["data_offsets"][0]]
    elif defect == "oversize":
        record["shape"] = [2**53, 2]
    else:
        record["dtype"] = "F4"
    replace_header(path, header, raw[8 + length :])
    assert ModelCatalogue(model_root).discover() == ()


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("defect", ["label", "bits", "group_size", "dynamic", "extra", "boolean"])
def test_rejects_unsupported_quantization_metadata(
    model_root: Path, kind: str, defect: str
) -> None:
    directory = quantized_model(model_root, kind)
    config = config_for(kind)
    quant = config["quantization_config"]
    if defect == "label":
        quant["quant_method"] = "other"
    elif defect == "extra":
        quant["unreviewed_encoding_option"] = True
    elif kind == "JunHowie":
        field, value = {
            "bits": ("bits", 8),
            "group_size": ("group_size", 32),
            "dynamic": ("desc_act", True),
            "boolean": ("sym", 1),
        }[defect]
        quant[field] = value
    else:
        field, value = {
            "bits": ("num_bits", 8),
            "group_size": ("group_size", 32),
            "dynamic": ("dynamic", True),
            "boolean": ("dynamic", 0),
        }[defect]
        quant["config_groups"]["group_0"]["weights"][field] = value
    (directory / "config.json").write_text(json.dumps(config))
    assert ModelCatalogue(model_root).discover() == ()


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("defect", ["missing_shard", "duplicate", "mapping", "traversal", "escape"])
def test_quantized_shard_and_root_protection(
    model_root: Path, tmp_path: Path, kind: str, defect: str
) -> None:
    directory = quantized_model(model_root, kind, indexed=True)
    index = directory / "model.safetensors.index.json"
    doc = json.loads(index.read_text())
    shard = directory / "parts/a.safetensors"
    if defect == "missing_shard":
        shard.unlink()
    elif defect == "duplicate":
        shutil.copyfile(shard, directory / "parts/b.safetensors")
    elif defect == "mapping":
        doc["weight_map"][packed_group(kind)[0][0]] = "parts/b.safetensors"
    elif defect == "traversal":
        doc["weight_map"][packed_group(kind)[0][0]] = "../parts/a.safetensors"
    else:
        outside = tmp_path / "outside.safetensors"
        shard.rename(outside)
        shard.symlink_to(outside)
    index.write_text(json.dumps(doc))
    assert ModelCatalogue(model_root).discover() == ()


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("asset", ["config.json", "model.safetensors", "processor_config.json"])
def test_quantized_snapshot_and_relocation(model_root: Path, kind: str, asset: str) -> None:
    directory = quantized_model(model_root, kind)
    (directory / "processor_config.json").write_text('{"processor_class":"Unexecuted"}')
    catalogue = ModelCatalogue(model_root)
    source = catalogue.pin("quantized")
    descriptors = source.tensors()
    directory.rename(model_root / "moved")
    moved = catalogue.pin("moved")
    assert moved.fingerprint == source.fingerprint
    assert moved.tensors() == descriptors
    stream = moved.iter_tensor(moved.tensors()[0].id, chunk_elements=1)
    next(stream)
    mutate_last_byte(model_root / "moved" / asset)
    for operation in [moved.tensors, moved.inventory, moved.physical_tensors, moved.configuration]:
        with pytest.raises(ModelError) as error:
            operation()
        assert error.value.code == "model_content_changed"
    with pytest.raises(ModelError, match="changed"):
        next(stream)


@pytest.mark.parametrize("kind", KINDS)
def test_quantized_discovery_is_header_only(
    model_root: Path, kind: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    quantized_model(model_root, kind)
    original = model_files.open_local
    reads: list[tuple[int, int]] = []

    @contextmanager
    def tracked(root: Path, path: Path) -> Iterator[BinaryIO]:
        with original(root, path) as stream:
            wrapper = Mock(wraps=stream)

            def read(size: int = -1) -> bytes:
                if path.suffix == ".safetensors":
                    reads.append((stream.tell(), size))
                    assert stream.tell() in {0, 8}
                    assert 0 < size <= model_files.MAX_METADATA_BYTES
                return stream.read(size)

            wrapper.read.side_effect = read
            yield cast(BinaryIO, wrapper)

    forbidden = Mock(side_effect=AssertionError("unexpected weights or model work"))
    monkeypatch.setattr(model_files, "open_local", tracked)
    monkeypatch.setattr(FileSnapshot, "fingerprint", forbidden)
    for name in ["load", "frombuffer", "tensor", "empty", "zeros"]:
        monkeypatch.setattr(torch, name, forbidden)
    monkeypatch.setattr("urllib.request.urlopen", forbidden)
    entries = ModelCatalogue(model_root).discover()
    assert len(entries) == 1
    assert len(entries[0].physical_tensors()) == 7
    assert len(reads) == 4
    forbidden.assert_not_called()


def test_vjepa_admission_without_tokenizer_or_original_weights(settings: Settings) -> None:
    directory = make_model(settings.model_root)
    (directory / "config.json").write_text(
        json.dumps({"model_type": "vjepa2", "architectures": ["VJEPA2Model"]})
    )
    (directory / "video_preprocessor_config.json").write_text('{"processor_class":"Unexecuted"}')
    write_weights(
        directory / "model.safetensors",
        [
            (
                "encoder.embeddings.patch_embeddings.projection.weight",
                "F32",
                [1, 1, 1, 1, 2],
                [1, -2],
            ),
            ("predictor.proj.weight", "F32", [2, 2], [1, 2, 3, 4]),
        ],
    )
    with TestClient(create_app(settings)) as client:
        model = client.get("/models").json()["models"][0]
        assert model["tokenizer_available"] is False
        session = client.post("/sessions", json={"model_id": "tiny"})
        assert session.status_code == 201
        base = f"/sessions/{session.json()['id']}"
        inventory = client.get(base + "/tensors").json()
        assert inventory["coverage"] == "complete" and inventory["diagnostics"] == []
        assert len(inventory["tensors"]) == 2
        for tensor in inventory["tensors"]:
            assert client.get(base + f"/tensors/{tensor['id']}/data").status_code == 200
        for endpoint, body in [("tokenize", {"text": "test"}), ("embeddings", {"token_ids": []})]:
            response = client.post(base + "/" + endpoint, json=body)
            assert response.status_code == 422
            assert response.json()["code"] == "unsupported_representation"


@pytest.mark.parametrize("kind", KINDS)
def test_orphan_native_scales_are_not_numeric_weights(model_root: Path, kind: str) -> None:
    directory = quantized_model(model_root, kind)
    suffix = "scales" if kind == "JunHowie" else "weight_scale_2"
    write_weights(directory / "orphan.safetensors", [(f"orphan.{suffix}", "F32", [], [1])])
    source = ModelCatalogue(model_root).pin("quantized")
    inventory = source.inventory()
    assert inventory["coverage"] == "partial"
    assert f"orphan.{suffix}" in str(inventory["diagnostics"])
    assert all(tensor.name != f"orphan.{suffix}" for tensor in source.tensors())


@pytest.mark.parametrize("kind", KINDS)
def test_quantization_label_requires_a_real_encoding_group(model_root: Path, kind: str) -> None:
    directory = quantized_model(model_root, kind)
    (directory / "model.safetensors").unlink()
    assert ModelCatalogue(model_root).discover() == ()


def test_nvfp4_ignore_must_agree_with_physical_storage(model_root: Path) -> None:
    directory = quantized_model(model_root, "AxionML")
    config = config_for("AxionML")
    config["quantization_config"]["ignore"] = ["model.language_model.layers.*"]
    (directory / "config.json").write_text(json.dumps(config))
    assert ModelCatalogue(model_root).discover() == ()


def test_gptq_conflicting_native_weight_is_ambiguous(model_root: Path) -> None:
    directory = quantized_model(model_root, "JunHowie")
    write_weights(
        directory / "conflict.safetensors", [(PREFIX + ".weight", "F32", [8, 128], [0] * 1024)]
    )
    assert ModelCatalogue(model_root).discover() == ()


@pytest.mark.parametrize("dtype", ["F32", "F16", "BF16"])
def test_native_baseline_keeps_auxiliary_named_entries(settings: Settings, dtype: str) -> None:
    directory = make_model(settings.model_root)
    write_weights(
        directory / "model.safetensors",
        [
            ("layer.scales", dtype, [2], [1.5, -2.25]),
            ("unresolved.region", dtype, [], [0.5]),
        ],
    )
    source = ModelCatalogue(settings.model_root).pin("test/tiny")
    assert source.inventory()["coverage"] == "complete"
    assert source.inventory()["diagnostics"] == []
    assert len(source.tensors()) == len(source.physical_tensors()) == 2
    for descriptor in source.tensors():
        assert descriptor.id == hashlib.sha256(descriptor.name.encode()).hexdigest()
    assert [v for chunk in source.iter_tensor(source.tensors()[0].id) for v in chunk.tolist()] == [
        1.5,
        -2.25,
    ]


@pytest.mark.parametrize("kind", KINDS)
def test_snapshot_change_is_an_http_conflict(settings: Settings, kind: str) -> None:
    directory = quantized_model(settings.model_root, kind)
    with TestClient(create_app(settings)) as client:
        response = client.post("/sessions", json={"model_id": "quantized"})
        assert response.status_code == 201
        base = f"/sessions/{response.json()['id']}"
        mutate_last_byte(directory / "model.safetensors")
        response = client.get(base + "/tensors")
        assert response.status_code == 409
        assert response.json()["code"] == "model_content_changed"
        assert str(directory) not in response.text

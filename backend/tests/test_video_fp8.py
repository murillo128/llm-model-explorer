"""Independent byte oracle for the selected Comfy tensor-wise FP8 format."""

import json
import struct
from pathlib import Path

import pytest
import torch
from fastapi.testclient import TestClient
from test_tensor_data import frames

from llm_model_explorer.app import create_app
from llm_model_explorer.model_files import FileSnapshot, ModelError
from llm_model_explorer.models import ModelCatalogue
from llm_model_explorer.quantized_inventory import logical_locations
from llm_model_explorer.settings import Settings
from llm_model_explorer.tensor_source import ModelSource, parse_header

PREFIX = "video.high_noise.blocks.1.self_attn.q"


def package(root: Path, layout: str = "legacy", defect: str = "") -> Path:
    directory = root / "video"
    directory.mkdir()
    config = {
        "model_type": "video_inspection",
        "quantization_config": {
            "quant_method": "fp8-e4m3-scaled",
            "format_version": 1,
            "components": {"video.high_noise": layout},
        },
    }
    (directory / "config.json").write_text(json.dumps(config))
    # E4M3FN: 0x38=1, 0xc0=-2, 0x30=0.5; two distinct rows.
    records = {
        PREFIX + ".weight": ("F8_E4M3", [2, 3], bytes([0x38, 0xC0, 0x30, 0x40, 0xB8, 0x00])),
        "video.vae.bias": ("F16", [2], struct.pack("<2e", 3.0, -4.0)),
    }
    weight_scale = PREFIX + (".scale_weight" if layout == "legacy" else ".weight_scale")
    input_scale = PREFIX + (".scale_input" if layout == "legacy" else ".input_scale")
    records[weight_scale] = ("F32", [], struct.pack("<f", 0.25))
    records[input_scale] = ("F32", [1], struct.pack("<f", 7.0))
    if layout == "legacy":
        records["video.high_noise.scaled_fp8"] = ("F32", [2], struct.pack("<2f", 13.0, 17.0))
    if defect == "missing_scale":
        del records[weight_scale]
    elif defect == "nonfinite":
        records[weight_scale] = ("F32", [], struct.pack("<f", float("nan")))
    elif defect == "scale_shape":
        records[weight_scale] = ("F32", [2], struct.pack("<2f", 0.25, 0.5))
    elif defect == "orphan":
        records["video.low_noise.other.scale_weight"] = ("F32", [], struct.pack("<f", 1.0))
    elif defect == "mixed_names":
        records[PREFIX + ".weight_scale"] = ("F32", [], struct.pack("<f", 1.0))
    elif defect == "undeclared":
        records["video.low_noise.other.weight"] = ("F8_E4M3", [1, 1], b"\x38")
    elif defect == "rank":
        records[PREFIX + ".weight"] = ("F8_E4M3", [6], records[PREFIX + ".weight"][2])
    header: dict[str, object] = {}
    payload = b""
    for name, (dtype, shape, data) in records.items():
        header[name] = {
            "dtype": dtype,
            "shape": shape,
            "data_offsets": [len(payload), len(payload) + len(data)],
        }
        payload += data
    raw = json.dumps(header).encode()
    (directory / "model.safetensors").write_bytes(struct.pack("<Q", len(raw)) + raw + payload)
    return directory


def pin(root: Path) -> ModelSource:
    return ModelCatalogue(root).pin("video@fp8-e4m3-scaled")


@pytest.mark.parametrize("layout", ["legacy", "normalized"])
def test_scaled_weights_and_native_values_are_independent(tmp_path: Path, layout: str) -> None:
    package(tmp_path, layout)
    source = pin(tmp_path)
    tensors = {tensor.name: tensor for tensor in source.tensors()}
    assert set(tensors) == {PREFIX + ".weight", "video.vae.bias"}
    weight = tensors[PREFIX + ".weight"]
    assert weight.shape == (2, 3)
    assert weight.storage_format == "fp8-e4m3-scaled"
    values = torch.cat(list(source.iter_tensor(weight.id, chunk_elements=2)))
    assert values.tolist() == [0.25, -0.5, 0.125, 0.5, -0.25, 0.0]
    assert values.dtype == torch.float32 and values.device.type == "cpu"
    rows = torch.cat(list(source.iter_rows(weight.id, (1, 0, 1), chunk_elements=2)))
    assert rows.tolist() == [0.5, -0.25, 0.0, 0.25, -0.5, 0.125, 0.5, -0.25, 0.0]
    assert torch.cat(list(source.iter_tensor(tensors["video.vae.bias"].id))).tolist() == [3, -4]
    assert source.inventory()["coverage"] == "complete"


@pytest.mark.parametrize(
    "defect",
    ["missing_scale", "nonfinite", "scale_shape", "orphan", "mixed_names", "undeclared", "rank"],
)
def test_invalid_groups_fail_admission(tmp_path: Path, defect: str) -> None:
    directory = package(tmp_path, defect=defect)
    snapshot = FileSnapshot.capture(tmp_path, directory, ("model.safetensors",))
    physical = parse_header(snapshot, "model.safetensors")
    config = json.loads((directory / "config.json").read_text())
    with pytest.raises(ModelError):
        logical_locations(config, physical, snapshot)
    assert ModelCatalogue(tmp_path).list_models() == ()


def test_mutation_between_chunks_never_refreshes_a_pinned_source(tmp_path: Path) -> None:
    directory = package(tmp_path)
    source = pin(tmp_path)
    weight = next(t for t in source.tensors() if t.name == PREFIX + ".weight")
    iterator = source.iter_tensor(weight.id, chunk_elements=2)
    assert next(iterator).tolist() == [0.25, -0.5]
    path = directory / "model.safetensors"
    with path.open("r+b") as stream:
        stream.seek(-1, 2)
        stream.write(b"\x01")
    with pytest.raises(ModelError, match="Model content changed"):
        next(iterator)


def test_logical_fp8_values_reach_binary_http_and_cached_reuse(tmp_path: Path) -> None:
    root = tmp_path / "models"
    root.mkdir()
    package(root)
    with TestClient(create_app(Settings(model_root=root, cache_dir=tmp_path / "cache"))) as client:
        model = client.get("/models").json()["models"][0]["id"]
        sid = client.post("/sessions", json={"model_id": model}).json()["id"]
        inventory = client.get(f"/sessions/{sid}/tensors").json()
        weight = next(t for t in inventory["tensors"] if t["name"] == PREFIX + ".weight")
        assert weight["storage_format"] == "fp8-e4m3-scaled"
        for _ in range(2):
            response = client.get(f"/sessions/{sid}/tensors/{weight['id']}/data")
            assert response.status_code == 200
            result = frames(response.content)
            assert result[-1] == (4, b"")
            assert b"".join(data for kind, data in result if kind == 2) == struct.pack(
                "<6f", 0.25, -0.5, 0.125, 0.5, -0.25, 0.0
            )
        # An incomplete family declaration does not disable independently valid storage.
        architecture = client.get(f"/sessions/{sid}/architecture").json()
        assert architecture["status"] == "unavailable"

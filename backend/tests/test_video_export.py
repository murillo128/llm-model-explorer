"""Export safety and byte preservation; full profile acceptance uses real files."""

import hashlib
import importlib.util
import json
import struct
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from llm_model_explorer.model_files import FileSnapshot, ModelError
from llm_model_explorer.models import ModelCatalogue
from llm_model_explorer.tensor_source import parse_header


@pytest.fixture
def exporter() -> ModuleType:
    path = Path(__file__).resolve().parents[2] / "examples/video/export.py"
    spec = importlib.util.spec_from_file_location("video_export", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def source(directory: Path, *, large: bool = False) -> tuple[Path, dict[str, Any], bytes]:
    directory.mkdir()
    # The bit patterns include an FP8 negative number and signed native zero.
    payload = bytes([0x38, 0xC0, 0x30]) * (400_000 if large else 1)
    header = {
        "__metadata__": {"license": "fixture-only license notice"},
        "q.weight": {
            "dtype": "F8_E4M3",
            "shape": [1, len(payload)],
            "data_offsets": [0, len(payload)],
        },
        "norm.bias": {
            "dtype": "F16",
            "shape": [2],
            "data_offsets": [len(payload), len(payload) + 4],
        },
    }
    encoded = json.dumps(header).encode()
    raw = struct.pack("<Q", len(encoded)) + encoded + payload + struct.pack("<2e", -0.0, 3.5)
    path = directory / "original.safetensors"
    path.write_bytes(raw)
    return (
        path,
        {
            "size_bytes": len(raw),
            "sha256_lfs_oid": hashlib.sha256(raw).hexdigest(),
            "git_blob_oid": None,
            "repository": "test/fixture",
            "revision": "0" * 40,
            "path": path.name,
            "license": "fixture-only",
        },
        raw,
    )


def test_namespaced_copy_preserves_dtype_shape_bytes_and_license(
    tmp_path: Path, exporter: ModuleType
) -> None:
    path, ref, original = source(tmp_path / "input", large=True)
    item = exporter.selected(path, ref)
    assert item.digest() == ref["sha256_lfs_oid"]
    output = tmp_path / "output.safetensors"
    exporter.copy_shard(output, item, [("video.high_noise." + t.name, t) for t in item.tensors])
    snapshot = FileSnapshot.capture(tmp_path, tmp_path, (output.name,))
    copied = parse_header(snapshot, output.name)
    raw = output.read_bytes()
    for before, after in zip(item.tensors, copied, strict=True):
        assert after.name == "video.high_noise." + before.name
        assert (after.dtype, after.shape) == (before.dtype, before.shape)
        assert (
            raw[after.offset : after.offset + after.byte_length]
            == (original[before.offset : before.offset + before.byte_length])
        )
    metadata = json.loads(raw[8 : 8 + struct.unpack("<Q", raw[:8])[0]])["__metadata__"]
    assert metadata["license"] == "fixture-only license notice"
    assert path.read_bytes() == original


def test_collision_and_changed_source_are_rejected(tmp_path: Path, exporter: ModuleType) -> None:
    path, ref, _ = source(tmp_path / "input")
    item = exporter.selected(path, ref)
    records = [("same.weight", t) for t in item.tensors]
    with pytest.raises(ModelError, match="collision"):
        exporter.copy_shard(tmp_path / "collision.safetensors", item, records)
    assert not (tmp_path / "collision.safetensors").exists()
    with path.open("r+b") as stream:
        stream.seek(-1, 2)
        stream.write(b"\x00")
    with pytest.raises(ModelError, match="changed|reference role"):
        item.digest()


def test_reference_identity_and_symlink_confinement(tmp_path: Path, exporter: ModuleType) -> None:
    path, ref, _ = source(tmp_path / "input")
    # Equal size/geometry and a plausible filename cannot establish expert identity.
    with pytest.raises(ModelError, match="reference role"):
        exporter.selected(path, {**ref, "sha256_lfs_oid": "0" * 64}).digest()
    confined = tmp_path / "confined"
    confined.mkdir()
    link = confined / path.name
    link.symlink_to(path)
    with pytest.raises(ModelError):
        exporter.selected(link, ref)


@pytest.mark.parametrize("destination", ["inside", "alias", "existing"])
def test_output_cannot_modify_inputs_or_existing_destination(
    tmp_path: Path, exporter: ModuleType, destination: str
) -> None:
    path, _, original = source(tmp_path / "input")
    output = path.parent / "exported"
    if destination == "alias":
        alias = tmp_path / "alias"
        alias.symlink_to(path.parent, target_is_directory=True)
        output = alias / "exported"
    elif destination == "existing":
        output = tmp_path / "existing"
        output.mkdir()
        (output / "sentinel").write_text("keep")
    with pytest.raises(ValueError):
        exporter.export("wan-ti2v-5b", {"denoiser": path}, path.parent, output)
    assert path.read_bytes() == original
    if destination == "existing":
        assert (output / "sentinel").read_text() == "keep"


def test_atomic_publication_never_replaces_a_concurrent_target(
    tmp_path: Path, exporter: ModuleType
) -> None:
    staging, destination = tmp_path / "staging", tmp_path / "final"
    staging.mkdir()
    (staging / "sentinel").write_text("new")
    destination.mkdir()
    (destination / "sentinel").write_text("old")
    with pytest.raises(FileExistsError):
        exporter.publish(staging, destination)
    assert (destination / "sentinel").read_text() == "old"
    target = tmp_path / "published"
    exporter.publish(staging, target)
    assert (target / "sentinel").read_text() == "new" and not staging.exists()


def test_failure_after_copy_cleans_unselectable_staging(
    tmp_path: Path, exporter: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    path, ref, original = source(tmp_path / "input")
    item = exporter.selected(path, ref)
    root = tmp_path / "models"
    root.mkdir()
    output = root / "video"
    # Inject an I/O failure at the copy boundary. Profile/graph expectations are
    # never replaced: this case must fail before publication or native admission.
    monkeypatch.setattr(exporter, "inputs", lambda *_: ({"denoiser": [item]}, []))
    real_copy = exporter.copy_shard

    def fail_copy(*args: Any) -> None:
        real_copy(*args)
        assert ModelCatalogue(root).list_models() == ()
        assert not output.exists()
        raise OSError("injected disk write failure")

    monkeypatch.setattr(exporter, "copy_shard", fail_copy)
    with pytest.raises(OSError, match="injected disk write"):
        exporter.export("wan-ti2v-5b", {"denoiser": path}, path.parent, output)
    assert list(root.iterdir()) == [] and path.read_bytes() == original

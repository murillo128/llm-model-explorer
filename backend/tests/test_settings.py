import os
from pathlib import Path
from unittest.mock import Mock

import pytest
import torch

from llm_model_explorer.device import validate_device
from llm_model_explorer.settings import Settings


def test_valid_paths_preserve_models_and_create_cache(model_root: Path, tmp_path: Path) -> None:
    before = {path.name: path.read_bytes() for path in model_root.iterdir()}
    settings = Settings(model_root=model_root, cache_dir=tmp_path / "derived" / "cache")
    assert settings.device == "cpu"
    assert settings.host == "127.0.0.1"
    assert settings.port == 8000
    assert settings.cache_dir.is_dir()
    assert list(settings.cache_dir.iterdir()) == []
    assert {path.name: path.read_bytes() for path in model_root.iterdir()} == before


@pytest.mark.parametrize("root", ["missing", "file"])
def test_invalid_model_root(root: str, tmp_path: Path) -> None:
    path = tmp_path / root
    if root == "file":
        path.touch()
    with pytest.raises(ValueError):
        Settings(model_root=path, cache_dir=tmp_path / "cache")
    assert not (tmp_path / "cache").exists()


@pytest.mark.parametrize("relation", ["same", "inside", "outside"])
def test_overlapping_paths_never_write(model_root: Path, relation: str) -> None:
    cache = {"same": model_root, "inside": model_root / "cache", "outside": model_root.parent}[
        relation
    ]
    with pytest.raises(ValueError, match="must not overlap"):
        Settings(model_root=model_root, cache_dir=cache)
    assert sorted(path.name for path in model_root.iterdir()) == ["synthetic.safetensors"]


@pytest.mark.parametrize("alias_side", ["model", "cache", "cache_parent"])
def test_symlink_overlap(model_root: Path, tmp_path: Path, alias_side: str) -> None:
    alias = tmp_path / "alias"
    alias.symlink_to(model_root, target_is_directory=True)
    root = alias if alias_side == "model" else model_root
    cache = model_root / "cache" if alias_side == "model" else alias
    if alias_side == "cache_parent":
        cache = cache / "new" / "cache"
    with pytest.raises(ValueError, match="must not overlap"):
        Settings(model_root=root, cache_dir=cache)
    assert not (model_root / "new").exists()
    assert not (model_root / "cache").exists()


def test_cache_path_is_a_file(model_root: Path, tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    cache.write_text("preserve")
    with pytest.raises(ValueError, match="model/cache"):
        Settings(model_root=model_root, cache_dir=cache)
    assert cache.read_text() == "preserve"


@pytest.mark.skipif(os.name != "posix" or os.geteuid() == 0, reason="needs POSIX non-root modes")
@pytest.mark.parametrize("target", ["model", "cache", "cache_parent"])
def test_real_permission_failures(model_root: Path, tmp_path: Path, target: str) -> None:
    cache = tmp_path / "cache"
    cache.mkdir()
    restricted = model_root if target == "model" else cache
    restricted.chmod(0 if target == "model" else 0o500)
    try:
        with pytest.raises(ValueError):
            Settings(
                model_root=model_root,
                cache_dir=cache / "new" if target == "cache_parent" else cache,
            )
    finally:
        restricted.chmod(0o700)


def test_cache_actual_write_failure(
    model_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "llm_model_explorer.settings.TemporaryFile",
        Mock(side_effect=PermissionError("read-only filesystem")),
    )
    with pytest.raises(ValueError, match="read-only filesystem"):
        Settings(model_root=model_root, cache_dir=tmp_path / "cache")


@pytest.mark.parametrize("port", [0, -1, 65536, True, "8000"])
def test_invalid_port(model_root: Path, tmp_path: Path, port: object) -> None:
    with pytest.raises(ValueError, match="port"):
        Settings(model_root=model_root, cache_dir=tmp_path / "cache", port=port)  # type: ignore[arg-type]
    assert not (tmp_path / "cache").exists()


@pytest.mark.parametrize("device", ["gpu", "mps", "cpu:0", "cuda:-1", "cuda:abc", "cuda:01"])
def test_invalid_device(device: str) -> None:
    with pytest.raises(ValueError, match="device must"):
        validate_device(device)


def test_cpu_does_not_probe_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    probe = Mock(side_effect=AssertionError("CPU must not probe CUDA"))
    monkeypatch.setattr(torch.cuda, "is_available", probe)
    assert validate_device("cpu") == "cpu"
    probe.assert_not_called()


def test_unavailable_cuda_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(ValueError, match="cuda:0 is unavailable"):
        validate_device("cuda:0")


def test_explicit_cuda_selection_without_tensor_allocation(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    properties = Mock()
    monkeypatch.setattr(torch.cuda, "get_device_properties", properties)
    monkeypatch.setattr(torch, "empty", Mock(side_effect=AssertionError("no allocation")))
    assert validate_device("cuda:1") == "cuda:1"
    properties.assert_called_once_with(1)
    assert validate_device("cuda") == "cuda:0"
    with pytest.raises(ValueError, match="unavailable"):
        validate_device("cuda:2")


def test_cuda_driver_failure_is_configuration_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        torch.cuda, "is_available", Mock(side_effect=RuntimeError("driver failure"))
    )
    with pytest.raises(ValueError, match="driver failure"):
        validate_device("cuda:0")


@pytest.mark.parametrize(
    "origin",
    [
        "*",
        "null",
        "https://*.example.com",
        "https://example.com/",
        "file://ui",
        "https://u:p@ui",
        "https://ui?q=x",
        "https://ui#x",
        "https://ui:70000",
        "http://[invalid",
        "https://ui\n",
    ],
)
def test_invalid_cors_origins(model_root: Path, tmp_path: Path, origin: str) -> None:
    with pytest.raises(ValueError, match="CORS origin"):
        Settings(model_root=model_root, cache_dir=tmp_path / "cache", cors_origins=(origin,))
    assert not (tmp_path / "cache").exists()

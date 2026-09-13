from pathlib import Path

import pytest

from llm_model_explorer.settings import Settings


@pytest.fixture
def model_root(tmp_path: Path) -> Path:
    root = tmp_path / "models"
    root.mkdir()
    (root / "synthetic.safetensors").write_bytes(b"synthetic weight sentinel")
    return root


@pytest.fixture
def settings(model_root: Path, tmp_path: Path) -> Settings:
    return Settings(model_root=model_root, cache_dir=tmp_path / "cache")

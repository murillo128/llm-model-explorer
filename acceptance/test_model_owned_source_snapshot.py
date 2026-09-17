"""Temporary source snapshot for the model-owned descriptor development PR.

The connected editing environment cannot clone this public repository. Capture
only tracked project source in the existing external acceptance artifact, never
credentials, model payloads, installed dependencies, or runner state. Remove this
bootstrap helper before the feature PR becomes ready for review.
"""

import os
import subprocess
import zipfile
from pathlib import Path

import pytest


def test_capture_public_project_sources() -> None:
    destination = os.environ.get("LMEX_EVIDENCE_DIR")
    if not destination:
        pytest.skip("source snapshot belongs to external acceptance evidence")
    root = Path(__file__).resolve().parents[1]
    tracked = (
        subprocess.run(["git", "ls-files", "-z"], cwd=root, check=True, capture_output=True)
        .stdout.decode()
        .split("\0")
    )
    suffixes = {
        ".py",
        ".json",
        ".yaml",
        ".yml",
        ".ts",
        ".tsx",
        ".md",
        ".sh",
        ".toml",
        ".lock",
        ".txt",
        ".mjs",
        ".js",
        ".css",
        ".html",
    }
    output = Path(destination) / "model-owned-source-snapshot.zip"
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in sorted(tracked):
            if not name or Path(name).suffix not in suffixes:
                continue
            source = root / name
            if source.is_symlink() or not source.is_file():
                continue
            if source.stat().st_size > 2_000_000:
                continue
            archive.write(source, name)
    assert output.stat().st_size > 0

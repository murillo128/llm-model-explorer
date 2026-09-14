"""Observe numeric publication independently of startup architecture artifacts."""

import json
from pathlib import Path


def numeric_cache_entries(root: Path) -> list[Path]:
    entries = []
    for entry in root.iterdir():
        manifest = entry / "manifest.json"
        if manifest.is_file() and json.loads(manifest.read_text()).get("format") == 2:
            continue
        entries.append(entry)
    return entries


def numeric_manifests(root: Path) -> list[Path]:
    return [
        entry / "manifest.json"
        for entry in numeric_cache_entries(root)
        if (entry / "manifest.json").is_file()
    ]

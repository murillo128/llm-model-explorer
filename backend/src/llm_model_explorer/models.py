"""Concrete local Hugging Face catalogue. No model loaders or provider abstraction."""

import logging
import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from pydantic import BaseModel, ConfigDict

from .model_files import FileSnapshot, ModelError, confined, invalid, read_json
from .quantized_inventory import encoding, logical_locations
from .tensor_source import (
    MAX_SAFE_INTEGER,
    ModelSource,
    PhysicalTensor,
    TensorDescriptor,
    TensorLocation,
    parse_header,
)

logger = logging.getLogger(__name__)
_NAME = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.-]*\Z")


class ModelSummary(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str
    display_name: str
    architectures: tuple[str, ...]
    tokenizer_available: bool
    model_type: str | None = None
    size_bytes: int | None = None


def _logical_name(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    parts = value.split("/")
    if len(parts) not in {1, 2} or any(not _NAME.fullmatch(p) or ".." in p for p in parts):
        return None
    return value


def _identity(config: dict[str, object], directory: Path) -> tuple[str, str]:
    identity = None
    for key in ("_name_or_path", "name_or_path"):
        candidate = _logical_name(config.get(key))
        # Existing relative paths are not evidence of a repository identity.
        if candidate and not (directory / candidate).exists() and not Path(candidate).exists():
            identity = candidate
            break
    display = identity or directory.name
    revision = config.get("_commit_hash", config.get("revision"))
    if isinstance(revision, str) and _NAME.fullmatch(revision) and ".." not in revision:
        return f"{display}@{revision}", display
    return display, display


def _shard_name(value: object) -> str:
    if not isinstance(value, str) or "\\" in value:
        raise invalid("Invalid shard reference.")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or ".." in path.parts
        or ":" in value
        or str(path) != value
        or path.suffix != ".safetensors"
    ):
        raise invalid("Invalid shard reference.")
    return value


@dataclass(frozen=True)
class CatalogueEntry:
    summary: ModelSummary
    _snapshot: FileSnapshot
    _locations: tuple[TensorLocation, ...]
    _physical: tuple[PhysicalTensor, ...]

    def physical_tensors(self) -> tuple[PhysicalTensor, ...]:
        self._snapshot.check()
        return self._physical

    def tensors(self) -> tuple[TensorDescriptor, ...]:
        """Inventory descriptors, without reading tensor payloads."""
        self._snapshot.check()
        return tuple(location.descriptor for location in self._locations)

    def pin(self) -> ModelSource:
        return ModelSource(
            self.summary.id,
            self._snapshot.fingerprint(),
            self._snapshot,
            self._locations,
            self._physical,
        )


class ModelCatalogue:
    def __init__(self, root: Path) -> None:
        # Settings has already resolved/validated root. Construction performs no I/O.
        self._root = root

    def inspect_directory(self, directory: Path) -> CatalogueEntry:
        """Admit one exact local package without scanning sibling models."""
        return self._inspect(directory)

    def _inspect(self, directory: Path) -> CatalogueEntry:
        confined(self._root, directory)
        # Capture config/index before parsing, so a race cannot pin stale descriptors.
        initial = FileSnapshot.capture(self._root, directory, ())
        config = read_json(self._root, directory / "config.json")
        if not isinstance(config.get("model_type"), str) or not config["model_type"]:
            raise invalid("Missing Hugging Face model type.")
        encoding(config)
        indexes = [name for name, _ in initial.files if name.endswith(".safetensors.index.json")]
        if len(indexes) > 1:
            raise invalid("Multiple safetensors indexes are ambiguous.")
        weight_map: dict[str, str] | None = None
        if indexes:
            index = read_json(self._root, directory / indexes[0])
            raw_map = index.get("weight_map")
            if not isinstance(raw_map, dict) or not raw_map:
                raise invalid("Invalid safetensors shard index.")
            weight_map = {key: _shard_name(value) for key, value in raw_map.items()}
            shards = tuple(sorted(set(weight_map.values())))
        else:
            shards = tuple(name for name, _ in initial.files if name.endswith(".safetensors"))
        if not shards:
            raise invalid("No safetensors weights found.")
        snapshot = FileSnapshot.capture(self._root, directory, shards)
        initial.check()
        locations: dict[str, PhysicalTensor] = {}
        for shard in shards:
            for location in parse_header(snapshot, shard):
                name = location.name
                if name in locations:
                    raise invalid("Duplicate tensor name across shards.")
                locations[name] = location
        if weight_map is not None and weight_map != {
            name: location.file for name, location in locations.items()
        }:
            raise invalid("Shard index does not match the available tensors.")
        snapshot.check()
        identity, display = _identity(config, directory)
        raw_architectures = config.get("architectures", [])
        architectures = (
            tuple(raw_architectures)
            if isinstance(raw_architectures, list)
            and all(isinstance(a, str) and _NAME.fullmatch(a) for a in raw_architectures)
            else ()
        )
        model_type = config["model_type"]
        names = {name for name, _ in snapshot.files}
        tokenizer_available = bool(
            {"tokenizer.json", "tokenizer.model", "spiece.model", "vocab.txt"} & names
            or {"vocab.json", "merges.txt"} <= names
        )
        size = sum(stamp.size for _, stamp in snapshot.files)
        summary = ModelSummary(
            id=identity,
            display_name=display,
            architectures=architectures,
            model_type=model_type
            if isinstance(model_type, str) and _NAME.fullmatch(model_type)
            else None,
            tokenizer_available=tokenizer_available,
            size_bytes=size if size <= MAX_SAFE_INTEGER else None,
        )
        physical = tuple(locations[k] for k in sorted(locations))
        logical = logical_locations(config, physical)
        snapshot.check()
        return CatalogueEntry(summary, snapshot, logical, physical)

    def discover(self) -> tuple[CatalogueEntry, ...]:
        """Fresh metadata scan; only identity collisions require full streamed hashing."""
        entries: dict[str, CatalogueEntry] = {}
        try:
            candidates = sorted(self._root.iterdir())
            for directory in candidates:
                if not directory.is_dir():
                    continue
                try:
                    confined(self._root, directory)
                    if not (directory / "config.json").exists():
                        continue
                    entry = self._inspect(directory)
                except (OSError, RuntimeError, ValueError, ModelError) as exc:
                    logger.warning("Skipping invalid local model candidate %s: %s", directory, exc)
                    continue
                identity = entry.summary.id
                if identity in entries:
                    if entry.pin().fingerprint != entries[identity].pin().fingerprint:
                        logger.error("Ambiguous local model identity: %s", identity)
                        raise ModelError(
                            "validation_error",
                            "Ambiguous model identity has different local contents.",
                        )
                else:
                    entries[identity] = entry
            return tuple(entries[k] for k in sorted(entries))
        except OSError as exc:
            logger.exception("Unable to scan configured model root")
            raise ModelError("internal_error", "Unable to discover local models.", 500) from exc

    def list_models(self) -> tuple[ModelSummary, ...]:
        return tuple(entry.summary for entry in self.discover())

    def pin(self, model_id: str) -> ModelSource:
        for entry in self.discover():
            if entry.summary.id == model_id:
                return entry.pin()
        raise ModelError("model_not_found", "Unknown model.", 404)

"""Concrete local Hugging Face catalogue. No model loaders or provider abstraction."""

import logging
import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from pydantic import BaseModel, ConfigDict

from .model_files import FileSnapshot, ModelError, confined, invalid, read_json
from .peft_adapters import AdapterSpec, bind_adapter, validate_adapter
from .quantized_inventory import encoding, logical_locations
from .tensor_source import (
    MAX_SAFE_INTEGER,
    ModelSource,
    PhysicalTensor,
    TensorDescriptor,
    TensorLocation,
    combined_fingerprint,
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


def _identity_details(
    config: dict[str, object], directory: Path
) -> tuple[str, str, str | None]:
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
        return f"{display}@{revision}", display, identity
    return display, display, identity


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
    _additional_snapshots: tuple[FileSnapshot, ...] = ()
    _composition_semantics: str | None = None

    def physical_tensors(self) -> tuple[PhysicalTensor, ...]:
        for snapshot in (self._snapshot, *self._additional_snapshots):
            snapshot.check()
        return self._physical

    def tensors(self) -> tuple[TensorDescriptor, ...]:
        """Inventory descriptors, without reading tensor payloads."""
        for snapshot in (self._snapshot, *self._additional_snapshots):
            snapshot.check()
        return tuple(location.descriptor for location in self._locations)

    def pin(self) -> ModelSource:
        snapshots = (self._snapshot, *self._additional_snapshots)
        fingerprints = tuple(snapshot.fingerprint() for snapshot in snapshots)
        for snapshot in snapshots:
            snapshot.check()
        return ModelSource(
            self.summary.id,
            combined_fingerprint(
                fingerprints,
                self._composition_semantics,
            ),
            self._snapshot,
            self._locations,
            self._physical,
            self._additional_snapshots,
            self._composition_semantics,
        )


@dataclass(frozen=True)
class _BaseCandidate:
    entry: CatalogueEntry
    config: dict[str, object]
    metadata_identity: str | None
    directory_name: str


@dataclass(frozen=True)
class _AdapterCandidate:
    identity: str
    display_name: str
    spec: AdapterSpec
    snapshot: FileSnapshot
    physical: tuple[PhysicalTensor, ...]


class ModelCatalogue:
    def __init__(self, root: Path) -> None:
        # Settings has already resolved/validated root. Construction performs no I/O.
        self._root = root

    def _inspect_base(self, directory: Path) -> _BaseCandidate:
        confined(self._root, directory)
        # Capture config/index before parsing, so a race cannot pin stale descriptors.
        initial = FileSnapshot.capture(self._root, directory, ())
        config = read_json(self._root, directory / "config.json")
        if not isinstance(config.get("model_type"), str) or not config["model_type"]:
            raise invalid("Missing Hugging Face model type.")
        layout = encoding(config)
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
        identity, display, metadata_identity = _identity_details(config, directory)
        if layout != "native":
            identity = f"{identity}@{layout}"
            display = f"{display} ({layout})"
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
        return _BaseCandidate(
            CatalogueEntry(summary, snapshot, logical, physical),
            config,
            metadata_identity,
            directory.name,
        )

    def _inspect_adapter(self, directory: Path) -> _AdapterCandidate:
        confined(self._root, directory)
        initial = FileSnapshot.capture(self._root, directory, ())
        config = read_json(self._root, directory / "adapter_config.json")
        indexes = [name for name, _ in initial.files if name.endswith(".safetensors.index.json")]
        direct_weights = [name for name, _ in initial.files if name.endswith(".safetensors")]
        if len(indexes) > 1:
            raise invalid("Multiple adapter safetensors indexes are ambiguous.")
        weight_map: dict[str, str] | None = None
        if indexes:
            index = read_json(self._root, directory / indexes[0])
            raw_map = index.get("weight_map")
            if not isinstance(raw_map, dict) or not raw_map:
                raise invalid("Invalid adapter safetensors shard index.")
            if any(not isinstance(name, str) or not name for name in raw_map):
                raise invalid("Invalid adapter tensor names in shard index.")
            weight_map = {name: _shard_name(value) for name, value in raw_map.items()}
            shards = tuple(sorted(set(weight_map.values())))
            if set(direct_weights) - set(shards):
                raise invalid("Adapter directory contains an unindexed safetensors payload.")
        else:
            if direct_weights != ["adapter_model.safetensors"]:
                raise invalid("Adapter requires one conventional safetensors payload or index.")
            shards = ("adapter_model.safetensors",)
        if not shards:
            raise invalid("No adapter safetensors weights found.")

        snapshot = FileSnapshot.capture(self._root, directory, shards)
        initial.check()
        locations: dict[str, PhysicalTensor] = {}
        for shard in shards:
            for location in parse_header(snapshot, shard):
                if location.name in locations:
                    raise invalid("Duplicate adapter tensor name across shards.")
                locations[location.name] = location
        if weight_map is not None and weight_map != {
            name: location.file for name, location in locations.items()
        }:
            raise invalid("Adapter shard index does not match its available tensors.")
        snapshot.check()
        physical = tuple(locations[name] for name in sorted(locations))
        spec = validate_adapter(config, physical)
        identity, display, _ = _identity_details(config, directory)
        snapshot.check()
        return _AdapterCandidate(identity, display, spec, snapshot, physical)

    @staticmethod
    def _matches_upstream(base: _BaseCandidate, adapter: _AdapterCandidate) -> bool:
        upstream = adapter.spec.upstream_name
        if base.metadata_identity and "/" in base.metadata_identity:
            return base.metadata_identity == upstream
        return base.directory_name == upstream.rsplit("/", 1)[-1]

    @staticmethod
    def _is_causal_lm(config: dict[str, object]) -> bool:
        architectures = config.get("architectures")
        return (
            isinstance(architectures, list)
            and len(architectures) == 1
            and isinstance(architectures[0], str)
            and architectures[0].endswith("ForCausalLM")
        )

    def _compose(
        self, base: _BaseCandidate, adapter: _AdapterCandidate
    ) -> CatalogueEntry | None:
        if not self._matches_upstream(base, adapter) or not self._is_causal_lm(base.config):
            return None
        try:
            base.entry._snapshot.check()
            adapter.snapshot.check()
            adapter_locations = bind_adapter(
                adapter.spec,
                adapter.identity,
                base.config,
                base.entry._locations,
                adapter.snapshot,
            )
            base.entry._snapshot.check()
            adapter.snapshot.check()
        except ModelError as exc:
            if exc.code == "model_content_changed":
                raise
            logger.warning("Skipping incompatible local PEFT composition: %s", exc)
            return None
        except ValueError as exc:
            logger.warning("Skipping incompatible local PEFT composition: %s", exc)
            return None
        base_entry = base.entry
        model_id = f"{base_entry.summary.id}+peft-lora:{adapter.identity}"
        adapter_size = sum(stamp.size for _, stamp in adapter.snapshot.files)
        base_size = base_entry.summary.size_bytes
        size = None if base_size is None else base_size + adapter_size
        summary = ModelSummary(
            id=model_id,
            display_name=f"{base_entry.summary.display_name} + LoRA {adapter.display_name}",
            architectures=base_entry.summary.architectures,
            tokenizer_available=base_entry.summary.tokenizer_available,
            model_type=base_entry.summary.model_type,
            size_bytes=size if size is not None and size <= MAX_SAFE_INTEGER else None,
        )
        return CatalogueEntry(
            summary,
            base_entry._snapshot,
            (*base_entry._locations, *adapter_locations),
            (*base_entry._physical, *adapter.physical),
            (adapter.snapshot,),
            "peft-lora-causal-lm-alpha-over-r-v1",
        )

    @staticmethod
    def _insert(entries: dict[str, CatalogueEntry], entry: CatalogueEntry) -> None:
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

    def discover(self) -> tuple[CatalogueEntry, ...]:
        """Fresh metadata scan; only identity collisions require full streamed hashing."""
        entries: dict[str, CatalogueEntry] = {}
        try:
            candidates = sorted(self._root.iterdir())
            bases: list[_BaseCandidate] = []
            adapters: list[_AdapterCandidate] = []
            for directory in candidates:
                if not directory.is_dir():
                    continue
                try:
                    confined(self._root, directory)
                    if (directory / "adapter_config.json").exists():
                        adapters.append(self._inspect_adapter(directory))
                    elif (directory / "config.json").exists():
                        bases.append(self._inspect_base(directory))
                except (OSError, RuntimeError, ValueError, ModelError) as exc:
                    logger.warning("Skipping invalid local model candidate %s: %s", directory, exc)
            for base in bases:
                self._insert(entries, base.entry)
            for adapter in adapters:
                for base in bases:
                    composition = self._compose(base, adapter)
                    if composition is not None:
                        self._insert(entries, composition)
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

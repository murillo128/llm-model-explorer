"""Filesystem artifacts and bounded live readers (blocking I/O; use BlockingWork).

Publication uses POSIX advisory directory locks and same-filesystem rename. Locks
cover only validation/publication, never production or consumer lifetimes.
"""

from __future__ import annotations

import errno
import fcntl
import hashlib
import json
import os
import re
import shutil
import stat
import tempfile
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from threading import RLock
from types import TracebackType
from typing import BinaryIO, Self

from .architecture_analysis import BindingContext, Producer, parse_graph
from .architecture_analysis.core import Scope, identity
from .architecture_analysis.records import ArchitectureGraph
from .architecture_analysis.validation import MAX_BYTES, json_chunks, preflight

_MANIFEST_LIMIT = 64 * 1024
_DIGEST = re.compile(r"[0-9a-f]{64}")


def _open_regular(directory: int, filename: str) -> BinaryIO | None:
    """Transfer ownership only after validating and wrapping a raw descriptor."""
    descriptor = os.open(filename, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory)
    payload: BinaryIO | None = None
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            return None
        payload = os.fdopen(descriptor, "rb", buffering=0)
        return payload
    finally:
        if payload is None:
            os.close(descriptor)


def _canonical(value: object) -> str:
    # Reject non-JSON values and non-string mapping keys, including nested ones.
    def validate(item: object) -> None:
        if isinstance(item, dict):
            for key, child in item.items():
                if not isinstance(key, str):
                    raise ValueError("artifact parameter keys must be strings")
                validate(child)
        elif isinstance(item, list):
            for child in item:
                validate(child)
        elif item is not None and type(item) not in (str, int, float, bool):
            raise ValueError("artifact parameters must be JSON values")

    validate(value)
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


@dataclass(frozen=True, init=False)
class ArtifactSpec:
    """Immutable snapshot of computation identity and expected representation.

    Parameters contain computation inputs only: callers must exclude session IDs,
    local paths, timestamps and UI styling. Shape may have arbitrary rank.
    """

    canonical: str
    key: str
    expected_bytes: int

    def __init__(
        self,
        *,
        model_fingerprint: str,
        source: str,
        operation: str,
        parameters: Mapping[str, object],
        dtype: str,
        layout: str,
        shape: tuple[int, ...],
        expected_bytes: int,
        producer: str,
    ) -> None:
        identity = {
            "model_fingerprint": model_fingerprint,
            "source": source,
            "operation": operation,
            "dtype": dtype,
            "layout": layout,
            "producer": producer,
        }
        if any(not isinstance(value, str) or not value for value in identity.values()):
            raise ValueError("artifact identity fields must be nonempty strings")
        if type(expected_bytes) is not int or expected_bytes < 0:
            raise ValueError("expected_bytes must be a nonnegative integer")
        if any(type(dimension) is not int or dimension < 0 for dimension in shape):
            raise ValueError("shape dimensions must be nonnegative integers")
        canonical = _canonical(
            dict(
                identity, parameters=dict(parameters), shape=list(shape), byte_length=expected_bytes
            )
        )
        if len(canonical.encode()) > _MANIFEST_LIMIT // 2:
            raise ValueError("artifact specification is too large")
        object.__setattr__(self, "canonical", canonical)
        object.__setattr__(self, "key", hashlib.sha256(canonical.encode()).hexdigest())
        object.__setattr__(self, "expected_bytes", expected_bytes)


class ArtifactConflict(RuntimeError):
    """A competing producer committed different bytes for the same identity."""


@dataclass(frozen=True, init=False)
class ArchitectureArtifactSpec:
    """Precomputable graph identity; neither output length nor public envelopes.

    Empty options use the existing Producer/GraphBuilder identity. For meaningful
    options, construction must use graph_id before allocating graph-local IDs.
    Serializer changes invalidate structured entries only.
    """

    canonical: str
    key: str
    graph_id: str

    def __init__(
        self,
        *,
        model_fingerprint: str,
        producer: Producer,
        scope: Scope,
        analysis_options: Mapping[str, object],
        serializer_revision: str = "architecture-json-1",
    ) -> None:
        fields = {
            "kind": "architecture_graph",
            "media_type": "application/json",
            "model_fingerprint": model_fingerprint,
            "description": producer.description,
            "description_revision": producer.revision,
            "source_revision": producer.source_revision,
            "analyzer_revision": producer.analyzer_revision,
            "schema_revision": producer.schema_revision,
            "serializer_revision": serializer_revision,
            "scope": scope,
        }
        if any(not isinstance(value, str) or not value for value in fields.values()):
            raise ValueError("architecture identity fields must be nonempty strings")
        if scope not in ("language_model", "visual_encoder_predictor"):
            raise ValueError("invalid architecture scope")
        options = _canonical(dict(analysis_options))
        canonical = _canonical(dict(fields, analysis_options=json.loads(options)))
        if len(canonical.encode()) > _MANIFEST_LIMIT // 2:
            raise ValueError("artifact specification is too large")
        graph_id = producer.graph_id(model_fingerprint, scope)
        if options != "{}":
            graph_id = identity("architecture-graph-options", graph_id, options)
        object.__setattr__(self, "canonical", canonical)
        object.__setattr__(self, "key", hashlib.sha256(canonical.encode()).hexdigest())
        object.__setattr__(self, "graph_id", graph_id)


def _validate_graph_payload(
    payload: BinaryIO,
    spec: ArchitectureArtifactSpec,
    context: BindingContext,
    length: int,
    digest: str,
) -> None:
    """Read at most the contract bound, including on corrupt or growing files."""
    if not 0 < length <= MAX_BYTES:
        raise ValueError("architecture payload exceeds its byte bounds")
    raw = payload.read(length + 1)
    if len(raw) != length or hashlib.sha256(raw).hexdigest() != digest:
        raise ValueError("architecture payload length or digest changed")
    graph = parse_graph(json.loads(raw.decode("utf-8")), context)
    if graph.graph_id != spec.graph_id or graph.scope != json.loads(spec.canonical)["scope"]:
        raise ValueError("architecture graph identity differs from its inputs")
    # A single serializer gives same-key producers a deterministic byte identity.
    canonical_digest = hashlib.sha256()
    for chunk in json_chunks(graph.document()):
        canonical_digest.update(chunk)
    if canonical_digest.hexdigest() != digest:
        raise ValueError("architecture payload is not canonical JSON")
    payload.seek(0)


@dataclass
class _Progress:
    available: int = 0
    complete: bool = False
    aborted: bool = False
    lock: RLock = field(default_factory=RLock)


class ArtifactReader:
    """Independent cursor over flushed bytes; close explicitly or use a context.

    An empty read on a live producer means 'nothing available yet', not EOF.
    Check complete/aborted to distinguish terminal states. Polling/wakeup policy
    belongs to the operation runtime. Existing handles survive rename and abort.
    """

    def __init__(
        self, payload: BinaryIO, spec: ArtifactSpec | ArchitectureArtifactSpec, progress: _Progress
    ) -> None:
        self._payload = payload
        self.spec = spec
        self._progress = progress

    @property
    def available_bytes(self) -> int:
        with self._progress.lock:
            return self._progress.available

    @property
    def complete(self) -> bool:
        with self._progress.lock:
            return self._progress.complete

    @property
    def aborted(self) -> bool:
        with self._progress.lock:
            return self._progress.aborted

    def read_available(self, max_bytes: int) -> bytes:
        """Read at most max_bytes, never expected-but-unwritten bytes."""
        if type(max_bytes) is not int or max_bytes <= 0:
            raise ValueError("max_bytes must be a positive integer")
        with self._progress.lock:
            size = min(max_bytes, self._progress.available - self._payload.tell())
            data = self._payload.read(size)
            if len(data) != size:
                raise OSError("artifact payload was truncated while a reader was attached")
            return data

    def close(self) -> None:
        self._payload.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()


class ArtifactStore:
    """Shared disposable cache; no model access, scheduler, or in-memory payloads.

    The configured paths must remain stable during use. Abandoned .tmp-* entries
    are ignored, not resumed or removed at startup (another process may own them).
    """

    def __init__(self, cache_dir: Path, *, model_root: Path) -> None:
        self.root = cache_dir.resolve()
        model_root = model_root.resolve()
        if self.root.is_relative_to(model_root) or model_root.is_relative_to(self.root):
            raise ValueError("cache directory and model root must not overlap")
        self.root.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def _publication_lock(self) -> Iterator[None]:
        # Separate opens make flock serialize distinct threads/store instances too.
        descriptor = os.open(self.root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            yield
        finally:
            os.close(descriptor)

    def _open_valid(
        self, spec: ArtifactSpec | ArchitectureArtifactSpec, context: BindingContext | None = None
    ) -> tuple[BinaryIO, str] | None:
        """Open fixed names relative to a non-symlink directory, validating lengths."""
        if not _DIGEST.fullmatch(spec.key):
            raise ValueError("invalid artifact key")
        try:
            directory = os.open(self.root / spec.key, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        except OSError as exc:
            if exc.errno in (errno.ENOENT, errno.ENOTDIR, errno.ELOOP):
                return None
            raise
        payload: BinaryIO | None = None
        try:
            manifest = _open_regular(directory, "manifest.json")
            if manifest is None:
                return None
            with manifest:
                raw = manifest.read(_MANIFEST_LIMIT + 1)
            if len(raw) > _MANIFEST_LIMIT:
                return None
            metadata = json.loads(raw)
            if not isinstance(metadata, dict) or set(metadata) != {
                "format",
                "key",
                "spec",
                "payload",
                "byte_length",
                "sha256",
            }:
                return None
            structured = isinstance(spec, ArchitectureArtifactSpec)
            length = metadata["byte_length"]
            if (
                type(metadata["format"]) is not int
                or metadata["format"] != (2 if structured else 1)
                or metadata["key"] != spec.key
                or _canonical(metadata["spec"]) != spec.canonical
                or metadata["payload"] != "payload.bin"
                or type(length) is not int
                or (isinstance(spec, ArtifactSpec) and length != spec.expected_bytes)
                or (structured and not 0 < length <= MAX_BYTES)
                or not isinstance(metadata["sha256"], str)
                or not _DIGEST.fullmatch(metadata["sha256"])
            ):
                return None
            payload = _open_regular(directory, "payload.bin")
            if payload is None:
                return None
            info = os.fstat(payload.fileno())
            if info.st_size != length:
                payload.close()
                return None
            if isinstance(spec, ArchitectureArtifactSpec):
                if context is None:
                    raise ValueError("architecture lookup requires binding context")
                _validate_graph_payload(payload, spec, context, length, metadata["sha256"])
            return payload, metadata["sha256"]
        except (OSError, ValueError, TypeError, RecursionError) as exc:
            if payload is not None:
                payload.close()
            if isinstance(exc, OSError) and exc.errno not in (
                errno.ENOENT,
                errno.ENOTDIR,
                errno.ELOOP,
            ):
                raise
            return None
        finally:
            os.close(directory)

    def lookup(self, spec: ArtifactSpec) -> ArtifactReader | None:
        valid = self._open_valid(spec)
        if valid is None:
            return None
        payload, _ = valid
        return ArtifactReader(payload, spec, _Progress(spec.expected_bytes, complete=True))

    def begin_write(self, spec: ArtifactSpec) -> ArtifactWriter:
        return ArtifactWriter(self, spec)

    def lookup_graph(
        self, spec: ArchitectureArtifactSpec, context: BindingContext
    ) -> ArtifactReader | None:
        """Validate prepared bytes without analysis or an expected output length."""
        valid = self._open_valid(spec, context)
        if valid is None:
            return None
        payload, _ = valid
        return ArtifactReader(
            payload, spec, _Progress(os.fstat(payload.fileno()).st_size, complete=True)
        )

    def begin_graph_write(
        self,
        spec: ArchitectureArtifactSpec,
        context: BindingContext,
        *,
        check_source: Callable[[], None],
    ) -> ArchitectureArtifactWriter:
        """The caller owns source hashing/cancellation; check_source must raise on change.

        Pass a final pinned-source check (including rehash when required) that
        also checks cancellation. Blocking work must settle before context exit.
        """
        return ArchitectureArtifactWriter(self, spec, context, check_source=check_source)


class ArtifactWriter:
    """One producer; use a context to abort automatically on errors/cancellation."""

    def __init__(self, store: ArtifactStore, spec: ArtifactSpec | ArchitectureArtifactSpec) -> None:
        self.store = store
        self.spec = spec
        self._progress = _Progress()
        self._hash = hashlib.sha256()
        store.root.mkdir(parents=True, exist_ok=True)
        self._temporary = Path(tempfile.mkdtemp(prefix=".tmp-", dir=store.root))
        try:
            self._payload = (self._temporary / "payload.bin").open("xb", buffering=0)
        except BaseException:
            shutil.rmtree(self._temporary)
            raise

    @property
    def available_bytes(self) -> int:
        with self._progress.lock:
            return self._progress.available

    def _require_writing(self) -> None:
        if self._progress.complete or self._progress.aborted:
            raise RuntimeError("artifact writer is already terminal")

    def append(self, data: bytes) -> None:
        with self._progress.lock:
            self._require_writing()
            try:
                limit = (
                    self.spec.expected_bytes if isinstance(self.spec, ArtifactSpec) else MAX_BYTES
                )
                if self._progress.available + len(data) > limit:
                    raise ValueError("payload exceeds declared length")
                view = memoryview(data)
                while view:
                    written = self._payload.write(view)
                    if not written:
                        raise OSError("payload write made no progress")
                    view = view[written:]
                self._payload.flush()
                self._hash.update(data)
                self._progress.available += len(data)
            except BaseException:
                self.abort()
                raise

    def open_reader(self) -> ArtifactReader:
        with self._progress.lock:
            self._require_writing()
            return ArtifactReader(
                (self._temporary / "payload.bin").open("rb", buffering=0),
                self.spec,
                self._progress,
            )

    def _validate_payload(self) -> None:
        if not isinstance(self.spec, ArtifactSpec):
            raise TypeError("structured artifacts require begin_graph_write")
        if self._progress.available != self.spec.expected_bytes:
            raise ValueError("payload is incomplete")
        if os.fstat(self._payload.fileno()).st_size != self._progress.available:
            raise ValueError("payload length changed externally")

    def _existing(self) -> tuple[BinaryIO, str] | None:
        return self.store._open_valid(self.spec)

    def _check_publication(self) -> None:
        """Numeric producers already check pinned sources in their operation runtime."""

    def commit(self) -> None:
        with self._progress.lock:
            self._require_writing()
            try:
                self._validate_payload()
                os.fsync(self._payload.fileno())
                digest = self._hash.hexdigest()
                metadata = {
                    "format": 1 if isinstance(self.spec, ArtifactSpec) else 2,
                    "key": self.spec.key,
                    "spec": json.loads(self.spec.canonical),
                    "payload": "payload.bin",
                    "byte_length": self._progress.available,
                    "sha256": digest,
                }
                with (self._temporary / "manifest.json").open("xb") as manifest:
                    manifest.write(_canonical(metadata).encode())
                    manifest.flush()
                    os.fsync(manifest.fileno())
                temporary_fd = os.open(self._temporary, os.O_RDONLY | os.O_DIRECTORY)
                try:
                    os.fsync(temporary_fd)
                finally:
                    os.close(temporary_fd)
                with self.store._publication_lock():
                    existing = self._existing()
                    if existing is not None:
                        payload, existing_digest = existing
                        payload.close()
                        if digest != existing_digest:
                            raise ArtifactConflict("same artifact key produced different bytes")
                        self._check_publication()
                        shutil.rmtree(self._temporary)
                    else:
                        self._check_publication()
                        target = self.store.root / self.spec.key
                        # Only invalid entries are removed, under the publication lock.
                        # Never follow links from corrupt cache entries.
                        if target.is_symlink() or (target.exists() and not target.is_dir()):
                            target.unlink()
                        elif target.exists():
                            shutil.rmtree(target)
                        os.rename(self._temporary, target)
                        root_fd = os.open(self.store.root, os.O_RDONLY | os.O_DIRECTORY)
                        try:
                            os.fsync(root_fd)
                        finally:
                            os.close(root_fd)
                    self._progress.complete = True
                self._payload.close()
            except BaseException:
                self.abort()
                raise

    def abort(self) -> None:
        with self._progress.lock:
            if self._progress.complete:
                return
            self._progress.aborted = True
            self._payload.close()
            if self._temporary.exists():
                shutil.rmtree(self._temporary)

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.abort()


class ArchitectureArtifactWriter(ArtifactWriter):
    """Full graph publication using the numeric store's writer/reader ownership.

    append may accept incremental serialization, but commit always validates the
    complete bytes. write_graph is the bounded, deterministic typed entry point.
    """

    spec: ArchitectureArtifactSpec

    def __init__(
        self,
        store: ArtifactStore,
        spec: ArchitectureArtifactSpec,
        context: BindingContext,
        *,
        check_source: Callable[[], None],
    ) -> None:
        self._context = context
        self._check_source = check_source
        super().__init__(store, spec)

    def write_graph(self, graph: ArchitectureGraph) -> None:
        with self._progress.lock:
            self._require_writing()
            try:
                if self.available_bytes:
                    raise ValueError("graph serialization has already started")
                preflight(graph)
                # Buffer small encoder fragments, without retaining the whole output.
                buffer = bytearray()
                for chunk in json_chunks(graph.document()):
                    buffer.extend(chunk)
                    if len(buffer) >= 64 * 1024:
                        self.append(bytes(buffer))
                        buffer.clear()
                if buffer:
                    self.append(bytes(buffer))
            except BaseException:
                self.abort()
                raise

    def _validate_payload(self) -> None:
        if os.fstat(self._payload.fileno()).st_size != self.available_bytes:
            raise ValueError("payload length changed externally")
        directory = os.open(self._temporary, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            payload = _open_regular(directory, "payload.bin")
            if payload is None:
                raise ValueError("architecture payload must be a regular file")
            with payload:
                if not os.path.samestat(
                    os.fstat(payload.fileno()), os.fstat(self._payload.fileno())
                ):
                    raise ValueError("architecture payload was replaced")
                _validate_graph_payload(
                    payload, self.spec, self._context, self.available_bytes, self._hash.hexdigest()
                )
        finally:
            os.close(directory)

    def _existing(self) -> tuple[BinaryIO, str] | None:
        return self.store._open_valid(self.spec, self._context)

    def _check_publication(self) -> None:
        self._check_source()

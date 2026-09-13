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
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from threading import RLock
from types import TracebackType
from typing import BinaryIO, Self

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

    def __init__(self, payload: BinaryIO, spec: ArtifactSpec, progress: _Progress) -> None:
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

    def _open_valid(self, spec: ArtifactSpec) -> tuple[BinaryIO, str] | None:
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
            if (
                type(metadata["format"]) is not int
                or metadata["format"] != 1
                or metadata["key"] != spec.key
                or _canonical(metadata["spec"]) != spec.canonical
                or metadata["payload"] != "payload.bin"
                or type(metadata["byte_length"]) is not int
                or metadata["byte_length"] != spec.expected_bytes
                or not isinstance(metadata["sha256"], str)
                or not _DIGEST.fullmatch(metadata["sha256"])
            ):
                return None
            payload = _open_regular(directory, "payload.bin")
            if payload is None:
                return None
            info = os.fstat(payload.fileno())
            if info.st_size != spec.expected_bytes:
                payload.close()
                return None
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


class ArtifactWriter:
    """One producer; use a context to abort automatically on errors/cancellation."""

    def __init__(self, store: ArtifactStore, spec: ArtifactSpec) -> None:
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
                if self._progress.available + len(data) > self.spec.expected_bytes:
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

    def commit(self) -> None:
        with self._progress.lock:
            self._require_writing()
            try:
                if self._progress.available != self.spec.expected_bytes:
                    raise ValueError("payload is incomplete")
                if os.fstat(self._payload.fileno()).st_size != self.spec.expected_bytes:
                    raise ValueError("payload length changed externally")
                os.fsync(self._payload.fileno())
                digest = self._hash.hexdigest()
                metadata = {
                    "format": 1,
                    "key": self.spec.key,
                    "spec": json.loads(self.spec.canonical),
                    "payload": "payload.bin",
                    "byte_length": self.spec.expected_bytes,
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
                    existing = self.store._open_valid(self.spec)
                    if existing is not None:
                        payload, existing_digest = existing
                        payload.close()
                        if digest != existing_digest:
                            raise ArtifactConflict("same artifact key produced different bytes")
                        shutil.rmtree(self._temporary)
                    else:
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

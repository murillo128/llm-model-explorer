"""Read-only local file snapshots. Paths and fingerprints stay inside the backend."""

import hashlib
import json
import os
import stat
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, cast

MAX_METADATA_BYTES = 100_000_000
HASH_CHUNK_BYTES = 1024 * 1024


class ModelError(Exception):
    """A path-free error reusable by HTTP and future stream delivery adapters."""

    def __init__(self, code: str, message: str, status: int = 422) -> None:
        super().__init__(message)
        self.code = code
        self.status = status


def invalid(message: str = "Invalid local model metadata.") -> ModelError:
    return ModelError("validation_error", message)


def changed() -> ModelError:
    return ModelError("model_content_changed", "Model content changed; create a new session.", 409)


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise invalid("Duplicate metadata key.")
        result[key] = value
    return result


def parse_json(raw: bytes) -> dict[str, object]:
    try:
        result = json.loads(raw, object_pairs_hook=_unique_object)
    except (ValueError, RecursionError) as exc:
        raise invalid() from exc
    if not isinstance(result, dict):
        raise invalid()
    return cast(dict[str, object], result)


def confined(root: Path, path: Path) -> Path:
    resolved = path.resolve(strict=True)
    if not resolved.is_relative_to(root):
        raise invalid("Model file escapes the configured root.")
    return resolved


@contextmanager
def open_local(root: Path, path: Path) -> Iterator[BinaryIO]:
    """Resolve in-root symlinks, then open without following replacement symlinks.

    On POSIX each component is opened relative to a held directory descriptor.
    The resolved path is never used to read through a symlink swapped in later.
    """
    resolved = confined(root, path)
    if os.open in os.supports_dir_fd:
        directory = os.open(root.anchor, os.O_RDONLY | os.O_DIRECTORY)
        try:
            for part in resolved.parts[1:-1]:
                child = os.open(
                    part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=directory
                )
                os.close(directory)
                directory = child
            fd = os.open(
                resolved.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory
            )
        finally:
            os.close(directory)
        with os.fdopen(fd, "rb") as stream:
            if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
                raise invalid("Model assets must be regular files.")
            yield stream
    else:
        with resolved.open("rb") as stream:
            if confined(root, path) != resolved or not stat.S_ISREG(
                os.fstat(stream.fileno()).st_mode
            ):
                raise invalid("Model assets must be stable regular files.")
            yield stream


@dataclass(frozen=True)
class FileStamp:
    resolved: Path
    device: int
    inode: int
    size: int
    modified_ns: int
    changed_ns: int

    @classmethod
    def capture(cls, root: Path, path: Path) -> "FileStamp":
        resolved = confined(root, path)
        info = resolved.stat()
        if not stat.S_ISREG(info.st_mode):
            raise invalid("Model assets must be regular files.")
        return cls.from_stat(resolved, info)

    @classmethod
    def from_stat(cls, resolved: Path, info: os.stat_result) -> "FileStamp":
        return cls(
            resolved, info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns
        )


def relevant_files(root: Path, directory: Path, shards: tuple[str, ...]) -> tuple[str, ...]:
    confined(root, directory)
    # Conventional local HF assets only; never recursively explore unrelated trees.
    names = {
        entry.name
        for entry in directory.iterdir()
        if entry.suffix in {".json", ".model", ".txt", ".tiktoken", ".safetensors"}
    }
    return tuple(sorted(names | set(shards)))


@dataclass(frozen=True)
class FileSnapshot:
    root: Path
    directory: Path
    shards: tuple[str, ...]
    files: tuple[tuple[str, FileStamp], ...]

    @classmethod
    def capture(cls, root: Path, directory: Path, shards: tuple[str, ...]) -> "FileSnapshot":
        names = relevant_files(root, directory, shards)
        return cls(
            root,
            directory,
            shards,
            tuple((name, FileStamp.capture(root, directory / name)) for name in names),
        )

    def check(self) -> None:
        try:
            if self != self.capture(self.root, self.directory, self.shards):
                raise changed()
        except (OSError, RuntimeError, ModelError) as exc:
            raise changed() from exc

    @contextmanager
    def open(self, name: str) -> Iterator[BinaryIO]:
        expected = dict(self.files)[name]
        self.check()
        try:
            with open_local(self.root, self.directory / name) as stream:
                actual = FileStamp.from_stat(expected.resolved, os.fstat(stream.fileno()))
                if actual != expected:
                    raise changed()
                yield stream
                if FileStamp.from_stat(expected.resolved, os.fstat(stream.fileno())) != expected:
                    raise changed()
            self.check()
        except (OSError, RuntimeError) as exc:
            raise changed() from exc

    def fingerprint(self) -> str:
        digest = hashlib.sha256(b"llm-model-content\x00")
        self.check()
        for name, stamp in self.files:
            encoded = name.encode("utf-8")
            digest.update(len(encoded).to_bytes(8, "little"))
            digest.update(encoded)
            digest.update(stamp.size.to_bytes(8, "little"))
            with self.open(name) as stream:
                while block := stream.read(HASH_CHUNK_BYTES):
                    digest.update(block)
        self.check()
        return digest.hexdigest()


def read_json(root: Path, path: Path) -> dict[str, object]:
    with open_local(root, path) as stream:
        raw = stream.read(MAX_METADATA_BYTES + 1)
    if len(raw) > MAX_METADATA_BYTES:
        raise ModelError("unsupported_size", "Model metadata exceeds the supported size.")
    return parse_json(raw)

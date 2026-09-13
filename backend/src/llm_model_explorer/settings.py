"""CLI-owned local configuration; filesystem paths never belong in public descriptors."""

import os
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryFile
from urllib.parse import urlsplit

from .device import validate_device


def _validate_origin(origin: str) -> None:
    try:
        parsed = urlsplit(origin)
        port = parsed.port
    except ValueError as exc:
        raise ValueError(f"invalid CORS origin: {origin!r}") from exc
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.path
        or parsed.query
        or parsed.fragment
        or "*" in origin
        or any(character.isspace() for character in origin)
        or origin != f"{parsed.scheme}://{parsed.netloc}"
        or (port is not None and not 1 <= port <= 65535)
    ):
        raise ValueError(
            f"CORS origin must be an explicit HTTP(S) origin without a path: {origin!r}"
        )


@dataclass(frozen=True)
class Settings:
    model_root: Path
    cache_dir: Path
    device: str = "cpu"
    host: str = "127.0.0.1"
    port: int = 8000
    cors_origins: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if (
            isinstance(self.port, bool)
            or not isinstance(self.port, int)
            or not 1 <= self.port <= 65535
        ):
            raise ValueError("port must be an integer between 1 and 65535")
        if not self.host or any(character.isspace() for character in self.host):
            raise ValueError("host must be a nonempty hostname or IP address without whitespace")
        for origin in self.cors_origins:
            _validate_origin(origin)
        device = validate_device(self.device)
        try:
            model_root = self.model_root.expanduser().resolve(strict=True)
            cache_dir = self.cache_dir.expanduser().resolve()
            if not model_root.is_dir() or not os.access(model_root, os.R_OK | os.X_OK):
                raise ValueError("model root must be a readable and searchable directory")
            # Probe directory enumeration, never model file contents.
            with os.scandir(model_root):
                pass
            if model_root.is_relative_to(cache_dir) or cache_dir.is_relative_to(model_root):
                raise ValueError(
                    "cache directory and model root must not overlap (including symlinks)"
                )
            cache_dir.mkdir(parents=True, exist_ok=True)
            if not os.access(cache_dir, os.W_OK | os.X_OK):
                raise ValueError("cache directory must be writable and searchable")
            # Verify actual filesystem writes as well as mode/ACL access; remove the probe.
            with TemporaryFile(dir=cache_dir) as probe:
                probe.write(b"cache write check")
                probe.flush()
        except (OSError, RuntimeError) as exc:
            raise ValueError(f"invalid model/cache configuration: {exc}") from exc
        object.__setattr__(self, "model_root", model_root)
        object.__setattr__(self, "cache_dir", cache_dir)
        object.__setattr__(self, "device", device)
        object.__setattr__(self, "cors_origins", tuple(self.cors_origins))

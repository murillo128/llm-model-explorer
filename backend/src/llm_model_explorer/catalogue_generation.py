"""Bounded filesystem metadata for catalogue invalidation; never read asset bytes."""

import stat
from dataclasses import dataclass
from pathlib import Path

from .model_files import FileStamp, ModelError, confined


@dataclass(frozen=True)
class _PathState:
    path: Path
    link: FileStamp | None
    target: FileStamp | None
    mode: int | None
    error: str | int | None = None

    @classmethod
    def capture(cls, root: Path, path: Path) -> "_PathState":
        link = None
        try:
            # Include the link itself as well as its confined resolved target.
            link = FileStamp.from_stat(path, path.lstat())
            resolved = confined(root, path)
            info = resolved.stat()
            return cls(path, link, FileStamp.from_stat(resolved, info), info.st_mode)
        except OSError as exc:
            return cls(path, link, None, None, exc.errno)
        except (RuntimeError, ModelError):
            return cls(path, link, None, None, "invalid")


@dataclass(frozen=True)
class CatalogueGeneration:
    paths: tuple[_PathState, ...]

    @classmethod
    def capture(cls, root: Path, shards: frozenset[Path]) -> "CatalogueGeneration":
        """Enumerate immediate candidates/assets and explicitly indexed nested shards.

        Invalid/missing assets are observations too: repairing an omitted candidate
        must invalidate the catalogue. Never walk unrelated subdirectory trees or
        enumerate a symlink target outside the configured root.
        """
        paths: dict[Path, _PathState] = {}

        def record(path: Path) -> _PathState:
            state = _PathState.capture(root, path)
            paths[path] = state
            return state

        try:
            root_state = record(root)
            if root_state.mode is None or not stat.S_ISDIR(root_state.mode):
                raise OSError("Unavailable model root")
            for candidate in sorted(root.iterdir()):
                state = record(candidate)
                if state.mode is None or not stat.S_ISDIR(state.mode):
                    continue
                try:
                    for asset in sorted(candidate.iterdir()):
                        record(asset)
                except OSError as exc:
                    # Discovery omits unreadable candidates rather than failing
                    # the whole root. Retain that state, including permissions.
                    paths[candidate] = _PathState(
                        candidate, state.link, state.target, state.mode, exc.errno
                    )
            for shard in sorted(shards):
                record(shard)
            return cls(tuple(paths[path] for path in sorted(paths)))
        except OSError as exc:
            raise ModelError("internal_error", "Unable to discover local models.", 500) from exc

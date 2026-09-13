"""Real-filesystem evidence for complete publication and bounded reader lifetimes."""

import errno
import json
import multiprocessing
import os
import shutil
import tracemalloc
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier
from typing import Any
from unittest.mock import patch

import pytest
from fastapi import APIRouter, Request
from fastapi.testclient import TestClient

from llm_model_explorer.app import create_app
from llm_model_explorer.artifacts import ArtifactConflict, ArtifactSpec, ArtifactStore
from llm_model_explorer.dependencies import get_artifacts
from llm_model_explorer.settings import Settings


def specification(**changes: Any) -> ArtifactSpec:
    fields: dict[str, Any] = dict(
        model_fingerprint="model-content-digest",
        source="layers.0.weight",
        operation="materialize",
        parameters={"scale": 1, "nested": {"b": 2, "a": 1}},
        dtype="float32",
        layout="row-major-little-endian",
        shape=(2, 2),
        expected_bytes=16,
        producer="materialize:1",
    )
    fields.update(changes)
    return ArtifactSpec(**fields)


@pytest.fixture
def store(settings: Settings) -> ArtifactStore:
    return ArtifactStore(settings.cache_dir, model_root=settings.model_root)


def publish(store: ArtifactStore, spec: ArtifactSpec, data: bytes = b"0123456789abcdef") -> None:
    with store.begin_write(spec) as writer:
        writer.append(data)
        writer.commit()


def contents(store: ArtifactStore, spec: ArtifactSpec) -> bytes:
    reader = store.lookup(spec)
    assert reader is not None
    with reader:
        assert reader.complete
        return reader.read_available(128)


def assert_empty(store: ArtifactStore) -> None:
    assert list(store.root.iterdir()) == []


def test_key_is_canonical_and_snapshots_nested_input() -> None:
    parameters: dict[str, object] = {"nested": {"a": 1, "b": 2}, "scale": 1}
    spec = specification(parameters=parameters)
    assert spec == specification()
    assert len(spec.key) == 64
    parameters["scale"] = 99
    assert spec == specification()


@pytest.mark.parametrize(
    "change",
    [
        {"model_fingerprint": "other-content"},
        {"source": "layers.1.weight"},
        {"operation": "statistics"},
        {"parameters": {"scale": 2}},
        {"dtype": "float64"},
        {"layout": "column-major"},
        {"shape": (4,)},
        {"expected_bytes": 32},
        {"producer": "materialize:2"},
    ],
)
def test_every_computation_input_invalidates_key(change: dict[str, object]) -> None:
    assert specification(**change).key != specification().key


@pytest.mark.parametrize(
    "change",
    [
        {"parameters": {"x": float("nan")}},
        {"parameters": {"x": float("inf")}},
        {"parameters": {"x": {1: "bad key"}}},
        {"parameters": {"x": object()}},
        {"parameters": {"x": "x" * 40000}},
        {"expected_bytes": -1},
        {"expected_bytes": True},
        {"shape": (-1,)},
        {"shape": (True,)},
        {"producer": ""},
    ],
)
def test_invalid_spec_is_rejected(change: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        specification(**change)


def test_live_readers_publish_and_restart(store: ArtifactStore, settings: Settings) -> None:
    spec = specification()
    assert store.lookup(spec) is None
    with store.begin_write(spec) as writer:
        with writer.open_reader() as first, writer.open_reader() as second:
            assert first.available_bytes == 0
            assert first.read_available(100) == b""
            writer.append(b"0123")
            assert writer.available_bytes == 4
            assert store.lookup(spec) is None
            assert first.read_available(100) == b"0123"
            assert second.read_available(2) == b"01"
            assert not first.complete
            writer.append(b"456789abcdef")
            writer.commit()
            assert first.complete and second.complete
            assert first.read_available(100) == b"456789abcdef"
            assert second.read_available(100) == b"23456789abcdef"
        with pytest.raises(RuntimeError):
            writer.append(b"!")
    assert sorted(path.name for path in store.root.iterdir()) == [spec.key]
    assert sorted(path.name for path in (store.root / spec.key).iterdir()) == [
        "manifest.json",
        "payload.bin",
    ]
    restarted = ArtifactStore(settings.cache_dir, model_root=settings.model_root)
    assert contents(restarted, spec) == b"0123456789abcdef"
    assert (
        settings.model_root / "synthetic.safetensors"
    ).read_bytes() == b"synthetic weight sentinel"


def test_many_independent_readers(store: ArtifactStore) -> None:
    spec = specification()
    publish(store, spec)

    def read() -> bytes:
        reader = store.lookup(spec)
        assert reader is not None
        with reader:
            return b"".join(reader.read_available(1) for _ in range(16))

    with ThreadPoolExecutor(max_workers=8) as pool:
        assert list(pool.map(lambda _: read(), range(32))) == [b"0123456789abcdef"] * 32


def test_abort_preserves_attached_prefix_but_no_entry(store: ArtifactStore) -> None:
    spec = specification()
    with store.begin_write(spec) as writer:
        with writer.open_reader() as reader:
            writer.append(b"0123")
            writer.abort()
            writer.abort()
            assert reader.aborted and not reader.complete
            assert reader.read_available(99) == b"0123"
            assert reader.read_available(99) == b""
            assert store.lookup(spec) is None
            assert_empty(store)


@pytest.mark.parametrize("length", [0, 4, 17])
def test_incomplete_or_oversized_producer_cannot_commit(store: ArtifactStore, length: int) -> None:
    spec = specification()
    with pytest.raises(ValueError), store.begin_write(spec) as writer:
        writer.append(b"x" * length)
        writer.commit()
    assert store.lookup(spec) is None
    assert_empty(store)


def test_context_exception_aborts(store: ArtifactStore) -> None:
    with pytest.raises(InterruptedError), store.begin_write(specification()) as writer:
        writer.append(b"1234")
        raise InterruptedError("producer cancelled")
    assert_empty(store)


def test_orphan_temporary_is_never_resumed(store: ArtifactStore, settings: Settings) -> None:
    orphan = store.root / ".tmp-crashed-process"
    orphan.mkdir()
    (orphan / "payload.bin").write_bytes(b"partial")
    (orphan / "manifest.json").write_text('{"format":1}')
    restarted = ArtifactStore(settings.cache_dir, model_root=settings.model_root)
    assert restarted.lookup(specification()) is None
    publish(restarted, specification())
    assert contents(restarted, specification()) == b"0123456789abcdef"
    assert (orphan / "payload.bin").read_bytes() == b"partial"


@pytest.mark.parametrize(
    "damage",
    [
        "json",
        "key",
        "spec",
        "path",
        "length",
        "truncated",
        "missing",
        "format",
        "huge",
        "payload_symlink",
        "manifest_symlink",
        "directory_symlink",
        "fifo",
        "boolean_length",
    ],
)
def test_corrupt_entries_miss_and_regenerate(
    store: ArtifactStore, tmp_path: Path, damage: str
) -> None:
    spec = specification()
    publish(store, spec)
    entry = store.root / spec.key
    manifest = entry / "manifest.json"
    metadata = json.loads(manifest.read_text())
    outside = tmp_path / "outside"
    outside.mkdir()
    sentinel = outside / "sentinel"
    sentinel.write_bytes(b"do not change")
    if damage == "json":
        manifest.write_text("{")
    elif damage == "huge":
        manifest.write_bytes(b" " * 65537)
    elif damage == "truncated":
        (entry / "payload.bin").write_bytes(b"short")
    elif damage == "missing":
        manifest.unlink()
    elif damage == "fifo":
        (entry / "payload.bin").unlink()
        os.mkfifo(entry / "payload.bin")
    elif damage in {"payload_symlink", "manifest_symlink"}:
        victim = entry / ("payload.bin" if damage == "payload_symlink" else "manifest.json")
        victim.unlink()
        victim.symlink_to(sentinel)
    elif damage == "directory_symlink":
        shutil.rmtree(entry)
        entry.symlink_to(outside, target_is_directory=True)
    else:
        if damage == "key":
            metadata["key"] = "../outside"
        elif damage == "spec":
            metadata["spec"]["dtype"] = "float64"
        elif damage == "path":
            metadata["payload"] = "../../outside/sentinel"
        elif damage == "length":
            metadata["byte_length"] = 100
        elif damage == "boolean_length":
            metadata["byte_length"] = True
        elif damage == "format":
            metadata["format"] = 2
        manifest.write_text(json.dumps(metadata))
    assert store.lookup(spec) is None
    publish(store, spec)
    assert contents(store, spec) == b"0123456789abcdef"
    assert sentinel.read_bytes() == b"do not change"
    assert list(store.root.iterdir()) == [entry]


def test_invalid_key_cannot_traverse(store: ArtifactStore) -> None:
    spec = specification()
    object.__setattr__(spec, "key", "../outside")
    with pytest.raises(ValueError, match="key"):
        store.lookup(spec)


@pytest.mark.parametrize("filename", ["manifest.json", "payload.bin"])
def test_directory_in_place_of_cache_file_misses_and_regenerates(
    store: ArtifactStore, filename: str
) -> None:
    spec = specification()
    publish(store, spec)
    corrupt = store.root / spec.key / filename
    corrupt.unlink()
    corrupt.mkdir()
    descriptors = Path("/proc/self/fd")
    before = len(list(descriptors.iterdir())) if descriptors.is_dir() else None
    for _ in range(20):
        assert store.lookup(spec) is None
    if before is not None:
        assert len(list(descriptors.iterdir())) == before
    assert corrupt.is_dir()
    publish(store, spec)
    assert contents(store, spec) == b"0123456789abcdef"
    assert corrupt.is_file()
    assert list(store.root.iterdir()) == [store.root / spec.key]


@pytest.mark.parametrize("filename", ["manifest.json", "payload.bin"])
def test_failed_file_wrapper_closes_raw_descriptor(store: ArtifactStore, filename: str) -> None:
    spec = specification()
    publish(store, spec)
    inode = (store.root / spec.key / filename).stat().st_ino
    failed: list[int] = []
    original = os.fdopen

    def fail_wrapper(descriptor: int, *args: Any, **kwargs: Any) -> Any:
        if os.fstat(descriptor).st_ino == inode:
            failed.append(descriptor)
            raise OSError(errno.EIO, "cannot wrap cache descriptor")
        return original(descriptor, *args, **kwargs)

    with patch("os.fdopen", side_effect=fail_wrapper):
        with pytest.raises(OSError) as error:
            store.lookup(spec)
        assert error.value.errno == errno.EIO
    assert len(failed) == 1
    with pytest.raises(OSError) as closed:
        os.fstat(failed[0])
    assert closed.value.errno == errno.EBADF
    assert contents(store, spec) == b"0123456789abcdef"


@pytest.mark.parametrize("error", [errno.ENOSPC, errno.EACCES, errno.EIO])
@pytest.mark.parametrize("stage", ["append", "manifest", "sync", "rename"])
def test_disk_failures_leave_no_partial_entry(
    store: ArtifactStore, monkeypatch: pytest.MonkeyPatch, error: int, stage: str
) -> None:
    spec = specification()
    with pytest.raises(OSError), store.begin_write(spec) as writer:
        if stage == "append":
            monkeypatch.setattr(
                writer._payload,
                "write",
                lambda _: (_ for _ in ()).throw(OSError(error, "injected write failure")),
            )
        writer.append(b"0123456789abcdef")
        if stage == "manifest":
            original_open = Path.open

            def fail_manifest(path: Path, *args: Any, **kwargs: Any) -> Any:
                if path.name == "manifest.json":
                    raise OSError(error, "injected manifest failure")
                return original_open(path, *args, **kwargs)

            monkeypatch.setattr(Path, "open", fail_manifest)
        elif stage == "sync":
            monkeypatch.setattr(
                os,
                "fsync",
                lambda _: (_ for _ in ()).throw(OSError(error, "injected sync failure")),
            )
        elif stage == "rename":
            monkeypatch.setattr(
                os,
                "rename",
                lambda *_: (_ for _ in ()).throw(OSError(error, "injected publication failure")),
            )
        writer.commit()
    assert store.lookup(spec) is None
    assert_empty(store)


def test_partial_os_write_failure_exposes_only_previously_flushed_bytes(
    store: ArtifactStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    with store.begin_write(specification()) as writer, writer.open_reader() as reader:
        writer.append(b"0123")
        original = writer._payload.write
        calls = 0

        def partial(data: bytes) -> int:
            nonlocal calls
            calls += 1
            if calls == 1:
                return original(data[:2])
            raise OSError(errno.ENOSPC, "disk full after partial write")

        monkeypatch.setattr(writer._payload, "write", partial)
        with pytest.raises(OSError):
            writer.append(b"456789abcdef")
        assert reader.available_bytes == 4
        assert reader.read_available(100) == b"0123"
        assert reader.aborted
    assert_empty(store)


def test_failure_does_not_destroy_existing_entry(
    store: ArtifactStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = specification()
    publish(store, spec)
    before = (store.root / spec.key / "manifest.json").read_bytes()
    with pytest.raises(OSError), store.begin_write(spec) as writer:
        writer.append(b"0123456789abcdef")
        with monkeypatch.context() as scoped:
            scoped.setattr(os, "fsync", lambda _: (_ for _ in ()).throw(OSError("disk error")))
            writer.commit()
    assert contents(store, spec) == b"0123456789abcdef"
    assert (store.root / spec.key / "manifest.json").read_bytes() == before
    assert list(store.root.iterdir()) == [store.root / spec.key]


def test_unreadable_existing_entry_is_not_removed(store: ArtifactStore) -> None:
    spec = specification()
    publish(store, spec)
    original = os.open

    def deny(path: Any, *args: Any, **kwargs: Any) -> int:
        if path == "manifest.json":
            raise PermissionError(errno.EACCES, "cannot read existing manifest")
        return original(path, *args, **kwargs)

    with pytest.raises(PermissionError), store.begin_write(spec) as writer:
        writer.append(b"0123456789abcdef")
        with patch("os.open", side_effect=deny):
            writer.commit()
    assert contents(store, spec) == b"0123456789abcdef"
    assert list(store.root.iterdir()) == [store.root / spec.key]


def test_concurrent_same_key_publication_reuses_immutable_entry(store: ArtifactStore) -> None:
    barrier = Barrier(8)
    spec = specification()

    def produce(_: int) -> None:
        with store.begin_write(spec) as writer, writer.open_reader() as reader:
            writer.append(b"0123456789abcdef")
            barrier.wait(timeout=10)
            writer.commit()
            assert reader.complete
            assert reader.read_available(16) == b"0123456789abcdef"

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(produce, range(8)))
    assert contents(store, spec) == b"0123456789abcdef"
    assert list(store.root.iterdir()) == [store.root / spec.key]
    inode = (store.root / spec.key).stat().st_ino
    publish(store, spec)
    assert (store.root / spec.key).stat().st_ino == inode


def _process_publish(cache: Path, models: Path, byte: bytes) -> None:
    store = ArtifactStore(cache, model_root=models)
    try:
        publish(store, specification(), byte * 16)
    except ArtifactConflict:
        pass


def test_cross_process_competition_preserves_one_complete_result(store: ArtifactStore) -> None:
    ctx = multiprocessing.get_context("spawn")
    processes = [
        ctx.Process(target=_process_publish, args=(store.root, store.root.parent / "models", byte))
        for byte in [b"a", b"b", b"c", b"d"]
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=30)
        assert process.exitcode == 0
    assert contents(store, specification()) in [byte * 16 for byte in [b"a", b"b", b"c", b"d"]]
    assert list(store.root.iterdir()) == [store.root / specification().key]


def test_conflicting_producer_is_rejected_and_attached_reader_aborts(store: ArtifactStore) -> None:
    spec = specification()
    publish(store, spec)
    with store.begin_write(spec) as writer, writer.open_reader() as reader:
        writer.append(b"x" * 16)
        with pytest.raises(ArtifactConflict):
            writer.commit()
        assert reader.aborted
        assert reader.read_available(16) == b"x" * 16
    assert contents(store, spec) == b"0123456789abcdef"
    assert list(store.root.iterdir()) == [store.root / spec.key]


def test_delete_entire_cache_and_recreate(store: ArtifactStore, settings: Settings) -> None:
    spec = specification()
    publish(store, spec)
    shutil.rmtree(store.root)
    restarted = ArtifactStore(settings.cache_dir, model_root=settings.model_root)
    assert restarted.lookup(spec) is None
    publish(restarted, spec)
    assert contents(restarted, spec) == b"0123456789abcdef"


def test_streaming_memory_is_bounded(store: ArtifactStore) -> None:
    size = 8 * 1024 * 1024
    block = b"a" * 65536
    spec = specification(expected_bytes=size, shape=(size // 4,))
    tracemalloc.start()
    try:
        with store.begin_write(spec) as writer, writer.open_reader() as reader:
            for _ in range(size // len(block)):
                writer.append(block)
                assert reader.read_available(len(block)) == block
            writer.commit()
        cached = store.lookup(spec)
        assert cached is not None
        with cached:
            for _ in range(size // len(block)):
                assert cached.read_available(len(block)) == block
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert peak < 1024 * 1024
    assert (store.root / spec.key / "payload.bin").stat().st_size == size


def test_zero_length_artifact_and_reader_lifetime(store: ArtifactStore) -> None:
    spec = specification(expected_bytes=0, shape=(0,))
    publish(store, spec, b"")
    reader = store.lookup(spec)
    assert reader is not None
    with reader:
        assert reader.read_available(1) == b""
        with pytest.raises(ValueError):
            reader.read_available(-1)
    with pytest.raises(ValueError):
        reader.read_available(1)


def test_external_truncation_after_lookup_raises(store: ArtifactStore) -> None:
    spec = specification()
    publish(store, spec)
    reader = store.lookup(spec)
    assert reader is not None
    with reader:
        (store.root / spec.key / "payload.bin").write_bytes(b"x")
        with pytest.raises(OSError, match="truncated"):
            reader.read_available(16)


@pytest.mark.parametrize("relative", [".", "inside"])
def test_cache_must_not_overlap_model_storage(tmp_path: Path, relative: str) -> None:
    with pytest.raises(ValueError, match="overlap"):
        ArtifactStore(tmp_path / relative, model_root=tmp_path)
    with pytest.raises(ValueError, match="overlap"):
        ArtifactStore(tmp_path, model_root=tmp_path / relative)


def test_lifespan_provides_configured_store_and_persists_entries(settings: Settings) -> None:
    router = APIRouter()

    @router.get("/test-cache")
    def cache(request: Request) -> dict[str, bool]:
        store = get_artifacts(request)
        assert store.root == settings.cache_dir
        reader = store.lookup(specification())
        if reader is not None:
            reader.close()
            return {"hit": True}
        publish(store, specification())
        return {"hit": False}

    for expected in [False, True]:
        with TestClient(create_app(settings, routers=[router])) as client:
            assert client.get("/test-cache").json() == {"hit": expected}


def test_publication_boundary_contains_both_complete_files(store: ArtifactStore) -> None:
    spec = specification()
    original = os.rename

    def inspect_and_publish(source: Path, target: Path) -> None:
        assert store.lookup(spec) is None
        assert (source / "payload.bin").read_bytes() == b"0123456789abcdef"
        manifest = json.loads((source / "manifest.json").read_text())
        assert manifest["key"] == spec.key
        assert manifest["byte_length"] == (source / "payload.bin").stat().st_size
        original(source, target)
        assert contents(store, spec) == b"0123456789abcdef"

    with patch("os.rename", side_effect=inspect_and_publish):
        publish(store, spec)
    assert list(store.root.iterdir()) == [store.root / spec.key]


def test_parent_sync_failure_leaves_only_a_fully_published_entry(store: ArtifactStore) -> None:
    spec = specification()
    root_inode = store.root.stat().st_ino
    original = os.fsync

    def fail_parent_sync(descriptor: int) -> None:
        if os.fstat(descriptor).st_ino == root_inode:
            raise OSError(errno.EIO, "parent sync failed after rename")
        original(descriptor)

    with patch("os.fsync", side_effect=fail_parent_sync), pytest.raises(OSError):
        publish(store, spec)
    assert contents(store, spec) == b"0123456789abcdef"
    assert list(store.root.iterdir()) == [store.root / spec.key]

"""Structured cache integrity and ownership against the shared graph oracle."""

import errno
import hashlib
import json
import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from threading import Barrier
from time import perf_counter
from typing import Any
from unittest.mock import Mock, patch

import pytest
from test_architecture_analysis import PRODUCER, context, inputs, registry
from test_artifacts import specification
from test_models import make_model

from llm_model_explorer.architecture_analysis import (
    AnalysisInput,
    GraphBuilder,
    parse_graph,
    serialize_graph,
)
from llm_model_explorer.architecture_analysis import records as r
from llm_model_explorer.architecture_analysis.validation import MAX_BYTES
from llm_model_explorer.artifacts import (
    ArchitectureArtifactSpec,
    ArtifactConflict,
    ArtifactStore,
    ArtifactWriter,
)
from llm_model_explorer.model_files import ModelError
from llm_model_explorer.models import ModelCatalogue
from llm_model_explorer.settings import Settings


@pytest.fixture
def store(settings: Settings) -> ArtifactStore:
    return ArtifactStore(settings.cache_dir, model_root=settings.model_root)


def spec(**changes: Any) -> ArchitectureArtifactSpec:
    return ArchitectureArtifactSpec(
        **{
            "model_fingerprint": inputs().fingerprint,
            "producer": PRODUCER,
            "scope": "language_model",
            "analysis_options": {},
            **changes,
        }
    )


def graph() -> r.ArchitectureGraph:
    result = registry().analyze(inputs())
    assert result.graph is not None
    return result.graph


def publish(store: ArtifactStore, value: r.ArchitectureGraph | None = None) -> bytes:
    value = graph() if value is None else value
    with store.begin_graph_write(spec(), context(), check_source=lambda: None) as writer:
        writer.write_graph(value)
        writer.commit()
    return serialize_graph(value)


def read(store: ArtifactStore) -> bytes:
    reader = store.lookup_graph(spec(), context())
    assert reader is not None
    with reader:
        assert reader.complete and not reader.aborted
        return reader.read_available(MAX_BYTES)


def rewrite(store: ArtifactStore, document: dict[str, Any]) -> None:
    entry = store.root / spec().key
    data = json.dumps(document, ensure_ascii=False, separators=(",", ":")).encode()
    (entry / "payload.bin").write_bytes(data)
    metadata = json.loads((entry / "manifest.json").read_bytes())
    metadata.update(byte_length=len(data), sha256=hashlib.sha256(data).hexdigest())
    (entry / "manifest.json").write_text(json.dumps(metadata))


def test_cold_warm_lookup_does_not_construct_graph(store: ArtifactStore) -> None:
    assert store.lookup_graph(spec(), context()) is None
    start = perf_counter()
    expected = publish(store)
    cold = perf_counter() - start
    entry = store.root / spec().key
    metadata = json.loads((entry / "manifest.json").read_bytes())
    assert metadata == {
        "format": 2,
        "key": spec().key,
        "spec": json.loads(spec().canonical),
        "payload": "payload.bin",
        "byte_length": len(expected),
        "sha256": hashlib.sha256(expected).hexdigest(),
    }
    assert metadata["spec"]["kind"] == "architecture_graph"
    assert metadata["spec"]["media_type"] == "application/json"
    assert "byte_length" not in metadata["spec"]
    denied = Mock(side_effect=AssertionError("warm lookup must not analyze"))
    with patch("llm_model_explorer.architecture_analysis.DescriptionRegistry.analyze", denied):
        with patch(
            "llm_model_explorer.artifacts.parse_graph",
            wraps=parse_graph,
        ) as validate:
            start = perf_counter()
            for _ in range(20):
                assert read(store) == expected
            warm = (perf_counter() - start) / 20
            assert validate.call_count == 20
    denied.assert_not_called()
    print(f"cold construction/publication={cold:.6f}s; warm validation mean={warm:.6f}s")


@pytest.mark.parametrize(
    "field", ["description", "revision", "source_revision", "analyzer_revision", "schema_revision"]
)
def test_semantic_revisions_invalidate(store: ArtifactStore, field: str) -> None:
    publish(store)
    changed = spec(producer=replace(PRODUCER, **{field: "changed"}))
    assert changed.key != spec().key and changed.graph_id != spec().graph_id
    assert store.lookup_graph(changed, context()) is None


@pytest.mark.parametrize(
    "changes",
    [
        {"model_fingerprint": "changed"},
        {"analysis_options": {"meaning": True}},
        {"scope": "visual_encoder_predictor"},
        {"serializer_revision": "changed"},
    ],
)
def test_other_input_changes_invalidate(store: ArtifactStore, changes: dict[str, Any]) -> None:
    publish(store)
    changed = spec(**changes)
    assert changed.key != spec().key
    assert store.lookup_graph(changed, context()) is None


def test_spec_snapshots_options_and_is_domain_separated() -> None:
    options: dict[str, Any] = {"nested": {"b": 2, "a": 1}}
    saved = spec(analysis_options=options)
    options["nested"]["a"] = 7
    assert saved == spec(analysis_options={"nested": {"a": 1, "b": 2}})
    assert saved.key != specification().key
    assert spec().graph_id == PRODUCER.graph_id(inputs().fingerprint, "language_model")


@pytest.mark.parametrize(
    "changes",
    [
        {"model_fingerprint": ""},
        {"scope": "unknown"},
        {"analysis_options": {"x": float("nan")}},
        {"analysis_options": {"x": object()}},
        {"analysis_options": {"x": {1: "invalid"}}},
        {"analysis_options": {"x": "x" * 40000}},
    ],
)
def test_invalid_input_identity(changes: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        spec(**changes)


def test_partial_coverage_is_complete_serialization(store: ArtifactStore) -> None:
    builder = GraphBuilder(inputs(), PRODUCER, "language_model")
    builder.unknown("opaque", "Unreviewed block", "Semantics are unavailable.")
    value = builder.finish()
    assert value.coverage == "partial"
    expected = publish(store, value)
    assert read(store) == expected
    assert json.loads(expected)["diagnostics"]


@pytest.mark.parametrize(
    "damage",
    [
        "schema",
        "reference",
        "identity",
        "scope",
        "inventory",
        "envelope",
        "nan",
        "unavailable",
    ],
)
def test_full_validation_even_with_matching_digest(store: ArtifactStore, damage: str) -> None:
    publish(store)
    document = graph().document()
    if damage == "schema":
        document["nodes"][0]["kind"] = "invalid"
    elif damage == "reference":
        document["nodes"][0]["parameter_ids"] = ["absent"]
    elif damage == "identity":
        document["graph_id"] = "another-analysis"
    elif damage == "scope":
        document["scope"] = "visual_encoder_predictor"
    elif damage == "inventory":
        document["parameters"][0]["inspection"]["tensor_id"] = "foreign"
    elif damage == "envelope":
        document["model_id"] = "public-model"
    elif damage == "nan":
        document["nodes"][0]["attributes"] = [
            {"name": "x", "value": float("nan"), "provenance": []}
        ]
    else:
        document = {"status": "unavailable", "reason": "analysis_failed"}
    rewrite(store, document)
    assert store.lookup_graph(spec(), context()) is None
    with (
        pytest.raises(ValueError),
        store.begin_graph_write(spec(), context(), check_source=lambda: None) as writer,
    ):
        writer.append((store.root / spec().key / "payload.bin").read_bytes())
        writer.commit()
    assert not list(store.root.glob(".tmp-*"))


@pytest.mark.parametrize(
    "damage",
    [
        "digest",
        "truncated",
        "oversize",
        "format",
        "manifest",
        "huge_manifest",
        "boolean_length",
        "negative_length",
        "huge_length",
        "path",
        "spec",
        "utf16",
        "noncanonical",
        "nested",
        "payload_symlink",
        "manifest_symlink",
        "directory_symlink",
        "fifo",
    ],
)
def test_corruption_is_bounded_miss_and_reconstructible(
    store: ArtifactStore, tmp_path: Path, damage: str
) -> None:
    expected = publish(store)
    entry = store.root / spec().key
    payload, manifest = entry / "payload.bin", entry / "manifest.json"
    metadata = json.loads(manifest.read_bytes())
    outside = tmp_path / "sentinel"
    outside.write_bytes(b"unchanged")
    if damage == "digest":
        payload.write_bytes(expected.replace(b'"complete"', b'"partial "'))
    elif damage == "truncated":
        payload.write_bytes(expected[:-1])
    elif damage == "oversize":
        with payload.open("r+b") as stream:
            stream.truncate(MAX_BYTES + 1)
    elif damage in ("utf16", "noncanonical", "nested"):
        data = (
            expected.decode().encode("utf-16")
            if damage == "utf16"
            else b" " + expected
            if damage == "noncanonical"
            else b"[" * 2000 + b"]" * 2000
        )
        payload.write_bytes(data)
        metadata.update(byte_length=len(data), sha256=hashlib.sha256(data).hexdigest())
    elif damage == "format":
        metadata["format"] = 1
    elif damage == "boolean_length":
        metadata["byte_length"] = True
    elif damage == "negative_length":
        metadata["byte_length"] = -1
    elif damage == "huge_length":
        metadata["byte_length"] = MAX_BYTES + 1
    elif damage == "path":
        metadata["payload"] = str(outside)
    elif damage == "spec":
        metadata["spec"]["analyzer_revision"] = "obsolete"
    elif damage in ("payload_symlink", "manifest_symlink"):
        victim = payload if damage == "payload_symlink" else manifest
        victim.unlink()
        victim.symlink_to(outside)
    elif damage == "directory_symlink":
        shutil.rmtree(entry)
        entry.symlink_to(tmp_path, target_is_directory=True)
    elif damage == "fifo":
        payload.unlink()
        os.mkfifo(payload)
    if damage not in ("manifest_symlink", "directory_symlink"):
        manifest.write_text(
            "{"
            if damage == "manifest"
            else " " * 65537
            if damage == "huge_manifest"
            else json.dumps(metadata)
        )
    assert store.lookup_graph(spec(), context()) is None
    assert outside.read_bytes() == b"unchanged"
    assert publish(store) == expected
    assert read(store) == expected
    assert outside.read_bytes() == b"unchanged"


def test_lookup_checks_current_inventory(store: ArtifactStore) -> None:
    publish(store)
    assert store.lookup_graph(spec(), replace(context(), numeric={})) is None


def test_live_readers_abort_and_preserve_other_writers(store: ArtifactStore) -> None:
    orphan = store.root / ".tmp-other-process"
    orphan.mkdir()
    (orphan / "payload.bin").write_bytes(b"abandoned")
    with store.begin_graph_write(spec(), context(), check_source=lambda: None) as other:
        other.write_graph(graph())
        with store.begin_graph_write(spec(), context(), check_source=lambda: None) as writer:
            with writer.open_reader() as first, writer.open_reader() as second:
                writer.append(b'{"graph_id":')
                assert first.read_available(1) == b"{"
                writer.abort()
                writer.abort()
                assert first.aborted and not first.complete
                assert second.read_available(100) == b'{"graph_id":'
        assert store.lookup_graph(spec(), context()) is None
        assert other._temporary.is_dir()
        assert (orphan / "payload.bin").read_bytes() == b"abandoned"
        other.commit()
    assert read(store) == serialize_graph(graph())
    assert orphan.is_dir()


@pytest.mark.parametrize("cause", ["cancelled", "incomplete", "oversize", "source", "tampered"])
def test_failed_generation_never_publishes(store: ArtifactStore, cause: str) -> None:
    def check() -> None:
        if cause == "source":
            raise RuntimeError("pinned source changed")

    with (
        pytest.raises((ValueError, RuntimeError, InterruptedError)),
        store.begin_graph_write(spec(), context(), check_source=check) as writer,
    ):
        if cause == "incomplete":
            writer.append(b'{"graph_id":')
        elif cause == "oversize":
            block = b" " * (1024 * 1024)
            for _ in range(33):
                writer.append(block)
        else:
            writer.write_graph(graph())
        if cause == "cancelled":
            raise InterruptedError("cancelled during serialization")
        if cause == "tampered":
            path = writer._temporary / "payload.bin"
            path.write_bytes(path.read_bytes().replace(b"complete", b"partial "))
        writer.commit()
    assert store.lookup_graph(spec(), context()) is None
    assert list(store.root.iterdir()) == []


def test_source_check_runs_at_publication_and_on_reuse(store: ArtifactStore) -> None:
    original = os.rename
    checked = Mock()
    with store.begin_graph_write(spec(), context(), check_source=checked) as writer:
        writer.write_graph(graph())

        def rename(source: Path, target: Path) -> None:
            checked.assert_called_once_with()
            assert store.lookup_graph(spec(), context()) is None
            original(source, target)

        with patch("os.rename", side_effect=rename):
            writer.commit()
    checked.reset_mock()
    with store.begin_graph_write(spec(), context(), check_source=checked) as writer:
        writer.write_graph(graph())
        writer.commit()
    checked.assert_called_once_with()


@pytest.mark.parametrize("stage", ["append", "manifest", "sync", "rename", "lookup", "reuse"])
@pytest.mark.parametrize("error", [errno.EACCES, errno.EIO, errno.ENOSPC])
def test_io_failures_propagate_without_destroying_data(
    store: ArtifactStore, stage: str, error: int
) -> None:
    expected = publish(store) if stage in ("lookup", "reuse") else None
    fail = Mock(side_effect=OSError(error, "injected storage failure"))
    if stage == "lookup":
        original = os.open

        def deny(path: Any, *args: Any, **kwargs: Any) -> int:
            if path == "manifest.json":
                fail()
                raise AssertionError("injected failure did not raise")
            return original(path, *args, **kwargs)

        with patch("os.open", side_effect=deny), pytest.raises(OSError):
            store.lookup_graph(spec(), context())
    else:
        with (
            pytest.raises(OSError),
            store.begin_graph_write(spec(), context(), check_source=lambda: None) as writer,
        ):
            if stage == "append":
                with patch.object(writer._payload, "write", fail):
                    writer.write_graph(graph())
            writer.write_graph(graph())
            target = {
                "manifest": "pathlib.Path.open",
                "sync": "os.fsync",
                "rename": "os.rename",
                "reuse": "llm_model_explorer.artifacts.ArtifactStore._open_valid",
            }[stage]
            original_open = Path.open

            def fail_manifest(path: Path, *args: Any, **kwargs: Any) -> Any:
                if path.name == "manifest.json":
                    return fail()
                return original_open(path, *args, **kwargs)

            with patch(target, fail_manifest if stage == "manifest" else fail):
                writer.commit()
    assert not list(store.root.glob(".tmp-*"))
    if expected is not None:
        assert read(store) == expected
    else:
        assert list(store.root.iterdir()) == []


def test_simultaneous_readers_and_same_key_writers(store: ArtifactStore) -> None:
    expected = serialize_graph(graph())
    barrier = Barrier(8)

    def work(_: int) -> None:
        with store.begin_graph_write(spec(), context(), check_source=lambda: None) as writer:
            with writer.open_reader() as first, writer.open_reader() as second:
                writer.write_graph(graph())
                assert first.read_available(3) == expected[:3]
                barrier.wait(timeout=10)
                writer.commit()
                assert first.complete and second.complete
                assert first.read_available(MAX_BYTES) == expected[3:]
                assert second.read_available(MAX_BYTES) == expected
                assert read(store) == expected

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(work, range(8)))
    assert list(store.root.iterdir()) == [store.root / spec().key]


def test_divergent_output_preserves_existing_and_aborts_reader(store: ArtifactStore) -> None:
    expected = publish(store)
    changed = graph().document()
    changed["nodes"][0]["label"] = "Different output"
    with store.begin_graph_write(spec(), context(), check_source=lambda: None) as writer:
        with writer.open_reader() as reader:
            writer.write_graph(r.ArchitectureGraph.model_validate(changed))
            with pytest.raises(ArtifactConflict):
                writer.commit()
            assert reader.aborted and not reader.complete
    assert read(store) == expected
    assert not list(store.root.glob(".tmp-*"))


def test_relocation_source_mutation_and_delete_restart(
    store: ArtifactStore, tmp_path: Path
) -> None:
    root = tmp_path / "original"
    root.mkdir()
    directory = make_model(root)
    (directory / "config.json").write_text('{"model_type":"synthetic"}')
    moved = tmp_path / "moved"
    moved.mkdir()
    shutil.copytree(directory, moved / "different-name")
    sources = [ModelCatalogue(p).discover()[0].pin() for p in (root, moved)]
    assert sources[0].model_id != sources[1].model_id
    assert sources[0].fingerprint == sources[1].fingerprint
    source = sources[0]
    data = AnalysisInput.from_source(source, tokenizer_available=False)
    builder = GraphBuilder(data, PRODUCER, "language_model")
    builder.unknown("scope", "Scope", "Fixture only")
    value = builder.finish()
    saved = spec(model_fingerprint=source.fingerprint)

    def produce() -> None:
        with store.begin_graph_write(
            saved, data.bindings, check_source=lambda: source.check_unchanged(rehash=True)
        ) as writer:
            writer.write_graph(value)
            writer.commit()

    produce()
    relocated = spec(model_fingerprint=sources[1].fingerprint)
    reader = store.lookup_graph(relocated, data.bindings)
    assert reader is not None
    with reader:
        assert reader.read_available(MAX_BYTES) == serialize_graph(value)
    shutil.rmtree(store.root)
    restarted = ArtifactStore(store.root, model_root=root)
    assert restarted.lookup_graph(saved, data.bindings) is None
    produce()
    (directory / "config.json").write_text('{"model_type":"mutated"}')
    with pytest.raises(ModelError, match="changed"):
        produce()
    reader = restarted.lookup_graph(saved, data.bindings)
    assert reader is not None
    with reader:
        assert reader.read_available(MAX_BYTES) == serialize_graph(value)
    assert not list(store.root.glob(".tmp-*"))


def test_numeric_manifest_and_key_unchanged(store: ArtifactStore) -> None:
    numeric = specification()
    canonical = (
        '{"byte_length":16,"dtype":"float32","layout":"row-major-little-endian",'
        '"model_fingerprint":"model-content-digest","operation":"materialize",'
        '"parameters":{"nested":{"a":1,"b":2},"scale":1},"producer":"materialize:1",'
        '"shape":[2,2],"source":"layers.0.weight"}'
    )
    assert numeric.canonical == canonical
    assert numeric.key == hashlib.sha256(canonical.encode()).hexdigest()
    data = b"0123456789abcdef"
    with store.begin_write(numeric) as writer:
        writer.append(data)
        writer.commit()
    before = (store.root / numeric.key / "manifest.json").read_bytes()
    assert json.loads(before) == {
        "format": 1,
        "key": numeric.key,
        "spec": json.loads(canonical),
        "payload": "payload.bin",
        "byte_length": 16,
        "sha256": hashlib.sha256(data).hexdigest(),
    }
    publish(store)
    assert (store.root / numeric.key / "manifest.json").read_bytes() == before
    reader = store.lookup(numeric)
    assert reader is not None
    with reader:
        assert reader.read_available(16) == data


def test_numeric_writer_cannot_bypass_graph_validation(store: ArtifactStore) -> None:
    with (
        pytest.raises(TypeError, match="begin_graph_write"),
        ArtifactWriter(store, spec()) as writer,
    ):
        writer.append(b'{"status":"unavailable"}')
        writer.commit()
    assert list(store.root.iterdir()) == []


def test_payload_read_is_bounded_and_oversize_rejected_before_read(
    store: ArtifactStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    from llm_model_explorer import artifacts

    expected = publish(store)
    original = artifacts._open_regular
    reads: list[int] = []

    def opened(directory: int, filename: str) -> Any:
        handle = original(directory, filename)
        if filename == "payload.bin" and handle is not None:
            actual_read = handle.read

            def bounded(size: int = -1) -> bytes:
                assert 0 < size <= MAX_BYTES + 1
                reads.append(size)
                return actual_read(size)

            monkeypatch.setattr(handle, "read", bounded)
        return handle

    with patch.object(artifacts, "_open_regular", side_effect=opened):
        reader = store.lookup_graph(spec(), context())
        assert reader is not None
        reader.close()
        assert reads == [len(expected) + 1]
        with (store.root / spec().key / "payload.bin").open("r+b") as payload:
            payload.truncate(MAX_BYTES + 1)
        assert store.lookup_graph(spec(), context()) is None
        assert reads == [len(expected) + 1]


def test_meaningful_options_round_trip(store: ArtifactStore) -> None:
    saved = spec(analysis_options={"declared_scope_option": "fixture"})
    builder = GraphBuilder(inputs(), PRODUCER, "language_model")
    builder.graph_id = saved.graph_id
    builder.unknown("scope", "Scope", "Explicit fixture scope option")
    value = builder.finish()
    with store.begin_graph_write(saved, context(), check_source=lambda: None) as writer:
        writer.write_graph(value)
        writer.commit()
    reader = store.lookup_graph(saved, context())
    assert reader is not None
    with reader:
        assert reader.read_available(MAX_BYTES) == serialize_graph(value)
    assert store.lookup_graph(spec(), context()) is None


@pytest.mark.parametrize("replacement", ["symlink", "directory", "file"])
def test_replaced_private_payload_cannot_publish(store: ArtifactStore, replacement: str) -> None:
    with (
        pytest.raises((ValueError, OSError)),
        store.begin_graph_write(spec(), context(), check_source=lambda: None) as writer,
    ):
        writer.write_graph(graph())
        path = writer._temporary / "payload.bin"
        saved = writer._temporary / "saved.bin"
        path.rename(saved)
        if replacement == "symlink":
            path.symlink_to(saved)
        elif replacement == "directory":
            path.mkdir()
        else:
            path.write_bytes(saved.read_bytes())
        writer.commit()
    assert list(store.root.iterdir()) == []

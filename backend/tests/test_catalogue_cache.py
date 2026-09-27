"""Catalogue reuse, filesystem invalidation and single-flight refresh oracles."""

import asyncio
import json
import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier, Event, Lock
from unittest.mock import Mock

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from quantized_oracles import bnb_nf4_fixture, compressed_tensors_fixture
from test_models import make_model, mutate_last_byte
from test_peft_adapters import composite_id, make_adapter, make_bases

from llm_model_explorer import model_files, models
from llm_model_explorer.app import create_app
from llm_model_explorer.dependencies import get_blocking_work, get_catalogue
from llm_model_explorer.execution import BlockingWork
from llm_model_explorer.model_files import FileSnapshot, ModelError
from llm_model_explorer.model_routes import router
from llm_model_explorer.models import ModelCatalogue
from llm_model_explorer.settings import Settings


@pytest.mark.parametrize("kind", ["native", "bnb", "compressed", "peft"])
def test_unchanged_calls_reuse_entries_and_diagnostics_without_asset_reads(
    model_root: Path, monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    if kind == "native":
        make_model(model_root)
    elif kind == "bnb":
        bnb_nf4_fixture().write(model_root)
    elif kind == "compressed":
        compressed_tensors_fixture().write(model_root)
    else:
        make_bases(model_root)
        make_adapter(model_root)
        rejected = model_root / "rejected"
        rejected.mkdir()
        (rejected / "adapter_config.json").write_text("{}")
    catalogue = ModelCatalogue(model_root)
    discover = Mock(wraps=catalogue._discover)
    monkeypatch.setattr(catalogue, "_discover", discover)
    entries = catalogue.discover()
    assert entries
    assert discover.call_count == 1
    cold = catalogue.list_catalogue()
    if kind == "peft":
        assert len(entries) == 4 and len(cold.diagnostics) == 1
    forbidden = Mock(side_effect=AssertionError("cache hit read or rebuilt model assets"))
    for name in ["read_json", "parse_header", "logical_locations", "bind_adapter"]:
        monkeypatch.setattr(models, name, forbidden)
    monkeypatch.setattr(model_files, "open_local", forbidden)
    monkeypatch.setattr(FileSnapshot, "open", forbidden)
    monkeypatch.setattr(FileSnapshot, "fingerprint", forbidden)
    assert catalogue.discover() is entries
    assert catalogue.list_models() == cold.models
    assert catalogue.list_catalogue() == cold
    assert discover.call_count == 1
    forbidden.assert_not_called()


def test_startup_warms_first_http_request_and_pin_still_hashes(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    make_model(settings.model_root)
    original = ModelCatalogue._discover
    discover = Mock()

    def tracked(catalogue: ModelCatalogue) -> models._Discovery:
        discover()
        return original(catalogue)

    monkeypatch.setattr(ModelCatalogue, "_discover", tracked)
    with TestClient(create_app(settings)) as client:
        assert discover.call_count == 1
        fingerprint = Mock(side_effect=AssertionError("listing hashed content"))
        with monkeypatch.context() as patches:
            patches.setattr(FileSnapshot, "fingerprint", fingerprint)
            cold = client.get("/models")
            warm = client.get("/models")
            assert cold.status_code == warm.status_code == 200
            assert cold.content == warm.content
            assert cold.headers["cache-control"] == "no-store"
            assert discover.call_count == 1
            fingerprint.assert_not_called()
        original_fingerprint = FileSnapshot.fingerprint
        hashes = Mock()

        def hashed(snapshot: FileSnapshot) -> str:
            hashes()
            return original_fingerprint(snapshot)

        monkeypatch.setattr(FileSnapshot, "fingerprint", hashed)
        assert client.post("/sessions", json={"model_id": "test/tiny"}).status_code == 201
        assert hashes.call_count == 1
        assert discover.call_count == 1


@pytest.mark.parametrize("change", ["add", "remove", "rename"])
def test_immediate_candidates_invalidate_next_call(
    model_root: Path, monkeypatch: pytest.MonkeyPatch, change: str
) -> None:
    directory = make_model(model_root, identity=None)
    catalogue = ModelCatalogue(model_root)
    assert [m.id for m in catalogue.list_models()] == ["tiny"]
    discover = Mock(wraps=catalogue._discover)
    monkeypatch.setattr(catalogue, "_discover", discover)
    if change == "add":
        make_model(model_root, "new", identity=None)
        expected = ["new", "tiny"]
    elif change == "remove":
        shutil.rmtree(directory)
        expected = []
    else:
        directory.rename(model_root / "renamed")
        expected = ["renamed"]
    assert [m.id for m in catalogue.list_models()] == expected
    catalogue.list_catalogue()
    assert discover.call_count == 1


@pytest.mark.parametrize(
    "change", ["add", "remove", "rename", "replace", "content", "size", "symlink", "fifo"]
)
def test_relevant_assets_invalidate_and_preserve_pinned_sources(
    model_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, change: str
) -> None:
    directory = make_model(model_root)
    weights = directory / "model.safetensors"
    catalogue = ModelCatalogue(model_root)
    source = catalogue.pin("test/tiny")
    discover = Mock(wraps=catalogue._discover)
    monkeypatch.setattr(catalogue, "_discover", discover)
    valid = True
    if change == "add":
        (directory / "tokenizer.json").write_text("{}")
    elif change == "remove":
        weights.unlink()
        valid = False
    elif change == "rename":
        weights.rename(directory / "renamed.safetensors")
    elif change == "replace":
        replacement = tmp_path / "replacement"
        replacement.write_bytes(weights.read_bytes())
        replacement.replace(weights)
    elif change == "content":
        before = weights.stat()
        mutate_last_byte(weights)
        os.utime(weights, ns=(before.st_atime_ns, before.st_mtime_ns))
        assert weights.stat().st_size == before.st_size
        assert weights.stat().st_mtime_ns == before.st_mtime_ns
    elif change == "size":
        with weights.open("ab") as stream:
            stream.write(b"x")
        valid = False
    elif change == "symlink":
        outside = make_model(tmp_path, "outside")
        weights.unlink()
        weights.symlink_to(outside / "model.safetensors")
        valid = False
    else:
        weights.unlink()
        os.mkfifo(weights)
        valid = False
    listing = catalogue.list_catalogue()
    assert len(listing.models) == int(valid)
    if change == "add":
        assert listing.models[0].tokenizer_available
    assert catalogue.list_catalogue() == listing
    assert discover.call_count == 1
    with pytest.raises(ModelError) as error:
        source.check_unchanged()
    assert error.value.code == "model_content_changed"
    if valid:
        fresh = catalogue.pin("test/tiny")
        assert len(fresh.tensors()) == 1
        if change == "content":
            assert fresh.fingerprint != source.fingerprint


def test_in_root_candidate_and_asset_symlink_retargeting(model_root: Path) -> None:
    first = make_model(model_root, "first", identity="org/first")
    second = make_model(model_root, "second", identity="org/second")
    link = model_root / "linked"
    link.symlink_to(first, target_is_directory=True)
    catalogue = ModelCatalogue(model_root)
    assert len(catalogue.list_models()) == 2
    entries = catalogue.discover()
    link.unlink()
    link.symlink_to(second, target_is_directory=True)
    assert catalogue.discover() is not entries
    weights = first / "model.safetensors"
    weights.unlink()
    weights.symlink_to(second / "model.safetensors")
    pinned = catalogue.pin("org/first")
    replacement = model_root / "replacement"
    replacement.write_bytes(weights.read_bytes())
    weights.unlink()
    weights.symlink_to(replacement)
    catalogue.list_catalogue()
    with pytest.raises(ModelError):
        pinned.check_unchanged()


@pytest.mark.parametrize("adapter", [False, True])
def test_rejected_indexed_candidates_observe_nested_shard_repairs(
    model_root: Path, adapter: bool
) -> None:
    if adapter:
        make_bases(model_root)
        directory = make_adapter(model_root, indexed=True)
        shard = directory / "weights/adapter-part.safetensors"
    else:
        directory = make_model(model_root)
        shard = directory / "weights/part.safetensors"
        shard.parent.mkdir()
        (directory / "model.safetensors").rename(shard)
        (directory / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": {"layer.weight": "weights/part.safetensors"}})
        )
    original = shard.read_bytes()
    shard.write_bytes(b"broken")
    catalogue = ModelCatalogue(model_root)
    rejected = catalogue.list_catalogue()
    assert len(rejected.models) == (2 if adapter else 0)
    if adapter:
        assert rejected.diagnostics[0].code == "peft_adapter_rejected"
    shard.write_bytes(original)
    repaired = catalogue.list_catalogue()
    assert len(repaired.models) == (4 if adapter else 1)
    assert repaired.diagnostics == ()
    assert catalogue.list_catalogue() == repaired
    shard.unlink()
    missing = catalogue.list_catalogue()
    assert missing.models == rejected.models
    assert missing == ModelCatalogue(model_root).list_catalogue()


@pytest.mark.parametrize("change", ["config", "index", "factors", "remove"])
def test_adapter_assets_refresh_compositions_and_diagnostics(
    model_root: Path, monkeypatch: pytest.MonkeyPatch, change: str
) -> None:
    make_bases(model_root)
    directory = make_adapter(model_root)
    catalogue = ModelCatalogue(model_root)
    source = catalogue.pin(composite_id())
    if change == "config":
        config = directory / "adapter_config.json"
        value = json.loads(config.read_text())
        value["r"] = 3
        config.write_text(json.dumps(value))
    elif change == "index":
        (directory / "adapter_model.safetensors.index.json").write_text(
            '{"weight_map":{"missing":"adapter_model.safetensors"}}'
        )
    elif change == "factors":
        weights = directory / "adapter_model.safetensors"
        raw = weights.read_bytes().replace(b'"shape": [2, 128]', b'"shape": [1, 256]')
        weights.write_bytes(raw)
    else:
        (directory / "adapter_model.safetensors").unlink()
    discover = Mock(wraps=catalogue._discover)
    monkeypatch.setattr(catalogue, "_discover", discover)
    listing = catalogue.list_catalogue()
    assert len(listing.models) == 2
    assert len(listing.diagnostics) == (2 if change in {"config", "factors"} else 1)
    assert listing.diagnostics[0].code == (
        "peft_composition_rejected" if change in {"config", "factors"} else "peft_adapter_rejected"
    )
    assert listing == ModelCatalogue(model_root).list_catalogue()
    assert catalogue.list_catalogue() == listing
    assert discover.call_count == 1
    with pytest.raises(ModelError, match="changed"):
        source.check_unchanged()


def test_duplicate_identity_refresh_hashes_and_failure_cannot_serve_old_result(
    model_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    directory = make_model(model_root)
    catalogue = ModelCatalogue(model_root)
    listing = catalogue.list_catalogue()
    duplicate = model_root / "duplicate"
    shutil.copytree(directory, duplicate)
    original = FileSnapshot.fingerprint
    hashes = Mock()

    def tracked(snapshot: FileSnapshot) -> str:
        hashes()
        return original(snapshot)

    monkeypatch.setattr(FileSnapshot, "fingerprint", tracked)
    assert catalogue.list_catalogue() == listing
    assert hashes.call_count == 2
    catalogue.list_catalogue()
    assert hashes.call_count == 2
    mutate_last_byte(duplicate / "model.safetensors")
    successful = catalogue._cached
    for _ in range(2):
        with pytest.raises(ModelError, match="Ambiguous"):
            catalogue.list_catalogue()
        assert catalogue._cached is successful
    assert hashes.call_count == 6
    shutil.rmtree(duplicate)
    assert catalogue.list_catalogue() == listing


def test_concurrent_invalidated_callers_share_one_discovery(
    model_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    make_model(model_root)
    catalogue = ModelCatalogue(model_root)
    catalogue.list_catalogue()
    make_model(model_root, "new", identity="org/new")
    barrier = Barrier(8)
    discover = Mock(wraps=catalogue._discover)
    monkeypatch.setattr(catalogue, "_discover", discover)

    def request(_: int) -> models.CatalogueListing:
        barrier.wait(timeout=5)
        return catalogue.list_catalogue()

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(request, range(8)))
    assert all(result == results[0] for result in results)
    assert len(results[0].models) == 2
    assert discover.call_count == 1


def test_abandoned_http_consumer_does_not_duplicate_refresh(
    model_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    make_model(model_root)
    catalogue = ModelCatalogue(model_root)
    catalogue.discover()
    make_model(model_root, "new", identity="org/new")
    entered, release = Event(), Event()
    subscribers_ready = Event()
    counter_lock = Lock()
    resolve_calls = 0
    original_resolve = catalogue._resolve

    def resolve() -> models._Discovery:
        nonlocal resolve_calls
        with counter_lock:
            resolve_calls += 1
            if resolve_calls == 4:
                subscribers_ready.set()
        return original_resolve()

    monkeypatch.setattr(catalogue, "_resolve", resolve)
    original = catalogue._discover
    calls = Mock()

    def delayed() -> models._Discovery:
        calls()
        entered.set()
        assert release.wait(5)
        return original()

    monkeypatch.setattr(catalogue, "_discover", delayed)

    async def scenario() -> None:
        work = BlockingWork()
        app = FastAPI()
        app.include_router(router)
        app.dependency_overrides[get_catalogue] = lambda: catalogue
        app.dependency_overrides[get_blocking_work] = lambda: work
        try:
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://test"
            ) as client:
                abandoned = asyncio.create_task(client.get("/models"))
                assert await asyncio.to_thread(entered.wait, 5)
                abandoned.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await abandoned
                subscribers = [asyncio.create_task(client.get("/models")) for _ in range(3)]
                assert await asyncio.to_thread(subscribers_ready.wait, 5)
                assert calls.call_count == 1
                release.set()
                responses = await asyncio.gather(*subscribers)
                assert all(response.status_code == 200 for response in responses)
                assert len({response.content for response in responses}) == 1
                assert len(responses[0].json()["models"]) == 2
                assert calls.call_count == 1
        finally:
            release.set()
            await work.aclose()

    asyncio.run(scenario())


@pytest.mark.parametrize("persistent", [False, True])
def test_mutation_during_refresh_never_publishes_mixed_generation(
    model_root: Path, monkeypatch: pytest.MonkeyPatch, persistent: bool
) -> None:
    directory = make_model(model_root)
    catalogue = ModelCatalogue(model_root)
    catalogue.list_catalogue()
    successful = catalogue._cached
    (directory / "tokenizer.json").write_text("{}")
    original = catalogue._discover
    calls = 0

    def racing() -> models._Discovery:
        nonlocal calls
        result = original()
        calls += 1
        if persistent or calls == 1:
            config = directory / "config.json"
            value = json.loads(config.read_text())
            value["_name_or_path"] = f"org/new{calls}"
            config.write_text(json.dumps(value))
        return result

    monkeypatch.setattr(catalogue, "_discover", racing)
    if persistent:
        with pytest.raises(ModelError, match="catalogue changed") as error:
            catalogue.list_catalogue()
        assert error.value.code == "validation_error"
        assert str(model_root) not in str(error.value)
        assert calls == 3 and catalogue._cached is successful
        monkeypatch.setattr(catalogue, "_discover", original)
        assert catalogue.list_models()[0].id == "org/new3"
    else:
        assert catalogue.list_models()[0].id == "org/new1"
        assert catalogue.list_models()[0].id == "org/new1"
        assert calls == 2


def test_refresh_failure_is_not_cached_and_later_request_retries(
    model_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    directory = make_model(model_root)
    catalogue = ModelCatalogue(model_root)
    catalogue.list_catalogue()
    successful = catalogue._cached
    (directory / "tokenizer.json").write_text("{}")
    original = catalogue._discover
    failure = Mock(side_effect=ModelError("internal_error", "Discovery failed.", 500))
    monkeypatch.setattr(catalogue, "_discover", failure)
    for _ in range(2):
        with pytest.raises(ModelError, match="Discovery failed"):
            catalogue.list_catalogue()
        assert catalogue._cached is successful
    assert failure.call_count == 2
    monkeypatch.setattr(catalogue, "_discover", original)
    assert catalogue.list_models()[0].tokenizer_available


def test_existing_http_session_rejects_change_after_catalogue_refresh(settings: Settings) -> None:
    directory = make_model(settings.model_root)
    with TestClient(create_app(settings)) as client:
        old = client.post("/sessions", json={"model_id": "test/tiny"}).json()["id"]
        mutate_last_byte(directory / "model.safetensors")
        assert client.get("/models").status_code == 200
        response = client.get(f"/sessions/{old}/tensors")
        assert response.status_code == 409
        assert response.json()["code"] == "model_content_changed"
        fresh = client.post("/sessions", json={"model_id": "test/tiny"})
        assert fresh.status_code == 201
        assert client.get(f"/sessions/{fresh.json()['id']}/tensors").status_code == 200

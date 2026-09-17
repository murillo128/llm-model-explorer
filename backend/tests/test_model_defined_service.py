"""Real files, startup/cache lifecycle and independent numeric-byte assertions."""

import asyncio
import json
import shutil
import struct
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import pytest

from llm_model_explorer import architecture_service
from llm_model_explorer.architecture_analysis.core import DescriptionRegistry
from llm_model_explorer.architecture_service import ArchitectureService
from llm_model_explorer.artifacts import ArtifactStore
from llm_model_explorer.execution import BlockingWork
from llm_model_explorer.model_files import ModelError
from llm_model_explorer.models import ModelCatalogue
from llm_model_explorer.settings import Settings
from llm_model_explorer.tensor_source import ModelSource

ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = ROOT / "examples/model-owned-architecture/architecture.json"


def local_model(root: Path, *, sidecar: bool = True) -> Path:
    directory = root / "owned"
    directory.mkdir()
    (directory / "config.json").write_text(json.dumps({"model_type": "my_new_model"}))
    header: dict[str, Any] = {"__metadata__": {"format": "pt"}}
    data = bytearray()
    for name, shape, values in [
        ("encoder.proj.weight", [4, 3], [i / 4 for i in range(12)]),
        ("encoder.proj.bias", [4], [0.0, 0.25, -0.25, 0.5]),
    ]:
        start = len(data)
        data.extend(struct.pack(f"<{len(values)}f", *values))
        header[name] = {"dtype": "F32", "shape": shape, "data_offsets": [start, len(data)]}
    raw = json.dumps(header).encode()
    raw += b" " * (-len(raw) % 8)
    (directory / "model.safetensors").write_bytes(struct.pack("<Q", len(raw)) + raw + data)
    if sidecar:
        shutil.copyfile(EXAMPLE, directory / "architecture.json")
    return directory


def prepare(settings: Settings) -> tuple[ArchitectureService, ModelCatalogue]:
    work = BlockingWork()
    service = ArchitectureService(
        ArtifactStore(settings.cache_dir, model_root=settings.model_root), work
    )
    catalogue = ModelCatalogue(settings.model_root)

    async def run() -> None:
        try:
            await service.prepare(catalogue)
        finally:
            await work.aclose()

    asyncio.run(run())
    return service, catalogue


def test_sidecar_precedence_native_values_read_only_and_warm_cache(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    directory = local_model(settings.model_root)
    (directory / "architecture_analysis.py").write_text("raise AssertionError('never import')\n")
    before = {p.name: p.read_bytes() for p in directory.iterdir()}
    for path in directory.iterdir():
        path.chmod(0o444)
    directory.chmod(0o555)
    forbidden = Mock(side_effect=AssertionError("model execution, fallback or payload access"))
    with monkeypatch.context() as patch:
        patch.setattr(ModelSource, "iter_tensor", forbidden)
        patch.setattr(DescriptionRegistry, "select", forbidden)
        service, catalogue = prepare(settings)
        source = catalogue.pin("owned")
        cold = service.lookup(source)
        assert json.loads(cold)["graph"]["scope"] == "model_defined"
        patch.setattr(architecture_service, "analyze_definition", forbidden)
        warm, warm_catalogue = prepare(settings)
        assert warm.lookup(warm_catalogue.pin("owned")) == cold
        forbidden.assert_not_called()
    graph = json.loads(cold)["graph"]
    assert graph["coverage"] == "complete"
    weight = next(p for p in graph["parameters"] if p["name"] == "encoder.proj.weight")
    actual = [
        v for chunk in source.iter_tensor(weight["inspection"]["tensor_id"]) for v in chunk.tolist()
    ]
    assert actual == [i / 4 for i in range(12)]
    assert before == {p.name: p.read_bytes() for p in directory.iterdir()}


@pytest.mark.parametrize("raw", ["{", '{"schema_version":2}', '{"a":1,"a":2}'])
def test_invalid_sidecar_never_falls_back_or_breaks_tensors(
    settings: Settings, monkeypatch: pytest.MonkeyPatch, raw: str
) -> None:
    directory = local_model(settings.model_root)
    (directory / "architecture.json").write_text(raw)
    # Even a known packaged discriminator cannot conceal a bad supplied definition.
    (directory / "config.json").write_text(
        json.dumps({"model_type": "llama", "architectures": ["LlamaForCausalLM"]})
    )
    forbidden = Mock(side_effect=AssertionError("fallback forbidden"))
    monkeypatch.setattr(DescriptionRegistry, "select", forbidden)
    service, catalogue = prepare(settings)
    source = catalogue.pin("owned")
    result = json.loads(service.lookup(source))
    assert result["status"] == "unavailable"
    assert result["reason"] == "analysis_failed"
    assert "no fallback" in result["diagnostics"][0]["message"]
    assert str(settings.model_root) not in json.dumps(result)
    assert len(source.tensors()) == 2
    assert sum(chunk.numel() for chunk in source.iter_tensor(source.tensors()[0].id)) > 0
    forbidden.assert_not_called()


def test_absent_sidecar_keeps_packaged_selection(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    local_model(settings.model_root, sidecar=False)
    calls: list[str] = []
    original = DescriptionRegistry.select

    def select(self: DescriptionRegistry, inputs: Any) -> Any:
        calls.append(str(inputs.configuration["model_type"]))
        return original(self, inputs)

    monkeypatch.setattr(DescriptionRegistry, "select", select)
    service, catalogue = prepare(settings)
    assert (
        json.loads(service.lookup(catalogue.pin("owned")))["reason"] == "unsupported_architecture"
    )
    assert calls == ["my_new_model"]


def test_definition_mutation_rejects_old_session_and_invalidates_cache(settings: Settings) -> None:
    directory = local_model(settings.model_root)
    service, catalogue = prepare(settings)
    old_source = catalogue.pin("owned")
    old = json.loads(service.lookup(old_source))
    sidecar = directory / "architecture.json"
    sidecar.write_text(sidecar.read_text().replace('"Encoder"', '"Changed"'))
    with pytest.raises(ModelError) as error:
        service.lookup(old_source)
    assert error.value.code == "model_content_changed"
    new_source = catalogue.pin("owned")
    assert new_source.fingerprint != old_source.fingerprint
    assert json.loads(service.lookup(new_source))["reason"] == "restart_required"
    restarted, fresh_catalogue = prepare(settings)
    new = json.loads(restarted.lookup(fresh_catalogue.pin("owned")))
    assert new["graph"]["graph_id"] != old["graph"]["graph_id"]
    assert any(n["label"] == "Changed" for n in new["graph"]["nodes"])
    sidecar.unlink()
    with pytest.raises(ModelError):
        restarted.lookup(new_source)


def test_bounded_definition_read_and_missing_file(settings: Settings) -> None:
    directory = local_model(settings.model_root, sidecar=False)
    source = ModelCatalogue(settings.model_root).pin("owned")
    assert source.architecture_definition(max_bytes=16) is None
    (directory / "architecture.json").write_bytes(b" " * 17)
    source = ModelCatalogue(settings.model_root).pin("owned")
    with pytest.raises(ModelError) as error:
        source.architecture_definition(max_bytes=16)
    assert error.value.code == "unsupported_size"


def test_http_model_owned_graph_contract_and_bad_definition_isolation(settings: Settings) -> None:
    # HTTP includes the unrelated tokenizer subsystem. Optional only in the
    # offline editing environment; Transformers is mandatory in normal CI.
    pytest.importorskip("transformers")
    from fastapi.testclient import TestClient

    from llm_model_explorer.app import create_app

    local_model(settings.model_root)
    with TestClient(create_app(settings)) as client:
        created = client.post("/sessions", json={"model_id": "owned"})
        assert created.status_code == 201
        sid = created.json()["id"]
        response = client.get(f"/sessions/{sid}/architecture")
        assert response.status_code == 200
        assert response.headers["cache-control"] == "no-store"
        assert "x-operation-id" not in response.headers
        assert response.json()["graph"]["scope"] == "model_defined"
        assert not client.get("/models").json()["models"][0]["tokenizer_available"]
        assert len(client.get(f"/sessions/{sid}/tensors").json()["tensors"]) == 2

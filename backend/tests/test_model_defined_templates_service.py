"""Shared sidecars cross the real checkpoint/startup/cache and HTTP boundaries."""

import json
import shutil
import struct
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import pytest

from llm_model_explorer import architecture_service
from llm_model_explorer.architecture_analysis.core import DescriptionRegistry
from llm_model_explorer.model_files import ModelError
from llm_model_explorer.settings import Settings
from llm_model_explorer.tensor_source import ModelSource
from test_model_defined_service import prepare

EXAMPLE = (
    Path(__file__).resolve().parents[2]
    / "examples/model-owned-architecture/shared-architecture.json"
)


def shared_model(root: Path) -> Path:
    directory = root / "shared"
    directory.mkdir()
    (directory / "config.json").write_text('{"model_type":"custom_shared_encoder"}\n')
    header: dict[str, Any] = {"__metadata__": {"format": "pt"}}
    payload = bytearray()
    for layer in range(2):
        entries = [(f"attn.{p}", [4, 4]) for p in ("q", "k", "v", "o")]
        entries += [("mlp.up", [8, 4]), ("mlp.down", [4, 8])]
        for suffix, shape in entries:
            start = len(payload)
            count = shape[0] * shape[1]
            payload.extend(struct.pack(f"<{count}f", *[float(layer + 1)] * count))
            header[f"blocks.{layer}.{suffix}.weight"] = {
                "dtype": "F32",
                "shape": shape,
                "data_offsets": [start, len(payload)],
            }
    raw = json.dumps(header).encode()
    raw += b" " * (-len(raw) % 8)
    (directory / "model.safetensors").write_bytes(struct.pack("<Q", len(raw)) + raw + payload)
    shutil.copyfile(EXAMPLE, directory / "architecture.json")
    return directory


def test_shared_cold_warm_cache_and_nonzero_tensor_values(
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    directory = shared_model(settings.model_root)
    before = {p.name: p.read_bytes() for p in directory.iterdir()}
    forbidden = Mock(side_effect=AssertionError("no execution/payload access/fallback"))
    with monkeypatch.context() as patch:
        patch.setattr(ModelSource, "iter_tensor", forbidden)
        patch.setattr(DescriptionRegistry, "select", forbidden)
        service, catalogue = prepare(settings)
        source = catalogue.pin("shared")
        cold = service.lookup(source)
        patch.setattr(architecture_service, "analyze_definition", forbidden)
        warm, warm_catalogue = prepare(settings)
        assert warm.lookup(warm_catalogue.pin("shared")) == cold
        forbidden.assert_not_called()
    graph = json.loads(cold)["graph"]
    assert [t["component_role"] for t in graph["templates"]] == ["attention", "mlp"]
    parameters = {p["id"]: p for p in graph["parameters"]}
    instance = graph["templates"][0]["instances"][1]
    q_id = next(p["parameter_id"] for p in instance["parameters"] if p["role"] == "q.weight")
    q = parameters[q_id]
    assert q["name"] == "blocks.1.attn.q.weight"
    values = [
        v for chunk in source.iter_tensor(q["inspection"]["tensor_id"]) for v in chunk.tolist()
    ]
    assert values == [2.0] * 16
    assert before == {p.name: p.read_bytes() for p in directory.iterdir()}


def test_template_mutation_invalidates_pin_and_bad_mapping_never_falls_back(
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    directory = shared_model(settings.model_root)
    service, catalogue = prepare(settings)
    source = catalogue.pin("shared")
    original = json.loads(service.lookup(source))
    sidecar = directory / "architecture.json"
    value = json.loads(sidecar.read_text())
    value["templates"][0]["label"] = "Renamed attention family"
    sidecar.write_text(json.dumps(value))
    with pytest.raises(ModelError) as error:
        service.lookup(source)
    assert error.value.code == "model_content_changed"
    assert json.loads(service.lookup(catalogue.pin("shared")))["reason"] == "restart_required"
    fresh, catalogue = prepare(settings)
    updated = json.loads(fresh.lookup(catalogue.pin("shared")))
    assert updated["graph"]["graph_id"] != original["graph"]["graph_id"]
    assert updated["graph"]["templates"][0]["label"] == "Renamed attention family"
    value["templates"][0]["instances"][1]["parameters"].pop()
    sidecar.write_text(json.dumps(value))
    forbidden = Mock(side_effect=AssertionError("packaged fallback forbidden"))
    monkeypatch.setattr(DescriptionRegistry, "select", forbidden)
    broken, catalogue = prepare(settings)
    result = json.loads(broken.lookup(catalogue.pin("shared")))
    assert result["status"] == "unavailable"
    assert result["reason"] == "analysis_failed"
    assert len(catalogue.pin("shared").tensors()) == 12
    forbidden.assert_not_called()


def test_http_publishes_model_supplied_shared_families(settings: Settings) -> None:
    pytest.importorskip("transformers")
    from fastapi.testclient import TestClient

    from llm_model_explorer.app import create_app

    shared_model(settings.model_root)
    with TestClient(create_app(settings)) as client:
        created = client.post("/sessions", json={"model_id": "shared"})
        assert created.status_code == 201
        response = client.get(f"/sessions/{created.json()['id']}/architecture")
        assert response.status_code == 200
        graph = response.json()["graph"]
        assert graph["scope"] == "model_defined" and len(graph["templates"]) == 2
        assert len(graph["repetitions"][0]["instances"]) == 2
        assert "x-operation-id" not in response.headers
        assert response.headers["cache-control"] == "no-store"

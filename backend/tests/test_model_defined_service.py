"""Real files, startup/cache lifecycle and independent numeric-byte assertions."""

import asyncio
import copy
import json
import shutil
import struct
import subprocess
import sys
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import pytest

from llm_model_explorer.architecture_analysis.core import DescriptionRegistry
from llm_model_explorer.architecture_analysis.model_defined import ModelDefinedValidator
from llm_model_explorer.architecture_service import ArchitectureService
from llm_model_explorer.artifacts import ArtifactStore
from llm_model_explorer.execution import BlockingWork
from llm_model_explorer.model_files import ModelError
from llm_model_explorer.models import ModelCatalogue
from llm_model_explorer.settings import Settings
from llm_model_explorer.tensor_source import ModelSource
from llm_model_explorer.validate_architecture_cli import main as validate_main
from llm_model_explorer.validate_architecture_cli import validate_directory

ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = ROOT / "examples/model-owned-architecture/architecture.json"
DIAGNOSTIC_CASES: list[dict[str, str]] = json.loads(
    (ROOT / "api/fixtures/model-defined-diagnostics.json").read_text()
)


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
    settings: Settings, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    directory = local_model(settings.model_root)
    (directory / "architecture_analysis.py").write_text("raise AssertionError('never import')\n")
    before = {p.name: p.read_bytes() for p in directory.iterdir()}
    for path in directory.iterdir():
        path.chmod(0o444)
    directory.chmod(0o555)
    assert validate_directory(directory)["status"] == "valid"
    assert validate_main([str(directory)]) == 0
    assert capsys.readouterr().out == "Valid model-owned architecture (complete).\n"
    forbidden = Mock(side_effect=AssertionError("model execution, fallback or payload access"))
    with monkeypatch.context() as patch:
        patch.setattr(ModelSource, "iter_tensor", forbidden)
        patch.setattr(DescriptionRegistry, "select", forbidden)
        assert validate_directory(directory)["status"] == "valid"
        service, catalogue = prepare(settings)
        source = catalogue.pin("owned")
        cold = service.lookup(source)
        assert json.loads(cold)["graph"]["scope"] == "model_defined"
        patch.setattr(ModelDefinedValidator, "validate", forbidden)
        warm, warm_catalogue = prepare(settings)
        assert warm.lookup(warm_catalogue.pin("owned")) == cold
        forbidden.assert_not_called()
    graph = json.loads(cold)["graph"]
    assert validate_directory(directory)["graph_id"] == graph["graph_id"]
    assert graph["coverage"] == "complete"
    weight = next(p for p in graph["parameters"] if p["name"] == "encoder.proj.weight")
    actual = [
        v for chunk in source.iter_tensor(weight["inspection"]["tensor_id"]) for v in chunk.tolist()
    ]
    assert actual == [i / 4 for i in range(12)]
    assert before == {p.name: p.read_bytes() for p in directory.iterdir()}


def test_installed_validator_accepts_relative_model_directory(tmp_path: Path) -> None:
    root = tmp_path / "models"
    root.mkdir()
    local_model(root)
    executable = Path(sys.executable).parent / "llm-model-explorer-validate-architecture"
    result = subprocess.run(
        [str(executable), "models/owned", "--json"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr or result.stdout
    document = json.loads(result.stdout)
    assert document["status"] == "valid"
    assert document["model_id"] == "owned"
    assert str(tmp_path) not in result.stdout


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
    assert result["diagnostics"][0]["message"].startswith("architecture.json")
    assert result["diagnostics"][0]["code"] != "analysis_failed"
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


def invalid_definition(name: str) -> bytes:
    value: dict[str, Any] = json.loads(EXAMPLE.read_text())
    if name == "invalid_utf8":
        return b"\xff"
    if name == "malformed_json":
        return b"{"
    if name == "duplicate_json_key":
        return b'{"nodes":1,"nodes":2}'
    if name == "non_finite_constant":
        return b'{"value":NaN}'
    if name == "unsupported_schema_version":
        value["schema_version"] = 2
    elif name == "missing_operation":
        value["nodes"][3].pop("operation")
    elif name == "unknown_field":
        value["nodes"][3]["unexpected"] = "untrusted"
    elif name == "duplicate_ids":
        value["nodes"].append(copy.deepcopy(value["nodes"][0]))
    elif name == "duplicate_port_ids":
        value["nodes"][3]["ports"].append(copy.deepcopy(value["nodes"][3]["ports"][0]))
    elif name == "duplicate_child":
        value["nodes"][1]["children"].append("projection")
    elif name == "broken_containment":
        value["nodes"][1]["children"].remove("projection")
    elif name == "cycle":
        value["nodes"][1]["parent_id"] = "encoder"
        value["nodes"][1]["children"].append("encoder")
    elif name == "unknown_endpoint":
        value["edges"][0]["target"]["node_id"] = "absent"
    elif name == "edge_direction":
        value["edges"][2]["source"]["port_id"] = "x"
    elif name == "boundary_bypass":
        value["edges"][0]["target"]["node_id"] = "projection"
    elif name == "dimension_mismatch":
        value["nodes"][3]["ports"][0]["shape"][1]["value"] = 5
    elif name == "unsafe_dimension_product":
        value["nodes"][0]["ports"][0]["shape"] = [
            {"kind": "constant", "value": 2**53 - 1},
            {"kind": "constant", "value": 2},
        ]
    elif name == "tokenizer_capability":
        value["nodes"][3]["references"] = [{"kind": "tokenizer"}]
    elif name == "unknown_shape_symbol":
        value["nodes"][0]["ports"][0]["shape"][0]["name"] = "missing"
    elif name == "unknown_parameter_reference":
        value["nodes"][2]["parameter_ids"].append("absent")
    elif name == "repetition_membership":
        value["repetitions"] = [
            {
                "id": "layers",
                "parent_id": "encoder",
                "label": "Layers",
                "instances": [{"node_id": "projection", "index": 0, "variant": "default"}],
            }
        ]
    elif name in {"repetition_order", "duplicate_repetition_instance", "template_mapping"}:
        value["nodes"] += [
            {
                "id": key,
                "kind": "group",
                "label": key.upper(),
                "parent_id": "encoder",
                "children": [],
                "attributes": [{"name": "semantic_role", "value": "attention"}],
            }
            for key in ("a", "b")
        ]
        value["nodes"][1]["children"] += ["a", "b"]
        if name in {"repetition_order", "duplicate_repetition_instance"}:
            value["repetitions"] = [
                {
                    "id": "layers",
                    "parent_id": "encoder",
                    "label": "Layers",
                    "instances": [
                        {
                            "node_id": "a",
                            "index": 1 if name == "repetition_order" else 0,
                            "variant": "default",
                        },
                        {
                            "node_id": "b" if name == "repetition_order" else "a",
                            "index": 0 if name == "repetition_order" else 1,
                            "variant": "default",
                        },
                    ],
                }
            ]
        else:
            value["templates"] = [
                {
                    "id": "shared",
                    "label": "Shared",
                    "component_role": "attention",
                    "instances": [
                        {"node_id": "a", "nodes": [], "ports": [], "edges": [], "parameters": []},
                        {
                            "node_id": "b",
                            "nodes": [{"role": "root", "node_id": "b"}],
                            "ports": [],
                            "edges": [],
                            "parameters": [],
                        },
                    ],
                }
            ]
    elif name == "absent_tensor":
        value["parameters"][0]["name"] = "absent.weight"
    elif name == "wrong_shape":
        value["parameters"][0]["shape"][0]["value"] = 5
    elif name == "unsupported_size":
        return b" " * (8 * 1024 * 1024 + 1)
    return json.dumps(value).encode()


def test_cli_keeps_admission_and_untrusted_source_text_private(
    settings: Settings, capsys: pytest.CaptureFixture[str]
) -> None:
    directory = local_model(settings.model_root)
    (directory / "architecture.json").write_bytes(
        ('{"' + "/private/secret-" * 1000 + '":1,"' + "/private/secret-" * 1000 + '":2}').encode()
    )
    result = validate_directory(directory)
    assert result["status"] == "invalid"
    serialized = json.dumps(result)
    assert "/private/" not in serialized
    assert "Traceback" not in serialized
    assert len(serialized) < 512
    assert validate_main([str(directory)]) == 1
    human = capsys.readouterr().out
    assert "json_duplicate_key" in human
    assert "/private/" not in human
    assert validate_directory(directory / "missing") == {
        "status": "invalid",
        "reason": "analysis_failed",
        "diagnostics": [
            {
                "code": "model_admission_failed",
                "message": (
                    "Model directory could not be admitted from local config "
                    "and Safetensors metadata."
                ),
            }
        ],
    }


@pytest.mark.parametrize("case", DIAGNOSTIC_CASES, ids=lambda case: case["name"])
def test_canonical_failure_matches_cli_startup_and_get(
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    case: dict[str, str],
) -> None:
    from fastapi.testclient import TestClient

    from llm_model_explorer.app import create_app

    directory = local_model(settings.model_root)
    (directory / "architecture.json").write_bytes(invalid_definition(case["name"]))
    assert validate_main([str(directory), "--json"]) == 1
    cli = json.loads(capsys.readouterr().out)
    finding = {"code": case["code"], "message": case["message"]}
    assert cli["status"] == "invalid"
    assert cli["diagnostics"] == [finding]
    app = create_app(settings)
    with TestClient(app) as client:
        services = app.state.services
        source = services.catalogue.pin("owned")
        prepared = services.architectures._prepared[("owned", source.fingerprint)]
        assert prepared.failure is not None
        assert prepared.failure["diagnostics"] == [finding]
        assert prepared.failure["reason"] == cli["reason"]
        assert prepared.failure["requires_restart"] is True
        monkeypatch.setattr(
            ModelDefinedValidator,
            "validate",
            Mock(side_effect=AssertionError("GET must not validate")),
        )
        created = client.post("/sessions", json={"model_id": "owned"})
        assert created.status_code == 201
        response = client.get(f"/sessions/{created.json()['id']}/architecture")
        assert response.status_code == 200
        assert response.json()["diagnostics"] == [finding]
        assert services.architectures.lookup(source) == response.content
    public = json.dumps(cli)
    assert str(directory) not in public
    assert "Traceback" not in public
    assert len(case["message"]) <= 512

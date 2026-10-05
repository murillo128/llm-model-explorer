"""CLM-owned gaps; generic Qwen/importer/transport matrices remain with their owners."""

import json
import math
import os
import pickle
import shutil
from dataclasses import replace
from pathlib import Path
from typing import Any, cast
from unittest.mock import Mock

import pytest
import requests
import torch
from clm_fixtures import exporter, fixture
from fastapi.testclient import TestClient
from safetensors.torch import load_file, save_file
from test_tensor_data import frames

from llm_model_explorer.app import create_app
from llm_model_explorer.architecture_analysis import AnalysisInput, DescriptionRegistry
from llm_model_explorer.architecture_service import packaged_registry
from llm_model_explorer.model_files import ModelError
from llm_model_explorer.models import ModelCatalogue
from llm_model_explorer.settings import Settings
from llm_model_explorer.tensor_source import ModelSource


def test_native_clm_selection_without_sidecar(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "models"
    encoder, head, _ = fixture(root)
    destination = root / "clm"
    export(encoder, head, destination)
    # Selection must depend on explicit inspection metadata and inventory, not a sidecar.
    assert not (destination / "architecture.json").exists()
    payload_access = Mock(side_effect=AssertionError("Analysis read numerical tensor values"))
    monkeypatch.setattr(ModelSource, "iter_tensor", payload_access)
    with TestClient(create_app(Settings(model_root=root, cache_dir=tmp_path / "cache"))) as client:
        identity = "Contrastive-LM/CLM-v0.1-8B@" + "2" * 40
        sid = client.post("/sessions", json={"model_id": identity}).json()["id"]
        response = client.get(f"/sessions/{sid}/architecture").json()
    assert response["status"] == "available", response
    graph = response["graph"]
    assert graph["scope"] == "language_model" and graph["coverage"] == "complete"
    assert graph["diagnostics"] == []
    layer_families = [t for t in graph.get("templates", []) if t["component_role"] == "layer"]
    assert len(layer_families) == 2  # Independent state/candidate encoder scopes.
    assert [len(t["instances"]) for t in layer_families] == [2, 2]

    payload_access.assert_not_called()
    assert any(
        p["source"] == "clm-inspection"
        and "bb42c6c5bf914fd449bed2f6ca65be80602cb1f7" in p.get("rule", "")
        for node in graph["nodes"]
        for p in node["provenance"]
    )
    semantic = native_graph(destination)
    nodes = {n["id"]: n for n in semantic["nodes"]}
    edges = {
        (
            e["source"]["node_id"],
            e["source"]["port_id"],
            e["target"]["node_id"],
            e["target"]["port_id"],
        )
        for e in semantic["edges"]
    }
    for call, batch, seq, head_name in (
        ("state_encoder", "B", "S_state", "state_head"),
        ("candidate_encoder", "C", "S_candidate", "action_head"),
    ):
        pool = nodes[call + ".pool"]
        assert pool["operation"] == "last_token_pool"
        assert pool["formula"] == "out = x[row, lengths[row] − 1]"
        assert next(p["shape"] for p in pool["ports"] if p["id"] == "x") == [
            {"kind": "symbol", "name": batch},
            {"kind": "symbol", "name": seq},
            {"kind": "constant", "value": 4},
        ]
        assert nodes[call + ".normalize"]["operation"] == "encoder_normalize"
        assert nodes[call + ".projection_normalize"]["operation"] == "l2_normalize"
        for source, target in (
            (call, call + ".pool"),
            (call + ".pool", call + ".normalize"),
            (call + ".normalize", "clm." + head_name),
            ("clm." + head_name, call + ".projection_normalize"),
        ):
            assert (source, "out", target, "x") in edges
    assert nodes["scale"]["operation"] == "exp_clamp"
    assert nodes["scale"]["parameter_ids"] == ["clm.logit_scale"]
    assert nodes["scale"]["references"] == [{"kind": "module", "name": "clm"}]
    for head_name in ("state_head", "action_head"):
        projection = nodes[f"clm.{head_name}.inp"]
        assert projection["formula"] == "out = x @ weightᵀ + bias"
        assert [(p["label"], p["direction"]) for p in projection["ports"]] == [
            ("x", "input"),
            ("out", "output"),
        ]
        assert projection["references"] == [{"kind": "module", "name": f"clm.{head_name}.inp"}]
        assert nodes[f"clm.{head_name}"].get("parameter_ids", []) == []
    assert nodes["probabilities"]["operation"] == "softmax"
    assert {a["name"]: a["value"] for a in nodes["probabilities"]["attributes"]} == {
        "axis": -1,
        "axis_meaning": "candidates for each state",
    }
    assert {
        ("cosine", "out", "scores", "cosine"),
        ("scale", "out", "scores", "scale"),
        ("temperature", "out", "scores", "temperature"),
        ("scores", "out", "probabilities", "x"),
    } <= edges


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("format_version",), True),
        (("encoder_repository",), "Qwen/Qwen3-8B-Base"),
        (("encoder_revision",), "main"),
        (("head_revision",), "3" * 40),
        (("pooling",), "mean"),
        (("head_configuration", "hidden_size"), 5),
        (("head_configuration", "depth"), 17),
        (("head_configuration", "width"), True),
        (("head_configuration", "layernorm"), "true"),
        (("head_configuration", "activation"), "swiglu"),
        (("head_configuration", "unknown_option"), True),
    ],
)
def test_native_clm_rejects_inconsistent_metadata(
    tmp_path: Path, path: tuple[str, ...], value: object
) -> None:
    root = tmp_path / "models"
    encoder, head, _ = fixture(root)
    destination = root / "clm"
    export(encoder, head, destination)
    entry = ModelCatalogue(root).inspect_directory(destination)
    inputs = AnalysisInput.from_source(entry.pin(), tokenizer_available=True)
    config = dict(inputs.configuration)
    target = cast(dict[str, Any], config["clm_inspection"])
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    result = packaged_registry().analyze(replace(inputs, configuration=config))
    assert result.graph is None


@pytest.mark.parametrize(
    "damage", ["scale_shape", "head_dtype", "extra_head", "numeric_geometry", "missing_encoder"]
)
def test_native_clm_rejects_inconsistent_inventory(tmp_path: Path, damage: str) -> None:
    root = tmp_path / "models"
    encoder, head, _ = fixture(root)
    destination = root / "clm"
    export(encoder, head, destination)
    entry = ModelCatalogue(root).inspect_directory(destination)
    inputs = AnalysisInput.from_source(entry.pin(), tokenizer_available=True)
    physical, numeric = dict(inputs.bindings.physical), dict(inputs.bindings.numeric)
    name = "clm.action_head.out.weight"
    if damage == "scale_shape":
        physical["clm.logit_scale"] = physical["clm.logit_scale"].model_copy(update={"shape": [1]})
    elif damage == "head_dtype":
        physical[name] = physical[name].model_copy(update={"dtype": "I32"})
    elif damage == "extra_head":
        physical["clm.action_head.orphan"] = physical[name].model_copy(
            update={"name": "clm.action_head.orphan"}
        )
    elif damage == "numeric_geometry":
        tensor = next(t for t in numeric.values() if t.name == name)
        numeric[tensor.id] = replace(tensor, shape=(3, 3))
    else:
        del physical["model.layers.0.self_attn.q_norm.weight"]
    bindings = replace(inputs.bindings, physical=physical, numeric=numeric)
    assert packaged_registry().analyze(replace(inputs, bindings=bindings)).graph is None


def export(encoder: Path, head: Path, destination: Path) -> dict[str, Any]:
    return dict(
        exporter().export_package(
            encoder, head, destination, encoder_revision="1" * 40, head_revision="2" * 40
        )
    )


def native_graph(directory: Path) -> dict[str, Any]:
    entry = ModelCatalogue(directory.parent).inspect_directory(directory)
    inputs = AnalysisInput.from_source(entry.pin(), tokenizer_available=True)
    result = packaged_registry().analyze(inputs)
    assert result.graph is not None and result.graph.coverage == "complete", result.diagnostics
    graph = result.graph.document()
    # Read public semantic provenance to normalize opaque IDs for the existing equations.
    node_keys = {
        n["id"]: next(
            p["source"]
            for p in n["provenance"]
            if p.get("rule") == "Semantic source key in the reviewed packaged description"
        )
        for n in graph["nodes"]
    }
    parameter_names = {p["id"]: p["name"] for p in graph["parameters"]}
    for node in graph["nodes"]:
        node["id"] = node_keys[node["id"]]
        node["parameter_ids"] = [parameter_names[p] for p in node["parameter_ids"]]
        if "parent_id" in node:
            node["parent_id"] = node_keys[node["parent_id"]]
        if "children" in node:
            node["children"] = [node_keys[c] for c in node["children"]]
    for edge in graph["edges"]:
        for end in ("source", "target"):
            edge[end]["node_id"] = node_keys[edge[end]["node_id"]]
    return graph


def input_tree(root: Path) -> dict[str, tuple[int, bytes | str | None]]:
    """Capture directories, symlinks, permissions and all original file bytes."""
    return {
        p.relative_to(root).as_posix(): (
            p.lstat().st_mode,
            p.readlink().as_posix() if p.is_symlink() else p.read_bytes() if p.is_file() else None,
        )
        for p in [root, *root.rglob("*")]
    }


@pytest.mark.parametrize("alias", ["symlink", "traversal"])
@pytest.mark.parametrize("copy_shards", [False, True])
def test_resolved_output_inside_encoder_is_rejected(
    tmp_path: Path, alias: str, copy_shards: bool
) -> None:
    root = tmp_path / "models"
    encoder, head, _ = fixture(root)
    (encoder / "notes").mkdir()
    (encoder / "notes" / "original.txt").write_bytes(b"unchanged nested input")
    (encoder / "notes" / "weight-link").symlink_to("../encoder.safetensors")
    if alias == "symlink":
        parent = tmp_path / "encoder-alias"
        parent.symlink_to(encoder, target_is_directory=True)
        destination = parent / "published"
    else:
        parent = tmp_path / "other"
        parent.mkdir()
        destination = parent / ".." / "models" / "qwen" / "published"
    before, head_bytes = input_tree(encoder), head.read_bytes()
    with pytest.raises(ValueError, match="Destination must be new and outside the encoder"):
        exporter().export_package(
            encoder,
            head,
            destination,
            encoder_revision="1" * 40,
            head_revision="2" * 40,
            copy_shards=copy_shards,
        )
    assert input_tree(encoder) == before
    assert head.read_bytes() == head_bytes
    assert not destination.exists()
    assert not list(tmp_path.rglob(".clm-export-*"))
    assert not any(
        entry.summary.id.startswith("Contrastive-LM/") for entry in ModelCatalogue(root).discover()
    )


@pytest.mark.skipif(os.name != "posix" or os.geteuid() == 0, reason="Requires non-root POSIX modes")
@pytest.mark.parametrize("copy_shards", [False, True])
def test_writable_output_under_read_only_ancestor(tmp_path: Path, copy_shards: bool) -> None:
    ancestor = tmp_path / "readonly"
    root = ancestor / "models"
    encoder, head, _ = fixture(root)
    before, head_bytes = input_tree(encoder), head.read_bytes()
    ancestor.chmod(0o555)
    try:
        # These are actual OS permissions, not mocked access checks.
        with pytest.raises(PermissionError):
            (ancestor / "forbidden").mkdir()
        destination = root / "clm"
        result = exporter().export_package(
            encoder,
            head,
            destination,
            encoder_revision="1" * 40,
            head_revision="2" * 40,
            copy_shards=copy_shards,
        )
        assert result["status"] == "valid"
        assert (destination / "encoder.safetensors").is_symlink() is not copy_shards
        assert (destination / "encoder.safetensors").read_bytes() == (
            encoder / "encoder.safetensors"
        ).read_bytes()
        assert input_tree(encoder) == before
        assert head.read_bytes() == head_bytes
        assert not list(root.glob(".clm-export-*"))
        source = ModelCatalogue(root).pin("Contrastive-LM/CLM-v0.1-8B@" + "2" * 40)
        source.check_unchanged()
    finally:
        ancestor.chmod(0o755)


@pytest.mark.parametrize("outcome", ["valid", "invalid_binding", "interrupted"])
def test_only_validated_atomic_publication_is_selectable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, outcome: str
) -> None:
    root = tmp_path / "models"
    encoder, head, _ = fixture(root)
    before, head_bytes = input_tree(encoder), head.read_bytes()
    destination = root / "clm"
    identity = "Contrastive-LM/CLM-v0.1-8B@" + "2" * 40
    catalogue = ModelCatalogue(root)
    original_ids = [e.summary.id for e in catalogue.discover()]
    module = exporter()
    real_validate = module.validate_directory
    validated_directory: list[tuple[int, int]] = []

    def observe_validation(directory: Path, **kwargs: Any) -> dict[str, object]:
        if outcome == "invalid_binding":
            path = directory / "config.json"
            config = json.loads(path.read_text())
            config["clm_inspection"]["head_configuration"]["width"] += 1
            path.write_text(json.dumps(config))
        assert not destination.exists()
        assert [e.summary.id for e in catalogue.discover()] == original_ids
        with pytest.raises(ModelError, match="Unknown model"):
            catalogue.pin(identity)
        stat = directory.stat()
        assert stat.st_dev == destination.parent.stat().st_dev
        result: dict[str, object] = real_validate(directory, **kwargs)
        assert result["status"] == ("invalid" if outcome == "invalid_binding" else "valid")
        assert not destination.exists()
        assert [e.summary.id for e in catalogue.discover()] == original_ids
        if outcome == "interrupted":
            raise RuntimeError("Interrupted after real validation")
        validated_directory.append((stat.st_dev, stat.st_ino))
        return result

    monkeypatch.setattr(module, "validate_directory", observe_validation)

    def run() -> dict[str, Any]:
        return dict(
            module.export_package(
                encoder, head, destination, encoder_revision="1" * 40, head_revision="2" * 40
            )
        )

    if outcome == "valid":
        assert run()["status"] == "valid"
        stat = destination.stat()
        # The validated directory itself was renamed; publication did not copy
        # assets into a selectable destination incrementally.
        assert validated_directory == [(stat.st_dev, stat.st_ino)]
        assert {e.summary.id for e in catalogue.discover()} == {*original_ids, identity}
        source = catalogue.pin(identity)
        assert len(source.physical_tensors()) == 42
        source.check_unchanged()
    else:
        with pytest.raises(
            (ValueError, RuntimeError),
            match="failed static import|Interrupted after real validation",
        ):
            run()
        assert not destination.exists()
        assert [e.summary.id for e in catalogue.discover()] == original_ids
        with pytest.raises(ModelError, match="Unknown model"):
            catalogue.pin(identity)
    assert input_tree(encoder) == before
    assert head.read_bytes() == head_bytes
    assert not list(tmp_path.rglob(".clm-export-*"))


@pytest.mark.parametrize(
    "metadata",
    [
        {"_name_or_path": "Qwen/Qwen3-8B-Base"},
        {"_name_or_path": "Qwen/Qwen3-8B-Base", "name_or_path": "Qwen/Qwen3-8B"},
        {"name_or_path": "Qwen/Qwen3-8B-Base"},
        {"_commit_hash": "3" * 40},
        {"revision": "3" * 40},
        {"_commit_hash": "1" * 40, "revision": "3" * 40},
    ],
)
def test_conflicting_encoder_binding_is_never_published(
    tmp_path: Path, metadata: dict[str, str]
) -> None:
    """Identical tensors/geometry do not make a different source CLM-compatible."""
    root = tmp_path / "models"
    encoder, head, _ = fixture(root)
    path = encoder / "config.json"
    config = json.loads(path.read_text())
    config.update(metadata)
    path.write_text(json.dumps(config))
    before = path.read_bytes()
    destination = root / "clm"
    with pytest.raises(ValueError, match="Encoder .* metadata"):
        export(encoder, head, destination)
    assert path.read_bytes() == before
    assert not destination.exists()
    assert not list(tmp_path.glob(".clm-export-*"))
    assert not any(
        model.id.startswith("Contrastive-LM/") for model in ModelCatalogue(root).list_models()
    )


def test_matching_encoder_binding_preserves_source_declarations(tmp_path: Path) -> None:
    root = tmp_path / "models"
    encoder, head, _ = fixture(root)
    path = encoder / "config.json"
    config = json.loads(path.read_text())
    metadata = {
        "_name_or_path": "Qwen/Qwen3-8B",
        "name_or_path": "Qwen/Qwen3-8B",
        "_commit_hash": "1" * 40,
        "revision": "1" * 40,
    }
    config.update(metadata)
    path.write_text(json.dumps(config))
    before = path.read_bytes()
    destination = root / "clm"
    assert export(encoder, head, destination)["status"] == "valid"
    provenance = json.loads((destination / "clm-provenance.json").read_text())
    assert provenance["encoder"]["binding"] == {
        "configuration": metadata,
        "repository_source": "configuration",
        "revision_source": "configuration",
    }
    assert path.read_bytes() == before


@pytest.mark.parametrize(
    "activation,layernorm,residual,depth,dtype",
    [
        ("gelu", True, False, 3, torch.float32),
        ("relu", False, True, 4, torch.float16),
        ("silu", True, True, 3, torch.bfloat16),
        ("gelu", False, False, 2, torch.float32),
    ],
)
def test_preserved_inventory_sharing_and_head_options(
    tmp_path: Path, activation: str, layernorm: bool, residual: bool, depth: int, dtype: torch.dtype
) -> None:
    root = tmp_path / "models"
    encoder, head, original = fixture(
        root,
        activation=activation,
        layernorm=layernorm,
        residual=residual,
        depth=depth,
        dtype=dtype,
    )
    before = {p.name: p.read_bytes() for p in encoder.iterdir() if p.is_file()}
    head_bytes = head.read_bytes()
    destination = root / "clm"
    result = export(encoder, head, destination)
    assert result["status"] == "valid" and result["coverage"] == "complete"
    assert {p.name: p.read_bytes() for p in encoder.iterdir() if p.is_file()} == before
    assert head.read_bytes() == head_bytes
    assert (destination / "encoder.safetensors").is_symlink()
    base = load_file(str(encoder / "encoder.safetensors"))
    exported_base = load_file(str(destination / "encoder.safetensors"))
    assert base.keys() == exported_base.keys()
    for name, value in base.items():
        assert torch.equal(value, exported_base[name])
        assert value.dtype == exported_base[name].dtype
    heads = load_file(str(destination / "clm-heads.safetensors"))
    for name in ("state_head", "action_head"):
        for key, value in original[name].items():
            actual = heads[f"clm.{name}.{key}"]
            assert torch.equal(value, actual) and value.dtype == actual.dtype
    assert torch.equal(heads["clm.logit_scale"], original["logit_scale"])
    index = json.loads((destination / "model.safetensors.index.json").read_text())["weight_map"]
    assert set(index) == set(base) | set(heads)
    assert not (destination / "architecture.json").exists()
    graph = native_graph(destination)
    names = {p["name"] for p in graph["parameters"]}
    assert names == (set(base) - {"lm_head.weight"}) | set(heads)
    for name in set(base) - {"lm_head.weight"}:
        consumers = [n for n in graph["nodes"] if name in n.get("parameter_ids", [])]
        assert len(consumers) == 2
        assert {n["id"].split(".")[0] for n in consumers} == {"state_encoder", "candidate_encoder"}
    for name in heads:
        assert sum(name in n.get("parameter_ids", []) for n in graph["nodes"]) == 1
    assert [len(r["instances"]) for r in graph["repetitions"]] == [2, 2]
    assert not any(n.get("operation") in {"tokenizer", "logits"} for n in graph["nodes"])
    assert not any("lm_head" in n.get("parameter_ids", []) for n in graph["nodes"])
    provenance = json.loads((destination / "clm-provenance.json").read_text())
    assert provenance["encoder"]["binding"] == {
        "configuration": {"_name_or_path": "Qwen/Qwen3-8B"},
        "repository_source": "configuration",
        "revision_source": "operator_selection",
    }
    pairs = [
        (
            e["source"]["node_id"],
            e["source"]["port_id"],
            e["target"]["node_id"],
            e["target"]["port_id"],
        )
        for e in graph["edges"]
    ]
    assert len(pairs) == len(set(pairs))
    for which in ("state_head", "action_head"):
        projected = evaluate_head(graph, heads, which, [0.25, -0.5, 0.75, 1.0])
        expected = oracle(original[which], original["cfg"], [0.25, -0.5, 0.75, 1.0])
        # Float32 graph evaluation versus explicit float64 scalar oracle; no model inference.
        assert projected.tolist() == pytest.approx(expected, abs=2e-6, rel=2e-6)
    vectors = [
        oracle(original["action_head"], original["cfg"], x)
        for x in [[1.0, 0.0, -1.0, 0.5], [-0.5, 0.25, 0.5, -1.0]]
    ]
    state = oracle(original["state_head"], original["cfg"], [0.25, -0.5, 0.75, 1.0])
    state = [v / max(math.sqrt(sum(z * z for z in state)), 1e-12) for v in state]
    vectors = [[v / max(math.sqrt(sum(z * z for z in row)), 1e-12) for v in row] for row in vectors]
    expected = [
        100 * sum(a * b for a, b in zip(state, candidate, strict=True)) for candidate in vectors
    ]
    actual_state = torch.nn.functional.normalize(
        evaluate_head(graph, heads, "state_head", [0.25, -0.5, 0.75, 1.0]), dim=-1
    )
    actual_candidates = torch.stack(
        [
            torch.nn.functional.normalize(evaluate_head(graph, heads, "action_head", x), dim=-1)
            for x in [[1.0, 0.0, -1.0, 0.5], [-0.5, 0.25, 0.5, -1.0]]
        ]
    )
    scale_node = next(n for n in graph["nodes"] if n["id"] == "scale")
    assert scale_node["operation"] == "exp_clamp"
    maximum = next(a["value"] for a in scale_node["attributes"] if a["name"] == "maximum")
    scores = heads["clm.logit_scale"].float().exp().clamp(max=maximum) * (
        actual_candidates @ actual_state
    )
    assert scores.tolist() == pytest.approx(expected, abs=2e-4)
    probabilities = [math.exp(x - max(expected)) for x in expected]
    probabilities = [p / sum(probabilities) for p in probabilities]
    assert torch.softmax(scores, dim=-1).tolist() == pytest.approx(probabilities, abs=2e-6)


def evaluate_head(
    graph: dict[str, Any], weights: dict[str, torch.Tensor], which: str, vector: list[float]
) -> torch.Tensor:
    """Execute the exported head's actual dependencies using documented operations."""
    key = "clm." + which
    x = torch.tensor(vector)
    x = x / (torch.linalg.vector_norm(x) + 1e-12)
    values = {key: x}
    for node in graph["nodes"]:
        if node.get("parent_id") != key:
            continue
        args = {
            e["target"]["port_id"]: values[e["source"]["node_id"]]
            for e in graph["edges"]
            if e["target"]["node_id"] == node["id"]
        }
        op = node["operation"]
        if op == "linear":
            w, b = [weights[p].float() for p in node["parameter_ids"]]
            out = torch.nn.functional.linear(args["x"], w, b)
        elif op == "layer_norm":
            w, b = [weights[p].float() for p in node["parameter_ids"]]
            epsilon = next(a["value"] for a in node["attributes"] if a["name"] == "epsilon")
            out = torch.nn.functional.layer_norm(args["x"], w.shape, w, b, epsilon)
        elif op == "add":
            out = args["skip"] + args["branch"]
        elif op == "gelu":
            out = torch.nn.functional.gelu(args["x"])
        elif op == "relu":
            out = torch.nn.functional.relu(args["x"])
        else:
            assert op == "silu"
            out = torch.nn.functional.silu(args["x"])
        values[node["id"]] = out
    output = next(e for e in graph["edges"] if e["target"] == {"node_id": key, "port_id": "out"})
    return cast(torch.Tensor, values[output["source"]["node_id"]])


def oracle(weights: dict[str, torch.Tensor], cfg: dict[str, Any], x: list[float]) -> list[float]:
    """Float64 scalar equations from pinned heads.py/embedder.py, independent of export."""
    x = [v / (math.sqrt(sum(z * z for z in x)) + 1e-12) for v in x]

    def linear(prefix: str, value: list[float]) -> list[float]:
        matrix = weights[prefix + ".weight"].double().tolist()
        bias = weights[prefix + ".bias"].double().tolist()
        return [
            sum(a * b for a, b in zip(row, value, strict=True)) + offset
            for row, offset in zip(matrix, bias, strict=True)
        ]

    def activation(value: list[float]) -> list[float]:
        if cfg["activation"] == "gelu":
            return [v * 0.5 * (1 + math.erf(v / math.sqrt(2))) for v in value]
        if cfg["activation"] == "relu":
            return [max(v, 0.0) for v in value]
        return [v / (1 + math.exp(-v)) for v in value]

    x = activation(linear("inp", x))
    for i in range(cfg["depth"] - 2):
        h = linear(f"hidden.{i}", x)
        if cfg["layernorm"]:
            mean = sum(h) / len(h)
            variance = sum((v - mean) ** 2 for v in h) / len(h)
            w, b = weights[f"norms.{i}.weight"].tolist(), weights[f"norms.{i}.bias"].tolist()
            h = [
                (v - mean) / math.sqrt(variance + 1e-5) * scale + offset
                for v, scale, offset in zip(h, w, b, strict=True)
            ]
        h = activation(h)
        x = [a + b for a, b in zip(x, h, strict=True)] if cfg["residual"] else h
    return linear("out", x)


@pytest.mark.parametrize(
    "damage",
    [
        "missing",
        "shape",
        "extra",
        "config",
        "nonfinite",
        "object",
        "duplicate_base",
        "wrong_encoder",
    ],
)
def test_failed_export_is_never_selectable(tmp_path: Path, damage: str) -> None:
    root = tmp_path / "models"
    encoder, head, original = fixture(root)
    if damage == "missing":
        del original["action_head"]
    elif damage == "shape":
        original["state_head"]["out.weight"] = torch.zeros(3, 3)
    elif damage == "extra":
        original["action_head"]["orphan.weight"] = torch.zeros(1)
    elif damage == "config":
        original["cfg"]["hidden_size"] = 5
    elif damage == "nonfinite":
        original["logit_scale"] = torch.tensor(float("nan"))
    elif damage == "object":
        original["cfg"]["bad"] = Path("checkpoint-python")
    elif damage == "duplicate_base":
        shutil.copyfile(encoder / "encoder.safetensors", encoder / "duplicate.safetensors")
    else:
        config = json.loads((encoder / "config.json").read_text())
        config["model_type"] = "llama"
        (encoder / "config.json").write_text(json.dumps(config))
    torch.save(original, head)
    destination = root / "clm"
    with pytest.raises((ValueError, pickle.UnpicklingError, RuntimeError, ModelError)):
        export(encoder, head, destination)
    assert not destination.exists()
    assert not list(tmp_path.glob(".clm-export-*"))
    assert not any(
        model.id.startswith("Contrastive-LM/") for model in ModelCatalogue(root).list_models()
    )


def test_import_consumer_cache_and_safe_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "models"
    encoder, head, _ = fixture(root)
    destination = root / "clm"
    export(encoder, head, destination)
    identity = "Contrastive-LM/CLM-v0.1-8B@" + "2" * 40
    forbidden = Mock(side_effect=AssertionError("Runtime checkpoint/network execution"))
    monkeypatch.setattr(torch, "load", forbidden)
    monkeypatch.setattr(requests.Session, "request", forbidden)
    settings = Settings(model_root=root, cache_dir=tmp_path / "cache")
    with TestClient(create_app(settings)) as client:
        sid = client.post("/sessions", json={"model_id": identity}).json()["id"]
        graph = client.get(f"/sessions/{sid}/architecture").json()["graph"]
        assert graph["scope"] == "language_model" and graph["coverage"] == "complete"
        assert not any(n["label"] == "Model-supplied definition" for n in graph["nodes"])
        tensor = next(p for p in graph["parameters"] if p["name"] == "clm.action_head.out.weight")
        response = client.get(f"/sessions/{sid}/tensors/{tensor['inspection']['tensor_id']}/data")
        assert response.status_code == 200
        expected = load_file(str(destination / "clm-heads.safetensors"))[tensor["name"]].float()
        assert (
            b"".join(data for kind, data in frames(response.content) if kind == 2)
            == expected.numpy().tobytes()
        )
        tokenized = client.post(
            f"/sessions/{sid}/tokenize",
            json={"text": "world hello world", "add_special_tokens": False},
        ).json()
        assert [t["id"] for t in tokenized["tokens"]] == [2, 1, 2]
        response = client.post(f"/sessions/{sid}/embeddings", json={"token_ids": [2, 1, 2]})
        assert response.status_code == 200, response.text
        rows = load_file(str(encoder / "encoder.safetensors"))["model.embed_tokens.weight"][
            [2, 1, 2]
        ].float()
        assert (
            b"".join(data for kind, data in frames(response.content) if kind == 2)
            == rows.numpy().tobytes()
        )
    monkeypatch.setattr(
        DescriptionRegistry, "analyze", Mock(side_effect=AssertionError("Warm graph rebuilt"))
    )
    with TestClient(create_app(settings)) as client:
        sid = client.post("/sessions", json={"model_id": identity}).json()["id"]
        assert (
            client.get(f"/sessions/{sid}/architecture").json()["graph"]["graph_id"]
            == graph["graph_id"]
        )
    forbidden.assert_not_called()


@pytest.mark.parametrize("asset", ["clm-heads.safetensors", "encoder.safetensors", "config.json"])
def test_content_changes_and_relocation(tmp_path: Path, asset: str) -> None:
    root = tmp_path / "models"
    encoder, head, _ = fixture(root)
    destination = root / "clm"
    export(encoder, head, destination)
    source = ModelCatalogue(root).inspect_directory(destination).pin()
    relocated = tmp_path / "relocated" / "clm"
    shutil.copytree(destination, relocated)
    assert (
        ModelCatalogue(relocated.parent).inspect_directory(relocated).pin().fingerprint
        == source.fingerprint
    )
    target = relocated / asset
    if asset.endswith(".json"):
        target.write_text(target.read_text() + " ")
    else:
        with target.open("r+b") as stream:
            stream.seek(-1, 2)
            last = stream.read(1)
            stream.seek(-1, 2)
            stream.write(bytes([last[0] ^ 1]))
    assert (
        ModelCatalogue(relocated.parent).inspect_directory(relocated).pin().fingerprint
        != source.fingerprint
    )


@pytest.mark.parametrize("damage", ["missing_marker", "missing_binding", "wrong_head_geometry"])
def test_full_clm_never_falls_back_to_bare_qwen(tmp_path: Path, damage: str) -> None:
    root = tmp_path / "models"
    encoder, head, _ = fixture(root)
    destination = root / "clm"
    export(encoder, head, destination)
    path = destination / "config.json"
    config = json.loads(path.read_text())
    if damage == "missing_marker":
        del config["clm_inspection"]
    elif damage == "wrong_head_geometry":
        config["clm_inspection"]["head_configuration"]["width"] += 1
    else:
        weights = load_file(str(destination / "clm-heads.safetensors"))
        del weights["clm.action_head.out.weight"]

        save_file(weights, str(destination / "clm-heads.safetensors"))
        index_path = destination / "model.safetensors.index.json"
        index = json.loads(index_path.read_text())
        del index["weight_map"]["clm.action_head.out.weight"]
        index_path.write_text(json.dumps(index))
    path.write_text(json.dumps(config))
    with TestClient(create_app(Settings(model_root=root, cache_dir=tmp_path / "cache"))) as client:
        identity = "Contrastive-LM/CLM-v0.1-8B@" + "2" * 40
        sid = client.post("/sessions", json={"model_id": identity}).json()["id"]
        response = client.get(f"/sessions/{sid}/architecture").json()
        assert response["status"] == "unavailable"
        assert "graph" not in response

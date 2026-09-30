"""CLM-owned gaps; generic Qwen/importer/transport matrices remain with their owners."""

import json
import math
import pickle
import shutil
from pathlib import Path
from typing import Any, cast
from unittest.mock import Mock

import pytest
import requests
import torch
from clm_fixtures import exporter, fixture
from fastapi.testclient import TestClient
from safetensors.torch import load_file
from test_tensor_data import frames

from llm_model_explorer.app import create_app
from llm_model_explorer.architecture_analysis.model_defined import ModelDefinedValidator
from llm_model_explorer.model_files import ModelError
from llm_model_explorer.models import ModelCatalogue
from llm_model_explorer.settings import Settings


def export(encoder: Path, head: Path, destination: Path) -> dict[str, Any]:
    return dict(
        exporter().export_package(
            encoder, head, destination, encoder_revision="1" * 40, head_revision="2" * 40
        )
    )


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
    graph = json.loads((destination / "architecture.json").read_text())
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
        assert graph["scope"] == "model_defined" and graph["coverage"] == "complete"
        assert any(n["label"] == "Model-supplied definition" for n in graph["nodes"])
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
        ModelDefinedValidator, "validate", Mock(side_effect=AssertionError("Warm graph rebuilt"))
    )
    with TestClient(create_app(settings)) as client:
        sid = client.post("/sessions", json={"model_id": identity}).json()["id"]
        assert (
            client.get(f"/sessions/{sid}/architecture").json()["graph"]["graph_id"]
            == graph["graph_id"]
        )
    forbidden.assert_not_called()


@pytest.mark.parametrize(
    "asset", ["clm-heads.safetensors", "encoder.safetensors", "config.json", "architecture.json"]
)
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


@pytest.mark.parametrize("damage", ["missing_sidecar", "missing_binding", "duplicate_edge"])
def test_full_clm_never_falls_back_to_bare_qwen(tmp_path: Path, damage: str) -> None:
    root = tmp_path / "models"
    encoder, head, _ = fixture(root)
    destination = root / "clm"
    export(encoder, head, destination)
    sidecar = destination / "architecture.json"
    if damage == "missing_sidecar":
        sidecar.unlink()
    else:
        graph = json.loads(sidecar.read_text())
        if damage == "missing_binding":
            graph["parameters"][-1]["name"] = "clm.missing.weight"
        else:
            graph["edges"].append(dict(graph["edges"][0]))
        sidecar.write_text(json.dumps(graph))
    with TestClient(create_app(Settings(model_root=root, cache_dir=tmp_path / "cache"))) as client:
        identity = "Contrastive-LM/CLM-v0.1-8B@" + "2" * 40
        sid = client.post("/sessions", json={"model_id": identity}).json()["id"]
        response = client.get(f"/sessions/{sid}/architecture").json()
        assert response["status"] == "unavailable"
        assert "graph" not in response

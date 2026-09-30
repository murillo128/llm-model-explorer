"""Kev-specific numerical, safety, hybrid semantics and consumer/lifecycle proof."""

import json
import pickle
import shutil
from pathlib import Path
from unittest.mock import Mock

import pytest
import requests
import torch
from fastapi.testclient import TestClient
from kev_fixtures import BASE_REV, KEV_REV, PREFIX, exporter, fixture
from safetensors.torch import load_file, save_file
from test_tensor_data import frames

from llm_model_explorer.app import create_app
from llm_model_explorer.architecture_analysis.model_defined import ModelDefinedValidator
from llm_model_explorer.model_files import ModelError
from llm_model_explorer.models import ModelCatalogue
from llm_model_explorer.settings import Settings


def export(base: Path, kev: Path, destination: Path) -> None:
    assert (
        exporter().export_package(
            base, kev, destination, base_revision=BASE_REV, kev_revision=KEV_REV
        )["status"]
        == "valid"
    )


def test_native_inventory_and_independent_algebra(tmp_path: Path) -> None:
    root = tmp_path / "models"
    base, kev, original = fixture(root)
    before = {p: p.read_bytes() for folder in (base, kev) for p in folder.iterdir() if p.is_file()}
    destination = root / "kev"
    export(base, kev, destination)
    assert all(p.read_bytes() == value for p, value in before.items())
    assert not (destination / "adapter_config.json").exists()
    native = load_file(str(base / "model.safetensors"))
    factors = load_file(str(kev / "adapter_model.safetensors"))
    converted = load_file(str(destination / "kev.safetensors"))
    assert set(converted) == {"kev.lora." + n for n in factors} | {
        "kev.head." + n for n in original["head"]
    }
    for name, tensor in factors.items():
        assert torch.equal(converted["kev.lora." + name], tensor)
        assert converted["kev.lora." + name].dtype == tensor.dtype
    for name, tensor in original["head"].items():
        assert torch.equal(converted["kev.head." + name], tensor)
    source = ModelCatalogue(root).inspect_directory(destination).pin()
    for descriptor in source.tensors():
        if descriptor.name in native:
            observed = torch.cat(list(source.iter_tensor(descriptor.id))).reshape(descriptor.shape)
            assert torch.equal(observed, native[descriptor.name].float())
    module = "base_model.model.layers.0.mlp.up_proj"
    x = torch.tensor([1.0, 2.0, 0.0, 0.0])
    a, b = (converted["kev.lora." + module + ".lora_" + f + ".weight"] for f in ("A", "B"))
    provenance = json.loads((destination / "kev-provenance.json").read_text())
    target = next(
        t
        for t in provenance["targets"]
        if t["base_weight"] == PREFIX + ".layers.0.mlp.up_proj.weight"
    )
    assert target["scale"] == 2
    actual = native[target["base_weight"]].float() @ x + target["scale"] * b @ (a @ x)
    assert actual.tolist() == [7.0, 18.0, 0.0, 0.0, 0.0, 0.0]
    # Independently specified example: q=[2,1,0,0], k=[[0,2,0,0],[2,1,0,0]].
    h_options = torch.tensor([[0.0, 1.0, 0.0, 0.0], [2.0, 0.0, 0.0, 0.0]])
    q = converted["kev.head.q.weight"] @ x + converted["kev.head.q.bias"]
    k = h_options @ converted["kev.head.k.weight"].T + converted["kev.head.k.bias"]
    assert q.tolist() == [2.0, 1.0, 0.0, 0.0]
    assert k.tolist() == [[0.0, 2.0, 0.0, 0.0], [2.0, 1.0, 0.0, 0.0]]
    metadata = provenance["kev"]["head_metadata"]
    assert metadata["head_dim"] == 4 and metadata["temperature"] == 2
    logits = (k @ q) / (metadata["head_dim"] ** 0.5) / metadata["temperature"]
    assert logits.tolist() == [0.5, 1.25]
    assert torch.softmax(logits, -1).tolist() == pytest.approx(
        [0.3208213008, 0.6791786992], abs=1e-7
    )
    assert not any("delta" in t.name or "effective" in t.name for t in source.tensors())


def test_graph_exact_factors_head_and_isolated_rows(tmp_path: Path) -> None:
    root = tmp_path / "models"
    base, kev, _ = fixture(root)
    export(base, kev, root / "kev")
    g = json.loads((root / "kev" / "architecture.json").read_text())
    nodes = {n["id"]: n for n in g["nodes"]}
    assert nodes["row.tokens"]["operation"] == "independent_causal_rows"
    assert "concat(state, question[row])" in nodes["row.tokens"]["formula"]
    assert "packed block-causal mask is not used" in nodes["backbone"]["description"]
    assert "no option permutation invariance" in nodes["row.tokens"]["description"]
    assert [i["variant"] for i in g["repetitions"][0]["instances"]] == [
        "linear_attention",
        "full_attention",
    ]
    assert nodes["pointer.q"]["parameter_ids"] == ["kev.head.q.weight", "kev.head.q.bias"]
    assert nodes["pointer.k"]["parameter_ids"] == ["kev.head.k.weight", "kev.head.k.bias"]
    assert nodes["pointer.temperature"]["attributes"] == [{"name": "temperature", "value": 2}]
    assert nodes["probabilities"]["ports"][0]["shape"] == [
        {"kind": "symbol", "name": "B"},
        {"kind": "symbol", "name": "O"},
    ]
    assert not any(n.get("operation") == "vocabulary_logits" for n in nodes.values())
    for target in json.loads((root / "kev" / "kev-provenance.json").read_text())["targets"]:
        group = nodes[target["base_weight"][:-7]]
        assert group["kind"] == "group"
        assert len(group["children"]) == 5
        assert nodes[group["id"] + ".A"]["parameter_ids"] == [target["A"]]
        assert nodes[group["id"] + ".B"]["parameter_ids"] == [target["B"]]
        assert nodes[group["id"] + ".base"]["parameter_ids"] == [target["base_weight"]]
    tuples = [
        (
            e["source"]["node_id"],
            e["source"]["port_id"],
            e["target"]["node_id"],
            e["target"]["port_id"],
        )
        for e in g["edges"]
    ]
    assert len(tuples) == len(set(tuples))
    states = {n["id"] for n in nodes.values() if n["kind"] == "state"}
    assert states and all(
        e["kind"] == "state"
        for e in g["edges"]
        if e["source"]["node_id"] in states or e["target"]["node_id"] in states
    )


@pytest.mark.parametrize(
    "damage",
    [
        "head_missing",
        "temperature_missing",
        "zero_temperature",
        "negative_temperature",
        "nan_temperature",
        "infinite_temperature",
        "wrong_head_shape",
        "missing_bias",
        "extra_head",
        "unsafe_object",
        "metadata_object",
        "base_identity",
        "base_revision",
        "rope_conflict",
        "head_revision",
        "adapter_revision",
        "unknown_target",
        "missing_A",
        "missing_pair",
        "duplicate_A",
        "orphan",
        "wrong_A_shape",
        "wrong_B_shape",
        "rank_mismatch",
        "head_rank_type",
        "dora",
        "rank_pattern",
        "unknown_option",
        "option_isolation",
        "special_embeddings",
        "question_lora",
        "missing_base_weight",
    ],
)
def test_invalid_export_never_publishes(tmp_path: Path, damage: str) -> None:
    root = tmp_path / "models"
    base, kev, payload = fixture(root)
    adapter = json.loads((kev / "adapter_config.json").read_text())
    config = json.loads((base / "config.json").read_text())
    factors = load_file(str(kev / "adapter_model.safetensors"))
    first = "base_model.model.layers.0.mlp.up_proj.lora_A.weight"
    if damage == "head_missing":
        del payload["head"]
    elif damage == "temperature_missing":
        del payload["temperature"]
    elif damage.endswith("temperature"):
        payload["temperature"] = {
            "zero_temperature": 0,
            "negative_temperature": -1,
            "nan_temperature": float("nan"),
            "infinite_temperature": float("inf"),
        }[damage]
    elif damage == "wrong_head_shape":
        payload["head"]["q.weight"] = torch.zeros(5, 4)
    elif damage == "missing_bias":
        del payload["head"]["k.bias"]
    elif damage == "extra_head":
        payload["head"]["orphan"] = torch.ones(1)
    elif damage == "unsafe_object":
        payload["bad"] = Path("untrusted-code")
    elif damage == "metadata_object":
        payload["args"] = {"tensor": torch.ones(1)}
    elif damage == "base_identity":
        config["name_or_path"] = "Qwen/different-model"
    elif damage == "base_revision":
        config["revision"] = "3" * 40
    elif damage == "rope_conflict":
        config["text_config"]["partial_rotary_factor"] = 0.5
    elif damage == "head_revision":
        payload["base_revision"] = "3" * 40
    elif damage == "adapter_revision":
        adapter["revision"] = "3" * 40
    elif damage == "unknown_target":
        adapter["target_modules"].append("unknown_proj")
    elif damage == "missing_A":
        del factors[first]
    elif damage == "missing_pair":
        del factors[first]
        del factors[first.replace("lora_A", "lora_B")]
    elif damage == "duplicate_A":
        factors[first.replace("lora_A.weight", "lora_A.default.weight")] = factors[first].clone()
    elif damage == "orphan":
        factors["base_model.model.unknown.lora_A.weight"] = torch.ones(2, 4)
    elif damage == "wrong_A_shape":
        factors[first] = torch.ones(2, 5)
    elif damage == "wrong_B_shape":
        factors[first.replace("lora_A", "lora_B")] = torch.ones(5, 2)
    elif damage == "rank_mismatch":
        payload["lora"] = 3
    elif damage == "head_rank_type":
        payload["lora"] = 2.0
    elif damage == "dora":
        adapter["use_dora"] = True
    elif damage == "rank_pattern":
        adapter["rank_pattern"] = {"up_proj": 3}
    elif damage == "unknown_option":
        adapter["unreviewed_extension"] = True
    elif damage == "option_isolation":
        payload["option_isolation"] = True
    elif damage == "special_embeddings":
        payload["special_embeddings"] = True
    elif damage == "question_lora":
        payload["lora_placement"] = "question"
    elif damage == "missing_base_weight":
        tensors = load_file(str(base / "model.safetensors"))
        del tensors[PREFIX + ".norm.weight"]
        save_file(tensors, str(base / "model.safetensors"))
    torch.save(payload, kev / "head.pt")
    save_file(factors, str(kev / "adapter_model.safetensors"))
    (kev / "adapter_config.json").write_text(json.dumps(adapter))
    (base / "config.json").write_text(json.dumps(config))
    with pytest.raises((ValueError, pickle.UnpicklingError, RuntimeError, ModelError)):
        export(base, kev, root / "kev")
    assert not (root / "kev").exists()
    assert not list(tmp_path.glob(".kev-export-*"))


def test_real_runtime_numeric_tokenizer_and_warm_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "models"
    base, kev, payload = fixture(root)
    export(base, kev, root / "kev")
    identity = "jaredpalmer/kev-0.8b-inspection@" + KEV_REV
    forbidden = Mock(side_effect=AssertionError("Runtime pickle or network"))
    monkeypatch.setattr(torch, "load", forbidden)
    monkeypatch.setattr(requests.Session, "request", forbidden)
    settings = Settings(model_root=root, cache_dir=tmp_path / "cache")
    with TestClient(create_app(settings)) as client:
        models = client.get("/models").json()["models"]
        assert len(models) == 2
        sid = client.post("/sessions", json={"model_id": identity}).json()["id"]
        graph = client.get(f"/sessions/{sid}/architecture").json()["graph"]
        assert graph["scope"] == "model_defined" and graph["coverage"] == "complete"
        original_factors = load_file(str(kev / "adapter_model.safetensors"))
        for name, expected in (
            ("kev.head.q.weight", payload["head"]["q.weight"]),
            (
                "kev.lora.base_model.model.layers.0.mlp.up_proj.lora_A.weight",
                original_factors["base_model.model.layers.0.mlp.up_proj.lora_A.weight"],
            ),
        ):
            p = next(p for p in graph["parameters"] if p["name"] == name)
            response = client.get(f"/sessions/{sid}/tensors/{p['inspection']['tensor_id']}/data")
            assert response.status_code == 200
            decoded = frames(response.content)
            assert decoded[-1][0] == 4
            assert b"".join(v for k, v in decoded if k == 2) == expected.float().numpy().tobytes()
        tokenized = client.post(
            f"/sessions/{sid}/tokenize",
            json={"text": "world hello world", "add_special_tokens": False},
        ).json()
        assert [t["id"] for t in tokenized["tokens"]] == [2, 1, 2]
        response = client.post(f"/sessions/{sid}/embeddings", json={"token_ids": [2, 1, 2]})
        original_rows = load_file(str(base / "model.safetensors"))[PREFIX + ".embed_tokens.weight"][
            [2, 1, 2]
        ].float()
        assert response.status_code == 200
        assert (
            b"".join(v for k, v in frames(response.content) if k == 2)
            == original_rows.numpy().tobytes()
        )
    monkeypatch.setattr(
        ModelDefinedValidator, "validate", Mock(side_effect=AssertionError("Warm rebuild"))
    )
    with TestClient(create_app(settings)) as client:
        sid = client.post("/sessions", json={"model_id": identity}).json()["id"]
        assert (
            client.get(f"/sessions/{sid}/architecture").json()["graph"]["graph_id"]
            == graph["graph_id"]
        )
    forbidden.assert_not_called()


@pytest.mark.parametrize(
    "asset,tensor_name",
    [
        ("base/model.safetensors", None),
        ("kev.safetensors", "kev.head.q.weight"),
        ("kev.safetensors", "kev.lora.base_model.model.layers.0.mlp.up_proj.lora_A.weight"),
        ("config.json", None),
        ("architecture.json", None),
    ],
)
def test_content_identity_and_pinned_mutation(
    tmp_path: Path, asset: str, tensor_name: str | None
) -> None:
    root = tmp_path / "models"
    base, kev, _ = fixture(root)
    export(base, kev, root / "kev")
    source = ModelCatalogue(root).inspect_directory(root / "kev").pin()
    relocated = tmp_path / "relocated" / "kev"
    shutil.copytree(root / "kev", relocated)
    pinned = ModelCatalogue(relocated.parent).inspect_directory(relocated).pin()
    assert pinned.fingerprint == source.fingerprint
    target = relocated / asset
    if asset.endswith(".json"):
        target.write_text(target.read_text() + " ")
    elif tensor_name:
        values = load_file(str(target))
        values[tensor_name][0, 0] += 1
        save_file(values, str(target))
    else:
        with target.open("r+b") as stream:
            stream.seek(-1, 2)
            last = stream.read(1)
            stream.seek(-1, 2)
            stream.write(bytes([last[0] ^ 1]))
    with pytest.raises(ModelError, match="changed"):
        pinned.check_unchanged()
    assert (
        ModelCatalogue(relocated.parent).inspect_directory(relocated).pin().fingerprint
        != source.fingerprint
    )


@pytest.mark.parametrize("damage", ["missing_sidecar", "wrong_binding", "duplicate_identity"])
def test_corrupt_package_never_falls_back_to_qwen(tmp_path: Path, damage: str) -> None:
    root = tmp_path / "models"
    base, kev, _ = fixture(root)
    export(base, kev, root / "kev")
    path = root / "kev" / "architecture.json"
    if damage == "missing_sidecar":
        path.unlink()
    else:
        g = json.loads(path.read_text())
        if damage == "wrong_binding":
            g["parameters"][-1]["name"] = "kev.missing.weight"
        else:
            g["edges"].append(dict(g["edges"][0]))
        path.write_text(json.dumps(g))
    with TestClient(create_app(Settings(model_root=root, cache_dir=tmp_path / "cache"))) as client:
        sid = client.post(
            "/sessions", json={"model_id": "jaredpalmer/kev-0.8b-inspection@" + KEV_REV}
        ).json()["id"]
        assert client.get(f"/sessions/{sid}/architecture").json()["status"] == "unavailable"

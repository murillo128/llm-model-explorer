"""Native package/composition proof over production CLI and independent TCP bytes."""

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest
from safetensors import safe_open

from acceptance.test_network import Frames, Service
from api.architecture_conformance import validate_architecture

REPO = Path(__file__).resolve().parents[1]
SOURCE_RULE = "Semantic source key in the reviewed packaged description"


def prepare(root, code):
    subprocess.run(
        [sys.executable, "-c", code, str(root)],
        cwd=REPO,
        env=os.environ
        | {"PYTHONPATH": os.pathsep.join([str(REPO / "backend/src"), str(REPO / "backend/tests")])},
        check=True,
        capture_output=True,
        text=True,
    )


def graph_for(service, predicate):
    catalogue = service.client.get("/models").json()
    model = next(model for model in catalogue["models"] if predicate(model["id"]))
    created = service.client.post("/sessions", json={"model_id": model["id"]})
    assert created.status_code == 201, created.text
    session = created.json()
    prefix = f"/sessions/{session['id']}"
    inventory = service.client.get(prefix + "/tensors").json()
    response = service.client.get(prefix + "/architecture")
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["status"] == "available"
    assert body["graph"]["scope"] == "language_model"
    assert body["graph"]["coverage"] == "complete"
    validate_architecture(
        body,
        {
            "session": session,
            "inventory": inventory,
            "tokenizer_available": model["tokenizer_available"],
        },
    )
    return prefix, body["graph"]


def raw_tensor(root, predicate):
    # The independent oracle reads physical Safetensors, never the application's
    # model source, graph storage resolver, numeric decoder or stream producer.
    for path in sorted(root.rglob("*.safetensors")):
        with safe_open(path, framework="pt", device="cpu") as file:
            for name in file.keys():
                if predicate(name):
                    return file.get_tensor(name).float()
    raise AssertionError("Expected native source tensor is absent")


def assert_native_stream(service, prefix, parameter, expected):
    assert parameter["binding"] == "native"
    assert parameter["inspection"]["status"] == "available"
    tensor_id = parameter["inspection"]["tensor_id"]
    with service.client.stream("GET", f"{prefix}/tensors/{tensor_id}/data") as response:
        frames = Frames(response)
        kind, metadata = frames.next()
        assert kind == 1
        assert json.loads(metadata)["shape"] == list(expected.shape)
        payload = bytearray()
        for kind, data in frames.rest():
            if kind == 2:
                payload.extend(data)
            else:
                assert kind == 4  # A failure/cancellation never counts as success.
        assert bytes(payload) == expected.numpy().tobytes()


@pytest.mark.parametrize("package", ["clm", "kev"])
def test_exported_packages_bind_head_and_factor_values_over_tcp(tmp_path, package):
    reference = os.environ.get(f"LMEX_{package.upper()}_REFERENCE_MODEL_ROOT")
    root = Path(reference) if reference else tmp_path / "models"
    if not reference:
        if package == "clm":
            prepare(
                root,
                "from pathlib import Path; import sys; from clm_fixtures import fixture, exporter; "
                "root=Path(sys.argv[1]); encoder,head,_=fixture(root); "
                'exporter().export_package(encoder,head,root/"clm",'
                'encoder_revision="1"*40,head_revision="2"*40)',
            )
        else:
            prepare(
                root,
                "from pathlib import Path; import sys; from kev_fixtures import fixture, exporter; "
                "root=Path(sys.argv[1]); base,checkpoint,_=fixture(root); "
                'exporter().export_package(base,checkpoint,root/"kev",'
                'base_revision="1"*40,kev_revision="2"*40)',
            )
    identity = (
        "Contrastive-LM/CLM-v0.1-8B@" if package == "clm" else "jaredpalmer/kev-0.8b-inspection@"
    )
    names = (
        ["clm.action_head.out.weight"]
        if package == "clm"
        else [
            "kev.lora.base_model.model.layers.0.linear_attn.in_proj_a.lora_A.weight",
            "kev.head.q.weight",
        ]
    )
    service = Service(tmp_path, model_root=root, startup_timeout=300)
    try:
        prefix, graph = graph_for(service, lambda model: model.startswith(identity))
        if package == "clm":
            assert not any(node["label"] == "Model-supplied definition" for node in graph["nodes"])
            assert any(
                node["label"] == "Action head" and node["kind"] == "group"
                for node in graph["nodes"]
            )
        for name in names:
            parameter = next(p for p in graph["parameters"] if p["name"] == name)
            node = next(n for n in graph["nodes"] if parameter["id"] in n["parameter_ids"])
            parent = next(n for n in graph["nodes"] if n["id"] == node["parent_id"])
            assert parent["kind"] == "group" and node["id"] in parent["children"]
            expected = raw_tensor(root, lambda key, name=name: key == name)
            assert_native_stream(service, prefix, parameter, expected)
        assert service.client.delete(prefix).status_code == 204
    finally:
        service.stop()
        service.client.close()


def test_adapted_projection_hierarchy_and_factor_values_over_tcp(tmp_path):
    root = tmp_path / "models"
    root.mkdir()
    reference = os.environ.get("LMEX_LORA_REFERENCE_MODEL_ROOT")
    if reference:
        supplied = Path(reference)
        for name in ["smollm2-135m-bf16", "smollm2-135m-smoltalk-lora"]:
            source = supplied / name
            required = "adapter_config.json" if name.endswith("lora") else "config.json"
            assert (source / required).is_file(), "Supplied local base/adapter pair is incomplete"
            destination = root / name
            destination.mkdir()
            for file in source.iterdir():
                if file.is_file():
                    (destination / file.name).hardlink_to(file)
    else:
        prepare(
            root,
            "import sys; from pathlib import Path; "
            "from test_lora_architecture import make_smollm2_lora; "
            "make_smollm2_lora(Path(sys.argv[1]))",
        )
    service = Service(tmp_path, model_root=root, startup_timeout=90)
    try:
        prefix, graph = graph_for(service, lambda model: "+peft-lora:" in model)
        nodes = {
            next(
                (p["source"] for p in node["provenance"] if p.get("rule") == SOURCE_RULE),
                node["id"],
            ): node
            for node in graph["nodes"]
        }
        target = "model.layers.0.self_attn.q_proj"
        group = nodes[target]
        assert group["kind"] == "group"
        assert group["children"] == [
            nodes[target + suffix]["id"]
            for suffix in [".base", ".lora_A", ".lora_B", ".lora_scale", ".lora_add"]
        ]
        for suffix, formula, scalar in [
            ("q_proj.lora_scale", "out = factor * x", ("factor", 2)),
            ("q_heads", "out = reshape(x, ...)", None),
            ("q_transpose", "out = transpose(x, ...)", None),
            ("softmax", "out = softmax(x, axis=axis)", ("axis", -1)),
        ]:
            node = nodes["model.layers.0.self_attn." + suffix]
            assert node["formula"] == formula
            if scalar:
                assert {a["name"]: a["value"] for a in node["attributes"]}[scalar[0]] == scalar[1]
        assert {p["id"] for p in nodes[target + ".lora_scale"]["ports"]} == {"x", "out"}
        for factor in ["A", "B"]:
            node = nodes[target + ".lora_" + factor]
            parameter = next(p for p in graph["parameters"] if p["id"] == node["parameter_ids"][0])
            source = re.compile(
                r"base_model\.model\.model\.layers\.0\.self_attn\.q_proj\.lora_"
                + factor
                + r"(?:\.default)?\.weight"
            )
            expected = raw_tensor(root, lambda name, source=source: source.fullmatch(name))
            assert_native_stream(service, prefix, parameter, expected)
        assert service.client.delete(prefix).status_code == 204
    finally:
        service.stop()
        service.client.close()

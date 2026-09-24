"""Optional end-to-end acceptance for the reviewed SmolLM2 LoRA reference pair."""

import os
import re
import struct
from pathlib import Path

import pytest
from llm_model_explorer.models import ModelCatalogue
from safetensors import safe_open

from acceptance.test_network import Frames, Service
from api.architecture_conformance import validate_architecture

MODEL_ROOT = os.environ.get("LMEX_LORA_REFERENCE_MODEL_ROOT")
pytestmark = pytest.mark.skipif(
    not MODEL_ROOT,
    reason="LMEX_LORA_REFERENCE_MODEL_ROOT not supplied; local SmolLM2 LoRA pair not tested",
)
TARGET = re.compile(
    r"__peft__\..*\.model\.layers\.(\d+)\.self_attn\.(q_proj|v_proj)\.lora_([AB])\.weight"
)
SEMANTIC_RULE = "Semantic source key in the reviewed packaged description"


def source_key(node: dict) -> str:
    return next(
        (item["source"] for item in node["provenance"] if item.get("rule") == SEMANTIC_RULE),
        node["label"],
    )


def parameter_shape(parameter: dict) -> tuple[int, ...]:
    return tuple(dimension["value"] for dimension in parameter["logical_shape"])


def attribute_signature(node: dict) -> tuple[tuple[str, object], ...]:
    return tuple((item["name"], item["value"]) for item in node["attributes"])


def read_logical_tensor(service: Service, session_id: str, tensor_id: str) -> bytes:
    with service.client.stream(
        "GET", f"/sessions/{session_id}/tensors/{tensor_id}/data"
    ) as response:
        frames = Frames(response)
        assert frames.next()[0] == 1
        payload = bytearray()
        while True:
            kind, data = frames.next()
            if kind == 4:
                return bytes(payload)
            assert kind == 2
            payload.extend(data)


def test_smollm2_lora_reference_pair_over_catalogue_sessions_and_architecture(tmp_path):
    root = Path(MODEL_ROOT).resolve(strict=True)
    summaries = ModelCatalogue(root).list_catalogue()
    assert summaries.diagnostics == ()
    composite_id = next(
        model.id
        for model in summaries.models
        if "+peft-lora:" in model.id and model.id.endswith(":smollm2-135m-smoltalk-lora")
    )
    base_id = composite_id.split("+peft-lora:", maxsplit=1)[0]
    assert {model.id for model in summaries.models} >= {base_id, composite_id}
    tokenizer = {model.id: model.tokenizer_available for model in summaries.models}
    adapter_dirs = [
        directory
        for directory in root.iterdir()
        if directory.is_dir()
        and (directory / "adapter_config.json").is_file()
        and (directory / "adapter_model.safetensors").is_file()
    ]
    assert len(adapter_dirs) == 1

    service = Service(tmp_path, model_root=root, startup_timeout=300)
    try:
        catalogue_response = service.client.get("/models")
        assert catalogue_response.status_code == 200
        assert str(root) not in catalogue_response.text
        catalogue = catalogue_response.json()
        assert catalogue["diagnostics"] == []
        assert {model["id"] for model in catalogue["models"]} >= {base_id, composite_id}

        sessions = {}
        graphs = {}
        inventories = {}
        for model_id in (base_id, composite_id):
            created = service.client.post("/sessions", json={"model_id": model_id})
            assert created.status_code == 201
            session = created.json()
            sessions[model_id] = session
            prefix = f"/sessions/{session['id']}"
            inventory_response = service.client.get(prefix + "/tensors")
            assert inventory_response.status_code == 200
            inventory = inventory_response.json()
            inventories[model_id] = inventory
            architecture_response = service.client.get(prefix + "/architecture")
            assert architecture_response.status_code == 200
            assert str(root) not in architecture_response.text
            body = architecture_response.json()
            validate_architecture(
                body,
                {
                    "session": session,
                    "inventory": inventory,
                    "tokenizer_available": tokenizer[model_id],
                },
            )
            assert body["status"] == "available"
            assert body["graph"]["coverage"] == "complete"
            graphs[model_id] = body["graph"]

        bare = graphs[base_id]
        graph = graphs[composite_id]
        base_nodes = {source_key(node): node for node in bare["nodes"]}
        nodes = {source_key(node): node for node in graph["nodes"]}
        base_by_id = {parameter["id"]: parameter for parameter in bare["parameters"]}
        composed_by_id = {parameter["id"]: parameter for parameter in graph["parameters"]}
        assert len(base_nodes) == len(bare["nodes"])
        assert len(nodes) == len(graph["nodes"])
        for key, node in base_nodes.items():
            composed = nodes[key]
            base_parameter_names = tuple(base_by_id[item]["name"] for item in node["parameter_ids"])
            composed_parameter_names = tuple(
                composed_by_id[item]["name"] for item in composed["parameter_ids"]
            )
            assert (
                composed["kind"],
                composed.get("operation"),
                composed["ports"],
                attribute_signature(composed),
                composed_parameter_names,
            ) == (
                node["kind"],
                node.get("operation"),
                node["ports"],
                attribute_signature(node),
                base_parameter_names,
            )

        base_parameters = {parameter["name"]: parameter for parameter in bare["parameters"]}
        parameters = {parameter["name"]: parameter for parameter in graph["parameters"]}
        parameters_by_id = {parameter["id"]: parameter for parameter in graph["parameters"]}
        assert set(base_parameters) <= set(parameters)
        for name, parameter in base_parameters.items():
            composed = parameters[name]
            assert (
                composed["binding"],
                composed["logical_shape"],
                composed["storage"],
                composed["inspection"],
            ) == (
                parameter["binding"],
                parameter["logical_shape"],
                parameter["storage"],
                parameter["inspection"],
            )

        factors = {}
        for parameter in graph["parameters"]:
            match = TARGET.fullmatch(parameter["name"])
            if match:
                layer, projection, factor = match.groups()
                factors.setdefault((int(layer), projection), {})[factor] = parameter
        expected_targets = {
            (layer, projection) for layer in range(30) for projection in ("q_proj", "v_proj")
        }
        assert set(factors) == expected_targets
        assert all(set(pair) == {"A", "B"} for pair in factors.values())
        inventory_by_id = {tensor["id"]: tensor for tensor in inventories[composite_id]["tensors"]}
        adapter_dir = adapter_dirs[0]
        checked_values = 0
        edges = {
            (
                edge["source"]["node_id"],
                edge["source"]["port_id"],
                edge["target"]["node_id"],
                edge["target"]["port_id"],
            )
            for edge in graph["edges"]
        }
        with safe_open(
            adapter_dir / "adapter_model.safetensors", framework="pt", device="cpu"
        ) as weights:
            for (layer, projection), pair in factors.items():
                target = f"model.layers.{layer}.self_attn.{projection}"
                base_node = nodes[target]
                assert base_node["operation"] == "linear"
                base_weight = next(
                    parameters_by_id[parameter_id]
                    for parameter_id in base_node["parameter_ids"]
                    if parameters_by_id[parameter_id]["name"] == target + ".weight"
                )
                assert base_weight["binding"] == "native"

                a_parameter, b_parameter = pair["A"], pair["B"]
                a_shape = (8, 576)
                b_shape = (576 if projection == "q_proj" else 192, 8)
                factor_nodes = {}
                for factor, parameter, shape in (
                    ("A", a_parameter, a_shape),
                    ("B", b_parameter, b_shape),
                ):
                    assert parameter_shape(parameter) == shape
                    assert parameter["binding"] == "native"
                    assert parameter["inspection"]["status"] == "available"
                    assert parameter["storage"][0]["role"] == "adapter_factor"
                    tensor_id = parameter["inspection"]["tensor_id"]
                    descriptor = inventory_by_id[tensor_id]
                    assert descriptor["name"] == parameter["name"]
                    assert descriptor["shape"] == list(shape)
                    factor_node = next(
                        node for node in graph["nodes"] if parameter["id"] in node["parameter_ids"]
                    )
                    factor_nodes[factor] = factor_node
                    assert factor_node["operation"] == "linear"
                    assert source_key(factor_node) == f"{target}.lora_{factor}"

                    if layer == 0:
                        payload = read_logical_tensor(
                            service,
                            sessions[composite_id]["id"],
                            tensor_id,
                        )
                        expected = weights.get_tensor(parameter["storage"][0]["name"])
                        expected = expected.float().contiguous()
                        assert len(payload) == expected.numel() * 4
                        actual_values = struct.unpack(f"<{expected.numel()}f", payload)
                        assert actual_values == tuple(expected.view(-1).tolist())
                        checked_values += len(actual_values)

                a_node, b_node = factor_nodes["A"], factor_nodes["B"]
                scale_node, add_node = (
                    nodes[target + ".lora_scale"],
                    nodes[target + ".lora_add"],
                )
                assert scale_node["operation"] == "scale"
                assert add_node["operation"] == "add"
                attributes = {item["name"]: item["value"] for item in scale_node["attributes"]}
                assert {key: attributes[key] for key in ("factor", "alpha", "rank")} == {
                    "factor": 2.0,
                    "alpha": 16.0,
                    "rank": 8.0,
                }
                assert (a_node["id"], "out", b_node["id"], "x") in edges
                assert (b_node["id"], "out", scale_node["id"], "x") in edges
                assert (base_node["id"], "out", add_node["id"], "base") in edges
                assert (scale_node["id"], "out", add_node["id"], "adapter") in edges
                base_input = next(
                    edge["source"]
                    for edge in graph["edges"]
                    if edge["target"] == {"node_id": base_node["id"], "port_id": "x"}
                )
                adapter_input = next(
                    edge["source"]
                    for edge in graph["edges"]
                    if edge["target"] == {"node_id": a_node["id"], "port_id": "x"}
                )
                assert base_input == adapter_input

        assert len(expected_targets) == 60
        assert checked_values > 0
    finally:
        service.stop()
        service.client.close()

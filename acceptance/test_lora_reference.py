"""Integrated native-LoRA and QLoRA-compatible SmolLM2 reference acceptance."""

import hashlib
import json
import os
import re
import struct
from pathlib import Path

import pytest
from safetensors import safe_open

from acceptance.architecture_reference import native_samples
from acceptance.quantized_reference import oracle_identity, scalar_sample
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
ADAPTER_NAME = "smollm2-135m-smoltalk-lora"


def source_key(node: dict) -> str:
    return next(
        (item["source"] for item in node["provenance"] if item.get("rule") == SEMANTIC_RULE),
        node["label"],
    )


def parameter_shape(parameter: dict) -> tuple[int, ...]:
    return tuple(dimension["value"] for dimension in parameter["logical_shape"])


def attribute_signature(node: dict) -> tuple[tuple[str, object], ...]:
    return tuple((item["name"], item["value"]) for item in node["attributes"])


def graph_topology(graph: dict) -> tuple[dict[str, tuple], set[tuple]]:
    nodes = {source_key(node): node for node in graph["nodes"]}
    nodes_by_id = {node["id"]: node for node in graph["nodes"]}
    parameters = {parameter["id"]: parameter for parameter in graph["parameters"]}
    signature = {
        key: (
            node["kind"],
            node.get("operation"),
            node["ports"],
            attribute_signature(node),
            tuple(parameters[item]["name"] for item in node["parameter_ids"]),
        )
        for key, node in nodes.items()
    }
    edges = {
        (
            source_key(nodes_by_id[edge["source"]["node_id"]]),
            edge["source"]["port_id"],
            source_key(nodes_by_id[edge["target"]["node_id"]]),
            edge["target"]["port_id"],
        )
        for edge in graph["edges"]
    }
    return signature, edges


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


def selected_samples(parameter: dict, descriptor: dict, directory: Path, *, quantized: bool):
    rows, columns = descriptor["shape"]
    coordinates = sorted(
        {
            (0, 0),
            (min(17, rows - 1), min(23, columns - 1)),
            (rows - 1, columns - 1),
        }
    )
    if quantized:
        expected = [
            scalar_sample(directory, parameter["name"], row, column) for row, column in coordinates
        ]
        oracle = oracle_identity("bnb-nf4-dq")
    else:
        expected = [
            {
                "offset": row * columns + column,
                "value": native_samples(directory, parameter["name"], row=row, column=column)[
                    "samples"
                ][0]["value"],
            }
            for row, column in coordinates
        ]
        oracle = {
            "package": "safetensors",
            "version": "0.7.0",
            "tolerance_absolute": 0.0,
        }
    return coordinates, expected, oracle


def test_smollm2_lora_and_qlora_references_over_catalogue_sessions_and_architecture(
    tmp_path,
):
    root = Path(MODEL_ROOT).resolve(strict=True)
    adapter_dirs = [
        directory
        for directory in root.iterdir()
        if directory.is_dir()
        and (directory / "adapter_config.json").is_file()
        and (directory / "adapter_model.safetensors").is_file()
    ]
    assert len(adapter_dirs) == 1
    adapter_dir = adapter_dirs[0]
    adapter_config = json.loads((adapter_dir / "adapter_config.json").read_text())
    assert adapter_config["peft_type"] == "LORA"
    assert adapter_config["r"] == 8 and adapter_config["lora_alpha"] == 16
    assert adapter_config["base_model_name_or_path"] == "HuggingFaceTB/SmolLM2-135M"
    download_manifest_path = os.environ.get("LMEX_HUB_DOWNLOAD_MANIFEST")
    assert download_manifest_path, "Record the pinned adapter download manifest"
    downloaded_adapter = json.loads(Path(download_manifest_path).read_text())["models"]["lora"]
    assert downloaded_adapter["revision"] == "fb39c7012c3b125f94d2bb1a093254994053f4ca"
    assert {path.name: path.stat().st_size for path in adapter_dir.iterdir() if path.is_file()} == {
        item["name"]: item["bytes"] for item in downloaded_adapter["files"]
    }

    adapter_file = adapter_dir / "adapter_model.safetensors"
    before_adapter = hashlib.sha256(adapter_file.read_bytes()).hexdigest()
    stat_before = (adapter_file.stat().st_size, adapter_file.stat().st_mtime_ns)
    native_vocab_size = json.loads((root / "SmolLM2-135M" / "config.json").read_text())[
        "vocab_size"
    ]
    quantized_vocab_size = json.loads((root / "SmolLM2-135M-bnb-4bit" / "config.json").read_text())[
        "vocab_size"
    ]
    assert quantized_vocab_size == native_vocab_size + 1
    service = Service(tmp_path, model_root=root, startup_timeout=1800, client_timeout=600)
    try:
        catalogue_response = service.readiness_response
        assert catalogue_response.status_code == 200
        assert str(root) not in catalogue_response.text
        catalogue = catalogue_response.json()
        assert catalogue["diagnostics"] == []
        model_ids = {model["id"] for model in catalogue["models"]}
        native_base = "SmolLM2-135M"
        bnb_base = "HuggingFaceTB/SmolLM2-135M@bnb-nf4-dq"
        assert {native_base, bnb_base} <= model_ids
        assert len(model_ids) == 7
        composites = {
            model_id.split("+peft-lora:", maxsplit=1)[0]: model_id
            for model_id in model_ids
            if "+peft-lora:" in model_id and model_id.endswith(f":{ADAPTER_NAME}")
        }
        assert composites == {
            native_base: f"{native_base}+peft-lora:{ADAPTER_NAME}",
            bnb_base: f"{bnb_base}+peft-lora:{ADAPTER_NAME}",
        }
        tokenizer = {model["id"]: model["tokenizer_available"] for model in catalogue["models"]}
        sessions = {}
        graphs = {}
        inventories = {}
        model_roles = [
            (native_base, "native base"),
            (composites[native_base], "native LoRA"),
            (bnb_base, "NF4 base"),
            (composites[bnb_base], "QLoRA-compatible"),
        ]
        for model_id, _role in model_roles:
            created = service.client.post("/sessions", json={"model_id": model_id})
            assert created.status_code == 201, created.text
            session = created.json()
            sessions[model_id] = session
            prefix = f"/sessions/{session['id']}"
            inventory_response = service.client.get(prefix + "/tensors")
            assert inventory_response.status_code == 200
            assert str(root) not in inventory_response.text
            inventory = inventory_response.json()
            inventories[model_id] = inventory
            assert inventory["coverage"] == "complete", inventory["diagnostics"]
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
            assert body["status"] == "available", body
            assert body["graph"]["coverage"] == "complete"
            graphs[model_id] = body["graph"]

        adapter_file_hashes = {}
        adapter_factor_values = {}
        with safe_open(adapter_file, framework="pt", device="cpu") as weights:
            adapter_file_hashes = {
                name: (
                    list(weights.get_slice(name).get_shape()),
                    weights.get_slice(name).get_dtype(),
                )
                for name in weights.keys()  # noqa: SIM118 - safetensors.safe_open is not a mapping
            }
            for name in adapter_file_hashes:
                marker = "model.layers.0."
                marker_index = name.rfind(marker)
                factor_key = name[marker_index:] if marker_index >= 0 else ""
                if factor_key.endswith((".lora_A.weight", ".lora_B.weight")):
                    tensor = weights.get_tensor(name).float().contiguous()
                    adapter_factor_values[factor_key] = tuple(tensor.view(-1).tolist())

        factor_names_by_base = {}
        topology_reports = {}
        for base_id, composite_id in composites.items():
            bare = graphs[base_id]
            graph = graphs[composite_id]
            bare_nodes, bare_edges = graph_topology(bare)
            composed_nodes, composed_edges = graph_topology(graph)
            assert set(bare_nodes) <= set(composed_nodes)
            assert all(composed_nodes[key] == bare_nodes[key] for key in bare_nodes)
            targeted_nodes = {
                f"model.layers.{layer}.self_attn.{projection}"
                for layer in range(30)
                for projection in ("q_proj", "v_proj")
            }
            expected_base_edges = {
                (
                    f"{source}.lora_add" if source in targeted_nodes and port == "out" else source,
                    port,
                    target,
                    target_port,
                )
                for source, port, target, target_port in bare_edges
            }
            assert expected_base_edges <= composed_edges

            bare_parameters = {parameter["name"]: parameter for parameter in bare["parameters"]}
            parameters = {parameter["name"]: parameter for parameter in graph["parameters"]}
            parameters_by_id = {parameter["id"]: parameter for parameter in graph["parameters"]}
            assert set(bare_parameters) <= set(parameters)
            for name, parameter in bare_parameters.items():
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
            factor_names_by_base[base_id] = sorted(
                parameter["name"] for pair in factors.values() for parameter in pair.values()
            )

            inventory_by_id = {
                tensor["id"]: tensor for tensor in inventories[composite_id]["tensors"]
            }
            checked_factor_values = 0
            edges = {
                (
                    edge["source"]["node_id"],
                    edge["source"]["port_id"],
                    edge["target"]["node_id"],
                    edge["target"]["port_id"],
                )
                for edge in graph["edges"]
            }
            nodes = {source_key(node): node for node in graph["nodes"]}
            base_binding = "native" if base_id == native_base else "quantized"
            base_tensor_names = []
            for (layer, projection), pair in factors.items():
                target = f"model.layers.{layer}.self_attn.{projection}"
                base_node = nodes[target]
                assert base_node["operation"] == "linear"
                base_weight = next(
                    parameters_by_id[parameter_id]
                    for parameter_id in base_node["parameter_ids"]
                    if parameters_by_id[parameter_id]["name"] == target + ".weight"
                )
                assert base_weight["binding"] == base_binding
                base_tensor_names.append(base_weight["name"])

                a_parameter, b_parameter = pair["A"], pair["B"]
                expected_shapes = {
                    "A": (8, 576),
                    "B": (576 if projection == "q_proj" else 192, 8),
                }
                factor_nodes = {}
                for factor, parameter in (("A", a_parameter), ("B", b_parameter)):
                    assert parameter_shape(parameter) == expected_shapes[factor]
                    assert parameter["binding"] == "native"
                    assert parameter["inspection"]["status"] == "available"
                    assert parameter["storage"][0]["role"] == "adapter_factor"
                    tensor_id = parameter["inspection"]["tensor_id"]
                    descriptor = inventory_by_id[tensor_id]
                    assert descriptor["name"] == parameter["name"]
                    assert descriptor["shape"] == list(expected_shapes[factor])
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
                        storage_name = parameter["storage"][0]["name"]
                        marker_index = storage_name.rfind("model.layers.0.")
                        assert marker_index >= 0
                        expected = adapter_factor_values[storage_name[marker_index:]]
                        assert len(payload) == len(expected) * 4
                        actual_values = struct.unpack(f"<{len(expected)}f", payload)
                        assert actual_values == expected
                        checked_factor_values += len(actual_values)

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
            assert checked_factor_values > 0
            topology_reports[base_id] = {
                "role": "QLoRA-compatible composition"
                if base_binding == "quantized"
                else "native LoRA composition",
                "base_binding": base_binding,
                "targets": len(factors),
                "graph_nodes": len(graph["nodes"]),
                "graph_edges": len(graph["edges"]),
                "base_graph_preserved": True,
                "adapter_factor_values_streamed": checked_factor_values,
            }

        # Stream the same targeted logical base weight from each representation;
        # compare native bytes to Safetensors and NF4 values to the independent oracle.
        base_sample_records = []
        for base_id, tensor_name, quantized in (
            (native_base, "model.layers.1.self_attn.q_proj.weight", False),
            (bnb_base, "model.layers.1.self_attn.q_proj.weight", True),
        ):
            composite_id = composites[base_id]
            graph = graphs[composite_id]
            parameter = next(item for item in graph["parameters"] if item["name"] == tensor_name)
            assert parameter["binding"] == ("quantized" if quantized else "native")
            descriptor = next(
                item
                for item in inventories[composite_id]["tensors"]
                if item["id"] == parameter["inspection"]["tensor_id"]
            )
            assert descriptor["name"] == tensor_name
            metadata_payload = read_logical_tensor(
                service, sessions[composite_id]["id"], descriptor["id"]
            )
            assert len(metadata_payload) == descriptor["numel"] * 4
            base_directory = root / ("SmolLM2-135M-bnb-4bit" if quantized else "SmolLM2-135M")
            coordinates, expected, oracle = selected_samples(
                parameter, descriptor, base_directory, quantized=quantized
            )
            for _coordinate, sample in zip(coordinates, expected, strict=True):
                actual = struct.unpack_from("<f", metadata_payload, sample["offset"] * 4)[0]
                assert abs(actual - sample["value"]) <= (1e-7 if quantized else 0.0)
            base_sample_records.append(
                {
                    "base_id": base_id,
                    "tensor": tensor_name,
                    "shape": descriptor["shape"],
                    "coordinates": [list(value) for value in coordinates],
                    "values": [sample["value"] for sample in expected],
                    "oracle": oracle,
                }
            )

        assert (
            adapter_file.stat().st_size,
            adapter_file.stat().st_mtime_ns,
        ) == stat_before
        assert hashlib.sha256(adapter_file.read_bytes()).hexdigest() == before_adapter

        native_topology, native_edges = graph_topology(graphs[composites[native_base]])
        qlora_topology, qlora_edges = graph_topology(graphs[composites[bnb_base]])
        native_lora_nodes = {key for key in native_topology if "lora_" in key}
        qlora_lora_nodes = {key for key in qlora_topology if "lora_" in key}
        assert native_lora_nodes == qlora_lora_nodes
        assert all(native_topology[key] == qlora_topology[key] for key in native_lora_nodes)
        native_lora_edges = {
            edge for edge in native_edges if "lora_" in edge[0] or "lora_" in edge[2]
        }
        qlora_lora_edges = {
            edge for edge in qlora_edges if "lora_" in edge[0] or "lora_" in edge[2]
        }
        assert native_lora_edges == qlora_lora_edges
        for session in sessions.values():
            assert service.client.delete(f"/sessions/{session['id']}").status_code == 204
    finally:
        service.stop()
        service.client.close()

    if evidence_path := os.environ.get("LMEX_EVIDENCE_DIR"):
        output = Path(evidence_path)
        output.mkdir(parents=True, exist_ok=True)
        (output / "issue-178-lora.json").write_text(
            json.dumps(
                {
                    "adapter_repository": "hfm8tr/smollm2-135m-smoltalk-lora",
                    "adapter_revision": "fb39c7012c3b125f94d2bb1a093254994053f4ca",
                    "selected_files": downloaded_adapter["files"],
                    "selected_file_count": downloaded_adapter["selected_file_count"],
                    "selected_bytes": downloaded_adapter["selected_bytes"],
                    "adapter_config": {
                        "peft_type": adapter_config["peft_type"],
                        "r": adapter_config["r"],
                        "lora_alpha": adapter_config["lora_alpha"],
                        "target_modules": sorted(adapter_config["target_modules"]),
                        "files": sorted(adapter_file_hashes),
                        "weight_file_bytes": stat_before[0],
                        "source_sha256": before_adapter,
                    },
                    "compositions": topology_reports,
                    "sampled_base_weights": base_sample_records,
                    "native_and_quantized_base_vocab_sizes": {
                        "native": native_vocab_size,
                        "quantized": quantized_vocab_size,
                    },
                    "historical_training_method_claim": None,
                },
                indent=2,
            )
        )

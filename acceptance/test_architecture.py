"""Production CLI/TCP acceptance of all packaged static descriptions."""

import hashlib
import json
import os
import platform
import re
import shutil
import struct
import subprocess
import sys
import time
from pathlib import Path

import pytest

from acceptance.architecture_fixtures import generate, packed_value, value
from acceptance.test_network import Frames, Service
from api.architecture_conformance import validate_architecture


def test_grouping_preserves_the_accepted_operation_level_contract(tmp_path):
    """Run #119's independent Node oracle on actual packaged producer exports."""
    repo = Path(__file__).resolve().parents[1]
    environment = os.environ | {
        "PYTHONPATH": os.pathsep.join([str(repo / "backend/src"), str(repo / "backend/tests")])
    }
    subprocess.run(
        [
            sys.executable,
            str(repo / "backend/tests/architecture_grouping_cases.py"),
            str(tmp_path),
        ],
        cwd=repo,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    checked = subprocess.run(
        [
            "node",
            str(repo / "ui/scripts/check-architecture-semantics.mjs"),
            str(tmp_path),
        ],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "PASS: 7 reviewed cases" in checked.stdout


def inspect_graph(service, model_id, *, validate_inventory=True):
    advertised = [
        model["id"]
        for model in service.client.get("/models").json()["models"]
        if model["id"] == model_id or model["id"].startswith(f"{model_id}@")
    ]
    assert len(advertised) == 1, (model_id, advertised)
    model_id = advertised[0]
    response = service.client.post("/sessions", json={"model_id": model_id})
    assert response.status_code == 201, response.text
    session = response.json()
    prefix = f"/sessions/{session['id']}"
    inventory = service.client.get(prefix + "/tensors").json()
    response = service.client.get(prefix + "/architecture")
    assert response.status_code == 200, response.text
    assert response.headers["cache-control"] == "no-store"
    assert "x-operation-id" not in response.headers
    assert str(service.model_root) not in response.text
    body = response.json()
    if validate_inventory:
        validate_architecture(
            body,
            {
                "session": session,
                "inventory": inventory,
                "tokenizer_available": False,
            },
        )
    else:
        # This synthetic Kimi shell exercises the complete graph topology, but
        # its full independent metadata oracle is not a physical tensor index.
        assert body["model_id"] == session["model_id"]
        validate_architecture(body, validate_storage=False)
    return prefix, inventory, body


def file_state(root):
    return {
        str(p.relative_to(root)): (
            p.stat().st_size,
            p.stat().st_mtime_ns,
            hashlib.sha256(p.read_bytes()).hexdigest(),
        )
        for p in root.rglob("*")
        if p.is_file()
    }


def test_all_descriptions_over_tcp_readonly_cold_warm_and_logical_values(tmp_path):
    root = tmp_path / "models"
    generate(root)
    before = file_state(root)
    for path in root.rglob("*"):
        path.chmod(0o555 if path.is_dir() else 0o444)
    root.chmod(0o555)
    service = Service(tmp_path, model_root=root)
    graphs = {}
    try:
        for family, counts in {
            "smollm2": [2],
            "qwen3": [2],
            "qwen35": [2],
            "vjepa2": [2, 1],
        }.items():
            prefix, inventory, body = inspect_graph(service, family)
            assert body["status"] == "available", body
            graph = body["graph"]
            graphs[family] = graph
            assert graph["coverage"] == "complete"
            assert [len(r["instances"]) for r in graph["repetitions"]] == counts
            assert inventory["coverage"] == "complete"
            assert inventory["diagnostics"] == []
            # Exercise a later concrete instance, both native ranks, and exact bytes.
            for rank in (1, 2):
                parameter = next(
                    p
                    for p in graph["parameters"]
                    if p["inspection"]["status"] == "available"
                    and p["binding"] == "native"
                    and len(p["logical_shape"]) == rank
                    and (rank == 2 or ".1." in p["name"])
                )
                tid = parameter["inspection"]["tensor_id"]
                descriptor = next(t for t in inventory["tensors"] if t["id"] == tid)
                assert descriptor["name"] == parameter["name"]
                with service.client.stream("GET", f"{prefix}/tensors/{tid}/data") as response:
                    frames = Frames(response)
                    assert frames.next()[0] == 1
                    payload = bytearray()
                    while True:
                        kind, data = frames.next()
                        if kind == 4:
                            break
                        assert kind == 2
                        payload.extend(data)
                    assert bytes(payload) == b"".join(
                        struct.pack("<f", value(i)) for i in range(descriptor["numel"])
                    )
            if family in {"qwen3", "qwen35"}:
                parameter = next(p for p in graph["parameters"] if p["binding"] == "quantized")
                assert parameter["inspection"]["status"] == "available"
                tid = parameter["inspection"]["tensor_id"]
                descriptor = next(t for t in inventory["tensors"] if t["id"] == tid)
                rows, columns = descriptor["shape"]
                expected = b"".join(
                    struct.pack("<f", packed_value(family, row, column, columns))
                    for row in range(rows)
                    for column in range(columns)
                )
                for _ in range(2):
                    with service.client.stream("GET", f"{prefix}/tensors/{tid}/data") as response:
                        assert response.status_code == 200
                        reader = Frames(response)
                        kind, metadata = reader.next()
                        assert kind == 1 and json.loads(metadata)["shape"] == [
                            rows,
                            columns,
                        ]
                        payload = bytearray()
                        while True:
                            kind, data = reader.next()
                            if kind == 4:
                                break
                            assert kind == 2
                            payload.extend(data)
                        assert payload == expected
            if family == "vjepa2":
                assert graph["scope"] == "visual_encoder_predictor"
                assert not any(
                    r["kind"] == "tokenizer" for n in graph["nodes"] for r in n["references"]
                )
                assert any(
                    p["inspection"].get("reason") == "unsupported_rank" for p in graph["parameters"]
                )
            assert service.client.delete(prefix).status_code == 204
        cached = file_state(tmp_path / "cache")
        service.stop()
        service.start()
        for family, graph in graphs.items():
            assert inspect_graph(service, family)[2]["graph"] == graph
        assert file_state(tmp_path / "cache") == cached
        assert file_state(root) == before
        log = (tmp_path / "service.log").read_text()
        assert log.count("mode=cache outcome=available") == 4
        assert "hashing_seconds=" in log
    finally:
        service.stop()
        service.client.close()
        root.chmod(0o755)
        for path in root.rglob("*"):
            path.chmod(0o755 if path.is_dir() else 0o644)
    if directory := os.environ.get("LMEX_EVIDENCE_DIR"):
        Path(directory).mkdir(parents=True, exist_ok=True)
        (Path(directory) / "architecture-fixtures.json").write_text(
            json.dumps(
                {
                    "kind": "synthetic local fixtures, not reference acceptance",
                    "graphs": {
                        key: {
                            "nodes": len(g["nodes"]),
                            "edges": len(g["edges"]),
                            "coverage": g["coverage"],
                        }
                        for key, g in graphs.items()
                    },
                    "startup_log": log,
                },
                indent=2,
            )
        )


def test_kimi_linear_complete_expert_graph_over_production_tcp(tmp_path):
    from acceptance.kimi_linear_fixture import generate as generate_kimi
    from api.architecture_conformance import expand_compact_graph

    generate_kimi(tmp_path / "models")
    service = Service(tmp_path, startup_timeout=120, kimi_architecture_fixture=True)
    try:
        prefix, inventory, body = inspect_graph(service, "kimi_linear", validate_inventory=False)
        assert body["status"] == "available"
        graph = body["graph"]
        expanded = expand_compact_graph(graph)
        assert graph["coverage"] == "complete"
        assert len(graph["repetitions"]) == 27

        expected = {
            f"model.layers.{layer}.block_sparse_moe.experts.{expert}.w{projection}.weight"
            for layer in range(1, 27)
            for expert in range(256)
            for projection in (1, 2, 3)
        }
        observed = {
            parameter["name"]
            for parameter in expanded["parameters"]
            if ".block_sparse_moe.experts." in parameter["name"]
        }
        assert observed == expected
        assert len({parameter["id"] for parameter in expanded["parameters"]}) == len(
            expanded["parameters"]
        )
        assert len({node["id"] for node in expanded["nodes"]}) == len(expanded["nodes"])
        assert len(expanded["nodes"]) > 7000
        assert len(expanded["edges"]) > 2000
        assert inventory["coverage"] == "complete"

        response = service.client.get(prefix + "/architecture")
        assert response.status_code == 200
        assert len(response.content) < 33_554_432
        assert response.json() == body
    finally:
        service.stop()
        service.client.close()


def test_issue_178_actual_quantized_architectures_cold_warm_and_numeric_samples(
    tmp_path,
):
    """Exercise the six pinned models through one real catalogue and cold/warm app."""
    if os.environ.get("LMEX_REQUIRE_ISSUE_178_REFERENCES") != "1":
        pytest.skip("Issue #178 full-reference acceptance is an explicit local gate")
    from acceptance.architecture_reference import (
        logical_value_stream,
        native_samples,
        selections,
        validate_pinned_architecture,
    )
    from acceptance.quantized_reference import oracle_identity, scalar_sample
    from api.architecture_conformance import expand_compact_graph

    required = {"deepseek_v2", "glm4_moe_lite", "kimi_linear"}
    quantized_families = required | {"smollm2_bnb"}
    reference_families = quantized_families | {"smollm2"}
    entries = selections()
    missing = reference_families - entries.keys()
    if missing:
        pytest.fail(f"Issue #178 requires complete local references: {sorted(missing)}")
    root = Path(entries["deepseek_v2"]["directory"]).resolve(strict=True).parent
    assert all(
        Path(entries[family]["directory"]).resolve(strict=True).parent == root
        for family in required
    )
    reports = {}
    download_manifest_path = os.environ.get("LMEX_HUB_DOWNLOAD_MANIFEST")
    assert download_manifest_path, (
        "Record the exact downloaded file manifest for all six repositories"
    )
    downloaded = json.loads(Path(download_manifest_path).read_text())
    assert downloaded["free_bytes_at_preflight"] >= 90_000_000_000
    for family in reference_families:
        selected = downloaded["models"][family]
        assert selected["repository"] == entries[family]["repository"]
        assert selected["revision"] == entries[family]["revision"]
        directory = Path(entries[family]["directory"])
        actual_files = {
            path.name: path.stat().st_size for path in directory.iterdir() if path.is_file()
        }
        selected_files = {item["name"]: item["bytes"] for item in selected["files"]}
        assert actual_files == selected_files
        config = json.loads((directory / "config.json").read_text())
        reports[family] = {
            "repository": selected["repository"],
            "revision": selected["revision"],
            "revision_source": "operator manifest",
            "selected_files": selected["files"],
            "selected_file_count": selected["selected_file_count"],
            "selected_bytes": selected["selected_bytes"],
            "total_bytes": selected["selected_bytes"],
            "model_type": config.get("model_type"),
            "architectures": config.get("architectures"),
            "quantization": config.get("quantization_config"),
            "tokenizer_available": family != "kimi_linear",
        }
    adapter = downloaded["models"]["lora"]
    assert adapter["repository"] == "hfm8tr/smollm2-135m-smoltalk-lora"
    adapter_directory = root / adapter["directory"]
    adapter_actual_files = {
        path.name: path.stat().st_size for path in adapter_directory.iterdir() if path.is_file()
    }
    adapter_selected_files = {item["name"]: item["bytes"] for item in adapter["files"]}
    assert adapter_actual_files == adapter_selected_files
    adapter_config = json.loads((adapter_directory / "adapter_config.json").read_text())
    reports["lora"] = {
        "repository": adapter["repository"],
        "revision": adapter["revision"],
        "selected_files": adapter["files"],
        "selected_file_count": adapter["selected_file_count"],
        "selected_bytes": adapter["selected_bytes"],
        "adapter_metadata": {
            "base_model_name_or_path": adapter_config["base_model_name_or_path"],
            "peft_type": adapter_config["peft_type"],
            "rank": adapter_config["r"],
            "alpha": adapter_config["lora_alpha"],
            "target_modules": sorted(adapter_config["target_modules"]),
        },
    }
    for family in reference_families:
        report = reports[family]
        assert report["revision"] and report["selected_bytes"] > 0
    assert reports["lora"]["revision"] and reports["lora"]["selected_bytes"] > 0

    evidence = Path(os.environ.get("LMEX_EVIDENCE_DIR", tmp_path)) / "issue-178-architectures.json"
    evidence.parent.mkdir(parents=True, exist_ok=True)
    cold_started = time.perf_counter()
    service_root = tmp_path / "reference-service"
    service_root.mkdir()
    service = Service(service_root, model_root=root, startup_timeout=1800, client_timeout=600)
    cold_startup_seconds = time.perf_counter() - cold_started
    graphs = {}
    measurements = []
    numeric = []
    tokenizer_result = None
    try:
        catalogue_response = service.readiness_response
        assert catalogue_response.status_code == 200
        assert str(root) not in catalogue_response.text
        catalogue = catalogue_response.json()
        assert catalogue["diagnostics"] == []
        model_ids = {model["id"] for model in catalogue["models"]}
        assert len(model_ids) == 7, sorted(model_ids)
        assert any(model.endswith("@bnb-nf4-dq") for model in model_ids)
        assert any(model.endswith("@compressed-tensors-w4a16-int4") for model in model_ids)
        assert sum("+peft-lora:" in model for model in model_ids) == 2
        cached_fingerprints = {}
        for manifest_path in (service.root / "cache").glob("*/manifest.json"):
            manifest = json.loads(manifest_path.read_text())
            if manifest["spec"].get("kind") != "architecture_graph":
                continue
            graph_path = manifest_path.parent / manifest["payload"]
            cached_graph = json.loads(graph_path.read_bytes())
            cached_fingerprints[cached_graph["graph_id"]] = manifest["spec"]["model_fingerprint"]

        for mode in ("cold", "warm"):
            if mode == "warm":
                cold_cache = file_state(service.root / "cache")
                service.stop()
                started = time.perf_counter()
                service.start()
                startup_seconds = time.perf_counter() - started
                assert file_state(service.root / "cache") == cold_cache
            else:
                startup_seconds = cold_startup_seconds
            service_measure = {"mode": mode, "startup_seconds": startup_seconds}
            status = Path(f"/proc/{service.process.pid}/status").read_text()
            service_measure["backend_peak_rss_kib"] = int(re.search(r"VmHWM:\s+(\d+)", status)[1])
            for family in sorted(quantized_families):
                selection = entries[family]
                # The public identity is metadata-derived and intentionally need
                # not match the accepted upstream repository owner.
                directory_name = Path(selection["directory"]).name.casefold()
                model_id = (
                    "HuggingFaceTB/SmolLM2-135M@bnb-nf4-dq"
                    if family == "smollm2_bnb"
                    else next(model for model in model_ids if directory_name in model.casefold())
                )
                created = service.client.post("/sessions", json={"model_id": model_id})
                assert created.status_code == 201, created.text
                session = created.json()
                prefix = f"/sessions/{session['id']}"
                architecture = service.client.get(prefix + "/architecture")
                assert architecture.status_code == 200, architecture.text
                assert str(root) not in architecture.text
                body = architecture.json()
                assert body["status"] == "available", {
                    "family": family,
                    "reason": body.get("reason"),
                    "diagnostics": body.get("diagnostics"),
                }
                inventory_response = service.client.get(prefix + "/tensors")
                assert inventory_response.status_code == 200
                assert str(root) not in inventory_response.text
                tensor_inventory = inventory_response.json()
                reports[family]["tensor_inventory_coverage"] = tensor_inventory["coverage"]
                diagnostic_codes = {}
                for diagnostic in tensor_inventory["diagnostics"]:
                    code = diagnostic["code"]
                    diagnostic_codes[code] = diagnostic_codes.get(code, 0) + 1
                reports[family]["tensor_inventory_diagnostic_codes"] = diagnostic_codes
                validate_architecture(
                    body,
                    {
                        "session": session,
                        "inventory": tensor_inventory,
                        "tokenizer_available": reports[family]["tokenizer_available"],
                    },
                )
                graph = body["graph"]
                expanded_graph = expand_compact_graph(graph)
                reports[family]["explorer_content_fingerprint"] = cached_fingerprints[
                    graph["graph_id"]
                ]
                if family in required:
                    validate_pinned_architecture(family, expanded_graph)
                else:
                    assert graph["coverage"] == "complete"
                if mode == "cold":
                    graphs[family] = graph
                    descriptors_by_name = {
                        tensor["name"]: tensor for tensor in tensor_inventory["tensors"]
                    }
                    expected_encoding = (
                        "bnb-nf4-dq"
                        if family in {"deepseek_v2", "smollm2_bnb"}
                        else "compressed-tensors-w4a16-int4"
                    )
                    quantized = [
                        parameter
                        for parameter in expanded_graph["parameters"]
                        if parameter["binding"] == "quantized"
                        and len(parameter["logical_shape"]) == 2
                        and parameter["name"].endswith(".weight")
                        and parameter["name"] in descriptors_by_name
                        and descriptors_by_name[parameter["name"]]["storage_format"]
                        == expected_encoding
                    ]
                    experts = [
                        parameter
                        for parameter in quantized
                        if ".experts." in parameter["name"]
                        and parameter["name"].endswith(("gate_proj.weight", "w1.weight"))
                    ]
                    attention = [
                        parameter for parameter in quantized if ".self_attn." in parameter["name"]
                    ]
                    if experts and attention:
                        selected = [attention[0], experts[0]]
                    elif experts:
                        assert len(experts) >= 2
                        selected = experts[:2]
                    else:
                        assert len(attention) >= 2
                        selected = attention[:2]
                    for parameter in selected:
                        descriptor = descriptors_by_name[parameter["name"]]
                        assert descriptor["name"] == parameter["name"]
                        shape = descriptor["shape"]
                        assert shape == [
                            dimension["value"] for dimension in parameter["logical_shape"]
                        ]
                        rows, columns = shape
                        coordinates = sorted(
                            {
                                (0, 0),
                                (min(17, rows - 1), min(23, columns - 1)),
                                (rows - 1, columns - 1),
                            }
                        )
                        oracle = [
                            scalar_sample(
                                Path(selection["directory"]),
                                parameter["name"],
                                row,
                                column,
                            )
                            for row, column in coordinates
                        ]
                        metadata, payload = logical_value_stream(
                            service,
                            session["id"],
                            descriptor["id"],
                            numel=descriptor["numel"],
                        )
                        assert metadata["shape"] == shape
                        assert len(payload) == descriptor["numel"] * 4
                        for (row, column), expected in zip(coordinates, oracle, strict=True):
                            offset = expected["offset"]
                            actual = struct.unpack_from("<f", payload, offset * 4)[0]
                            assert abs(actual - expected["value"]) <= 1e-7, (
                                family,
                                parameter["name"],
                                row,
                                column,
                                actual,
                                expected,
                            )
                        numeric.append(
                            {
                                "family": family,
                                "tensor": parameter["name"],
                                "shape": shape,
                                "coordinates": [list(value) for value in coordinates],
                                "values": [value["value"] for value in oracle],
                                "oracle": oracle_identity(expected_encoding),
                            }
                        )
                    if family == "deepseek_v2":
                        native = next(
                            tensor
                            for tensor in tensor_inventory["tensors"]
                            if tensor["storage_format"] == "safetensors"
                            and tensor["rank"] == 1
                            and tensor["name"].startswith("model.layers.0.")
                        )
                        metadata, payload = logical_value_stream(
                            service,
                            session["id"],
                            native["id"],
                            numel=native["numel"],
                        )
                        assert metadata["shape"] == native["shape"]
                        assert len(payload) == native["numel"] * 4
                        independent = native_samples(
                            Path(entries[family]["directory"]), native["name"]
                        )
                        for sample in independent["samples"]:
                            actual = struct.unpack_from("<f", payload, sample["offset"] * 4)[0]
                            assert actual == sample["value"]
                        numeric.append(
                            {
                                "family": family,
                                "tensor": native["name"],
                                "samples": independent["samples"],
                                "native_stream_bytes": len(payload),
                            }
                        )
                else:
                    assert graph == graphs[family], f"Warm graph changed for {family}"
                assert service.client.delete(prefix).status_code == 204
            service_measure["architecture_log"] = [
                line
                for line in (service.root / "service.log").read_text().splitlines()
                if "Architecture model=" in line
            ]
            measurements.append(service_measure)

        # Bare SmolLM2 exercises the live tokenizer in the same production app.
        native_id = next(model for model in model_ids if model == "SmolLM2-135M")
        session = service.client.post("/sessions", json={"model_id": native_id}).json()
        tokenizer_response = service.client.post(
            f"/sessions/{session['id']}/tokenize",
            json={"text": "real SmolLM2 tokenizer ✓"},
        )
        assert tokenizer_response.status_code == 200, tokenizer_response.text
        tokenizer_result = tokenizer_response.json()
        assert tokenizer_result["text"] == "real SmolLM2 tokenizer ✓"
        assert tokenizer_result["tokens"]
        assert str(root) not in tokenizer_response.text
        assert service.client.delete(f"/sessions/{session['id']}").status_code == 204
    finally:
        service.stop()
        service.client.close()

    log = (service.root / "service.log").read_text()
    assert "hashing_seconds=" in log
    assert "analysis_seconds=" in log
    assert "cache_read_seconds=" in log
    evidence.write_text(
        json.dumps(
            {
                "issue": 178,
                "environment": {
                    "platform": platform.platform(),
                    "python": sys.version.split()[0],
                    "machine": platform.machine(),
                    "cpu_count": os.cpu_count(),
                    "device": "cpu",
                },
                "references": reports,
                "download_preflight": {
                    "selected_bytes_total": downloaded["selected_bytes_total"],
                    "free_bytes_before_large_downloads": downloaded["free_bytes_at_preflight"],
                },
                "catalogue_entries": 7,
                "architecture": {
                    family: {
                        "coverage": graph["coverage"],
                        "nodes": len(expand_compact_graph(graph)["nodes"]),
                        "edges": len(expand_compact_graph(graph)["edges"]),
                        "wire_nodes": len(graph["nodes"]),
                        "wire_edges": len(graph["edges"]),
                        "compact_families": len(graph.get("compact_components", [])),
                        "parameters": len(graph["parameters"]),
                        "graph_id": graph["graph_id"],
                    }
                    for family, graph in graphs.items()
                },
                "cold_warm": measurements,
                "numeric_samples": numeric,
                "tokenizer_tokens": len(tokenizer_result["tokens"]),
                "backend_log_measurements": [
                    line for line in log.splitlines() if "Architecture model=" in line
                ],
            },
            indent=2,
        )
    )


def test_tcp_unavailable_restart_cache_loss_and_partial_predictor(tmp_path):
    root = tmp_path / "models"
    generate(root)
    # Remove the predictor storage, preserving a valid admitted encoder checkpoint.
    from acceptance.architecture_fixtures import FIXTURES, write_checkpoint

    fixture = json.loads((FIXTURES / "vjepa2-tiny.json").read_text())
    shutil.rmtree(root / "vjepa2")
    write_checkpoint(
        root / "vjepa2",
        fixture["configuration"],
        {k: v for k, v in fixture["storage"].items() if not k.startswith("predictor.")},
    )
    config = root / "smollm2/config.json"
    data = json.loads(config.read_text())
    data["hidden_act"] = "unsupported"
    config.write_text(json.dumps(data))
    service = Service(tmp_path, model_root=root)
    try:
        prefix, inventory, body = inspect_graph(service, "smollm2")
        assert body["reason"] == "unsupported_architecture"
        assert inventory["tensors"]  # Architecture failure preserves numeric capability.
        assert (
            service.client.get(f"{prefix}/tensors/{inventory['tensors'][0]['id']}/data").status_code
            == 200
        )
        assert inspect_graph(service, "vjepa2")[2]["graph"]["coverage"] == "partial"
        prefix, _, body = inspect_graph(service, "qwen3")
        assert body["graph"]["coverage"] == "complete"
        shutil.rmtree(tmp_path / "cache")
        response = service.client.get(prefix + "/architecture").json()
        assert response["reason"] == "cache_unavailable" and response["requires_restart"]
        config = root / "qwen3/config.json"
        config.write_text(config.read_text() + " ")
        assert service.client.get(prefix + "/architecture").status_code == 409
        assert inspect_graph(service, "qwen3")[2]["reason"] == "restart_required"
    finally:
        service.stop()
        service.client.close()


@pytest.mark.parametrize("family", ["qwen3", "qwen35", "vjepa2", "smollm2"])
def test_complete_local_architecture_reference(tmp_path, family):
    from acceptance.architecture_reference import REFERENCES, measure, selections

    entries = selections()
    if family not in entries:
        if os.environ.get("LMEX_REQUIRE_ARCHITECTURE_REFERENCES") == "1":
            pytest.fail(f"Required complete local {REFERENCES[family]} is missing")
        pytest.skip(f"Missing complete local {REFERENCES[family]}; actual-reference gate pending")
    output = Path(os.environ.get("LMEX_EVIDENCE_DIR", tmp_path)) / f"architecture-{family}"
    measure(family, entries[family], output)


def test_integrated_startup_and_retrieval_never_execute_or_materialize(tmp_path, monkeypatch):
    import socket
    from unittest.mock import Mock

    import safetensors
    import torch
    import transformers
    from fastapi.testclient import TestClient
    from llm_model_explorer.app import create_app
    from llm_model_explorer.settings import Settings
    from llm_model_explorer.tensor_source import ModelSource

    root = tmp_path / "models"
    generate(root)
    forbidden = Mock(side_effect=AssertionError("Static startup executed or materialized weights"))
    for owner, names in (
        (socket, ["create_connection"]),
        (torch, ["load"]),
        (torch.nn.Module, ["__init__", "_call_impl"]),
        (torch.jit, ["trace", "script"]),
        (safetensors, ["safe_open"]),
        (transformers.AutoModel, ["from_config", "from_pretrained"]),
        (transformers.AutoModelForCausalLM, ["from_config", "from_pretrained"]),
        (transformers.AutoTokenizer, ["from_pretrained"]),
        (ModelSource, ["iter_tensor", "iter_rows"]),
    ):
        for name in names:
            monkeypatch.setattr(owner, name, forbidden)
    with TestClient(create_app(Settings(model_root=root, cache_dir=tmp_path / "cache"))) as client:
        for model in client.get("/models").json()["models"]:
            session = client.post("/sessions", json={"model_id": model["id"]}).json()["id"]
            body = client.get(f"/sessions/{session}/architecture").json()
            assert body["status"] == "available" and body["graph"]["coverage"] == "complete"
    forbidden.assert_not_called()

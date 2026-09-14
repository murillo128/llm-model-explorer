"""Production CLI/TCP acceptance of all packaged static descriptions."""

import hashlib
import json
import os
import shutil
import struct
from pathlib import Path

import pytest

from acceptance.architecture_fixtures import generate, value
from acceptance.test_network import Frames, Service
from api.architecture_conformance import validate_architecture


def inspect_graph(service, model_id):
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
    validate_architecture(
        body,
        {
            "session": session,
            "inventory": inventory,
            "tokenizer_available": False,
        },
    )
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


def test_all_descriptions_over_tcp_readonly_cold_warm_and_native_values(tmp_path):
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
            assert inventory["coverage"] == (
                "partial" if family in {"qwen3", "qwen35"} else "complete"
            )
            # Exercise a later concrete instance, both native ranks, and exact bytes.
            for rank in (1, 2):
                parameter = next(
                    p
                    for p in graph["parameters"]
                    if p["inspection"]["status"] == "available"
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

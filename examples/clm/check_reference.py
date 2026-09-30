"""Operator-selected actual CLM acceptance over production TCP, without inference."""

import argparse
import json
import struct
import time
from pathlib import Path

import torch
from safetensors import safe_open
from transformers import AutoTokenizer

from acceptance.test_network import Service
from examples.clm.export import ENCODER_REVISION, HEAD_REVISION, SOURCE_REVISION, digest


def numeric(response):
    response.raise_for_status()
    raw, values, terminal = response.content, bytearray(), None
    while raw:
        magic, kind, flags, reserved, length = struct.unpack("<4sBBHI", raw[:12])
        assert (magic, flags, reserved) == (b"LMEX", 0, 0)
        assert len(raw) >= 12 + length
        if kind == 2:
            values.extend(raw[12 : 12 + length])
        if kind >= 4:
            assert terminal is None
            terminal = kind
        raw = raw[12 + length :]
    assert terminal == 4
    return bytes(values)


def check(encoder, head, model_root, output):
    output.mkdir(parents=True, exist_ok=False)
    package = model_root / "clm"
    provenance = json.loads((package / "clm-provenance.json").read_text())
    assert provenance["encoder"]["revision"] == ENCODER_REVISION
    assert provenance["head"]["revision"] == HEAD_REVISION
    assert provenance["source"]["revision"] == SOURCE_REVISION
    assert digest(head) == provenance["head"]["sha256"]
    original = torch.load(head, map_location="cpu", weights_only=True, mmap=True)
    original_tokenizer = AutoTokenizer.from_pretrained(
        encoder, local_files_only=True, trust_remote_code=False
    )
    index = json.loads((encoder / "model.safetensors.index.json").read_text())["weight_map"]
    identity = "Contrastive-LM/CLM-v0.1-8B@" + HEAD_REVISION
    report = {
        "source_revision": SOURCE_REVISION,
        "encoder_revision": ENCODER_REVISION,
        "head_revision": HEAD_REVISION,
        "startup_seconds": {},
        "samples": {},
    }
    started = time.perf_counter()
    service = Service(output, model_root=model_root, startup_timeout=300, client_timeout=300)
    report["startup_seconds"]["cold"] = time.perf_counter() - started
    try:
        client = service.client
        models = client.get("/models").json()["models"]
        assert len({m["id"] for m in models}) == len(models)
        assert any(m["id"] == identity for m in models)
        assert any(m["id"] != identity for m in models)  # Native encoder coexists.
        sid = client.post("/sessions", json={"model_id": identity}).json()["id"]
        body = client.get(f"/sessions/{sid}/architecture").json()
        assert body["status"] == "available" and body["graph"]["coverage"] == "complete"
        graph = body["graph"]
        assert graph["scope"] == "model_defined"
        assert any(n["label"] == "Model-supplied definition" for n in graph["nodes"])
        assert graph["diagnostics"] == []
        report["graph"] = {
            "id": graph["graph_id"],
            "nodes": len(graph["nodes"]),
            "edges": len(graph["edges"]),
            "parameters": len(graph["parameters"]),
            "repetitions": [len(r["instances"]) for r in graph["repetitions"]],
            "sidecar_bytes": (package / "architecture.json").stat().st_size,
            "response_bytes": len(json.dumps(body).encode()),
        }
        inventory = client.get(f"/sessions/{sid}/tensors").json()["tensors"]
        report["inventory_count"] = len(inventory)
        for name in (
            "clm.state_head.out.weight",
            "clm.action_head.out.weight",
            "model.norm.weight",
        ):
            descriptor = next(t for t in inventory if t["name"] == name)
            if name.startswith("clm."):
                _, which, key = name.split(".", 2)
                expected = original[which][key].float()
            else:
                with safe_open(encoder / index[name], framework="pt", device="cpu") as weights:
                    expected = weights.get_tensor(name).float()
            data = numeric(client.get(f"/sessions/{sid}/tensors/{descriptor['id']}/data"))
            assert data == expected.numpy().tobytes()
            report["samples"][name] = {
                "shape": descriptor["shape"],
                "dtype": descriptor["storage_dtype"],
                "first_values": expected.flatten()[:4].tolist(),
                "exact_complete_stream": True,
            }
        text = "Hello world, España 😀"
        expected_ids = original_tokenizer(text, add_special_tokens=False)["input_ids"]
        tokenized = client.post(
            f"/sessions/{sid}/tokenize", json={"text": text, "add_special_tokens": False}
        ).json()
        ids = [token["id"] for token in tokenized["tokens"]]
        assert ids == expected_ids
        ids = ids + [ids[0]]
        name = "model.embed_tokens.weight"
        with safe_open(encoder / index[name], framework="pt", device="cpu") as weights:
            table = weights.get_slice(name)
            expected = torch.cat([table[row : row + 1].float() for row in ids])
        data = numeric(client.post(f"/sessions/{sid}/embeddings", json={"token_ids": ids}))
        assert data == expected.numpy().tobytes()
        report["tokenizer"] = {
            "text": text,
            "ids": ids,
            "exact_embedding_rows": True,
            "first_row_values": expected[0, :4].tolist(),
        }
    finally:
        service.stop()
        service.client.close()
    started = time.perf_counter()
    service = Service(output, model_root=model_root, startup_timeout=300, client_timeout=300)
    report["startup_seconds"]["warm"] = time.perf_counter() - started
    try:
        sid = service.client.post("/sessions", json={"model_id": identity}).json()["id"]
        body = service.client.get(f"/sessions/{sid}/architecture").json()
        assert body["graph"]["graph_id"] == report["graph"]["id"]
        assert f"model={identity} mode=cache" in (output / "service.log").read_text()
        report["warm_cache_reused"] = True
    finally:
        service.stop()
        service.client.close()
    (output / "reference.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--encoder", type=Path, required=True)
    parser.add_argument("--head", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    args = parser.parse_args()
    check(args.encoder, args.head, args.model_root, args.evidence)

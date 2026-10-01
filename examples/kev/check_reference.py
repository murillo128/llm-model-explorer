"""Actual Kev static acceptance over production TCP; no backbone inference."""

import argparse
import json
import time
from pathlib import Path

import torch
from safetensors import safe_open
from transformers import AutoTokenizer

from acceptance.test_network import Service
from examples.clm.check_reference import numeric
from examples.kev.export import (
    BASE_REVISION,
    EXPORT_IDENTITY,
    KEV_REVISION,
    PREFIX,
    SOURCE_REVISION,
    SPECIAL,
    digest,
)


def check(base, kev, model_root, output):
    output.mkdir(parents=True, exist_ok=False)
    package = model_root / "kev"
    provenance = json.loads((package / "kev-provenance.json").read_text())
    assert provenance["base"]["revision"] == BASE_REVISION
    assert provenance["kev"]["revision"] == KEV_REVISION
    assert provenance["source"]["revision"] == SOURCE_REVISION
    assert provenance["producer_sha256"] == digest(Path(__file__).with_name("export.py"))
    assert digest(package / "architecture.json") == provenance["definition_sha256"]
    for name, folder in (("base", base), ("kev", kev)):
        for f in provenance[name]["files"]:
            assert digest(folder / f["name"]) == f["sha256"]
    head = torch.load(kev / "head.pt", weights_only=True, map_location="cpu", mmap=True)
    assert head["base_revision"] == BASE_REVISION
    assert provenance["kev"]["head_metadata"]["temperature"] == head["temperature"]
    tokenizer = AutoTokenizer.from_pretrained(base, local_files_only=True, trust_remote_code=False)
    index = json.loads((base / "model.safetensors.index.json").read_text())["weight_map"]
    identity = EXPORT_IDENTITY + "@" + KEV_REVISION
    report = {
        "source_revision": SOURCE_REVISION,
        "base_revision": BASE_REVISION,
        "kev_revision": KEV_REVISION,
        "samples": {},
        "startup_seconds": {},
        "export_content": provenance["base"]["content_fingerprint"],
        "temperature": head["temperature"],
        "adapted_projections": len(provenance["targets"]),
    }
    started = time.perf_counter()
    service = Service(output, model_root=model_root, startup_timeout=300, client_timeout=300)
    report["startup_seconds"]["cold"] = time.perf_counter() - started
    try:
        client = service.client
        models = client.get("/models").json()["models"]
        assert any(m["id"] == identity for m in models)
        assert len({m["id"] for m in models}) == len(models)
        sid = client.post("/sessions", json={"model_id": identity}).json()["id"]
        body = client.get(f"/sessions/{sid}/architecture").json()
        assert body["status"] == "available"
        g = body["graph"]
        assert g["scope"] == "model_defined" and g["coverage"] == "complete"
        assert g["diagnostics"] == []
        assert any(n["label"] == "Model-supplied definition" for n in g["nodes"])
        report["graph"] = {
            "id": g["graph_id"],
            "nodes": len(g["nodes"]),
            "edges": len(g["edges"]),
            "parameters": len(g["parameters"]),
            "repetitions": [len(r["instances"]) for r in g["repetitions"]],
            "sidecar_bytes": (package / "architecture.json").stat().st_size,
            "response_bytes": len(json.dumps(body).encode()),
        }
        inventory = client.get(f"/sessions/{sid}/tensors").json()["tensors"]
        assert {t["name"] for t in inventory} == set(index) | {
            "kev.lora." + n for n in safe_keys(kev / "adapter_model.safetensors")
        } | {"kev.head." + n for n in head["head"]}
        report["inventory_count"] = len(inventory)
        names = [
            PREFIX + ".norm.weight",
            PREFIX + ".layers.0.linear_attn.in_proj_a.weight",
            "kev.lora.base_model.model.layers.0.linear_attn.in_proj_qkv.lora_A.weight",
            "kev.lora.base_model.model.layers.0.linear_attn.in_proj_qkv.lora_B.weight",
            "kev.head.q.weight",
            "kev.head.k.weight",
            "kev.head.q.bias",
            "kev.head.k.bias",
        ]
        for name in names:
            if name.startswith("kev.head."):
                expected = head["head"][name.removeprefix("kev.head.")].float()
            else:
                original = name.removeprefix("kev.lora.")
                shard = (
                    kev / "adapter_model.safetensors"
                    if name.startswith("kev.lora.")
                    else base / index[name]
                )
                with safe_open(shard, framework="pt", device="cpu") as f:
                    expected = f.get_tensor(original).float()
            descriptor = next(t for t in inventory if t["name"] == name)
            assert (
                numeric(client.get(f"/sessions/{sid}/tensors/{descriptor['id']}/data"))
                == expected.numpy().tobytes()
            )
            report["samples"][name] = {
                "shape": descriptor["shape"],
                "dtype": descriptor["storage_dtype"],
                "first_values": expected.flatten()[:4].tolist(),
                "exact_complete_stream": True,
            }
        text = "Hello world, España 😀"
        expected_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        tokenized = client.post(
            f"/sessions/{sid}/tokenize", json={"text": text, "add_special_tokens": False}
        ).json()
        ids = [t["id"] for t in tokenized["tokens"]]
        assert ids == expected_ids
        delimiters = tokenizer.convert_tokens_to_ids(SPECIAL)
        assert len(set(delimiters)) == 5 and all(type(i) is int and i >= 0 for i in delimiters)
        ids += delimiters + [ids[0]]
        name = PREFIX + ".embed_tokens.weight"
        with safe_open(base / index[name], framework="pt", device="cpu") as f:
            table = f.get_slice(name)
            expected = torch.cat([table[row : row + 1].float() for row in ids])
        assert (
            numeric(client.post(f"/sessions/{sid}/embeddings", json={"token_ids": ids}))
            == expected.numpy().tobytes()
        )
        report["tokenizer"] = {
            "text": text,
            "ids": ids,
            "delimiters": dict(zip(SPECIAL, delimiters, strict=True)),
            "exact_original_embedding_rows": True,
        }
    finally:
        service.stop()
        service.client.close()
    started = time.perf_counter()
    service = Service(output, model_root=model_root, startup_timeout=300, client_timeout=300)
    report["startup_seconds"]["warm"] = time.perf_counter() - started
    try:
        sid = service.client.post("/sessions", json={"model_id": identity}).json()["id"]
        assert (
            service.client.get(f"/sessions/{sid}/architecture").json()["graph"]["graph_id"]
            == report["graph"]["id"]
        )
        assert f"model={identity} mode=cache" in (output / "service.log").read_text()
        report["warm_cache_reused"] = True
    finally:
        service.stop()
        service.client.close()
    (output / "reference.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, sort_keys=True))


def safe_keys(path):
    with safe_open(path, framework="pt", device="cpu") as f:
        return f.keys()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--kev", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    args = parser.parse_args()
    check(args.base, args.kev, args.model_root, args.evidence)

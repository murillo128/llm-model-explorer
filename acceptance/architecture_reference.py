"""Offline, operator-selected reference evidence; never downloads or copies weights."""

import argparse
import json
import os
import platform
import re
import struct
import time
from pathlib import Path

from llm_model_explorer.models import ModelCatalogue

from acceptance.architecture_fixtures import FIXTURES
from acceptance.test_network import Service
from api.architecture_conformance import validate_architecture

REFERENCES = {
    "qwen3": "JunHowie/Qwen3-0.6B-GPTQ-Int4",
    "qwen35": "AxionML/Qwen3.5-0.8B-NVFP4",
    "vjepa2": "facebook/vjepa2-vitl-fpc64-256",
    "smollm2": "HuggingFaceTB/SmolLM2-135M",
}


def selections():
    supplied = os.environ.get("LMEX_ARCHITECTURE_REFERENCES")
    if not supplied:
        return {}
    entries = json.loads(Path(supplied).read_text())
    if not isinstance(entries, dict) or set(entries) - REFERENCES.keys():
        raise ValueError("Reference manifest must map approved family keys to selections")
    for family, entry in entries.items():
        if entry["repository"] != REFERENCES[family]:
            raise ValueError(f"Unapproved reference substitution for {family}")
        revision = entry.get("revision")
        if revision is None:
            if not entry.get("revision_absence_reason"):
                raise ValueError("Explain absent upstream revision")
        elif not re.fullmatch(r"[0-9a-f]{40}", revision):
            raise ValueError("Reference revision must be a full upstream SHA")
        Path(entry["directory"]).resolve(strict=True)
    return entries


def expected_storage(family):
    if family in {"qwen3", "smollm2"}:
        fixture = json.loads((FIXTURES / "dense-reference-metadata.json").read_text())[family]
        storage = dict(fixture["global_storage"])
        for i in range(fixture["config"]["num_hidden_layers"]):
            storage.update(
                {f"model.layers.{i}.{k}": v for k, v in fixture["layer_storage"].items()}
            )
        return {k: {"dtype": v[0], "shape": v[1]} for k, v in storage.items()}
    return json.loads((FIXTURES / f"{family}-reference.json").read_text())["storage"]


def inventory(family, selection):
    directory = Path(selection["directory"]).resolve(strict=True)
    entry = next(
        e for e in ModelCatalogue(directory.parent).discover() if e._snapshot.directory == directory
    )
    source = entry.pin()
    actual = {t.name: {"dtype": t.dtype, "shape": list(t.shape)} for t in source.physical_tensors()}
    assert actual == expected_storage(family), (
        "Actual checkpoint storage differs from approved reference"
    )
    assert entry.summary.tokenizer_available == (family != "vjepa2"), (
        "Missing advertised tokenizer assets"
    )
    files = [{"name": name, "bytes": stamp.size} for name, stamp in source._snapshot.files]
    config = source.configuration()
    if family in {"qwen3", "smollm2"}:
        reviewed = json.loads((FIXTURES / "dense-reference-metadata.json").read_text())[family][
            "config"
        ]
    else:
        reviewed = json.loads((FIXTURES / f"{family}-reference.json").read_text())["configuration"]
    assert config == reviewed, "Reference configuration differs from reviewed checkpoint metadata"
    return (
        directory,
        entry.summary.id,
        {
            "repository": selection["repository"],
            "revision": selection.get("revision"),
            "revision_source": "operator manifest",
            "revision_absence_reason": selection.get("revision_absence_reason"),
            "fingerprint": source.fingerprint,
            "files": files,
            "total_bytes": sum(f["bytes"] for f in files),
            "physical_tensors": len(actual),
            "quantization": config.get("quantization_config"),
            "checkpoint_declared_transformers_version": config.get("transformers_version"),
            "tokenizer_available": entry.summary.tokenizer_available,
        },
    )


def measure(family, selection, output):
    output.mkdir(parents=True, exist_ok=True)
    assert not (output / "cache").exists(), "Use a fresh evidence directory for a cold run"
    directory, model_id, report = inventory(family, selection)
    report["environment"] = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "machine": platform.machine(),
        "cpu_count": os.cpu_count(),
    }
    runs = []
    graph = None
    for mode in ("cold", "warm"):
        started = time.perf_counter()
        service = Service(output, model_root=directory.parent, startup_timeout=300)
        ready = time.perf_counter() - started
        try:
            session = service.client.post("/sessions", json={"model_id": model_id}).json()
            prefix = f"/sessions/{session['id']}"
            tensors = service.client.get(prefix + "/tensors").json()
            assert tensors["coverage"] == "complete", tensors["diagnostics"]
            response = service.client.get(prefix + "/architecture")
            response.raise_for_status()
            body = response.json()
            validate_architecture(
                body,
                {
                    "session": session,
                    "inventory": tensors,
                    "tokenizer_available": report["tokenizer_available"],
                },
            )
            assert body["status"] == "available", body
            current = body["graph"]
            assert current["coverage"] == "complete", current["diagnostics"]
            counts = [len(r["instances"]) for r in current["repetitions"]]
            assert (
                counts
                == {"qwen3": [28], "qwen35": [24], "vjepa2": [24, 12], "smollm2": [30]}[family]
            )
            if family == "qwen35":
                assert [
                    i["index"]
                    for i in current["repetitions"][0]["instances"]
                    if i["variant"] == "full_attention"
                ] == [3, 7, 11, 15, 19, 23]
            if graph is not None:
                assert current == graph
            graph = current
            assert str(directory.parent) not in response.text
            status = Path(f"/proc/{service.process.pid}/status").read_text()
            peak = int(re.search(r"VmHWM:\s+(\d+)", status)[1])
            runs.append({"mode": mode, "readiness_seconds": ready, "peak_rss_kib": peak})
        finally:
            service.stop()
            service.client.close()
    assert graph is not None
    manifests = [json.loads(p.read_text()) for p in (output / "cache").glob("*/manifest.json")]
    report.update(
        {
            "runs": runs,
            "scope": graph["scope"],
            "coverage": graph["coverage"],
            "nodes": len(graph["nodes"]),
            "edges": len(graph["edges"]),
            "parameters": len(graph["parameters"]),
            "graph_id": graph["graph_id"],
            "structured_manifests": [m for m in manifests if m["format"] != 1],
            "startup_measurements": [
                line
                for line in (output / "service.log").read_text().splitlines()
                if "Architecture model=" in line
            ],
        }
    )
    assert any(f"model={model_id} mode=cache" in line for line in report["startup_measurements"])
    (output / "reference.json").write_text(json.dumps(report, indent=2))
    return report


def native_samples(directory, name, *, row=None, column=None):
    """Independent safetensors slices of explicitly requested native values."""
    from safetensors import safe_open

    entry = next(
        e for e in ModelCatalogue(directory.parent).discover() if e._snapshot.directory == directory
    )
    tensor = next(t for t in entry.tensors() if t.name == name)
    assert tensor.rank in (1, 2)
    assert tensor.storage_format == "safetensors"
    assert tensor.storage_dtype in {"F32", "F16", "BF16"}, "Native sample needs native storage"
    for shard in entry._snapshot.shards:
        with safe_open(directory / shard, framework="pt", device="cpu") as weights:
            if name not in weights.keys():
                continue
            view = weights.get_slice(name)
            count = tensor.numel
            offsets = sorted({0, min(8, count - 1), count - 1})
            if row is not None or column is not None:
                assert tensor.rank == 2 and row is not None and column is not None
                assert 0 <= row < tensor.shape[0] and 0 <= column < tensor.shape[1]
                offsets = [row * tensor.shape[1] + column]
            values = []
            for offset in offsets:
                if tensor.rank == 1:
                    scalar = view[offset : offset + 1].float()[0]
                else:
                    row, column = divmod(offset, tensor.shape[1])
                    scalar = view[row : row + 1, column : column + 1].float()[0, 0]
                values.append({"offset": offset, "value": float(scalar)})
            return {"shape": list(tensor.shape), "samples": values}
    raise AssertionError("Admitted native tensor missing from checkpoint")


def packed_sample(directory, family, name, row, column):
    """Reuse the independent bounded scalar oracle, never the production decoder."""
    from backend.scripts.check_quantized_reference import CHECKPOINTS, scalar_reference

    encoding = CHECKPOINTS[family]["encoding"]
    entry = next(
        e for e in ModelCatalogue(directory.parent).discover() if e._snapshot.directory == directory
    )
    source = entry.pin()
    physical = {tensor.name: tensor for tensor in source.physical_tensors()}
    prefix = name.removesuffix(".weight")
    assert name == prefix + ".weight"
    suffixes = (
        ("qweight", "qzeros", "scales", "g_idx")
        if encoding == "gptq-int4"
        else ("weight", "weight_scale", "weight_scale_2", "input_scale")
    )
    storage = tuple(physical[prefix + "." + suffix] for suffix in suffixes)
    packed = storage[0]
    shape = (
        (packed.shape[1], packed.shape[0] * 8)
        if encoding == "gptq-int4"
        else (packed.shape[0], packed.shape[1] * 2)
    )
    assert row is not None and column is not None
    assert 0 <= row < shape[0] and 0 <= column < shape[1]
    offset = row * shape[1] + column
    expected, _, _ = scalar_reference(source._snapshot, storage, encoding, shape, offset, 1)
    return {
        "shape": list(shape),
        "samples": [{"offset": offset, "value": struct.unpack("<f", expected)[0]}],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("family", choices=REFERENCES)
    parser.add_argument("--tensor")
    parser.add_argument("--row", type=int)
    parser.add_argument("--column", type=int)
    parser.add_argument("--packed", action="store_true")
    args = parser.parse_args()
    directory, model_id, report = inventory(args.family, selections()[args.family])
    if args.tensor:
        result = (
            packed_sample(directory, args.family, args.tensor, args.row, args.column)
            if args.packed
            else native_samples(directory, args.tensor, row=args.row, column=args.column)
        )
        print(json.dumps(result))
    else:
        print(json.dumps({"directory": str(directory), "model_id": model_id, "report": report}))

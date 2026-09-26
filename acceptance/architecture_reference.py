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
    "smollm2_bnb": "unsloth/SmolLM2-135M-bnb-4bit",
    "deepseek_v2": "slowfastai/DeepSeek-V2-Lite-bnb-4bit",
    "glm4_moe_lite": "cyankiwi/GLM-4.7-Flash-AWQ-4bit",
    "kimi_linear": "cyankiwi/Kimi-Linear-48B-A3B-Instruct-AWQ-4bit",
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


def header_storage(directory, shards):
    """Read Safetensors descriptors only; do not materialize any model weight."""
    from safetensors import safe_open

    storage = {}
    for shard in shards:
        with safe_open(directory / shard, framework="pt", device="cpu") as weights:
            for name in weights.keys():  # noqa: SIM118 - safetensors.safe_open is not a mapping
                tensor = weights.get_slice(name)
                storage[name] = {
                    "dtype": tensor.get_dtype(),
                    "shape": list(tensor.get_shape()),
                }
    return storage


def inventory(family, selection):
    directory = Path(selection["directory"]).resolve(strict=True)
    # Inspect only the selected checkpoint. A full catalogue scan belongs to
    # the integrated application check and would needlessly rehash every
    # multi-gigabyte sibling for each browser oracle sample.
    entry = ModelCatalogue(directory.parent)._inspect_base(directory).entry
    source = entry.pin()
    actual = {t.name: {"dtype": t.dtype, "shape": list(t.shape)} for t in source.physical_tensors()}
    if family in {
        "smollm2_bnb",
        "deepseek_v2",
        "glm4_moe_lite",
        "kimi_linear",
    }:
        expected = header_storage(directory, source._snapshot.shards)
        assert actual == expected, "Catalogue physical inventory differs from Safetensors headers"
    else:
        assert actual == expected_storage(family), (
            "Actual checkpoint storage differs from approved reference"
        )
    assert entry.summary.tokenizer_available == (family not in {"vjepa2", "kimi_linear"}), (
        "Missing advertised tokenizer assets"
    )
    files = [{"name": name, "bytes": stamp.size} for name, stamp in source._snapshot.files]
    config = source.configuration()
    compatibility_deviations = {}
    if family == "smollm2_bnb":
        reviewed = json.loads((FIXTURES / "dense-reference-metadata.json").read_text())["smollm2"][
            "config"
        ]
        for key, value in reviewed.items():
            if key not in {
                "_name_or_path",
                "transformers_version",
                "head_dim",
                "mlp_bias",
                "pad_token_id",
                "vocab_size",
            }:
                assert config[key] == value, f"SmolLM2 quantized base disagrees on {key}"
        assert config["vocab_size"] == reviewed["vocab_size"] + 1
        compatibility_deviations = {
            key: {"base": reviewed.get(key), "quantized": config.get(key)}
            for key in sorted(set(reviewed) | set(config))
            if reviewed.get(key) != config.get(key)
        }
        quantization = config["quantization_config"]
        assert quantization["quant_method"] == "bitsandbytes"
        assert quantization["bnb_4bit_quant_type"] == "nf4"
        assert quantization["bnb_4bit_use_double_quant"] is True
    elif family in {"qwen3", "smollm2"}:
        reviewed = json.loads((FIXTURES / "dense-reference-metadata.json").read_text())[family][
            "config"
        ]
    else:
        fixture_name = {
            "deepseek_v2": "deepseek-v2-lite",
            "glm4_moe_lite": "glm4-moe-lite",
            "kimi_linear": "kimi-linear",
        }[family]
        reviewed = json.loads((FIXTURES / f"{fixture_name}-reference.json").read_text())[
            "configuration"
        ]
    if family != "smollm2_bnb":
        comparable = dict(config)
        if family == "kimi_linear" and "pad_token_id" not in reviewed:
            assert config.get("pad_token_id") == 163839
            comparable.pop("pad_token_id")
            compatibility_deviations["pad_token_id"] = {
                "reviewed": None,
                "checkpoint": config["pad_token_id"],
            }
        assert comparable == reviewed, (
            "Reference configuration differs from reviewed checkpoint metadata"
        )
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
            "configuration_compatibility_deviations": compatibility_deviations,
            "checkpoint_declared_transformers_version": config.get("transformers_version"),
            "tokenizer_available": entry.summary.tokenizer_available,
        },
    )


def link_model_root(family, destination, *, include_adapter=False):
    """Make a small hard-link view of actual local files for isolated UI startups."""
    import os
    import shutil

    entries = selections()
    model_directory = Path(entries[family]["directory"]).resolve(strict=True)
    destination = Path(destination)
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True)

    def add_tree(source, target):
        target.mkdir()
        for child in source.iterdir():
            if child.is_file():
                os.link(child, target / child.name)

    add_tree(model_directory, destination / model_directory.name)
    if include_adapter:
        adapter_directories = [
            path
            for path in model_directory.parent.iterdir()
            if path.is_dir()
            and (path / "adapter_config.json").is_file()
            and (path / "adapter_model.safetensors").is_file()
        ]
        if len(adapter_directories) != 1:
            raise AssertionError("Expected exactly one selected LoRA adapter")
        add_tree(adapter_directories[0], destination / adapter_directories[0].name)
    for path in destination.rglob("*"):
        path.chmod(0o755 if path.is_dir() else 0o444)
    destination.chmod(0o755)
    return destination


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
        service = Service(
            output,
            model_root=directory.parent,
            startup_timeout=300,
            client_timeout=600,
        )
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
                == {
                    "qwen3": [28],
                    "qwen35": [24],
                    "vjepa2": [24, 12],
                    "smollm2": [30],
                    "deepseek_v2": [27],
                    "glm4_moe_lite": [47],
                    "kimi_linear": [27],
                }[family]
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

    index = directory / "model.safetensors.index.json"
    weight_map = json.loads(index.read_text())["weight_map"] if index.is_file() else {}
    shards = (
        [weight_map[name]]
        if name in weight_map
        else [p.name for p in directory.glob("*.safetensors")]
    )
    for shard in shards:
        with safe_open(directory / shard, framework="pt", device="cpu") as weights:
            if name not in weights.keys():  # noqa: SIM118 - safetensors.safe_open is not a mapping
                continue
            view = weights.get_slice(name)
            shape = tuple(view.get_shape())
            dtype = view.get_dtype()
            assert len(shape) in (1, 2)
            assert dtype in {"F32", "F16", "BF16"}, "Native sample needs native storage"
            count = 1
            for dimension in shape:
                count *= dimension
            offsets = sorted({0, min(8, count - 1), count - 1})
            if row is not None or column is not None:
                assert len(shape) == 2 and row is not None and column is not None
                assert 0 <= row < shape[0] and 0 <= column < shape[1]
                offsets = [row * shape[1] + column]
            values = []
            for offset in offsets:
                if len(shape) == 1:
                    scalar = view[offset : offset + 1].float()[0]
                else:
                    row, column = divmod(offset, shape[1])
                    scalar = view[row : row + 1, column : column + 1].float()[0, 0]
                values.append({"offset": offset, "value": float(scalar)})
            return {"shape": list(shape), "samples": values}
    raise AssertionError("Admitted native tensor missing from checkpoint")


def packed_sample(directory, family, name, row, column):
    """Reuse the independent bounded scalar oracle, never the production decoder."""
    if family in {"deepseek_v2", "glm4_moe_lite", "kimi_linear"}:
        from acceptance.quantized_reference import scalar_sample

        sample = scalar_sample(directory, name, row, column)
        return {
            "shape": sample["shape"],
            "samples": [{"offset": sample["offset"], "value": sample["value"]}],
        }
    from backend.scripts.check_quantized_reference import CHECKPOINTS, scalar_reference

    encoding = CHECKPOINTS[family]["encoding"]
    entry = ModelCatalogue(directory.parent)._inspect_base(directory).entry
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


def logical_value_stream(service, session_id, tensor_id, *, numel):
    """Collect a production float32 stream while validating its independent framing."""
    from acceptance.test_network import Frames

    with service.client.stream(
        "GET", f"/sessions/{session_id}/tensors/{tensor_id}/data"
    ) as response:
        frames = Frames(response)
        kind, metadata = frames.next()
        assert kind == 1
        header = json.loads(metadata)
        payload = bytearray()
        while True:
            kind, data = frames.next()
            if kind == 4:
                break
            assert kind == 2
            payload.extend(data)
        assert len(payload) == 4 * numel
        return header, payload


def validate_pinned_architecture(family, graph):
    """Check the issue's model-specific structural claims against the HTTP graph."""
    from api.architecture_conformance import expand_compact_graph

    graph = expand_compact_graph(graph)
    parameters = {parameter["name"]: parameter for parameter in graph["parameters"]}
    assert graph["coverage"] == "complete"

    def attributes(node):
        return {item["name"]: item["value"] for item in node["attributes"]}

    def role_nodes(role):
        return [node for node in graph["nodes"] if attributes(node).get("semantic_role") == role]

    if family == "deepseek_v2":
        layer_repetition = next(
            item for item in graph["repetitions"] if item["label"] == "Decoder layers"
        )
        assert [(item["index"], item["variant"]) for item in layer_repetition["instances"]] == [
            (0, "dense"),
            *[(index, "moe") for index in range(1, 27)],
        ]
        expert_repetitions = [
            item for item in graph["repetitions"] if item["label"] == "Routed experts"
        ]
        assert len(expert_repetitions) == 26
        assert all(len(item["instances"]) == 64 for item in expert_repetitions)
        assert "model.layers.1.self_attn.kv_a_proj_with_mqa.weight" in parameters
        assert "model.layers.1.mlp.experts.0.gate_proj.weight" in parameters
        assert "model.layers.1.mlp.shared_experts.gate_proj.weight" in parameters
        selections = role_nodes("topk_selection")
        assert len(selections) == 26
        assert all(attributes(node).get("top_k") == 6 for node in selections)
    elif family == "glm4_moe_lite":
        layers = next(item for item in graph["repetitions"] if item["label"] == "Decoder layers")
        assert len(layers["instances"]) == 47
        assert layers["instances"][0]["variant"] == "dense"
        assert all(item["variant"] == "sparse" for item in layers["instances"][1:])
        expert_repetitions = [
            item for item in graph["repetitions"] if item["label"] == "Routed experts"
        ]
        assert len(expert_repetitions) == 46
        assert all(len(item["instances"]) == 64 for item in expert_repetitions)
        assert any(name.startswith("model.layers.1.mlp.experts.") for name in parameters)
        assert not any(name.startswith("model.layers.47.") for name in parameters)
        root = next(node for node in graph["nodes"] if "parent_id" not in node)
        attrs = attributes(root)
        assert attrs["num_nextn_predict_layers"] == 1
        assert "metadata only" in attrs["nextn_evaluation"]
        selections = role_nodes("topk_selection")
        assert len(selections) == 92
        assert sum(attributes(node).get("top_k") == 4 for node in selections) == 46
        assert sum(attributes(node).get("top_k") == 1 for node in selections) == 46
    elif family == "kimi_linear":
        layers = next(item for item in graph["repetitions"] if item["label"] == "Decoder layers")
        assert len(layers["instances"]) == 27
        assert sum(".block_sparse_moe.experts." in name for name in parameters) == 26 * 256 * 3
        assert "model.layers.1.block_sparse_moe.experts.0.w1.weight" in parameters
        layer_nodes = [
            node
            for node in graph["nodes"]
            if node["label"].startswith("Decoder layer ") and "attention_type" in attributes(node)
        ]
        attention_types = {
            int(node["label"].rsplit(" ", maxsplit=1)[1]): attributes(node)["attention_type"]
            for node in layer_nodes
        }
        assert len(attention_types) == 27
        assert sum(value == "KDA" for value in attention_types.values()) == 20
        assert sum(value == "MLA" for value in attention_types.values()) == 7
        selections = role_nodes("kimi_grouped_topk_selection")
        assert len(selections) == 26
        assert all(attributes(node).get("top_k") == 8 for node in selections)
    else:
        raise ValueError(f"No pinned architecture oracle for {family!r}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("family", choices=REFERENCES)
    parser.add_argument("--tensor")
    parser.add_argument("--row", type=int)
    parser.add_argument("--column", type=int)
    parser.add_argument("--packed", action="store_true")
    parser.add_argument(
        "--link-root", help="Build an isolated hard-link model root for a browser run"
    )
    args = parser.parse_args()
    if args.tensor:
        directory = Path(selections()[args.family]["directory"]).resolve(strict=True)
        result = (
            packed_sample(directory, args.family, args.tensor, args.row, args.column)
            if args.packed
            else native_samples(directory, args.tensor, row=args.row, column=args.column)
        )
        print(json.dumps(result))
    else:
        directory, model_id, report = inventory(args.family, selections()[args.family])
        model_root = None
        if args.link_root:
            model_root = str(link_model_root(args.family, args.link_root))
        print(
            json.dumps(
                {
                    "directory": str(directory),
                    "model_id": model_id,
                    "report": report,
                    "model_root": model_root,
                }
            )
        )

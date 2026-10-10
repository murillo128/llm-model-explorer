"""Tiny local checkpoints for filesystem-to-browser hot reload (no mocked events)."""

import argparse
import copy
import json
from pathlib import Path

import torch
from safetensors.torch import save_file
from tokenizers import Tokenizer, models, pre_tokenizers, processors
from transformers import PreTrainedTokenizerFast


def weights(directory: Path, value: int) -> None:
    tensors = {
        "encoder.proj.weight": torch.full((4, 3), float(value)),
        "encoder.proj.bias": torch.full((4,), float(value)),
        "model.embed_tokens.weight": torch.arange(32).reshape(8, 4).float() + value,
    }
    temporary = directory / "replacement.tmp"
    save_file(tensors, temporary)
    temporary.replace(directory / "model.safetensors")


def generate(root: Path) -> None:
    example = (
        Path(__file__).resolve().parents[1] / "examples/model-owned-architecture/architecture.json"
    )
    source = json.loads(example.read_text())
    shape3 = source["nodes"][1]["ports"][0]["shape"]
    shape4 = source["nodes"][1]["ports"][1]["shape"]
    document = {k: copy.deepcopy(v) for k, v in source.items() if k not in {"nodes", "edges"}}
    document["nodes"] = [
        {
            "id": "bank",
            "kind": "group",
            "label": "Parallel encoders",
            "children": ["stage0", "stage1"],
            "ports": [],
        }
    ]
    document["edges"] = []
    for index in range(2):
        suffix = str(index)
        stage = {
            "id": f"stage{index}",
            "kind": "group",
            "label": f"Stage {index}",
            "parent_id": "bank",
            "children": [f"encoder{index}"],
            "ports": [],
        }
        document["nodes"].append(stage)
        for original in source["nodes"]:
            node = copy.deepcopy(original)
            node["id"] += suffix
            if "parent_id" in node:
                node["parent_id"] += suffix
            elif node["id"] == f"encoder{index}":
                node["parent_id"] = stage["id"]
            if "children" in node:
                node["children"] = [child + suffix for child in node["children"]]
            if node["id"] == f"projection{index}":
                node["formula"] = "out = x W^T + b"
            document["nodes"].append(node)
        for original in source["edges"]:
            edge = copy.deepcopy(original)
            edge["id"] += suffix
            for end in ("source", "target"):
                edge[end]["node_id"] += suffix
            document["edges"].append(edge)
    # Give enclosing stages exact forwarded interfaces (never skip boundaries).
    for index in range(2):
        stage = next(n for n in document["nodes"] if n["id"] == f"stage{index}")
        stage["ports"] = [
            {"id": "x", "label": "x", "direction": "input", "shape": shape3},
            {"id": "out", "label": "out", "direction": "output", "shape": shape4},
        ]
        # The bank is a grouping boundary with explicit per-branch ports.
        bank = document["nodes"][0]
        bank["ports"].extend([{**p, "id": p["id"] + str(index)} for p in stage["ports"]])
        for edge in document["edges"]:
            if edge["id"] == f"input{index}":
                edge["target"]["node_id"] = "bank"
                edge["target"]["port_id"] = f"x{index}"
            if edge["id"] == f"output{index}":
                edge["source"]["node_id"] = "bank"
                edge["source"]["port_id"] = f"out{index}"
        for name, source_id, source_port, target_id, target_port in [
            ("enter", "bank", f"x{index}", stage["id"], "x"),
            ("inner", stage["id"], "x", f"encoder{index}", "x"),
            ("leave", f"encoder{index}", "out", stage["id"], "out"),
            ("outer", stage["id"], "out", "bank", f"out{index}"),
        ]:
            document["edges"].append(
                {
                    "id": name + str(index),
                    "source": {"node_id": source_id, "port_id": source_port},
                    "target": {"node_id": target_id, "port_id": target_port},
                }
            )
    document["repetitions"] = [
        {
            "id": "stages",
            "parent_id": "bank",
            "label": "Stages",
            "instances": [
                {"node_id": f"stage{i}", "index": i, "variant": "encoder"} for i in range(2)
            ],
        }
    ]
    for name in ("a", "b"):
        directory = root / name
        directory.mkdir(parents=True)
        (directory / "config.json").write_text(
            json.dumps(
                {
                    "_name_or_path": f"reload/{name}",
                    "model_type": "llama",
                    "architectures": ["LlamaForCausalLM"],
                    "vocab_size": 8,
                    "hidden_size": 4,
                }
            )
        )
        weights(directory, 1)
        native = Tokenizer(
            models.WordLevel(
                {"<unk>": 0, "<bos>": 1, "alpha": 2, "beta": 3, "!": 4}, unk_token="<unk>"
            )
        )
        native.pre_tokenizer = pre_tokenizers.Whitespace()
        native.post_processor = processors.TemplateProcessing(
            single="<bos> $A", special_tokens=[("<bos>", 1)]
        )
        PreTrainedTokenizerFast(
            tokenizer_object=native, unk_token="<unk>", bos_token="<bos>"
        ).save_pretrained(directory)
        (directory / "architecture.json").write_text(json.dumps(document))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--value", type=int)
    args = parser.parse_args()
    torch.set_num_threads(2)
    if args.value is None:
        generate(args.root)
    else:
        weights(args.root, args.value)

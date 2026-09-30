"""Explicit offline, unmerged Kev inspection export; never imported by the server."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import re
import shutil
import tempfile
from collections import OrderedDict
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch
from llm_model_explorer.architecture_analysis import AnalysisInput, GraphBuilder
from llm_model_explorer.architecture_analysis.model_defined_schema import (
    MAX_DEFINITION_BYTES,
)
from llm_model_explorer.architecture_analysis.qwen35 import (
    PREFIX,
    build,
    configuration,
    parameter_shapes,
)
from llm_model_explorer.architecture_analysis.qwen35 import (
    PRODUCER as QWEN_PRODUCER,
)
from llm_model_explorer.model_files import FileSnapshot, read_json
from llm_model_explorer.models import ModelCatalogue
from llm_model_explorer.peft_adapters import _factor_module, _target_patterns
from llm_model_explorer.tensor_source import parse_header
from llm_model_explorer.validate_architecture_cli import validate_directory
from safetensors import safe_open
from safetensors.torch import save_file

SOURCE_REVISION = "45923b7a3460b6d36358e2e143455902c1eb856b"
KEV_REVISION = "9a45d25eb2ab761841196625383fa1dff0e56c1e"
BASE_REVISION = "dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68"
BASE_REPOSITORY = "Qwen/Qwen3.5-0.8B-Base"
KEV_REPOSITORY = "jaredpalmer/kev-0.8b"
EXPORT_IDENTITY = "jaredpalmer/kev-0.8b-inspection"
PRODUCER = "kev-inspection-export-1"
SOURCE_FILES = {
    "kev/model.py": "2634ffe7747d69473fb6596942c248e5df2586df3610bd1adb47e7e9acd99f96",
    "kev/checkpoint.py": "f3edb4d159c0aa12d7006fc599fe6ebc3f2c288b3b69890c289f7dac451dd03d",
    "LICENSE": "6b08bb37982c233aa12bcbdf19106da12f3f4fcf800773ea42e28ebeddd34fb8",
}
NATIVE = {torch.float32, torch.float16, torch.bfloat16}
SPECIAL = [
    "<|fim_prefix|>",
    "<|fim_middle|>",
    "<|box_start|>",
    "<|box_end|>",
    "<|fim_suffix|>",
]


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            result.update(block)
    return result.hexdigest()


def primitive(value: Any, depth: int = 0) -> None:
    """Accept only bounded JSON-like metadata, without checkpoint objects."""
    if depth > 16:
        raise ValueError("Checkpoint metadata nesting is excessive")
    if value is None or type(value) in (str, bool, int):
        return
    if type(value) is float and math.isfinite(value):
        return
    if type(value) is list and len(value) <= 10000:
        for item in value:
            primitive(item, depth + 1)
        return
    if type(value) is dict and len(value) <= 10000 and all(type(k) is str for k in value):
        for item in value.values():
            primitive(item, depth + 1)
        return
    raise ValueError("Unsupported checkpoint metadata object")


def checked_head(
    path: Path, hidden: int, revision: str
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    if not path.is_file() or path.stat().st_size > 64 * 1024 * 1024:
        raise ValueError("Head must be a regular checkpoint of at most 64 MiB")
    payload = torch.load(path, weights_only=True, map_location="cpu", mmap=True)
    fields = {
        "base",
        "base_revision",
        "head",
        "head_dim",
        "lora",
        "option_isolation",
        "special_embeddings",
        "weights_dtype",
        "temperature",
        "holdout",
        "lora_placement",
        "args",
        "suite_sha256",
        "init_source",
        "temperature_fit",
    }
    required = {
        "head",
        "base",
        "base_revision",
        "head_dim",
        "lora",
        "option_isolation",
        "special_embeddings",
        "weights_dtype",
        "temperature",
    }
    if type(payload) is not dict or set(payload) - fields or not required <= payload.keys():
        raise ValueError("Unsupported head payload")
    metadata = {k: v for k, v in payload.items() if k != "head"}
    primitive(metadata)
    if metadata.get("base") != BASE_REPOSITORY or metadata.get("base_revision") != revision:
        raise ValueError("Head base repository/revision mismatch")
    args = metadata.get("args", {})
    if type(args) is not dict:
        raise ValueError("Invalid head training metadata")
    for k in ("base", "base_revision", "head_dim", "lora", "lora_placement"):
        if k in args and args[k] != metadata.get(k):
            raise ValueError("Conflicting head training metadata")
    if (
        metadata.get("option_isolation") is not False
        or metadata.get("special_embeddings") is not False
        or metadata.get("lora_placement", "full") != "full"
        or metadata.get("weights_dtype") != "fp32"
    ):
        raise ValueError("Unsupported hybrid inference options")
    temperature = metadata.get("temperature")
    if type(temperature) not in (int, float) or not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("Inference temperature must be finite and positive")
    rank = metadata.get("lora")
    if type(rank) is not int or not 0 < rank <= 16384:
        raise ValueError("Invalid head LoRA rank")
    dp = metadata.get("head_dim")
    if type(dp) is not int or not 0 < dp <= 16384:
        raise ValueError("Invalid pointer dimension")
    expected = {
        "q.weight": (dp, hidden),
        "q.bias": (dp,),
        "k.weight": (dp, hidden),
        "k.bias": (dp,),
    }
    state = payload["head"]
    if type(state) not in (dict, OrderedDict) or set(state) != set(expected):
        raise ValueError("Missing or extra pointer head tensor")
    tensors = {}
    logical_bytes = 0
    for key, dims in expected.items():
        t = state[key]
        if (
            type(t) is not torch.Tensor
            or t.dtype not in NATIVE
            or tuple(t.shape) != dims
            or t.layout != torch.strided
        ):
            raise ValueError("Pointer tensor shape, dtype or value mismatch")
        logical_bytes += t.numel() * t.element_size()
        if logical_bytes > 64 * 1024 * 1024:
            raise ValueError("Pointer tensors exceed the conversion budget")
        if not bool(torch.isfinite(t).all()):
            raise ValueError("Nonfinite pointer tensor")
        tensors["kev.head." + key] = t.contiguous()
    # Preserve scalar precision as metadata; do not synthesize a weight tensor.
    selected = {
        k: metadata[k]
        for k in (
            "base",
            "base_revision",
            "head_dim",
            "lora",
            "option_isolation",
            "special_embeddings",
            "weights_dtype",
            "temperature",
        )
    }
    selected["lora_placement"] = metadata.get("lora_placement", "full")
    return selected, tensors


def checked_adapter(kev: Path, expected: dict[str, list[int]], revision: str):
    snapshot = FileSnapshot.capture(kev.parent, kev, ("adapter_model.safetensors",))
    config = read_json(kev.parent, kev / "adapter_config.json")
    allowed = {
        "alora_invocation_tokens",
        "alpha_pattern",
        "arrow_config",
        "auto_mapping",
        "base_model_name_or_path",
        "bias",
        "corda_config",
        "ensure_weight_tying",
        "eva_config",
        "exclude_modules",
        "fan_in_fan_out",
        "inference_mode",
        "init_lora_weights",
        "kasa_config",
        "layer_replication",
        "layers_pattern",
        "layers_to_transform",
        "loftq_config",
        "lora_alpha",
        "lora_bias",
        "lora_dropout",
        "lora_ga_config",
        "megatron_config",
        "megatron_core",
        "modules_to_save",
        "monteclora_config",
        "peft_type",
        "peft_version",
        "qalora_group_size",
        "r",
        "rank_pattern",
        "revision",
        "target_modules",
        "target_parameters",
        "task_type",
        "trainable_token_indices",
        "use_bdlora",
        "use_dora",
        "use_qalora",
        "use_rslora",
        "velora_config",
    }
    if (
        set(config) - allowed
        or config.get("peft_type") != "LORA"
        or config.get("task_type") != "FEATURE_EXTRACTION"
        or config.get("base_model_name_or_path") != BASE_REPOSITORY
        or config.get("revision") not in (None, revision)
    ):
        raise ValueError("Unsupported Kev adapter or base revision")
    for k in allowed - {
        "base_model_name_or_path",
        "bias",
        "fan_in_fan_out",
        "inference_mode",
        "init_lora_weights",
        "lora_alpha",
        "lora_dropout",
        "megatron_core",
        "peft_type",
        "peft_version",
        "qalora_group_size",
        "r",
        "revision",
        "target_modules",
        "task_type",
        "ensure_weight_tying",
        "lora_bias",
        "use_dora",
        "use_qalora",
        "use_rslora",
        "use_bdlora",
    }:
        if config.get(k) not in (None, {}, []):
            raise ValueError("Unsupported inference-changing adapter option: " + k)
    for k in (
        "ensure_weight_tying",
        "lora_bias",
        "use_dora",
        "use_qalora",
        "use_rslora",
        "use_bdlora",
    ):
        if config.get(k) not in (None, False):
            raise ValueError("Unsupported adapter extension")
    if (
        config.get("bias") != "none"
        or config.get("fan_in_fan_out") is not False
        or config.get("megatron_core") not in (None, "megatron.core")
    ):
        raise ValueError("Unsupported LoRA orientation or bias")
    for flag in ("inference_mode", "init_lora_weights"):
        if flag in config and type(config[flag]) is not bool:
            raise ValueError("Unsupported adapter flag")
    rank, alpha, dropout = (
        config.get("r"),
        config.get("lora_alpha"),
        config.get("lora_dropout", 0),
    )
    if (
        type(rank) is not int
        or not 0 < rank <= 16384
        or type(alpha) not in (int, float)
        or not math.isfinite(alpha)
        or alpha <= 0
        or type(dropout) not in (int, float)
        or not math.isfinite(dropout)
        or not 0 <= dropout <= 1
    ):
        raise ValueError("Invalid adapter rank, alpha or dropout")
    targets = _target_patterns(config.get("target_modules"))
    modules = {
        name[:-7]: dims
        for name, dims in expected.items()
        if len(dims) == 2
        and re.fullmatch(
            re.escape(PREFIX) + r"\.layers\.\d+\.(?:mlp\.(?:gate_proj|up_proj|down_proj)|"
            r"self_attn\.(?:q_proj|k_proj|v_proj|o_proj)|"
            r"linear_attn\.(?:in_proj_qkv|in_proj_z|in_proj_a|in_proj_b|out_proj))\.weight",
            name,
        )
    }
    # PEFT AutoModel sees the text backbone as layers.*, while the native
    # conditional-generation checkpoint stores model.language_model.layers.*.
    aliases = {name.removeprefix(PREFIX + "."): name for name in modules}
    wanted = set()
    for pattern, regex in targets:
        matches = {
            native
            for alias, native in aliases.items()
            if (
                re.fullmatch(pattern, alias)
                if regex
                else alias == pattern or alias.endswith("." + pattern)
            )
        }
        if not matches:
            raise ValueError("Unknown adapter target: " + pattern)
        wanted.update(matches)
    physical = parse_header(snapshot, "adapter_model.safetensors")
    factors: dict[str, dict[str, str]] = {}
    names = set()
    for p in physical:
        module, factor, adapter_name = _factor_module(p.name)
        names.add(adapter_name)
        if (
            module not in aliases
            or aliases[module] not in wanted
            or p.dtype not in ("F32", "F16", "BF16")
        ):
            raise ValueError("Orphan or unsupported adapter factor")
        native = aliases[module]
        pair = factors.setdefault(native, {})
        if factor in pair:
            raise ValueError("Duplicate adapter factor")
        out, inp = modules[native]
        if p.shape != ((rank, inp) if factor == "A" else (out, rank)):
            raise ValueError("Adapter factor shape mismatch")
        pair[factor] = p.name
    if (
        set(factors) != wanted
        or len(names) != 1
        or any(set(v) != {"A", "B"} for v in factors.values())
    ):
        raise ValueError("Missing or ambiguous adapter factors")
    if {n for n, _ in snapshot.files if n.endswith(".safetensors")} != {
        "adapter_model.safetensors"
    }:
        raise ValueError("Unsupported extra adapter shards")
    return config, factors, snapshot


def shape(*dims):
    return [
        {"kind": "constant", "value": d} if type(d) is int else {"kind": "symbol", "name": d}
        for d in dims
    ]


def port(key, dims, output=False):
    return {
        "id": key,
        "label": key,
        "direction": "output" if output else "input",
        "shape": dims,
    }


class Definition:
    def __init__(self):
        self.nodes = []
        self.edges = []

    def edge(self, source, target, sp="out", tp="x", kind="data"):
        self.edges.append(
            {
                "id": "kev.edge." + str(len(self.edges)),
                "source": {"node_id": source, "port_id": sp},
                "target": {"node_id": target, "port_id": tp},
                "kind": kind,
            }
        )

    def op(
        self,
        key,
        label,
        inputs,
        output,
        *,
        parameters=(),
        formula=None,
        attributes=None,
        parent=None,
        kind="operation",
    ):
        n = {
            "id": key,
            "kind": kind,
            "label": label,
            "ports": [port(k, v) for k, v in inputs.items()] + [port("out", output, True)],
            "parameter_ids": list(parameters),
        }
        if kind == "operation":
            n["operation"] = key.split(".")[-1]
        if formula:
            n["formula"] = formula
        if attributes:
            n["attributes"] = [{"name": k, "value": v} for k, v in attributes.items()]
        if parent:
            n["parent_id"] = parent
        self.nodes.append(n)
        return key


def architecture(inputs, config, factors, adapter, head, tensors):
    normalized = copy.deepcopy(config)
    for key in ("_commit_hash", "revision", "name_or_path"):
        normalized.pop(key, None)
    # The pinned Qwen configuration inherits this from rope_parameters.
    normalized["text_config"].setdefault(
        "partial_rotary_factor",
        normalized["text_config"]["rope_parameters"]["partial_rotary_factor"],
    )
    inputs = replace(inputs, configuration=normalized)
    c = configuration(inputs)
    if c is None:
        raise ValueError("Unreviewed Qwen3.5 text configuration")
    b = GraphBuilder(inputs, QWEN_PRODUCER, "language_model")
    build(inputs, b)
    graph = b.finish().document()
    d = Definition()
    pid = {p["id"]: p["name"] for p in graph["parameters"]}
    # Fused query/gate regions are descriptions, not new tensors. Their split
    # consumes the projected value and needs no extra matrix binding.
    excluded = {
        n["id"]
        for n in graph["nodes"]
        if n.get("operation") in ("linear", "vocabulary_logits")
        and ("lm_head.weight" in [pid[p] for p in n["parameter_ids"]] or n["kind"] == "output")
    }
    excluded.update(n["id"] for n in graph["nodes"] if n["kind"] == "input")
    root = next(n for n in graph["nodes"] if n["kind"] == "group" and not n.get("parent_id"))
    ids = {}
    for n in graph["nodes"]:
        key = next(
            (
                p["source"]
                for p in n["provenance"]
                if p.get("rule") == "Semantic source key in the reviewed packaged description"
            ),
            n["id"],
        )
        ids[n["id"]] = key
    ids[root["id"]] = "backbone"
    for original in graph["nodes"]:
        if original["id"] in excluded:
            continue
        n = {
            k: copy.deepcopy(v)
            for k, v in original.items()
            if k
            in {
                "id",
                "kind",
                "label",
                "ports",
                "operation",
                "parent_id",
                "children",
                "parameter_ids",
                "formula",
                "description",
                "attributes",
                "references",
            }
        }
        n["id"] = ids[original["id"]]
        n["parameter_ids"] = [
            pid[p] for p in original["parameter_ids"] if pid[p] in inputs.bindings.physical
        ]
        # Module references from the packaged description remain valid native paths.
        n["references"] = [
            r for r in n.get("references", []) if r["kind"] in ("module", "tokenizer")
        ]
        if "attributes" in n:
            n["attributes"] = [{"name": a["name"], "value": a["value"]} for a in n["attributes"]]
        if n.get("parent_id"):
            n["parent_id"] = ids[n["parent_id"]]
        if "children" in n:
            n["children"] = [ids[k] for k in n["children"] if k not in excluded]
        if original["id"] == root["id"]:
            n["label"] = "Independent question causal rows"
            n["description"] = (
                "Each row contains the same state prefix and exactly one question branch. "
                "No question consumes another question's state. Serving may clone the "
                "state-prefix KV, convolution and recurrent caches per row; the packed "
                "block-causal mask is not used for this hybrid backbone."
            )
        if n["kind"] == "state":
            n["description"] = (
                "Per-question state only: zero on a complete state+question row; "
                "a serving branch receives an independent copy of its state-prefix cache."
            )
        d.nodes.append(n)
    # Keep the original native graph topology and explicit state edge kinds.
    for e in graph["edges"]:
        sn, tn = e["source"]["node_id"], e["target"]["node_id"]
        if sn in excluded or tn in excluded:
            continue
        d.edge(ids[sn], ids[tn], e["source"]["port_id"], e["target"]["port_id"], e["kind"])
    native_ports = {
        "symbolic_token_ids": "tokens",
        "symbolic_thw_positions": "positions",
        "symbolic_padding_mask": "mask",
        "current_sequence_padding_mask": "current_mask",
    }
    for n in graph["nodes"]:
        if n["kind"] == "input":
            name = native_ports[n["operation"]]
            dimensions = n["ports"][0]["shape"]
            inp = d.op("row." + name, name, {}, dimensions, kind="input")
            d.edge(inp, "backbone", tp=name)
    # Symbol B is the independent question-row axis. Prefix/branch composition
    # is an explicit operation outside tokenizer/typed-request serialization.
    token_node = next(n for n in d.nodes if n["id"] == "row.tokens")
    token_node.update(
        kind="operation",
        operation="independent_causal_rows",
        label="Shared state plus isolated questions",
        formula="out[row] = concat(state, question[row])",
        description=(
            "Options within a question remain sequential; "
            "no option permutation invariance is claimed."
        ),
    )
    token_node["ports"] = [
        port("state", shape("S_state")),
        port("question", shape("B", "S_question")),
        port("out", shape("B", "T"), True),
    ]
    for key, dims in (
        ("state", shape("S_state")),
        ("question", shape("B", "S_question")),
    ):
        d.op(
            key,
            "State prefix" if key == "state" else "Question branches",
            {},
            dims,
            kind="input",
        )
        d.edge(key, "row.tokens", tp=key)
    h, dp = c["hidden_size"], head["head_dim"]
    for key, dims in (("decide_idx", shape("B")), ("opt_idx", shape("B", "O"))):
        d.op(key, key, {}, dims, kind="input")
    for name, index, dims in (
        ("decide", "decide_idx", shape("B", h)),
        ("options", "opt_idx", shape("B", "O", h)),
    ):
        selected = d.op(
            "select." + name,
            "Decision hidden states" if name == "decide" else "Option boundary hidden states",
            {
                "x": shape("B", "T", h),
                "index": shape("B") if name == "decide" else shape("B", "O"),
            },
            dims,
            formula="out = float32(x[row, index[row]])",
        )
        d.edge("backbone", selected)
        d.edge(index, selected, tp="index")
    for name, dims, out in (
        ("q", shape("B", h), shape("B", dp)),
        ("k", shape("B", "O", h), shape("B", "O", dp)),
    ):
        d.op(
            "pointer." + name,
            "Query projection" if name == "q" else "Key projection",
            {"x": dims},
            out,
            parameters=("kev.head." + name + ".weight", "kev.head." + name + ".bias"),
            formula="out = x weightᵀ + bias",
            parent="pointer",
        )
        d.edge("pointer", "pointer." + name, sp="decide" if name == "q" else "options")
    d.op(
        "pointer.dot",
        "Option dot products",
        {"q": shape("B", dp), "k": shape("B", "O", dp)},
        shape("B", "O"),
        formula="out[row, option] = dot(k[row, option], q[row])",
        parent="pointer",
    )
    d.edge("pointer.q", "pointer.dot", tp="q")
    d.edge("pointer.k", "pointer.dot", tp="k")
    d.op(
        "pointer.scale",
        "Pointer scaling",
        {"x": shape("B", "O")},
        shape("B", "O"),
        formula="out = x / sqrt(head_dim)",
        attributes={"head_dim": dp},
        parent="pointer",
    )
    d.edge("pointer.dot", "pointer.scale")
    d.op(
        "pointer.temperature",
        "Inference calibration",
        {"x": shape("B", "O")},
        shape("B", "O"),
        formula="out = x / temperature",
        attributes={"temperature": head["temperature"]},
        parent="pointer",
    )
    d.edge("pointer.scale", "pointer.temperature")
    d.op(
        "pointer.softmax",
        "Per-question option probabilities",
        {"x": shape("B", "O")},
        shape("B", "O"),
        formula="out = softmax(x, axis)",
        attributes={"axis": -1},
        parent="pointer",
    )
    d.edge("pointer.temperature", "pointer.softmax")
    d.edge("pointer.softmax", "pointer", tp="out")
    d.nodes.append(
        {
            "id": "pointer",
            "kind": "group",
            "label": "Pointer decision head",
            "ports": [
                port("decide", shape("B", h)),
                port("options", shape("B", "O", h)),
                port("out", shape("B", "O"), True),
            ],
            "children": [n["id"] for n in d.nodes if n.get("parent_id") == "pointer"],
        }
    )
    d.edge("select.decide", "pointer", tp="decide")
    d.edge("select.options", "pointer", tp="options")
    d.op(
        "probabilities",
        "Option probabilities",
        {"x": shape("B", "O")},
        shape("B", "O"),
        kind="output",
    )
    d.edge("pointer", "probabilities")
    # Turn each adapted operation into a navigable group with exact additive math.
    for module, pair in factors.items():
        n = next(
            n
            for n in d.nodes
            if n["parameter_ids"] == [module + ".weight"] and n.get("operation") == "linear"
        )
        key = n["id"]
        inp = next(p["shape"] for p in n["ports"] if p["direction"] == "input")
        out = next(p["shape"] for p in n["ports"] if p["direction"] == "output")
        base = copy.deepcopy(n)
        base.update(
            id=key + ".base",
            parent_id=key,
            formula="out = x weightᵀ",
            label="Base projection",
        )
        n.update(
            kind="group",
            label=module.split(".")[-1] + " + LoRA",
            children=[
                key + ".base",
                key + ".A",
                key + ".B",
                key + ".scale",
                key + ".add",
            ],
            parameter_ids=[],
        )
        n.pop("operation", None)
        n.pop("formula", None)
        d.nodes.append(base)
        a, bb = ("kev.lora." + pair[f] for f in ("A", "B"))
        middle = inp[:-1] + shape(adapter["r"])
        d.op(
            key + ".A",
            "LoRA A",
            {"x": inp},
            middle,
            parameters=(a,),
            formula="out = x Aᵀ",
            parent=key,
        )
        d.op(
            key + ".B",
            "LoRA B",
            {"x": middle},
            out,
            parameters=(bb,),
            formula="out = x Bᵀ",
            parent=key,
        )
        d.op(
            key + ".scale",
            "LoRA scaling",
            {"x": out},
            out,
            formula="out = x * alpha / rank",
            attributes={"alpha": adapter["lora_alpha"], "rank": adapter["r"]},
            parent=key,
        )
        d.op(
            key + ".add",
            "Adapted projection",
            {"base": out, "lora": out},
            out,
            formula="out = base + lora",
            parent=key,
        )
        d.edge(key, key + ".base", sp="x")
        d.edge(key, key + ".A", sp="x")
        d.edge(key + ".A", key + ".B")
        d.edge(key + ".B", key + ".scale")
        d.edge(key + ".scale", key + ".add", tp="lora")
        d.edge(key + ".base", key + ".add", tp="base")
        d.edge(key + ".add", key, tp="out")
    inventory = {p.name: list(p.shape) for p in inputs.bindings.physical.values()}
    inventory.update({n: list(t.shape) for n, t in tensors.items()})
    return {
        "schema_version": 1,
        "architecture_revision": PRODUCER + "-" + SOURCE_REVISION,
        "name": "Kev-0.8B decision model",
        "scope": "option_decision",
        "symbols": [
            {**s, "meaning": "Independent question rows"} if s["name"] == "B" else s
            for s in graph["symbols"]
        ]
        + [
            {
                "name": "O",
                "meaning": (
                    "Options for this question; separate questions may have different counts"
                ),
            },
            {"name": "S_state", "meaning": "Shared state prefix length"},
            {"name": "S_question", "meaning": "Question branch length"},
        ],
        "nodes": d.nodes,
        "edges": d.edges,
        "parameters": [
            {"id": name, "name": name, "shape": shape(*dims)} for name, dims in inventory.items()
        ],
        "repetitions": [
            {
                **r,
                "parent_id": ids[r["parent_id"]],
                "instances": [{**i, "node_id": ids[i["node_id"]]} for i in r["instances"]],
            }
            for r in graph["repetitions"]
        ],
    }


def export_package(
    base: Path,
    kev: Path,
    destination: Path,
    *,
    base_revision: str,
    kev_revision: str,
    copy_shards: bool = False,
):
    for rev in (base_revision, kev_revision):
        if not re.fullmatch(r"[0-9a-f]{40}", rev):
            raise ValueError("Immutable 40-hex revisions required")
    base, kev = base.resolve(strict=True), kev.resolve(strict=True)
    destination = destination.absolute()
    if destination.exists() or destination.is_relative_to(base) or destination.is_relative_to(kev):
        raise ValueError("Destination must be new and outside inputs")
    config = read_json(base.parent, base / "config.json")
    if (
        config.get("model_type") != "qwen3_5"
        or config.get("architectures") != ["Qwen3_5ForConditionalGeneration"]
        or config.get("quantization_config") is not None
    ):
        raise ValueError("Complete native Qwen3.5 base required")
    binding = {
        k: config[k]
        for k in ("_name_or_path", "name_or_path", "_commit_hash", "revision")
        if config.get(k) is not None
    }
    if any(
        binding[k] != BASE_REPOSITORY for k in ("_name_or_path", "name_or_path") if k in binding
    ) or any(binding[k] != base_revision for k in ("_commit_hash", "revision") if k in binding):
        raise ValueError("Base configuration identity/revision mismatch")
    entry = ModelCatalogue(base.parent).inspect_directory(base)
    source = entry.pin()
    if not entry.summary.tokenizer_available:
        raise ValueError("Base tokenizer required")
    inputs = AnalysisInput.from_source(source, tokenizer_available=True)
    normalized = copy.deepcopy(config)
    for key in ("_commit_hash", "revision", "name_or_path"):
        normalized.pop(key, None)
    normalized["text_config"].setdefault(
        "partial_rotary_factor",
        normalized["text_config"]["rope_parameters"]["partial_rotary_factor"],
    )
    c = configuration(replace(inputs, configuration=normalized))
    if c is None:
        raise ValueError("Unreviewed Qwen3.5 configuration")
    expected = parameter_shapes(c)
    physical = {p.name: p for p in source.physical_tensors()}
    language = {n for n in physical if n.startswith(PREFIX + ".")}
    if language != set(expected) or any(
        list(physical[n].shape) != dims or physical[n].dtype not in ("F32", "F16", "BF16")
        for n, dims in expected.items()
    ):
        raise ValueError("Incomplete or incompatible native text backbone")
    checkpoint_snapshot = FileSnapshot.capture(
        kev.parent, kev, ("head.pt", "adapter_model.safetensors")
    )
    head_path = kev / "head.pt"
    before_head = digest(head_path)
    head, tensors = checked_head(head_path, c["hidden_size"], base_revision)
    adapter, factors, adapter_snapshot = checked_adapter(kev, expected, base_revision)
    if head["lora"] != adapter["r"]:
        raise ValueError("Head/adapter rank mismatch")
    for name in ("training_config.json", "provenance.json"):
        if (kev / name).is_file():
            meta = read_json(kev.parent, kev / name)
            selected = meta.get("config", meta.get("args", meta))
            if (
                selected.get("base", BASE_REPOSITORY) != BASE_REPOSITORY
                or selected.get("base_revision", base_revision) != base_revision
            ):
                raise ValueError("Checkpoint metadata base mismatch")
    if (kev / "adapter_model.safetensors").stat().st_size > 256 * 1024 * 1024:
        raise ValueError("Adapter exceeds the 256 MiB conversion budget")
    with safe_open(kev / "adapter_model.safetensors", framework="pt", device="cpu") as f:
        for original in f.keys():
            t = f.get_tensor(original)
            if not bool(torch.isfinite(t).all()):
                raise ValueError("Nonfinite adapter factor")
            tensors["kev.lora." + original] = t.contiguous()
    definition = architecture(inputs, config, factors, adapter, head, tensors)
    raw = json.dumps(definition, separators=(",", ":"), ensure_ascii=True) + "\n"
    if len(raw.encode()) > MAX_DEFINITION_BYTES:
        raise ValueError("Sidecar exceeds existing budget")
    weights = {p.name: "base/" + p.file for p in source.physical_tensors()}
    if set(tensors) & set(weights):
        raise ValueError("Tensor namespace collision")
    shards = sorted({p.file for p in source.physical_tensors()})
    destination.parent.mkdir(parents=True, exist_ok=True)
    required = 128 * 1024 * 1024 + sum(t.numel() * t.element_size() for t in tensors.values())
    if copy_shards:
        required += sum((base / n).stat().st_size for n in shards)
    if shutil.disk_usage(destination.parent).free < required:
        raise ValueError("Insufficient disk")
    if not copy_shards and not base.is_relative_to(destination.parent.resolve()):
        raise ValueError("Shared shards must remain within model root; use --copy-shards")
    staging = Path(tempfile.mkdtemp(prefix=".kev-export-", dir=destination.parent.parent))
    try:
        for name in shards:
            target = staging / "base" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            if copy_shards:
                shutil.copyfile(base / name, target)
            else:
                target.symlink_to(base / name)
        assets = {
            "tokenizer.json",
            "tokenizer_config.json",
            "vocab.json",
            "merges.txt",
            "special_tokens_map.json",
            "added_tokens.json",
            "LICENSE",
        }
        for name in assets:
            if (base / name).is_file():
                shutil.copyfile(base / name, staging / name)
        save_file(tensors, str(staging / "kev.safetensors"))
        weights.update({n: "kev.safetensors" for n in tensors})
        (staging / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": weights}, sort_keys=True) + "\n"
        )
        config["_name_or_path"] = EXPORT_IDENTITY
        config["_commit_hash"] = kev_revision
        config["kev_inspection"] = {
            "base_repository": BASE_REPOSITORY,
            "base_revision": base_revision,
            "kev_revision": kev_revision,
            "head": head,
            "required_definition": "architecture.json",
        }
        (staging / "config.json").write_text(json.dumps(config, sort_keys=True) + "\n")
        (staging / "architecture.json").write_text(raw)
        provenance = {
            "producer": PRODUCER,
            "producer_sha256": digest(Path(__file__)),
            "source": {
                "repository": "jaredpalmer/kev",
                "revision": SOURCE_REVISION,
                "license": "Apache-2.0",
                "files": SOURCE_FILES,
            },
            "qwen_description": QWEN_PRODUCER.__dict__,
            "base": {
                "repository": BASE_REPOSITORY,
                "revision": base_revision,
                "configuration_binding": binding,
                "revision_origin": "configuration"
                if any(k in binding for k in ("_commit_hash", "revision"))
                else "operator_selection_checked_against_head",
                "content_fingerprint": source.fingerprint,
                "license": "Apache-2.0",
                "files": [
                    {
                        "name": n,
                        "sha256": digest(base / n),
                        "bytes": (base / n).stat().st_size,
                    }
                    for n in sorted(
                        set(shards) | assets | {"config.json", "model.safetensors.index.json"}
                    )
                    if (base / n).is_file()
                ],
            },
            "kev": {
                "repository": KEV_REPOSITORY,
                "revision": kev_revision,
                "license": "Apache-2.0",
                "adapter_configuration": adapter,
                "head_metadata": head,
                "files": [
                    {
                        "name": n,
                        "sha256": digest(kev / n),
                        "bytes": (kev / n).stat().st_size,
                    }
                    for n in (
                        "head.pt",
                        "adapter_model.safetensors",
                        "adapter_config.json",
                        "training_config.json",
                        "provenance.json",
                        "tokenizer.json",
                        "tokenizer_config.json",
                        "README.md",
                    )
                    if (kev / n).is_file()
                ],
            },
            "namespace": {
                "base": "unchanged",
                "factors": "kev.lora.<exact-original-key>",
                "head": "kev.head.<exact-original-key>",
            },
            "targets": [
                {
                    "base_weight": module + ".weight",
                    "A": "kev.lora." + pair["A"],
                    "B": "kev.lora." + pair["B"],
                    "scale": adapter["lora_alpha"] / adapter["r"],
                    "orientation": "W[out,in], A[rank,in], B[out,rank]",
                }
                for module, pair in sorted(factors.items())
            ],
            "definition_sha256": digest(staging / "architecture.json"),
        }
        (staging / "kev-provenance.json").write_text(json.dumps(provenance, sort_keys=True) + "\n")
        source.check_unchanged()
        adapter_snapshot.check()
        checkpoint_snapshot.check()
        if digest(head_path) != before_head:
            raise ValueError("Head changed during conversion")
        result = validate_directory(staging)
        if result["status"] != "valid":
            raise ValueError("Export failed static validation: " + str(result))
        if destination.exists():
            raise ValueError("Destination appeared during export")
        staging.rename(destination)
        return result
    finally:
        if staging.exists():
            shutil.rmtree(staging)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--kev", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-revision", required=True)
    parser.add_argument("--kev-revision", required=True)
    parser.add_argument("--copy-shards", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(
            export_package(
                args.base,
                args.kev,
                args.output,
                base_revision=args.base_revision,
                kev_revision=args.kev_revision,
                copy_shards=args.copy_shards,
            ),
            sort_keys=True,
        )
    )

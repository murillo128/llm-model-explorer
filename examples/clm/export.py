"""Offline CLM inspection-package exporter; never imported by the server.

Correspondence: Contrastive-LM/CLM at SOURCE_REVISION, heads.py and embedder.py.
Only the small head is materialized. Encoder shards remain native and shared.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import shutil
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from llm_model_explorer.architecture_analysis import (
    AnalysisInput,
    DescriptionRegistry,
    register_dense_descriptions,
)
from llm_model_explorer.architecture_analysis.dense_config import (
    SOURCE_REVISION as QWEN_SOURCE_REVISION,
)
from llm_model_explorer.architecture_analysis.model_defined_schema import MAX_DEFINITION_BYTES
from llm_model_explorer.models import ModelCatalogue
from llm_model_explorer.validate_architecture_cli import validate_directory
from safetensors.torch import save_file

SOURCE_REVISION = "bb42c6c5bf914fd449bed2f6ca65be80602cb1f7"
ENCODER_REVISION = "b968826d9c46dd6066d109eabc6255188de91218"
HEAD_REVISION = "e939398d4556fcd9400c76fa8c5a513202f42b0a"
ENCODER_REPOSITORY = "Qwen/Qwen3-8B"
HEAD_REPOSITORY = "Contrastive-LM/CLM-v0.1-8B"
PRODUCER = "clm-inspection-export-1"
NATIVE = {torch.float32, torch.float16, torch.bfloat16}
SOURCE_FILES = {
    "src/clm/heads.py": "3f3b880e940a47b45879614b140fd873f7de9b13ccb8254b07989af7ea92e093",
    "src/clm/embedder.py": "d0ca78073aa1c59676e17e35d9c8cda8131da03010841887f833fbfe8fc69dbc",
    "src/clm/engine.py": "838897b99b1b09ce141f35272450e1c59fd290f092d1362570e07a453f06f5e7",
    "LICENSE": "cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30",
}


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()


def checked_head(path: Path, hidden: int) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    """Restricted CPU admission of tensors and primitive configuration only."""
    if not path.is_file() or path.stat().st_size > 512 * 1024 * 1024:
        raise ValueError("Head must be a regular checkpoint of at most 512 MiB")
    # Explicit weights_only, no safe-global extensions, no checkpoint Python.
    payload = torch.load(path, weights_only=True, map_location="cpu", mmap=True)
    fields = {"state_head", "action_head", "logit_scale", "cfg", "hidden_size", "projection_dim"}
    if (
        not isinstance(payload, dict)
        or set(payload) - fields
        or not fields.difference({"hidden_size", "projection_dim"}) <= payload.keys()
    ):
        raise ValueError("Unsupported head payload")
    cfg = payload["cfg"]
    allowed = {
        "model",
        "hidden_size",
        "projection_dim",
        "width",
        "depth",
        "activation",
        "layernorm",
        "residual",
    }
    if not isinstance(cfg, dict) or set(cfg) - allowed:
        raise ValueError("Unsupported head configuration")
    cfg = dict(cfg)
    for key, default in (("hidden_size", 4096), ("projection_dim", 512)):
        cfg.setdefault(key, default)
        if key in payload and payload[key] != cfg[key]:
            raise ValueError("Conflicting head dimensions")
    cfg.setdefault("activation", "gelu")
    cfg.setdefault("layernorm", False)
    cfg.setdefault("residual", False)
    cfg.setdefault("model", ENCODER_REPOSITORY)
    if cfg["model"] != ENCODER_REPOSITORY or cfg["hidden_size"] != hidden:
        raise ValueError("Head requires the selected Qwen3-8B encoder width")
    for key in ("hidden_size", "width", "projection_dim", "depth"):
        if type(cfg.get(key)) is not int or not 0 < cfg[key] <= 16384:
            raise ValueError("Invalid head dimensions")
    if not 2 <= cfg["depth"] <= 16 or cfg["activation"] not in {"gelu", "relu", "silu"}:
        raise ValueError("Unsupported head depth or activation")
    if any(type(cfg[k]) is not bool for k in ("layernorm", "residual")):
        raise ValueError("Invalid head options")
    width, proj = cfg["width"], cfg["projection_dim"]
    expected = {
        "inp.weight": (width, hidden),
        "inp.bias": (width,),
        "out.weight": (proj, width),
        "out.bias": (proj,),
    }
    for i in range(cfg["depth"] - 2):
        expected.update({f"hidden.{i}.weight": (width, width), f"hidden.{i}.bias": (width,)})
        if cfg["layernorm"]:
            expected.update({f"norms.{i}.weight": (width,), f"norms.{i}.bias": (width,)})
    tensors = {}
    logical_bytes = 0
    for head in ("state_head", "action_head"):
        state = payload[head]
        if not isinstance(state, Mapping) or set(state) != set(expected):
            raise ValueError("Missing, extra or malformed head tensors")
        for name, dims in expected.items():
            tensor = state[name]
            if (
                type(tensor) is not torch.Tensor
                or tensor.dtype not in NATIVE
                or tuple(tensor.shape) != dims
            ):
                raise ValueError("Head tensor shape or dtype mismatch")
            logical_bytes += tensor.numel() * tensor.element_size()
            if logical_bytes > 512 * 1024 * 1024 or tensor.layout != torch.strided:
                raise ValueError("Head tensors exceed the native conversion budget")
            if not bool(torch.isfinite(tensor).all()):
                raise ValueError("Nonfinite head tensor")
            tensors[f"clm.{head}.{name}"] = tensor.contiguous()
    scale = payload["logit_scale"]
    if type(scale) in {int, float}:
        scale = torch.tensor(scale, dtype=torch.float32)
    if (
        type(scale) is not torch.Tensor
        or scale.shape != torch.Size([])
        or scale.dtype not in NATIVE
        or not bool(torch.isfinite(scale))
    ):
        raise ValueError("Invalid learned logit scale")
    tensors["clm.logit_scale"] = scale.contiguous()
    return cfg, tensors


def shape(*dims: int | str) -> list[dict[str, Any]]:
    return [
        {"kind": "constant", "value": d} if type(d) is int else {"kind": "symbol", "name": d}
        for d in dims
    ]


def port(name: str, dims: list[dict[str, Any]], output: bool = False) -> dict[str, Any]:
    return {"id": name, "label": name, "direction": "output" if output else "input", "shape": dims}


class Definition:
    def __init__(self) -> None:
        self.nodes: list[dict[str, Any]] = []
        self.edges: list[dict[str, Any]] = []
        self.parameters: dict[str, dict[str, Any]] = {}
        self.repetitions: list[dict[str, Any]] = []

    def edge(self, source: str, target: str, sp: str = "out", tp: str = "x") -> None:
        self.edges.append(
            {
                "id": f"edge.{len(self.edges)}",
                "source": {"node_id": source, "port_id": sp},
                "target": {"node_id": target, "port_id": tp},
                "kind": "data",
            }
        )

    def node(
        self,
        key: str,
        label: str,
        inputs: dict[str, list[dict[str, Any]]],
        output: list[dict[str, Any]],
        *,
        operation: str | None = None,
        parent: str | None = None,
        parameters: tuple[str, ...] = (),
        formula: str | None = None,
        attributes: dict[str, Any] | None = None,
        kind: str = "operation",
    ) -> str:
        n: dict[str, Any] = {
            "id": key,
            "kind": kind,
            "label": label,
            "ports": [port(k, v) for k, v in inputs.items()] + [port("out", output, True)],
            "parameter_ids": list(parameters),
        }
        if operation:
            n["operation"] = operation
        if parent:
            n["parent_id"] = parent
        if formula:
            n["formula"] = formula
        if attributes:
            n["attributes"] = [{"name": k, "value": v} for k, v in attributes.items()]
        self.nodes.append(n)
        return key

    def group(
        self,
        key: str,
        label: str,
        inputs: dict[str, list[dict[str, Any]]],
        output: list[dict[str, Any]],
    ) -> None:
        self.nodes.append(
            {
                "id": key,
                "kind": "group",
                "label": label,
                "ports": [port(k, v) for k, v in inputs.items()] + [port("out", output, True)],
                "children": [n["id"] for n in self.nodes if n.get("parent_id") == key],
            }
        )

    def head(self, head: str, batch: str, cfg: dict[str, Any]) -> str:
        key = f"clm.{head}"
        hidden, width, proj = cfg["hidden_size"], cfg["width"], cfg["projection_dim"]

        def op(
            suffix: str,
            name: str,
            operation: str,
            inputs: dict[str, list[dict[str, Any]]],
            dims: list[dict[str, Any]],
            *,
            binding: str | None = None,
            formula: str | None = None,
            attributes: dict[str, Any] | None = None,
        ) -> str:
            params = (
                ()
                if binding is None
                else (f"clm.{head}.{binding}.weight", f"clm.{head}.{binding}.bias")
            )
            return self.node(
                key + "." + suffix,
                name,
                inputs,
                dims,
                operation=operation,
                parent=key,
                parameters=params,
                formula=formula,
                attributes=attributes,
            )

        current = op(
            "inp",
            "Input projection",
            "linear",
            {"x": shape(batch, hidden)},
            shape(batch, width),
            binding="inp",
            formula="out = x Wᵀ + b",
        )
        self.edge(key, current, "x")
        activated = op(
            "inp_activation",
            cfg["activation"].upper(),
            cfg["activation"],
            {"x": shape(batch, width)},
            shape(batch, width),
            formula=f"out = {cfg['activation']}(x)",
            attributes={"approximate": "none"} if cfg["activation"] == "gelu" else None,
        )
        self.edge(current, activated)
        current = activated
        for i in range(cfg["depth"] - 2):
            skip = current
            nxt = op(
                f"hidden.{i}",
                "Hidden projection",
                "linear",
                {"x": shape(batch, width)},
                shape(batch, width),
                binding=f"hidden.{i}",
                formula="out = x Wᵀ + b",
            )
            self.edge(current, nxt)
            current = nxt
            if cfg["layernorm"]:
                nxt = op(
                    f"norms.{i}",
                    "LayerNorm",
                    "layer_norm",
                    {"x": shape(batch, width)},
                    shape(batch, width),
                    binding=f"norms.{i}",
                    formula="out = (x − mean(x)) / sqrt(var(x) + epsilon) * W + b",
                    attributes={"epsilon": 1e-5, "axis": -1, "variance": "population"},
                )
                self.edge(current, nxt)
                current = nxt
            nxt = op(
                f"activation.{i}",
                cfg["activation"].upper(),
                cfg["activation"],
                {"x": shape(batch, width)},
                shape(batch, width),
                formula=f"out = {cfg['activation']}(x)",
            )
            self.edge(current, nxt)
            current = nxt
            if cfg["residual"]:
                nxt = op(
                    f"residual.{i}",
                    "Residual addition",
                    "add",
                    {"skip": shape(batch, width), "branch": shape(batch, width)},
                    shape(batch, width),
                    formula="out = skip + branch",
                )
                self.edge(skip, nxt, tp="skip")
                self.edge(current, nxt, tp="branch")
                current = nxt
        nxt = op(
            "out",
            "Output projection",
            "linear",
            {"x": shape(batch, width)},
            shape(batch, proj),
            binding="out",
            formula="out = x Wᵀ + b",
        )
        self.edge(current, nxt)
        self.edge(nxt, key, tp="out")
        self.group(
            key,
            "State head" if head == "state_head" else "Action head",
            {"x": shape(batch, hidden)},
            shape(batch, proj),
        )
        return key


def architecture(
    inputs: AnalysisInput, cfg: dict[str, Any], tensors: dict[str, torch.Tensor]
) -> dict[str, Any]:
    """Reuse the reviewed Qwen mathematical graph, without its generation head."""
    registry = DescriptionRegistry()
    register_dense_descriptions(registry)
    result = registry.analyze(inputs)
    if result.status != "complete" or result.graph is None or result.graph.diagnostics:
        raise ValueError("Encoder is outside the complete native Qwen3 description")
    graph = result.graph.document()
    by_id = {n["id"]: n for n in graph["nodes"]}
    root = next(n for n in graph["nodes"] if n["kind"] == "group" and "parent_id" not in n)
    excluded = {
        n["id"]
        for n in graph["nodes"]
        if n["kind"] in {"input", "output", "context"}
        or any(
            p["name"] == "lm_head.weight"
            for p in graph["parameters"]
            if p["id"] in n["parameter_ids"]
        )
    }
    parameter_ids = {p["id"]: p["name"] for p in graph["parameters"]}
    d = Definition()
    for p in graph["parameters"]:
        if p["name"] != "lm_head.weight":
            if p["binding"] != "native":
                raise ValueError("CLM requires native, directly bound encoder weights")
            d.parameters[p["name"]] = {
                "id": p["name"],
                "name": p["name"],
                "shape": p["logical_shape"],
            }
    for name, tensor in tensors.items():
        d.parameters[name] = {"id": name, "name": name, "shape": shape(*tuple(tensor.shape))}
    for call, batch, seq in (
        ("state_encoder", "B", "S_state"),
        ("candidate_encoder", "C", "S_candidate"),
    ):
        ids = {
            n["id"]: call if n["id"] == root["id"] else f"{call}.{n['id']}" for n in graph["nodes"]
        }

        def dims(value: Any, batch: str = batch, seq: str = seq) -> Any:
            if isinstance(value, list):
                return [dims(v) for v in value]
            if isinstance(value, dict):
                value = {k: dims(v) for k, v in value.items()}
                if value.get("kind") == "symbol":
                    value["name"] = {"B": batch, "S": seq}.get(value["name"], value["name"])
                if value.get("kind") == "expression":
                    value["symbols"] = [{"B": batch, "S": seq}.get(s, s) for s in value["symbols"]]
                    value["text"] = re.sub(
                        r"\b[BS]\b", lambda m: {"B": batch, "S": seq}[m[0]], value["text"]
                    )
                return value
            return value

        for original in graph["nodes"]:
            if original["id"] in excluded:
                continue
            n = dims(
                {
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
                    }
                }
            )
            if "attributes" in n:
                n["attributes"] = [
                    {"name": a["name"], "value": a["value"]} for a in n["attributes"]
                ]
            n["id"] = ids[original["id"]]
            n["parameter_ids"] = [parameter_ids[p] for p in original["parameter_ids"]]
            if "parent_id" in n:
                n["parent_id"] = ids[n["parent_id"]]
            if "children" in n:
                n["children"] = [ids[c] for c in n["children"] if c not in excluded]
            if original["id"] == root["id"]:
                n["label"] = "State encoder" if batch == "B" else "Candidate encoder"
                n["ports"] = [
                    port("input_ids", shape(batch, seq)),
                    port("position_ids", shape(batch, seq)),
                    port("attention_mask", shape(batch, 1, seq, seq)),
                    port("out", shape(batch, seq, cfg["hidden_size"]), True),
                ]
            d.nodes.append(n)
        for e in graph["edges"]:
            sn, tn = e["source"]["node_id"], e["target"]["node_id"]
            if tn in excluded or (sn in excluded and by_id[sn]["kind"] != "input"):
                continue
            sp = e["source"]["port_id"]
            if sn in excluded:
                input_node = by_id[sn]
                sp = {
                    "token_ids": "input_ids",
                    "positions": "position_ids",
                    "causal_mask": "attention_mask",
                }[input_node["operation"]]
                sn = root["id"]
            d.edge(ids[sn], ids[tn], sp, e["target"]["port_id"])
        norm = next(
            n
            for n in graph["nodes"]
            if "model.norm.weight" in [parameter_ids[p] for p in n["parameter_ids"]]
        )
        d.edge(ids[norm["id"]], call, tp="out")
        for repetition in graph["repetitions"]:
            d.repetitions.append(
                {
                    "id": call + ".layers",
                    "parent_id": call,
                    "label": "Decoder layers",
                    "instances": [
                        {**i, "node_id": ids[i["node_id"]]} for i in repetition["instances"]
                    ],
                }
            )
        # Model interfaces are passive; tokenization remains an external capability.
        for name, dimensions in (
            ("input_ids", shape(batch, seq)),
            ("position_ids", shape(batch, seq)),
            ("attention_mask", shape(batch, 1, seq, seq)),
        ):
            key = d.node(call + "." + name, name, {}, dimensions, kind="input")
            d.edge(key, call, tp=name)
        lengths = d.node(call + ".lengths", "Sequence lengths", {}, shape(batch), kind="input")
        pooled = d.node(
            call + ".pool",
            "Last-token pooling",
            {"x": shape(batch, seq, cfg["hidden_size"]), "lengths": shape(batch)},
            shape(batch, cfg["hidden_size"]),
            operation="last_token_pool",
            formula="out = x[row, lengths[row] − 1]",
            attributes={"padding": "right; lengths exclude padding"},
        )
        d.edge(call, pooled)
        d.edge(lengths, pooled, tp="lengths")
        normalized = d.node(
            call + ".normalize",
            "Encoder L2 normalization",
            {"x": shape(batch, cfg["hidden_size"])},
            shape(batch, cfg["hidden_size"]),
            operation="encoder_normalize",
            formula="out = x / (norm₂(x) + epsilon)",
            attributes={"epsilon": 1e-12, "axis": -1},
        )
        d.edge(pooled, normalized)
        head = d.head("state_head" if batch == "B" else "action_head", batch, cfg)
        d.edge(normalized, head)
        z = d.node(
            call + ".projection_normalize",
            "Projection L2 normalization",
            {"x": shape(batch, cfg["projection_dim"])},
            shape(batch, cfg["projection_dim"]),
            operation="l2_normalize",
            formula="out = x / max(norm₂(x), epsilon)",
            attributes={"epsilon": 1e-12, "axis": -1},
        )
        d.edge(head, z)
    cosine = d.node(
        "cosine",
        "Candidate cosine similarities",
        {
            "state": shape("B", cfg["projection_dim"]),
            "candidates": shape("C", cfg["projection_dim"]),
        },
        shape("B", "C"),
        operation="dot_product",
        formula="out = state candidatesᵀ",
    )
    d.edge("state_encoder.projection_normalize", cosine, tp="state")
    d.edge("candidate_encoder.projection_normalize", cosine, tp="candidates")
    scale = d.node(
        "scale",
        "Learned similarity scale",
        {},
        [],
        operation="exp_clamp",
        parameters=("clm.logit_scale",),
        formula="out = min(exp(float32(logit_scale)), maximum)",
        attributes={"maximum": 100.0},
    )
    temperature = d.node("temperature", "Temperature", {}, [], kind="input")
    scores = d.node(
        "scores",
        "Candidate scores",
        {"cosine": shape("B", "C"), "scale": [], "temperature": []},
        shape("B", "C"),
        operation="scale",
        formula="out = cosine * scale / temperature",
        attributes={
            "temperature_default": 1.0,
            "temperature_range": "0 < temperature ≤ 100",
            "broadcast": "scalar scale and temperature across state and candidate axes",
        },
    )
    d.edge(cosine, scores, tp="cosine")
    d.edge(scale, scores, tp="scale")
    d.edge(temperature, scores, tp="temperature")
    probabilities = d.node(
        "probabilities",
        "Candidate probabilities",
        {"x": shape("B", "C")},
        shape("B", "C"),
        operation="softmax",
        formula="out = softmax(x, axis)",
        attributes={"axis": -1, "axis_meaning": "candidates for each state"},
    )
    d.edge(scores, probabilities)
    for key, source in (("score_output", scores), ("probability_output", probabilities)):
        out = d.node(
            key,
            "Candidate scores" if key == "score_output" else "Candidate probabilities",
            {"x": shape("B", "C")},
            shape("B", "C"),
            kind="output",
        )
        d.edge(source, out)
    return {
        "schema_version": 1,
        "architecture_revision": PRODUCER + "-" + SOURCE_REVISION,
        "name": "CLM-v0.1-8B",
        "scope": "candidate_decision",
        "symbols": [
            {"name": k, "meaning": v}
            for k, v in {
                "B": "Independent states",
                "C": "Candidates scored independently against each state",
                "S_state": "State sequence length",
                "S_candidate": "Candidate sequence length",
            }.items()
        ],
        "nodes": d.nodes,
        "edges": d.edges,
        "parameters": list(d.parameters.values()),
        "repetitions": d.repetitions,
    }


def export_package(
    encoder: Path,
    head: Path,
    destination: Path,
    *,
    encoder_revision: str,
    head_revision: str,
    copy_shards: bool = False,
) -> dict[str, Any]:
    for revision in (encoder_revision, head_revision):
        if not re.fullmatch(r"[0-9a-f]{40}", revision):
            raise ValueError("Inputs require immutable 40-hex revisions")
    encoder, head = encoder.resolve(strict=True), head.resolve(strict=True)
    destination = destination.absolute()
    if destination.exists() or destination.is_relative_to(encoder):
        raise ValueError("Destination must be new and outside the encoder")
    config = json.loads((encoder / "config.json").read_text())
    # Bind any declared source identity before replacing it with the CLM identity.
    # Bare upstream configs can omit these fields; their provenance must identify
    # the operator's verified reference selection rather than claim a metadata match.
    identity_fields = ("_name_or_path", "name_or_path")
    revision_fields = ("_commit_hash", "revision")
    metadata = {
        key: config[key]
        for key in (*identity_fields, *revision_fields)
        if config.get(key) is not None
    }
    if any(metadata[key] != ENCODER_REPOSITORY for key in identity_fields if key in metadata):
        raise ValueError("Encoder repository metadata does not match Qwen/Qwen3-8B")
    if any(metadata[key] != encoder_revision for key in revision_fields if key in metadata):
        raise ValueError("Encoder revision metadata does not match the selected revision")
    binding = {
        "configuration": metadata,
        "repository_source": "configuration"
        if any(key in metadata for key in identity_fields)
        else "operator_selection",
        "revision_source": "configuration"
        if any(key in metadata for key in revision_fields)
        else "operator_selection",
    }
    if (
        config.get("model_type") != "qwen3"
        or config.get("architectures") != ["Qwen3ForCausalLM"]
        or config.get("quantization_config") is not None
    ):
        raise ValueError("A complete native Qwen3 encoder is required")
    before_head = digest(head)
    cfg, tensors = checked_head(head, config.get("hidden_size"))
    entry = ModelCatalogue(encoder.parent).inspect_directory(encoder)
    if not entry.summary.tokenizer_available:
        raise ValueError("Encoder tokenizer is required")
    source = entry.pin()
    inputs = AnalysisInput.from_source(source, tokenizer_available=True)
    definition = architecture(inputs, cfg, tensors)
    raw_definition = json.dumps(definition, separators=(",", ":"), ensure_ascii=True)
    if len(raw_definition.encode()) > MAX_DEFINITION_BYTES:
        raise ValueError("CLM definition exceeds the existing sidecar budget")
    weight_map = {p.name: p.file for p in source.physical_tensors()}
    if set(tensors) & set(weight_map):
        raise ValueError("Head namespace collides with encoder tensors")
    shards = sorted(set(weight_map.values()))
    assets = [
        p.name
        for p in encoder.iterdir()
        if p.is_file()
        and p.name
        in {
            "config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "vocab.json",
            "merges.txt",
            "special_tokens_map.json",
            "added_tokens.json",
            "LICENSE",
            "model.safetensors.index.json",
        }
    ]
    required_bytes = head.stat().st_size * 2 + 32 * 1024 * 1024
    if copy_shards:
        required_bytes += sum((encoder / f).stat().st_size for f in shards)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(destination.parent).free < required_bytes:
        raise ValueError("Insufficient free disk for export")
    if not copy_shards and not encoder.is_relative_to(destination.parent.resolve()):
        raise ValueError(
            "Shared shards must remain inside the destination model root; use --copy-shards"
        )
    staging = Path(tempfile.mkdtemp(prefix=".clm-export-", dir=destination.parent.parent))
    try:
        for name in shards:
            target = staging / name
            target.parent.mkdir(parents=True, exist_ok=True)
            if copy_shards:
                shutil.copyfile(encoder / name, target)
            else:
                target.symlink_to(encoder / name)
        for name in assets:
            if name not in {"config.json", "model.safetensors.index.json"}:
                shutil.copyfile(encoder / name, staging / name)
        head_license = head.parent / "LICENSE"
        if head_license.is_file():
            shutil.copyfile(head_license, staging / "CLM-LICENSE.txt")
        if (staging / "clm-heads.safetensors").exists():
            raise ValueError("Head shard name collision")
        save_file(tensors, str(staging / "clm-heads.safetensors"))
        weight_map.update({name: "clm-heads.safetensors" for name in tensors})
        (staging / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": weight_map}, sort_keys=True) + "\n"
        )
        config["_name_or_path"] = HEAD_REPOSITORY
        config["_commit_hash"] = head_revision
        # Preserve the Qwen text layout for exact input-table lookup, while
        # making packaged dense selection fail closed if the sidecar is lost.
        config["clm_inspection"] = {
            "encoder_repository": ENCODER_REPOSITORY,
            "encoder_revision": encoder_revision,
            "head_revision": head_revision,
            "head_configuration": cfg,
            "pooling": "last_token",
            "required_definition": "architecture.json",
        }
        (staging / "config.json").write_text(json.dumps(config, sort_keys=True) + "\n")
        (staging / "architecture.json").write_text(raw_definition + "\n")
        provenance = {
            "producer": PRODUCER,
            "producer_sha256": digest(Path(__file__)),
            "qwen_description_source_revision": QWEN_SOURCE_REVISION,
            "source": {
                "repository": "Contrastive-LM/CLM",
                "revision": SOURCE_REVISION,
                "license": "Apache-2.0",
                "files": [{"name": name, "sha256": sha} for name, sha in SOURCE_FILES.items()],
            },
            "encoder": {
                "repository": ENCODER_REPOSITORY,
                "revision": encoder_revision,
                "binding": binding,
                "content_fingerprint": source.fingerprint,
                "license": "Apache-2.0",
                "files": [
                    {
                        "name": name,
                        "bytes": (encoder / name).stat().st_size,
                        "sha256": digest(encoder / name),
                    }
                    for name in sorted(set(assets + shards))
                ],
            },
            "head": {
                "repository": HEAD_REPOSITORY,
                "filename": head.name,
                "revision": head_revision,
                "license": "Apache-2.0",
                "sha256": before_head,
                "bytes": head.stat().st_size,
                "configuration": cfg,
                "files": [
                    {"name": name, "sha256": digest(head.parent / name)}
                    for name in ("config.json", "LICENSE", "README.md")
                    if (head.parent / name).is_file()
                ],
            },
            "namespace": {
                "encoder": "unchanged",
                "state_head": "clm.state_head.<original-name>",
                "action_head": "clm.action_head.<original-name>",
                "logit_scale": "clm.logit_scale",
            },
            "inventory": [
                {"name": p.name, "shape": list(p.shape), "dtype": p.dtype}
                for p in source.physical_tensors()
            ]
            + [
                {"name": name, "shape": list(t.shape), "dtype": str(t.dtype)}
                for name, t in sorted(tensors.items())
            ],
        }
        (staging / "clm-provenance.json").write_text(json.dumps(provenance, sort_keys=True) + "\n")
        source.check_unchanged()
        if digest(head) != before_head:
            raise ValueError("Head changed during export")
        validation = validate_directory(staging)
        if validation["status"] != "valid":
            raise ValueError(f"Exported package failed static import: {validation}")
        if destination.exists():
            raise ValueError("Destination appeared during export")
        staging.rename(destination)
        return validation
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--encoder", type=Path, required=True)
    parser.add_argument("--head", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--encoder-revision", required=True)
    parser.add_argument("--head-revision", required=True)
    parser.add_argument("--copy-shards", action="store_true")
    args = parser.parse_args()
    result = export_package(
        args.encoder,
        args.head,
        args.output,
        encoder_revision=args.encoder_revision,
        head_revision=args.head_revision,
        copy_shards=args.copy_shards,
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()

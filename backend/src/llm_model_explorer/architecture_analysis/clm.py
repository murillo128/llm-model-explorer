"""Reviewed packaged CLM decision graph, migrated from the offline exporter.

Contrastive-LM/CLM heads.py, embedder.py and engine.py at SOURCE_REVISION.
Reuses the dense Qwen3 description; only guarded metadata reaches this module.
"""

from __future__ import annotations

import copy
import re
from typing import Any

from .clm_config import SOURCE_REVISION, configuration, encoder_inputs
from .core import AnalysisInput, Description, DescriptionRegistry, GraphBuilder, Producer
from .dense import register_dense_descriptions
from .packaged import publish_definition
from .validation import require

PRODUCER = Producer("clm-inspection", "1", SOURCE_REVISION)


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
            "references": [{"kind": "module", "name": parameters[0].rsplit(".", 1)[0]}]
            if parameters
            else [],
        }
        if operation:
            n["operation"] = operation
        if parent:
            n["parent_id"] = parent
        if formula:
            n["formula"] = formula
        if operation == "gelu":
            attributes = {"approximate": "none", **(attributes or {})}
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
            formula="out = x weightᵀ + bias",
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
                formula="out = x weightᵀ + bias",
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
                    formula="out = (x − mean(x)) / sqrt(var(x) + epsilon) * weight + bias",
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
            formula="out = x weightᵀ + bias",
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


def architecture(inputs: AnalysisInput, cfg: dict[str, Any]) -> dict[str, Any]:
    """Reuse the reviewed Qwen mathematical graph, without its generation head."""
    registry = DescriptionRegistry()
    register_dense_descriptions(registry)
    result = registry.analyze(encoder_inputs(inputs))
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
                "provenance": p["provenance"],
            }
    for name, storage in inputs.bindings.physical.items():
        if name.startswith("clm."):
            d.parameters[name] = {"id": name, "name": name, "shape": shape(*storage.shape)}
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
                        "provenance",
                        "references",
                    }
                }
            )
            n["id"] = ids[original["id"]]
            n["parameter_ids"] = [parameter_ids[p] for p in original["parameter_ids"]]
            n["references"] = [
                {**ref, "parameter_id": parameter_ids[ref["parameter_id"]]}
                if ref["kind"] == "parameter"
                else ref
                for ref in n.get("references", [])
            ]
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
        "name": "CLM-v0.1-8B",
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


def build(inputs: AnalysisInput, builder: GraphBuilder) -> None:
    cfg = configuration(inputs)
    require(cfg is not None, "Invalid CLM inspection metadata or head bindings.")
    assert cfg is not None
    definition = architecture(inputs, cfg)
    publish_definition(definition, builder)


def register_clm(registry: DescriptionRegistry) -> None:
    registry.register(
        Description(
            PRODUCER,
            "language_model",
            frozenset({"qwen3"}),
            frozenset({"Qwen3ForCausalLM"}),
            lambda inputs: configuration(inputs) is not None,
            build,
        )
    )

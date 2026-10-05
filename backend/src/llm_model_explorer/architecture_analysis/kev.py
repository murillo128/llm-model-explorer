"""Reviewed Kev option-decision graph; no inference or checkpoint code execution.

Kev encode/rows_of/forward_rows_batch/PointerHead at SOURCE_REVISION,
with the existing reviewed Qwen3.5 hybrid description.
"""

from __future__ import annotations

import copy
from typing import Any

from .core import AnalysisInput, Description, DescriptionRegistry, GraphBuilder, Producer
from .kev_config import SOURCE_REVISION, backbone_inputs, configuration
from .packaged import publish_definition
from .qwen35 import PRODUCER as QWEN_PRODUCER
from .qwen35 import build_single_pass as build_qwen
from .qwen35 import configuration as qwen_configuration
from .validation import require

PRODUCER = Producer("kev-inspection", "1-qwen35-" + QWEN_PRODUCER.revision, SOURCE_REVISION)


def shape(*dims: int | str) -> list[dict[str, Any]]:
    return [
        {"kind": "constant", "value": d} if type(d) is int else {"kind": "symbol", "name": d}
        for d in dims
    ]


def port(key: str, dims: list[dict[str, Any]] | None, output: bool = False) -> dict[str, Any]:
    return {
        "id": key,
        "label": key,
        "direction": "output" if output else "input",
        "shape": dims,
    }


class Definition:
    def __init__(self) -> None:
        self.nodes: list[dict[str, Any]] = []
        self.edges: list[dict[str, Any]] = []

    def edge(
        self, source: str, target: str, sp: str = "out", tp: str = "x", kind: str = "data"
    ) -> None:
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
        key: str,
        label: str,
        inputs: dict[str, list[dict[str, Any]] | None],
        output: list[dict[str, Any]] | None,
        *,
        parameters: tuple[str, ...] = (),
        formula: str | None = None,
        attributes: dict[str, Any] | None = None,
        parent: str | None = None,
        kind: str = "operation",
    ) -> str:
        n: dict[str, Any] = {
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


def architecture(inputs: AnalysisInput, cfg: dict[str, Any]) -> dict[str, Any]:
    factors, adapter, head = (cfg[k] for k in ("factors", "adapter", "head"))
    native = backbone_inputs(inputs)
    c = qwen_configuration(native)
    if c is None:
        raise ValueError("Unreviewed Qwen3.5 text configuration")
    b = GraphBuilder(native, QWEN_PRODUCER, "language_model")
    build_qwen(native, b)
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
                "provenance",
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
    # Caller-visible values are semantic text, never forward() arguments.
    # encode derives packed ids/segments/positions/options and readout offsets;
    # rows_of and forward_rows_batch remap them into independent causal rows.
    native_ports = {
        "symbolic_token_ids": "tokens",
        "symbolic_thw_positions": "positions",
        "symbolic_padding_mask": "mask",
        "current_sequence_padding_mask": "current_mask",
    }
    row_outputs = {
        native_ports[n["operation"]]: n["ports"][0]["shape"]
        for n in graph["nodes"]
        if n["kind"] == "input"
    }
    row_outputs.update(decide_idx=shape("B"), opt_idx=shape("B", "O"))
    packed = {k: shape("L") for k in ("ids", "seg", "pos", "opt")}
    packed.update(decide_idx=shape("B"), opt_idx=shape("B", "O"))
    for key, label in (("state", "State"), ("questions", "Questions")):
        d.op(key, label, {}, None, kind="input")
        d.edge(key, "request", tp=key)
    d.nodes.append(
        {
            "id": "request",
            "kind": "group",
            "label": "Kev request encoder / branch packer",
            "ports": [port("state", None), port("questions", None)]
            + [port(k, dims, True) for k, dims in row_outputs.items()],
            "children": ["request.encode", "row.tokens"],
            "description": (
                "Only state and questions (instruction plus options) are caller inputs. "
                "Tokenization, delimiter escaping, positions, masks and readout indices "
                "are deterministic internal preparation; no API question IDs enter the model."
            ),
        }
    )
    d.nodes.append(
        {
            "id": "request.encode",
            "kind": "operation",
            "operation": "kev_request_encoding",
            "label": "Request encoding",
            "parent_id": "request",
            "ports": [port("state", None), port("questions", None)]
            + [port(k, dims, True) for k, dims in packed.items()],
            "formula": "ids, seg, pos, opt, decide_idx, opt_idx = encode(state, questions)",
            "description": (
                "Escape user delimiter syntax, tokenize without added special tokens, "
                "then serialize state and each instruction/options branch with the existing "
                "state/question/option-start/option-end/decision delimiters. Segment zero "
                "is the state; each question has its own segment. Branch positions restart "
                "after the state; readouts locate decision and option-closing tokens."
            ),
        }
    )
    d.edge("request", "request.encode", sp="state", tp="state")
    d.edge("request", "request.encode", sp="questions", tp="questions")
    d.nodes.append(
        {
            "id": "row.tokens",
            "kind": "operation",
            "operation": "independent_causal_rows",
            "label": "Isolated question rows",
            "parent_id": "request",
            "ports": [port(k, dims) for k, dims in packed.items()]
            + [port(k, dims, True) for k, dims in row_outputs.items()],
            "formula": (
                "tokens, positions, mask, current_mask, decide_idx, opt_idx = "
                "rows_of(ids, seg, pos, opt, decide_idx, opt_idx)"
            ),
            "description": (
                "Each causal row concatenates the shared state prefix and exactly one question. "
                "Right-padding masks and text rotary coordinates come from those rows. "
                "Readout indices are remapped into the row, including the state prefix. "
                "KV, convolution and recurrent caches are independent per question; serving "
                "copies prefix caches per row. Options remain sequential; no option "
                "permutation invariance is claimed. A packed block-causal mask is not used."
            ),
        }
    )
    # Row-local readout outputs need distinct IDs from their packed input offsets.
    rows = d.nodes[-1]
    for p in rows["ports"]:
        if p["direction"] == "output" and p["id"] in ("decide_idx", "opt_idx"):
            p["id"] = "row_" + p["id"]
            p["label"] = p["id"]
    rows["formula"] = rows["formula"].replace(
        "current_mask, decide_idx, opt_idx =", "current_mask, row_decide_idx, row_opt_idx ="
    )
    for key in packed:
        d.edge("request.encode", "row.tokens", sp=key, tp=key)
    for key in row_outputs:
        sp = "row_" + key if key in ("decide_idx", "opt_idx") else key
        d.edge("row.tokens", "request", sp=sp, tp=key)
        if key not in ("decide_idx", "opt_idx"):
            d.edge("request", "backbone", sp=key, tp=key)
    h, dp = c["hidden_size"], head["head_dim"]
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
        d.edge("request", selected, sp=index, tp="index")
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
    parameter_keys: dict[str, str] = {}
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
        n["references"] = []
        n.pop("operation", None)
        n.pop("formula", None)
        base["provenance"] = [
            {**p, "source": key + ".base"}
            if p.get("rule") == "Semantic source key in the reviewed packaged description"
            else p
            for p in base.get("provenance", [])
        ]
        d.nodes.append(base)
        a, bb = ("kev.lora." + pair[f] for f in ("A", "B"))
        parameter_keys[a] = key + ".lora_A.weight"
        parameter_keys[bb] = key + ".lora_B.weight"
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
    return {
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
            {"name": "L", "meaning": "Packed request token length before row separation"},
        ],
        "nodes": d.nodes,
        "edges": d.edges,
        "parameters": [
            {
                "id": name,
                "name": name,
                "shape": shape(*dims),
                **({"semantic_key": parameter_keys[name]} if name in parameter_keys else {}),
                "provenance": [p.document() for p in QWEN_PRODUCER.provenance()]
                if not name.startswith("kev.")
                else [],
            }
            for name, dims in inventory.items()
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


def build(inputs: AnalysisInput, builder: GraphBuilder) -> None:
    cfg = configuration(inputs)
    require(cfg is not None, "Invalid Kev inspection metadata or bindings.")
    assert cfg is not None
    publish_definition(architecture(inputs, cfg), builder)


def register_kev(registry: DescriptionRegistry) -> None:
    registry.register(
        Description(
            PRODUCER,
            "language_model",
            frozenset({"qwen3_5"}),
            frozenset({"Qwen3_5ForConditionalGeneration"}),
            lambda inputs: configuration(inputs) is not None,
            build,
        )
    )

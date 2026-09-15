"""Reviewed Transformers V-JEPA 2 static encoder/predictor description.

No Transformers imports or numerical model construction. See the source review in
backend/evidence/vjepa2-description.md for the exact implementation and scope.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Literal

from . import records as r
from .core import AnalysisInput, Description, DescriptionRegistry, GraphBuilder, Producer
from .semantic import operation_role, role_attribute, source_key

PRODUCER = Producer(
    "transformers-vjepa2", "2", "transformers/753d61104116eefc8ffc977327b441ee0c8d599f"
)

# Only defaults read from VJEPA2Config at the reviewed revision are normalized.
DEFAULTS: dict[str, Any] = {
    "patch_size": 16,
    "crop_size": 256,
    "frames_per_clip": 64,
    "tubelet_size": 2,
    "hidden_size": 1024,
    "in_chans": 3,
    "num_attention_heads": 16,
    "num_hidden_layers": 24,
    "drop_path_rate": 0.0,
    "mlp_ratio": 4.0,
    "layer_norm_eps": 1e-6,
    "qkv_bias": True,
    "attention_probs_dropout_prob": 0.0,
    "hidden_act": "gelu",
    "initializer_range": 0.02,
    "attention_dropout": 0.0,
    "num_pooler_layers": 3,
    "pred_hidden_size": 384,
    "pred_num_attention_heads": 12,
    "pred_num_hidden_layers": 12,
    "pred_num_mask_tokens": 10,
    "pred_zero_init_mask_tokens": True,
    "pred_mlp_ratio": 4.0,
}
# These extra fields do not alter VJEPA2Model's reviewed computation. Unknown
# fields fail closed rather than accidentally accepting a future structural option.
METADATA = {
    "model_type",
    "architectures",
    "_name_or_path",
    "name_or_path",
    "_commit_hash",
    "revision",
    "torch_dtype",
    "dtype",
    "transformers_version",
    "image_size",
    "hidden_dropout_prob",
    "use_SiLU",
    "wide_SiLU",
    "return_dict",
    "output_hidden_states",
    "output_attentions",
}


def configuration(inputs: AnalysisInput) -> dict[str, Any] | None:
    raw = inputs.configuration
    if set(raw) - DEFAULTS.keys() - METADATA:
        return None
    c = DEFAULTS | dict(raw)
    ints = (
        "patch_size",
        "crop_size",
        "frames_per_clip",
        "tubelet_size",
        "hidden_size",
        "in_chans",
        "num_attention_heads",
        "num_hidden_layers",
        "pred_hidden_size",
        "pred_num_attention_heads",
        "pred_num_hidden_layers",
        "pred_num_mask_tokens",
    )
    if any(type(c[k]) is not int or not 0 < c[k] <= 1_000_000 for k in ints):
        return None
    if c["num_hidden_layers"] > 256 or c["pred_num_hidden_layers"] > 256:
        return None
    for prefix in ("", "pred_"):
        width, heads = c[prefix + "hidden_size"], c[prefix + "num_attention_heads"]
        ratio = c[prefix + "mlp_ratio"]
        if (
            width % heads
            or width // heads < 6
            or type(ratio) not in (int, float)
            or not math.isfinite(ratio)
            or not 0 < ratio <= 64
            or int(width * ratio) < 1
        ):
            return None
    if (
        type(c["qkv_bias"]) is not bool
        or type(c["pred_zero_init_mask_tokens"]) is not bool
        or c["hidden_act"] != "gelu"
        or c["crop_size"] < c["patch_size"]
        or c.get("image_size", c["crop_size"]) != c["crop_size"]
    ):
        return None
    if any(
        c.get(k, 0) != 0
        for k in (
            "drop_path_rate",
            "attention_probs_dropout_prob",
            "attention_dropout",
            "hidden_dropout_prob",
        )
    ):
        return None
    if (
        type(c["layer_norm_eps"]) not in (int, float)
        or not math.isfinite(c["layer_norm_eps"])
        or c["layer_norm_eps"] <= 0
    ):
        return None
    # Legacy fields are ignored by this Transformers implementation; reject a
    # contradictory activation declaration rather than imply SiLU support.
    if c.get("use_SiLU", False) is not False or c.get("wide_SiLU", True) is not True:
        return None
    return c


def parameter_shapes(c: dict[str, Any]) -> dict[str, list[int]]:
    shapes: dict[str, list[int]] = {}

    def affine(name: str, out: int, incoming: int | None = None, bias: bool = True) -> None:
        shapes[name + ".weight"] = [out] if incoming is None else [out, incoming]
        if bias:
            shapes[name + ".bias"] = [out]

    d, p = c["hidden_size"], c["pred_hidden_size"]
    patch = "encoder.embeddings.patch_embeddings.proj"
    shapes[patch + ".weight"] = [
        d,
        c["in_chans"],
        c["tubelet_size"],
        c["patch_size"],
        c["patch_size"],
    ]
    shapes[patch + ".bias"] = [d]
    shapes["predictor.embeddings.mask_tokens"] = [c["pred_num_mask_tokens"], 1, 1, p]
    affine("predictor.embeddings.predictor_embeddings", p, d)
    affine("predictor.proj", d, p)
    for stack, prefix in (("encoder", ""), ("predictor", "pred_")):
        width = c[prefix + "hidden_size"]
        mlp = int(width * c[prefix + "mlp_ratio"])
        affine(stack + ".layernorm", width)
        for i in range(c[prefix + "num_hidden_layers"]):
            base = f"{stack}.layer.{i}"
            for norm in ("norm1", "norm2"):
                affine(base + "." + norm, width)
            for projection in ("query", "key", "value", "proj"):
                affine(
                    base + ".attention." + projection,
                    width,
                    width,
                    c["qkv_bias"] if projection != "proj" else True,
                )
            affine(base + ".mlp.fc1", mlp, width)
            affine(base + ".mlp.fc2", width, mlp)
    return shapes


def supports(inputs: AnalysisInput) -> bool:
    c = configuration(inputs)
    if c is None:
        return False
    expected = parameter_shapes(c)
    physical = inputs.bindings.physical
    # A missing parameter can be localized by native_parameter; unexpected storage
    # or wrong geometry is contradictory evidence and cannot claim this variant.
    if set(physical) - expected.keys():
        return False
    if not any(name.startswith("encoder.layer.") for name in physical):
        return False
    return all(
        t.shape == expected[name] and t.dtype in ("F32", "F16", "BF16")
        for name, t in physical.items()
    )


def shape(*dims: int | str) -> list[r.ArchitectureDimension]:
    return [
        r.ArchitectureConstantDimension(kind="constant", value=d)
        if isinstance(d, int)
        else r.ArchitectureSymbolDimension(kind="symbol", name=d)
        for d in dims
    ]


@dataclass(frozen=True)
class _Value:
    node: str
    port: str
    shape: r.ArchitectureShape


class _Graph:
    """Small description-local notation; all records still go through GraphBuilder."""

    def __init__(self, builder: GraphBuilder, c: dict[str, Any]) -> None:
        self.b, self.c = builder, c
        self.children: dict[str, list[str]] = {}
        self.parameters = {
            name: builder.native_parameter(
                name,
                name,
                shape(*dims),
                [
                    *PRODUCER.provenance(),
                    r.ArchitectureProvenance(kind="storage", source=name),
                ],
            )
            for name, dims in parameter_shapes(c).items()
        }

    def nid(self, key: str) -> str:
        return self.b.record_id("node", key)

    def link(self, value: _Value, node: str, port: str) -> None:
        key = f"{value.node}:{value.port}->{node}:{port}"
        self.b.add_edge(
            r.ArchitectureEdge(
                id=self.b.record_id("edge", key),
                kind="data",
                source=r.ArchitectureEndpoint(node_id=self.nid(value.node), port_id=value.port),
                target=r.ArchitectureEndpoint(node_id=self.nid(node), port_id=port),
                provenance=PRODUCER.provenance(),
            )
        )

    def node(
        self,
        key: str,
        operation: str,
        inputs: dict[str, _Value],
        outputs: dict[str, r.ArchitectureShape],
        *,
        parent: str | None = None,
        kind: Literal["operation", "input", "output"] = "operation",
        parameters: tuple[str, ...] = (),
        formula: str | None = None,
        attributes: dict[str, Any] | None = None,
    ) -> dict[str, _Value]:
        args: dict[str, Any] = {}
        if parent is not None:
            args["parent_id"] = self.nid(parent)
            self.children[parent].append(self.nid(key))
        if formula is not None:
            args["formula"] = formula
        ids = [self.parameters[p] for p in parameters]
        self.b.add_node(
            r.ArchitectureLeafNode(
                id=self.nid(key),
                kind=kind,
                label=operation_role(key, operation).replace("_", " "),
                operation=operation,
                ports=[
                    r.ArchitecturePort(id=k, direction="input", label=k, shape=v.shape)
                    for k, v in inputs.items()
                ]
                + [
                    r.ArchitecturePort(id=k, direction="output", label=k, shape=s)
                    for k, s in outputs.items()
                ],
                parameter_ids=ids,
                references=[
                    r.ArchitectureParameterReference(kind="parameter", parameter_id=p) for p in ids
                ],
                attributes=[
                    r.ArchitectureAttribute(name=k, value=v, provenance=PRODUCER.provenance())
                    for k, v in (attributes or {}).items()
                ]
                + [role_attribute(PRODUCER, operation_role(key, operation))],
                provenance=PRODUCER.provenance() + [source_key(PRODUCER, key)],
                **args,
            )
        )
        for port, value in inputs.items():
            self.link(value, key, port)
        return {k: _Value(key, k, s) for k, s in outputs.items()}

    def op(
        self,
        key: str,
        operation: str,
        inputs: dict[str, _Value],
        output: r.ArchitectureShape,
        **kwargs: Any,
    ) -> _Value:
        return self.node(key, operation, inputs, {"out": output}, **kwargs)["out"]

    def group(self, key: str, inputs: dict[str, _Value]) -> dict[str, _Value]:
        self.children[key] = []
        for port, value in inputs.items():
            self.link(value, key, port)
        return {port: _Value(key, port, value.shape) for port, value in inputs.items()}

    def end_group(
        self,
        key: str,
        inputs: dict[str, _Value],
        output: _Value,
        parent: str | None = None,
        *,
        role: str | None = None,
    ) -> _Value:
        self.link(output, key, "out")
        args: dict[str, Any] = {}
        if parent is not None:
            args["parent_id"] = self.nid(parent)
            self.children[parent].append(self.nid(key))
        self.b.add_node(
            r.ArchitectureGroupNode(
                id=self.nid(key),
                kind="group",
                label=role.upper() if role == "mlp" else role.title() if role else key,
                children=self.children[key],
                ports=[
                    r.ArchitecturePort(id=k, direction="input", label=k, shape=v.shape)
                    for k, v in inputs.items()
                ]
                + [
                    r.ArchitecturePort(
                        id="out", direction="output", label="out", shape=output.shape
                    )
                ],
                parameter_ids=[],
                references=[r.ArchitectureModuleReference(kind="module", name=key)],
                attributes=(
                    [
                        r.ArchitectureAttribute(
                            name=name,
                            value=self.c[name],
                            provenance=[
                                r.ArchitectureProvenance(
                                    kind="configuration",
                                    source="/" + name,
                                    rule="Explicit declaration or reviewed VJEPA2Config default",
                                )
                            ],
                        )
                        for name in DEFAULTS
                        if name
                        not in {"num_pooler_layers", "attention_dropout", "initializer_range"}
                    ]
                    + [
                        r.ArchitectureAttribute(
                            name="described_path",
                            value="Evaluation; one context/target mask pair per item; no optional "
                            "attention head mask; predictor included",
                            provenance=PRODUCER.provenance(),
                        )
                    ]
                    if parent is None
                    else []
                )
                + ([] if role is None else [role_attribute(PRODUCER, role)]),
                provenance=PRODUCER.provenance() + [source_key(PRODUCER, key)],
                **args,
            )
        )
        return _Value(key, "out", output.shape)

    def affine(
        self, key: str, x: _Value, output: r.ArchitectureShape, parent: str, *, norm: bool = False
    ) -> _Value:
        params = tuple(name for name in (key + ".weight", key + ".bias") if name in self.parameters)
        return self.op(
            key,
            "layer_norm" if norm else "linear",
            {"x": x},
            output,
            parent=parent,
            parameters=params,
            attributes={"epsilon": self.c["layer_norm_eps"]} if norm else {},
            formula="LayerNorm(x; weight, bias, epsilon)"
            if norm
            else ("x W^T + bias" if key + ".bias" in params else "x W^T"),
        )

    def stack(self, stack: str, inputs: dict[str, _Value], prefix: str) -> _Value:
        # The stack group is already open; this method fills its repeated blocks.
        width = self.c[prefix + "hidden_size"]
        heads = self.c[prefix + "num_attention_heads"]
        head = width // heads
        mlp = int(width * self.c[prefix + "mlp_ratio"])
        x, positions = inputs["x"], inputs["positions"]
        assert x.shape is not None
        batch, sequence = x.shape[:2]
        hidden = x.shape
        head_shape = [batch, *shape(heads), sequence, *shape(head)]
        scores_shape = [batch, *shape(heads), sequence, sequence]
        instances = []
        for i in range(self.c[prefix + "num_hidden_layers"]):
            key = f"{stack}.layer.{i}"
            ins = self.group(key, {"x": x, "positions": positions})
            residual = ins["x"]
            norm = self.affine(key + ".norm1", residual, hidden, key, norm=True)
            attention = key + ".attention"
            attention_inputs = {"x": norm, "positions": ins["positions"]}
            av = self.group(attention, attention_inputs)
            qkv = {}
            for role in ("query", "key", "value"):
                projected = self.affine(key + ".attention." + role, av["x"], hidden, attention)
                projected = self.op(
                    key + "." + role + ".heads",
                    "split_heads_transpose",
                    {"x": projected},
                    head_shape,
                    parent=attention,
                )
                if role != "value":
                    projected = self.op(
                        key + "." + role + ".rope",
                        "rotary_3d",
                        {"x": projected, "positions": av["positions"]},
                        head_shape,
                        parent=attention,
                        attributes={
                            "axis_width": 2 * ((head // 3) // 2),
                            "unrotated_width": head - 6 * ((head // 3) // 2),
                            "frequency_base": 10000,
                            "grid_size": self.c["crop_size"] // self.c["patch_size"],
                        },
                        formula="Rotate disjoint temporal/height/width Q or K pairs; "
                        "leave remaining channels unchanged",
                    )
                qkv[role] = projected
            scores = self.op(
                key + ".scores",
                "scaled_query_key_product",
                {"query": qkv["query"], "key": qkv["key"]},
                scores_shape,
                parent=attention,
                formula="Q K^T / sqrt(head_dim)",
                attributes={"causal": False},
            )
            probs = self.op(
                key + ".softmax",
                "softmax",
                {"scores": scores},
                scores_shape,
                parent=attention,
                attributes={"axis": -1},
            )
            values = self.op(
                key + ".weighted_values",
                "attention_value_product",
                {"probabilities": probs, "value": qkv["value"]},
                head_shape,
                parent=attention,
                formula="softmax(scores) V",
            )
            merged = self.op(
                key + ".merge_heads",
                "transpose_merge_heads",
                {"x": values},
                hidden,
                parent=attention,
            )
            attn = self.affine(key + ".attention.proj", merged, hidden, attention)
            attn = self.end_group(attention, attention_inputs, attn, key, role="attention")
            residual = self.op(
                key + ".attention_residual",
                "residual_add",
                {"skip": residual, "branch": attn},
                hidden,
                parent=key,
            )
            norm = self.affine(key + ".norm2", residual, hidden, key, norm=True)
            mlp_group = key + ".mlp"
            mlp_inputs = {"x": norm}
            mv = self.group(mlp_group, mlp_inputs)
            up = self.affine(key + ".mlp.fc1", mv["x"], [batch, sequence, *shape(mlp)], mlp_group)
            active = self.op(key + ".gelu", "gelu", {"x": up}, up.shape, parent=mlp_group)
            down = self.affine(key + ".mlp.fc2", active, hidden, mlp_group)
            down = self.end_group(mlp_group, mlp_inputs, down, key, role="mlp")
            output = self.op(
                key + ".mlp_residual",
                "residual_add",
                {"skip": residual, "branch": down},
                hidden,
                parent=key,
            )
            x = self.end_group(key, ins, output, stack)
            instances.append(
                r.ArchitectureRepetitionInstance(
                    node_id=self.nid(key), index=i, variant="vjepa2_rope_block"
                )
            )
        self.b.add_repetition(
            r.ArchitectureRepetition(
                id=self.b.record_id("repetition", stack),
                parent_id=self.nid(stack),
                label=stack + " blocks",
                instances=instances,
            )
        )
        return self.affine(stack + ".layernorm", x, hidden, stack, norm=True)


def build(inputs: AnalysisInput, builder: GraphBuilder) -> None:
    c = configuration(inputs)
    assert c is not None
    for name, meaning in {
        "B": "Symbolic input batch; one context and one target mask per item",
        "F": "Symbolic input frame count",
        "H": "Symbolic input image height",
        "W": "Symbolic input image width",
        "F_eff": "F * tubelet_size if F < tubelet_size, else F",
        "D_grid": "floor(F_eff / tubelet_size)",
        "H_grid": "floor(H / patch_size)",
        "W_grid": "floor(W / patch_size)",
        "N": "D_grid * H_grid * W_grid patches",
        "C": "Selected context positions per item",
        "T": "Selected target positions per item",
        "P": "C + T concatenated predictor positions",
    }.items():
        builder.add_symbol(name, meaning)
    g = _Graph(builder, c)
    d, p = c["hidden_size"], c["pred_hidden_size"]
    visual = g.op(
        "visual",
        "symbolic_video",
        {},
        shape("B", "F", c["in_chans"], "H", "W"),
        kind="input",
        attributes={"executed": False},
    )
    context = g.op(
        "context_positions",
        "context_position_indices",
        {},
        shape("B", "C"),
        kind="input",
        attributes={"default": "all encoder positions"},
    )
    target = g.op(
        "target_positions",
        "target_position_indices",
        {},
        shape("B", "T"),
        kind="input",
        attributes={"default": "all encoder positions"},
    )
    enc = g.group("encoder", {"video": visual})
    prepared = g.op(
        "encoder.prepare",
        "permute_repeat_short_clip",
        {"video": enc["video"]},
        shape("B", c["in_chans"], "F_eff", "H", "W"),
        parent="encoder",
        formula="Permute B,F,C,H,W to B,C,F,H,W; repeat frames tubelet_size times "
        "only when F < tubelet_size",
    )
    patch = "encoder.embeddings.patch_embeddings.proj"
    projected = g.op(
        patch,
        "patch_projection_3d",
        {"video": prepared},
        shape("B", d, "D_grid", "H_grid", "W_grid"),
        parent="encoder",
        parameters=(patch + ".weight", patch + ".bias"),
        attributes={
            "kernel_and_stride": [c["tubelet_size"], c["patch_size"], c["patch_size"]],
            "padding": 0,
        },
    )
    tokens = g.op(
        "encoder.flatten",
        "flatten_spatiotemporal_transpose",
        {"x": projected},
        shape("B", "N", d),
        parent="encoder",
    )
    positions = g.op(
        "encoder.positions",
        "raster_position_indices",
        {"tokens": tokens},
        shape("N"),
        parent="encoder",
        formula="arange(N); no learned position weights",
    )
    encoded = g.stack("encoder", {"x": tokens, "positions": positions}, "")
    encoded = g.end_group("encoder", enc, encoded)
    g.node("encoder_output", "encoder_representations", {"x": encoded}, {}, kind="output")
    context_repr = g.op(
        "context_selection",
        "gather_context_representations",
        {"x": encoded, "indices": context},
        shape("B", "C", d),
    )
    g.node("context_output", "context_representations", {"x": context_repr}, {}, kind="output")
    target_repr = g.op(
        "target_selection",
        "gather_target_representations",
        {"x": encoded, "indices": target},
        shape("B", "T", d),
    )
    g.node("target_output", "target_representations", {"x": target_repr}, {}, kind="output")
    pred = g.group("predictor", {"encoded": encoded, "context": context, "target": target})
    selected = g.op(
        "predictor.context_selection",
        "gather_context_representations",
        {"x": pred["encoded"], "indices": pred["context"]},
        shape("B", "C", d),
        parent="predictor",
        formula="apply_masks after encoder final LayerNorm",
    )
    projected = g.affine(
        "predictor.embeddings.predictor_embeddings", selected, shape("B", "C", p), "predictor"
    )
    masks = g.op(
        "predictor.mask_tokens",
        "select_repeat_target_mask_token",
        {"indices": pred["target"]},
        shape("B", "T", p),
        parent="predictor",
        parameters=("predictor.embeddings.mask_tokens",),
        attributes={"mask_index": 1 % c["pred_num_mask_tokens"]},
        formula="Select mask_tokens[1 % num_mask_tokens], repeat, gather target positions",
    )
    combined = g.op(
        "predictor.concat",
        "concatenate_context_target",
        {"context": projected, "target": masks},
        shape("B", "P", p),
        parent="predictor",
    )
    pos = g.op(
        "predictor.positions",
        "concatenate_position_indices",
        {"context": pred["context"], "target": pred["target"]},
        shape("B", "P"),
        parent="predictor",
    )
    ordered = g.node(
        "predictor.sort",
        "sort_tokens_by_position",
        {"x": combined, "positions": pos},
        {"tokens": combined.shape, "sorted_positions": pos.shape, "order": pos.shape},
        parent="predictor",
        formula="argsort(positions); gather tokens and positions",
    )
    prediction = g.stack(
        "predictor", {"x": ordered["tokens"], "positions": ordered["sorted_positions"]}, "pred_"
    )
    restored = g.op(
        "predictor.unsort",
        "restore_context_target_order",
        {"x": prediction, "order": ordered["order"]},
        prediction.shape,
        parent="predictor",
        formula="gather(x, argsort(order))",
    )
    predicted_targets = g.op(
        "predictor.target_selection",
        "select_target_suffix",
        {"x": restored, "context_indices": pred["context"]},
        shape("B", "T", p),
        parent="predictor",
        formula="x[:, C:]",
    )
    out = g.affine("predictor.proj", predicted_targets, shape("B", "T", d), "predictor")
    out = g.end_group("predictor", pred, out)
    g.node("predictor_output", "predicted_target_representations", {"x": out}, {}, kind="output")


DESCRIPTION = Description(
    producer=PRODUCER,
    scope="visual_encoder_predictor",
    model_types=frozenset({"vjepa2"}),
    architectures=frozenset({"VJEPA2Model"}),
    supports=supports,
    build=build,
)


def register_vjepa2(registry: DescriptionRegistry) -> None:
    """Register the packaged description with the shared registry explicitly."""
    registry.register(DESCRIPTION)

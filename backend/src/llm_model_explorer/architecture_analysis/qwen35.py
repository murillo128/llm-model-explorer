"""Static Qwen3.5 NVFP4 language description; no model imports or execution.

The source/configuration review and the deliberately bounded selection rules are
recorded in evidence/qwen35-description.md. Storage validation belongs to admission.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Literal

from ..model_files import ModelError
from ..quantized_inventory import encoding, nvfp4_storage_names
from . import records as r
from .core import AnalysisInput, Description, DescriptionRegistry, GraphBuilder, Producer
from .semantic import operation_role, role_attribute, source_key
from .validation import require

PRODUCER = Producer(
    "transformers-qwen35-nvfp4", "2", "transformers/2cba19507be799b7bef247ca6c1c4708bf881b5b"
)
PREFIX = "model.language_model"
# Require explicit structural dimensions; do not transplant another size's defaults.
DIMS = {
    "hidden_size",
    "intermediate_size",
    "vocab_size",
    "num_hidden_layers",
    "num_attention_heads",
    "num_key_value_heads",
    "head_dim",
    "linear_conv_kernel_dim",
    "linear_key_head_dim",
    "linear_value_head_dim",
    "linear_num_key_heads",
    "linear_num_value_heads",
}
OPTIONS = {
    "attention_bias": False,
    "attention_dropout": 0.0,
    "attn_output_gate": True,
    "hidden_act": "silu",
    "model_type": "qwen3_5_text",
    "mlp_only_layers": [],
    "mamba_ssm_dtype": "float32",
    "mtp_num_hidden_layers": 1,
    "mtp_use_dedicated_embeddings": False,
    "tie_word_embeddings": True,
}
METADATA = {
    "dtype",
    "torch_dtype",
    "bos_token_id",
    "eos_token_id",
    "pad_token_id",
    "initializer_range",
    "use_cache",
    "max_position_embeddings",
}
TOP_METADATA = {
    "dtype",
    "torch_dtype",
    "transformers_version",
    "_name_or_path",
    "architectures",
    "model_type",
    "image_token_id",
    "video_token_id",
    "vision_start_token_id",
    "vision_end_token_id",
}


def configuration(inputs: AnalysisInput) -> dict[str, Any] | None:
    raw = inputs.configuration
    if (
        set(raw)
        - TOP_METADATA
        - {"text_config", "vision_config", "tie_word_embeddings", "quantization_config"}
    ):
        return None
    c = raw.get("text_config")
    if not isinstance(c, dict) or not DIMS <= c.keys():
        return None
    if (
        set(c)
        - DIMS
        - OPTIONS.keys()
        - METADATA
        - {
            "layer_types",
            "full_attention_interval",
            "partial_rotary_factor",
            "rope_parameters",
            "rms_norm_eps",
        }
    ):
        return None
    if any(type(c[k]) is not int or not 0 < c[k] <= 1_000_000 for k in DIMS):
        return None
    if c["num_hidden_layers"] > 256:
        return None
    if any(c.get(k) != v or type(c.get(k)) is not type(v) for k, v in OPTIONS.items()):
        return None
    if raw.get("tie_word_embeddings") is not True:
        return None
    vision = raw.get("vision_config")
    if not isinstance(vision, dict) or vision.get("out_hidden_size") != c["hidden_size"]:
        return None
    c = dict(c)
    layers = c.get("layer_types")
    if layers is None:
        interval = c.get("full_attention_interval", 4)
        if type(interval) is not int or interval <= 0:
            return None
        layers = [
            "full_attention" if (i + 1) % interval == 0 else "linear_attention"
            for i in range(c["num_hidden_layers"])
        ]
        c["layer_types"] = layers
    if (
        not isinstance(layers, list)
        or len(layers) != c["num_hidden_layers"]
        or any(x not in ("linear_attention", "full_attention") for x in layers)
    ):
        return None
    if (
        c["num_attention_heads"] % c["num_key_value_heads"]
        or c["linear_num_value_heads"] % c["linear_num_key_heads"]
    ):
        return None
    rope = c.get("rope_parameters")
    if not isinstance(rope, dict) or set(rope) != {
        "rope_type",
        "rope_theta",
        "partial_rotary_factor",
        "mrope_interleaved",
        "mrope_section",
    }:
        return None
    if (
        rope["rope_type"] != "default"
        or rope["mrope_interleaved"] is not True
        or rope["partial_rotary_factor"] != 0.25
        or c.get("partial_rotary_factor") != 0.25
    ):
        return None
    sections = rope["mrope_section"]
    if (
        not isinstance(sections, list)
        or len(sections) != 3
        or any(type(x) is not int or x < 0 for x in sections)
        or sum(sections) != c["head_dim"] // 8
        or c["head_dim"] % 8
    ):
        return None
    for value in (rope["rope_theta"], c.get("rms_norm_eps")):
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(value)
            or value <= 0
        ):
            return None
    return c


def parameter_shapes(c: dict[str, Any]) -> dict[str, list[int]]:
    h, m, v = (c[k] for k in ("hidden_size", "intermediate_size", "vocab_size"))
    shapes = {PREFIX + ".embed_tokens.weight": [v, h], PREFIX + ".norm.weight": [h]}
    for i, variant in enumerate(c["layer_types"]):
        p = f"{PREFIX}.layers.{i}"
        shapes.update(
            {
                p + ".input_layernorm.weight": [h],
                p + ".post_attention_layernorm.weight": [h],
                p + ".mlp.gate_proj.weight": [m, h],
                p + ".mlp.up_proj.weight": [m, h],
                p + ".mlp.down_proj.weight": [h, m],
            }
        )
        if variant == "full_attention":
            d, nh, nk = c["head_dim"], c["num_attention_heads"], c["num_key_value_heads"]
            shapes.update(
                {
                    p + ".self_attn." + k + ".weight": s
                    for k, s in {
                        "q_proj": [2 * nh * d, h],
                        "k_proj": [nk * d, h],
                        "v_proj": [nk * d, h],
                        "o_proj": [h, nh * d],
                        "q_norm": [d],
                        "k_norm": [d],
                    }.items()
                }
            )
        else:
            nk, nv = c["linear_num_key_heads"], c["linear_num_value_heads"]
            kd, vd = nk * c["linear_key_head_dim"], nv * c["linear_value_head_dim"]
            shapes.update(
                {
                    p + ".linear_attn." + k: s
                    for k, s in {
                        "in_proj_qkv.weight": [2 * kd + vd, h],
                        "in_proj_z.weight": [vd, h],
                        "in_proj_a.weight": [nv, h],
                        "in_proj_b.weight": [nv, h],
                        "conv1d.weight": [2 * kd + vd, 1, c["linear_conv_kernel_dim"]],
                        "A_log": [nv],
                        "dt_bias": [nv],
                        "norm.weight": [c["linear_value_head_dim"]],
                        "out_proj.weight": [h, vd],
                    }.items()
                }
            )
    return shapes


def supports(inputs: AnalysisInput) -> bool:
    c = configuration(inputs)
    if c is None:
        return False
    try:
        if encoding(dict(inputs.configuration)) != "nvfp4":
            return False
        packed = nvfp4_storage_names(inputs.bindings.physical, dict(inputs.configuration))
    except ModelError:
        return False
    if not packed:
        return False
    expected = parameter_shapes(c)
    for name, t in inputs.bindings.physical.items():
        # Original implementation does not use checkpoint MTP weights on its
        # language forward path; visual and MTP assets remain explicitly contextual.
        if name.startswith(("model.visual.", "mtp.")):
            continue
        if name in packed and not name.endswith(".weight"):
            continue
        if name not in expected:
            return False
        dims = expected[name]
        logical = [t.shape[0], t.shape[1] * 2] if name in packed else t.shape
        if logical != dims or (name not in packed and t.dtype not in ("BF16", "F16", "F32")):
            return False
    return True


def shape(*dims: int | str) -> list[r.ArchitectureDimension]:
    return [
        r.ArchitectureConstantDimension(kind="constant", value=d)
        if isinstance(d, int)
        else r.ArchitectureSymbolDimension(kind="symbol", name=d)
        for d in dims
    ]


@dataclass(frozen=True)
class Value:
    node: str
    port: str
    shape: r.ArchitectureShape
    kind: Literal["data", "state", "context"] = "data"


class Graph:
    """Description-local notation, emitting all records through the bounded core."""

    def __init__(self, b: GraphBuilder, c: dict[str, Any]):
        self.b, self.c = b, c
        self.children: dict[str, list[str]] = {}
        self.parameters: dict[str, str] = {}
        numeric_by_name = {tensor.name: tensor for tensor in b.inputs.bindings.numeric.values()}
        packed = nvfp4_storage_names(b.inputs.bindings.physical, dict(b.inputs.configuration))
        for name, dims in parameter_shapes(c).items():
            provenance = PRODUCER.provenance() + [
                r.ArchitectureProvenance(
                    kind="configuration",
                    source="/text_config",
                    rule="Checked logical parameter geometry",
                )
            ]
            if name in b.inputs.bindings.physical:
                provenance.append(r.ArchitectureProvenance(kind="storage", source=name))
            if name not in packed:
                pid = b.native_parameter(name, name, shape(*dims), provenance)
            else:
                prefix = name.removesuffix(".weight")
                numeric = numeric_by_name.get(name)
                storage = [
                    b.inputs.bindings.physical[prefix + suffix].model_copy(update={"role": role})
                    for suffix, role in [
                        (".weight", "packed_nvfp4"),
                        (".weight_scale", "block_scale"),
                        (".weight_scale_2", "global_weight_scale"),
                        (".input_scale", "input_scale"),
                    ]
                ]
                param = r.ArchitectureDirectParameter(
                    id=b.record_id("parameter", name),
                    name=name,
                    logical_shape=shape(*dims),
                    binding="quantized",
                    storage=storage,
                    inspection=r.ArchitectureAvailableInspection(
                        status="available", tensor_id=numeric.id
                    )
                    if numeric is not None
                    else r.ArchitectureUnavailableInspection(
                        status="unavailable",
                        reason="unsupported_representation",
                        message="No admitted complete NVFP4 numeric tensor exists.",
                    ),
                    provenance=provenance
                    + [r.ArchitectureProvenance(kind="storage", source=s.name) for s in storage],
                )
                pid = b.add_parameter(param)
            self.parameters[name] = pid
        # The reviewed checkpoint omits lm_head.weight and declares tying in both configs.
        name = PREFIX + ".embed_tokens.weight"
        embedding_storage = b.inputs.bindings.physical.get(name)
        numeric = next((t for t in b.inputs.bindings.numeric.values() if t.name == name), None)
        self.parameters["lm_head.weight"] = b.add_parameter(
            r.ArchitectureAliasParameter(
                id=b.record_id("parameter", "lm_head.weight"),
                name="lm_head.weight",
                logical_shape=shape(c["vocab_size"], c["hidden_size"]),
                binding="alias",
                alias_of=self.parameters[name],
                storage=[] if embedding_storage is None else [embedding_storage],
                inspection=r.ArchitectureAvailableInspection(
                    status="available", tensor_id=numeric.id
                )
                if numeric is not None
                else r.ArchitectureUnavailableInspection(
                    status="unavailable",
                    reason="unresolved_binding"
                    if embedding_storage is None
                    else "unsupported_representation",
                    message="Tied embedding has no admitted numeric view.",
                ),
                provenance=PRODUCER.provenance()
                + [
                    r.ArchitectureProvenance(
                        kind="configuration",
                        source="/tie_word_embeddings",
                        rule="Tied to model.language_model.embed_tokens.weight",
                    )
                ],
            )
        )

    def nid(self, key: str) -> str:
        return self.b.record_id("node", key)

    def link(self, value: Value, key: str, port: str) -> None:
        self.b.add_edge(
            r.ArchitectureEdge(
                id=self.b.record_id("edge", f"{value.node}:{value.port}>{key}:{port}"),
                source=r.ArchitectureEndpoint(node_id=self.nid(value.node), port_id=value.port),
                target=r.ArchitectureEndpoint(node_id=self.nid(key), port_id=port),
                kind=value.kind,
                provenance=PRODUCER.provenance(),
            )
        )

    def node(
        self,
        key: str,
        op: str,
        inputs: dict[str, Value],
        outputs: dict[str, r.ArchitectureShape],
        *,
        parent: str | None = None,
        kind: Literal["operation", "input", "output", "context", "state"] = "operation",
        parameters: tuple[str, ...] = (),
        formula: str | None = None,
        attributes: dict[str, Any] | None = None,
    ) -> dict[str, Value]:
        args: dict[str, Any] = {}
        if parent is not None:
            args["parent_id"] = self.nid(parent)
            self.children[parent].append(self.nid(key))
        if formula is not None:
            args["formula"] = formula
        ids = [self.parameters[p] for p in parameters]
        role = "query_gate_projection" if key.endswith(".q_proj") else operation_role(key, op)
        self.b.add_node(
            r.ArchitectureLeafNode(
                id=self.nid(key),
                kind=kind,
                label=role.replace("_", " "),
                operation=op,
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
                + [role_attribute(PRODUCER, role)],
                provenance=PRODUCER.provenance() + [source_key(PRODUCER, key)],
                **args,
            )
        )
        for k, v in inputs.items():
            self.link(v, key, k)
        return {
            k: Value(key, k, s, "state" if kind == "state" or k.endswith("_state") else "data")
            for k, s in outputs.items()
        }

    def op(
        self, key: str, op: str, inputs: dict[str, Value], out: r.ArchitectureShape, **kwargs: Any
    ) -> Value:
        return self.node(key, op, inputs, {"out": out}, **kwargs)["out"]

    def group(self, key: str, inputs: dict[str, Value]) -> dict[str, Value]:
        self.children[key] = []
        for k, v in inputs.items():
            self.link(v, key, k)
        return {k: Value(key, k, v.shape, v.kind) for k, v in inputs.items()}

    def end(
        self,
        key: str,
        inputs: dict[str, Value],
        out: Value,
        parent: str | None = None,
        *,
        role: str | None = None,
    ) -> Value:
        self.link(out, key, "out")
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
                + [r.ArchitecturePort(id="out", direction="output", label="out", shape=out.shape)],
                parameter_ids=[],
                references=[r.ArchitectureModuleReference(kind="module", name=key)],
                attributes=[] if role is None else [role_attribute(PRODUCER, role)],
                provenance=PRODUCER.provenance() + [source_key(PRODUCER, key)],
                **args,
            )
        )
        return Value(key, "out", out.shape)

    def linear(self, p: str, x: Value, out: int, parent: str) -> Value:
        return self.op(
            p, "linear", {"x": x}, shape("B", "T", out), parent=parent, parameters=(p + ".weight",)
        )

    def norm(self, p: str, x: Value, parent: str, *, centered: bool = True) -> Value:
        return self.op(
            p,
            "rms_norm_zero_centered" if centered else "rms_norm",
            {"x": x},
            x.shape,
            parent=parent,
            parameters=(p + ".weight",),
            attributes={"epsilon": self.c["rms_norm_eps"]},
            formula="x / sqrt(mean(x^2) + eps) * " + ("(1 + weight)" if centered else "weight"),
        )

    def region(self, name: str, source: str, dims: list[int], description: str) -> str:
        storage = self.b.inputs.bindings.physical.get(source)
        if storage is None:
            self.parameters[name] = self.b.native_parameter(
                name, name, shape(*dims), PRODUCER.provenance()
            )
            return self.parameters[name]
        prefix = source.removesuffix(".weight")
        records = [
            s
            for n, s in self.b.inputs.bindings.physical.items()
            if n == source
            or n in {prefix + ".weight_scale", prefix + ".weight_scale_2", prefix + ".input_scale"}
        ]
        pid = self.b.add_parameter(
            r.ArchitectureFusedParameter(
                id=self.b.record_id("parameter", name),
                name=name,
                logical_shape=shape(*dims),
                binding="fused_region",
                storage=records,
                region=r.ArchitectureRegion(storage_name=source, description=description),
                inspection=r.ArchitectureUnavailableInspection(
                    status="unavailable",
                    reason="requires_view",
                    message="This logical region requires a view of the fused parameter; no "
                    "complete native endpoint exists.",
                ),
                provenance=PRODUCER.provenance()
                + [r.ArchitectureProvenance(kind="storage", source=source)],
            )
        )
        self.parameters[name] = pid
        return pid


def full_attention(g: Graph, p: str, x: Value, positions: Value, mask: Value) -> Value:
    c = g.c
    h = c["hidden_size"]
    nh = c["num_attention_heads"]
    nk = c["num_key_value_heads"]
    d = c["head_dim"]
    inputs = {"x": x, "positions": positions, "mask": mask}
    v = g.group(p, inputs)
    x = v["x"]
    fused = g.linear(p + ".q_proj", x, 2 * nh * d, p)
    heads = g.op(
        p + ".query_gate_heads", "reshape", {"x": fused}, shape("B", "T", nh, 2 * d), parent=p
    )
    for role, start in [("query", 0), ("gate", d)]:
        name = p + "." + role + ".weight"
        g.region(
            name,
            p + ".q_proj.weight",
            [nh * d, h],
            f"Logical output rows per head j: [j*{2 * d}+{start}, j*{2 * d}+{start + d}); all "
            f"{h} input columns. NVFP4 packing is on input columns.",
        )
        if name not in g.parameters:
            g.parameters[name] = g.parameters[p + ".q_proj.weight"]
    split = g.node(
        p + ".query_gate_split",
        "split_query_gate_per_head",
        {"x": heads},
        {"query": shape("B", "T", nh, d), "gate": shape("B", "T", nh, d)},
        parent=p,
        parameters=(p + ".query.weight", p + ".gate.weight"),
        attributes={"axis": -1, "head_dim": d},
    )
    q = g.norm(p + ".q_norm", split["query"], p)
    k = g.linear(p + ".k_proj", x, nk * d, p)
    k = g.op(p + ".key_heads", "reshape", {"x": k}, shape("B", "T", nk, d), parent=p)
    k = g.norm(p + ".k_norm", k, p)
    value = g.linear(p + ".v_proj", x, nk * d, p)
    value = g.op(
        p + ".value_heads", "reshape_transpose", {"x": value}, shape("B", nk, "T", d), parent=p
    )
    rotary = {}
    for role, tensor, heads_n in [("query", q, nh), ("key", k, nk)]:
        t = g.op(
            p + "." + role + "_rope",
            "partial_interleaved_rope",
            {"x": tensor, "positions": v["positions"]},
            tensor.shape,
            parent=p,
            attributes={
                "rotary_dim": d // 4,
                "rope_theta": c["rope_parameters"]["rope_theta"],
                "mrope_section": c["rope_parameters"]["mrope_section"],
            },
            formula="Rotate first head_dim/4 channels using interleaved T/H/W positions; pass "
            "remaining channels unchanged.",
        )
        rotary[role] = g.op(
            p + "." + role + "_transpose",
            "transpose",
            {"x": t},
            shape("B", heads_n, "T", d),
            parent=p,
        )
    # Symbolic prior/next KV dependencies, not a captured cache or executed timeline.
    state = g.node(
        p + ".prior_kv",
        "prior_kv",
        {},
        {"key_state": shape("B", nk, "P", d), "value_state": shape("B", nk, "P", d)},
        kind="state",
        parent=p,
    )
    joined = g.node(
        p + ".kv_concat",
        "concatenate_prior_kv",
        {"key": rotary["key"], "value": value, **{"prior_" + k: v for k, v in state.items()}},
        {"key_state": shape("B", nk, "K", d), "value_state": shape("B", nk, "K", d)},
        parent=p,
        attributes={"key_length": "K = P + T; P may be zero"},
    )
    g.node(p + ".next_kv", "next_kv", joined, {}, kind="state", parent=p)
    k = g.op(
        p + ".repeat_key",
        "repeat_kv_heads",
        {"x": joined["key_state"]},
        shape("B", nh, "K", d),
        parent=p,
        attributes={"groups": nh // nk},
    )
    value = g.op(
        p + ".repeat_value",
        "repeat_kv_heads",
        {"x": joined["value_state"]},
        shape("B", nh, "K", d),
        parent=p,
        attributes={"groups": nh // nk},
    )
    kt = g.op(
        p + ".key_matrix_transpose",
        "transpose_last_axes",
        {"x": k},
        shape("B", nh, d, "K"),
        parent=p,
    )
    scores = g.op(
        p + ".scores",
        "scaled_query_key_product",
        {"query": rotary["query"], "key": kt},
        shape("B", nh, "T", "K"),
        parent=p,
        formula="Q @ transpose(K) / sqrt(head_dim)",
    )
    masked = g.op(
        p + ".causal_mask",
        "causal_padding_mask",
        {"scores": scores, "mask": v["mask"]},
        scores.shape,
        parent=p,
    )
    probs = g.op(
        p + ".softmax", "softmax", {"x": masked}, masked.shape, parent=p, attributes={"axis": -1}
    )
    weighted = g.op(
        p + ".weighted_values",
        "attention_value_product",
        {"probabilities": probs, "value": value},
        shape("B", nh, "T", d),
        parent=p,
    )
    merged = g.op(
        p + ".merge_heads", "transpose_reshape", {"x": weighted}, shape("B", "T", nh * d), parent=p
    )
    gate = g.op(p + ".gate_flatten", "reshape", {"x": split["gate"]}, merged.shape, parent=p)
    gate = g.op(p + ".sigmoid", "sigmoid", {"x": gate}, gate.shape, parent=p)
    gated = g.op(
        p + ".output_gate", "multiply", {"attention": merged, "gate": gate}, merged.shape, parent=p
    )
    return g.end(
        p, inputs, g.linear(p + ".o_proj", gated, h, p), p.rsplit(".", 1)[0], role="attention"
    )


def linear_attention(g: Graph, p: str, x: Value, mask: Value) -> Value:
    c = g.c
    h = c["hidden_size"]
    nk = c["linear_num_key_heads"]
    nv = c["linear_num_value_heads"]
    dk = c["linear_key_head_dim"]
    dv = c["linear_value_head_dim"]
    kd = nk * dk
    vd = nv * dv
    width = 2 * kd + vd
    inputs = {"x": x, "mask": mask}
    v = g.group(p, inputs)
    x = g.op(p + ".padding_mask", "mask_padding_states", v, v["x"].shape, parent=p)
    qkv = g.linear(p + ".in_proj_qkv", x, width, p)
    z = g.linear(p + ".in_proj_z", x, vd, p)
    a = g.linear(p + ".in_proj_a", x, nv, p)
    b = g.linear(p + ".in_proj_b", x, nv, p)
    trans = g.op(p + ".conv_transpose", "transpose", {"x": qkv}, shape("B", width, "T"), parent=p)
    prior = g.op(
        p + ".prior_conv",
        "prior_convolution_state",
        {},
        shape("B", width, c["linear_conv_kernel_dim"]),
        kind="state",
        parent=p,
    )
    conv = g.node(
        p + ".conv1d",
        "causal_depthwise_convolution",
        {"x": trans, "prior_state": prior},
        {"out": trans.shape, "next_state": prior.shape},
        parent=p,
        parameters=(p + ".conv1d.weight",),
        attributes={"kernel_size": c["linear_conv_kernel_dim"], "groups": width, "bias": False},
        formula="Causal local mixing of projected QKV; retain newest kernel-width inputs as "
        "symbolic next state.",
    )
    g.node(
        p + ".next_conv",
        "next_convolution_state",
        {"state": conv["next_state"]},
        {},
        kind="state",
        parent=p,
    )
    activated = g.op(p + ".conv_silu", "silu", {"x": conv["out"]}, trans.shape, parent=p)
    mixed = g.op(
        p + ".conv_to_sequence", "transpose", {"x": activated}, shape("B", "T", width), parent=p
    )
    branches = g.node(
        p + ".qkv_split",
        "split_reshape_qkv",
        {"x": mixed},
        {
            "query": shape("B", "T", nk, dk),
            "key": shape("B", "T", nk, dk),
            "value": shape("B", "T", nv, dv),
        },
        parent=p,
        attributes={"split_widths": [kd, kd, vd]},
    )
    qk = {}
    for role in ("query", "key"):
        repeated = g.op(
            p + "." + role + "_repeat",
            "repeat_key_heads",
            {"x": branches[role]},
            shape("B", "T", nv, dk),
            parent=p,
            attributes={"groups": nv // nk},
        )
        normalized = g.op(
            p + "." + role + "_l2",
            "l2_normalize",
            {"x": repeated},
            repeated.shape,
            parent=p,
            formula="x / sqrt(sum(x^2) + 1e-6)",
        )
        qk[role] = normalized
    qk["query"] = g.op(
        p + ".query_scale",
        "scale_query",
        {"x": qk["query"]},
        qk["query"].shape,
        parent=p,
        formula="q / sqrt(key_head_dim)",
    )
    beta = g.op(p + ".beta", "sigmoid", {"x": b}, b.shape, parent=p)
    decay = g.op(
        p + ".decay",
        "log_decay",
        {"a": a},
        a.shape,
        parent=p,
        parameters=(p + ".A_log", p + ".dt_bias"),
        formula="g = -exp(A_log) * softplus(a + dt_bias)",
    )
    prior = g.op(
        p + ".prior_recurrent",
        "prior_delta_state",
        {},
        shape("B", nv, dk, dv),
        kind="state",
        parent=p,
    )
    update = g.node(
        p + ".delta_rule",
        "gated_delta_rule",
        dict(qk, value=branches["value"], beta=beta, log_decay=decay, prior_state=prior),
        {"out": shape("B", "T", nv, dv), "next_state": prior.shape},
        parent=p,
        formula="D = exp(g_t) * S_prev; delta = beta_t * (v_t - k_t^T D); S_next = D + k_t "
        "delta^T; y_t = q_t^T S_next. Symbolic algorithm, not an executed recurrence.",
    )
    g.node(
        p + ".next_recurrent",
        "next_delta_state",
        {"state": update["next_state"]},
        {},
        kind="state",
        parent=p,
    )
    norm = g.norm(p + ".norm", update["out"], p, centered=False)
    z = g.op(p + ".z_heads", "reshape", {"x": z}, norm.shape, parent=p)
    gate = g.op(p + ".z_silu", "silu", {"x": z}, z.shape, parent=p)
    gated = g.op(
        p + ".output_gate", "multiply", {"normalized": norm, "gate": gate}, norm.shape, parent=p
    )
    merged = g.op(p + ".merge_heads", "reshape", {"x": gated}, shape("B", "T", vd), parent=p)
    return g.end(
        p, inputs, g.linear(p + ".out_proj", merged, h, p), p.rsplit(".", 1)[0], role="attention"
    )


def build(inputs: AnalysisInput, b: GraphBuilder) -> None:
    c = configuration(inputs)
    require(c is not None, "Unsupported Qwen3.5 configuration.")
    assert c is not None
    for symbol, meaning in [
        ("B", "Batch"),
        ("T", "Current token sequence"),
        ("P", "Prior KV sequence, possibly zero"),
        ("K", "Total key sequence P + T"),
    ]:
        b.add_symbol(symbol, meaning)
    g = Graph(b, c)
    h = c["hidden_size"]
    tokens = g.op("token_ids", "symbolic_token_ids", {}, shape("B", "T"), kind="input")
    positions = g.op("positions", "symbolic_thw_positions", {}, shape(3, "B", "T"), kind="input")
    mask = g.op("mask", "symbolic_padding_mask", {}, shape("B", "K"), kind="input")
    # Linear path receives a current-sequence padding mask, separate from KV mask.
    current_mask = g.op(
        "current_mask", "current_sequence_padding_mask", {}, shape("B", "T"), kind="input"
    )
    for key, label in [
        ("visual_context", "visual_components_context_only"),
        ("mtp_context", "auxiliary_mtp_weights_unused_by_reviewed_language_path"),
    ]:
        g.node(key, label, {}, {}, kind="context")
    if inputs.bindings.tokenizer_available:
        b.add_node(
            r.ArchitectureLeafNode(
                id=g.nid("tokenizer"),
                kind="context",
                label="Tokenizer capability",
                ports=[],
                parameter_ids=[],
                references=[r.ArchitectureTokenizerReference(kind="tokenizer")],
                attributes=[],
                provenance=PRODUCER.provenance(),
            )
        )
    root_inputs = {
        "tokens": tokens,
        "positions": positions,
        "mask": mask,
        "current_mask": current_mask,
    }
    v = g.group(PREFIX, root_inputs)
    x = g.op(
        PREFIX + ".embed_tokens",
        "embedding_lookup",
        {"tokens": v["tokens"]},
        shape("B", "T", h),
        parent=PREFIX,
        parameters=(PREFIX + ".embed_tokens.weight",),
    )
    instances = []
    for i, variant in enumerate(c["layer_types"]):
        p = f"{PREFIX}.layers.{i}"
        layer_inputs = {
            "x": x,
            "positions": v["positions"],
            "mask": v["mask"],
            "current_mask": v["current_mask"],
        }
        lv = g.group(p, layer_inputs)
        residual = lv["x"]
        normalized = g.norm(p + ".input_layernorm", residual, p)
        attn_key = p + (".self_attn" if variant == "full_attention" else ".linear_attn")
        attn = (
            full_attention(g, attn_key, normalized, lv["positions"], lv["mask"])
            if variant == "full_attention"
            else linear_attention(g, attn_key, normalized, lv["current_mask"])
        )
        post = g.op(
            p + ".attention_residual",
            "add",
            {"skip": residual, "branch": attn},
            residual.shape,
            parent=p,
        )
        norm = g.norm(p + ".post_attention_layernorm", post, p)
        mlp = p + ".mlp"
        mlp_inputs = {"x": norm}
        mv = g.group(mlp, mlp_inputs)
        gate = g.linear(p + ".mlp.gate_proj", mv["x"], c["intermediate_size"], mlp)
        up = g.linear(p + ".mlp.up_proj", mv["x"], c["intermediate_size"], mlp)
        activated = g.op(p + ".mlp.silu", "silu", {"x": gate}, gate.shape, parent=mlp)
        multiplied = g.op(
            p + ".mlp.multiply", "multiply", {"gate": activated, "up": up}, up.shape, parent=mlp
        )
        down = g.linear(p + ".mlp.down_proj", multiplied, h, mlp)
        down = g.end(mlp, mlp_inputs, down, p, role="mlp")
        out = g.op(p + ".mlp_residual", "add", {"skip": post, "branch": down}, post.shape, parent=p)
        x = g.end(p, layer_inputs, out, PREFIX)
        instances.append(
            r.ArchitectureRepetitionInstance(node_id=g.nid(p), index=i, variant=variant)
        )
    final = g.norm(PREFIX + ".norm", x, PREFIX)
    x = g.end(PREFIX, root_inputs, final)
    logits = g.op(
        "lm_head",
        "linear",
        {"x": x},
        shape("B", "T", c["vocab_size"]),
        parameters=("lm_head.weight",),
    )
    g.node("logits", "vocabulary_logits", {"x": logits}, {}, kind="output")
    b.add_repetition(
        r.ArchitectureRepetition(
            id=b.record_id("repetition", "decoder_layers"),
            parent_id=g.nid(PREFIX),
            label="Configured hybrid decoder layers",
            instances=instances,
        )
    )


def register_qwen35(registry: DescriptionRegistry) -> None:
    registry.register(
        Description(
            PRODUCER,
            "language_model",
            frozenset({"qwen3_5"}),
            frozenset({"Qwen3_5ForConditionalGeneration"}),
            supports,
            build,
        )
    )

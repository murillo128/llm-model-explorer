"""Packaged static dense-language descriptions; never an inference implementation."""

import re
from dataclasses import dataclass
from math import isfinite
from typing import TYPE_CHECKING, Literal

from . import records as r
from .core import AnalysisInput, Description, DescriptionRegistry, GraphBuilder, Producer
from .dense_config import SOURCE_REVISION, DenseConfig, checked
from .semantic import operation_role, role_attribute, source_key
from .validation import GraphError, require

if TYPE_CHECKING:
    from ..tensor_source import PeftLoraTarget


def shape(*dims: int | str) -> list[r.ArchitectureDimension]:
    return [
        r.ArchitectureConstantDimension(kind="constant", value=d)
        if isinstance(d, int)
        else r.ArchitectureSymbolDimension(kind="symbol", name=d)
        for d in dims
    ]


@dataclass(frozen=True)
class LinearNodes:
    base: str
    output: str
    adapter_input: str | None = None


_ADAPTER_FACTOR = re.compile(r"\.lora_[AB](?:\.[A-Za-z0-9_-]+)?\.weight$")


class DenseGraph:
    """Shared fragments only where reviewed Qwen3/Llama mathematics agrees."""

    def __init__(self, inputs: AnalysisInput, builder: GraphBuilder, config: DenseConfig):
        self.inputs, self.b, self.c = inputs, builder, config
        self.children: dict[str, list[str]] = {}
        self.parameters: dict[str, r.ArchitectureParameter] = {}
        self.used_storage: set[str] = set()
        self.numeric = {t.name: t for t in inputs.bindings.numeric.values()}
        self.lora = inputs.lora_composition
        self.lora_targets = self._validate_lora()

    def _lora_error(self, code: str, message: str) -> None:
        raise GraphError(code, message)

    def _validate_lora(self) -> dict[str, "PeftLoraTarget"]:
        composition = self.lora
        context = self.inputs.bindings
        if composition is None:
            if context.adapter_tensor_storage:
                self._lora_error(
                    "lora_orphan_factor",
                    "Adapter tensor bindings exist without a PEFT composition.",
                )
            return {}
        if (
            type(composition.rank) is not int
            or composition.rank <= 0
            or isinstance(composition.alpha, bool)
            or not isinstance(composition.alpha, (int, float))
            or not isfinite(composition.alpha)
            or composition.alpha <= 0
            or isinstance(composition.scale, bool)
            or not isinstance(composition.scale, (int, float))
            or not isfinite(composition.scale)
            or composition.scale <= 0
            or composition.scale != composition.alpha / composition.rank
        ):
            self._lora_error(
                "lora_scale", "LoRA scale must equal the configured alpha divided by rank."
            )
        expected: dict[str, tuple[int, int]] = {"lm_head": (self.c.vocab, self.c.hidden)}
        for index in range(self.c.layers):
            prefix = f"model.layers.{index}."
            expected.update(
                {
                    prefix + "self_attn.q_proj": (self.c.heads * self.c.head_dim, self.c.hidden),
                    prefix + "self_attn.k_proj": (
                        self.c.kv_heads * self.c.head_dim,
                        self.c.hidden,
                    ),
                    prefix + "self_attn.v_proj": (
                        self.c.kv_heads * self.c.head_dim,
                        self.c.hidden,
                    ),
                    prefix + "self_attn.o_proj": (
                        self.c.hidden,
                        self.c.heads * self.c.head_dim,
                    ),
                    prefix + "mlp.gate_proj": (self.c.intermediate, self.c.hidden),
                    prefix + "mlp.up_proj": (self.c.intermediate, self.c.hidden),
                    prefix + "mlp.down_proj": (self.c.hidden, self.c.intermediate),
                }
            )
        targets: dict[str, PeftLoraTarget] = {}
        logical_names: set[str] = set()
        storage_names: set[str] = set()
        for target in composition.targets:
            if target.module_name in targets:
                self._lora_error(
                    "lora_ambiguous_target",
                    "A LoRA target resolves to more than one graph operation.",
                )
            dims = expected.get(target.module_name)
            if dims is None:
                self._lora_error(
                    "lora_unsupported_target",
                    "A LoRA target has no reviewed linear graph operation.",
                )
            assert dims is not None
            a_name, b_name = target.a_tensor_name, target.b_tensor_name
            if a_name == b_name or a_name in logical_names or b_name in logical_names:
                self._lora_error(
                    "lora_ambiguous_target", "LoRA factors do not have unique logical identities."
                )
            logical_names.update((a_name, b_name))
            storage_names.update((target.a_storage_name, target.b_storage_name))
            for logical_name, physical_name, geometry in (
                (a_name, target.a_storage_name, (composition.rank, dims[1])),
                (b_name, target.b_storage_name, (dims[0], composition.rank)),
            ):
                matches = [t for t in context.numeric.values() if t.name == logical_name]
                storage = context.physical.get(physical_name)
                if len(matches) != 1 or storage is None:
                    self._lora_error(
                        "lora_missing_factor",
                        "Each LoRA target requires one verified A and B tensor.",
                    )
                tensor = matches[0]
                if (
                    tuple(storage.shape) != geometry
                    or tensor.shape != geometry
                    or tensor.dtype != storage.dtype
                    or storage.dtype not in {"F32", "F16", "BF16"}
                    or context.adapter_tensor_storage.get(logical_name) != physical_name
                ):
                    self._lora_error(
                        "lora_factor_geometry",
                        "LoRA A/B tensor shapes or storage do not match the target.",
                    )
            base = self.numeric.get(target.module_name + ".weight")
            if base is None or base.shape != dims:
                self._lora_error(
                    "lora_base_binding", "A LoRA target has no matching base linear weight."
                )
            targets[target.module_name] = target
        logical_adapter_names = {
            tensor.name
            for tensor in context.numeric.values()
            if tensor.name.startswith("__peft__.")
        }
        actual_factor_storage = {name for name in context.physical if _ADAPTER_FACTOR.search(name)}
        if (
            not targets
            or set(context.adapter_tensor_storage) != logical_names
            or logical_adapter_names != logical_names
            or actual_factor_storage != storage_names
        ):
            self._lora_error(
                "lora_orphan_factor",
                "LoRA factors are missing, duplicated, or outside the verified targets.",
            )
        return targets

    def provenance(self, *fields: str) -> list[r.ArchitectureProvenance]:
        return self.b.producer.provenance() + [
            r.ArchitectureProvenance(kind="configuration", source=f"config.json#/{f}")
            for f in fields
            if f in self.inputs.configuration
        ]

    def nid(self, key: str) -> str:
        return self.b.record_id("node", key)

    def port(
        self, name: str, dims: r.ArchitectureShape, output: bool = False
    ) -> r.ArchitecturePort:
        return r.ArchitecturePort(
            id=name, direction="output" if output else "input", label=name, shape=dims
        )

    def edge(self, source: str, target: str, port: str = "x", source_port: str = "out") -> None:
        self.b.add_edge(
            r.ArchitectureEdge(
                id=self.b.record_id("edge", f"{source}:{source_port}->{target}:{port}"),
                source=r.ArchitectureEndpoint(node_id=self.nid(source), port_id=source_port),
                target=r.ArchitectureEndpoint(node_id=self.nid(target), port_id=port),
                kind="data",
                provenance=self.b.producer.provenance(),
            )
        )

    def node(
        self,
        key: str,
        parent: str,
        operation: str,
        inputs: dict[str, r.ArchitectureShape],
        output: r.ArchitectureShape,
        *,
        parameters: tuple[str, ...] = (),
        attributes: dict[str, str | float | bool] | None = None,
        formula: str | None = None,
        kind: Literal["operation", "input", "output"] = "operation",
        fields: tuple[str, ...] = (),
        extra_provenance: tuple[r.ArchitectureProvenance, ...] = (),
    ) -> str:
        self.children.setdefault(parent, []).append(self.nid(key))
        kwargs = {} if formula is None else {"formula": formula}
        references: list[r.ArchitectureReference] = [
            r.ArchitectureParameterReference(kind="parameter", parameter_id=p) for p in parameters
        ]
        if parameters:
            references.insert(0, r.ArchitectureModuleReference(kind="module", name=key))
        self.b.add_node(
            r.ArchitectureLeafNode(
                id=self.nid(key),
                parent_id=self.nid(parent),
                kind=kind,
                label=operation_role(key, operation).replace("_", " "),
                operation=operation,
                ports=[self.port(k, v) for k, v in inputs.items()]
                + [self.port("out", output, True)],
                parameter_ids=list(parameters),
                references=references,
                attributes=[
                    r.ArchitectureAttribute(
                        name=k,
                        value=v,
                        provenance=self.provenance(*fields) + list(extra_provenance),
                    )
                    for k, v in (attributes or {}).items()
                ]
                + [role_attribute(self.b.producer, operation_role(key, operation))],
                provenance=self.provenance(*fields)
                + list(extra_provenance)
                + [source_key(self.b.producer, key)],
                **kwargs,
            ),
            semantic_key=key,
        )
        return key

    def group(
        self,
        key: str,
        parent: str | None,
        ports: list[r.ArchitecturePort],
        *,
        role: str | None = None,
    ) -> None:
        kwargs = {} if parent is None else {"parent_id": self.nid(parent)}
        if parent is not None:
            self.children.setdefault(parent, []).append(self.nid(key))
        self.b.add_node(
            r.ArchitectureGroupNode(
                id=self.nid(key),
                kind="group",
                label=role.upper() if role == "mlp" else role.title() if role else key,
                ports=ports,
                children=self.children.get(key, []),
                parameter_ids=[],
                references=[r.ArchitectureModuleReference(kind="module", name=key)],
                attributes=[] if role is None else [role_attribute(self.b.producer, role)],
                provenance=self.b.producer.provenance() + [source_key(self.b.producer, key)],
                **kwargs,
            ),
            semantic_key=key,
        )

    def parameter(self, name: str, dims: tuple[int, ...], *, packed: bool = False) -> str:
        physical = self.inputs.bindings.physical
        pid = self.b.record_id("parameter", name)
        storage = [physical[name]] if name in physical else []
        binding: Literal["native", "quantized", "unresolved"] = "unresolved"
        reason: Literal["unresolved_binding", "unsupported_representation"] = "unresolved_binding"
        message = "Required parameter is absent or its geometry is incompatible."
        if packed and name.removesuffix(".weight") + ".qweight" in physical:
            prefix = name.removesuffix(".weight")
            out, inp = dims
            expected = {
                "qweight": ("I32", [inp // 8, out], "packed_data"),
                "qzeros": ("I32", [inp // 128, out // 8], "zero_points"),
                "scales": ("F16", [inp // 128, out], "scales"),
                "g_idx": ("I32", [inp], "group_indices"),
            }
            storage += [physical[f"{prefix}.{s}"] for s in expected if f"{prefix}.{s}" in physical]
            if (
                inp % 128 == 0
                and out % 8 == 0
                and name not in physical
                and all(
                    f"{prefix}.{s}" in physical
                    and physical[f"{prefix}.{s}"].dtype == dt
                    and physical[f"{prefix}.{s}"].shape == sh
                    for s, (dt, sh, _) in expected.items()
                )
            ):
                binding, reason = "quantized", "unsupported_representation"
                message = "No admitted complete GPTQ Int4 numeric tensor exists."
                storage = [
                    r.ArchitectureStorage(name=f"{prefix}.{s}", dtype=dt, shape=sh, role=role)
                    for s, (dt, sh, role) in expected.items()
                ]
        elif packed and name in self.numeric and self.numeric[name].storage_format == "bnb-nf4-dq":
            roles = {
                ".absmax": "scales",
                ".quant_map": "codebook",
                ".nested_absmax": "scales",
                ".nested_quant_map": "codebook",
                ".quant_state.bitsandbytes__nf4": "quantization_state",
            }
            storage = [physical[name].model_copy(update={"role": "packed_data"})]
            storage.extend(
                physical[name + suffix].model_copy(update={"role": role})
                for suffix, role in roles.items()
                if name + suffix in physical
            )
            binding, reason = "quantized", "unsupported_representation"
            message = "No admitted complete bitsandbytes NF4 numeric tensor exists."
        elif (
            storage
            and storage[0].shape == list(dims)
            and storage[0].dtype in {"F32", "F16", "BF16"}
        ):
            binding, reason = "native", "unsupported_representation"
            message = "No admitted complete native numeric tensor exists."
        inspection: r.ArchitectureInspection = r.ArchitectureUnavailableInspection(
            status="unavailable", reason=reason, message=message
        )
        numeric = self.numeric.get(name)
        if binding in ("native", "quantized") and numeric is not None:
            require(
                numeric.shape == dims and numeric.dtype == storage[0].dtype,
                "Numeric inventory disagrees with dense parameter geometry.",
            )
            inspection = r.ArchitectureAvailableInspection(status="available", tensor_id=numeric.id)
        param = r.ArchitectureDirectParameter(
            id=pid,
            name=name,
            logical_shape=shape(*dims),
            binding=binding,
            storage=storage,
            inspection=inspection,
            provenance=self.provenance(
                "hidden_size",
                "intermediate_size",
                "vocab_size",
                "num_attention_heads",
                "num_key_value_heads",
                "head_dim",
                "quantization_config",
            )
            + [r.ArchitectureProvenance(kind="storage", source=s.name) for s in storage],
        )
        self.parameters[name] = param
        self.b.add_parameter(param)
        self.used_storage.update(s.name for s in storage)
        if binding == "unresolved":
            self.b.diagnose(
                r.ArchitectureDiagnostic(
                    code="unresolved_binding", message=message, parameter_id=pid
                )
            )
        return pid

    def alias(self, name: str, target: str) -> str:
        native = self.parameters[target]
        pid = self.b.record_id("parameter", name)
        param = r.ArchitectureAliasParameter(
            id=pid,
            name=name,
            logical_shape=native.logical_shape,
            binding="alias",
            alias_of=native.id,
            storage=native.storage,
            inspection=native.inspection,
            provenance=self.provenance("tie_word_embeddings")
            + [r.ArchitectureProvenance(kind="storage", source=s.name) for s in native.storage],
        )
        self.parameters[name] = param
        self.b.add_parameter(param)
        return pid

    def norm(self, key: str, parent: str, dims: list[r.ArchitectureDimension], width: int) -> str:
        return self.node(
            key,
            parent,
            "rms_norm",
            {"x": dims},
            dims,
            parameters=(self.parameter(key + ".weight", (width,)),),
            attributes={"epsilon": self.c.epsilon, "axis": -1.0, "weight_offset": 0.0},
            formula="y = weight * x / sqrt(mean(x², axis=-1) + epsilon)",
            fields=("rms_norm_eps",),
        )

    def linear(
        self,
        key: str,
        parent: str,
        inp: int,
        out: int,
        bias: bool,
        *,
        weight_parameter: str | None = None,
        extra_attributes: dict[str, str | float | bool] | None = None,
        suppress_formula: bool = False,
    ) -> LinearNodes:
        weight = (
            weight_parameter
            if weight_parameter is not None
            else self.parameter(key + ".weight", (out, inp), packed=self.c.quantized)
        )
        params = [weight]
        if bias:
            params.append(self.parameter(key + ".bias", (out,)))
        base = self.node(
            key,
            parent,
            "linear",
            {"x": shape("B", "S", inp)},
            shape("B", "S", out),
            parameters=tuple(params),
            attributes={"bias": bias, **(extra_attributes or {})},
            formula=None
            if suppress_formula
            else ("y = x Wᵀ + bias" if bias else "y = x Wᵀ"),
        )
        target = self.lora_targets.get(key)
        if target is None:
            return LinearNodes(base, base)

        composition = self.lora
        assert composition is not None
        adapter_provenance = (
            r.ArchitectureProvenance(kind="configuration", source="adapter_config.json#/r"),
            r.ArchitectureProvenance(
                kind="configuration", source="adapter_config.json#/lora_alpha"
            ),
        )
        a_parameter = self.b.adapter_parameter(
            key + ".lora_A.weight",
            target.a_tensor_name,
            target.a_storage_name,
            shape(composition.rank, inp),
            self.provenance()
            + [r.ArchitectureProvenance(kind="configuration", source="adapter_config.json#/r")],
        )
        b_parameter = self.b.adapter_parameter(
            key + ".lora_B.weight",
            target.b_tensor_name,
            target.b_storage_name,
            shape(out, composition.rank),
            self.provenance()
            + [r.ArchitectureProvenance(kind="configuration", source="adapter_config.json#/r")],
        )
        a = self.node(
            key + ".lora_A",
            parent,
            "linear",
            {"x": shape("B", "S", inp)},
            shape("B", "S", composition.rank),
            parameters=(a_parameter,),
            attributes={"bias": False},
            formula="y = x Aᵀ",
            extra_provenance=adapter_provenance,
        )
        b = self.node(
            key + ".lora_B",
            parent,
            "linear",
            {"x": shape("B", "S", composition.rank)},
            shape("B", "S", out),
            parameters=(b_parameter,),
            attributes={"bias": False},
            formula="y = x Bᵀ",
            extra_provenance=adapter_provenance,
        )
        scale = self.node(
            key + ".lora_scale",
            parent,
            "scale",
            {"x": shape("B", "S", out)},
            shape("B", "S", out),
            attributes={
                "factor": composition.scale,
                "alpha": composition.alpha,
                "rank": float(composition.rank),
            },
            formula="y = (alpha / r) * x",
            extra_provenance=adapter_provenance,
        )
        residual = self.node(
            key + ".lora_add",
            parent,
            "add",
            {"base": shape("B", "S", out), "adapter": shape("B", "S", out)},
            shape("B", "S", out),
            formula="y = base + adapter",
            extra_provenance=adapter_provenance,
        )
        self.edge(a, b)
        self.edge(b, scale)
        self.edge(base, residual, "base")
        self.edge(scale, residual, "adapter")
        self.used_storage.update((target.a_storage_name, target.b_storage_name))
        return LinearNodes(base, residual, a)

    def connect_linear(
        self,
        source: str,
        linear: LinearNodes,
        *,
        source_port: str = "out",
        target_port: str = "x",
    ) -> str:
        self.edge(source, linear.base, target_port, source_port)
        if linear.adapter_input is not None:
            self.edge(source, linear.adapter_input, target_port, source_port)
        return linear.output

    def layer(self, index: int) -> str:
        c = self.c
        key = f"model.layers.{index}"
        hidden, rotary = shape("B", "S", c.hidden), shape("B", "S", c.head_dim)
        mask, scores = shape("B", 1, "S", "S"), shape("B", c.heads, "S", "S")
        norm = self.norm(key + ".input_layernorm", key, hidden, c.hidden)
        self.edge(key, norm, source_port="x")
        attention = key + ".self_attn"
        self.b.begin_template(attention, attention, "dense_attention", "attention")
        self.edge(norm, attention)
        for auxiliary in ("cos", "sin", "mask"):
            self.edge(key, attention, auxiliary, auxiliary)
        projected: dict[str, str] = {}
        for branch, heads in (("q", c.heads), ("k", c.kv_heads), ("v", c.kv_heads)):
            base = key + ".self_attn."
            linear = self.linear(
                base + branch + "_proj", attention, c.hidden, heads * c.head_dim, c.attention_bias
            )
            proj = self.connect_linear(attention, linear, source_port="x")
            split_shape = shape("B", "S", heads, c.head_dim)
            split = self.node(
                base + branch + "_heads",
                attention,
                "reshape",
                {"x": shape("B", "S", heads * c.head_dim)},
                split_shape,
            )
            self.edge(proj, split)
            if c.qwen and branch in {"q", "k"}:
                normalized = self.norm(base + branch + "_norm", attention, split_shape, c.head_dim)
                self.edge(split, normalized)
                split = normalized
            head_shape = shape("B", heads, "S", c.head_dim)
            trans = self.node(
                base + branch + "_transpose",
                attention,
                "transpose",
                {"x": split_shape},
                head_shape,
                attributes={"axes": "0,2,1,3"},
            )
            self.edge(split, trans)
            if branch in {"q", "k"}:
                rope = self.node(
                    base + branch + "_rotary",
                    attention,
                    "rotary_position",
                    {"x": head_shape, "cos": rotary, "sin": rotary},
                    head_shape,
                    formula="x * cos + rotate_half(x) * sin",
                )
                self.edge(trans, rope)
                self.edge(attention, rope, "cos", "cos")
                self.edge(attention, rope, "sin", "sin")
                trans = rope
            if branch in {"k", "v"}:
                repeat = self.node(
                    base + branch + "_repeat",
                    attention,
                    "repeat_kv",
                    {"x": head_shape},
                    shape("B", c.heads, "S", c.head_dim),
                    attributes={"groups": float(c.heads // c.kv_heads)},
                    fields=("num_attention_heads", "num_key_value_heads"),
                )
                self.edge(trans, repeat)
                trans = repeat
            projected[branch] = trans
        base = key + ".self_attn."
        heads_shape = shape("B", c.heads, "S", c.head_dim)
        kt = self.node(
            base + "key_transpose",
            attention,
            "transpose",
            {"x": heads_shape},
            shape("B", c.heads, c.head_dim, "S"),
            attributes={"axes": "0,1,3,2"},
        )
        self.edge(projected["k"], kt)
        qk = self.node(
            base + "qk_product",
            attention,
            "matmul",
            {"q": heads_shape, "kt": shape("B", c.heads, c.head_dim, "S")},
            scores,
        )
        self.edge(projected["q"], qk, "q")
        self.edge(kt, qk, "kt")
        scale = self.node(
            base + "scale",
            attention,
            "scale",
            {"x": scores},
            scores,
            attributes={"factor": c.head_dim**-0.5},
            fields=("head_dim",),
        )
        self.edge(qk, scale)
        masked = self.node(
            base + "mask", attention, "add_mask", {"x": scores, "mask": mask}, scores
        )
        self.edge(scale, masked)
        self.edge(attention, masked, "mask", "mask")
        softmax = self.node(
            base + "softmax", attention, "softmax", {"x": scores}, scores, attributes={"axis": -1.0}
        )
        self.edge(masked, softmax)
        av = self.node(
            base + "value_product",
            attention,
            "matmul",
            {"probabilities": scores, "v": heads_shape},
            heads_shape,
        )
        self.edge(softmax, av, "probabilities")
        self.edge(projected["v"], av, "v")
        trans = self.node(
            base + "output_transpose",
            attention,
            "transpose",
            {"x": heads_shape},
            shape("B", "S", c.heads, c.head_dim),
            attributes={"axes": "0,2,1,3"},
        )
        self.edge(av, trans)
        merge = self.node(
            base + "merge_heads",
            attention,
            "reshape",
            {"x": shape("B", "S", c.heads, c.head_dim)},
            shape("B", "S", c.heads * c.head_dim),
        )
        self.edge(trans, merge)
        linear = self.linear(
            base + "o_proj", attention, c.heads * c.head_dim, c.hidden, c.attention_bias
        )
        out = self.connect_linear(merge, linear)
        self.edge(out, attention, "out")
        self.group(
            attention,
            key,
            [
                self.port("x", hidden),
                self.port("cos", rotary),
                self.port("sin", rotary),
                self.port("mask", mask),
                self.port("out", hidden, True),
            ],
            role="attention",
        )
        residual = self.node(
            key + ".attention_residual", key, "add", {"skip": hidden, "branch": hidden}, hidden
        )
        self.edge(key, residual, "skip", "x")
        self.edge(attention, residual, "branch")
        post = self.norm(key + ".post_attention_layernorm", key, hidden, c.hidden)
        self.edge(residual, post)
        mlp = key + ".mlp"
        self.b.begin_template(mlp, mlp, "gated_mlp", "mlp")
        self.edge(post, mlp)
        gate = self.connect_linear(
            mlp,
            self.linear(key + ".mlp.gate_proj", mlp, c.hidden, c.intermediate, c.mlp_bias),
            source_port="x",
        )
        up = self.connect_linear(
            mlp,
            self.linear(key + ".mlp.up_proj", mlp, c.hidden, c.intermediate, c.mlp_bias),
            source_port="x",
        )
        intermediate = shape("B", "S", c.intermediate)
        act = self.node(
            key + ".mlp.activation",
            mlp,
            "silu",
            {"x": intermediate},
            intermediate,
            formula="silu(x) = x * sigmoid(x)",
            fields=("hidden_act",),
        )
        self.edge(gate, act)
        multiply = self.node(
            key + ".mlp.multiply",
            mlp,
            "multiply",
            {"gate": intermediate, "up": intermediate},
            intermediate,
        )
        self.edge(act, multiply, "gate")
        self.edge(up, multiply, "up")
        down = self.connect_linear(
            multiply,
            self.linear(key + ".mlp.down_proj", mlp, c.intermediate, c.hidden, c.mlp_bias),
        )
        self.edge(down, mlp, "out")
        self.group(mlp, key, [self.port("x", hidden), self.port("out", hidden, True)], role="mlp")
        final = self.node(
            key + ".mlp_residual", key, "add", {"skip": hidden, "branch": hidden}, hidden
        )
        self.edge(residual, final, "skip")
        self.edge(mlp, final, "branch")
        self.edge(final, key, "out")
        self.group(
            key,
            "model",
            [
                self.port("x", hidden),
                self.port("cos", rotary),
                self.port("sin", rotary),
                self.port("mask", mask),
                self.port("out", hidden, True),
            ],
        )
        return key

    def build(self) -> None:
        c, b = self.c, self.b
        b.add_symbol("B", "Symbolic batch size; no input has been executed.")
        b.add_symbol("S", "Symbolic full input sequence length; no captured KV cache.")
        tokens = self.node("input_ids", "model", "token_ids", {}, shape("B", "S"), kind="input")
        positions = self.node(
            "position_ids", "model", "positions", {}, shape("B", "S"), kind="input"
        )
        mask = self.node(
            "attention_mask",
            "model",
            "causal_mask",
            {},
            shape("B", 1, "S", "S"),
            kind="input",
            attributes={"causal": True},
        )
        # A saved tied checkpoint may retain either name. Two stored copies do not
        # prove identical values without reading weights, so preserve uncertainty.
        embed_name, head_name = "model.embed_tokens.weight", "lm_head.weight"
        physical = self.inputs.bindings.physical
        if c.tied and embed_name not in physical and head_name in physical:
            head = self.parameter(head_name, (c.vocab, c.hidden))
            embed = self.alias(embed_name, head_name)
        else:
            embed = self.parameter(embed_name, (c.vocab, c.hidden))
            if c.tied and head_name not in physical:
                head = self.alias(head_name, embed_name)
            else:
                head = self.parameter(head_name, (c.vocab, c.hidden))
                if c.tied:
                    b.diagnose(
                        r.ArchitectureDiagnostic(
                            code="unverified_tie",
                            message=(
                                "Both tied names have stored tensors; "
                                "metadata cannot establish value identity."
                            ),
                            parameter_id=head,
                        )
                    )
        embedded = self.node(
            "model.embed_tokens",
            "model",
            "embedding",
            {"x": shape("B", "S")},
            shape("B", "S", c.hidden),
            parameters=(embed,),
        )
        self.edge(tokens, embedded)
        rotary = shape("B", "S", c.head_dim)
        cos = self.node(
            "model.rotary_cos",
            "model",
            "rotary_cos",
            {"positions": shape("B", "S")},
            rotary,
            attributes={
                "theta": c.theta,
                "max_positions": float(c.max_positions),
                "layout": "split_half",
            },
            formula="cos(concat(f, f)); f = position * theta^(-2i/head_dim)",
            fields=("rope_theta", "head_dim", "max_position_embeddings"),
        )
        sin = self.node(
            "model.rotary_sin",
            "model",
            "rotary_sin",
            {"positions": shape("B", "S")},
            rotary,
            attributes={"theta": c.theta, "layout": "split_half"},
            formula="sin(concat(f, f)); f = position * theta^(-2i/head_dim)",
            fields=("rope_theta", "head_dim"),
        )
        self.edge(positions, cos, "positions")
        self.edge(positions, sin, "positions")
        previous = embedded
        instances = []
        for index in range(c.layers):
            layer = self.layer(index)
            self.edge(previous, layer)
            self.edge(cos, layer, "cos")
            self.edge(sin, layer, "sin")
            self.edge(mask, layer, "mask")
            instances.append(
                r.ArchitectureRepetitionInstance(
                    node_id=self.nid(layer),
                    index=index,
                    variant="qwen3_dense" if c.qwen else "smollm2_dense",
                )
            )
            previous = layer
        norm = self.norm("model.norm", "model", shape("B", "S", c.hidden), c.hidden)
        self.edge(previous, norm)
        output = self.connect_linear(
            norm,
            self.linear(
                "lm_head",
                "model",
                c.hidden,
                c.vocab,
                False,
                weight_parameter=head,
                extra_attributes={"mode": "evaluation; full-sequence static architecture"},
                suppress_formula=True,
            ),
        )
        logits = self.node(
            "logits",
            "model",
            "logits",
            {"x": shape("B", "S", c.vocab)},
            shape("B", "S", c.vocab),
            kind="output",
        )
        self.edge(output, logits)
        self.group("model", None, [])
        b.add_repetition(
            r.ArchitectureRepetition(
                id=b.record_id("repetition", "layers"),
                parent_id=self.nid("model"),
                label="Decoder layers",
                instances=instances,
            )
        )
        if self.inputs.bindings.tokenizer_available:
            b.add_node(
                r.ArchitectureLeafNode(
                    id=self.nid("tokenizer"),
                    kind="context",
                    label="Tokenizer",
                    ports=[],
                    parameter_ids=[],
                    references=[r.ArchitectureTokenizerReference(kind="tokenizer")],
                    attributes=[],
                    provenance=b.producer.provenance(),
                )
            )
        if set(physical) - self.used_storage:
            b.diagnose(
                r.ArchitectureDiagnostic(
                    code="unrecognized_storage",
                    message="Checkpoint contains storage outside the reviewed dense description.",
                )
            )


def build(inputs: AnalysisInput, builder: GraphBuilder) -> None:
    config = checked(inputs.configuration)
    require(config is not None, "Unsupported dense configuration.")
    assert config is not None
    DenseGraph(inputs, builder, config).build()


def register_dense_descriptions(registry: DescriptionRegistry) -> None:
    """Register with the caller-owned registry; startup/cache/HTTP stay outside this module."""
    for family, architecture, name in (
        ("qwen3", "Qwen3ForCausalLM", "qwen3-dense"),
        ("llama", "LlamaForCausalLM", "smollm2-dense"),
    ):
        registry.register(
            Description(
                Producer(name, "3", SOURCE_REVISION),
                "language_model",
                frozenset({family}),
                frozenset({architecture}),
                lambda inputs: (
                    inputs.lora_composition is None and checked(inputs.configuration) is not None
                ),
                build,
            )
        )
        registry.register(
            Description(
                Producer(name + "-lora", "1", SOURCE_REVISION),
                "language_model",
                frozenset({family}),
                frozenset({architecture}),
                lambda inputs: (
                    inputs.lora_composition is not None
                    and checked(inputs.configuration) is not None
                ),
                build,
            )
        )

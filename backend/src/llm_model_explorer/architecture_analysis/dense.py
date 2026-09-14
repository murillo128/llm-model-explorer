"""Packaged static dense-language descriptions; never an inference implementation."""

from typing import Literal

from . import records as r
from .core import AnalysisInput, Description, DescriptionRegistry, GraphBuilder, Producer
from .dense_config import SOURCE_REVISION, DenseConfig, checked
from .validation import require


def shape(*dims: int | str) -> list[r.ArchitectureDimension]:
    return [
        r.ArchitectureConstantDimension(kind="constant", value=d)
        if isinstance(d, int)
        else r.ArchitectureSymbolDimension(kind="symbol", name=d)
        for d in dims
    ]


class DenseGraph:
    """Shared fragments only where reviewed Qwen3/Llama mathematics agrees."""

    def __init__(self, inputs: AnalysisInput, builder: GraphBuilder, config: DenseConfig):
        self.inputs, self.b, self.c = inputs, builder, config
        self.children: dict[str, list[str]] = {}
        self.parameters: dict[str, r.ArchitectureParameter] = {}
        self.used_storage: set[str] = set()
        self.numeric = {t.name: t for t in inputs.bindings.numeric.values()}

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
                label=key,
                operation=operation,
                ports=[self.port(k, v) for k, v in inputs.items()]
                + [self.port("out", output, True)],
                parameter_ids=list(parameters),
                references=references,
                attributes=[
                    r.ArchitectureAttribute(name=k, value=v, provenance=self.provenance(*fields))
                    for k, v in (attributes or {}).items()
                ],
                provenance=self.provenance(*fields),
                **kwargs,
            )
        )
        return key

    def group(self, key: str, parent: str | None, ports: list[r.ArchitecturePort]) -> None:
        kwargs = {} if parent is None else {"parent_id": self.nid(parent)}
        if parent is not None:
            self.children.setdefault(parent, []).append(self.nid(key))
        self.b.add_node(
            r.ArchitectureGroupNode(
                id=self.nid(key),
                kind="group",
                label=key,
                ports=ports,
                children=self.children.get(key, []),
                parameter_ids=[],
                references=[r.ArchitectureModuleReference(kind="module", name=key)],
                attributes=[],
                provenance=self.b.producer.provenance(),
                **kwargs,
            )
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
                message = "GPTQ Int4 decoding is not available; storage is metadata only."
                storage = [
                    r.ArchitectureStorage(name=f"{prefix}.{s}", dtype=dt, shape=sh, role=role)
                    for s, (dt, sh, role) in expected.items()
                ]
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
        if binding == "native" and numeric is not None:
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

    def linear(self, key: str, parent: str, inp: int, out: int, bias: bool) -> str:
        params = [self.parameter(key + ".weight", (out, inp), packed=self.c.quantized)]
        if bias:
            params.append(self.parameter(key + ".bias", (out,)))
        return self.node(
            key,
            parent,
            "linear",
            {"x": shape("B", "S", inp)},
            shape("B", "S", out),
            parameters=tuple(params),
            attributes={"bias": bias},
            formula="y = x Wᵀ + bias" if bias else "y = x Wᵀ",
        )

    def layer(self, index: int) -> str:
        c = self.c
        key = f"model.layers.{index}"
        hidden, rotary = shape("B", "S", c.hidden), shape("B", "S", c.head_dim)
        mask, scores = shape("B", 1, "S", "S"), shape("B", c.heads, "S", "S")
        norm = self.norm(key + ".input_layernorm", key, hidden, c.hidden)
        self.edge(key, norm, source_port="x")
        projected: dict[str, str] = {}
        for branch, heads in (("q", c.heads), ("k", c.kv_heads), ("v", c.kv_heads)):
            base = key + ".self_attn."
            proj = self.linear(
                base + branch + "_proj", key, c.hidden, heads * c.head_dim, c.attention_bias
            )
            self.edge(norm, proj)
            split_shape = shape("B", "S", heads, c.head_dim)
            split = self.node(
                base + branch + "_heads",
                key,
                "reshape",
                {"x": shape("B", "S", heads * c.head_dim)},
                split_shape,
            )
            self.edge(proj, split)
            if c.qwen and branch in {"q", "k"}:
                normalized = self.norm(base + branch + "_norm", key, split_shape, c.head_dim)
                self.edge(split, normalized)
                split = normalized
            head_shape = shape("B", heads, "S", c.head_dim)
            trans = self.node(
                base + branch + "_transpose",
                key,
                "transpose",
                {"x": split_shape},
                head_shape,
                attributes={"axes": "0,2,1,3"},
            )
            self.edge(split, trans)
            if branch in {"q", "k"}:
                rope = self.node(
                    base + branch + "_rotary",
                    key,
                    "rotary_position",
                    {"x": head_shape, "cos": rotary, "sin": rotary},
                    head_shape,
                    formula="x * cos + rotate_half(x) * sin",
                )
                self.edge(trans, rope)
                self.edge(key, rope, "cos", "cos")
                self.edge(key, rope, "sin", "sin")
                trans = rope
            if branch in {"k", "v"}:
                repeat = self.node(
                    base + branch + "_repeat",
                    key,
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
            key,
            "transpose",
            {"x": heads_shape},
            shape("B", c.heads, c.head_dim, "S"),
            attributes={"axes": "0,1,3,2"},
        )
        self.edge(projected["k"], kt)
        qk = self.node(
            base + "qk_product",
            key,
            "matmul",
            {"q": heads_shape, "kt": shape("B", c.heads, c.head_dim, "S")},
            scores,
        )
        self.edge(projected["q"], qk, "q")
        self.edge(kt, qk, "kt")
        scale = self.node(
            base + "scale",
            key,
            "scale",
            {"x": scores},
            scores,
            attributes={"factor": c.head_dim**-0.5},
            fields=("head_dim",),
        )
        self.edge(qk, scale)
        masked = self.node(base + "mask", key, "add_mask", {"x": scores, "mask": mask}, scores)
        self.edge(scale, masked)
        self.edge(key, masked, "mask", "mask")
        softmax = self.node(
            base + "softmax", key, "softmax", {"x": scores}, scores, attributes={"axis": -1.0}
        )
        self.edge(masked, softmax)
        av = self.node(
            base + "value_product",
            key,
            "matmul",
            {"probabilities": scores, "v": heads_shape},
            heads_shape,
        )
        self.edge(softmax, av, "probabilities")
        self.edge(projected["v"], av, "v")
        trans = self.node(
            base + "output_transpose",
            key,
            "transpose",
            {"x": heads_shape},
            shape("B", "S", c.heads, c.head_dim),
            attributes={"axes": "0,2,1,3"},
        )
        self.edge(av, trans)
        merge = self.node(
            base + "merge_heads",
            key,
            "reshape",
            {"x": shape("B", "S", c.heads, c.head_dim)},
            shape("B", "S", c.heads * c.head_dim),
        )
        self.edge(trans, merge)
        out = self.linear(base + "o_proj", key, c.heads * c.head_dim, c.hidden, c.attention_bias)
        self.edge(merge, out)
        residual = self.node(
            key + ".attention_residual", key, "add", {"skip": hidden, "branch": hidden}, hidden
        )
        self.edge(key, residual, "skip", "x")
        self.edge(out, residual, "branch")
        post = self.norm(key + ".post_attention_layernorm", key, hidden, c.hidden)
        self.edge(residual, post)
        gate = self.linear(key + ".mlp.gate_proj", key, c.hidden, c.intermediate, c.mlp_bias)
        up = self.linear(key + ".mlp.up_proj", key, c.hidden, c.intermediate, c.mlp_bias)
        self.edge(post, gate)
        self.edge(post, up)
        intermediate = shape("B", "S", c.intermediate)
        act = self.node(
            key + ".mlp.activation",
            key,
            "silu",
            {"x": intermediate},
            intermediate,
            formula="silu(x) = x * sigmoid(x)",
            fields=("hidden_act",),
        )
        self.edge(gate, act)
        multiply = self.node(
            key + ".mlp.multiply",
            key,
            "multiply",
            {"gate": intermediate, "up": intermediate},
            intermediate,
        )
        self.edge(act, multiply, "gate")
        self.edge(up, multiply, "up")
        down = self.linear(key + ".mlp.down_proj", key, c.intermediate, c.hidden, c.mlp_bias)
        self.edge(multiply, down)
        final = self.node(
            key + ".mlp_residual", key, "add", {"skip": hidden, "branch": hidden}, hidden
        )
        self.edge(residual, final, "skip")
        self.edge(down, final, "branch")
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
        output = self.node(
            "lm_head",
            "model",
            "linear",
            {"x": shape("B", "S", c.hidden)},
            shape("B", "S", c.vocab),
            parameters=(head,),
            attributes={"bias": False, "mode": "evaluation; full-sequence static architecture"},
        )
        self.edge(norm, output)
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
                Producer(name, "1", SOURCE_REVISION),
                "language_model",
                frozenset({family}),
                frozenset({architecture}),
                lambda inputs: checked(inputs.configuration) is not None,
                build,
            )
        )

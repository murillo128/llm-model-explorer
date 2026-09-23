"""Reviewed static DeepSeek-V2-Lite architecture; never imports checkpoint code."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

from . import records as r
from .core import AnalysisInput, Description, DescriptionRegistry, GraphBuilder, Producer
from .semantic import operation_role
from .validation import require

SOURCE_REVISION = "slowfastai/DeepSeek-V2-Lite-bnb-4bit@9fc357346aeba67950a34a86f3520fc276ca2daf"
PRODUCER = Producer("deepseek-v2-lite", "1", SOURCE_REVISION)

REFERENCE = {
    "model_type": "deepseek_v2",
    "hidden_size": 2048,
    "intermediate_size": 10944,
    "vocab_size": 102400,
    "num_hidden_layers": 27,
    "num_attention_heads": 16,
    "num_key_value_heads": 16,
    "first_k_dense_replace": 1,
    "moe_layer_freq": 1,
    "moe_intermediate_size": 1408,
    "n_routed_experts": 64,
    "num_experts_per_tok": 6,
    "n_shared_experts": 2,
    "kv_lora_rank": 512,
    "q_lora_rank": None,
    "qk_nope_head_dim": 128,
    "qk_rope_head_dim": 64,
    "v_head_dim": 128,
    "max_position_embeddings": 163840,
    "rms_norm_eps": 1e-6,
    "rope_theta": 10000,
    "attention_bias": False,
    "attention_dropout": 0.0,
    "mlp_bias": False,
    "hidden_act": "silu",
    "pretraining_tp": 1,
    "tie_word_embeddings": False,
    "bos_token_id": 100000,
    "eos_token_id": 100001,
    "initializer_range": 0.02,
    "use_cache": True,
    "n_group": 1,
    "topk_group": 1,
    "topk_method": "greedy",
    "scoring_func": "softmax",
    "norm_topk_prob": False,
    "routed_scaling_factor": 1.0,
    "seq_aux": True,
    "aux_loss_alpha": 0.001,
}
ROPE_SCALING = {
    "type": "yarn",
    "factor": 40,
    "original_max_position_embeddings": 4096,
    "beta_fast": 32,
    "beta_slow": 1,
    "mscale": 0.707,
    "mscale_all_dim": 0.707,
}
ARCHITECTURE = "DeepseekV2ForCausalLM"
AUTO_MAP_SUFFIXES = {
    "AutoConfig": "configuration_deepseek.DeepseekV2Config",
    "AutoModel": "modeling_deepseek.DeepseekV2Model",
    "AutoModelForCausalLM": "modeling_deepseek.DeepseekV2ForCausalLM",
}
BNB_CONFIG = {
    "_load_in_4bit": True,
    "_load_in_8bit": False,
    "bnb_4bit_compute_dtype": "bfloat16",
    "bnb_4bit_quant_storage": "uint8",
    "bnb_4bit_quant_type": "nf4",
    "bnb_4bit_use_double_quant": True,
    "llm_int8_enable_fp32_cpu_offload": False,
    "llm_int8_has_fp16_weight": False,
    "llm_int8_skip_modules": None,
    "llm_int8_threshold": 6.0,
    "load_in_4bit": True,
    "load_in_8bit": False,
    "quant_method": "bitsandbytes",
}
METADATA = {
    "_commit_hash",
    "_name_or_path",
    "dtype",
    "torch_dtype",
    "transformers_version",
}


def checked(configuration: Mapping[str, object]) -> dict[str, Any] | None:
    """Accept the reviewed reference structure and harmless reference metadata only."""
    allowed = (
        set(REFERENCE)
        | METADATA
        | {
            "architectures",
            "auto_map",
            "rope_scaling",
            "ep_size",
            "quantization_config",
        }
    )
    if set(configuration) - allowed:
        return None
    if configuration.get("architectures") != [ARCHITECTURE]:
        return None
    for key, expected in REFERENCE.items():
        actual = configuration.get(key, False if key == "mlp_bias" else None)
        if type(actual) is not type(expected) or actual != expected:
            return None
    scaling = configuration.get("rope_scaling")
    if not isinstance(scaling, dict) or set(scaling) != set(ROPE_SCALING):
        return None
    if any(
        type(scaling[key]) is not type(expected) or scaling[key] != expected
        for key, expected in ROPE_SCALING.items()
    ):
        return None
    if type(configuration.get("rope_theta")) is not int:
        return None
    if type(configuration.get("rms_norm_eps")) is not float:
        return None
    if type(configuration.get("attention_dropout")) is not float:
        return None
    if type(configuration.get("routed_scaling_factor")) is not float:
        return None
    if type(configuration.get("aux_loss_alpha")) is not float:
        return None
    if configuration.get("ep_size", 1) != 1 or type(configuration.get("ep_size", 1)) is not int:
        return None
    if not _valid_auto_map(configuration.get("auto_map")):
        return None
    quantization = configuration.get("quantization_config")
    if quantization is not None and not _valid_bnb_config(quantization):
        return None
    for key in METADATA:
        value = configuration.get(key)
        if value is not None and not isinstance(value, str):
            return None
    result = dict(configuration)
    result["mlp_bias"] = False  # Reviewed DeepseekV2Config default; absent in both references.
    return result


def _valid_auto_map(value: object) -> bool:
    if value is None:
        return True
    if not isinstance(value, dict) or set(value) != set(AUTO_MAP_SUFFIXES):
        return False
    return all(
        isinstance(value[key], str) and value[key].endswith(suffix)
        for key, suffix in AUTO_MAP_SUFFIXES.items()
    )


def _valid_bnb_config(value: object) -> bool:
    if not isinstance(value, dict) or set(value) != set(BNB_CONFIG):
        return False
    return all(
        type(value[key]) is type(expected) and value[key] == expected
        for key, expected in BNB_CONFIG.items()
    )


def _required_int(configuration: Mapping[str, object], key: str) -> int:
    value = configuration.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"Expected integer DeepSeek-V2 option: {key}.")
    return value


def parameter_shapes(configuration: Mapping[str, object]) -> dict[str, tuple[int, ...]]:
    """Logical bindings for the verified DeepSeek-V2-Lite CausalLM path."""
    hidden = _required_int(configuration, "hidden_size")
    vocab = _required_int(configuration, "vocab_size")
    heads = _required_int(configuration, "num_attention_heads")
    qk_width = _required_int(configuration, "qk_nope_head_dim") + _required_int(
        configuration, "qk_rope_head_dim"
    )
    value_width = _required_int(configuration, "v_head_dim")
    kv_rank = _required_int(configuration, "kv_lora_rank")
    expert_width = _required_int(configuration, "moe_intermediate_size")
    shared_width = expert_width * _required_int(configuration, "n_shared_experts")
    result = {
        "model.embed_tokens.weight": (vocab, hidden),
        "model.norm.weight": (hidden,),
        "lm_head.weight": (vocab, hidden),
    }
    for index in range(_required_int(configuration, "num_hidden_layers")):
        layer = f"model.layers.{index}"
        result.update(
            {
                f"{layer}.input_layernorm.weight": (hidden,),
                f"{layer}.post_attention_layernorm.weight": (hidden,),
                f"{layer}.self_attn.q_proj.weight": (heads * qk_width, hidden),
                f"{layer}.self_attn.kv_a_proj_with_mqa.weight": (
                    kv_rank + _required_int(configuration, "qk_rope_head_dim"),
                    hidden,
                ),
                f"{layer}.self_attn.kv_a_layernorm.weight": (kv_rank,),
                f"{layer}.self_attn.kv_b_proj.weight": (
                    heads * (_required_int(configuration, "qk_nope_head_dim") + value_width),
                    kv_rank,
                ),
                f"{layer}.self_attn.o_proj.weight": (hidden, heads * value_width),
            }
        )
        mlp = f"{layer}.mlp"
        if index == 0:
            width = _required_int(configuration, "intermediate_size")
            result.update(
                {
                    f"{mlp}.gate_proj.weight": (width, hidden),
                    f"{mlp}.up_proj.weight": (width, hidden),
                    f"{mlp}.down_proj.weight": (hidden, width),
                }
            )
            continue
        result[f"{mlp}.gate.weight"] = (
            _required_int(configuration, "n_routed_experts"),
            hidden,
        )
        for expert in range(_required_int(configuration, "n_routed_experts")):
            prefix = f"{mlp}.experts.{expert}"
            result.update(
                {
                    f"{prefix}.gate_proj.weight": (expert_width, hidden),
                    f"{prefix}.up_proj.weight": (expert_width, hidden),
                    f"{prefix}.down_proj.weight": (hidden, expert_width),
                }
            )
        shared = f"{mlp}.shared_experts"
        result.update(
            {
                f"{shared}.gate_proj.weight": (shared_width, hidden),
                f"{shared}.up_proj.weight": (shared_width, hidden),
                f"{shared}.down_proj.weight": (hidden, shared_width),
            }
        )
    return result


def supports(inputs: AnalysisInput) -> bool:
    return checked(inputs.configuration) is not None


def shape(*dimensions: int | str | r.ArchitectureDimension) -> r.ArchitectureShape:
    result: list[r.ArchitectureDimension] = []
    for dimension in dimensions:
        if isinstance(
            dimension,
            (
                r.ArchitectureConstantDimension,
                r.ArchitectureSymbolDimension,
                r.ArchitectureExpressionDimension,
                r.ArchitectureUnknownDimension,
            ),
        ):
            result.append(dimension)
        elif isinstance(dimension, int):
            result.append(r.ArchitectureConstantDimension(kind="constant", value=dimension))
        else:
            result.append(r.ArchitectureSymbolDimension(kind="symbol", name=dimension))
    return result


def _expression(text: str, *symbols: str) -> r.ArchitectureExpressionDimension:
    return r.ArchitectureExpressionDimension(kind="expression", text=text, symbols=list(symbols))


@dataclass(frozen=True)
class Value:
    node: str
    port: str
    shape: r.ArchitectureShape
    kind: Literal["data", "state"] = "data"


class DeepseekGraph:
    def __init__(self, inputs: AnalysisInput, builder: GraphBuilder, configuration: dict[str, Any]):
        self.inputs, self.b, self.c = inputs, builder, configuration
        self.children: dict[str, list[str]] = {}
        self.parameter_ids: dict[str, str] = {}
        self.used_storage: set[str] = set()
        self.physical = inputs.bindings.physical
        self.numeric = {tensor.name: tensor for tensor in inputs.bindings.numeric.values()}
        self.shapes = parameter_shapes(configuration)
        for name, dims in self.shapes.items():
            self.parameter_ids[name] = self.parameter(name, dims)

    def nid(self, key: str) -> str:
        return self.b.record_id("node", key)

    def provenance(self, *fields: str) -> list[r.ArchitectureProvenance]:
        return [
            r.ArchitectureProvenance(
                kind="description",
                source=self.b.producer.description,
                revision=self.b.producer.revision,
            )
        ] + [
            r.ArchitectureProvenance(kind="configuration", source=f"config.json#/{field}")
            for field in fields
            if field in self.inputs.configuration
        ]

    def port(
        self, name: str, dimensions: r.ArchitectureShape, direction: Literal["input", "output"]
    ) -> r.ArchitecturePort:
        return r.ArchitecturePort(id=name, direction=direction, label=name, shape=dimensions)

    def role_attribute(self, role: str) -> r.ArchitectureAttribute:
        return r.ArchitectureAttribute(
            name="semantic_role", value=role, provenance=self.provenance()
        )

    def source_key(self, key: str) -> r.ArchitectureProvenance:
        return r.ArchitectureProvenance(
            kind="description", source=key, revision=self.b.producer.revision
        )

    def link(self, value: Value, target: str, port: str) -> None:
        self.b.add_edge(
            r.ArchitectureEdge(
                id=self.b.record_id("edge", f"{value.node}:{value.port}>{target}:{port}"),
                source=r.ArchitectureEndpoint(node_id=self.nid(value.node), port_id=value.port),
                target=r.ArchitectureEndpoint(node_id=self.nid(target), port_id=port),
                kind=value.kind,
                # The endpoint nodes carry the full pinned-source provenance. Keep the
                # repeated edge records compact while retaining their description identity.
                provenance=[
                    r.ArchitectureProvenance(
                        kind="description",
                        source=self.b.producer.description,
                        revision=self.b.producer.revision,
                    )
                ],
            )
        )

    def _parameter_storage(self, name: str) -> list[r.ArchitectureStorage]:
        prefix = name.removesuffix(".weight")
        selected = [
            tensor
            for storage_name, tensor in self.physical.items()
            if storage_name == name or storage_name.startswith(prefix + ".")
        ]
        return [
            tensor.model_copy(
                update={
                    "role": "packed_weight" if tensor.name == name else "quantization_auxiliary"
                }
            )
            for tensor in sorted(selected, key=lambda value: value.name)
        ]

    def parameter(self, name: str, dims: tuple[int, ...]) -> str:
        storage = self.physical.get(name)
        numeric = self.numeric.get(name)
        bnb = self.c.get("quantization_config") is not None
        binding: Literal["native", "quantized", "unresolved"] = "unresolved"
        parameter_storage: list[r.ArchitectureStorage] = []
        inspection: r.ArchitectureInspection = r.ArchitectureUnavailableInspection(
            status="unavailable",
            reason="unresolved_binding",
            message="Required parameter storage is absent.",
        )
        expected_shape = list(dims)

        if (
            storage is not None
            and list(storage.shape) == expected_shape
            and storage.dtype in {"F32", "F16", "BF16"}
        ):
            binding = "native"
            parameter_storage = [storage]
            self.used_storage.add(name)
            if numeric is None:
                inspection = r.ArchitectureUnavailableInspection(
                    status="unavailable",
                    reason="unsupported_representation",
                    message="No admitted complete native numeric view exists.",
                )
            else:
                require(
                    numeric.shape == dims and numeric.dtype == storage.dtype,
                    "Native numeric inventory disagrees with parameter geometry.",
                )
                inspection = r.ArchitectureAvailableInspection(
                    status="available", tensor_id=numeric.id
                )
        elif (
            storage is not None
            and bnb
            and name.endswith(".weight")
            and storage.dtype == "U8"
            and tuple(storage.shape) == ((math.prod(dims) + 1) // 2, 1)
        ):
            # Model admission validates the NF4 group. This descriptor retains its physical
            # records while keeping architecture completeness separate from numeric decoding.
            binding = "quantized"
            parameter_storage = self._parameter_storage(name)
            self.used_storage.update(value.name for value in parameter_storage)
            inspection = r.ArchitectureUnavailableInspection(
                status="unavailable",
                reason="unsupported_representation",
                message=(
                    "The admitted packed weight has no logical inspection view in this analyzer."
                ),
            )
        elif storage is not None:
            parameter_storage = [storage]
            self.used_storage.add(name)
            inspection = r.ArchitectureUnavailableInspection(
                status="unavailable",
                reason="unresolved_binding",
                message="Stored tensor geometry is incompatible with the reviewed parameter.",
            )
        else:
            inspection = r.ArchitectureUnavailableInspection(
                status="unavailable",
                reason="unresolved_binding",
                message="Required parameter storage is absent.",
            )

        parameter_id = self.b.record_id("parameter", name)
        provenance = self.provenance(
            "hidden_size",
            "intermediate_size",
            "vocab_size",
            "num_hidden_layers",
            "num_attention_heads",
            "qk_nope_head_dim",
            "qk_rope_head_dim",
            "kv_lora_rank",
            "moe_intermediate_size",
            "n_routed_experts",
            "quantization_config",
        ) + [
            r.ArchitectureProvenance(kind="storage", source=item.name) for item in parameter_storage
        ]
        self.b.add_parameter(
            r.ArchitectureDirectParameter(
                id=parameter_id,
                name=name,
                logical_shape=shape(*dims),
                binding=binding,
                storage=parameter_storage,
                inspection=inspection,
                provenance=provenance,
            )
        )
        if binding == "unresolved":
            self.b.diagnose(
                r.ArchitectureDiagnostic(
                    code="unresolved_binding",
                    message="A required parameter does not bind to verified storage geometry.",
                    parameter_id=parameter_id,
                )
            )
        return parameter_id

    def node(
        self,
        key: str,
        operation: str,
        inputs: dict[str, Value],
        outputs: dict[str, r.ArchitectureShape],
        *,
        parent: str,
        parameters: tuple[str, ...] = (),
        attributes: dict[str, str | float | bool] | None = None,
        fields: tuple[str, ...] = (),
        formula: str | None = None,
        module_name: str | None = None,
        kind: Literal["operation", "input", "output", "context", "state"] = "operation",
        state_outputs: tuple[str, ...] = (),
    ) -> dict[str, Value]:
        node_id = self.nid(key)
        self.children.setdefault(parent, []).append(node_id)
        parameter_ids = [self.parameter_ids[name] for name in parameters]
        references: list[r.ArchitectureReference] = [
            r.ArchitectureParameterReference(kind="parameter", parameter_id=value)
            for value in parameter_ids
        ]
        if parameters:
            references.insert(
                0,
                r.ArchitectureModuleReference(kind="module", name=module_name or key),
            )
        role = operation_role(key, operation)
        kwargs: dict[str, Any] = {"formula": formula} if formula else {}
        self.b.add_node(
            r.ArchitectureLeafNode(
                id=node_id,
                parent_id=self.nid(parent),
                kind=kind,
                label=role.replace("_", " "),
                operation=operation,
                ports=[self.port(name, value.shape, "input") for name, value in inputs.items()]
                + [self.port(name, value, "output") for name, value in outputs.items()],
                parameter_ids=parameter_ids,
                references=references,
                attributes=[
                    r.ArchitectureAttribute(
                        name=name, value=value, provenance=self.provenance(*fields)
                    )
                    for name, value in (attributes or {}).items()
                ]
                + [self.role_attribute(role)],
                provenance=self.provenance(*fields) + [self.source_key(key)],
                **kwargs,
            ),
            semantic_key=key,
        )
        for name, value in inputs.items():
            self.link(value, key, name)
        return {
            name: Value(
                key,
                name,
                dimensions,
                "state" if name in state_outputs or kind == "state" else "data",
            )
            for name, dimensions in outputs.items()
        }

    def op(
        self,
        key: str,
        operation: str,
        inputs: dict[str, Value],
        output: r.ArchitectureShape,
        *,
        parent: str,
        parameters: tuple[str, ...] = (),
        attributes: dict[str, str | float | bool] | None = None,
        fields: tuple[str, ...] = (),
        formula: str | None = None,
        module_name: str | None = None,
        state_output: bool = False,
        kind: Literal["operation", "input", "output", "context", "state"] = "operation",
    ) -> Value:
        name = "out"
        return self.node(
            key,
            operation,
            inputs,
            {name: output},
            parent=parent,
            parameters=parameters,
            attributes=attributes,
            fields=fields,
            formula=formula,
            module_name=module_name,
            kind=kind,
            state_outputs=(name,) if state_output else (),
        )[name]

    def begin_group(self, key: str) -> None:
        self.children.setdefault(key, [])

    def end_group(
        self,
        key: str,
        label: str,
        parent: str | None,
        inputs: dict[str, Value],
        outputs: dict[str, Value],
        *,
        role: str | None = None,
        attributes: dict[str, str | float | bool] | None = None,
        fields: tuple[str, ...] = (),
    ) -> dict[str, Value]:
        self.children.setdefault(key, [])
        for name, value in inputs.items():
            # Root model inputs are already container ports; a self-edge would
            # invent a wire between the same port rather than an operation.
            if parent is not None or value.node != key:
                self.link(value, key, name)
        for name, value in outputs.items():
            self.link(value, key, name)
        args: dict[str, Any] = {}
        if parent is not None:
            args["parent_id"] = self.nid(parent)
            self.children.setdefault(parent, []).append(self.nid(key))
        attrs = [
            r.ArchitectureAttribute(name=name, value=value, provenance=self.provenance(*fields))
            for name, value in (attributes or {}).items()
        ]
        if role is not None:
            attrs.append(self.role_attribute(role))
        provenance = self.provenance(*fields)
        if parent is None:
            provenance = self.b.producer.provenance() + provenance[1:]
        self.b.add_node(
            r.ArchitectureGroupNode(
                id=self.nid(key),
                kind="group",
                label=label,
                ports=[self.port(name, value.shape, "input") for name, value in inputs.items()]
                + [self.port(name, value.shape, "output") for name, value in outputs.items()],
                children=self.children[key],
                parameter_ids=[],
                references=[r.ArchitectureModuleReference(kind="module", name=key)],
                attributes=attrs,
                provenance=provenance + [self.source_key(key)],
                **args,
            ),
            semantic_key=key,
        )
        return {name: Value(key, name, value.shape, value.kind) for name, value in outputs.items()}

    def linear(
        self,
        key: str,
        x: Value,
        input_width: int,
        output_width: int,
        parent: str,
        *,
        fields: tuple[str, ...] = ("hidden_size",),
        attributes: dict[str, str | float | bool] | None = None,
    ) -> Value:
        return self.op(
            key,
            "linear",
            {"x": x},
            shape(*x.shape[:-1], output_width) if x.shape is not None else None,
            parent=parent,
            parameters=(key + ".weight",),
            attributes={"bias": False, **(attributes or {})},
            fields=fields,
            formula="y = x Wᵀ",
        )

    def norm(
        self, key: str, x: Value, width: int, parent: str, *, fields: tuple[str, ...]
    ) -> Value:
        return self.op(
            key,
            "rms_norm",
            {"x": x},
            x.shape,
            parent=parent,
            parameters=(key + ".weight",),
            attributes={"epsilon": self.c["rms_norm_eps"], "axis": -1.0, "weight_offset": 0.0},
            fields=fields,
            formula="y = weight * x / sqrt(mean(x², axis=-1) + epsilon)",
        )

    def build_attention(
        self,
        layer: str,
        x: Value,
        positions: Value,
        cosine: Value,
        sine: Value,
        mask: Value,
        prior_key: Value,
        prior_value: Value,
    ) -> dict[str, Value]:
        c = self.c
        key = layer + ".self_attn"
        self.begin_group(key)
        self.b.begin_template(key, key, "deepseek_v2_mla_attention", "attention")
        group_positions = Value(key, "positions", positions.shape)
        group_cosine = Value(key, "cos", cosine.shape)
        group_sine = Value(key, "sin", sine.shape)
        group_mask = Value(key, "causal_mask", mask.shape)
        group_prior_key = Value(key, "prior_key_state", prior_key.shape, "state")
        group_prior_value = Value(key, "prior_value_state", prior_value.shape, "state")
        hidden = shape("B", "S", c["hidden_size"])
        rotary_width = int(c["qk_rope_head_dim"])
        non_rotary_width = int(c["qk_nope_head_dim"])
        q_width = non_rotary_width + rotary_width
        heads = int(c["num_attention_heads"])
        values_width = int(c["v_head_dim"])
        q_heads = shape("B", heads, "S", q_width)
        q_nonrotary = shape("B", heads, "S", non_rotary_width)
        q_rotary = shape("B", heads, "S", rotary_width)
        kv_latent = shape("B", "S", c["kv_lora_rank"])
        key_rotary = shape("B", 1, "S", rotary_width)
        key_nonrotary = shape("B", heads, "S", non_rotary_width)
        value_heads = shape("B", heads, "S", values_width)
        full_key = shape("B", heads, "S", q_width)
        cached_key = shape("B", heads, _expression("K + S", "K", "S"), q_width)
        cached_value = shape("B", heads, _expression("K + S", "K", "S"), values_width)
        current_x = Value(key, "x", hidden)

        query_projection = self.linear(
            key + ".q_proj",
            current_x,
            int(c["hidden_size"]),
            heads * q_width,
            key,
            fields=("hidden_size", "num_attention_heads", "qk_nope_head_dim", "qk_rope_head_dim"),
            attributes={"q_lora_rank": "null; direct query projection"},
        )
        query_heads = self.op(
            key + ".q_heads",
            "reshape_transpose",
            {"x": query_projection},
            q_heads,
            parent=key,
            attributes={"heads": float(heads), "head_width": float(q_width)},
            fields=("num_attention_heads", "qk_nope_head_dim", "qk_rope_head_dim"),
            formula="[B,S,H·D] → [B,H,S,D]",
        )
        query_split = self.node(
            key + ".q_split",
            "split",
            {"x": query_heads},
            {"non_rotary": q_nonrotary, "rotary": q_rotary},
            parent=key,
            attributes={
                "non_rotary_width": float(non_rotary_width),
                "rotary_width": float(rotary_width),
            },
            fields=("qk_nope_head_dim", "qk_rope_head_dim"),
            formula="split each query head into non-positional and rotary subspaces",
        )
        query_rope = self.op(
            key + ".q_rope",
            "rotary_position",
            {
                "x": query_split["rotary"],
                "cos": group_cosine,
                "sin": group_sine,
                "positions": group_positions,
            },
            q_rotary,
            parent=key,
            attributes={"rope_type": "yarn", "rotary_width": float(rotary_width)},
            fields=("rope_scaling", "rope_theta", "qk_rope_head_dim"),
            formula="apply YARN RoPE to the rotary query subspace only",
        )
        query = self.op(
            key + ".q_recombine",
            "concat",
            {"non_rotary": query_split["non_rotary"], "rotary": query_rope},
            q_heads,
            parent=key,
            attributes={"axis": -1.0},
            fields=("qk_nope_head_dim", "qk_rope_head_dim"),
        )

        kv_projection = self.linear(
            key + ".kv_a_proj_with_mqa",
            current_x,
            int(c["hidden_size"]),
            int(c["kv_lora_rank"]) + rotary_width,
            key,
            fields=("hidden_size", "kv_lora_rank", "qk_rope_head_dim"),
            attributes={"shared_rotary_key_heads": 1.0},
        )
        kv_split = self.node(
            key + ".kv_a_split",
            "split",
            {"x": kv_projection},
            {"latent": kv_latent, "rotary_key": shape("B", "S", rotary_width)},
            parent=key,
            attributes={
                "latent_rank": float(c["kv_lora_rank"]),
                "rotary_width": float(rotary_width),
            },
            fields=("kv_lora_rank", "qk_rope_head_dim"),
            formula="split compressed KV latent from the shared rotary-key branch",
        )
        kv_latent_norm = self.norm(
            key + ".kv_a_layernorm",
            kv_split["latent"],
            int(c["kv_lora_rank"]),
            key,
            fields=("kv_lora_rank", "rms_norm_eps"),
        )
        kv_reconstruction = self.linear(
            key + ".kv_b_proj",
            kv_latent_norm,
            int(c["kv_lora_rank"]),
            heads * (non_rotary_width + values_width),
            key,
            fields=("kv_lora_rank", "num_attention_heads", "qk_nope_head_dim", "v_head_dim"),
            attributes={
                "reconstructed_key_width": float(non_rotary_width),
                "value_width": float(values_width),
            },
        )
        kv_heads = self.op(
            key + ".kv_heads",
            "reshape_transpose",
            {"x": kv_reconstruction},
            shape("B", heads, "S", non_rotary_width + values_width),
            parent=key,
            attributes={"heads": float(heads)},
        )
        kv_b_split = self.node(
            key + ".kv_b_split",
            "split",
            {"x": kv_heads},
            {"key_non_rotary": key_nonrotary, "value": value_heads},
            parent=key,
            attributes={"key_width": float(non_rotary_width), "value_width": float(values_width)},
            fields=("qk_nope_head_dim", "v_head_dim"),
            formula="reconstructed per-head key and value projections",
        )
        key_rotary_heads = self.op(
            key + ".k_pe_heads",
            "reshape_transpose",
            {"x": kv_split["rotary_key"]},
            key_rotary,
            parent=key,
            attributes={"heads": 1.0, "shared_across_query_heads": True},
            fields=("qk_rope_head_dim",),
        )
        key_rope = self.op(
            key + ".k_rope",
            "rotary_position",
            {
                "x": key_rotary_heads,
                "cos": group_cosine,
                "sin": group_sine,
                "positions": group_positions,
            },
            key_rotary,
            parent=key,
            attributes={"rope_type": "yarn", "rotary_width": float(rotary_width)},
            fields=("rope_scaling", "rope_theta", "qk_rope_head_dim"),
            formula="apply YARN RoPE to the single shared rotary-key branch",
        )
        key_rotary_repeated = self.op(
            key + ".k_pe_repeat",
            "repeat_kv",
            {"x": key_rope},
            shape("B", heads, "S", rotary_width),
            parent=key,
            attributes={"groups": float(heads), "shared_rotary_key": True},
            fields=("num_attention_heads",),
        )
        key_current = self.op(
            key + ".key_recombine",
            "concat",
            {"non_rotary": kv_b_split["key_non_rotary"], "rotary": key_rotary_repeated},
            full_key,
            parent=key,
            attributes={"axis": -1.0},
            fields=("qk_nope_head_dim", "qk_rope_head_dim"),
        )

        key_cache = self.op(
            key + ".key_cache_update",
            "state_concat",
            {"prior_state": group_prior_key, "current_key": key_current},
            cached_key,
            parent=key,
            attributes={"sequence_axis": 2.0},
            fields=("num_hidden_layers",),
            formula="symbolic prior key state concatenated with current reconstructed keys",
            state_output=True,
        )
        value_cache = self.op(
            key + ".value_cache_update",
            "state_concat",
            {"prior_state": group_prior_value, "current_value": kv_b_split["value"]},
            cached_value,
            parent=key,
            attributes={"sequence_axis": 2.0},
            fields=("num_hidden_layers",),
            formula="symbolic prior value state concatenated with current values",
            state_output=True,
        )
        next_key = self.op(
            key + ".next_key_state",
            "key_value_state",
            {"x": key_cache},
            cached_key,
            parent=key,
            kind="state",
            state_output=True,
            attributes={"state": "per-layer key cache; symbolic, no sample"},
        )
        next_value = self.op(
            key + ".next_value_state",
            "key_value_state",
            {"x": value_cache},
            cached_value,
            parent=key,
            kind="state",
            state_output=True,
            attributes={"state": "per-layer value cache; symbolic, no sample"},
        )
        key_transpose = self.op(
            key + ".key_transpose",
            "transpose",
            {"x": key_cache},
            shape("B", heads, q_width, _expression("K + S", "K", "S")),
            parent=key,
            attributes={"axes": "0,1,3,2"},
        )
        scores = self.op(
            key + ".scores",
            "matmul",
            {"query": query, "key_transposed": key_transpose},
            shape("B", heads, "S", _expression("K + S", "K", "S")),
            parent=key,
            formula="Q Kᵀ",
        )
        scaling = self.op(
            key + ".score_scale",
            "scale",
            {"x": scores},
            scores.shape,
            parent=key,
            attributes={"head_width": float(q_width), "rope_type": "yarn", "mscale_all_dim": 0.707},
            fields=("qk_nope_head_dim", "qk_rope_head_dim", "rope_scaling"),
            formula="softmax_scale = 192⁻¹ᐟ² × yarn_get_mscale(40, 0.707)²",
        )
        masked = self.op(
            key + ".causal_mask",
            "add_mask",
            {"scores": scaling, "mask": group_mask},
            scores.shape,
            parent=key,
            attributes={"causal": True},
        )
        probabilities = self.op(
            key + ".softmax",
            "softmax",
            {"x": masked},
            scores.shape,
            parent=key,
            attributes={"axis": -1.0},
            formula="softmax over cached and current key positions",
        )
        attended = self.op(
            key + ".weighted_values",
            "matmul",
            {"probabilities": probabilities, "value": value_cache},
            shape("B", heads, "S", values_width),
            parent=key,
        )
        output_transpose = self.op(
            key + ".output_transpose",
            "transpose",
            {"x": attended},
            shape("B", "S", heads, values_width),
            parent=key,
            attributes={"axes": "0,2,1,3"},
        )
        merged = self.op(
            key + ".merge_heads",
            "reshape",
            {"x": output_transpose},
            shape("B", "S", heads * values_width),
            parent=key,
        )
        projected = self.linear(
            key + ".o_proj",
            merged,
            heads * values_width,
            int(c["hidden_size"]),
            key,
            fields=("num_attention_heads", "v_head_dim", "hidden_size"),
        )
        return self.end_group(
            key,
            "Multi-head Latent Attention",
            layer,
            {
                "x": x,
                "positions": positions,
                "cos": cosine,
                "sin": sine,
                "causal_mask": mask,
                "prior_key_state": prior_key,
                "prior_value_state": prior_value,
            },
            {"out": projected, "next_key_state": next_key, "next_value_state": next_value},
            role="attention",
            attributes={
                "attention_type": "Multi-head Latent Attention (MLA)",
                "causal": True,
                "attention_dropout": float(c["attention_dropout"]),
            },
            fields=(
                "kv_lora_rank",
                "qk_nope_head_dim",
                "qk_rope_head_dim",
                "v_head_dim",
                "rope_scaling",
                "attention_dropout",
            ),
        )

    def dense_mlp(self, layer: str, x: Value) -> Value:
        c = self.c
        key = layer + ".mlp"
        self.begin_group(key)
        width = int(c["intermediate_size"])
        gate = self.linear(
            key + ".gate_proj",
            Value(key, "x", x.shape),
            int(c["hidden_size"]),
            width,
            key,
            fields=("hidden_size", "intermediate_size", "hidden_act"),
        )
        up = self.linear(
            key + ".up_proj",
            Value(key, "x", x.shape),
            int(c["hidden_size"]),
            width,
            key,
            fields=("hidden_size", "intermediate_size"),
        )
        activated = self.op(
            key + ".silu",
            "silu",
            {"x": gate},
            gate.shape,
            parent=key,
            fields=("hidden_act",),
            formula="silu(x) = x · sigmoid(x)",
        )
        product = self.op(
            key + ".gated_product",
            "multiply",
            {"gate": activated, "up": up},
            gate.shape,
            parent=key,
        )
        down = self.linear(
            key + ".down_proj",
            product,
            width,
            int(c["hidden_size"]),
            key,
            fields=("hidden_size", "intermediate_size"),
        )
        result = self.end_group(
            key,
            "Dense MLP",
            layer,
            {"x": x},
            {"out": down},
            role="mlp",
            attributes={"feed_forward_path": "dense", "intermediate_width": float(width)},
            fields=("first_k_dense_replace", "intermediate_size", "hidden_act"),
        )
        return result["out"]

    def expert(
        self, layer: str, expert_index: int, dispatch: dict[str, Value], hidden_width: int
    ) -> Value:
        key = f"{layer}.mlp.experts.{expert_index}"
        self.begin_group(key)
        width = int(self.c["moe_intermediate_size"])
        tokens = Value(dispatch["tokens"].node, dispatch["tokens"].port, dispatch["tokens"].shape)
        weights = Value(
            dispatch["weights"].node, dispatch["weights"].port, dispatch["weights"].shape
        )
        require(tokens.shape is not None, "Expert dispatch has no token rank.")
        assert tokens.shape is not None
        mlp = self.op(
            key + ".swiglu_mlp",
            "swiglu_mlp",
            {"x": Value(key, "x", tokens.shape)},
            shape(*tokens.shape[:-1], hidden_width),
            parent=key,
            parameters=(
                key + ".gate_proj.weight",
                key + ".up_proj.weight",
                key + ".down_proj.weight",
            ),
            attributes={"activation": "silu", "intermediate_width": float(width)},
            fields=("hidden_size", "moe_intermediate_size", "hidden_act"),
            formula="down_proj(silu(gate_proj(x)) * up_proj(x))",
            module_name=key,
        )
        weighted = self.op(
            key + ".routing_weight",
            "multiply",
            {"expert_values": mlp, "routing_weights": Value(key, "routing_weights", weights.shape)},
            mlp.shape,
            parent=key,
            formula="weighted expert output = expert output × selected routing weight per token",
        )
        output = self.end_group(
            key,
            f"Routed expert {expert_index}",
            layer + ".mlp",
            {"x": tokens, "routing_weights": weights},
            {"out": weighted},
            role="mlp",
            attributes={"expert_index": float(expert_index), "intermediate_width": float(width)},
            fields=("n_routed_experts", "moe_intermediate_size"),
        )
        return output["out"]

    def shared_mlp(self, layer: str, x: Value, hidden_width: int) -> Value:
        key = layer + ".mlp.shared_experts"
        self.begin_group(key)
        width = int(self.c["moe_intermediate_size"]) * int(self.c["n_shared_experts"])
        gate = self.linear(
            key + ".gate_proj",
            Value(key, "x", x.shape),
            hidden_width,
            width,
            key,
            fields=("hidden_size", "n_shared_experts", "moe_intermediate_size", "hidden_act"),
        )
        up = self.linear(
            key + ".up_proj",
            Value(key, "x", x.shape),
            hidden_width,
            width,
            key,
            fields=("hidden_size", "n_shared_experts", "moe_intermediate_size"),
        )
        activated = self.op(
            key + ".silu",
            "silu",
            {"x": gate},
            gate.shape,
            parent=key,
            fields=("hidden_act",),
            formula="silu(x) = x · sigmoid(x)",
        )
        product = self.op(
            key + ".gated_product",
            "multiply",
            {"gate": activated, "up": up},
            gate.shape,
            parent=key,
        )
        down = self.linear(
            key + ".down_proj",
            product,
            width,
            hidden_width,
            key,
            fields=("hidden_size", "n_shared_experts", "moe_intermediate_size"),
        )
        result = self.end_group(
            key,
            "Shared experts (combined MLP)",
            layer + ".mlp",
            {"x": x},
            {"out": down},
            role="mlp",
            attributes={
                "shared_expert_count": float(self.c["n_shared_experts"]),
                "combined_intermediate_width": float(width),
            },
            fields=("n_shared_experts", "moe_intermediate_size"),
        )
        return result["out"]

    def moe_mlp(self, layer: str, x: Value) -> Value:
        c = self.c
        key = layer + ".mlp"
        self.begin_group(key)
        hidden = int(c["hidden_size"])
        experts = int(c["n_routed_experts"])
        top_k = int(c["num_experts_per_tok"])
        scores_shape = shape("B", "S", experts)
        router_scores = self.linear(
            key + ".gate",
            Value(key, "x", x.shape),
            hidden,
            experts,
            key,
            fields=("hidden_size", "n_routed_experts", "scoring_func"),
            attributes={"scoring_function": "softmax"},
        )
        scores = self.op(
            key + ".gate_softmax",
            "softmax",
            {"logits": router_scores},
            scores_shape,
            parent=key,
            attributes={"axis": -1.0, "score_type": "float32 softmax"},
            fields=("scoring_func",),
        )
        self.node(
            key + ".topk",
            "topk_selection",
            {"scores": scores},
            {"expert_indices": shape("B", "S", top_k), "expert_weights": shape("B", "S", top_k)},
            parent=key,
            attributes={
                "top_k": float(top_k),
                "selection_method": "greedy top-k over softmax scores",
                "norm_topk_prob": False,
                "routed_scaling_factor": float(c["routed_scaling_factor"]),
                "selected_weight_rule": "selected softmax scores × routed_scaling_factor",
                "runtime_selection": "symbolic; no tokens or experts selected during analysis",
            },
            fields=(
                "num_experts_per_tok",
                "topk_method",
                "norm_topk_prob",
                "routed_scaling_factor",
            ),
            formula=(
                "select configured top-k expert IDs; retain their softmax scores "
                "and multiply by routed_scaling_factor because norm_topk_prob=false"
            ),
        )
        weighted_values: list[Value] = []
        token_positions: list[Value] = []
        instances: list[r.ArchitectureRepetitionInstance] = []
        for index in range(experts):
            count_symbol = f"R{int(layer.rsplit('.', 1)[-1])}_{index}"
            self.b.add_symbol(
                count_symbol,
                f"Symbolic token positions routed to expert {index} in {layer}; not computed.",
            )
            routed_tokens = shape(count_symbol, hidden)
            routed_weights = shape(count_symbol)
            positions = shape(count_symbol, 2)
            dispatch_key = f"{key}.dispatch.{index}"
            dispatch = self.node(
                dispatch_key,
                "symbolic_expert_dispatch",
                {
                    "hidden_states": Value(key, "x", x.shape),
                    "expert_indices": Value(
                        key + ".topk", "expert_indices", shape("B", "S", top_k)
                    ),
                    "expert_weights": Value(
                        key + ".topk", "expert_weights", shape("B", "S", top_k)
                    ),
                },
                {"tokens": routed_tokens, "weights": routed_weights, "positions": positions},
                parent=key,
                attributes={"expert_index": float(index), "selection_is_runtime_data": True},
                fields=("n_routed_experts", "num_experts_per_tok"),
                formula=(
                    "symbolically gather positions whose configured top-k IDs contain this expert"
                ),
            )
            expert_key = f"{key}.experts.{index}"
            expert_output = self.expert(layer, index, dispatch, hidden)
            weighted_values.append(expert_output)
            token_positions.append(dispatch["positions"])
            instances.append(
                r.ArchitectureRepetitionInstance(
                    node_id=self.nid(expert_key), index=index, variant="routed_expert"
                )
            )

        scatter_inputs: dict[str, Value] = {
            "expert_values": Value(weighted_values[0].node, weighted_values[0].port, None),
            "token_positions": Value(token_positions[0].node, token_positions[0].port, None),
        }
        scatter = self.node(
            key + ".routed_scatter_sum",
            "weighted_scatter_sum",
            scatter_inputs,
            {"out": shape("B", "S", hidden)},
            parent=key,
            attributes={"top_k": float(top_k), "reduction": "sum at original token positions"},
            fields=("num_experts_per_tok",),
            formula=(
                "scatter-add every selected, weighted expert result to its symbolic "
                "source token position"
            ),
        )["out"]
        # The scatter ports collect every expert; repeated edges retain each identity.
        for expert_value in weighted_values[1:]:
            self.link(
                Value(expert_value.node, expert_value.port, None),
                key + ".routed_scatter_sum",
                "expert_values",
            )
        for position_value in token_positions[1:]:
            self.link(
                Value(position_value.node, position_value.port, None),
                key + ".routed_scatter_sum",
                "token_positions",
            )

        shared = self.shared_mlp(layer, Value(key, "x", x.shape), hidden)
        combined = self.op(
            key + ".shared_routed_add",
            "add",
            {"routed": scatter, "shared": shared},
            shape("B", "S", hidden),
            parent=key,
            attributes={"branches": "routed experts plus combined shared MLP"},
            fields=("n_shared_experts",),
            formula="routed weighted expert reduction + shared expert MLP output",
        )
        result = self.end_group(
            key,
            "DeepSeekMoE (routed and shared)",
            layer,
            {"x": x},
            {"out": combined},
            role="mlp",
            attributes={
                "routed_expert_count": float(experts),
                "top_k": float(top_k),
                "shared_expert_count": float(c["n_shared_experts"]),
                "routing_is_symbolic": True,
            },
            fields=(
                "first_k_dense_replace",
                "moe_layer_freq",
                "n_routed_experts",
                "num_experts_per_tok",
                "n_shared_experts",
            ),
        )
        self.b.add_repetition(
            r.ArchitectureRepetition(
                id=self.b.record_id("repetition", key + ".experts"),
                parent_id=self.nid(key),
                label="Routed experts",
                instances=instances,
            )
        )
        return result["out"]

    def layer(
        self,
        index: int,
        incoming: Value,
        positions: Value,
        cosine: Value,
        sine: Value,
        mask: Value,
        prior_key: Value,
        prior_value: Value,
    ) -> dict[str, Value]:
        c = self.c
        key = f"model.layers.{index}"
        hidden = shape("B", "S", c["hidden_size"])
        self.begin_group(key)
        x = Value(key, "x", hidden)
        normalized = self.norm(
            key + ".input_layernorm",
            x,
            int(c["hidden_size"]),
            key,
            fields=("hidden_size", "rms_norm_eps"),
        )
        attention = self.build_attention(
            key,
            normalized,
            Value(key, "positions", positions.shape),
            Value(key, "cos", cosine.shape),
            Value(key, "sin", sine.shape),
            Value(key, "causal_mask", mask.shape),
            Value(key, "prior_key_state", prior_key.shape, "state"),
            Value(key, "prior_value_state", prior_value.shape, "state"),
        )
        attention_residual = self.op(
            key + ".attention_residual",
            "add",
            {"skip": x, "branch": attention["out"]},
            hidden,
            parent=key,
            formula="input hidden states + MLA output",
        )
        post_norm = self.norm(
            key + ".post_attention_layernorm",
            attention_residual,
            int(c["hidden_size"]),
            key,
            fields=("hidden_size", "rms_norm_eps"),
        )
        mlp = self.dense_mlp(key, post_norm) if index == 0 else self.moe_mlp(key, post_norm)
        output = self.op(
            key + ".mlp_residual",
            "add",
            {"skip": attention_residual, "branch": mlp},
            hidden,
            parent=key,
            formula="post-attention residual stream + dense or MoE feed-forward output",
        )
        group_outputs = self.end_group(
            key,
            f"Decoder layer {index}",
            "model",
            {
                "x": incoming,
                "positions": positions,
                "cos": cosine,
                "sin": sine,
                "causal_mask": mask,
                "prior_key_state": prior_key,
                "prior_value_state": prior_value,
            },
            {
                "out": output,
                "next_key_state": attention["next_key_state"],
                "next_value_state": attention["next_value_state"],
            },
            attributes={
                "sequence_index": float(index),
                "feed_forward_path": "dense" if index == 0 else "moe",
            },
            fields=("first_k_dense_replace", "moe_layer_freq"),
        )
        return group_outputs

    def build(self) -> None:
        c = self.c
        hidden_width = int(c["hidden_size"])
        heads = int(c["num_attention_heads"])
        q_width = int(c["qk_nope_head_dim"]) + int(c["qk_rope_head_dim"])
        value_width = int(c["v_head_dim"])
        layers = int(c["num_hidden_layers"])
        sequence = shape("B", "S")
        hidden = shape("B", "S", hidden_width)
        key_state = shape(layers, "B", heads, "K", q_width)
        value_state = shape(layers, "B", heads, "K", value_width)
        updated_key_state = shape(layers, "B", heads, _expression("K + S", "K", "S"), q_width)
        updated_value_state = shape(layers, "B", heads, _expression("K + S", "K", "S"), value_width)
        mask_shape = shape("B", 1, "S", _expression("K + S", "K", "S"))
        self.b.add_symbol("B", "Symbolic batch size; no input has been executed.")
        self.b.add_symbol(
            "S", "Symbolic current token sequence length; no prompt has been supplied."
        )
        self.b.add_symbol(
            "K", "Symbolic prior per-layer key/value cache length; no cache sample exists."
        )

        root_inputs: dict[str, Value] = {
            "input_ids": Value("model", "input_ids", sequence),
            "position_ids": Value("model", "position_ids", sequence),
            "attention_mask": Value("model", "attention_mask", mask_shape),
            "past_key_values_key": Value("model", "past_key_values_key", key_state, "state"),
            "past_key_values_value": Value("model", "past_key_values_value", value_state, "state"),
        }
        tokens = self.op(
            "model.embed_tokens",
            "embedding",
            {"input_ids": root_inputs["input_ids"]},
            hidden,
            parent="model",
            parameters=("model.embed_tokens.weight",),
            attributes={
                "vocabulary_size": float(c["vocab_size"]),
                "hidden_size": float(hidden_width),
            },
            fields=("vocab_size", "hidden_size"),
        )
        rotary_width = int(c["qk_rope_head_dim"])
        rotary_shape = shape("B", "S", rotary_width)
        rope_attributes: dict[str, str | float | bool] = {
            "rope_type": "yarn",
            "rope_theta": float(c["rope_theta"]),
            "max_position_embeddings": float(c["max_position_embeddings"]),
            "factor": float(c["rope_scaling"]["factor"]),
            "original_max_position_embeddings": float(
                c["rope_scaling"]["original_max_position_embeddings"]
            ),
            "beta_fast": float(c["rope_scaling"]["beta_fast"]),
            "beta_slow": float(c["rope_scaling"]["beta_slow"]),
            "mscale": float(c["rope_scaling"]["mscale"]),
            "mscale_all_dim": float(c["rope_scaling"]["mscale_all_dim"]),
        }
        cosine = self.op(
            "model.rotary_cos",
            "yarn_rotary_cosine",
            {"positions": root_inputs["position_ids"]},
            rotary_shape,
            parent="model",
            attributes=rope_attributes,
            fields=("rope_theta", "rope_scaling", "qk_rope_head_dim"),
            formula="YARN cosine frequencies at symbolic position_ids; no positions are evaluated.",
        )
        sine = self.op(
            "model.rotary_sin",
            "yarn_rotary_sine",
            {"positions": root_inputs["position_ids"]},
            rotary_shape,
            parent="model",
            attributes=rope_attributes,
            fields=("rope_theta", "rope_scaling", "qk_rope_head_dim"),
            formula="YARN sine frequencies at symbolic position_ids; no positions are evaluated.",
        )
        key_states: list[Value] = []
        value_states: list[Value] = []
        repetitions: list[r.ArchitectureRepetitionInstance] = []
        previous = tokens
        for index in range(layers):
            prior_key = self.op(
                f"model.layer_state.{index}.prior_key",
                "select_layer_state",
                {"state_bank": root_inputs["past_key_values_key"]},
                shape("B", heads, "K", q_width),
                parent="model",
                attributes={"layer_index": float(index), "state_role": "prior key"},
                fields=(
                    "num_hidden_layers",
                    "num_attention_heads",
                    "qk_nope_head_dim",
                    "qk_rope_head_dim",
                ),
                state_output=True,
            )
            prior_value = self.op(
                f"model.layer_state.{index}.prior_value",
                "select_layer_state",
                {"state_bank": root_inputs["past_key_values_value"]},
                shape("B", heads, "K", value_width),
                parent="model",
                attributes={"layer_index": float(index), "state_role": "prior value"},
                fields=("num_hidden_layers", "num_attention_heads", "v_head_dim"),
                state_output=True,
            )
            built = self.layer(
                index,
                previous,
                root_inputs["position_ids"],
                cosine,
                sine,
                root_inputs["attention_mask"],
                prior_key,
                prior_value,
            )
            previous = built["out"]
            key_states.append(built["next_key_state"])
            value_states.append(built["next_value_state"])
            repetitions.append(
                r.ArchitectureRepetitionInstance(
                    node_id=self.nid(f"model.layers.{index}"),
                    index=index,
                    variant="dense" if index == 0 else "moe",
                )
            )
        final_norm = self.norm(
            "model.norm", previous, hidden_width, "model", fields=("hidden_size", "rms_norm_eps")
        )
        logits = self.linear(
            "lm_head",
            final_norm,
            hidden_width,
            int(c["vocab_size"]),
            "model",
            fields=("hidden_size", "vocab_size"),
        )
        keys_stacked = self.node(
            "model.next_key_values",
            "stack_layer_states",
            {f"layer_{index}": value for index, value in enumerate(key_states)},
            {"out": updated_key_state},
            parent="model",
            attributes={"layer_count": float(layers), "state_role": "next keys"},
            fields=("num_hidden_layers",),
            state_outputs=("out",),
        )["out"]
        values_stacked = self.node(
            "model.next_value_values",
            "stack_layer_states",
            {f"layer_{index}": value for index, value in enumerate(value_states)},
            {"out": updated_value_state},
            parent="model",
            attributes={"layer_count": float(layers), "state_role": "next values"},
            fields=("num_hidden_layers",),
            state_outputs=("out",),
        )["out"]
        self.end_group(
            "model",
            "DeepSeek-V2-Lite CausalLM",
            None,
            root_inputs,
            {
                "logits": logits,
                "next_key_values_key": keys_stacked,
                "next_key_values_value": values_stacked,
            },
            attributes={"architecture": ARCHITECTURE, "static_evaluation_path": True},
            fields=("architectures", "model_type", "num_hidden_layers"),
        )
        self.b.add_repetition(
            r.ArchitectureRepetition(
                id=self.b.record_id("repetition", "model.layers"),
                parent_id=self.nid("model"),
                label="Decoder layers",
                instances=repetitions,
            )
        )
        if self.inputs.bindings.tokenizer_available:
            self.b.add_node(
                r.ArchitectureLeafNode(
                    id=self.nid("tokenizer"),
                    kind="context",
                    label="Tokenizer",
                    ports=[],
                    parameter_ids=[],
                    references=[r.ArchitectureTokenizerReference(kind="tokenizer")],
                    attributes=[],
                    provenance=self.b.producer.provenance(),
                )
            )
        if set(self.physical) - self.used_storage:
            self.b.diagnose(
                r.ArchitectureDiagnostic(
                    code="unrecognized_storage",
                    message=(
                        "Checkpoint contains storage outside the reviewed DeepSeek-V2-Lite "
                        "language graph."
                    ),
                )
            )


def build(inputs: AnalysisInput, builder: GraphBuilder) -> None:
    configuration = checked(inputs.configuration)
    require(configuration is not None, "Unsupported DeepSeek-V2 configuration.")
    assert configuration is not None
    DeepseekGraph(inputs, builder, configuration).build()


def register_deepseek_v2(registry: DescriptionRegistry) -> None:
    registry.register(
        Description(
            producer=PRODUCER,
            scope="language_model",
            model_types=frozenset({"deepseek_v2"}),
            architectures=frozenset({ARCHITECTURE}),
            supports=supports,
            build=build,
        )
    )

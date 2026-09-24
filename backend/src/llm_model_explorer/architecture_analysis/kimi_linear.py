"""Verified static Kimi Linear KDA/MLA/MoE description; no checkpoint code executes."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from . import records as r
from .core import AnalysisInput, Description, DescriptionRegistry, GraphBuilder, Producer
from .deepseek_v2 import Value, _expression, shape
from .glm4_moe_lite import Glm4MoeLiteGraph
from .validation import require

CONFIG_REVISION = "5d029d1844aa64ec302e14466be7d0353c6e697f"
MODELING_REVISION = "d64a5299ded33ab2609617e05f6bd2cf9b6eef35"
SOURCE_REVISION = (
    "moonshotai/Kimi-Linear-48B-A3B-Instruct/modeling_kimi.py@"
    + MODELING_REVISION
    + "; cyankiwi/Kimi-Linear-48B-A3B-Instruct-AWQ-4bit/config.json@"
    + CONFIG_REVISION
)
PRODUCER = Producer("kimi-linear-kda-mla-moe", "3", SOURCE_REVISION)
ARCHITECTURE = "KimiLinearForCausalLM"
MODEL_TYPE = "kimi_linear"

# Values required to describe this reviewed reference. They intentionally remain
# strict: model names and compatible-looking dimensions do not establish coverage.
REFERENCE: dict[str, Any] = {
    "model_type": MODEL_TYPE,
    "architectures": [ARCHITECTURE],
    "vocab_size": 163840,
    "hidden_size": 2304,
    "head_dim": 72,  # Legacy config field; Kimi uses the explicit KDA/MLA geometry below.
    "intermediate_size": 9216,
    "num_hidden_layers": 27,
    "num_attention_heads": 32,
    "num_key_value_heads": 32,
    "hidden_act": "silu",
    "rms_norm_eps": 1e-5,
    "rope_theta": 10000.0,
    "rope_scaling": None,
    "tie_word_embeddings": False,
    "first_k_dense_replace": 1,
    "moe_intermediate_size": 1024,
    "moe_layer_freq": 1,
    "moe_renormalize": True,
    "moe_router_activation_func": "sigmoid",
    "num_experts": 256,
    "num_experts_per_token": 8,
    "num_shared_experts": 1,
    "routed_scaling_factor": 2.446,
    "use_grouped_topk": True,
    "num_expert_group": 1,
    "topk_group": 1,
    "q_lora_rank": None,
    "kv_lora_rank": 512,
    "qk_nope_head_dim": 128,
    "qk_rope_head_dim": 64,
    "v_head_dim": 128,
    "mla_use_nope": True,
    "num_nextn_predict_layers": 0,
    "linear_attn_config": {
        "full_attn_layers": [4, 8, 12, 16, 20, 24, 27],
        "head_dim": 128,
        "kda_layers": [
            1,
            2,
            3,
            5,
            6,
            7,
            9,
            10,
            11,
            13,
            14,
            15,
            17,
            18,
            19,
            21,
            22,
            23,
            25,
            26,
        ],
        "num_heads": 32,
        "short_conv_kernel_size": 4,
    },
    "use_cache": True,
}

METADATA = {
    "_commit_hash",
    "_name_or_path",
    "auto_map",
    "bos_token_id",
    "dtype",
    "eos_token_id",
    "initializer_range",
    "model_max_length",
    "pad_token_id",
    "quantization_config",
    "transformers_version",
    "torch_dtype",
    "attention_dropout",
}
AUTO_MAP = {
    "AutoConfig": "configuration_kimi.KimiLinearConfig",
    "AutoModel": "modeling_kimi.KimiLinearModel",
    "AutoModelForCausalLM": "modeling_kimi.KimiLinearForCausalLM",
}


def checked(configuration: Mapping[str, object]) -> dict[str, Any] | None:
    """Select only the pinned Kimi Linear structural option set."""

    def same_option(actual: object, expected: object) -> bool:
        if type(actual) is not type(expected):
            return False
        if isinstance(expected, dict):
            if not isinstance(actual, dict) or set(actual) != set(expected):
                return False
            return all(same_option(actual[key], expected[key]) for key in expected)
        if isinstance(expected, list):
            return (
                isinstance(actual, list)
                and len(actual) == len(expected)
                and all(
                    same_option(left, right) for left, right in zip(actual, expected, strict=True)
                )
            )
        return actual == expected

    if set(configuration) - (set(REFERENCE) | METADATA):
        return None
    for key, expected in REFERENCE.items():
        actual = configuration.get(key)
        if not same_option(actual, expected):
            return None

    auto_map = configuration.get("auto_map")
    if auto_map is not None and auto_map != AUTO_MAP:
        return None
    quantization = configuration.get("quantization_config")
    if quantization is not None and not isinstance(quantization, dict):
        return None
    for key in ("dtype", "torch_dtype", "transformers_version", "_name_or_path", "_commit_hash"):
        value = configuration.get(key)
        if value is not None and not isinstance(value, str):
            return None
    for key in ("bos_token_id", "eos_token_id", "pad_token_id", "model_max_length"):
        value = configuration.get(key)
        if value is not None and type(value) is not int:
            return None
    for key in ("initializer_range", "attention_dropout"):
        value = configuration.get(key)
        if value is not None and (
            type(value) not in (int, float)
            or (isinstance(value, (int, float)) and not math.isfinite(value))
        ):
            return None

    linear = configuration["linear_attn_config"]
    assert isinstance(linear, dict)
    kda = set(linear["kda_layers"])
    result = dict(configuration)
    result["layer_types"] = [
        "kda" if index + 1 in kda else "full_attention"
        for index in range(REFERENCE["num_hidden_layers"])
    ]
    return result


def supports(inputs: AnalysisInput) -> bool:
    return checked(inputs.configuration) is not None


def parameter_shapes(configuration: Mapping[str, object]) -> dict[str, tuple[int, ...]]:
    """Complete logical CausalLM parameter inventory for the reviewed reference."""
    c = checked(configuration)
    require(c is not None, "Unsupported Kimi Linear configuration.")
    assert c is not None
    h = int(c["hidden_size"])
    vocab = int(c["vocab_size"])
    heads = int(c["num_attention_heads"])
    kda_heads = int(c["linear_attn_config"]["num_heads"])
    kda_dim = int(c["linear_attn_config"]["head_dim"])
    conv = int(c["linear_attn_config"]["short_conv_kernel_size"])
    qk_dim = int(c["qk_nope_head_dim"]) + int(c["qk_rope_head_dim"])
    kv_rank = int(c["kv_lora_rank"])
    value_dim = int(c["v_head_dim"])
    expert_count = int(c["num_experts"])
    expert_width = int(c["moe_intermediate_size"])
    shared_width = expert_width * int(c["num_shared_experts"])
    result: dict[str, tuple[int, ...]] = {
        "model.embed_tokens.weight": (vocab, h),
        "model.norm.weight": (h,),
        "lm_head.weight": (vocab, h),
    }
    for index, attention_type in enumerate(c["layer_types"]):
        layer = f"model.layers.{index}"
        result.update(
            {
                f"{layer}.input_layernorm.weight": (h,),
                f"{layer}.post_attention_layernorm.weight": (h,),
            }
        )
        attention = layer + ".self_attn"
        if attention_type == "kda":
            width = kda_heads * kda_dim
            result.update(
                {
                    f"{attention}.q_proj.weight": (width, h),
                    f"{attention}.k_proj.weight": (width, h),
                    f"{attention}.v_proj.weight": (width, h),
                    f"{attention}.q_conv1d.weight": (width, 1, conv),
                    f"{attention}.k_conv1d.weight": (width, 1, conv),
                    f"{attention}.v_conv1d.weight": (width, 1, conv),
                    f"{attention}.A_log": (1, 1, kda_heads, 1),
                    f"{attention}.dt_bias": (width,),
                    f"{attention}.f_a_proj.weight": (kda_dim, h),
                    f"{attention}.f_b_proj.weight": (width, kda_dim),
                    f"{attention}.b_proj.weight": (kda_heads, h),
                    f"{attention}.g_a_proj.weight": (kda_dim, h),
                    f"{attention}.g_b_proj.weight": (width, kda_dim),
                    f"{attention}.o_norm.weight": (kda_dim,),
                    f"{attention}.o_proj.weight": (h, width),
                }
            )
        else:
            rope_dim = int(c["qk_rope_head_dim"])
            nope_dim = int(c["qk_nope_head_dim"])
            result.update(
                {
                    f"{attention}.q_proj.weight": (heads * qk_dim, h),
                    f"{attention}.kv_a_proj_with_mqa.weight": (kv_rank + rope_dim, h),
                    f"{attention}.kv_a_layernorm.weight": (kv_rank,),
                    f"{attention}.kv_b_proj.weight": (heads * (nope_dim + value_dim), kv_rank),
                    f"{attention}.o_proj.weight": (h, heads * value_dim),
                }
            )

        if index == 0:
            mlp = layer + ".mlp"
            width = int(c["intermediate_size"])
            result.update(
                {
                    f"{mlp}.gate_proj.weight": (width, h),
                    f"{mlp}.up_proj.weight": (width, h),
                    f"{mlp}.down_proj.weight": (h, width),
                }
            )
        else:
            moe = layer + ".block_sparse_moe"
            result[f"{moe}.gate.weight"] = (expert_count, h)
            result[f"{moe}.gate.e_score_correction_bias"] = (expert_count,)
            for expert in range(expert_count):
                prefix = f"{moe}.experts.{expert}"
                result.update(
                    {
                        f"{prefix}.w1.weight": (expert_width, h),
                        f"{prefix}.w2.weight": (h, expert_width),
                        f"{prefix}.w3.weight": (expert_width, h),
                    }
                )
            shared = moe + ".shared_experts"
            result.update(
                {
                    f"{shared}.gate_proj.weight": (shared_width, h),
                    f"{shared}.up_proj.weight": (shared_width, h),
                    f"{shared}.down_proj.weight": (h, shared_width),
                }
            )
    return result


class KimiLinearGraph(Glm4MoeLiteGraph):
    """Kimi paths over the shared checked graph/binding record mechanics."""

    def __init__(self, inputs: AnalysisInput, builder: GraphBuilder, configuration: dict[str, Any]):
        self.inputs, self.b, self.c = inputs, builder, configuration
        self.children: dict[str, list[str]] = {}
        self.parameter_ids: dict[str, str] = {}
        self.used_storage: set[str] = set()
        self.physical = inputs.bindings.physical
        self.numeric = {tensor.name: tensor for tensor in inputs.bindings.numeric.values()}
        self.shapes = parameter_shapes(inputs.configuration)
        for name, dimensions in self.shapes.items():
            self.parameter_ids[name] = self.parameter(name, dimensions)

    def provenance(self, *fields: str) -> list[r.ArchitectureProvenance]:
        result = [
            r.ArchitectureProvenance(
                kind="description",
                source=self.b.producer.description,
                revision=self.b.producer.revision,
            )
        ]
        paths = [field for field in fields if field in self.inputs.configuration]
        if paths:
            result.append(
                r.ArchitectureProvenance(
                    kind="configuration",
                    source="config.json",
                    rule="Checked fields: " + ", ".join("/" + key for key in paths),
                )
            )
        return result

    def _unresolved(
        self,
        name: str,
        dimensions: tuple[int, ...],
        storage: list[r.ArchitectureStorage],
        message: str,
    ) -> str:
        parameter_id = self.b.record_id("parameter", name)
        self.b.add_parameter(
            r.ArchitectureDirectParameter(
                id=parameter_id,
                name=name,
                logical_shape=shape(*dimensions),
                binding="unresolved",
                storage=storage,
                inspection=r.ArchitectureUnavailableInspection(
                    status="unavailable", reason="unresolved_binding", message=message
                ),
                provenance=self.provenance()
                + [r.ArchitectureProvenance(kind="storage", source=item.name) for item in storage],
            )
        )
        self.b.diagnose(
            r.ArchitectureDiagnostic(
                code="unresolved_binding",
                message="A required Kimi Linear parameter does not bind to reviewed storage.",
                parameter_id=parameter_id,
            )
        )
        return parameter_id

    def parameter(self, name: str, dimensions: tuple[int, ...]) -> str:
        """Bind one logical tensor with compact provenance for large expert inventories."""
        physical = self.physical.get(name)
        numeric = self.numeric.get(name)
        packed = self._compressed_storage(name, dimensions)
        parameter_id = self.b.record_id("parameter", name)
        # `storage` already records every exact companion name and geometry. Keep concise
        # description provenance on each logical parameter without repeating those storage
        # names a second time in provenance.
        provenance = self.provenance()
        if packed is not None:
            parameter = r.ArchitectureDirectParameter(
                id=parameter_id,
                name=name,
                logical_shape=shape(*dimensions),
                binding="quantized",
                storage=packed,
                inspection=r.ArchitectureUnavailableInspection(
                    status="unavailable",
                    reason="unsupported_representation",
                    message="Packed W4A16 data is retained without decoding.",
                ),
                provenance=provenance,
            )
        elif (
            physical is None
            or physical.shape != list(dimensions)
            or physical.dtype not in {"F32", "F16", "BF16"}
        ):
            storage = [physical] if physical is not None else []
            return self._unresolved(
                name,
                dimensions,
                storage,
                "Required Kimi Linear parameter storage is absent or incompatible.",
            )
        else:
            self.used_storage.add(name)
            if numeric is not None:
                require(
                    numeric.shape == dimensions and numeric.dtype == physical.dtype,
                    "Kimi Linear numeric inventory disagrees with parameter geometry.",
                )
            if len(dimensions) not in (1, 2):
                inspection: r.ArchitectureInspection = r.ArchitectureUnavailableInspection(
                    status="unavailable",
                    reason="unsupported_rank",
                    message="Numeric inspection supports complete rank-1/rank-2 tensors.",
                )
            elif numeric is None:
                inspection = r.ArchitectureUnavailableInspection(
                    status="unavailable",
                    reason="unsupported_representation",
                    message="No logical numeric view is attached.",
                )
            else:
                inspection = r.ArchitectureAvailableInspection(
                    status="available", tensor_id=numeric.id
                )
            parameter = r.ArchitectureDirectParameter(
                id=parameter_id,
                name=name,
                logical_shape=shape(*dimensions),
                binding="native",
                storage=[physical],
                inspection=inspection,
                provenance=provenance,
            )
        self.b.add_parameter(parameter)
        return parameter_id

    def _group_state(
        self,
        key: str,
        operation: str,
        x: Value,
        prior: Value,
        state_shape: r.ArchitectureShape,
        parent: str,
        *,
        kernel_size: int,
        sequence_offsets: Value,
    ) -> tuple[Value, Value]:
        outputs = self.node(
            key,
            operation,
            {"x": x, "prior_state": prior, "sequence_offsets": sequence_offsets},
            {"out": x.shape, "next_state": state_shape},
            parent=parent,
            parameters=(key + ".weight",),
            attributes={"kernel_size": float(kernel_size), "activation": "silu", "causal": True},
            fields=("linear_attn_config",),
            formula=(
                "causal channel-wise short convolution with SiLU, respecting per-example "
                "sequence_offsets in the packed token sequence; prior/next history is symbolic "
                "and is not a captured activation"
            ),
            state_outputs=("next_state",),
        )
        return outputs["out"], outputs["next_state"]

    def kda_attention(
        self,
        layer: str,
        x: Value,
        padding_mask: Value,
        prior_q_conv: Value,
        prior_k_conv: Value,
        prior_v_conv: Value,
        prior_recurrent: Value,
    ) -> dict[str, Value]:
        c = self.c
        key = layer + ".self_attn"
        self.begin_group(key)
        h = int(c["hidden_size"])
        heads = int(c["linear_attn_config"]["num_heads"])
        dim = int(c["linear_attn_config"]["head_dim"])
        width = heads * dim
        kernel = int(c["linear_attn_config"]["short_conv_kernel_size"])
        hidden = shape("B", "S", h)
        projected_shape = shape(1, "T", width)
        packed_head_shape = shape(1, "T", heads, dim)
        conv_state_shape = shape("B", width, kernel - 1)
        recurrent_shape = shape("B", heads, dim, dim)
        current = Value(key, "x", hidden)
        compaction = self.node(
            key + ".input_compaction",
            "compact_valid_tokens",
            {
                "padded_hidden_states": current,
                "padding_mask": Value(key, "padding_mask", shape("B", "S")),
            },
            {
                "compact_hidden_states": shape(1, "T", h),
                "token_indices": shape("T"),
                "sequence_offsets": shape(_expression("B + 1", "B")),
            },
            parent=key,
            attributes={"packed_batch_size": 1.0},
            formula=(
                "compact each example's unmasked tokens into one packed sequence; emit original "
                "flattened token indices and per-example cumulative sequence offsets. T is the "
                "symbolic total valid-token count; no token values are evaluated"
            ),
        )
        packed_hidden = compaction["compact_hidden_states"]
        token_indices = compaction["token_indices"]
        sequence_offsets = compaction["sequence_offsets"]

        projections: dict[str, Value] = {}
        conv_states: dict[str, Value] = {}
        for branch, prior in (
            ("q", prior_q_conv),
            ("k", prior_k_conv),
            ("v", prior_v_conv),
        ):
            projection = self.linear(
                key + f".{branch}_proj",
                packed_hidden,
                h,
                width,
                key,
                fields=("hidden_size", "linear_attn_config"),
            )
            projections[branch], conv_states[branch] = self._group_state(
                key + f".{branch}_conv1d",
                "short_convolution",
                projection,
                Value(key, f"prior_{branch}_conv_state", prior.shape, "state"),
                conv_state_shape,
                key,
                kernel_size=kernel,
                sequence_offsets=sequence_offsets,
            )

        decay_low = self.linear(
            key + ".f_a_proj",
            packed_hidden,
            h,
            dim,
            key,
            fields=("hidden_size", "linear_attn_config"),
        )
        decay_raw = self.linear(
            key + ".f_b_proj",
            decay_low,
            dim,
            width,
            key,
            fields=("linear_attn_config",),
        )
        decay = self.op(
            key + ".fused_kda_decay",
            "kda_decay_gate",
            {"raw_gate": decay_raw},
            packed_head_shape,
            parent=key,
            parameters=(key + ".A_log", key + ".dt_bias"),
            attributes={"head_count": float(heads), "head_width": float(dim)},
            fields=("linear_attn_config",),
            formula="g = -exp(A_log) * softplus(f_b_proj(f_a_proj(x)) + dt_bias)",
        )
        beta_logits = self.linear(
            key + ".b_proj",
            packed_hidden,
            h,
            heads,
            key,
            fields=("hidden_size", "linear_attn_config"),
        )
        beta = self.op(
            key + ".beta_sigmoid",
            "sigmoid",
            {"x": beta_logits},
            shape(1, "T", heads),
            parent=key,
            formula="beta = sigmoid(b_proj(x))",
        )
        q_heads = self.op(
            key + ".q_heads",
            "reshape",
            {"x": projections["q"]},
            packed_head_shape,
            parent=key,
            attributes={"heads": float(heads), "head_width": float(dim)},
        )
        k_heads = self.op(
            key + ".k_heads",
            "reshape",
            {"x": projections["k"]},
            packed_head_shape,
            parent=key,
            attributes={"heads": float(heads), "head_width": float(dim)},
        )
        v_heads = self.op(
            key + ".v_heads",
            "reshape",
            {"x": projections["v"]},
            packed_head_shape,
            parent=key,
            attributes={"heads": float(heads), "head_width": float(dim)},
        )
        update = self.node(
            key + ".kda_delta_update",
            "kda_delta_state_update",
            {
                "q": q_heads,
                "k": k_heads,
                "v": v_heads,
                "decay": decay,
                "beta": beta,
                "prior_state": Value(key, "prior_recurrent_state", prior_recurrent.shape, "state"),
                "sequence_offsets": sequence_offsets,
            },
            {"out": packed_head_shape, "next_state": recurrent_shape},
            parent=key,
            attributes={"qk_l2_normalized": True, "head_count": float(heads)},
            fields=("linear_attn_config", "moe_intermediate_size"),
            formula=(
                "S'_t = exp(g_t) ⊙ S_{t-1}; r_t = v_t - k_tᵀS'_t; "
                "S_t = S'_t + beta_t k_t r_tᵀ; y_t = q_tᵀS_t, "
                "with normalized q/k and symbolic prior/next state"
            ),
            state_outputs=("next_state",),
        )
        output_gate_low = self.linear(
            key + ".g_a_proj",
            packed_hidden,
            h,
            dim,
            key,
            fields=("hidden_size", "linear_attn_config"),
        )
        output_gate = self.linear(
            key + ".g_b_proj",
            output_gate_low,
            dim,
            width,
            key,
            fields=("linear_attn_config",),
        )
        output_gate_heads = self.op(
            key + ".output_gate_heads",
            "reshape",
            {"x": output_gate},
            packed_head_shape,
            parent=key,
            attributes={"heads": float(heads), "head_width": float(dim)},
        )
        normalized = self.op(
            key + ".o_norm",
            "rms_norm_gated",
            {
                "x": Value(update["out"].node, "out", packed_head_shape),
                "gate": output_gate_heads,
            },
            packed_head_shape,
            parent=key,
            parameters=(key + ".o_norm.weight",),
            attributes={"epsilon": float(c["rms_norm_eps"]), "gate_activation": "sigmoid"},
            fields=("rms_norm_eps",),
            formula="RMSNorm(KDA output) * sigmoid(output gate)",
        )
        merged = self.op(
            key + ".merge_heads",
            "reshape",
            {"x": normalized},
            projected_shape,
            parent=key,
            attributes={"heads": float(heads), "head_width": float(dim)},
        )
        result = self.linear(
            key + ".o_proj",
            merged,
            width,
            h,
            key,
            fields=("hidden_size", "linear_attn_config"),
        )
        padded_result = self.op(
            key + ".output_repadding",
            "restore_padding",
            {"compact_output": result, "token_indices": token_indices},
            hidden,
            parent=key,
            attributes={"padding_value": 0.0},
            formula=(
                "scatter compact output rows back to their original batch/sequence positions "
                "using token_indices and fill masked positions with zero"
            ),
        )
        grouped = self.end_group(
            key,
            "Kimi Delta Attention",
            layer,
            {
                "x": x,
                "padding_mask": padding_mask,
                "prior_q_conv_state": Value(
                    layer, "prior_q_conv_state", prior_q_conv.shape, "state"
                ),
                "prior_k_conv_state": Value(
                    layer, "prior_k_conv_state", prior_k_conv.shape, "state"
                ),
                "prior_v_conv_state": Value(
                    layer, "prior_v_conv_state", prior_v_conv.shape, "state"
                ),
                "prior_recurrent_state": Value(
                    layer, "prior_recurrent_state", prior_recurrent.shape, "state"
                ),
            },
            {
                "out": padded_result,
                "next_q_conv_state": conv_states["q"],
                "next_k_conv_state": conv_states["k"],
                "next_v_conv_state": conv_states["v"],
                "next_recurrent_state": Value(
                    update["next_state"].node, "next_state", recurrent_shape, "state"
                ),
            },
            role="attention",
            attributes={
                "attention_type": "Kimi Delta Attention (KDA)",
                "head_count": float(heads),
                "head_dimension": float(dim),
                "short_convolution_kernel_size": float(kernel),
            },
            fields=("linear_attn_config", "hidden_size", "rms_norm_eps"),
        )
        return grouped

    def mla_attention(
        self,
        layer: str,
        x: Value,
        causal_mask: Value,
        prior_key: Value,
        prior_value: Value,
    ) -> dict[str, Value]:
        c = self.c
        key = layer + ".self_attn"
        self.begin_group(key)
        hidden = int(c["hidden_size"])
        heads = int(c["num_attention_heads"])
        kv_rank = int(c["kv_lora_rank"])
        nope = int(c["qk_nope_head_dim"])
        rope = int(c["qk_rope_head_dim"])
        value_dim = int(c["v_head_dim"])
        q_width = nope + rope
        q_heads = shape("B", heads, "S", q_width)
        q_nope = shape("B", heads, "S", nope)
        q_rope = shape("B", heads, "S", rope)
        latent = shape("B", "S", kv_rank)
        key_rope = shape("B", 1, "S", rope)
        key_rope_heads = shape("B", heads, "S", rope)
        key_heads = shape("B", heads, "S", q_width)
        value_heads = shape("B", heads, "S", value_dim)
        cached_key = shape("B", heads, _expression("K + S", "K", "S"), q_width)
        cached_value = shape("B", heads, _expression("K + S", "K", "S"), value_dim)
        current = Value(key, "x", shape("B", "S", hidden))

        q_projected = self.linear(
            key + ".q_proj",
            current,
            hidden,
            heads * q_width,
            key,
            fields=("hidden_size", "num_attention_heads", "qk_nope_head_dim", "qk_rope_head_dim"),
        )
        q_by_head = self.op(
            key + ".q_heads",
            "reshape_transpose",
            {"x": q_projected},
            q_heads,
            parent=key,
            attributes={"heads": float(heads), "head_width": float(q_width)},
        )
        q_split = self.node(
            key + ".q_split",
            "split",
            {"x": q_by_head},
            {"non_rotary": q_nope, "unrotated_rope_slice": q_rope},
            parent=key,
            fields=("qk_nope_head_dim", "qk_rope_head_dim"),
            formula="split the configured 128/64 query subspaces; MLA NoPE applies no rotation",
        )
        query = self.op(
            key + ".query_recombine",
            "concat",
            {
                "non_rotary": q_split["non_rotary"],
                "unrotated_rope_slice": q_split["unrotated_rope_slice"],
            },
            q_heads,
            parent=key,
            attributes={"axis": -1.0, "positional_encoding": "none"},
            fields=("mla_use_nope", "qk_rope_head_dim"),
            formula="concatenate the unchanged query slices; mla_use_nope=true",
        )

        kv_projected = self.linear(
            key + ".kv_a_proj_with_mqa",
            current,
            hidden,
            kv_rank + rope,
            key,
            fields=("hidden_size", "kv_lora_rank", "qk_rope_head_dim"),
        )
        kv_split = self.node(
            key + ".kv_a_split",
            "split",
            {"x": kv_projected},
            {"latent": latent, "unrotated_rope_slice": shape("B", "S", rope)},
            parent=key,
            attributes={"latent_rank": float(kv_rank), "head_slice_width": float(rope)},
            fields=("kv_lora_rank", "qk_rope_head_dim"),
        )
        compressed = self.norm(
            key + ".kv_a_layernorm",
            kv_split["latent"],
            kv_rank,
            key,
            fields=("kv_lora_rank", "rms_norm_eps"),
        )
        expanded = self.linear(
            key + ".kv_b_proj",
            compressed,
            kv_rank,
            heads * (nope + value_dim),
            key,
            fields=("kv_lora_rank", "num_attention_heads", "qk_nope_head_dim", "v_head_dim"),
        )
        expanded_heads = self.op(
            key + ".kv_heads",
            "reshape_transpose",
            {"x": expanded},
            shape("B", heads, "S", nope + value_dim),
            parent=key,
            attributes={"heads": float(heads)},
        )
        kv_b_split = self.node(
            key + ".kv_b_split",
            "split",
            {"x": expanded_heads},
            {"key_non_rotary": shape("B", heads, "S", nope), "value": value_heads},
            parent=key,
            fields=("qk_nope_head_dim", "v_head_dim"),
        )
        key_rope_one = self.op(
            key + ".key_rope_heads",
            "reshape_transpose",
            {"x": kv_split["unrotated_rope_slice"]},
            key_rope,
            parent=key,
            attributes={"heads": 1.0, "positional_encoding": "none"},
        )
        key_rope_repeated = self.op(
            key + ".key_rope_expand",
            "expand_heads",
            {"x": key_rope_one},
            key_rope_heads,
            parent=key,
            attributes={"heads": float(heads)},
            fields=("num_attention_heads",),
            formula="broadcast the shared, unrotated key slice to every attention head",
        )
        full_key = self.op(
            key + ".key_recombine",
            "concat",
            {"non_rotary": kv_b_split["key_non_rotary"], "unrotated_rope_slice": key_rope_repeated},
            key_heads,
            parent=key,
            attributes={"axis": -1.0, "positional_encoding": "none"},
            fields=("mla_use_nope", "qk_nope_head_dim", "qk_rope_head_dim"),
            formula="concatenate the reconstructed 128-wide key and unchanged 64-wide slice",
        )
        keys = self.op(
            key + ".key_cache_update",
            "append_cache_state",
            {
                "prior_state": Value(key, "prior_key_state", prior_key.shape, "state"),
                "current": full_key,
            },
            cached_key,
            parent=key,
            state_output=True,
            formula="symbolically append current keys to prior per-layer MLA state",
        )
        values = self.op(
            key + ".value_cache_update",
            "append_cache_state",
            {
                "prior_state": Value(key, "prior_value_state", prior_value.shape, "state"),
                "current": kv_b_split["value"],
            },
            cached_value,
            parent=key,
            state_output=True,
            formula="symbolically append current values to prior per-layer MLA state",
        )
        key_transpose = self.op(
            key + ".key_transpose",
            "transpose",
            {"x": keys},
            shape("B", heads, q_width, _expression("K + S", "K", "S")),
            parent=key,
            attributes={"axes": "0,1,3,2"},
        )
        scores = shape("B", heads, "S", _expression("K + S", "K", "S"))
        product = self.op(
            key + ".qk_product",
            "matmul",
            {"q": query, "kt": key_transpose},
            scores,
            parent=key,
        )
        scaled = self.op(
            key + ".scale",
            "scale",
            {"x": product},
            scores,
            parent=key,
            attributes={"factor": q_width**-0.5},
            fields=("qk_nope_head_dim", "qk_rope_head_dim"),
            formula="scores / sqrt(128 + 64)",
        )
        masked = self.op(
            key + ".causal_mask",
            "add_mask",
            {"x": scaled, "mask": Value(key, "causal_mask", causal_mask.shape)},
            scores,
            parent=key,
            attributes={"causal": True},
        )
        probabilities = self.op(
            key + ".softmax",
            "softmax",
            {"x": masked},
            scores,
            parent=key,
            attributes={"axis": -1.0},
        )
        attended = self.op(
            key + ".weighted_values",
            "matmul",
            {"probabilities": probabilities, "value": values},
            value_heads,
            parent=key,
        )
        transposed = self.op(
            key + ".output_transpose",
            "transpose",
            {"x": attended},
            shape("B", "S", heads, value_dim),
            parent=key,
            attributes={"axes": "0,2,1,3"},
        )
        merged = self.op(
            key + ".merge_heads",
            "reshape",
            {"x": transposed},
            shape("B", "S", heads * value_dim),
            parent=key,
        )
        projected = self.linear(
            key + ".o_proj",
            merged,
            heads * value_dim,
            hidden,
            key,
            fields=("num_attention_heads", "v_head_dim", "hidden_size"),
        )
        return self.end_group(
            key,
            "Multi-head Latent Attention (NoPE)",
            layer,
            {
                "x": x,
                "causal_mask": causal_mask,
                "prior_key_state": Value(layer, "prior_key_state", prior_key.shape, "state"),
                "prior_value_state": Value(layer, "prior_value_state", prior_value.shape, "state"),
            },
            {
                "out": projected,
                "next_key_state": Value(keys.node, "out", cached_key, "state"),
                "next_value_state": Value(values.node, "out", cached_value, "state"),
            },
            role="attention",
            attributes={
                "attention_type": "Multi-head Latent Attention (MLA)",
                "positional_encoding": "none",
                "kv_lora_rank": float(kv_rank),
                "query_head_dimension": float(q_width),
                "causal": True,
            },
            fields=(
                "mla_use_nope",
                "kv_lora_rank",
                "qk_nope_head_dim",
                "qk_rope_head_dim",
                "v_head_dim",
            ),
        )

    def _compact_expert(
        self,
        key: str,
        index: int,
        parent: str,
        width: int,
    ) -> None:
        """Retain one compact, fully bound repeated component for each expert identity."""
        node_id = self.nid(key)
        self.children.setdefault(key, [])
        parameters = [
            self.parameter_ids[key + ".w1.weight"],
            self.parameter_ids[key + ".w3.weight"],
            self.parameter_ids[key + ".w2.weight"],
        ]
        self.children.setdefault(parent, []).append(node_id)
        self.b.add_node(
            r.ArchitectureGroupNode(
                id=node_id,
                parent_id=self.nid(parent),
                kind="group",
                label=f"Routed expert {index}",
                operation="weighted_swiglu_mlp",
                formula=(
                    "weighted w2(silu(w1(x)) * w3(x)); the enclosing MoE operation "
                    "binds this expert's symbolic dispatch and routing weight"
                ),
                ports=[],
                children=[],
                parameter_ids=parameters,
                references=[],
                attributes=[
                    r.ArchitectureAttribute(
                        name="expert_index",
                        value=float(index),
                        provenance=self.provenance("num_experts"),
                    ),
                    r.ArchitectureAttribute(
                        name="intermediate_width",
                        value=float(width),
                        provenance=self.provenance("moe_intermediate_size"),
                    ),
                    self.role_attribute("mlp"),
                ],
                provenance=self.provenance("num_experts", "moe_intermediate_size", "hidden_act"),
            ),
            semantic_key=key,
        )

    def kimi_shared_mlp(self, parent: str, x: Value) -> Value:
        """Describe Kimi's shared expert using its direct checkpoint parameter paths."""
        key = parent + ".shared_experts"
        hidden = int(self.c["hidden_size"])
        width = int(self.c["moe_intermediate_size"]) * int(self.c["num_shared_experts"])
        self.begin_group(key)
        gate = self.linear(
            key + ".gate_proj",
            Value(key, "x", x.shape),
            hidden,
            width,
            key,
            fields=("hidden_size", "moe_intermediate_size", "num_shared_experts"),
        )
        up = self.linear(
            key + ".up_proj",
            Value(key, "x", x.shape),
            hidden,
            width,
            key,
            fields=("hidden_size", "moe_intermediate_size", "num_shared_experts"),
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
            hidden,
            key,
            fields=("hidden_size", "moe_intermediate_size", "num_shared_experts"),
        )
        result = self.end_group(
            key,
            "Shared SwiGLU Expert",
            parent,
            {"x": x},
            {"out": down},
            role="mlp",
            attributes={"intermediate_width": float(width)},
            fields=("hidden_size", "moe_intermediate_size", "num_shared_experts", "hidden_act"),
        )
        return result["out"]

    def moe_mlp(self, layer: str, x: Value) -> Value:
        c = self.c
        key = layer + ".block_sparse_moe"
        self.begin_group(key)
        hidden = int(c["hidden_size"])
        count = int(c["num_experts"])
        top_k = int(c["num_experts_per_token"])
        width = int(c["moe_intermediate_size"])
        score_shape = shape("B", "S", count)
        router = self.linear(
            key + ".gate",
            Value(key, "x", x.shape),
            hidden,
            count,
            key,
            fields=("hidden_size", "num_experts", "moe_router_activation_func"),
            attributes={
                "bias": False,
                "router_weight_layout": "experts_by_hidden",
                "computation_dtype": "float32",
            },
        )
        scores = self.op(
            key + ".router_sigmoid",
            "sigmoid",
            {"x": router},
            score_shape,
            parent=key,
            attributes={"score_type": "float32 sigmoid"},
            fields=("moe_router_activation_func",),
            formula="independent sigmoid router score for each configured expert",
        )
        selection = self.node(
            key + ".topk",
            "kimi_grouped_topk_selection",
            {"scores": scores},
            {"expert_indices": shape("B", "S", top_k), "expert_weights": shape("B", "S", top_k)},
            parent=key,
            parameters=(key + ".gate.e_score_correction_bias",),
            attributes={
                "top_k": float(top_k),
                "expert_group_count": float(c["num_expert_group"]),
                "topk_group": float(c["topk_group"]),
                "renormalize": bool(c["moe_renormalize"]),
                "routed_scaling_factor": float(c["routed_scaling_factor"]),
                "selected_weight_rule": "correction-adjusted sigmoid scores",
                "selection_is_runtime_data": True,
            },
            fields=(
                "num_experts",
                "num_experts_per_token",
                "num_expert_group",
                "topk_group",
                "moe_renormalize",
                "routed_scaling_factor",
            ),
            formula=(
                "add e_score_correction_bias in-place through scores_for_choice, a view of "
                "the sigmoid score tensor; rank the single all-expert group by the sum of its "
                "two highest corrected scores and select the top 8. Gather the mutated "
                "correction-adjusted sigmoid scores, renormalize them, then multiply by "
                "routed_scaling_factor 2.446"
            ),
        )

        instances: list[r.ArchitectureRepetitionInstance] = []
        for expert in range(count):
            expert_key = f"{key}.experts.{expert}"
            self._compact_expert(expert_key, expert, key, width)
            instances.append(
                r.ArchitectureRepetitionInstance(
                    node_id=self.nid(expert_key), index=expert, variant="routed_swiglu"
                )
            )

        scattered = self.op(
            key + ".dispatch_execute_scatter_sum",
            "symbolic_expert_dispatch_execute_scatter_sum",
            {
                "hidden_states": Value(key, "x", x.shape),
                "expert_indices": Value(
                    selection["expert_indices"].node,
                    "expert_indices",
                    selection["expert_indices"].shape,
                ),
                "expert_weights": Value(
                    selection["expert_weights"].node,
                    "expert_weights",
                    selection["expert_weights"].shape,
                ),
            },
            shape("B", "S", hidden),
            parent=key,
            attributes={
                "expert_count": float(count),
                "top_k": float(top_k),
                "selection_is_runtime_data": True,
                "expert_instances": key + ".experts.0.." + str(count - 1),
            },
            fields=("num_experts", "num_experts_per_token"),
            formula=(
                "for each expert e, symbolically dispatch selected tokens and source positions; "
                "compute weight[e] * w2[e](silu(w1[e](x)) * w3[e](x)); scatter-add weighted "
                "results to original token positions. Expert parameters are bound by the "
                "matching experts.e repeated group; no routes or tokens are evaluated"
            ),
        )
        shared = self.kimi_shared_mlp(key, Value(key, "x", x.shape))
        combined = self.op(
            key + ".shared_routed_add",
            "add",
            {"routed": scattered, "shared": shared},
            shape("B", "S", hidden),
            parent=key,
            attributes={"shared_expert_count": float(c["num_shared_experts"])},
            fields=("num_shared_experts",),
            formula="weighted routed-expert scatter sum + one shared SwiGLU expert",
        )
        result = self.end_group(
            key,
            "Kimi Sparse MoE",
            layer,
            {"x": x},
            {"out": combined},
            role="mlp",
            attributes={
                "routed_expert_count": float(count),
                "top_k": float(top_k),
                "shared_expert_count": float(c["num_shared_experts"]),
                "routing_is_symbolic": True,
            },
            fields=(
                "num_experts",
                "num_experts_per_token",
                "num_shared_experts",
                "first_k_dense_replace",
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

    def decoder_layer(
        self,
        index: int,
        incoming: Value,
        padding_mask: Value,
        causal_mask: Value,
        states: dict[str, Value],
    ) -> dict[str, Value]:
        c = self.c
        key = f"model.layers.{index}"
        hidden = shape("B", "S", int(c["hidden_size"]))
        attention_type = c["layer_types"][index]
        self.begin_group(key)
        x = Value(key, "x", hidden)
        normalized = self.norm(
            key + ".input_layernorm",
            x,
            int(c["hidden_size"]),
            key,
            fields=("hidden_size", "rms_norm_eps"),
        )
        if attention_type == "kda":
            attention = self.kda_attention(
                key,
                normalized,
                Value(key, "padding_mask", padding_mask.shape),
                states["prior_q_conv_state"],
                states["prior_k_conv_state"],
                states["prior_v_conv_state"],
                states["prior_recurrent_state"],
            )
            attention_residual = self.op(
                key + ".attention_residual",
                "add",
                {"skip": x, "branch": attention["out"]},
                hidden,
                parent=key,
                formula="input hidden states + KDA output",
            )
            post_norm = self.norm(
                key + ".post_attention_layernorm",
                attention_residual,
                int(c["hidden_size"]),
                key,
                fields=("hidden_size", "rms_norm_eps"),
            )
            feed_forward = (
                self.dense_mlp(key, post_norm) if index == 0 else self.moe_mlp(key, post_norm)
            )
            output = self.op(
                key + ".mlp_residual",
                "add",
                {"skip": attention_residual, "branch": feed_forward},
                hidden,
                parent=key,
                formula="post-attention residual stream + dense or sparse MoE output",
            )
            group_inputs = {
                "x": incoming,
                "padding_mask": padding_mask,
                **states,
            }
            group_outputs = {
                "out": output,
                "next_q_conv_state": attention["next_q_conv_state"],
                "next_k_conv_state": attention["next_k_conv_state"],
                "next_v_conv_state": attention["next_v_conv_state"],
                "next_recurrent_state": attention["next_recurrent_state"],
            }
        else:
            attention = self.mla_attention(
                key,
                normalized,
                Value(key, "causal_mask", causal_mask.shape),
                states["prior_key_state"],
                states["prior_value_state"],
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
            feed_forward = self.moe_mlp(key, post_norm)
            output = self.op(
                key + ".mlp_residual",
                "add",
                {"skip": attention_residual, "branch": feed_forward},
                hidden,
                parent=key,
                formula="post-attention residual stream + sparse MoE output",
            )
            group_inputs = {
                "x": incoming,
                "causal_mask": causal_mask,
                "prior_key_state": states["prior_key_state"],
                "prior_value_state": states["prior_value_state"],
            }
            group_outputs = {
                "out": output,
                "next_key_state": attention["next_key_state"],
                "next_value_state": attention["next_value_state"],
            }
        result = self.end_group(
            key,
            f"Decoder layer {index}",
            "model",
            group_inputs,
            group_outputs,
            attributes={
                "sequence_index": float(index),
                "attention_type": "KDA" if attention_type == "kda" else "MLA",
                "feed_forward_path": "dense" if index == 0 else "moe",
            },
            fields=("linear_attn_config", "first_k_dense_replace", "moe_layer_freq"),
        )
        return result

    def _state_select(
        self,
        key: str,
        bank: Value,
        output_shape: r.ArchitectureShape,
        layer_index: int,
        ordinal: int,
        state_role: str,
    ) -> Value:
        return self.op(
            key,
            "select_layer_state",
            {"state_bank": bank},
            output_shape,
            parent="model",
            attributes={
                "layer_index": float(layer_index),
                "state_index": float(ordinal),
                "state_role": state_role,
            },
            fields=("num_hidden_layers", "linear_attn_config"),
            state_output=True,
        )

    def _state_stack(
        self,
        key: str,
        states: list[Value],
        output_shape: r.ArchitectureShape,
        state_role: str,
    ) -> Value:
        return self.node(
            key,
            "stack_layer_states",
            {f"layer_{index}": value for index, value in enumerate(states)},
            {"out": output_shape},
            parent="model",
            attributes={"layer_count": float(len(states)), "state_role": state_role},
            fields=("num_hidden_layers", "linear_attn_config"),
            state_outputs=("out",),
        )["out"]

    def build(self) -> None:
        c = self.c
        hidden_width = int(c["hidden_size"])
        layers = int(c["num_hidden_layers"])
        kda_layers = [i for i, value in enumerate(c["layer_types"]) if value == "kda"]
        mla_layers = [i for i, value in enumerate(c["layer_types"]) if value == "full_attention"]
        heads = int(c["linear_attn_config"]["num_heads"])
        dim = int(c["linear_attn_config"]["head_dim"])
        width = heads * dim
        kernel = int(c["linear_attn_config"]["short_conv_kernel_size"])
        q_width = int(c["qk_nope_head_dim"]) + int(c["qk_rope_head_dim"])
        value_width = int(c["v_head_dim"])
        vocab = int(c["vocab_size"])
        sequence = shape("B", "S")
        hidden = shape("B", "S", hidden_width)
        kda_conv_bank = shape(len(kda_layers), "B", width, kernel - 1)
        kda_recurrent_bank = shape(len(kda_layers), "B", heads, dim, dim)
        next_mla_key_bank = shape(
            len(mla_layers),
            "B",
            int(c["num_attention_heads"]),
            _expression("K + S", "K", "S"),
            q_width,
        )
        next_mla_value_bank = shape(
            len(mla_layers),
            "B",
            int(c["num_attention_heads"]),
            _expression("K + S", "K", "S"),
            value_width,
        )
        prior_mla_key_bank = shape(
            len(mla_layers), "B", int(c["num_attention_heads"]), "K", q_width
        )
        prior_mla_value_bank = shape(
            len(mla_layers), "B", int(c["num_attention_heads"]), "K", value_width
        )
        causal_mask = shape("B", 1, "S", _expression("K + S", "K", "S"))
        self.b.add_symbol("B", "Symbolic batch size; no input has been executed.")
        self.b.add_symbol("S", "Symbolic current sequence length; no prompt has been supplied.")
        self.b.add_symbol("K", "Symbolic prior MLA key/value length; no cache sample exists.")
        self.b.add_symbol("T", "Symbolic count of unmasked KDA tokens after sequence compaction.")

        root_inputs: dict[str, Value] = {
            "input_ids": Value("model", "input_ids", sequence),
            "padding_mask": Value("model", "padding_mask", sequence),
            "causal_mask": Value("model", "causal_mask", causal_mask),
            "prior_kda_q_conv_state": Value(
                "model", "prior_kda_q_conv_state", kda_conv_bank, "state"
            ),
            "prior_kda_k_conv_state": Value(
                "model", "prior_kda_k_conv_state", kda_conv_bank, "state"
            ),
            "prior_kda_v_conv_state": Value(
                "model", "prior_kda_v_conv_state", kda_conv_bank, "state"
            ),
            "prior_kda_recurrent_state": Value(
                "model", "prior_kda_recurrent_state", kda_recurrent_bank, "state"
            ),
            "prior_mla_key_state": Value(
                "model", "prior_mla_key_state", prior_mla_key_bank, "state"
            ),
            "prior_mla_value_state": Value(
                "model", "prior_mla_value_state", prior_mla_value_bank, "state"
            ),
        }
        tokens = self.op(
            "model.embed_tokens",
            "embedding",
            {"input_ids": root_inputs["input_ids"]},
            hidden,
            parent="model",
            parameters=("model.embed_tokens.weight",),
            attributes={"vocabulary_size": float(vocab), "hidden_size": float(hidden_width)},
            fields=("vocab_size", "hidden_size"),
        )

        selected: dict[int, dict[str, Value]] = {}
        kda_state_banks = {
            "prior_q_conv_state": root_inputs["prior_kda_q_conv_state"],
            "prior_k_conv_state": root_inputs["prior_kda_k_conv_state"],
            "prior_v_conv_state": root_inputs["prior_kda_v_conv_state"],
        }
        for ordinal, index in enumerate(kda_layers):
            states = {
                name: self._state_select(
                    f"model.kda_state.{name}.{index}",
                    bank,
                    shape("B", width, kernel - 1),
                    index,
                    ordinal,
                    name,
                )
                for name, bank in kda_state_banks.items()
            }
            states["prior_recurrent_state"] = self._state_select(
                f"model.kda_state.recurrent.{index}",
                root_inputs["prior_kda_recurrent_state"],
                shape("B", heads, dim, dim),
                index,
                ordinal,
                "prior recurrent state",
            )
            selected[index] = states
        for ordinal, index in enumerate(mla_layers):
            selected[index] = {
                "prior_key_state": self._state_select(
                    f"model.mla_state.key.{index}",
                    root_inputs["prior_mla_key_state"],
                    shape("B", int(c["num_attention_heads"]), "K", q_width),
                    index,
                    ordinal,
                    "prior key state",
                ),
                "prior_value_state": self._state_select(
                    f"model.mla_state.value.{index}",
                    root_inputs["prior_mla_value_state"],
                    shape("B", int(c["num_attention_heads"]), "K", value_width),
                    index,
                    ordinal,
                    "prior value state",
                ),
            }

        next_q_conv: list[Value] = []
        next_k_conv: list[Value] = []
        next_v_conv: list[Value] = []
        next_recurrent: list[Value] = []
        next_keys: list[Value] = []
        next_values: list[Value] = []
        layer_instances: list[r.ArchitectureRepetitionInstance] = []
        previous = tokens
        for index in range(layers):
            built = self.decoder_layer(
                index,
                previous,
                root_inputs["padding_mask"],
                root_inputs["causal_mask"],
                selected[index],
            )
            previous = built["out"]
            if c["layer_types"][index] == "kda":
                next_q_conv.append(built["next_q_conv_state"])
                next_k_conv.append(built["next_k_conv_state"])
                next_v_conv.append(built["next_v_conv_state"])
                next_recurrent.append(built["next_recurrent_state"])
            else:
                next_keys.append(built["next_key_state"])
                next_values.append(built["next_value_state"])
            variant = c["layer_types"][index] + ("_dense" if index == 0 else "_moe")
            layer_instances.append(
                r.ArchitectureRepetitionInstance(
                    node_id=self.nid(f"model.layers.{index}"), index=index, variant=variant
                )
            )

        final_norm = self.norm(
            "model.norm",
            previous,
            hidden_width,
            "model",
            fields=("hidden_size", "rms_norm_eps"),
        )
        logits = self.linear(
            "lm_head",
            final_norm,
            hidden_width,
            vocab,
            "model",
            fields=("hidden_size", "vocab_size", "tie_word_embeddings"),
        )
        next_q = self._state_stack(
            "model.next_kda_q_conv_state", next_q_conv, kda_conv_bank, "next q convolution state"
        )
        next_k = self._state_stack(
            "model.next_kda_k_conv_state", next_k_conv, kda_conv_bank, "next k convolution state"
        )
        next_v = self._state_stack(
            "model.next_kda_v_conv_state", next_v_conv, kda_conv_bank, "next v convolution state"
        )
        next_r = self._state_stack(
            "model.next_kda_recurrent_state",
            next_recurrent,
            kda_recurrent_bank,
            "next recurrent state",
        )
        next_key = self._state_stack(
            "model.next_mla_key_state", next_keys, next_mla_key_bank, "next MLA keys"
        )
        next_value = self._state_stack(
            "model.next_mla_value_state", next_values, next_mla_value_bank, "next MLA values"
        )
        self.end_group(
            "model",
            "Kimi Linear CausalLM",
            None,
            root_inputs,
            {
                "logits": logits,
                "next_kda_q_conv_state": next_q,
                "next_kda_k_conv_state": next_k,
                "next_kda_v_conv_state": next_v,
                "next_kda_recurrent_state": next_r,
                "next_mla_key_state": next_key,
                "next_mla_value_state": next_value,
            },
            attributes={
                "architecture": ARCHITECTURE,
                "static_evaluation_path": True,
                "position_encoding": "none",
            },
            fields=("architectures", "model_type", "num_hidden_layers", "mla_use_nope"),
        )
        self.b.add_repetition(
            r.ArchitectureRepetition(
                id=self.b.record_id("repetition", "model.layers"),
                parent_id=self.nid("model"),
                label="Decoder layers",
                instances=layer_instances,
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
                    message="Checkpoint contains storage outside the reviewed Kimi Linear graph.",
                )
            )


def build(inputs: AnalysisInput, builder: GraphBuilder) -> None:
    configuration = checked(inputs.configuration)
    require(configuration is not None, "Unsupported Kimi Linear configuration.")
    assert configuration is not None
    KimiLinearGraph(inputs, builder, configuration).build()


def register_kimi_linear(registry: DescriptionRegistry) -> None:
    registry.register(
        Description(
            producer=PRODUCER,
            scope="language_model",
            model_types=frozenset({MODEL_TYPE}),
            architectures=frozenset({ARCHITECTURE}),
            supports=supports,
            build=build,
            compact_experts=True,
        )
    )

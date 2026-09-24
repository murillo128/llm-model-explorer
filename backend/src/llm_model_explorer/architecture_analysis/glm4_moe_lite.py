"""Verified static GLM-4.7-Flash description; checkpoint code is never imported."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from . import records as r
from .core import AnalysisInput, Description, DescriptionRegistry, GraphBuilder, Producer
from .deepseek_v2 import DeepseekGraph, Value, shape
from .validation import require

SOURCE_REVISION = (
    "huggingface/transformers@c8b81b63232be35ab1774dd3cabbf499d8b9808f; "
    "cyankiwi/GLM-4.7-Flash-AWQ-4bit@25624b53414e585bcf7dcb9584667c3106c6089b"
)
PRODUCER = Producer("glm4-moe-lite", "1", SOURCE_REVISION)
ARCHITECTURE = "Glm4MoeLiteForCausalLM"
MODEL_TYPE = "glm4_moe_lite"

# These are the structural values in the reviewed GLM-4.7-Flash checkpoint.
# Source defaults are accepted only for fields explicitly listed in DEFAULTS.
REFERENCE: dict[str, Any] = {
    "vocab_size": 154880,
    "hidden_size": 2048,
    "intermediate_size": 10240,
    "moe_intermediate_size": 1536,
    "num_hidden_layers": 47,
    "num_attention_heads": 20,
    "num_key_value_heads": 20,
    "n_shared_experts": 1,
    "n_routed_experts": 64,
    "routed_scaling_factor": 1.8,
    "kv_lora_rank": 512,
    "q_lora_rank": 768,
    "qk_rope_head_dim": 64,
    "v_head_dim": 256,
    "qk_nope_head_dim": 192,
    "n_group": 1,
    "topk_group": 1,
    "num_experts_per_tok": 4,
    "norm_topk_prob": True,
    "hidden_act": "silu",
    "max_position_embeddings": 202752,
    "rms_norm_eps": 1e-5,
    "tie_word_embeddings": False,
    "rope_interleave": True,
    "attention_bias": False,
    "attention_dropout": 0.0,
    "partial_rotary_factor": 1.0,
    "num_nextn_predict_layers": 1,
    "topk_method": "noaux_tc",
    "pretraining_tp": 1,
    "use_cache": True,
    "first_k_dense_replace": 1,
    "head_dim": 64,
    "qk_head_dim": 256,
}
DEFAULTS = {
    "n_shared_experts": 1,
    "routed_scaling_factor": 1.8,
    "q_lora_rank": 768,
    "kv_lora_rank": 512,
    "qk_rope_head_dim": 64,
    "v_head_dim": 256,
    "qk_nope_head_dim": 192,
    "n_group": 1,
    "topk_group": 1,
    "num_experts_per_tok": 4,
    "norm_topk_prob": True,
    "hidden_act": "silu",
    "max_position_embeddings": 202752,
    "rms_norm_eps": 1e-5,
    "tie_word_embeddings": False,
    "rope_interleave": True,
    "attention_bias": False,
    "attention_dropout": 0.0,
    "partial_rotary_factor": 1.0,
    "pretraining_tp": 1,
    "use_cache": True,
}
OPTIONAL_METADATA = {
    "_commit_hash",
    "_name_or_path",
    "bos_token_id",
    "eos_token_id",
    "pad_token_id",
    "dtype",
    "torch_dtype",
    "initializer_range",
    "transformers_version",
    "quantization_config",
    "rope_scaling",
    "rope_theta",
    "mlp_layer_types",
    "rope_parameters",
    "first_k_dense_replace",
    "head_dim",
    "qk_head_dim",
}
OPTIONAL_METADATA |= set(REFERENCE) - set(DEFAULTS)
LAYER_TYPES = ["dense"] + ["sparse"] * (REFERENCE["num_hidden_layers"] - 1)
ROPE_PARAMETERS = {
    "partial_rotary_factor": 1.0,
    "rope_theta": 1_000_000,
    "rope_type": "default",
}
COMPRESSED_FORMAT = "compressed-tensors-w4a16-int4"


def checked(configuration: Mapping[str, object]) -> dict[str, Any] | None:
    """Accept only the reviewed layer, latent-attention, routing and RoPE options."""
    raw = dict(configuration)
    if set(raw) - (set(REFERENCE) | OPTIONAL_METADATA | {"model_type", "architectures"}):
        return None
    if raw.get("model_type") != MODEL_TYPE or raw.get("architectures") != [ARCHITECTURE]:
        return None

    normalized = dict(raw)
    for key, expected in REFERENCE.items():
        actual = raw.get(key, DEFAULTS.get(key))
        if type(actual) is not type(expected) or actual != expected:
            return None
        normalized[key] = actual

    # The reviewed Transformers config creates exactly this list when absent.
    layer_types = raw.get("mlp_layer_types", LAYER_TYPES)
    if (
        not isinstance(layer_types, list)
        or len(layer_types) != REFERENCE["num_hidden_layers"]
        or layer_types != LAYER_TYPES
    ):
        return None
    normalized["mlp_layer_types"] = list(layer_types)

    rope = raw.get("rope_parameters", ROPE_PARAMETERS)
    if not isinstance(rope, dict) or set(rope) != set(ROPE_PARAMETERS):
        return None
    if any(type(rope[k]) is not type(v) or rope[k] != v for k, v in ROPE_PARAMETERS.items()):
        return None
    if raw.get("rope_scaling") is not None or raw.get("rope_theta") is not None:
        return None
    normalized["rope_parameters"] = dict(rope)

    # first_k_dense_replace is conversion metadata; it must agree with the
    # explicit pattern and cannot silently change the checked layer topology.
    first_dense = raw.get("first_k_dense_replace", 1)
    if type(first_dense) is not int or first_dense != 1:
        return None
    quantization = raw.get("quantization_config")
    if quantization is not None and not isinstance(quantization, dict):
        return None
    for key in ("dtype", "torch_dtype", "transformers_version", "_name_or_path", "_commit_hash"):
        value = raw.get(key)
        if value is not None and not isinstance(value, str):
            return None

    if (
        raw.get("partial_rotary_factor", REFERENCE["partial_rotary_factor"])
        != rope["partial_rotary_factor"]
    ):
        return None
    if raw.get("head_dim", REFERENCE["head_dim"]) != REFERENCE["qk_rope_head_dim"]:
        return None
    if raw.get("qk_head_dim", REFERENCE["qk_head_dim"]) != (
        REFERENCE["qk_nope_head_dim"] + REFERENCE["qk_rope_head_dim"]
    ):
        return None
    normalized["first_k_dense_replace"] = 1
    return normalized


def parameter_shapes(configuration: Mapping[str, object]) -> dict[str, tuple[int, ...]]:
    """Logical direct and per-expert parameter identities for the CausalLM path."""
    c = checked(configuration)
    require(c is not None, "Unsupported GLM-4.7-Flash configuration.")
    assert c is not None
    h, vocab = int(c["hidden_size"]), int(c["vocab_size"])
    heads = int(c["num_attention_heads"])
    q_rank, kv_rank = int(c["q_lora_rank"]), int(c["kv_lora_rank"])
    nope, rope = int(c["qk_nope_head_dim"]), int(c["qk_rope_head_dim"])
    value = int(c["v_head_dim"])
    routed = int(c["n_routed_experts"])
    width = int(c["moe_intermediate_size"])
    shared_width = width * int(c["n_shared_experts"])
    result = {
        "model.embed_tokens.weight": (vocab, h),
        "model.norm.weight": (h,),
        "lm_head.weight": (vocab, h),
    }
    for index, kind in enumerate(c["mlp_layer_types"]):
        layer = f"model.layers.{index}"
        result.update(
            {
                f"{layer}.input_layernorm.weight": (h,),
                f"{layer}.post_attention_layernorm.weight": (h,),
                f"{layer}.self_attn.q_a_proj.weight": (q_rank, h),
                f"{layer}.self_attn.q_a_layernorm.weight": (q_rank,),
                f"{layer}.self_attn.q_b_proj.weight": (heads * (nope + rope), q_rank),
                f"{layer}.self_attn.kv_a_proj_with_mqa.weight": (kv_rank + rope, h),
                f"{layer}.self_attn.kv_a_layernorm.weight": (kv_rank,),
                f"{layer}.self_attn.kv_b_proj.weight": (heads * (nope + value), kv_rank),
                f"{layer}.self_attn.o_proj.weight": (h, heads * value),
            }
        )
        mlp = f"{layer}.mlp"
        if kind == "dense":
            dense = int(c["intermediate_size"])
            result.update(
                {
                    f"{mlp}.gate_proj.weight": (dense, h),
                    f"{mlp}.up_proj.weight": (dense, h),
                    f"{mlp}.down_proj.weight": (h, dense),
                }
            )
        else:
            result[f"{mlp}.gate.weight"] = (routed, h)
            result[f"{mlp}.gate.e_score_correction_bias"] = (routed,)
            for expert in range(routed):
                prefix = f"{mlp}.experts.{expert}"
                result.update(
                    {
                        f"{prefix}.gate_proj.weight": (width, h),
                        f"{prefix}.up_proj.weight": (width, h),
                        f"{prefix}.down_proj.weight": (h, width),
                    }
                )
            result.update(
                {
                    f"{mlp}.shared_experts.gate_proj.weight": (shared_width, h),
                    f"{mlp}.shared_experts.up_proj.weight": (shared_width, h),
                    f"{mlp}.shared_experts.down_proj.weight": (h, shared_width),
                }
            )
    return result


def expert_storage_shapes(configuration: Mapping[str, object]) -> dict[str, tuple[int, ...]]:
    """Stacked parameter shapes declared by Glm4MoeLiteExperts."""
    c = checked(configuration)
    require(c is not None, "Unsupported GLM-4.7-Flash configuration.")
    assert c is not None
    experts = int(c["n_routed_experts"])
    hidden = int(c["hidden_size"])
    width = int(c["moe_intermediate_size"])
    result: dict[str, tuple[int, ...]] = {}
    for index, kind in enumerate(c["mlp_layer_types"]):
        if kind != "sparse":
            continue
        prefix = f"model.layers.{index}.mlp.experts"
        result[prefix + ".gate_up_proj"] = (experts, 2 * width, hidden)
        result[prefix + ".down_proj"] = (experts, hidden, width)
    return result


def fused_expert_regions(
    configuration: Mapping[str, object],
) -> dict[str, tuple[str, str]]:
    c = checked(configuration)
    require(c is not None, "Unsupported GLM-4.7-Flash configuration.")
    assert c is not None
    width = int(c["moe_intermediate_size"])
    regions: dict[str, tuple[str, str]] = {}
    for index, kind in enumerate(c["mlp_layer_types"]):
        if kind != "sparse":
            continue
        base = f"model.layers.{index}.mlp.experts"
        for expert in range(int(c["n_routed_experts"])):
            prefix = f"model.layers.{index}.mlp.experts.{expert}"
            regions[prefix + ".gate_proj.weight"] = (
                base + ".gate_up_proj",
                f"expert index {expert}; gate rows [0, {width}) of the stacked gate_up_proj "
                f"record for layer {index}; columns are the complete hidden dimension.",
            )
            regions[prefix + ".up_proj.weight"] = (
                base + ".gate_up_proj",
                f"expert index {expert}; up rows [{width}, {2 * width}) of the stacked "
                f"gate_up_proj record for layer {index}; columns are the complete hidden "
                "dimension.",
            )
            regions[prefix + ".down_proj.weight"] = (
                base + ".down_proj",
                f"expert index {expert} on axis 0 of the stacked down_proj record "
                f"for layer {index}; "
                "all output and intermediate dimensions are retained.",
            )
    return regions


def supports(inputs: AnalysisInput) -> bool:
    return checked(inputs.configuration) is not None


class Glm4MoeLiteGraph(DeepseekGraph):
    """Reuse graph-record mechanics while replacing every GLM-specific path."""

    def __init__(self, inputs: AnalysisInput, builder: GraphBuilder, configuration: dict[str, Any]):
        self.inputs, self.b, self.c = inputs, builder, configuration
        self.children: dict[str, list[str]] = {}
        self.parameter_ids: dict[str, str] = {}
        self.used_storage: set[str] = set()
        self.physical = inputs.bindings.physical
        self.numeric = {tensor.name: tensor for tensor in inputs.bindings.numeric.values()}
        self.shapes = parameter_shapes(configuration)
        self.expert_regions = fused_expert_regions(configuration)
        self.expert_storage = expert_storage_shapes(configuration)
        for name, dims in self.shapes.items():
            if name in self.expert_regions:
                base, _ = self.expert_regions[name]
                if self._has_direct_parameter(name, dims):
                    self.parameter_ids[name] = self.parameter(name, dims)
                else:
                    self.parameter_ids[name] = self.fused_parameter(name, dims, base)
            else:
                self.parameter_ids[name] = self.parameter(name, dims)

    def provenance(self, *fields: str) -> list[r.ArchitectureProvenance]:
        """Keep per-node config provenance precise without repeating one record per field."""
        result = [
            r.ArchitectureProvenance(
                kind="description",
                source=self.b.producer.description,
                revision=self.b.producer.revision,
            )
        ]
        if fields:
            paths = ", ".join(f"/{field}" for field in fields if field in self.inputs.configuration)
            if paths:
                result.append(
                    r.ArchitectureProvenance(
                        kind="configuration",
                        source="config.json",
                        rule=f"Checked configuration fields: {paths}",
                    )
                )
        return result

    def _storage_provenance(
        self, values: list[r.ArchitectureStorage]
    ) -> list[r.ArchitectureProvenance]:
        return [r.ArchitectureProvenance(kind="storage", source=value.name) for value in values]

    def _unresolved(
        self,
        name: str,
        dims: tuple[int, ...],
        storage: list[r.ArchitectureStorage],
        message: str,
    ) -> str:
        pid = self.b.record_id("parameter", name)
        self.b.add_parameter(
            r.ArchitectureDirectParameter(
                id=pid,
                name=name,
                logical_shape=shape(*dims),
                binding="unresolved",
                storage=storage,
                inspection=r.ArchitectureUnavailableInspection(
                    status="unavailable", reason="unresolved_binding", message=message
                ),
                provenance=self.b.producer.provenance() + self._storage_provenance(storage),
            )
        )
        self.b.diagnose(
            r.ArchitectureDiagnostic(
                code="unresolved_binding",
                message="A GLM parameter does not bind to verified storage geometry.",
                parameter_id=pid,
            )
        )
        return pid

    def _compressed_storage(
        self, name: str, dims: tuple[int, ...]
    ) -> list[r.ArchitectureStorage] | None:
        numeric = self.numeric.get(name)
        if numeric is None or numeric.storage_format != COMPRESSED_FORMAT or len(dims) != 2:
            return None
        if numeric.shape != dims:
            return None
        prefix = name.removesuffix(".weight")
        packed_name = prefix + ".weight_packed"
        scale_name = prefix + ".weight_scale"
        shape_name = prefix + ".weight_shape"
        packed = self.physical.get(packed_name)
        scale = self.physical.get(scale_name)
        logical_shape = self.physical.get(shape_name)
        out, inp = dims
        expected_scale_dtype = "BF16" if self.c.get("dtype") == "bfloat16" else "F16"
        if (
            packed is None
            or scale is None
            or logical_shape is None
            or packed.dtype != "I32"
            or packed.shape != [out, inp // 8]
            or scale.dtype != expected_scale_dtype
            or scale.shape != [out, inp // 32]
            or logical_shape.dtype != "I64"
            or logical_shape.shape != [2]
            or prefix + ".weight_zero_point" in self.physical
        ):
            return None
        result = [
            packed.model_copy(update={"role": "packed_data"}),
            scale.model_copy(update={"role": "scales"}),
            logical_shape.model_copy(update={"role": "logical_shape"}),
        ]
        self.used_storage.update(value.name for value in result)
        return result

    def _has_direct_parameter(self, name: str, dims: tuple[int, ...]) -> bool:
        storage = self.physical.get(name)
        if storage is not None:
            return storage.shape == list(dims) and storage.dtype in {"F32", "F16", "BF16"}
        return self._compressed_storage(name, dims) is not None

    def parameter(self, name: str, dims: tuple[int, ...]) -> str:
        storage = self.physical.get(name)
        numeric = self.numeric.get(name)
        packed_storage = self._compressed_storage(name, dims)
        if packed_storage is not None:
            pid = self.b.record_id("parameter", name)
            self.b.add_parameter(
                r.ArchitectureDirectParameter(
                    id=pid,
                    name=name,
                    logical_shape=shape(*dims),
                    binding="quantized",
                    storage=packed_storage,
                    inspection=r.ArchitectureUnavailableInspection(
                        status="unavailable",
                        reason="unsupported_representation",
                        message=(
                            "Architecture metadata retains the packed group; "
                            "no logical view is attached."
                        ),
                    ),
                    provenance=self.b.producer.provenance()
                    + [
                        r.ArchitectureProvenance(kind="storage", source=value.name)
                        for value in packed_storage
                    ],
                )
            )
            return pid

        if storage is None:
            return self._unresolved(name, dims, [], "Required parameter storage is absent.")
        if storage.shape != list(dims) or storage.dtype not in {"F32", "F16", "BF16"}:
            return self._unresolved(
                name, dims, [storage], "Stored parameter geometry is incompatible with GLM."
            )
        self.used_storage.add(name)
        inspection: r.ArchitectureInspection
        if numeric is None:
            inspection = r.ArchitectureUnavailableInspection(
                status="unavailable",
                reason="unsupported_representation",
                message="No admitted complete native numeric view exists.",
            )
        else:
            require(
                numeric.shape == dims and numeric.dtype == storage.dtype,
                "GLM numeric and physical inventories disagree.",
            )
            inspection = r.ArchitectureAvailableInspection(status="available", tensor_id=numeric.id)
        pid = self.b.record_id("parameter", name)
        self.b.add_parameter(
            r.ArchitectureDirectParameter(
                id=pid,
                name=name,
                logical_shape=shape(*dims),
                binding="native",
                storage=[storage],
                inspection=inspection,
                provenance=self.b.producer.provenance()
                + [r.ArchitectureProvenance(kind="storage", source=name)],
            )
        )
        return pid

    def fused_parameter(self, name: str, dims: tuple[int, ...], base: str) -> str:
        storage = self.physical.get(base)
        expected = self.expert_storage[base]
        if storage is None:
            return self._unresolved(name, dims, [], "Stacked expert parameter storage is absent.")
        if storage.shape != list(expected) or storage.dtype not in {"F32", "F16", "BF16"}:
            return self._unresolved(
                name, dims, [storage], "Stacked expert storage geometry is incompatible with GLM."
            )
        self.used_storage.add(base)
        pid = self.b.record_id("parameter", name)
        self.b.add_parameter(
            r.ArchitectureFusedParameter(
                id=pid,
                name=name,
                logical_shape=shape(*dims),
                binding="fused_region",
                storage=[storage],
                region=r.ArchitectureRegion(
                    storage_name=base, description=self.expert_regions[name][1]
                ),
                inspection=r.ArchitectureUnavailableInspection(
                    status="unavailable",
                    reason="requires_view",
                    message="This expert slice belongs to a stacked rank-3 parameter.",
                ),
                provenance=self.b.producer.provenance()
                + [r.ArchitectureProvenance(kind="storage", source=base)],
            )
        )
        return pid

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
        h, heads = int(c["hidden_size"]), int(c["num_attention_heads"])
        q_rank, kv_rank = int(c["q_lora_rank"]), int(c["kv_lora_rank"])
        nope, rope = int(c["qk_nope_head_dim"]), int(c["qk_rope_head_dim"])
        value = int(c["v_head_dim"])
        q_width = nope + rope
        sequence = "S"
        extended = _expr("K + S", "K", "S")
        hidden = shape("B", sequence, h)
        q_heads = shape("B", heads, sequence, q_width)
        q_nonrotary = shape("B", heads, sequence, nope)
        q_rotary = shape("B", heads, sequence, rope)
        latent = shape("B", sequence, kv_rank)
        cached_latent = shape("B", 1, extended, kv_rank)
        cached_rotary = shape("B", 1, extended, rope)
        key_nonrotary = shape("B", heads, extended, nope)
        value_heads = shape("B", heads, extended, value)
        full_key = shape("B", heads, extended, q_width)
        query = Value(key, "x", hidden)

        self.begin_group(key)
        self.b.begin_template(key, key, "glm4_moe_lite_mla", "attention")

        q_a = self.linear(
            key + ".q_a_proj",
            query,
            h,
            q_rank,
            key,
            fields=("hidden_size", "q_lora_rank", "attention_bias"),
        )
        q_latent = self.norm(
            key + ".q_a_layernorm",
            q_a,
            q_rank,
            key,
            fields=("q_lora_rank", "rms_norm_eps"),
        )
        q_b = self.linear(
            key + ".q_b_proj",
            q_latent,
            q_rank,
            heads * q_width,
            key,
            fields=("q_lora_rank", "num_attention_heads", "qk_nope_head_dim", "qk_rope_head_dim"),
        )
        q_view = self.op(
            key + ".q_heads",
            "reshape_transpose",
            {"x": q_b},
            q_heads,
            parent=key,
            attributes={"heads": float(heads), "head_width": float(q_width)},
            formula="[B,S,H·D] → [B,H,S,D]",
        )
        q_split = self.node(
            key + ".q_split",
            "split",
            {"x": q_view},
            {"non_rotary": q_nonrotary, "rotary": q_rotary},
            parent=key,
            attributes={"non_rotary_width": float(nope), "rotary_width": float(rope)},
            fields=("qk_nope_head_dim", "qk_rope_head_dim"),
            formula="split each query head into the non-RoPE and RoPE dimensions",
        )
        q_rope = self.op(
            key + ".q_rope",
            "rotary_position",
            {
                "x": q_split["rotary"],
                "cos": Value(key, "cos", cosine.shape),
                "sin": Value(key, "sin", sine.shape),
                "positions": Value(key, "positions", positions.shape),
            },
            q_rotary,
            parent=key,
            attributes={"rope_interleave": True, "rope_type": "default", "rope_theta": 1_000_000.0},
            fields=("rope_parameters", "rope_interleave", "qk_rope_head_dim"),
            formula="apply interleaved default RoPE to the rotary query dimensions",
        )
        q = self.op(
            key + ".q_recombine",
            "concat",
            {"non_rotary": q_split["non_rotary"], "rotary": q_rope},
            q_heads,
            parent=key,
            attributes={"axis": -1.0},
            fields=("qk_nope_head_dim", "qk_rope_head_dim"),
        )

        kv_a = self.linear(
            key + ".kv_a_proj_with_mqa",
            query,
            h,
            kv_rank + rope,
            key,
            fields=("hidden_size", "kv_lora_rank", "qk_rope_head_dim", "attention_bias"),
        )
        kv_split = self.node(
            key + ".kv_a_split",
            "split",
            {"x": kv_a},
            {"latent": latent, "rotary_key": shape("B", sequence, rope)},
            parent=key,
            attributes={"latent_rank": float(kv_rank), "rotary_width": float(rope)},
            fields=("kv_lora_rank", "qk_rope_head_dim"),
            formula="split the low-rank KV latent from the shared rotary-key branch",
        )
        kv_latent = self.norm(
            key + ".kv_a_layernorm",
            kv_split["latent"],
            kv_rank,
            key,
            fields=("kv_lora_rank", "rms_norm_eps"),
        )
        kv_latent_heads = self.op(
            key + ".kv_latent_head",
            "reshape",
            {"x": kv_latent},
            shape("B", 1, sequence, kv_rank),
            parent=key,
            attributes={"head_count": 1.0, "cached_representation": "normalized_latent"},
        )
        k_rope_head = self.op(
            key + ".k_rope_head",
            "reshape",
            {"x": kv_split["rotary_key"]},
            shape("B", 1, sequence, rope),
            parent=key,
            attributes={"head_count": 1.0, "shared_across_query_heads": True},
        )
        k_rope = self.op(
            key + ".k_rope",
            "rotary_position",
            {
                "x": k_rope_head,
                "cos": Value(key, "cos", cosine.shape),
                "sin": Value(key, "sin", sine.shape),
                "positions": Value(key, "positions", positions.shape),
            },
            shape("B", 1, sequence, rope),
            parent=key,
            attributes={"rope_interleave": True, "rope_type": "default", "rope_theta": 1_000_000.0},
            fields=("rope_parameters", "rope_interleave", "qk_rope_head_dim"),
            formula="apply interleaved default RoPE to the one shared rotary-key head",
        )

        next_latent = self.op(
            key + ".kv_latent_cache_update",
            "state_concat",
            {
                "prior_state": Value(key, "prior_key_state", prior_key.shape, "state"),
                "current_state": kv_latent_heads,
            },
            cached_latent,
            parent=key,
            attributes={"sequence_axis": 2.0, "state_role": "normalized compressed KV latent"},
            fields=("num_hidden_layers", "kv_lora_rank"),
            formula="symbolic prior normalized KV latent concatenated with the current latent",
            state_output=True,
        )
        next_rotary = self.op(
            key + ".rotary_key_cache_update",
            "state_concat",
            {
                "prior_state": Value(key, "prior_value_state", prior_value.shape, "state"),
                "current_state": k_rope,
            },
            cached_rotary,
            parent=key,
            attributes={"sequence_axis": 2.0, "state_role": "shared rotary key"},
            fields=("num_hidden_layers", "qk_rope_head_dim"),
            formula="symbolic prior rotated key concatenated with the current rotated key",
            state_output=True,
        )

        kv_b = self.linear(
            key + ".kv_b_proj",
            next_latent,
            kv_rank,
            heads * (nope + value),
            key,
            fields=("kv_lora_rank", "num_attention_heads", "qk_nope_head_dim", "v_head_dim"),
            attributes={"reconstructed_key_width": float(nope), "value_width": float(value)},
        )
        kv_heads = self.op(
            key + ".kv_heads",
            "reshape_transpose",
            {"x": kv_b},
            shape("B", heads, extended, nope + value),
            parent=key,
            attributes={"heads": float(heads)},
            formula="[B,1,K+S,H·D] → [B,H,K+S,D]",
        )
        kv_split_b = self.node(
            key + ".kv_b_split",
            "split",
            {"x": kv_heads},
            {"key_non_rotary": key_nonrotary, "value": value_heads},
            parent=key,
            attributes={"key_width": float(nope), "value_width": float(value)},
            fields=("qk_nope_head_dim", "v_head_dim"),
            formula="split reconstructed per-head non-RoPE keys and values",
        )
        expanded_k = self.op(
            key + ".k_rope_repeat",
            "repeat_kv",
            {"x": next_rotary},
            shape("B", heads, extended, rope),
            parent=key,
            attributes={"groups": float(heads), "shared_rotary_key": True},
            fields=("num_attention_heads",),
        )
        k = self.op(
            key + ".key_recombine",
            "concat",
            {"non_rotary": kv_split_b["key_non_rotary"], "rotary": expanded_k},
            full_key,
            parent=key,
            attributes={"axis": -1.0},
            fields=("qk_nope_head_dim", "qk_rope_head_dim"),
        )
        scores = self.op(
            key + ".attention_scores",
            "attention_product",
            {"query": q, "key": k},
            shape("B", heads, sequence, extended),
            parent=key,
            attributes={"scale": 1.0 / math.sqrt(q_width), "causal": True},
            fields=("qk_nope_head_dim", "qk_rope_head_dim", "attention_dropout"),
            formula="Q Kᵀ / sqrt(qk_head_dim), with the configured causal mask",
        )
        probabilities = self.op(
            key + ".attention_softmax",
            "softmax",
            {"scores": scores, "mask": Value(key, "causal_mask", mask.shape)},
            shape("B", heads, sequence, extended),
            parent=key,
            attributes={"axis": -1.0, "masked": True},
            formula="softmax over prior plus current key positions",
        )
        context = self.op(
            key + ".attention_values",
            "attention_product",
            {"probabilities": probabilities, "values": kv_split_b["value"]},
            shape("B", heads, sequence, value),
            parent=key,
            formula="attention probabilities multiplied by reconstructed values",
        )
        merged = self.op(
            key + ".merge_heads",
            "reshape_transpose",
            {"x": context},
            shape("B", sequence, heads * value),
            parent=key,
            attributes={"heads": float(heads), "head_width": float(value)},
        )
        output = self.linear(
            key + ".o_proj",
            merged,
            heads * value,
            h,
            key,
            fields=("hidden_size", "num_attention_heads", "v_head_dim", "attention_bias"),
        )
        return self.end_group(
            key,
            "GLM latent attention",
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
            {"out": output, "next_key_state": next_latent, "next_value_state": next_rotary},
            role="attention",
            attributes={
                "q_lora_rank": float(q_rank),
                "kv_lora_rank": float(kv_rank),
                "query_non_rotary_width": float(nope),
                "query_rotary_width": float(rope),
                "value_head_width": float(value),
                "rope_interleave": True,
            },
            fields=(
                "q_lora_rank",
                "kv_lora_rank",
                "qk_nope_head_dim",
                "qk_rope_head_dim",
                "v_head_dim",
            ),
        )

    def routed_expert(
        self,
        layer: str,
        expert_index: int,
        hidden_states: Value,
        indices: Value,
        weights: Value,
        count_symbol: str,
    ) -> dict[str, Value]:
        key = f"{layer}.mlp.experts.{expert_index}"
        width = int(self.c["moe_intermediate_size"])
        parent = layer + ".mlp"
        parameter_names = (
            key + ".gate_proj.weight",
            key + ".up_proj.weight",
            key + ".down_proj.weight",
        )
        parameter_ids = [self.parameter_ids[name] for name in parameter_names]
        references: list[r.ArchitectureReference] = []
        references.append(r.ArchitectureModuleReference(kind="module", name=key))
        references.extend(
            r.ArchitectureParameterReference(kind="parameter", parameter_id=pid)
            for pid in parameter_ids
        )
        input_values = {
            "hidden_states": hidden_states,
            "expert_indices": indices,
            "expert_weights": weights,
        }
        output_shapes = {
            "expert_values": shape(count_symbol, int(self.c["hidden_size"])),
            "positions": shape(count_symbol, 2),
        }
        formula = (
            "symbolically select this expert's token positions, compute "
            "down_proj(silu(gate_proj(x)) * up_proj(x)), then multiply each result "
            "by its selected normalized router weight; gate/up are fused stacked regions"
        )
        self.children.setdefault(parent, []).append(self.nid(key))
        fields = ("hidden_size", "moe_intermediate_size", "hidden_act")
        self.b.add_node(
            r.ArchitectureGroupNode(
                id=self.nid(key),
                parent_id=self.nid(parent),
                kind="group",
                label=f"Routed expert {expert_index}",
                operation="routed_swiglu_expert",
                formula=formula,
                children=[],
                ports=[
                    self.port(name, value.shape, "input") for name, value in input_values.items()
                ]
                + [self.port(name, value, "output") for name, value in output_shapes.items()],
                parameter_ids=parameter_ids,
                references=references,
                attributes=[
                    r.ArchitectureAttribute(
                        name="activation", value="silu", provenance=self.provenance(*fields)
                    ),
                    r.ArchitectureAttribute(
                        name="intermediate_width",
                        value=float(width),
                        provenance=self.provenance(*fields),
                    ),
                    r.ArchitectureAttribute(
                        name="expert_index",
                        value=float(expert_index),
                        provenance=self.provenance("n_routed_experts"),
                    ),
                    r.ArchitectureAttribute(
                        name="routing_is_symbolic", value=True, provenance=self.provenance()
                    ),
                    r.ArchitectureAttribute(
                        name="gate_up_storage",
                        value=layer + ".mlp.experts.gate_up_proj",
                        provenance=self.provenance("moe_intermediate_size"),
                    ),
                    self.role_attribute("mlp"),
                ],
                provenance=self.provenance(*fields) + [self.source_key(key)],
            ),
            semantic_key=key,
        )
        for name, value in input_values.items():
            self.link(value, key, name)
        return {name: Value(key, name, value) for name, value in output_shapes.items()}

    def moe_mlp(self, layer: str, x: Value) -> Value:
        c = self.c
        key = layer + ".mlp"
        hidden = int(c["hidden_size"])
        experts = int(c["n_routed_experts"])
        top_k = int(c["num_experts_per_tok"])
        groups = int(c["n_group"])
        group_top_k = int(c["topk_group"])
        scores_shape = shape("B", "S", experts)
        grouped_shape = shape("B", "S", groups)
        selected_shape = shape("B", "S", top_k)
        self.begin_group(key)

        logits = self.linear(
            key + ".gate",
            Value(key, "x", x.shape),
            hidden,
            experts,
            key,
            fields=("hidden_size", "n_routed_experts", "topk_method"),
            attributes={"accumulation_dtype": "float32", "router": "linear"},
        )
        scores = self.op(
            key + ".router_sigmoid",
            "sigmoid",
            {"logits": logits},
            scores_shape,
            parent=key,
            attributes={"score_function": "sigmoid"},
            fields=("topk_method",),
            formula="sigmoid(float32 router logits)",
        )
        corrected = self.op(
            key + ".selection_scores",
            "add",
            {"scores": scores},
            scores_shape,
            parent=key,
            parameters=(key + ".gate.e_score_correction_bias",),
            attributes={"bias_role": "expert selection correction only"},
            fields=("n_routed_experts", "topk_method"),
            formula="choice_scores = sigmoid(router_logits) + e_score_correction_bias",
        )
        group_scores = self.op(
            key + ".group_top2_score_sum",
            "group_topk_score_sum",
            {"scores": corrected},
            grouped_shape,
            parent=key,
            attributes={
                "experts_per_group": float(experts // groups),
                "top_scores": 2.0,
                "group_count": float(groups),
            },
            fields=("n_group", "n_routed_experts", "topk_method"),
            formula="sum the two highest corrected expert scores in each group",
        )
        selected_groups = self.op(
            key + ".group_topk",
            "topk_selection",
            {"scores": group_scores},
            shape("B", "S", group_top_k),
            parent=key,
            attributes={"top_k": float(group_top_k), "selection_method": "topk groups"},
            fields=("topk_group", "n_group"),
        )
        group_mask = self.op(
            key + ".group_mask",
            "group_expert_mask",
            {"group_indices": selected_groups},
            scores_shape,
            parent=key,
            attributes={"experts_per_group": float(experts // groups)},
            formula="expand selected groups to an expert eligibility mask",
        )
        masked_scores = self.op(
            key + ".masked_selection_scores",
            "mask_invalid",
            {"scores": corrected, "mask": group_mask},
            scores_shape,
            parent=key,
            attributes={"masked_value": "negative_infinity"},
        )
        indices = self.op(
            key + ".expert_topk",
            "topk_selection",
            {"scores": masked_scores},
            selected_shape,
            parent=key,
            attributes={"top_k": float(top_k), "selection_method": "noaux_tc"},
            fields=("num_experts_per_tok", "topk_method"),
            formula="select top-k experts from corrected scores inside selected groups",
        )
        selected_weights = self.op(
            key + ".selected_raw_weights",
            "gather",
            {"scores": scores, "indices": indices},
            selected_shape,
            parent=key,
            formula="gather selected weights from sigmoid scores without correction bias",
        )
        normalized = self.op(
            key + ".normalize_topk_weights",
            "normalize",
            {"weights": selected_weights},
            selected_shape,
            parent=key,
            attributes={"epsilon": 1e-20, "normalize_topk_prob": True},
            fields=("norm_topk_prob",),
            formula="weights / (sum(weights, axis=-1) + 1e-20)",
        )
        weights = self.op(
            key + ".scale_topk_weights",
            "multiply",
            {"weights": normalized},
            selected_shape,
            parent=key,
            attributes={"routed_scaling_factor": float(c["routed_scaling_factor"])},
            fields=("routed_scaling_factor",),
            formula="normalized selected weights × routed_scaling_factor",
        )

        outputs: list[Value] = []
        positions: list[Value] = []
        instances: list[r.ArchitectureRepetitionInstance] = []
        layer_index = int(layer.rsplit(".", 1)[-1])
        for expert in range(experts):
            count_symbol = f"R{layer_index}_{expert}"
            self.b.add_symbol(
                count_symbol,
                f"Symbolic number of token positions routed to expert {expert} in {layer}; "
                "not computed.",
            )
            expert_key = f"{key}.experts.{expert}"
            expert_outputs = self.routed_expert(
                layer, expert, Value(key, "x", x.shape), indices, weights, count_symbol
            )
            outputs.append(expert_outputs["expert_values"])
            positions.append(expert_outputs["positions"])
            instances.append(
                r.ArchitectureRepetitionInstance(
                    node_id=self.nid(expert_key), index=expert, variant="routed_expert"
                )
            )

        scatter_inputs = {
            "expert_values": Value(outputs[0].node, outputs[0].port, None),
            "token_positions": Value(positions[0].node, positions[0].port, None),
        }
        routed = self.node(
            key + ".routed_scatter_sum",
            "weighted_scatter_sum",
            scatter_inputs,
            {"out": shape("B", "S", hidden)},
            parent=key,
            attributes={"top_k": float(top_k), "reduction": "sum at original token positions"},
            fields=("num_experts_per_tok",),
            formula="scatter-add all selected weighted expert results to symbolic source positions",
        )["out"]
        for expert_output, position in zip(outputs[1:], positions[1:], strict=True):
            self.link(
                Value(expert_output.node, expert_output.port, None),
                key + ".routed_scatter_sum",
                "expert_values",
            )
            self.link(
                Value(position.node, position.port, None),
                key + ".routed_scatter_sum",
                "token_positions",
            )

        shared = self.shared_mlp(layer, Value(key, "x", x.shape), hidden)
        combined = self.op(
            key + ".shared_routed_add",
            "add",
            {"routed": routed, "shared": shared},
            shape("B", "S", hidden),
            parent=key,
            attributes={"branches": "routed experts plus shared expert"},
            fields=("n_shared_experts",),
            formula="weighted routed expert sum + shared expert MLP output",
        )
        result = self.end_group(
            key,
            "GLM MoE (routed and shared)",
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
                "n_routed_experts",
                "num_experts_per_tok",
                "n_shared_experts",
                "norm_topk_prob",
                "routed_scaling_factor",
                "topk_method",
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

    def build(self) -> None:
        c = self.c
        hidden_width = int(c["hidden_size"])
        kv_rank = int(c["kv_lora_rank"])
        rope_width = int(c["qk_rope_head_dim"])
        layers = int(c["num_hidden_layers"])
        sequence = shape("B", "S")
        hidden = shape("B", "S", hidden_width)
        prior_latent_bank = shape(layers, "B", 1, "K", kv_rank)
        prior_rope_bank = shape(layers, "B", 1, "K", rope_width)
        next_latent_bank = shape(layers, "B", 1, _expr("K + S", "K", "S"), kv_rank)
        next_rope_bank = shape(layers, "B", 1, _expr("K + S", "K", "S"), rope_width)
        mask_shape = shape("B", 1, "S", _expr("K + S", "K", "S"))
        self.b.add_symbol("B", "Symbolic batch size; no input has been executed.")
        self.b.add_symbol("S", "Symbolic current token sequence length; no prompt was supplied.")
        self.b.add_symbol("K", "Symbolic prior compressed KV-cache length; no cache sample exists.")

        root_inputs: dict[str, Value] = {
            "input_ids": Value("model", "input_ids", sequence),
            "position_ids": Value("model", "position_ids", sequence),
            "attention_mask": Value("model", "attention_mask", mask_shape),
            "past_key_values_kv_latent": Value(
                "model", "past_key_values_kv_latent", prior_latent_bank, "state"
            ),
            "past_key_values_rotary_key": Value(
                "model", "past_key_values_rotary_key", prior_rope_bank, "state"
            ),
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
        rope_shape = shape("B", "S", rope_width)
        rope_attributes: dict[str, str | float | bool] = {
            "rope_type": "default",
            "rope_theta": 1_000_000.0,
            "partial_rotary_factor": 1.0,
            "interleaved_application": True,
            "max_position_embeddings": float(c["max_position_embeddings"]),
        }
        cosine = self.op(
            "model.rotary_cos",
            "rotary_cosine",
            {"positions": root_inputs["position_ids"]},
            rope_shape,
            parent="model",
            attributes=rope_attributes,
            fields=("rope_parameters", "partial_rotary_factor", "rope_interleave"),
            formula=(
                "default RoPE cosine frequencies at symbolic positions; no positions are evaluated"
            ),
        )
        sine = self.op(
            "model.rotary_sin",
            "rotary_sine",
            {"positions": root_inputs["position_ids"]},
            rope_shape,
            parent="model",
            attributes=rope_attributes,
            fields=("rope_parameters", "partial_rotary_factor", "rope_interleave"),
            formula=(
                "default RoPE sine frequencies at symbolic positions; no positions are evaluated"
            ),
        )

        latent_states: list[Value] = []
        rotary_states: list[Value] = []
        layer_instances: list[r.ArchitectureRepetitionInstance] = []
        previous = tokens
        for index, kind in enumerate(c["mlp_layer_types"]):
            prior_latent = self.op(
                f"model.layer_state.{index}.prior_kv_latent",
                "select_layer_state",
                {"state_bank": root_inputs["past_key_values_kv_latent"]},
                shape("B", 1, "K", kv_rank),
                parent="model",
                attributes={
                    "layer_index": float(index),
                    "state_role": "prior normalized KV latent",
                },
                fields=("num_hidden_layers", "kv_lora_rank"),
                state_output=True,
            )
            prior_rotary = self.op(
                f"model.layer_state.{index}.prior_rotary_key",
                "select_layer_state",
                {"state_bank": root_inputs["past_key_values_rotary_key"]},
                shape("B", 1, "K", rope_width),
                parent="model",
                attributes={"layer_index": float(index), "state_role": "prior rotated shared key"},
                fields=("num_hidden_layers", "qk_rope_head_dim"),
                state_output=True,
            )
            built = self.layer(
                index,
                previous,
                root_inputs["position_ids"],
                cosine,
                sine,
                root_inputs["attention_mask"],
                prior_latent,
                prior_rotary,
            )
            previous = built["out"]
            latent_states.append(built["next_key_state"])
            rotary_states.append(built["next_value_state"])
            layer_instances.append(
                r.ArchitectureRepetitionInstance(
                    node_id=self.nid(f"model.layers.{index}"), index=index, variant=kind
                )
            )

        normalized = self.norm(
            "model.norm",
            previous,
            hidden_width,
            "model",
            fields=("hidden_size", "rms_norm_eps"),
        )
        logits = self.linear(
            "lm_head",
            normalized,
            hidden_width,
            int(c["vocab_size"]),
            "model",
            fields=("hidden_size", "vocab_size", "tie_word_embeddings"),
        )
        latent_output = self.node(
            "model.next_kv_latent_values",
            "stack_layer_states",
            {f"layer_{index}": value for index, value in enumerate(latent_states)},
            {"out": next_latent_bank},
            parent="model",
            attributes={"layer_count": float(layers), "state_role": "next normalized KV latents"},
            fields=("num_hidden_layers", "kv_lora_rank"),
            state_outputs=("out",),
        )["out"]
        rotary_output = self.node(
            "model.next_rotary_key_values",
            "stack_layer_states",
            {f"layer_{index}": value for index, value in enumerate(rotary_states)},
            {"out": next_rope_bank},
            parent="model",
            attributes={"layer_count": float(layers), "state_role": "next rotated shared keys"},
            fields=("num_hidden_layers", "qk_rope_head_dim"),
            state_outputs=("out",),
        )["out"]
        self.end_group(
            "model",
            "GLM-4.7-Flash CausalLM",
            None,
            root_inputs,
            {
                "logits": logits,
                "next_key_values_kv_latent": latent_output,
                "next_key_values_rotary_key": rotary_output,
            },
            attributes={
                "architecture": ARCHITECTURE,
                "static_evaluation_path": True,
                "num_nextn_predict_layers": float(c["num_nextn_predict_layers"]),
                "nextn_evaluation": (
                    "metadata only: ordinary CausalLM forward evaluates configured decoder layers "
                    "and lm_head; upstream ignores unexpected model.layers.47.* checkpoint keys"
                ),
            },
            fields=("architectures", "model_type", "num_hidden_layers", "num_nextn_predict_layers"),
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

        # The pinned implementation intentionally ignores this one stored MTP layer
        # while ordinary CausalLM.forward evaluates only layers [0, num_hidden_layers).
        unrecognized = {
            name
            for name in set(self.physical) - self.used_storage
            if not name.startswith("model.layers.47.")
        }
        if unrecognized:
            self.b.diagnose(
                r.ArchitectureDiagnostic(
                    code="unrecognized_storage",
                    message=(
                        "Checkpoint contains storage outside the reviewed "
                        "GLM-4.7-Flash evaluation graph."
                    ),
                )
            )


def _expr(text: str, *symbols: str) -> r.ArchitectureExpressionDimension:
    return r.ArchitectureExpressionDimension(kind="expression", text=text, symbols=list(symbols))


def build(inputs: AnalysisInput, builder: GraphBuilder) -> None:
    configuration = checked(inputs.configuration)
    require(configuration is not None, "Unsupported GLM-4.7-Flash configuration.")
    assert configuration is not None
    Glm4MoeLiteGraph(inputs, builder, configuration).build()


def register_glm4_moe_lite(registry: DescriptionRegistry) -> None:
    registry.register(
        Description(
            producer=PRODUCER,
            scope="language_model",
            model_types=frozenset({MODEL_TYPE}),
            architectures=frozenset({ARCHITECTURE}),
            supports=supports,
            build=build,
        )
    )

"""Fail-closed metadata and tensor binding for the accepted PEFT LoRA subset."""

import hashlib
import math
import re
from dataclasses import dataclass

from .model_files import FileSnapshot, ModelError, invalid
from .tensor_source import (
    MAX_SAFE_INTEGER,
    PhysicalTensor,
    TensorDescriptor,
    TensorLocation,
    native_location,
)

_NAME = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.-]*\Z")
_FACTOR = re.compile(
    r"^(?P<module>.+)\.lora_(?P<factor>A|B)(?:\.(?P<adapter>[A-Za-z0-9_-]+))?\.weight$"
)
_DENSE_LINEAR_MODULE = re.compile(
    r"^model\.layers\.(?P<layer>0|[1-9][0-9]*)\."
    r"(?:self_attn\.(?:q_proj|k_proj|v_proj|o_proj)|"
    r"mlp\.(?:gate_proj|up_proj|down_proj))$"
)
_SUPPORTED_CAUSAL_ARCHITECTURES = {
    ("llama", "LlamaForCausalLM"),
    ("qwen3", "Qwen3ForCausalLM"),
}

_KNOWN_FIELDS = {
    "_commit_hash",
    "_name_or_path",
    "adapter_name",
    "alora_invocation_tokens",
    "alpha_pattern",
    "arrow_config",
    "auto_mapping",
    "base_model_name_or_path",
    "bias",
    "corda_config",
    "exclude_modules",
    "eva_config",
    "fan_in_fan_out",
    "inference_mode",
    "init_lora_weights",
    "layer_replication",
    "layers_pattern",
    "layers_to_transform",
    "loftq_config",
    "lora_alpha",
    "lora_bias",
    "lora_dropout",
    "lora_ga_config",
    "megatron_config",
    "megatron_core",
    "modules_to_save",
    "name_or_path",
    "peft_type",
    "peft_version",
    "qalora_group_size",
    "r",
    "rank_pattern",
    "revision",
    "target_modules",
    "target_parameters",
    "task_type",
    "trainable_token_indices",
    "use_dora",
    "use_bdlora",
    "use_qalora",
    "use_rslora",
    "ensure_weight_tying",
}
# PEFT 0.19 serializes several inactive/default fields on the accepted reference
# adapter; validate their inert values below instead of rejecting that metadata.


@dataclass(frozen=True)
class AdapterSpec:
    upstream_name: str
    rank: int
    alpha: float
    scale: float
    target_patterns: tuple[tuple[str, bool], ...]
    factors: dict[str, tuple[PhysicalTensor, PhysicalTensor]]


def _unsupported(message: str = "Unsupported local PEFT adapter representation.") -> ModelError:
    return ModelError("unsupported_representation", message)


def _upstream_name(value: object) -> str:
    if not isinstance(value, str) or not value or len(value) > 512 or "\\" in value:
        raise invalid("Invalid PEFT upstream model identity.")
    name, marker, revision = value.partition("@")
    parts = name.split("/")
    if (
        len(parts) not in {1, 2}
        or any(not _NAME.fullmatch(part) or ".." in part for part in parts)
        or (marker and (not revision or not _NAME.fullmatch(revision) or ".." in revision))
        or ":" in value
        or value.startswith(".")
    ):
        raise invalid("Invalid PEFT upstream model identity.")
    return name


def _target_patterns(value: object) -> tuple[tuple[str, bool], ...]:
    if isinstance(value, str):
        if not value or len(value) > 1024:
            raise invalid("Invalid PEFT target module expression.")
        try:
            re.compile(value)
        except re.error as exc:
            raise invalid("Invalid PEFT target module expression.") from exc
        return ((value, True),)
    if not isinstance(value, list) or not value:
        raise invalid("PEFT target modules must be a nonempty list or regex.")
    result: list[tuple[str, bool]] = []
    for pattern in value:
        if (
            not isinstance(pattern, str)
            or not pattern
            or len(pattern) > 512
            or any(not _NAME.fullmatch(part) or ".." in part for part in pattern.split("."))
        ):
            raise invalid("Invalid PEFT literal target module.")
        result.append((pattern, False))
    if len({pattern for pattern, _ in result}) != len(result):
        raise invalid("Duplicate PEFT target module.")
    return tuple(result)


def _factor_module(name: str) -> tuple[str, str, str | None]:
    match = _FACTOR.fullmatch(name)
    if match is None:
        raise _unsupported("Adapter tensors must be standard LoRA A/B weight factors.")
    module = match.group("module")
    for prefix in ("base_model.model.", "base_model."):
        if module.startswith(prefix):
            module = module.removeprefix(prefix)
            break
    if (
        not module
        or module.startswith(".")
        or module.endswith(".")
        or any(not _NAME.fullmatch(part) or ".." in part for part in module.split("."))
    ):
        raise _unsupported("Unsupported PEFT target module name.")
    return module, match.group("factor"), match.group("adapter")


def _supported_linear_module(config: dict[str, object], module: str) -> bool:
    architectures = config.get("architectures")
    if (
        not isinstance(architectures, list)
        or len(architectures) != 1
        or not isinstance(architectures[0], str)
        or (config.get("model_type"), architectures[0]) not in _SUPPORTED_CAUSAL_ARCHITECTURES
    ):
        return False
    if module == "lm_head":
        return True
    layer_count = config.get("num_hidden_layers")
    match = _DENSE_LINEAR_MODULE.fullmatch(module)
    if (
        type(layer_count) is not int
        or not 0 < layer_count <= MAX_SAFE_INTEGER
        or match is None
        or len(match.group("layer")) > 16
    ):
        return False
    return int(match.group("layer")) < layer_count


def factor_logical_names(adapter_id: str, module: str) -> tuple[str, str]:
    """Return the stable composite tensor names for one already validated target."""
    namespace = "".join(
        character
        if character.isascii() and (character.isalnum() or character in "_-")
        else f"%{ord(character):02X}"
        for character in adapter_id
    )
    prefix = f"__peft__.{namespace}.{module}"
    return f"{prefix}.lora_A.weight", f"{prefix}.lora_B.weight"


def validate_adapter(
    config: dict[str, object], physical: tuple[PhysicalTensor, ...]
) -> AdapterSpec:
    """Validate one adapter config and its complete Safetensors factor inventory."""
    if config.get("peft_type") != "LORA" or config.get("task_type") != "CAUSAL_LM":
        raise _unsupported("Only PEFT LORA adapters for CAUSAL_LM are supported.")
    if set(config) - _KNOWN_FIELDS:
        raise _unsupported("Adapter metadata contains an unsupported option.")

    upstream_name = _upstream_name(config.get("base_model_name_or_path"))
    rank = config.get("r")
    alpha = config.get("lora_alpha")
    if type(rank) is not int or rank <= 0 or rank > MAX_SAFE_INTEGER:
        raise invalid("PEFT LoRA rank must be a positive safe integer.")
    if not isinstance(alpha, (int, float)) or isinstance(alpha, bool):
        raise invalid("PEFT LoRA alpha must be finite and positive.")
    scale = 0.0
    try:
        alpha_value = float(alpha)
        scale = alpha_value / rank
        valid_alpha = math.isfinite(alpha_value) and alpha_value > 0
        valid_alpha = valid_alpha and math.isfinite(scale) and scale > 0
    except OverflowError:
        valid_alpha = False
    if not valid_alpha:
        raise invalid("PEFT LoRA alpha must be finite and positive.")
    if config.get("rank_pattern", {}) != {} or config.get("alpha_pattern", {}) != {}:
        raise _unsupported("Per-target PEFT rank or alpha patterns are unsupported.")
    if config.get("bias") != "none" or config.get("fan_in_fan_out") is not False:
        raise _unsupported("Only bias-free, fan-in/fan-out-disabled LoRA is supported.")
    for flag in ("use_dora", "use_rslora", "use_qalora", "lora_bias", "ensure_weight_tying"):
        if flag in config and config[flag] is not False:
            raise _unsupported("The adapter enables an unsupported LoRA extension.")
    if any(config.get(key) is not None for key in ("modules_to_save", "layer_replication")):
        raise _unsupported("Saved modules and layer replication are unsupported.")
    if config.get("auto_mapping") is not None:
        raise _unsupported("Custom adapter or base mappings are unsupported.")
    for key in (
        "layers_to_transform",
        "layers_pattern",
        "target_parameters",
        "trainable_token_indices",
        "alora_invocation_tokens",
        "exclude_modules",
        "arrow_config",
        "corda_config",
        "megatron_config",
        "loftq_config",
        "eva_config",
        "lora_ga_config",
    ):
        value = config.get(key)
        if value is not None and value not in ({}, [], ()):
            raise _unsupported("The adapter enables an unsupported PEFT option.")
    if config.get("megatron_core") not in (None, "megatron.core"):
        raise _unsupported("Custom Megatron LoRA integration is unsupported.")
    if config.get("use_bdlora") not in (None, False):
        raise _unsupported("The adapter enables an unsupported LoRA extension.")
    qalora_group_size = config.get("qalora_group_size")
    if qalora_group_size is not None and (
        type(qalora_group_size) is not int
        or not 0 < qalora_group_size <= MAX_SAFE_INTEGER
    ):
        raise invalid("Invalid PEFT QALoRA group size metadata.")
    dropout = config.get("lora_dropout", 0.0)
    if not isinstance(dropout, (int, float)) or isinstance(dropout, bool):
        raise invalid("Invalid PEFT LoRA dropout metadata.")
    try:
        dropout_value = float(dropout)
        valid_dropout = math.isfinite(dropout_value) and 0 <= dropout_value <= 1
    except OverflowError:
        valid_dropout = False
    if not valid_dropout:
        raise invalid("Invalid PEFT LoRA dropout metadata.")
    if "init_lora_weights" in config and type(config["init_lora_weights"]) is not bool:
        raise _unsupported("Nonstandard LoRA initialization metadata is unsupported.")
    if "inference_mode" in config and type(config["inference_mode"]) is not bool:
        raise invalid("Invalid PEFT inference_mode metadata.")
    if any(config.get(key) is not None for key in ("layers_to_transform", "layers_pattern")):
        raise _unsupported("Layer-restricted PEFT targets are unsupported.")

    targets = _target_patterns(config.get("target_modules"))
    factor_map: dict[str, dict[str, PhysicalTensor]] = {}
    adapter_names: set[str | None] = set()
    for tensor in physical:
        if tensor.dtype not in {"F32", "F16", "BF16"}:
            raise _unsupported("PEFT factors must use F32, F16, or BF16 Safetensors storage.")
        module, factor, adapter_name = _factor_module(tensor.name)
        adapter_names.add(adapter_name)
        factors = factor_map.setdefault(module, {})
        if factor in factors:
            raise invalid("Duplicate PEFT LoRA factor for one target module.")
        factors[factor] = tensor
    if len(adapter_names) != 1:
        raise _unsupported("Ambiguous PEFT adapter factor names.")
    if not factor_map or any(set(factors) != {"A", "B"} for factors in factor_map.values()):
        raise invalid("Every PEFT target must have exactly one LoRA A/B pair.")

    return AdapterSpec(
        upstream_name,
        rank,
        alpha_value,
        scale,
        targets,
        {module: (factors["A"], factors["B"]) for module, factors in factor_map.items()},
    )


def bind_adapter(
    spec: AdapterSpec,
    adapter_id: str,
    base_config: dict[str, object],
    base_locations: tuple[TensorLocation, ...],
    snapshot: FileSnapshot,
) -> tuple[TensorLocation, ...]:
    """Bind all saved factors to exactly the configured, actionable base weights."""
    from dataclasses import replace

    weights = {
        location.descriptor.name.removesuffix(".weight"): location
        for location in base_locations
        if location.descriptor.name.endswith(".weight")
        and location.descriptor.rank == 2
        and _supported_linear_module(base_config, location.descriptor.name.removesuffix(".weight"))
    }
    matched: dict[str, TensorLocation] = {}
    matched_patterns: set[tuple[str, bool]] = set()
    for module in weights:
        pattern_indexes: list[tuple[str, bool]] = []
        for pattern, is_regex in spec.target_patterns:
            if is_regex:
                if re.fullmatch(pattern, module):
                    pattern_indexes.append((pattern, is_regex))
            elif module == pattern or module.endswith("." + pattern):
                pattern_indexes.append((pattern, is_regex))
        if len(pattern_indexes) > 1:
            raise invalid("PEFT target patterns ambiguously select one base module.")
        if pattern_indexes:
            matched[module] = weights[module]
            matched_patterns.add(pattern_indexes[0])
    if matched_patterns != set(spec.target_patterns):
        raise invalid("A configured PEFT target does not match a base logical weight.")
    if set(matched) != set(spec.factors):
        raise invalid("PEFT adapter factors do not cover exactly the configured base targets.")

    locations: list[TensorLocation] = []
    for module in sorted(matched):
        base = matched[module].descriptor
        a, b = spec.factors[module]
        if (
            base.shape[0] <= 0
            or base.shape[1] <= 0
            or len(a.shape) != 2
            or len(b.shape) != 2
            or a.shape != (spec.rank, base.shape[1])
            or b.shape != (base.shape[0], spec.rank)
        ):
            raise invalid("PEFT LoRA A/B dimensions do not match the base weight.")
        for factor_name, storage in (("lora_A", a), ("lora_B", b)):
            a_name, b_name = factor_logical_names(adapter_id, module)
            name = a_name if factor_name == "lora_A" else b_name
            if any(location.descriptor.name == name for location in base_locations):
                raise invalid("PEFT logical tensor name collides with the base inventory.")
            location = native_location(storage)
            descriptor = TensorDescriptor(
                id=hashlib.sha256(name.encode("utf-8")).hexdigest(),
                name=name,
                path=tuple(name.split(".")),
                shape=location.descriptor.shape,
                rank=location.descriptor.rank,
                numel=location.descriptor.numel,
                storage_dtype=location.descriptor.storage_dtype,
            )
            locations.append(
                replace(
                    location,
                    descriptor=descriptor,
                    snapshot=snapshot,
                    physical_names=(storage.name,),
                )
            )
    return tuple(locations)

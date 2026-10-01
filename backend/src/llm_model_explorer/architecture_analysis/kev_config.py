"""Guarded native Kev inspection metadata, hybrid geometry and unmerged bindings."""

from __future__ import annotations

import copy
import math
import re
from dataclasses import replace
from typing import Any

from ..model_files import ModelError
from ..peft_adapters import _factor_module, _target_patterns
from .core import AnalysisInput
from .qwen35 import PREFIX, parameter_shapes
from .qwen35 import configuration as qwen_configuration

SOURCE_REVISION = "45923b7a3460b6d36358e2e143455902c1eb856b"
BASE_REPOSITORY = "Qwen/Qwen3.5-0.8B-Base"
HEAD_FIELDS = {
    "base",
    "base_revision",
    "head_dim",
    "lora",
    "option_isolation",
    "special_embeddings",
    "weights_dtype",
    "temperature",
    "lora_placement",
}


def head_configuration(value: object, revision: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != HEAD_FIELDS:
        raise ValueError("Unsupported pointer head metadata")
    if (
        value["base"] != BASE_REPOSITORY
        or value["base_revision"] != revision
        or value["option_isolation"] is not False
        or value["special_embeddings"] is not False
        or value["weights_dtype"] != "fp32"
        or value["lora_placement"] != "full"
    ):
        raise ValueError("Incompatible Kev base or hybrid options")
    for key in ("head_dim", "lora"):
        if type(value[key]) is not int or not 0 < value[key] <= 16384:
            raise ValueError("Invalid pointer dimension or LoRA rank")
    temperature = value["temperature"]
    if type(temperature) not in (int, float) or not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("Inference temperature must be finite and positive")
    return dict(value)


def backbone_inputs(inputs: AnalysisInput) -> AnalysisInput:
    config = copy.deepcopy(dict(inputs.configuration))
    for key in ("kev_inspection", "_commit_hash", "revision", "name_or_path"):
        config.pop(key, None)
    config["_name_or_path"] = BASE_REPOSITORY
    text = config.get("text_config")
    if isinstance(text, dict) and isinstance(text.get("rope_parameters"), dict):
        text.setdefault(
            "partial_rotary_factor", text["rope_parameters"].get("partial_rotary_factor")
        )
    return replace(
        inputs,
        configuration=config,
        bindings=replace(
            inputs.bindings,
            physical={
                k: v for k, v in inputs.bindings.physical.items() if not k.startswith("kev.")
            },
            numeric={
                k: v for k, v in inputs.bindings.numeric.items() if not v.name.startswith("kev.")
            },
        ),
    )


def factor_bindings(
    physical: dict[str, Any], expected: dict[str, list[int]], adapter: dict[str, Any]
) -> dict[str, dict[str, str]]:
    """Validate the reviewed PEFT text-backbone mapping, not arbitrary FEATURE_EXTRACTION."""
    if set(adapter) != {"r", "lora_alpha", "target_modules"}:
        raise ValueError("Unsupported packaged adapter metadata")
    rank, alpha = adapter["r"], adapter["lora_alpha"]
    if (
        type(rank) is not int
        or not 0 < rank <= 16384
        or type(alpha) not in (int, float)
        or not math.isfinite(alpha)
        or alpha <= 0
    ):
        raise ValueError("Invalid adapter rank or scaling")
    modules = {
        name[:-7]: dims
        for name, dims in expected.items()
        if len(dims) == 2
        and re.fullmatch(
            re.escape(PREFIX) + r"\.layers\.\d+\.(?:mlp\.(?:gate_proj|up_proj|down_proj)|"
            r"self_attn\.(?:q_proj|k_proj|v_proj|o_proj)|"
            r"linear_attn\.(?:in_proj_qkv|in_proj_z|in_proj_a|in_proj_b|out_proj))\.weight",
            name,
        )
    }
    aliases = {name.removeprefix(PREFIX + "."): name for name in modules}
    wanted = set()
    for pattern, regex in _target_patterns(adapter["target_modules"]):
        matches = {
            native
            for alias, native in aliases.items()
            if (
                re.fullmatch(pattern, alias)
                if regex
                else alias == pattern or alias.endswith("." + pattern)
            )
        }
        if not matches:
            raise ValueError("Unsupported adapter target mapping")
        wanted.update(matches)
    factors: dict[str, dict[str, str]] = {}
    names = set()
    for name, tensor in physical.items():
        original = name.removeprefix("kev.lora.")
        module, factor, adapter_name = _factor_module(original)
        names.add(adapter_name)
        if module not in aliases or aliases[module] not in wanted:
            raise ValueError("Orphan or unsupported adapter factor")
        native = aliases[module]
        pair = factors.setdefault(native, {})
        out, inp = modules[native]
        if factor in pair or list(tensor.shape) != ([rank, inp] if factor == "A" else [out, rank]):
            raise ValueError("Duplicate or malformed adapter factor")
        pair[factor] = original
    if (
        set(factors) != wanted
        or len(names) != 1
        or any(set(p) != {"A", "B"} for p in factors.values())
    ):
        raise ValueError("Missing or ambiguous adapter factors")
    return factors


def configuration(inputs: AnalysisInput) -> dict[str, Any] | None:
    marker = inputs.configuration.get("kev_inspection")
    if not isinstance(marker, dict) or set(marker) != {
        "format_version",
        "base_repository",
        "base_revision",
        "kev_revision",
        "head",
        "adapter",
    }:
        return None
    if (
        type(marker["format_version"]) is not int
        or marker["format_version"] != 1
        or marker["base_repository"] != BASE_REPOSITORY
        or inputs.configuration.get("model_type") != "qwen3_5"
        or inputs.configuration.get("architectures") != ["Qwen3_5ForConditionalGeneration"]
        or inputs.configuration.get("quantization_config") is not None
        or inputs.lora_composition is not None
        or inputs.bindings.adapter_tensor_storage
    ):
        return None
    for key in ("base_revision", "kev_revision"):
        if not isinstance(marker[key], str) or not re.fullmatch(r"[0-9a-f]{40}", marker[key]):
            return None
    if (
        inputs.configuration.get("_commit_hash", marker["kev_revision"]) != marker["kev_revision"]
        or inputs.configuration.get("revision", marker["base_revision"]) != marker["base_revision"]
        or inputs.configuration.get("name_or_path", BASE_REPOSITORY) != BASE_REPOSITORY
    ):
        return None
    c = qwen_configuration(backbone_inputs(inputs))
    if c is None or "linear_attention" not in c["layer_types"]:
        return None
    expected = parameter_shapes(c)
    physical = inputs.bindings.physical
    if {n for n in physical if n.startswith(PREFIX + ".")} != set(expected):
        return None
    try:
        head = head_configuration(marker["head"], marker["base_revision"])
        adapter = marker["adapter"]
        if not isinstance(adapter, dict) or adapter.get("r") != head["lora"]:
            return None
        factors = factor_bindings(
            {k: v for k, v in physical.items() if k.startswith("kev.lora.")}, expected, adapter
        )
    except (ValueError, TypeError, ModelError):
        return None
    h, dp = c["hidden_size"], head["head_dim"]
    heads = {
        "kev.head.q.weight": [dp, h],
        "kev.head.k.weight": [dp, h],
        "kev.head.q.bias": [dp],
        "kev.head.k.bias": [dp],
    }
    if {n for n in physical if n.startswith("kev.")} != set(heads) | {
        "kev.lora." + name for pair in factors.values() for name in pair.values()
    }:
        return None
    numeric = {t.name: t for t in inputs.bindings.numeric.values()}
    geometry = expected | heads
    for name, tensor in physical.items():
        if tensor.dtype not in {"F32", "F16", "BF16"}:
            return None
        if name in geometry and tensor.shape != geometry[name]:
            return None
        if (
            not name.startswith((PREFIX + ".", "kev.", "model.visual.", "mtp."))
            and name != "lm_head.weight"
        ):
            return None
        if len(tensor.shape) in (1, 2) and (
            name not in numeric
            or list(numeric[name].shape) != tensor.shape
            or numeric[name].dtype != tensor.dtype
            or numeric[name].storage_format != "safetensors"
        ):
            return None
    return {"head": head, "adapter": adapter, "factors": factors}

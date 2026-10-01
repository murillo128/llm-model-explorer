"""Explicit CLM inspection metadata and native head geometry; no checkpoint execution."""

import re
from collections.abc import Mapping
from dataclasses import replace
from typing import Any

from .core import AnalysisInput
from .dense_config import checked

SOURCE_REVISION = "bb42c6c5bf914fd449bed2f6ca65be80602cb1f7"
ENCODER_REPOSITORY = "Qwen/Qwen3-8B"
HEAD_REPOSITORY = "Contrastive-LM/CLM-v0.1-8B"
HEAD_FIELDS = {
    "model",
    "hidden_size",
    "projection_dim",
    "width",
    "depth",
    "activation",
    "layernorm",
    "residual",
}


def reserved(inputs: AnalysisInput) -> bool:
    """Prevent bare-Qwen fallback even when an inspection marker is damaged or removed."""
    return (
        "clm_inspection" in inputs.configuration
        or any(name.startswith("clm.") for name in inputs.bindings.physical)
        or any(
            inputs.configuration.get(key) == HEAD_REPOSITORY
            for key in ("_name_or_path", "name_or_path")
        )
    )


def head_configuration(value: object, hidden: int) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != HEAD_FIELDS:
        raise ValueError("Unsupported head configuration")
    cfg = dict(value)
    if cfg["model"] != ENCODER_REPOSITORY or cfg["hidden_size"] != hidden:
        raise ValueError("Head requires the selected Qwen3-8B encoder width")
    for key in ("hidden_size", "width", "projection_dim", "depth"):
        if type(cfg[key]) is not int or not 0 < cfg[key] <= 16384:
            raise ValueError("Invalid head dimensions")
    if not 2 <= cfg["depth"] <= 16 or cfg["activation"] not in ("gelu", "relu", "silu"):
        raise ValueError("Unsupported head depth or activation")
    if any(type(cfg[k]) is not bool for k in ("layernorm", "residual")):
        raise ValueError("Invalid head options")
    return cfg


def head_shapes(cfg: Mapping[str, Any]) -> dict[str, tuple[int, ...]]:
    width, hidden, proj = cfg["width"], cfg["hidden_size"], cfg["projection_dim"]
    expected = {
        "inp.weight": (width, hidden),
        "inp.bias": (width,),
        "out.weight": (proj, width),
        "out.bias": (proj,),
    }
    for i in range(cfg["depth"] - 2):
        expected.update({f"hidden.{i}.weight": (width, width), f"hidden.{i}.bias": (width,)})
        if cfg["layernorm"]:
            expected.update({f"norms.{i}.weight": (width,), f"norms.{i}.bias": (width,)})
    return expected


def encoder_inputs(inputs: AnalysisInput) -> AnalysisInput:
    """Expose just the shared Qwen backbone to its existing reviewed description."""
    config = {k: v for k, v in inputs.configuration.items() if k != "clm_inspection"}
    config["_name_or_path"] = ENCODER_REPOSITORY
    config.pop("name_or_path", None)
    bindings = replace(
        inputs.bindings,
        physical={k: v for k, v in inputs.bindings.physical.items() if not k.startswith("clm.")},
        numeric={k: v for k, v in inputs.bindings.numeric.items() if not v.name.startswith("clm.")},
    )
    return replace(inputs, configuration=config, bindings=bindings)


def configuration(inputs: AnalysisInput) -> dict[str, Any] | None:
    marker = inputs.configuration.get("clm_inspection")
    if not isinstance(marker, dict) or set(marker) != {
        "format_version",
        "encoder_repository",
        "encoder_revision",
        "head_revision",
        "head_configuration",
        "pooling",
    }:
        return None
    if (
        type(marker["format_version"]) is not int
        or marker["format_version"] != 1
        or marker["encoder_repository"] != ENCODER_REPOSITORY
        or marker["pooling"] != "last_token"
        or inputs.configuration.get("model_type") != "qwen3"
        or inputs.configuration.get("architectures") != ["Qwen3ForCausalLM"]
        or inputs.configuration.get("quantization_config") is not None
        or inputs.lora_composition is not None
        or inputs.bindings.adapter_tensor_storage
    ):
        return None
    for key in ("encoder_revision", "head_revision"):
        if not isinstance(marker[key], str) or not re.fullmatch(r"[0-9a-f]{40}", marker[key]):
            return None
    if inputs.configuration.get("_commit_hash", marker["head_revision"]) != marker["head_revision"]:
        return None
    dense = checked(encoder_inputs(inputs).configuration)
    if dense is None:
        return None
    try:
        cfg = head_configuration(marker["head_configuration"], dense.hidden)
    except (ValueError, TypeError):
        return None
    expected = {
        f"clm.{head}.{key}": dims
        for head in ("state_head", "action_head")
        for key, dims in head_shapes(cfg).items()
    }
    expected["clm.logit_scale"] = ()
    physical = inputs.bindings.physical
    numeric = {tensor.name: tensor for tensor in inputs.bindings.numeric.values()}
    if {name for name in physical if name.startswith("clm.")} != set(expected):
        return None
    for name, dims in expected.items():
        if tuple(physical[name].shape) != dims:
            return None
        if len(dims) in (1, 2) and (
            name not in numeric
            or numeric[name].shape != dims
            or numeric[name].dtype != physical[name].dtype
            or numeric[name].storage_format != "safetensors"
        ):
            return None
    if any(storage.dtype not in {"F32", "F16", "BF16"} for storage in physical.values()):
        return None
    return cfg

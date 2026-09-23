"""Admission and logical bindings for reviewed packed layouts.

These rules validate physical groups and bind complete mathematical tensors.
Numeric decoding stays in quantized_decoding; graph semantics stay in analysis.
"""

import hashlib
import re
import struct
from collections.abc import Mapping, Sequence
from fnmatch import fnmatchcase
from typing import Literal, Protocol

from .bnb_config import is_supported_config
from .bnb_nf4 import NF4_ENCODING, NF4State, validate_group
from .model_files import FileSnapshot, ModelError, changed, invalid
from .tensor_source import (
    DTYPES,
    PhysicalTensor,
    TensorDescriptor,
    TensorLocation,
    native_location,
    safe_integer,
)

Encoding = Literal[
    "native",
    "gptq-int4",
    "nvfp4",
    "compressed-tensors-w4a16-int4",
    "bnb-nf4-dq",
]
_MODULE_NAME = re.compile(r"[A-Za-z0-9_]+(?:\.[A-Za-z0-9_]+)*\Z")


class StorageMetadata(Protocol):
    @property
    def dtype(self) -> str: ...
    @property
    def shape(self) -> Sequence[int]: ...


def unsupported() -> ModelError:
    return ModelError("unsupported_representation", "Unsupported checkpoint encoding layout.")


def encoding(config: dict[str, object]) -> Encoding:
    quant = config.get("quantization_config")
    if quant is None:
        return "native"
    if not isinstance(quant, dict):
        raise unsupported()
    if config.get("model_type") == "qwen3" and quant.get("quant_method") == "gptq":
        required = {
            "bits": 4,
            "checkpoint_format": "gptq",
            "desc_act": False,
            "group_size": 128,
            "lm_head": False,
            "pack_dtype": "int32",
            "quant_method": "gptq",
            "sym": True,
        }
        if any(type(quant.get(k)) is not type(v) or quant[k] != v for k, v in required.items()):
            raise unsupported()
        if set(quant) - (required.keys() | {"meta", "hyb_act"}):
            raise unsupported()
        if quant.get("hyb_act", False) is not False:
            raise unsupported()
        return "gptq-int4"
    if config.get("model_type") == "qwen3_5" and quant.get("quant_method") == "modelopt":
        if set(quant) != {"config_groups", "ignore", "quant_algo", "producer", "quant_method"}:
            raise unsupported()
        groups = quant["config_groups"]
        if not isinstance(groups, dict) or set(groups) != {"group_0"}:
            raise unsupported()
        group = groups["group_0"]
        if not isinstance(group, dict) or set(group) != {"weights", "input_activations", "targets"}:
            raise unsupported()
        for key in ("weights", "input_activations"):
            schema = group[key]
            required = {"dynamic": False, "num_bits": 4, "type": "float", "group_size": 16}
            if not isinstance(schema, dict) or set(schema) != set(required):
                raise unsupported()
            if any(type(schema[k]) is not type(v) or schema[k] != v for k, v in required.items()):
                raise unsupported()
        producer = quant["producer"]
        if (
            quant["quant_algo"] != "NVFP4"
            or group["targets"] != ["Linear"]
            or not isinstance(producer, dict)
            or producer.get("name") != "modelopt"
            or not isinstance(quant["ignore"], list)
            or any(not isinstance(item, str) or not item for item in quant["ignore"])
        ):
            raise unsupported()
        return "nvfp4"
    if (
        config.get("model_type") in {"glm4_moe_lite", "kimi_linear"}
        and quant.get("quant_method") == "compressed-tensors"
    ):
        _validate_compressed_tensors_config(config, quant)
        return "compressed-tensors-w4a16-int4"
    if quant.get("quant_method") == "bitsandbytes" and is_supported_config(quant):
        return NF4_ENCODING
    raise unsupported()


def _validate_compressed_tensors_config(
    config: dict[str, object], quant: dict[str, object]
) -> None:
    """Admit only the pinned GLM/Kimi compressed-tensors W4A16 config subset."""
    expected_version = (
        "0.13.1.a20260219"
        if config.get("model_type") == "glm4_moe_lite"
        else "0.12.3.dev20+gd429903"
    )
    expected_quant_keys = {
        "config_groups",
        "format",
        "global_compression_ratio",
        "ignore",
        "kv_cache_scheme",
        "quant_method",
        "quantization_status",
        "sparsity_config",
        "transform_config",
        "version",
    }
    if set(quant) != expected_quant_keys:
        raise unsupported()
    if (
        quant["format"] != "pack-quantized"
        or quant["quant_method"] != "compressed-tensors"
        or quant["quantization_status"] != "compressed"
        or quant["global_compression_ratio"] is not None
        or quant["kv_cache_scheme"] is not None
        or quant["sparsity_config"] != {}
        or quant["transform_config"] != {}
        or quant["version"] != expected_version
    ):
        raise unsupported()

    groups = quant["config_groups"]
    if not isinstance(groups, dict) or set(groups) != {"group_0"}:
        raise unsupported()
    group = groups["group_0"]
    if not isinstance(group, dict) or set(group) != {
        "format",
        "input_activations",
        "output_activations",
        "targets",
        "weights",
    }:
        raise unsupported()
    if (
        group["format"] != "pack-quantized"
        or group["input_activations"] is not None
        or group["output_activations"] is not None
        or group["targets"] != ["Linear"]
    ):
        raise unsupported()

    weights = group["weights"]
    base_weights: dict[str, object] = {
        "actorder": None,
        "block_structure": None,
        "dynamic": False,
        "group_size": 32,
        "num_bits": 4,
        "observer": "mse",
        "observer_kwargs": {},
        "strategy": "group",
        "symmetric": True,
        "type": "int",
    }
    optional_dtype_fields = {"scale_dtype", "zp_dtype"}
    if not isinstance(weights, dict) or frozenset(weights) not in {
        frozenset(base_weights),
        frozenset(base_weights) | optional_dtype_fields,
    }:
        raise unsupported()
    if any(
        type(weights[key]) is not type(value) or weights[key] != value
        for key, value in base_weights.items()
    ):
        raise unsupported()
    if any(weights.get(key) is not None for key in optional_dtype_fields):
        raise unsupported()

    ignored = quant["ignore"]
    if (
        not isinstance(ignored, list)
        or not ignored
        or any(not isinstance(name, str) or not _MODULE_NAME.fullmatch(name) for name in ignored)
        or len(set(ignored)) != len(ignored)
    ):
        raise unsupported()

    dtypes = [config.get(key) for key in ("dtype", "torch_dtype") if key in config]
    if (
        not dtypes
        or any(
            not isinstance(value, str) or value not in {"bfloat16", "float16"} for value in dtypes
        )
        or len(set(dtypes)) != 1
    ):
        raise unsupported()


def _expect(
    tensors: Mapping[str, StorageMetadata], name: str, dtype: str, shape: tuple[int, ...]
) -> None:
    tensor = tensors.get(name)
    if tensor is None or tensor.dtype != dtype or tuple(tensor.shape) != shape:
        raise invalid("Incomplete or inconsistent quantized storage group.")


def _gptq(tensors: dict[str, PhysicalTensor]) -> set[str]:
    excluded: set[str] = set()
    for name, tensor in tensors.items():
        if not name.endswith(".qweight"):
            continue
        prefix = name.removesuffix(".qweight")
        if tensor.dtype != "I32" or len(tensor.shape) != 2 or min(tensor.shape) <= 0:
            raise invalid("Invalid GPTQ packed weight geometry.")
        packed_input, output = tensor.shape
        inputs = safe_integer(packed_input * 8)
        safe_integer(inputs * output * 4)
        if inputs % 128 or output % 8 or prefix + ".weight" in tensors:
            raise invalid("Unsupported or ambiguous GPTQ packed weight geometry.")
        _expect(tensors, prefix + ".qzeros", "I32", (inputs // 128, output // 8))
        _expect(tensors, prefix + ".scales", "F16", (inputs // 128, output))
        _expect(tensors, prefix + ".g_idx", "I32", (inputs,))
        excluded.update(prefix + suffix for suffix in (".qweight", ".qzeros", ".scales", ".g_idx"))
    return excluded


def nvfp4_storage_names(
    tensors: Mapping[str, StorageMetadata], config: dict[str, object]
) -> set[str]:
    """Validate admitted NVFP4 metadata without requiring file locations or tensor reads."""
    excluded: set[str] = set()
    quant = config["quantization_config"]
    assert isinstance(quant, dict)
    for name, tensor in tensors.items():
        if tensor.dtype != "U8" or not name.endswith(".weight"):
            continue
        prefix = name.removesuffix(".weight")
        if len(tensor.shape) != 2 or min(tensor.shape) <= 0:
            raise invalid("Invalid NVFP4 packed weight geometry.")
        output, packed_input = tensor.shape
        safe_integer(output * packed_input * 2 * 4)
        if packed_input % 8 or any(fnmatchcase(prefix, pattern) for pattern in quant["ignore"]):
            raise invalid("NVFP4 storage disagrees with its quantization configuration.")
        _expect(tensors, prefix + ".weight_scale", "F8_E4M3", (output, packed_input // 8))
        _expect(tensors, prefix + ".weight_scale_2", "F32", ())
        _expect(tensors, prefix + ".input_scale", "F32", ())
        excluded.update(
            prefix + suffix
            for suffix in (".weight", ".weight_scale", ".weight_scale_2", ".input_scale")
        )
    return excluded


def _compressed_tensors_storage(
    tensors: Mapping[str, PhysicalTensor],
    config: dict[str, object],
    snapshot: FileSnapshot,
) -> tuple[set[str], dict[str, tuple[int, int]]]:
    """Validate packed groups and their serialized logical shape metadata."""
    quant = config["quantization_config"]
    assert isinstance(quant, dict)
    ignored = set(quant["ignore"])
    configured_dtype = config.get("dtype", config.get("torch_dtype"))
    scale_dtype = "BF16" if configured_dtype == "bfloat16" else "F16"
    excluded: set[str] = set()
    logical_shapes: dict[str, tuple[int, int]] = {}

    for name, packed in tensors.items():
        if not name.endswith(".weight_packed"):
            continue
        prefix = name.removesuffix(".weight_packed")
        if (
            packed.dtype != "I32"
            or len(packed.shape) != 2
            or min(packed.shape) <= 0
            or prefix + ".weight" in tensors
            or prefix in ignored
        ):
            raise invalid("Invalid or ignored compressed-tensors packed weight group.")
        output, packed_input = packed.shape
        if packed_input % 4:
            raise invalid("Compressed-tensors input dimension is not group aligned.")
        inputs = safe_integer(packed_input * 8)
        if inputs % 32:
            raise invalid("Compressed-tensors input dimension is not group aligned.")
        safe_integer(output * inputs * 4)

        scale_name = prefix + ".weight_scale"
        shape_name = prefix + ".weight_shape"
        zero_name = prefix + ".weight_zero_point"
        if zero_name in tensors:
            raise unsupported()
        _expect(tensors, scale_name, scale_dtype, (output, inputs // 32))
        _expect(tensors, shape_name, "I64", (2,))

        shape_tensor = tensors[shape_name]
        with snapshot.open(shape_tensor.file) as stream:
            stream.seek(shape_tensor.offset)
            raw_shape = stream.read(16)
        if len(raw_shape) != 16:
            raise changed()
        declared_output, declared_input = struct.unpack("<qq", raw_shape)
        if (
            declared_output != output
            or declared_input != inputs
            or declared_output <= 0
            or declared_input <= 0
        ):
            raise invalid("Compressed-tensors logical shape metadata disagrees with its group.")
        safe_integer(declared_output)
        safe_integer(declared_input)

        logical_shapes[prefix] = (declared_output, declared_input)
        excluded.update((name, scale_name, shape_name))

    return excluded, logical_shapes


def _bnb_nf4_storage_groups(
    tensors: dict[str, PhysicalTensor], snapshot: FileSnapshot
) -> tuple[dict[str, tuple[tuple[PhysicalTensor, ...], NF4State]], set[str]]:
    groups: dict[str, tuple[tuple[PhysicalTensor, ...], NF4State]] = {}
    excluded: set[str] = set()
    for tensor in tensors.values():
        if tensor.name.endswith(".weight") and tensor.dtype == "U8":
            group, state = validate_group(snapshot, tensors, tensor)
            groups[tensor.name] = (group, state)
            excluded.update(item.name for item in group)
    if not groups:
        raise invalid("Bitsandbytes NF4 checkpoint has no supported packed storage groups.")
    return groups, excluded


def logical_locations(
    config: dict[str, object], physical: tuple[PhysicalTensor, ...], snapshot: FileSnapshot
) -> tuple[TensorLocation, ...]:
    layout = encoding(config)
    if layout == "native":
        # Never filter/rename baseline native tensors, including scalars and buffers.
        return tuple(native_location(tensor) for tensor in physical)
    tensors = {tensor.name: tensor for tensor in physical}
    logical_shapes: dict[str, tuple[int, int]] = {}
    bnb_groups: dict[str, tuple[tuple[PhysicalTensor, ...], NF4State]] = {}
    if layout == "gptq-int4":
        excluded = _gptq(tensors)
    elif layout == "nvfp4":
        excluded = nvfp4_storage_names(tensors, config)
    elif layout == "compressed-tensors-w4a16-int4":
        excluded, logical_shapes = _compressed_tensors_storage(tensors, config, snapshot)
    else:
        bnb_groups, excluded = _bnb_nf4_storage_groups(tensors, snapshot)
    if not excluded:
        raise invalid("Quantized checkpoint has no supported packed storage groups.")
    locations = []
    auxiliaries = {
        "qweight",
        "qzeros",
        "scales",
        "g_idx",
        "weight_scale",
        "weight_scale_2",
        "input_scale",
        "weight_packed",
        "weight_shape",
        "weight_zero_point",
        "absmax",
        "quant_map",
        "nested_absmax",
        "nested_quant_map",
    }
    for tensor in physical:
        if tensor.name in excluded:
            if layout == NF4_ENCODING and tensor.name in bnb_groups:
                group, state = bnb_groups[tensor.name]
                name = tensor.name
                shape = state.shape
                storage = group
            else:
                packed_suffix = {
                    "gptq-int4": ".qweight",
                    "nvfp4": ".weight",
                    "compressed-tensors-w4a16-int4": ".weight_packed",
                }.get(layout)
                if packed_suffix is None or not tensor.name.endswith(packed_suffix):
                    continue
                prefix = tensor.name.removesuffix(packed_suffix)
                name = prefix + ".weight"
                if layout == "gptq-int4":
                    suffixes: tuple[str, ...] = (".qweight", ".qzeros", ".scales", ".g_idx")
                    shape = (tensor.shape[1], tensor.shape[0] * 8)
                elif layout == "nvfp4":
                    suffixes = (".weight", ".weight_scale", ".weight_scale_2", ".input_scale")
                    shape = (tensor.shape[0], tensor.shape[1] * 2)
                elif layout == "compressed-tensors-w4a16-int4":
                    suffixes = (".weight_packed", ".weight_scale", ".weight_shape")
                    shape = logical_shapes[prefix]
                else:
                    raise invalid("Unsupported compressed storage layout.")
                storage = tuple(tensors[prefix + suffix] for suffix in suffixes)
            locations.append(
                TensorLocation(
                    TensorDescriptor(
                        id=hashlib.sha256(name.encode("utf-8")).hexdigest(),
                        name=name,
                        path=tuple(name.split(".")),
                        shape=shape,
                        rank=2,
                        numel=safe_integer(shape[0] * shape[1]),
                        storage_dtype=tensor.dtype,
                        storage_format=layout,
                    ),
                    tensor.file,
                    tensor.offset,
                    storage,
                )
            )
            continue
        leaf = tensor.name.rsplit(".", 1)[-1]
        if tensor.dtype not in DTYPES or leaf in auxiliaries:
            # Unknown/orphan records remain visible to structural analysis and
            # inventory diagnostics, never a guessed actionable numeric tensor.
            continue
        # In these formats native weight/bias fields are complete, uncompressed
        # tensors, unlike encoding auxiliaries above. Physical shape is therefore
        # the full logical shape; no axis transform, alias or region is invented.
        # Unknown native fields remain available to analysis but have no numeric ID.
        if (leaf == "weight" and len(tensor.shape) >= 1) or (
            leaf in {"bias", "A_log", "dt_bias"} and len(tensor.shape) == 1
        ):
            locations.append(native_location(tensor))
    return tuple(sorted(locations, key=lambda location: location.descriptor.name))

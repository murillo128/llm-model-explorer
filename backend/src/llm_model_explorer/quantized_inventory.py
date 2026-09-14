"""Admission rules for the two reviewed packed layouts, without numeric decoders.

These rules validate physical encoding groups. They do not infer a semantic graph
or manufacture logical parameters for packed storage. See evidence/quantized-admission.md.
"""

from collections.abc import Mapping, Sequence
from fnmatch import fnmatchcase
from typing import Literal, Protocol

from .model_files import ModelError, invalid
from .tensor_source import DTYPES, PhysicalTensor, TensorLocation, native_location, safe_integer

Encoding = Literal["native", "gptq-int4", "nvfp4"]


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


def logical_locations(
    config: dict[str, object], physical: tuple[PhysicalTensor, ...]
) -> tuple[TensorLocation, ...]:
    layout = encoding(config)
    if layout == "native":
        # Never filter/rename baseline native tensors, including scalars and buffers.
        return tuple(native_location(tensor) for tensor in physical)
    tensors = {tensor.name: tensor for tensor in physical}
    excluded = _gptq(tensors) if layout == "gptq-int4" else nvfp4_storage_names(tensors, config)
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
    }
    for tensor in physical:
        if tensor.name in excluded:
            continue
        leaf = tensor.name.rsplit(".", 1)[-1]
        if tensor.dtype not in DTYPES or leaf in auxiliaries:
            raise invalid("Unrecognized or orphaned quantized storage.")
        # In these formats native weight/bias fields are complete, uncompressed
        # tensors, unlike encoding auxiliaries above. Physical shape is therefore
        # the full logical shape; no axis transform, alias or region is invented.
        # Unknown native fields remain available to analysis but have no numeric ID.
        if (leaf == "weight" and len(tensor.shape) >= 1) or (
            leaf in {"bias", "A_log", "dt_bias"} and len(tensor.shape) == 1
        ):
            locations.append(native_location(tensor))
    return tuple(locations)

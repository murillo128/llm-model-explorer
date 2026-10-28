"""The selected video packages' tensor-wise E4M3FN storage, not activation math.

Format reference: ComfyUI 79cf64c21a1dba50749945ba9fe2e3a692f45438,
comfy/utils.py::convert_old_quants. The global marker selects serialization;
scale_input/input_scale belongs to activations and never scales stored weights.
"""

import hashlib
from collections.abc import Mapping

import torch

from .model_files import FileSnapshot, invalid
from .tensor_source import DTYPES, PhysicalTensor, TensorDescriptor, TensorLocation, native_location

ENCODING = "fp8-e4m3-scaled"
COMPONENTS = frozenset({"video.text_encoder", "video.high_noise", "video.low_noise"})
AUXILIARIES = frozenset(
    {"scale_weight", "scale_input", "weight_scale", "input_scale", "scaled_fp8"}
)


def component_layouts(config: Mapping[str, object]) -> dict[str, str]:
    quant = config.get("quantization_config")
    if (
        config.get("model_type") != "video_inspection"
        or not isinstance(quant, dict)
        or set(quant) != {"quant_method", "format_version", "components"}
        or quant.get("quant_method") != ENCODING
        or type(quant.get("format_version")) is not int
        or quant["format_version"] != 1
    ):
        raise invalid("Unsupported video FP8 declaration.")
    components = quant["components"]
    if (
        not isinstance(components, dict)
        or not components
        or not set(components) <= COMPONENTS
        or any(value not in ("legacy", "normalized") for value in components.values())
    ):
        raise invalid("Unsupported video FP8 component layout.")
    return dict(components)


def _scale(snapshot: FileSnapshot, tensor: PhysicalTensor) -> None:
    if tensor.dtype not in DTYPES or tensor.shape not in ((), (1,)):
        raise invalid("FP8 scale must be one native floating scalar.")
    # Reuse the guarded endian-aware reader rather than interpreting scale bytes
    # as host-endian here. Its return is owned and bounded to one element.
    from .quantized_decoding import _gather

    value = _gather(snapshot, tensor, torch.zeros(1, dtype=torch.int64)).float()
    if not bool(torch.isfinite(value).all()):
        raise invalid("FP8 scale must be finite.")


def logical_locations(
    config: Mapping[str, object], physical: tuple[PhysicalTensor, ...], snapshot: FileSnapshot
) -> tuple[TensorLocation, ...]:
    layouts = component_layouts(config)
    tensors = {item.name: item for item in physical}
    owned: set[str] = set()
    locations: list[TensorLocation] = []
    for component, layout in layouts.items():
        prefix = component + "."
        marker = tensors.get(prefix + "scaled_fp8")
        if layout == "legacy":
            if (
                marker is None
                or marker.dtype not in {"F32", "F8_E4M3"}
                or marker.shape not in ((), (0,), (1,), (2,))
            ):
                raise invalid("Missing or invalid legacy FP8 marker.")
            owned.add(marker.name)
        elif marker is not None:
            raise invalid("Normalized FP8 declaration contains a legacy marker.")
        found = False
        for tensor in physical:
            if not tensor.name.startswith(prefix) or tensor.dtype != "F8_E4M3":
                continue
            if tensor is marker:
                continue
            if (
                not tensor.name.endswith(".weight")
                or len(tensor.shape) != 2
                or min(tensor.shape) < 1
            ):
                raise invalid("Unsupported FP8 weight geometry.")
            module = tensor.name.removesuffix(".weight")
            weight_key = module + (".scale_weight" if layout == "legacy" else ".weight_scale")
            input_key = module + (".scale_input" if layout == "legacy" else ".input_scale")
            weight_scale = tensors.get(weight_key)
            if weight_scale is None:
                raise invalid("Missing FP8 weight scale.")
            _scale(snapshot, weight_scale)
            group = [tensor, weight_scale]
            input_scale = tensors.get(input_key)
            if input_scale is not None:
                _scale(snapshot, input_scale)
                group.append(input_scale)
            if marker is not None:
                group.append(marker)
            owned.update(item.name for item in group)
            locations.append(
                TensorLocation(
                    TensorDescriptor(
                        id=hashlib.sha256(tensor.name.encode()).hexdigest(),
                        name=tensor.name,
                        path=tuple(tensor.name.split(".")),
                        shape=tensor.shape,
                        rank=2,
                        numel=tensor.numel,
                        storage_dtype=tensor.dtype,
                        storage_format=ENCODING,
                    ),
                    tensor.file,
                    tensor.offset,
                    tuple(group),
                )
            )
            found = True
        if not found:
            raise invalid("Declared FP8 component has no scaled weights.")
    for tensor in physical:
        if tensor.name in owned:
            continue
        if tensor.name.rsplit(".", 1)[-1] in AUXILIARIES or tensor.dtype not in DTYPES:
            raise invalid("Orphan or inconsistent FP8 storage auxiliary.")
        locations.append(native_location(tensor))
    snapshot.check()
    return tuple(sorted(locations, key=lambda item: item.descriptor.name))

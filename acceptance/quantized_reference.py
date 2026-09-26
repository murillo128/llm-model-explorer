"""Independent, bounded reference samples for the accepted packed formats.

NF4 uses the pinned bitsandbytes codebook and nested-scale decoder. Compressed
W4A16 uses direct scalar unpacking from the reviewed pack-quantized layout. This
module never imports a checkpoint's Python files or the product quantization
decoder.
"""

from __future__ import annotations

import json
import struct
from contextlib import ExitStack
from importlib.metadata import version
from pathlib import Path

import torch
from safetensors import safe_open

BNB_VERSION = "0.50.2"
BNB_SOURCE_REVISION = "833649043474794b8fe7a4136e0c40faf077b2e0"
COMPRESSED_TENSORS_SOURCE_REVISION = "4a696625b7ada2cb857c75f67ce02dc56b381323"
TOLERANCE = 1e-7


def oracle_identity(encoding: str) -> dict[str, str | float]:
    if encoding == "bnb-nf4-dq":
        return {
            "package": "bitsandbytes",
            "version": BNB_VERSION,
            "source_revision": BNB_SOURCE_REVISION,
            "tolerance_absolute": TOLERANCE,
        }
    if encoding == "compressed-tensors-w4a16-int4":
        return {
            "package": "compressed-tensors",
            "version": "source-derived scalar oracle",
            "source_revision": COMPRESSED_TENSORS_SOURCE_REVISION,
            "tolerance_absolute": TOLERANCE,
        }
    raise ValueError(f"No independent oracle for {encoding!r}")


def _weight_map(directory: Path) -> dict[str, str]:
    index = directory / "model.safetensors.index.json"
    if not index.is_file():
        return {}
    return json.loads(index.read_text())["weight_map"]


def _record_path(directory: Path, name: str, weight_map: dict[str, str]) -> Path:
    shard = weight_map.get(name)
    if shard is not None:
        return directory / shard
    candidates = sorted(directory.glob("*.safetensors"))
    for candidate in candidates:
        with safe_open(candidate, framework="pt", device="cpu") as source:
            if name in source.keys():  # noqa: SIM118 - safetensors.safe_open is not a mapping
                return candidate
    raise AssertionError(f"Reference record {name!r} is absent")


def _open_record(
    directory: Path,
    name: str,
    weight_map: dict[str, str],
    stack: ExitStack,
    opened: dict[Path, object],
):
    path = _record_path(directory, name, weight_map)
    source = opened.get(path)
    if source is None:
        source = stack.enter_context(safe_open(path, framework="pt", device="cpu"))
        opened[path] = source
    return source


def _nf4_sample(
    directory: Path, name: str, row: int, column: int, weight_map: dict[str, str]
) -> dict[str, object]:
    import bitsandbytes.functional as bnb_functional

    if version("bitsandbytes") != BNB_VERSION:
        raise AssertionError(f"NF4 oracle requires bitsandbytes {BNB_VERSION}")
    quant_state_name = name + ".quant_state.bitsandbytes__nf4"
    with ExitStack() as stack:
        opened: dict[Path, object] = {}
        source = _open_record(directory, name, weight_map, stack, opened)
        packed = source.get_slice(name)
        state_source = _open_record(directory, quant_state_name, weight_map, stack, opened)
        state = json.loads(bytes(state_source.get_tensor(quant_state_name).tolist()))
        shape = tuple(int(value) for value in state["shape"])
        if len(shape) != 2 or not (0 <= row < shape[0] and 0 <= column < shape[1]):
            raise AssertionError("NF4 reference coordinates are outside the logical matrix")
        if tuple(packed.get_shape()) != ((shape[0] * shape[1] + 1) // 2, 1):
            raise AssertionError("Serialized NF4 packed shape disagrees with its quant state")

        absmax_name = name + ".absmax"
        nested_absmax_name = name + ".nested_absmax"
        nested_map_name = name + ".nested_quant_map"
        quant_map_name = name + ".quant_map"
        absmax_source = _open_record(directory, absmax_name, weight_map, stack, opened)
        nested_absmax_source = _open_record(
            directory, nested_absmax_name, weight_map, stack, opened
        )
        nested_map_source = _open_record(directory, nested_map_name, weight_map, stack, opened)
        quant_map_source = _open_record(directory, quant_map_name, weight_map, stack, opened)
        nested_state = bnb_functional.QuantState(
            absmax=nested_absmax_source.get_tensor(nested_absmax_name),
            blocksize=int(state["nested_blocksize"]),
            code=nested_map_source.get_tensor(nested_map_name),
            dtype=getattr(torch, state["nested_dtype"]),
        )
        scales = bnb_functional.dequantize_blockwise(
            absmax_source.get_tensor(absmax_name), nested_state
        ) + float(state["nested_offset"])
        codebook = bnb_functional.get_4bit_type("nf4", device=torch.device("cpu"))
        recorded_codebook = quant_map_source.get_tensor(quant_map_name)
        if not torch.equal(recorded_codebook, codebook):
            raise AssertionError("Recorded NF4 codebook differs from bitsandbytes")

        logical_index = row * shape[1] + column
        packed_byte = int(packed[logical_index // 2 : logical_index // 2 + 1, 0:1].item())
        code = (packed_byte >> 4) & 15 if logical_index % 2 == 0 else packed_byte & 15
        expected = float((codebook[code] * scales[logical_index // int(state["blocksize"])]).item())
        return {
            "shape": list(shape),
            "offset": logical_index,
            "value": struct.unpack("<f", struct.pack("<f", expected))[0],
            "encoding": "bnb-nf4-dq",
        }


def _compressed_sample(
    directory: Path, name: str, row: int, column: int, weight_map: dict[str, str]
) -> dict[str, object]:
    config = json.loads((directory / "config.json").read_text())
    quantization = config.get("quantization_config", {})
    group = quantization.get("config_groups", {}).get("group_0", {})
    weights_config = group.get("weights", {})
    if not (
        quantization.get("quant_method") == "compressed-tensors"
        and quantization.get("format") == "pack-quantized"
        and group.get("format") == "pack-quantized"
        and weights_config.get("num_bits") == 4
        and weights_config.get("group_size") == 32
        and weights_config.get("symmetric") is True
    ):
        raise AssertionError("Checkpoint does not declare the reviewed compressed W4A16 group")

    prefix = name.removesuffix(".weight")
    packed_name = prefix + ".weight_packed"
    scale_name = prefix + ".weight_scale"
    with ExitStack() as stack:
        opened: dict[Path, object] = {}
        packed_source = _open_record(directory, packed_name, weight_map, stack, opened)
        scale_source = _open_record(directory, scale_name, weight_map, stack, opened)
        packed = packed_source.get_slice(packed_name)
        scales = scale_source.get_slice(scale_name)
        packed_shape = tuple(int(value) for value in packed.get_shape())
        scale_shape = tuple(int(value) for value in scales.get_shape())
        logical_shape = (packed_shape[0], scale_shape[1] * 32)
        if packed_shape != (logical_shape[0], logical_shape[1] // 8):
            raise AssertionError("Packed W4A16 tensor and scales disagree on geometry")
        if not (0 <= row < logical_shape[0] and 0 <= column < logical_shape[1]):
            raise AssertionError("Compressed W4A16 reference coordinates are outside the matrix")

        word = int(packed[row : row + 1, column // 8 : column // 8 + 1].item()) & 0xFFFFFFFF
        nibble = (word >> (4 * (column % 8))) & 15
        signed = nibble - 8
        scale = float(scales[row : row + 1, column // 32 : column // 32 + 1].float().item())
        expected = struct.unpack("<f", struct.pack("<f", signed * scale))[0]
        return {
            "shape": list(logical_shape),
            "offset": row * logical_shape[1] + column,
            "value": expected,
            "encoding": "compressed-tensors-w4a16-int4",
        }


def scalar_sample(directory: Path, name: str, row: int, column: int) -> dict[str, object]:
    directory = directory.resolve(strict=True)
    config = json.loads((directory / "config.json").read_text())
    quantization = config.get("quantization_config", {})
    weight_map = _weight_map(directory)
    if quantization.get("quant_method") == "bitsandbytes":
        return _nf4_sample(directory, name, row, column, weight_map)
    if quantization.get("quant_method") == "compressed-tensors":
        return _compressed_sample(directory, name, row, column, weight_map)
    raise ValueError("Reference model is not one of the accepted packed formats")

"""Validated bitsandbytes NF4/double-quant Safetensors metadata.

This module handles only the reviewed packed ``Params4bit`` representation.
It does not import bitsandbytes and never reads packed weight payloads while
checking the small metadata companions.
"""

import math
import struct
from dataclasses import dataclass
from math import isfinite
from typing import Final

from .model_files import FileSnapshot, ModelError, changed, invalid, parse_json
from .tensor_source import PhysicalTensor, safe_integer

NF4_ENCODING: Final = "bnb-nf4-dq"
NF4_BLOCK_SIZE: Final = 64
NESTED_BLOCK_SIZE: Final = 256
MAX_QUANT_STATE_BYTES: Final = 4096

# bitsandbytes.functional.get_4bit_type("nf4"), serialized as float32.
NF4_CODEBOOK: Final[tuple[float, ...]] = (
    -1.0,
    -0.6961928009986877,
    -0.5250730514526367,
    -0.39491748809814453,
    -0.28444138169288635,
    -0.18477343022823334,
    -0.09105003625154495,
    0.0,
    0.07958029955625534,
    0.16093020141124725,
    0.24611230194568634,
    0.33791524171829224,
    0.44070982933044434,
    0.5626170039176941,
    0.7229568362236023,
    1.0,
)


@dataclass(frozen=True)
class NF4State:
    shape: tuple[int, int]
    dtype: str
    nested_offset: float


def read_physical(snapshot: FileSnapshot, tensor: PhysicalTensor, limit: int) -> bytes:
    """Read one small, owned companion payload under the pinned snapshot."""
    if tensor.byte_length > limit:
        raise ModelError("unsupported_size", "Quantization metadata exceeds the supported size.")
    with snapshot.open(tensor.file) as stream:
        stream.seek(tensor.offset)
        data = stream.read(tensor.byte_length)
    if len(data) != tensor.byte_length:
        raise changed()
    return data


def parse_state(snapshot: FileSnapshot, tensor: PhysicalTensor) -> NF4State:
    """Read the serialized non-tensor fields packed by bitsandbytes as JSON U8."""
    if (
        not tensor.name.endswith(".quant_state.bitsandbytes__nf4")
        or tensor.dtype != "U8"
        or len(tensor.shape) != 1
        or not 0 < tensor.numel <= MAX_QUANT_STATE_BYTES
    ):
        raise invalid("Invalid bitsandbytes NF4 quantization-state record.")
    raw = read_physical(snapshot, tensor, MAX_QUANT_STATE_BYTES)
    state = parse_json(raw)
    expected = {
        "quant_type",
        "blocksize",
        "dtype",
        "shape",
        "nested_blocksize",
        "nested_dtype",
        "nested_offset",
    }
    if set(state) != expected:
        raise ModelError(
            "unsupported_representation", "Unsupported bitsandbytes quantization state."
        )
    shape = state["shape"]
    if not isinstance(shape, list) or len(shape) != 2:
        raise invalid("Bitsandbytes NF4 weights must record a matrix shape.")
    dims = tuple(safe_integer(dimension) for dimension in shape)
    if min(dims) <= 0:
        raise invalid("Bitsandbytes NF4 weights must have positive matrix dimensions.")
    if (
        state["quant_type"] != "nf4"
        or type(state["blocksize"]) is not int
        or state["blocksize"] != NF4_BLOCK_SIZE
        or type(state["nested_blocksize"]) is not int
        or state["nested_blocksize"] != NESTED_BLOCK_SIZE
        or not isinstance(state["dtype"], str)
        or state["dtype"] not in {"float16", "bfloat16", "float32"}
        or state["nested_dtype"] != "float32"
    ):
        raise ModelError("unsupported_representation", "Unsupported bitsandbytes NF4 state values.")
    offset = state["nested_offset"]
    if (
        not isinstance(offset, (int, float))
        or isinstance(offset, bool)
        or not isfinite(offset)
        or offset < 0
    ):
        raise invalid("Invalid bitsandbytes nested quantization offset.")
    return NF4State((dims[0], dims[1]), str(state["dtype"]), float(offset))


def validate_codebooks(
    snapshot: FileSnapshot, quant_map: PhysicalTensor, nested_quant_map: PhysicalTensor
) -> None:
    """Check the recorded NF4 and nested blockwise codebooks before admission."""
    if quant_map.dtype != "F32" or quant_map.shape != (len(NF4_CODEBOOK),):
        raise invalid("Invalid bitsandbytes NF4 codebook geometry.")
    expected = struct.pack("<16f", *NF4_CODEBOOK)
    if read_physical(snapshot, quant_map, len(expected)) != expected:
        raise ModelError("unsupported_representation", "Unreviewed bitsandbytes NF4 codebook.")
    if nested_quant_map.dtype != "F32" or nested_quant_map.shape != (256,):
        raise invalid("Invalid bitsandbytes nested codebook geometry.")
    raw = read_physical(snapshot, nested_quant_map, 256 * 4)
    values = tuple(value for (value,) in struct.iter_unpack("<f", raw))
    if (
        len(values) != 256
        or not all(math.isfinite(value) for value in values)
        or values[0] < -1.0
        or values[-1] != 1.0
        or any(left >= right for left, right in zip(values, values[1:], strict=False))
    ):
        raise ModelError("unsupported_representation", "Invalid bitsandbytes nested codebook.")


def validate_group(
    snapshot: FileSnapshot,
    tensors: dict[str, PhysicalTensor],
    weight: PhysicalTensor,
) -> tuple[tuple[PhysicalTensor, ...], NF4State]:
    """Validate one full physical group and return it in stable decoder order."""
    if weight.dtype != "U8" or not weight.name.endswith(".weight"):
        raise invalid("Invalid bitsandbytes packed weight record.")
    prefix = weight.name
    names = (
        "",
        ".absmax",
        ".quant_map",
        ".nested_absmax",
        ".nested_quant_map",
        ".quant_state.bitsandbytes__nf4",
    )
    group = tuple(tensors.get(prefix + suffix) for suffix in names)
    if any(item is None for item in group):
        raise invalid("Incomplete bitsandbytes NF4 storage group.")
    complete = tuple(item for item in group if item is not None)
    state = parse_state(snapshot, complete[-1])
    output, inputs = state.shape
    numel = safe_integer(output * inputs)
    absmax_count = safe_integer((numel + NF4_BLOCK_SIZE - 1) // NF4_BLOCK_SIZE)
    nested_count = safe_integer((absmax_count + NESTED_BLOCK_SIZE - 1) // NESTED_BLOCK_SIZE)
    expected: tuple[tuple[str, tuple[int, ...]], ...] = (
        ("U8", ((numel + 1) // 2, 1)),
        ("U8", (absmax_count,)),
        ("F32", (16,)),
        ("F32", (nested_count,)),
        ("F32", (256,)),
        ("U8", (complete[-1].numel,)),
    )
    if any(
        item.dtype != dtype or item.shape != shape
        for item, (dtype, shape) in zip(complete, expected, strict=True)
    ):
        raise invalid("Bitsandbytes NF4 companions disagree with the recorded logical shape.")
    if len(complete[-1].shape) != 1 or not 0 < complete[-1].numel <= MAX_QUANT_STATE_BYTES:
        raise invalid("Invalid bitsandbytes NF4 quantization-state geometry.")
    validate_codebooks(snapshot, complete[2], complete[4])
    return complete, state

"""Bounded logical float32 reads for the two admitted packed weight layouts.

Only requested storage elements are gathered through bounded owned file reads.
Never convert a complete physical array or expose a live file mapping.
Admission owns configuration, companion membership, dtypes and physical geometry.
"""

import sys

import torch

from .model_files import FileSnapshot, ModelError, changed
from .tensor_source import DEFAULT_CHUNK_ELEMENTS, PhysicalTensor, safe_integer

_DTYPES = {
    "I32": (torch.int32, 4),
    "U8": (torch.uint8, 1),
    "F16": (torch.float16, 2),
    "F32": (torch.float32, 4),
    "F8_E4M3": (torch.float8_e4m3fn, 1),
}


def _gather(snapshot: FileSnapshot, storage: PhysicalTensor, indices: torch.Tensor) -> torch.Tensor:
    """Read O(requested elements) owned bytes, including strided GPTQ columns.

    Coalesce adjacent unique storage positions into bounded I/O spans, then
    restore requested order with a vectorized gather. Python iterates file spans
    only; all per-value indexing and numeric decoding remain PyTorch operations.
    Owned reads let concurrent truncation fail through snapshot/short-read checks
    instead of risking SIGBUS by dereferencing a live file mapping.
    """
    dtype, width = _DTYPES[storage.dtype]
    if bool(torch.any(indices < 0)) or bool(torch.any(indices >= storage.numel)):
        raise ModelError("unsupported_representation", "Invalid packed tensor storage index.")
    unique, inverse = torch.unique(indices, sorted=True, return_inverse=True)
    breaks = torch.nonzero(unique[1:] != unique[:-1] + 1).flatten() + 1
    boundaries = [0, *breaks.tolist(), unique.numel()]
    starts = unique[boundaries[:-1]].tolist()
    raw = bytearray(unique.numel() * width)
    with snapshot.open(storage.file) as stream:
        for start, begin, end in zip(starts, boundaries[:-1], boundaries[1:], strict=True):
            stream.seek(storage.offset + start * width)
            length = (end - begin) * width
            block = stream.read(length)
            if len(block) != length:
                raise changed()
            raw[begin * width : end * width] = block
    view = torch.frombuffer(raw, dtype=torch.uint8)
    positions = inverse[:, None] * width + torch.arange(width)
    selected = torch.index_select(view, 0, positions.reshape(-1))
    if sys.byteorder != "little" and width > 1:
        selected = selected.reshape(-1, width).flip(1).contiguous().reshape(-1)
    return selected.view(dtype)


def decode_range(
    snapshot: FileSnapshot,
    storage: tuple[PhysicalTensor, ...],
    encoding: str,
    start: int,
    count: int,
) -> torch.Tensor:
    """Return one owned flat CPU chunk in logical [output,input] C order.

    GPTQ checkpoint_format=gptq stores zero_point - 1 in each qzeros nibble;
    adding one is not modulo 16. The stored g_idx selects groups verbatim.
    NVFP4 uses low-nibble-first E2M1, E4M3 block scales and the per-tensor
    weight scale. input_scale belongs to activations, not stored weights.
    Independent reference provenance and scalar oracles live in the tests.
    """
    safe_integer(start)
    safe_integer(count)
    if count > DEFAULT_CHUNK_ELEMENTS:
        raise ValueError("Packed reads exceed the logical chunk bound")
    tensors = {item.name.rsplit(".", 1)[-1]: item for item in storage}
    if encoding == "gptq-int4":
        packed_inputs, outputs = tensors["qweight"].shape
        inputs = safe_integer(packed_inputs * 8)
    elif encoding == "nvfp4":
        outputs, packed_inputs = tensors["weight"].shape
        inputs = safe_integer(packed_inputs * 2)
    else:
        raise ModelError("unsupported_representation", "Unsupported packed tensor encoding.")
    numel = safe_integer(outputs * inputs)
    if safe_integer(start + count) > numel:
        raise ModelError("validation_error", "Packed tensor range exceeds its logical shape.")
    snapshot.check()
    if not count:
        return torch.empty(0, dtype=torch.float32)
    try:
        logical = torch.arange(start, start + count, dtype=torch.int64)
        row, column = logical // inputs, logical % inputs

        def gather(name: str, indices: torch.Tensor) -> torch.Tensor:
            return _gather(snapshot, tensors[name], indices)

        if encoding == "gptq-int4":
            groups = gather("g_idx", column).to(torch.int64)
            if bool(torch.any(groups < 0)) or bool(torch.any(groups >= inputs // 128)):
                raise ModelError("unsupported_representation", "Invalid GPTQ group mapping.")
            packed = gather("qweight", (column // 8) * outputs + row)
            quantized = (packed >> ((column % 8) * 4)) & 15
            packed_zeros = gather("qzeros", groups * (outputs // 8) + row // 8)
            zeros = ((packed_zeros >> ((row % 8) * 4)) & 15) + 1
            scales = gather("scales", groups * outputs + row).to(torch.float32)
            values = (quantized - zeros).to(torch.float32) * scales
        else:
            packed = gather("weight", row * (inputs // 2) + column // 2)
            codes = ((packed >> ((column % 2) * 4)) & 15).to(torch.int64)
            table = torch.tensor(
                [
                    0.0,
                    0.5,
                    1.0,
                    1.5,
                    2.0,
                    3.0,
                    4.0,
                    6.0,
                    -0.0,
                    -0.5,
                    -1.0,
                    -1.5,
                    -2.0,
                    -3.0,
                    -4.0,
                    -6.0,
                ],
                dtype=torch.float32,
            )
            scales = gather("weight_scale", row * (inputs // 16) + column // 16).float()
            global_scale = gather("weight_scale_2", torch.zeros(1, dtype=torch.int64))
            # ModelOpt reconstructs the block scale in float32 before applying
            # it to E2M1 values; keep that evaluation order for rounding parity.
            values = table[codes] * (scales * global_scale)
    except torch.OutOfMemoryError as exc:
        raise MemoryError("Insufficient memory for packed tensor conversion.") from exc
    snapshot.check()
    return values

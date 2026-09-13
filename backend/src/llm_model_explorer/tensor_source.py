"""Safetensors header validation and bounded canonical float32 reads."""

import hashlib
import math
import struct
import sys
from collections.abc import Generator, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import torch
from pydantic import BaseModel, ConfigDict

from .model_files import MAX_METADATA_BYTES, FileSnapshot, ModelError, changed, invalid, parse_json

MAX_SAFE_INTEGER = 2**53 - 1
DEFAULT_CHUNK_ELEMENTS = 256 * 1024
DTYPES = {"F32": (torch.float32, 4), "F16": (torch.float16, 2), "BF16": (torch.bfloat16, 2)}


class TensorDescriptor(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str
    name: str
    path: tuple[str, ...]
    shape: tuple[int, ...]
    rank: int
    numel: int
    storage_dtype: str
    storage_format: str = "safetensors"
    logical_dtype: str = "float32"


@dataclass(frozen=True)
class TensorLocation:
    descriptor: TensorDescriptor
    file: str
    offset: int


def safe_integer(value: object) -> int:
    if type(value) is not int or not 0 <= value <= MAX_SAFE_INTEGER:
        raise ModelError("unsupported_size", "Tensor dimensions or offsets exceed safe limits.")
    return value


def parse_header(snapshot: FileSnapshot, name: str) -> tuple[TensorLocation, ...]:
    with snapshot.open(name) as stream:
        prefix = stream.read(8)
        if len(prefix) != 8:
            raise invalid("Truncated safetensors header.")
        length = struct.unpack("<Q", prefix)[0]
        size = dict(snapshot.files)[name].size
        if length > MAX_METADATA_BYTES:
            raise ModelError("unsupported_size", "Safetensors header exceeds the supported size.")
        if length < 2 or length > size - 8:
            raise invalid("Invalid safetensors header length.")
        raw = stream.read(length)
    if len(raw) != length or not raw.startswith(b"{"):
        raise invalid("Invalid safetensors header.")
    header = parse_json(raw)
    metadata = header.pop("__metadata__", {})
    if not isinstance(metadata, dict) or any(not isinstance(v, str) for v in metadata.values()):
        raise invalid("Invalid safetensors metadata.")
    locations: list[TensorLocation] = []
    ranges: list[tuple[int, int]] = []
    for tensor_name, info in header.items():
        if not tensor_name or not isinstance(info, dict):
            raise invalid("Invalid tensor descriptor.")
        dtype = info.get("dtype")
        if not isinstance(dtype, str) or dtype not in DTYPES:
            raise ModelError("unsupported_representation", "Unsupported tensor storage encoding.")
        shape = info.get("shape")
        offsets = info.get("data_offsets")
        if not isinstance(shape, list) or not isinstance(offsets, list) or len(offsets) != 2:
            raise invalid("Invalid tensor shape or offsets.")
        dimensions = tuple(safe_integer(d) for d in shape)
        numel = safe_integer(math.prod(dimensions))
        safe_integer(numel * 4)
        start, end = (safe_integer(v) for v in offsets)
        if end < start or end - start != numel * DTYPES[dtype][1] or end > size - 8 - length:
            raise invalid("Invalid tensor data offsets.")
        ranges.append((start, end))
        descriptor = TensorDescriptor(
            id=hashlib.sha256(tensor_name.encode("utf-8")).hexdigest(),
            name=tensor_name,
            path=tuple(tensor_name.split(".")),
            shape=dimensions,
            rank=len(dimensions),
            numel=numel,
            storage_dtype=dtype,
        )
        locations.append(TensorLocation(descriptor, name, 8 + length + start))
    position = 0
    for start, end in sorted(ranges):
        if start != position:
            raise invalid("Safetensors data contains gaps or overlapping tensors.")
        position = end
    if position != size - 8 - length:
        raise invalid("Safetensors data is not fully described.")
    return tuple(locations)


@dataclass(frozen=True)
class ModelSource:
    """A session-pinnable snapshot; each consumer owns its iterator/file handle."""

    model_id: str
    fingerprint: str
    _snapshot: FileSnapshot
    _locations: tuple[TensorLocation, ...]

    def check_unchanged(self, *, rehash: bool = False) -> None:
        self._snapshot.check()
        if rehash and self._snapshot.fingerprint() != self.fingerprint:
            raise changed()

    def tensors(self) -> tuple[TensorDescriptor, ...]:
        self.check_unchanged()
        return tuple(location.descriptor for location in self._locations)

    @contextmanager
    def local_directory(self) -> Iterator[Path]:
        """Backend-only seam for local tokenizer loading/use; never serialize this path.

        The caller must disable downloads and remote code and only use conventional
        tokenizer assets covered by this snapshot. Reuse this guard on every operation.
        """
        self.check_unchanged()
        try:
            yield self._snapshot.directory
        finally:
            self.check_unchanged()

    def iter_tensor(
        self, tensor_id: str, *, chunk_elements: int = DEFAULT_CHUNK_ELEMENTS
    ) -> Generator[torch.Tensor, None, None]:
        """Yield owned flat CPU float32 chunks, in native C order, with bounded reads.

        The caller selects any later compute device. Advancing after the final chunk
        performs the final snapshot check; consume to exhaustion before publication.
        """
        if type(chunk_elements) is not int or not 0 < chunk_elements <= DEFAULT_CHUNK_ELEMENTS:
            raise ValueError(f"chunk_elements must be between 1 and {DEFAULT_CHUNK_ELEMENTS}")
        self.check_unchanged()
        location = next((loc for loc in self._locations if loc.descriptor.id == tensor_id), None)
        if location is None:
            raise ModelError("tensor_not_found", "Unknown tensor.", 404)
        dtype, width = DTYPES[location.descriptor.storage_dtype]
        with self._snapshot.open(location.file) as stream:
            stream.seek(location.offset)
            remaining = location.descriptor.numel
            while remaining:
                self.check_unchanged()
                count = min(remaining, chunk_elements)
                raw = bytearray(stream.read(count * width))
                if len(raw) != count * width:
                    raise changed()
                # safetensors is little-endian; torch.frombuffer uses native order.
                if sys.byteorder != "little":
                    import array

                    words = array.array("I" if width == 4 else "H", raw)
                    words.byteswap()
                    raw = bytearray(words.tobytes())
                values = torch.frombuffer(raw, dtype=dtype).to(dtype=torch.float32)
                self.check_unchanged()
                yield values
                remaining -= count
        self.check_unchanged()

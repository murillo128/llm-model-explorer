"""Independent scalar format oracles and deterministic, synthetic checkpoint bytes.

These helpers intentionally use struct/Python arithmetic, never production decode
helpers or torch conversion tables. They are not actual-checkpoint acceptance.

Reviewed primary sources:
- GPTQ v1 packing, +1 zero restoration and g_idx lookup: AutoGPTQ qlinear_cuda.py
  https://github.com/AutoGPTQ/AutoGPTQ/blob/9f7d37072917ab3a7545835f23e808294a542153/auto_gptq/nn_modules/qlinear/qlinear_cuda.py
- Selected GPTQModel 4.0 producer's v1 conversion: utils/model.py
  https://github.com/ModelCloud/GPTQModel/blob/40759cdf06c17ea6c57637fa075f8c10547b4a6f/gptqmodel/utils/model.py
- Selected ModelOpt producer's low-first packing and block * global weight scale:
  https://github.com/NVIDIA/Model-Optimizer/blob/82f1d216d1a9022e60b8e1143a77f36c00b885a4/modelopt/torch/quantization/qtensor/nvfp4_tensor.py
  Activation scaling is a separate export method and absent from dequantize().
- Compressed-tensors `pack-quantized` signed offset and dense cross-element I32 packing:
  https://github.com/vllm-project/compressed-tensors/tree/4a696625b7ada2cb857c75f67ce02dc56b381323/src/compressed_tensors/compressors/pack_quantized
- E2M1/E4M3 sign/exponent/mantissa definitions: OCP MX v1.0, Tables 2 and 5
  https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf
  The standard E2M1 sign bit is retained, including -0; ModelOpt's slow Python
  lookup happens to collapse negative zero, so compare its nonzero values only
  when using that path as an external reference.
"""

import json
import math
import struct
from dataclasses import dataclass
from pathlib import Path

from test_quantized_models import config_for

PREFIX = "model.layers.0.self_attn.q_proj"


def f32(value: float) -> float:
    """Round once to canonical IEEE float32, including signed zero."""
    return float(struct.unpack("<f", struct.pack("<f", value))[0])


def e2m1(code: int) -> float:
    """Decode sign/exponent/mantissa directly, independent of a lookup table."""
    exponent, mantissa = (code >> 1) & 3, code & 1
    magnitude = mantissa * 0.5 if exponent == 0 else (1 + mantissa / 2) * 2 ** (exponent - 1)
    return math.copysign(magnitude, -1 if code & 8 else 1)


def e4m3(code: int) -> float:
    exponent, mantissa = (code >> 3) & 15, code & 7
    if exponent == 15 and mantissa == 7:
        return math.nan
    magnitude = mantissa * 2**-9 if exponent == 0 else (1 + mantissa / 8) * 2 ** (exponent - 7)
    return math.copysign(magnitude, -1 if code & 128 else 1)


@dataclass(frozen=True)
class Stored:
    name: str
    dtype: str
    shape: tuple[int, ...]
    data: bytes


@dataclass(frozen=True)
class PackedFixture:
    encoding: str
    shape: tuple[int, int]
    storage: tuple[Stored, ...]
    config_kind: str | None = None

    @property
    def name(self) -> str:
        name = self.storage[0].name
        prefix = (
            name.removesuffix(".weight_packed")
            if self.encoding == "compressed-tensors-w4a16-int4"
            else name.rsplit(".", 1)[0]
        )
        return prefix + ".weight"

    def expected(self) -> bytes:
        """Walk output rows and input columns, decoding one scalar at a time."""
        fields = {entry.name.rsplit(".", 1)[1]: entry.data for entry in self.storage}
        output, inputs = self.shape
        values = bytearray()
        for row in range(output):
            for column in range(inputs):
                if self.encoding == "gptq-int4":
                    group = struct.unpack_from("<i", fields["g_idx"], column * 4)[0]
                    word = struct.unpack_from(
                        "<I", fields["qweight"], ((column // 8) * output + row) * 4
                    )[0]
                    weight = (word >> (4 * (column % 8))) & 15
                    zero_word = struct.unpack_from(
                        "<I", fields["qzeros"], (group * (output // 8) + row // 8) * 4
                    )[0]
                    zero = ((zero_word >> (4 * (row % 8))) & 15) + 1
                    scale = struct.unpack_from("<e", fields["scales"], (group * output + row) * 2)[
                        0
                    ]
                    value = (weight - zero) * scale
                elif self.encoding == "nvfp4":
                    packed = fields["weight"][row * (inputs // 2) + column // 2]
                    code = (packed >> (4 * (column % 2))) & 15
                    scale = e4m3(fields["weight_scale"][row * (inputs // 16) + column // 16])
                    global_scale = struct.unpack("<f", fields["weight_scale_2"])[0]
                    value = e2m1(code) * f32(scale * global_scale)
                else:
                    word = struct.unpack_from(
                        "<I", fields["weight_packed"], (row * (inputs // 8) + column // 8) * 4
                    )[0]
                    signed = ((word >> (4 * (column % 8))) & 15) - 8
                    scale_bits = struct.unpack_from(
                        "<H", fields["weight_scale"], (row * (inputs // 32) + column // 32) * 2
                    )[0]
                    scale = struct.unpack("<f", struct.pack("<I", scale_bits << 16))[0]
                    value = signed * scale
                values.extend(struct.pack("<f", value))
        return bytes(values)

    def write(self, root: Path, *, split: bool = False, name: str = "numeric") -> Path:
        directory = root / name
        directory.mkdir(parents=True)
        kind = self.config_kind or ("JunHowie" if self.encoding == "gptq-int4" else "AxionML")
        (directory / "config.json").write_text(json.dumps(config_for(kind)))
        groups = [(entry,) for entry in self.storage] if split else [self.storage]
        mapping: dict[str, str] = {}
        for number, entries in enumerate(groups):
            file = f"part-{number}.safetensors"
            header: dict[str, object] = {}
            payload = bytearray()
            for entry in entries:
                begin = len(payload)
                payload.extend(entry.data)
                header[entry.name] = {
                    "dtype": entry.dtype,
                    "shape": entry.shape,
                    "data_offsets": [begin, len(payload)],
                }
                mapping[entry.name] = file
            raw_header = json.dumps(header, separators=(",", ":")).encode()
            (directory / file).write_bytes(
                struct.pack("<Q", len(raw_header)) + raw_header + payload
            )
        (directory / "model.safetensors.index.json").write_text(json.dumps({"weight_map": mapping}))
        return directory


def gptq_fixture(*, inputs: int = 256, output: int = 16) -> PackedFixture:
    """All eight word positions, two groups and nontrivial repeated g_idx."""
    groups = inputs // 128
    packed = bytearray()
    for word in range(inputs // 8):
        for row in range(output):
            value = sum(
                ((3 * (word * 8 + nibble) + 5 * row + word) % 16) << (4 * nibble)
                for nibble in range(8)
            )
            packed.extend(struct.pack("<I", value))
    zeros = bytearray()
    for group in range(groups):
        for word in range(output // 8):
            value = sum(
                ((group * 7 + word * 8 + nibble) % 16) << (4 * nibble) for nibble in range(8)
            )
            zeros.extend(struct.pack("<I", value))
    scale_values = [0.0, -0.0, 2**-24, 2**-14, 0.1, 1.25, 65504.0, 0.3333]
    scales = b"".join(
        struct.pack("<e", scale_values[(group * 3 + row) % len(scale_values)])
        for group in range(groups)
        for row in range(output)
    )
    indices = b"".join(
        struct.pack("<i", (column // 5 + column // 128) % groups) for column in range(inputs)
    )
    return PackedFixture(
        "gptq-int4",
        (output, inputs),
        (
            Stored(PREFIX + ".qweight", "I32", (inputs // 8, output), bytes(packed)),
            Stored(PREFIX + ".qzeros", "I32", (groups, output // 8), bytes(zeros)),
            Stored(PREFIX + ".scales", "F16", (groups, output), scales),
            Stored(PREFIX + ".g_idx", "I32", (inputs,), indices),
        ),
    )


def nvfp4_fixture(*, inputs: int = 64, output: int = 4, input_scale: float = 31.0) -> PackedFixture:
    """All E2M1 codes in both positions and E4M3 normal/subnormal endpoints."""
    packed = bytes(
        ((column + row * 3 + column // 16) % 16)
        | (((column + 1 + row * 3 + (column + 1) // 16) % 16) << 4)
        for row in range(output)
        for column in range(0, inputs, 2)
    )
    scale_codes = [
        0,
        1,
        7,
        8,
        0x10,
        0x30,
        0x38,
        0x3C,
        0x40,
        0x48,
        0x50,
        0x60,
        0x70,
        0x78,
        0x7D,
        0x7E,
    ]
    scales = bytes(scale_codes[index % len(scale_codes)] for index in range(output * inputs // 16))
    prefix = PREFIX.replace("model.layers", "model.language_model.layers")
    return PackedFixture(
        "nvfp4",
        (output, inputs),
        (
            Stored(prefix + ".weight", "U8", (output, inputs // 2), packed),
            Stored(prefix + ".weight_scale", "F8_E4M3", (output, inputs // 16), scales),
            Stored(prefix + ".weight_scale_2", "F32", (), struct.pack("<f", 0.1234567)),
            Stored(prefix + ".input_scale", "F32", (), struct.pack("<f", input_scale)),
        ),
    )


def compressed_tensors_fixture(
    *, kimi: bool = False, inputs: int = 64, output: int = 4
) -> PackedFixture:
    """Small CT W4A16 fixture with both 32-column groups and all signed codes."""
    if inputs <= 0 or inputs % 32 or output <= 0:
        raise ValueError("compressed-tensors fixtures require positive 32-aligned input width")
    kind = "Kimi" if kimi else "GLM"
    prefix = (
        "model.layers.1.block_sparse_moe.experts.0.w1"
        if kimi
        else "model.layers.1.mlp.experts.0.down_proj"
    )

    packed = bytearray()
    for row in range(output):
        for word_index in range(inputs // 8):
            word = 0
            for lane in range(8):
                column = word_index * 8 + lane
                signed = ((row * 7 + column * 5 + column // 32 * 3) % 16) - 8
                word |= (signed + 8) << (4 * lane)
            packed.extend(struct.pack("<I", word))

    scale_values = (0.5, 1.25, 0.03125, 2.0)
    scales = bytearray()
    for row in range(output):
        for group in range(inputs // 32):
            value = scale_values[(row * 3 + group) % len(scale_values)]
            bits = struct.unpack("<I", struct.pack("<f", value))[0]
            scales.extend(struct.pack("<H", bits >> 16))

    return PackedFixture(
        "compressed-tensors-w4a16-int4",
        (output, inputs),
        (
            Stored(prefix + ".weight_packed", "I32", (output, inputs // 8), bytes(packed)),
            Stored(prefix + ".weight_scale", "BF16", (output, inputs // 32), bytes(scales)),
            Stored(prefix + ".weight_shape", "I64", (2,), struct.pack("<2q", output, inputs)),
        ),
        config_kind=kind,
    )

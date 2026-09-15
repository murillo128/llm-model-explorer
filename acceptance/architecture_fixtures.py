"""Reduced local checkpoints from independently reviewed component-test metadata.

Synthetic bytes prove integration, never support for an installed reference model.
No application description is used to generate the expected storage or values.
"""

import argparse
import json
import math
import struct
import sys
from pathlib import Path

FIXTURES = Path(__file__).resolve().parents[1] / "backend/tests/fixtures"
WIDTHS = {"F32": 4, "F16": 2, "BF16": 2, "I32": 4, "U8": 1, "F8_E4M3": 1}


def value(index):
    return (index % 29 - 14) / 8


def packed_value(family, row, column, inputs):
    """Scalar reference for our synthetic packed bytes, never production decoding.

    GPTQ v1 uses mask-then-+1 zeros and the stored g_idx. ModelOpt NVFP4 uses
    low-first E2M1 and block E4M3 times global weight scale, never input_scale.
    Exact pinned upstream provenance is in backend/tests/quantized_oracles.py.
    """
    if family == "qwen3":
        group = (column // 3 + column // 128) % (inputs // 128)
        integer = (3 * column + 5 * row) % 16
        zero = (group + row) % 14 + 1
        scale = (group + row % 7 + 1) / 8
        return (integer - zero) * scale
    assert family == "qwen35"
    code = (column + row * 3) % 16
    exponent, mantissa = (code >> 1) & 3, code & 1
    magnitude = mantissa / 2 if exponent == 0 else (1 + mantissa / 2) * 2 ** (exponent - 1)
    scalar = math.copysign(magnitude, -1 if code & 8 else 1)
    scale = (0.5, 1.0, 2.0, 3.0)[(row + column // 16) % 4]
    return scalar * scale * 0.5


def packed_storage(name, dtype, shape):
    """Produce deterministic physical words with nonzero, asymmetric values."""
    leaf = name.rsplit(".", 1)[-1]
    if leaf == "qweight" and dtype == "I32":
        return b"".join(
            struct.pack(
                "<I",
                sum(
                    ((3 * (word * 8 + nibble) + 5 * row) % 16) << (4 * nibble)
                    for nibble in range(8)
                ),
            )
            for word in range(shape[0])
            for row in range(shape[1])
        )
    if leaf == "qzeros" and dtype == "I32":
        return b"".join(
            struct.pack(
                "<I", sum(((group + word * 8 + nibble) % 14) << (4 * nibble) for nibble in range(8))
            )
            for group in range(shape[0])
            for word in range(shape[1])
        )
    if leaf == "g_idx" and dtype == "I32":
        return b"".join(
            struct.pack("<i", (column // 3 + column // 128) % (shape[0] // 128))
            for column in range(shape[0])
        )
    if leaf == "scales" and dtype == "F16":
        return b"".join(
            struct.pack("<e", (group + row % 7 + 1) / 8)
            for group in range(shape[0])
            for row in range(shape[1])
        )
    if leaf == "weight" and dtype == "U8":
        return bytes(
            ((column * 2 + row * 3) % 16) | (((column * 2 + 1 + row * 3) % 16) << 4)
            for row in range(shape[0])
            for column in range(shape[1])
        )
    if leaf == "weight_scale" and dtype == "F8_E4M3":
        return bytes(
            (0x30, 0x38, 0x40, 0x44)[(row + block) % 4]
            for row in range(shape[0])
            for block in range(shape[1])
        )
    if leaf in {"weight_scale_2", "input_scale"} and dtype == "F32":
        return struct.pack("<f", 0.5 if leaf == "weight_scale_2" else 19.0)
    return None


def write_checkpoint(directory, config, storage):
    directory.mkdir(parents=True)
    (directory / "config.json").write_text(json.dumps(config))
    header, payload = {}, bytearray()
    for name, descriptor in storage.items():
        dtype, shape = descriptor["dtype"], descriptor["shape"]
        count = math.prod(shape)
        start = len(payload)
        encoded = packed_storage(name, dtype, shape)
        if encoded is not None:
            assert len(encoded) == count * WIDTHS[dtype]
            payload.extend(encoded)
        elif dtype in {"F32", "F16", "BF16"}:
            for i in range(count):
                scalar = struct.pack("<f", value(i))
                payload.extend(
                    scalar
                    if dtype == "F32"
                    else scalar[2:]
                    if dtype == "BF16"
                    else struct.pack("<e", value(i))
                )
        else:
            payload.extend(bytes(count * WIDTHS[dtype]))
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [start, len(payload)]}
    raw = json.dumps(header, separators=(",", ":")).encode()
    raw += b" " * (-len(raw) % 8)
    (directory / "model.safetensors").write_bytes(struct.pack("<Q", len(raw)) + raw + payload)
    (directory / "modeling_custom.py").write_text('raise AssertionError("checkpoint executed")\n')


def generate(root):
    # Reuse the independently authored dense fixture geometry, not production rules.
    sys.path.insert(0, str(FIXTURES.parent))
    try:
        from dense_fixtures import small_config, small_storage

        for family in ("smollm2", "qwen3"):
            qwen = family == "qwen3"
            write_checkpoint(
                root / family,
                small_config(qwen),
                {
                    name: {"dtype": "F16" if dtype == "F32" else dtype, "shape": shape}
                    for name, dtype, shape in small_storage(qwen)
                },
            )
    finally:
        sys.path.pop(0)
    for family in ("qwen35", "vjepa2"):
        fixture = json.loads((FIXTURES / f"{family}-tiny.json").read_text())
        write_checkpoint(root / family, fixture["configuration"], fixture["storage"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    generate(parser.parse_args().root)

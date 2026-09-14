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


def write_checkpoint(directory, config, storage):
    directory.mkdir(parents=True)
    (directory / "config.json").write_text(json.dumps(config))
    header, payload = {}, bytearray()
    for name, descriptor in storage.items():
        dtype, shape = descriptor["dtype"], descriptor["shape"]
        count = math.prod(shape)
        start = len(payload)
        if dtype in {"F32", "F16", "BF16"}:
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

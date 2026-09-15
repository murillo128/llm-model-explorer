"""Offline bounded comparison of reviewed local checkpoints with scalar references.

The independent reference uses Python integer/struct arithmetic derived from the
pinned upstream sources below. Source files are hash-checked as provenance only;
neither downloaded code nor code from a checkpoint is executed. PyTorch is used
only by the production decoder being compared. Complete representative layers
are compared in bounded chunks, never retained as a decompressed layer.

From the repository root, with the locked backend interpreter:
  PYTHONPATH=backend/src python backend/scripts/check_quantized_reference.py \
    --references /path/to/references.json --sources /path/to/reviewed-sources \
    --output backend/evidence/quantized-reference-comparison.json

The external manifest uses acceptance/architecture.md's family/repository/
revision/directory entries. Missing capabilities are SKIP; supplied mismatched
checkpoints or reference sources fail. The sources directory must contain the
named files below, obtained separately from their pinned URLs.
"""

import argparse
import hashlib
import inspect
import json
import math
import platform
import struct
from contextlib import ExitStack
from importlib.metadata import version
from pathlib import Path

from llm_model_explorer.model_files import FileSnapshot
from llm_model_explorer.quantized_decoding import decode_range
from llm_model_explorer.tensor_source import parse_header

REFERENCES = {
    "gptq": {
        "file": "gptq-cuda-reference.py",
        "url": "https://github.com/AutoGPTQ/AutoGPTQ/blob/"
        "9f7d37072917ab3a7545835f23e808294a542153/auto_gptq/nn_modules/qlinear/qlinear_cuda.py",
        "sha256": "5a6a483784adc7d9ee55e62f6dc4fdaf3a642f56796d2a6b8afea0b2364d751d",
        "lines": "258-304",
    },
    "nvfp4": {
        "file": "modelopt-export-reference.py",
        "url": "https://github.com/NVIDIA/Model-Optimizer/blob/"
        "82f1d216d1a9022e60b8e1143a77f36c00b885a4/"
        "modelopt/torch/quantization/qtensor/nvfp4_tensor.py",
        "sha256": "bd2073b89d45965011aa7cc3b80b7aae1472b9fa10b6c72dd50f648eb93e509e",
        "lines": "314-377",
    },
}
CHECKPOINTS = {
    "qwen3": {
        "repository": "JunHowie/Qwen3-0.6B-GPTQ-Int4",
        "revision": "b9d87006067b0c0c2dea836370d6288e14f112ab",
        "encoding": "gptq-int4",
        "reference": "gptq",
        "complete_parameter": "model.layers.0.self_attn.k_proj",
        "parameters": [
            "model.layers.0.self_attn.k_proj",
            "model.layers.0.self_attn.q_proj",
            "model.layers.13.mlp.down_proj",
            "model.layers.27.self_attn.o_proj",
        ],
    },
    "qwen35": {
        "repository": "AxionML/Qwen3.5-0.8B-NVFP4",
        "revision": "2ac1e750cda67cc8538d731f6216f77b9c3a6f72",
        "encoding": "nvfp4",
        "reference": "nvfp4",
        "complete_parameter": "model.language_model.layers.0.linear_attn.in_proj_a",
        "parameters": [
            "model.language_model.layers.0.linear_attn.in_proj_a",
            "model.language_model.layers.3.self_attn.q_proj",
            "model.language_model.layers.11.mlp.down_proj",
            "model.language_model.layers.23.self_attn.q_proj",
        ],
    },
}


def f32(value):
    return struct.unpack("<f", struct.pack("<f", value))[0]


def floating(code, exponent_bits, mantissa_bits, bias):
    """IEEE-like finite formats, preserving the encoded sign of zero."""
    exponent = (code >> mantissa_bits) & ((1 << exponent_bits) - 1)
    mantissa = code & ((1 << mantissa_bits) - 1)
    if exponent_bits == 4 and exponent == 15 and mantissa == 7:
        return math.nan
    magnitude = (
        mantissa * 2 ** (1 - bias - mantissa_bits)
        if exponent == 0
        else (1 + mantissa / 2**mantissa_bits) * 2 ** (exponent - bias)
    )
    return math.copysign(magnitude, -1 if code >> (exponent_bits + mantissa_bits) else 1)


def scalar_reference(snapshot, storage, encoding, shape, start, count):
    """Seek only requested scalar storage; cache metadata within this one range."""
    output, inputs = shape
    fields = {tensor.name.rsplit(".", 1)[1]: tensor for tensor in storage}
    formats = {"I32": "i", "F16": "e", "F32": "f", "U8": "B", "F8_E4M3": "B"}
    values, upstream_values = bytearray(), bytearray()
    observations = {"codes": set(), "groups": set(), "bytes_read": 0}
    with ExitStack() as stack:
        streams = {name: stack.enter_context(snapshot.open(name)) for name in snapshot.shards}
        cache = {}

        def read(name, index):
            key = name, index
            if key not in cache:
                tensor = fields[name]
                assert 0 <= index < tensor.numel
                fmt = "<" + formats[tensor.dtype]
                width = struct.calcsize(fmt)
                stream = streams[tensor.file]
                stream.seek(tensor.offset + index * width)
                raw = stream.read(width)
                assert len(raw) == width
                cache[key] = struct.unpack(fmt, raw)[0]
                observations["bytes_read"] += width
            return cache[key]

        if encoding == "nvfp4":
            observations["input_scale"] = read("input_scale", 0)
            observations["weight_scale_2"] = read("weight_scale_2", 0)
        for index in range(start, start + count):
            row, column = divmod(index, inputs)
            if encoding == "gptq-int4":
                group = read("g_idx", column)
                packed = read("qweight", (column // 8) * output + row)
                code = (packed >> (4 * (column % 8))) & 15
                zero_word = read("qzeros", group * (output // 8) + row // 8)
                zero = ((zero_word >> (4 * (row % 8))) & 15) + 1
                value = f32((code - zero) * read("scales", group * output + row))
                # Upstream inference commonly rounds the reconstructed weight to
                # its matmul dtype. The API exposes its logical float32 value.
                upstream = struct.unpack("<e", struct.pack("<e", value))[0]
            else:
                packed = read("weight", row * (inputs // 2) + column // 2)
                code = (packed >> (4 * (column % 2))) & 15
                group = column // 16
                block = floating(read("weight_scale", row * (inputs // 16) + group), 4, 3, 7)
                scale = f32(block * observations["weight_scale_2"])
                value = f32(floating(code, 2, 1, 1) * scale)
                # The pinned ModelOpt slow LUT uses +0 for both zero codes.
                upstream = f32((0.0 if code == 8 else floating(code, 2, 1, 1)) * scale)
            assert math.isfinite(value), "Nonfinite reviewed reference sample"
            observations["codes"].add(code)
            observations["groups"].add(group)
            values.extend(struct.pack("<f", value))
            upstream_values.extend(struct.pack("<f", upstream))
    return bytes(values), bytes(upstream_values), observations


def compare(snapshot, physical, config):
    reports = []
    encoding = config["encoding"]
    suffixes = (
        ("qweight", "qzeros", "scales", "g_idx")
        if encoding == "gptq-int4"
        else ("weight", "weight_scale", "weight_scale_2", "input_scale")
    )
    for prefix in config["parameters"]:
        storage = tuple(physical[prefix + "." + suffix] for suffix in suffixes)
        packed = storage[0]
        output, inputs = (
            (packed.shape[1], packed.shape[0] * 8)
            if encoding == "gptq-int4"
            else (packed.shape[0], packed.shape[1] * 2)
        )
        ranges = [
            ("first-eight-rows", 0, 8 * inputs),
            ("row-boundary", inputs - 7, 37),
            ("middle-row-multiple-groups", output // 2 * inputs + 117, 513),
            ("last-row", (output - 1) * inputs, inputs),
        ]
        complete = prefix == config["complete_parameter"]
        if complete:
            ranges = [("complete-projection", 0, output * inputs)]
        digest = hashlib.sha256()
        codes, groups = set(), set()
        compared = read_bytes = upstream_differences = 0
        upstream_max_error = 0.0
        chunks = (
            (chunk_start, min(65_536, start + count - chunk_start))
            for _, start, count in ranges
            for chunk_start in range(start, start + count, 65_536)
        )
        for start, count in chunks:
            expected, upstream, observations = scalar_reference(
                snapshot, storage, encoding, (output, inputs), start, count
            )
            actual = decode_range(snapshot, storage, encoding, start, count)
            actual_bytes = actual.numpy().astype("<f4", copy=False).tobytes()
            assert actual_bytes == expected, f"Float32 bit mismatch: {prefix} at {start}"
            for offset in range(0, len(expected), 4):
                a, b = expected[offset : offset + 4], upstream[offset : offset + 4]
                upstream_differences += a != b
                upstream_max_error = max(
                    upstream_max_error, abs(struct.unpack("<f", a)[0] - struct.unpack("<f", b)[0])
                )
            digest.update(expected)
            codes.update(observations["codes"])
            groups.update(observations["groups"])
            read_bytes += observations["bytes_read"]
            compared += count
        report = {
            "parameter": prefix + ".weight",
            "shape": [output, inputs],
            "complete_parameter": complete,
            "ranges": [
                {"kind": kind, "start": start, "count": count} for kind, start, count in ranges
            ],
            "compared_float32_values": compared,
            "canonical_bit_mismatches": 0,
            "canonical_max_absolute_error": 0.0,
            "canonical_sample_sha256": digest.hexdigest(),
            "observed_nibble_codes_hex": "".join(format(code, "x") for code in sorted(codes)),
            "observed_groups": {
                "count": len(groups),
                "minimum": min(groups),
                "maximum": max(groups),
            },
            "oracle_storage_bytes_read": read_bytes,
            "upstream_inference_or_lut_bit_differences": upstream_differences,
            "upstream_inference_or_lut_max_absolute_error": upstream_max_error,
        }
        if encoding == "nvfp4":
            report["input_scale_excluded"] = observations["input_scale"]
            report["weight_scale_2"] = observations["weight_scale_2"]
        reports.append(report)
    return reports


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--references", type=Path)
    parser.add_argument("--sources", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    selections = json.loads(args.references.read_text()) if args.references else {}
    report = {
        "method": "Independent bounded Python scalar reference; production decode_range seam",
        "decoder_source_sha256": hashlib.sha256(
            Path(inspect.getfile(decode_range)).read_bytes()
        ).hexdigest(),
        "environment": {
            "python": platform.python_version(),
            "torch": version("torch"),
            "platform": platform.platform(),
        },
        "reproduce": "PYTHONPATH=backend/src python backend/scripts/check_quantized_reference.py "
        '--references "$LMEX_ARCHITECTURE_REFERENCES" '
        '--sources "$LMEX_QUANTIZED_REFERENCE_SOURCES" '
        "--output backend/evidence/quantized-reference-comparison.json",
        "reference_sources": REFERENCES,
        "scope": "One complete projection per format plus selected ranges across other layers; "
        "no inference or whole-checkpoint equivalence claim",
        "maximum_decode_chunk_elements": 65_536,
        "precision_notes": [
            "GPTQ restores masked qzeros + 1 without wrapping, uses stored g_idx, and preserves "
            "logical float32 before an upstream inference float16 rounding step.",
            "NVFP4 rounds block_scale * weight_scale_2 to float32 before multiplying E2M1. "
            "input_scale is activation metadata and is excluded.",
            "Canonical E2M1 retains negative zero; ModelOpt slow Python LUT collapses its sign. "
            "Reported LUT differences are distinct from canonical bit comparisons.",
        ],
        "checkpoints": {},
    }
    for family, config in CHECKPOINTS.items():
        selection = selections.get(family)
        reference = REFERENCES[config["reference"]]
        if not selection or not args.sources or not (args.sources / reference["file"]).is_file():
            report["checkpoints"][family] = {
                "status": "SKIP",
                "reason": "Reviewed local checkpoint or pinned reference source not supplied",
            }
            continue
        assert (
            hashlib.sha256((args.sources / reference["file"]).read_bytes()).hexdigest()
            == reference["sha256"]
        )
        assert all(selection[key] == config[key] for key in ("repository", "revision"))
        directory = Path(selection["directory"]).resolve(strict=True)
        for name in ("config.json", "model.safetensors"):
            metadata = directory / ".cache/huggingface/download" / (name + ".metadata")
            if metadata.exists():
                assert metadata.read_text().splitlines()[0] == config["revision"]
        snapshot = FileSnapshot.capture(directory.parent, directory, ("model.safetensors",))
        physical = {tensor.name: tensor for tensor in parse_header(snapshot, "model.safetensors")}
        fixtures = Path(__file__).resolve().parents[1] / "tests/fixtures"
        if family == "qwen3":
            reviewed = json.loads((fixtures / "dense-reference-metadata.json").read_text())[family]
            expected_config = reviewed["config"]
            expected_storage = dict(reviewed["global_storage"])
            for layer in range(expected_config["num_hidden_layers"]):
                expected_storage.update(
                    {
                        f"model.layers.{layer}.{name}": item
                        for name, item in reviewed["layer_storage"].items()
                    }
                )
            expected_storage = {
                name: {"dtype": dtype, "shape": shape}
                for name, (dtype, shape) in expected_storage.items()
            }
        else:
            reviewed = json.loads((fixtures / "qwen35-reference.json").read_text())
            expected_config, expected_storage = reviewed["configuration"], reviewed["storage"]
        with snapshot.open("config.json") as stream:
            assert json.load(stream) == expected_config, (
                "Configuration differs from reviewed checkpoint"
            )
        assert {
            name: {"dtype": tensor.dtype, "shape": list(tensor.shape)}
            for name, tensor in physical.items()
        } == expected_storage, "Storage differs from reviewed checkpoint"
        results = compare(snapshot, physical, config)
        report["checkpoints"][family] = {
            "status": "PASS",
            "repository": config["repository"],
            "revision": config["revision"],
            "revision_source": "Operator manifest; present Hugging Face download metadata checked",
            "fingerprint": snapshot.fingerprint(),
            "physical_tensor_count": len(physical),
            "weight_file_bytes": dict(snapshot.files)["model.safetensors"].size,
            "compared_float32_values": sum(layer["compared_float32_values"] for layer in results),
            "canonical_bit_mismatches": 0,
            "layers": results,
        }
    raw = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.write_text(raw)
    else:
        print(raw, end="")


if __name__ == "__main__":
    main()

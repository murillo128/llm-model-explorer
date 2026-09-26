"""Compare a bounded local NF4 tensor range with bitsandbytes reference math.

This optional acceptance script needs one local saved bitsandbytes checkpoint
group and an environment that has bitsandbytes installed. It never downloads
models. The independent oracle uses bitsandbytes' codebook and nested
blockwise dequantizer plus the pinned upstream high-first NF4 kernel mapping.

From the repository root:
  PYTHONPATH=backend/src python backend/scripts/check_bnb_nf4_reference.py \
    --model-root /path/to/local/model-root \
    --model-id 'HuggingFaceTB/SmolLM2-135M@bnb-nf4-dq' \
    --output backend/evidence/bnb-nf4-smollm2-reference.json
"""

import argparse
import hashlib
import json
import platform
from importlib.metadata import version
from pathlib import Path

import bitsandbytes as bnb
import bitsandbytes.functional as bnb_functional
import torch
import transformers
from safetensors import safe_open

from llm_model_explorer.models import ModelCatalogue
from llm_model_explorer.quantized_decoding import decode_range

REFERENCE_REPOSITORY = "unsloth/SmolLM2-135M-bnb-4bit"
REFERENCE_REVISION = "738fd459cdfaf82f75846cf075738987ed6eb2b8"
REFERENCE_SOURCE = (
    "https://github.com/bitsandbytes-foundation/bitsandbytes/blob/0.50.2/csrc/kernels.cu#L517-L522"
)
REFERENCE_MODEL_ID = "HuggingFaceTB/SmolLM2-135M@bnb-nf4-dq"
REFERENCE_TRANSFORMERS_VERSION = "4.46.1"
TENSOR_NAME = "model.layers.0.self_attn.k_proj.weight"


def tensor_bytes(tensor: torch.Tensor) -> bytes:
    return tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()


def compare(model_root: Path, model_id: str, start: int, count: int) -> dict[str, object]:
    if model_id != REFERENCE_MODEL_ID:
        raise ValueError(f"Expected the pinned reference model ID {REFERENCE_MODEL_ID!r}.")
    if bnb.__version__ != "0.50.2":
        raise ValueError("This comparison is pinned to bitsandbytes 0.50.2.")
    source = ModelCatalogue(model_root).pin(model_id)
    if not any(item.storage_format == "bnb-nf4-dq" for item in source.tensors()):
        raise ValueError("Selected model is not an admitted bitsandbytes NF4 checkpoint.")
    physical = source.physical_tensors()
    by_name = {item.name: item for item in physical}
    group = tuple(
        by_name[TENSOR_NAME + suffix]
        for suffix in (
            "",
            ".absmax",
            ".quant_map",
            ".nested_absmax",
            ".nested_quant_map",
            ".quant_state.bitsandbytes__nf4",
        )
    )
    logical = next(item for item in source.tensors() if item.name == TENSOR_NAME)
    if start < 0 or count <= 0 or start + count > logical.numel:
        raise ValueError("Requested range is outside the logical tensor.")

    model_directory = source._snapshot.directory
    checkpoint_config = json.loads((model_directory / "config.json").read_text())
    if checkpoint_config.get("transformers_version") != REFERENCE_TRANSFORMERS_VERSION:
        raise ValueError("Checkpoint configuration disagrees with the pinned reference.")

    def load_record(name: str) -> torch.Tensor:
        item = by_name[name]
        with safe_open(model_directory / item.file, framework="pt", device="cpu") as f:
            return f.get_tensor(name)

    packed = load_record(TENSOR_NAME).reshape(-1)
    state_bytes = bytes(load_record(TENSOR_NAME + ".quant_state.bitsandbytes__nf4").tolist())
    state = json.loads(state_bytes)
    absmax_codes = load_record(TENSOR_NAME + ".absmax")
    quant_map = load_record(TENSOR_NAME + ".quant_map")
    nested_absmax = load_record(TENSOR_NAME + ".nested_absmax")
    nested_map = load_record(TENSOR_NAME + ".nested_quant_map")
    if tuple(state["shape"]) != logical.shape:
        raise ValueError("Serialized NF4 shape disagrees with the logical inventory.")

    nested_state = bnb_functional.QuantState(
        absmax=nested_absmax,
        blocksize=state["nested_blocksize"],
        code=nested_map,
        dtype=getattr(torch, state["nested_dtype"]),
    )
    scales = bnb_functional.dequantize_blockwise(absmax_codes, nested_state)
    scales = scales + state["nested_offset"]
    codebook = bnb_functional.get_4bit_type("nf4", device=torch.device("cpu"))
    if not torch.equal(quant_map, codebook):
        raise ValueError("Recorded NF4 codebook disagrees with bitsandbytes.")
    logical_indices = torch.arange(start, start + count, dtype=torch.int64)
    packed_bytes = packed[logical_indices // 2].to(torch.int64)
    codes = torch.where(
        logical_indices.remainder(2) == 0,
        (packed_bytes >> 4) & 15,
        packed_bytes & 15,
    )
    oracle = codebook[codes] * scales[logical_indices // state["blocksize"]]
    decoded = decode_range(source._snapshot, physical, "bnb-nf4-dq", start, count)
    difference = (decoded - oracle).abs()

    records = []
    for item in group:
        with safe_open(model_directory / item.file, framework="pt", device="cpu") as f:
            data = tensor_bytes(f.get_tensor(item.name))
        records.append(
            {
                "name": item.name,
                "dtype": item.dtype,
                "shape": list(item.shape),
                "bytes": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
            }
        )
    expected_hashes = {
        TENSOR_NAME: "1bd072552f674c83ff84b4990cb10013ec3d9a42b8daf00fc9f9640dd3dc4d6e",
        TENSOR_NAME + ".absmax": "b9951dc1ce3e1cb658da5744bf0ed9908ff60680021739e67a8f33f3de276765",
        TENSOR_NAME
        + ".quant_map": "8501941daa1b8a90ad1bbfeb632e5101b5dddbc4bb52d6e55abcfd777e60c06a",
        TENSOR_NAME
        + ".nested_absmax": "f4b034d1053af346d1b1584d998cef388f7f856ea8d65de510d5c245e720df14",
        TENSOR_NAME
        + ".nested_quant_map": "e732639a65f497b4ad684bb166a4467708255edd5207757de8b8f0c7e1fda89c",
        TENSOR_NAME + ".quant_state.bitsandbytes__nf4": (
            "175e42fd82e2a24fa4760e989cc74f135049b59219f65d4db24d5b0973857a80"
        ),
    }
    if {item["name"]: item["sha256"] for item in records} != expected_hashes:
        raise ValueError("Physical records do not match the pinned reference group hashes.")

    return {
        "reference": {"repository": REFERENCE_REPOSITORY, "revision": REFERENCE_REVISION},
        "tensor": {"name": TENSOR_NAME, "shape": list(logical.shape)},
        "range": {"start": start, "count": count},
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "bitsandbytes": bnb.__version__,
            "transformers": transformers.__version__,
            "safetensors": version("safetensors"),
        },
        "checkpoint_transformers_version": checkpoint_config.get("transformers_version"),
        "oracle": {
            "description": (
                "bitsandbytes.functional.get_4bit_type('nf4') and "
                "dequantize_blockwise for nested scales, using the pinned "
                "high-first NF4 nibble mapping"
            ),
            "limitation": (
                "The installed CPU wheel has no CPU kernel for functional.dequantize_4bit; "
                "the comparison composes its CPU codebook and nested-scale APIs with the "
                "pinned kernel's documented high-first mapping."
            ),
            "nibble_mapping_source": REFERENCE_SOURCE,
            "maximum_absolute_error": float(difference.max()),
            "float32_bit_mismatches": int(
                torch.count_nonzero(decoded.view(torch.int32) != oracle.view(torch.int32))
            ),
            "decoded_sha256": hashlib.sha256(tensor_bytes(decoded)).hexdigest(),
            "oracle_sha256": hashlib.sha256(tensor_bytes(oracle)).hexdigest(),
        },
        "physical_group": records,
        "weight_bytes_committed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--count", type=int, default=4096)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = compare(args.model_root, args.model_id, args.start, args.count)
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(rendered, end="")
    else:
        args.output.write_text(rendered)


if __name__ == "__main__":
    main()

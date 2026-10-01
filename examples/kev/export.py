"""Explicit offline, unmerged Kev inspection export; never imported by the server."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import re
import shutil
import tempfile
from collections import OrderedDict
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch
from llm_model_explorer.architecture_analysis import AnalysisInput
from llm_model_explorer.architecture_analysis.kev import PRODUCER as KEV_PRODUCER
from llm_model_explorer.architecture_analysis.kev_config import (
    BASE_REPOSITORY,
    SOURCE_REVISION,
    factor_bindings,
    head_configuration,
)
from llm_model_explorer.architecture_analysis.qwen35 import (
    PREFIX,
    configuration,
    parameter_shapes,
)
from llm_model_explorer.architecture_analysis.qwen35 import (
    PRODUCER as QWEN_PRODUCER,
)
from llm_model_explorer.model_files import FileSnapshot, read_json
from llm_model_explorer.models import ModelCatalogue
from llm_model_explorer.tensor_source import parse_header
from llm_model_explorer.validate_architecture_cli import validate_directory
from safetensors import safe_open
from safetensors.torch import save_file

KEV_REVISION = "9a45d25eb2ab761841196625383fa1dff0e56c1e"
BASE_REVISION = "dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68"
KEV_REPOSITORY = "jaredpalmer/kev-0.8b"
EXPORT_IDENTITY = "jaredpalmer/kev-0.8b-inspection"
PRODUCER = "kev-inspection-export-2"
SOURCE_FILES = {
    "kev/model.py": "2634ffe7747d69473fb6596942c248e5df2586df3610bd1adb47e7e9acd99f96",
    "kev/checkpoint.py": "f3edb4d159c0aa12d7006fc599fe6ebc3f2c288b3b69890c289f7dac451dd03d",
    "LICENSE": "6b08bb37982c233aa12bcbdf19106da12f3f4fcf800773ea42e28ebeddd34fb8",
}
NATIVE = {torch.float32, torch.float16, torch.bfloat16}
SPECIAL = [
    "<|fim_prefix|>",
    "<|fim_middle|>",
    "<|box_start|>",
    "<|box_end|>",
    "<|fim_suffix|>",
]


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            result.update(block)
    return result.hexdigest()


def primitive(value: Any, depth: int = 0) -> None:
    """Accept only bounded JSON-like metadata, without checkpoint objects."""
    if depth > 16:
        raise ValueError("Checkpoint metadata nesting is excessive")
    if value is None or type(value) in (str, bool, int):
        return
    if type(value) is float and math.isfinite(value):
        return
    if type(value) is list and len(value) <= 10000:
        for item in value:
            primitive(item, depth + 1)
        return
    if type(value) is dict and len(value) <= 10000 and all(type(k) is str for k in value):
        for item in value.values():
            primitive(item, depth + 1)
        return
    raise ValueError("Unsupported checkpoint metadata object")


def checked_head(
    path: Path, hidden: int, revision: str
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    if not path.is_file() or path.stat().st_size > 64 * 1024 * 1024:
        raise ValueError("Head must be a regular checkpoint of at most 64 MiB")
    payload = torch.load(path, weights_only=True, map_location="cpu", mmap=True)
    fields = {
        "base",
        "base_revision",
        "head",
        "head_dim",
        "lora",
        "option_isolation",
        "special_embeddings",
        "weights_dtype",
        "temperature",
        "holdout",
        "lora_placement",
        "args",
        "suite_sha256",
        "init_source",
        "temperature_fit",
    }
    required = {
        "head",
        "base",
        "base_revision",
        "head_dim",
        "lora",
        "option_isolation",
        "special_embeddings",
        "weights_dtype",
        "temperature",
    }
    if type(payload) is not dict or set(payload) - fields or not required <= payload.keys():
        raise ValueError("Unsupported head payload")
    metadata = {k: v for k, v in payload.items() if k != "head"}
    primitive(metadata)
    if metadata.get("base") != BASE_REPOSITORY or metadata.get("base_revision") != revision:
        raise ValueError("Head base repository/revision mismatch")
    args = metadata.get("args", {})
    if type(args) is not dict:
        raise ValueError("Invalid head training metadata")
    for k in ("base", "base_revision", "head_dim", "lora", "lora_placement"):
        if k in args and args[k] != metadata.get(k):
            raise ValueError("Conflicting head training metadata")
    if (
        metadata.get("option_isolation") is not False
        or metadata.get("special_embeddings") is not False
        or metadata.get("lora_placement", "full") != "full"
        or metadata.get("weights_dtype") != "fp32"
    ):
        raise ValueError("Unsupported hybrid inference options")
    temperature = metadata.get("temperature")
    if type(temperature) not in (int, float) or not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("Inference temperature must be finite and positive")
    rank = metadata.get("lora")
    if type(rank) is not int or not 0 < rank <= 16384:
        raise ValueError("Invalid head LoRA rank")
    dp = metadata.get("head_dim")
    if type(dp) is not int or not 0 < dp <= 16384:
        raise ValueError("Invalid pointer dimension")
    expected = {
        "q.weight": (dp, hidden),
        "q.bias": (dp,),
        "k.weight": (dp, hidden),
        "k.bias": (dp,),
    }
    state = payload["head"]
    if type(state) not in (dict, OrderedDict) or set(state) != set(expected):
        raise ValueError("Missing or extra pointer head tensor")
    tensors = {}
    logical_bytes = 0
    for key, dims in expected.items():
        t = state[key]
        if (
            type(t) is not torch.Tensor
            or t.dtype not in NATIVE
            or tuple(t.shape) != dims
            or t.layout != torch.strided
        ):
            raise ValueError("Pointer tensor shape, dtype or value mismatch")
        logical_bytes += t.numel() * t.element_size()
        if logical_bytes > 64 * 1024 * 1024:
            raise ValueError("Pointer tensors exceed the conversion budget")
        if not bool(torch.isfinite(t).all()):
            raise ValueError("Nonfinite pointer tensor")
        tensors["kev.head." + key] = t.contiguous()
    # Preserve scalar precision as metadata; do not synthesize a weight tensor.
    selected = {
        k: metadata[k]
        for k in (
            "base",
            "base_revision",
            "head_dim",
            "lora",
            "option_isolation",
            "special_embeddings",
            "weights_dtype",
            "temperature",
        )
    }
    selected["lora_placement"] = metadata.get("lora_placement", "full")
    return head_configuration(selected, revision), tensors


def checked_adapter(kev: Path, expected: dict[str, list[int]], revision: str):
    snapshot = FileSnapshot.capture(kev.parent, kev, ("adapter_model.safetensors",))
    config = read_json(kev.parent, kev / "adapter_config.json")
    allowed = {
        "alora_invocation_tokens",
        "alpha_pattern",
        "arrow_config",
        "auto_mapping",
        "base_model_name_or_path",
        "bias",
        "corda_config",
        "ensure_weight_tying",
        "eva_config",
        "exclude_modules",
        "fan_in_fan_out",
        "inference_mode",
        "init_lora_weights",
        "kasa_config",
        "layer_replication",
        "layers_pattern",
        "layers_to_transform",
        "loftq_config",
        "lora_alpha",
        "lora_bias",
        "lora_dropout",
        "lora_ga_config",
        "megatron_config",
        "megatron_core",
        "modules_to_save",
        "monteclora_config",
        "peft_type",
        "peft_version",
        "qalora_group_size",
        "r",
        "rank_pattern",
        "revision",
        "target_modules",
        "target_parameters",
        "task_type",
        "trainable_token_indices",
        "use_bdlora",
        "use_dora",
        "use_qalora",
        "use_rslora",
        "velora_config",
    }
    if (
        set(config) - allowed
        or config.get("peft_type") != "LORA"
        or config.get("task_type") != "FEATURE_EXTRACTION"
        or config.get("base_model_name_or_path") != BASE_REPOSITORY
        or config.get("revision") not in (None, revision)
    ):
        raise ValueError("Unsupported Kev adapter or base revision")
    for k in allowed - {
        "base_model_name_or_path",
        "bias",
        "fan_in_fan_out",
        "inference_mode",
        "init_lora_weights",
        "lora_alpha",
        "lora_dropout",
        "megatron_core",
        "peft_type",
        "peft_version",
        "qalora_group_size",
        "r",
        "revision",
        "target_modules",
        "task_type",
        "ensure_weight_tying",
        "lora_bias",
        "use_dora",
        "use_qalora",
        "use_rslora",
        "use_bdlora",
    }:
        if config.get(k) not in (None, {}, []):
            raise ValueError("Unsupported inference-changing adapter option: " + k)
    for k in (
        "ensure_weight_tying",
        "lora_bias",
        "use_dora",
        "use_qalora",
        "use_rslora",
        "use_bdlora",
    ):
        if config.get(k) not in (None, False):
            raise ValueError("Unsupported adapter extension")
    if (
        config.get("bias") != "none"
        or config.get("fan_in_fan_out") is not False
        or config.get("megatron_core") not in (None, "megatron.core")
    ):
        raise ValueError("Unsupported LoRA orientation or bias")
    for flag in ("inference_mode", "init_lora_weights"):
        if flag in config and type(config[flag]) is not bool:
            raise ValueError("Unsupported adapter flag")
    rank, alpha, dropout = (
        config.get("r"),
        config.get("lora_alpha"),
        config.get("lora_dropout", 0),
    )
    if (
        type(rank) is not int
        or not 0 < rank <= 16384
        or type(alpha) not in (int, float)
        or not math.isfinite(alpha)
        or alpha <= 0
        or type(dropout) not in (int, float)
        or not math.isfinite(dropout)
        or not 0 <= dropout <= 1
    ):
        raise ValueError("Invalid adapter rank, alpha or dropout")
    physical = parse_header(snapshot, "adapter_model.safetensors")
    if any(p.dtype not in ("F32", "F16", "BF16") for p in physical):
        raise ValueError("Unsupported adapter factor dtype")
    factors = factor_bindings(
        {p.name: p for p in physical},
        expected,
        {k: config[k] for k in ("r", "lora_alpha", "target_modules")},
    )
    if {n for n, _ in snapshot.files if n.endswith(".safetensors")} != {
        "adapter_model.safetensors"
    }:
        raise ValueError("Unsupported extra adapter shards")
    return config, factors, snapshot


def export_package(
    base: Path,
    kev: Path,
    destination: Path,
    *,
    base_revision: str,
    kev_revision: str,
    copy_shards: bool = False,
):
    for rev in (base_revision, kev_revision):
        if not re.fullmatch(r"[0-9a-f]{40}", rev):
            raise ValueError("Immutable 40-hex revisions required")
    base, kev = base.resolve(strict=True), kev.resolve(strict=True)
    # Resolve existing symlink parents and traversal before enforcing separation.
    destination = destination.resolve(strict=False)
    if destination.exists() or destination.is_relative_to(base) or destination.is_relative_to(kev):
        raise ValueError("Destination must be new and outside inputs")
    config = read_json(base.parent, base / "config.json")
    if (
        config.get("model_type") != "qwen3_5"
        or config.get("architectures") != ["Qwen3_5ForConditionalGeneration"]
        or config.get("quantization_config") is not None
    ):
        raise ValueError("Complete native Qwen3.5 base required")
    binding = {
        k: config[k]
        for k in ("_name_or_path", "name_or_path", "_commit_hash", "revision")
        if config.get(k) is not None
    }
    if any(
        binding[k] != BASE_REPOSITORY for k in ("_name_or_path", "name_or_path") if k in binding
    ) or any(binding[k] != base_revision for k in ("_commit_hash", "revision") if k in binding):
        raise ValueError("Base configuration identity/revision mismatch")
    entry = ModelCatalogue(base.parent).inspect_directory(base)
    source = entry.pin()
    if not entry.summary.tokenizer_available:
        raise ValueError("Base tokenizer required")
    inputs = AnalysisInput.from_source(source, tokenizer_available=True)
    normalized = copy.deepcopy(config)
    for key in ("_commit_hash", "revision", "name_or_path"):
        normalized.pop(key, None)
    normalized["text_config"].setdefault(
        "partial_rotary_factor",
        normalized["text_config"]["rope_parameters"]["partial_rotary_factor"],
    )
    c = configuration(replace(inputs, configuration=normalized))
    if c is None:
        raise ValueError("Unreviewed Qwen3.5 configuration")
    expected = parameter_shapes(c)
    physical = {p.name: p for p in source.physical_tensors()}
    language = {n for n in physical if n.startswith(PREFIX + ".")}
    if language != set(expected) or any(
        list(physical[n].shape) != dims or physical[n].dtype not in ("F32", "F16", "BF16")
        for n, dims in expected.items()
    ):
        raise ValueError("Incomplete or incompatible native text backbone")
    checkpoint_snapshot = FileSnapshot.capture(
        kev.parent, kev, ("head.pt", "adapter_model.safetensors")
    )
    head_path = kev / "head.pt"
    before_head = digest(head_path)
    head, tensors = checked_head(head_path, c["hidden_size"], base_revision)
    adapter, factors, adapter_snapshot = checked_adapter(kev, expected, base_revision)
    if head["lora"] != adapter["r"]:
        raise ValueError("Head/adapter rank mismatch")
    for name in ("training_config.json", "provenance.json"):
        if (kev / name).is_file():
            meta = read_json(kev.parent, kev / name)
            selected = meta.get("config", meta.get("args", meta))
            if (
                selected.get("base", BASE_REPOSITORY) != BASE_REPOSITORY
                or selected.get("base_revision", base_revision) != base_revision
            ):
                raise ValueError("Checkpoint metadata base mismatch")
    if (kev / "adapter_model.safetensors").stat().st_size > 256 * 1024 * 1024:
        raise ValueError("Adapter exceeds the 256 MiB conversion budget")
    with safe_open(kev / "adapter_model.safetensors", framework="pt", device="cpu") as f:
        for original in f.keys():
            t = f.get_tensor(original)
            if not bool(torch.isfinite(t).all()):
                raise ValueError("Nonfinite adapter factor")
            tensors["kev.lora." + original] = t.contiguous()
    weights = {p.name: "base/" + p.file for p in source.physical_tensors()}
    if set(tensors) & set(weights):
        raise ValueError("Tensor namespace collision")
    shards = sorted({p.file for p in source.physical_tensors()})
    destination.parent.mkdir(parents=True, exist_ok=True)
    required = 128 * 1024 * 1024 + sum(t.numel() * t.element_size() for t in tensors.values())
    if copy_shards:
        required += sum((base / n).stat().st_size for n in shards)
    if shutil.disk_usage(destination.parent).free < required:
        raise ValueError("Insufficient disk")
    if not copy_shards and not base.is_relative_to(destination.parent.resolve()):
        raise ValueError("Shared shards must remain within model root; use --copy-shards")
    # The container has no model config, so immediate catalogue scans cannot
    # select its nested package. Keep it on the checked output filesystem.
    container = Path(tempfile.mkdtemp(prefix=".kev-export-", dir=destination.parent))
    staging = container / "package"
    try:
        staging.mkdir(mode=0o700)
        for name in shards:
            target = staging / "base" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            if copy_shards:
                shutil.copyfile(base / name, target)
            else:
                target.symlink_to(base / name)
        assets = {
            "tokenizer.json",
            "tokenizer_config.json",
            "vocab.json",
            "merges.txt",
            "special_tokens_map.json",
            "added_tokens.json",
            "LICENSE",
        }
        for name in assets:
            if (base / name).is_file():
                shutil.copyfile(base / name, staging / name)
        save_file(tensors, str(staging / "kev.safetensors"))
        weights.update({n: "kev.safetensors" for n in tensors})
        (staging / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": weights}, sort_keys=True) + "\n"
        )
        config["_name_or_path"] = EXPORT_IDENTITY
        config["_commit_hash"] = kev_revision
        config["kev_inspection"] = {
            "format_version": 1,
            "base_repository": BASE_REPOSITORY,
            "base_revision": base_revision,
            "kev_revision": kev_revision,
            "head": head,
            "adapter": {k: adapter[k] for k in ("r", "lora_alpha", "target_modules")},
        }
        (staging / "config.json").write_text(json.dumps(config, sort_keys=True) + "\n")
        provenance = {
            "producer": PRODUCER,
            "producer_sha256": digest(Path(__file__)),
            "source": {
                "repository": "jaredpalmer/kev",
                "revision": SOURCE_REVISION,
                "license": "Apache-2.0",
                "files": SOURCE_FILES,
            },
            "qwen_description": QWEN_PRODUCER.__dict__,
            "base": {
                "repository": BASE_REPOSITORY,
                "revision": base_revision,
                "configuration_binding": binding,
                "revision_origin": "configuration"
                if any(k in binding for k in ("_commit_hash", "revision"))
                else "operator_selection_checked_against_head",
                "content_fingerprint": source.fingerprint,
                "license": "Apache-2.0",
                "files": [
                    {
                        "name": n,
                        "sha256": digest(base / n),
                        "bytes": (base / n).stat().st_size,
                    }
                    for n in sorted(
                        set(shards) | assets | {"config.json", "model.safetensors.index.json"}
                    )
                    if (base / n).is_file()
                ],
            },
            "kev": {
                "repository": KEV_REPOSITORY,
                "revision": kev_revision,
                "license": "Apache-2.0",
                "adapter_configuration": adapter,
                "head_metadata": head,
                "files": [
                    {
                        "name": n,
                        "sha256": digest(kev / n),
                        "bytes": (kev / n).stat().st_size,
                    }
                    for n in (
                        "head.pt",
                        "adapter_model.safetensors",
                        "adapter_config.json",
                        "training_config.json",
                        "provenance.json",
                        "tokenizer.json",
                        "tokenizer_config.json",
                        "README.md",
                    )
                    if (kev / n).is_file()
                ],
            },
            "namespace": {
                "base": "unchanged",
                "factors": "kev.lora.<exact-original-key>",
                "head": "kev.head.<exact-original-key>",
            },
            "targets": [
                {
                    "base_weight": module + ".weight",
                    "A": "kev.lora." + pair["A"],
                    "B": "kev.lora." + pair["B"],
                    "scale": adapter["lora_alpha"] / adapter["r"],
                    "orientation": "W[out,in], A[rank,in], B[out,rank]",
                }
                for module, pair in sorted(factors.items())
            ],
            "packaged_description": KEV_PRODUCER.__dict__,
        }
        (staging / "kev-provenance.json").write_text(json.dumps(provenance, sort_keys=True) + "\n")
        source.check_unchanged()
        adapter_snapshot.check()
        checkpoint_snapshot.check()
        if digest(head_path) != before_head:
            raise ValueError("Head changed during conversion")
        result = validate_directory(staging, model_root=destination.parent)
        if result["status"] != "valid":
            raise ValueError("Export failed static validation: " + str(result))
        if destination.exists():
            raise ValueError("Destination appeared during export")
        staging.rename(destination)
        return result
    finally:
        shutil.rmtree(container)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--kev", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-revision", required=True)
    parser.add_argument("--kev-revision", required=True)
    parser.add_argument("--copy-shards", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(
            export_package(
                args.base,
                args.kev,
                args.output,
                base_revision=args.base_revision,
                kev_revision=args.kev_revision,
                copy_shards=args.copy_shards,
            ),
            sort_keys=True,
        )
    )

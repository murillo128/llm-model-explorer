"""Offline CLM inspection-package exporter; never imported by the server.

Correspondence: Contrastive-LM/CLM at SOURCE_REVISION, heads.py and embedder.py.
Only the small head is materialized. Encoder shards remain native and shared.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from llm_model_explorer.architecture_analysis import (
    AnalysisInput,
    DescriptionRegistry,
    register_dense_descriptions,
)
from llm_model_explorer.architecture_analysis.clm import PRODUCER as ARCHITECTURE_PRODUCER
from llm_model_explorer.architecture_analysis.clm_config import (
    ENCODER_REPOSITORY,
    HEAD_REPOSITORY,
    SOURCE_REVISION,
    head_configuration,
    head_shapes,
)
from llm_model_explorer.architecture_analysis.dense_config import (
    SOURCE_REVISION as QWEN_SOURCE_REVISION,
)
from llm_model_explorer.models import ModelCatalogue
from llm_model_explorer.validate_architecture_cli import validate_directory
from safetensors.torch import save_file

ENCODER_REVISION = "b968826d9c46dd6066d109eabc6255188de91218"
HEAD_REVISION = "e939398d4556fcd9400c76fa8c5a513202f42b0a"
PRODUCER = "clm-inspection-export-2"
NATIVE = {torch.float32, torch.float16, torch.bfloat16}
SOURCE_FILES = {
    "src/clm/heads.py": "3f3b880e940a47b45879614b140fd873f7de9b13ccb8254b07989af7ea92e093",
    "src/clm/embedder.py": "d0ca78073aa1c59676e17e35d9c8cda8131da03010841887f833fbfe8fc69dbc",
    "src/clm/engine.py": "838897b99b1b09ce141f35272450e1c59fd290f092d1362570e07a453f06f5e7",
    "LICENSE": "cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30",
}


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()


def checked_head(path: Path, hidden: int) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    """Restricted CPU admission of tensors and primitive configuration only."""
    if not path.is_file() or path.stat().st_size > 512 * 1024 * 1024:
        raise ValueError("Head must be a regular checkpoint of at most 512 MiB")
    # Explicit weights_only, no safe-global extensions, no checkpoint Python.
    payload = torch.load(path, weights_only=True, map_location="cpu", mmap=True)
    fields = {"state_head", "action_head", "logit_scale", "cfg", "hidden_size", "projection_dim"}
    if (
        not isinstance(payload, dict)
        or set(payload) - fields
        or not fields.difference({"hidden_size", "projection_dim"}) <= payload.keys()
    ):
        raise ValueError("Unsupported head payload")
    cfg = payload["cfg"]
    allowed = {
        "model",
        "hidden_size",
        "projection_dim",
        "width",
        "depth",
        "activation",
        "layernorm",
        "residual",
    }
    if not isinstance(cfg, dict) or set(cfg) - allowed:
        raise ValueError("Unsupported head configuration")
    cfg = dict(cfg)
    for key, default in (("hidden_size", 4096), ("projection_dim", 512)):
        cfg.setdefault(key, default)
        if key in payload and payload[key] != cfg[key]:
            raise ValueError("Conflicting head dimensions")
    cfg.setdefault("activation", "gelu")
    cfg.setdefault("layernorm", False)
    cfg.setdefault("residual", False)
    cfg.setdefault("model", ENCODER_REPOSITORY)
    cfg = head_configuration(cfg, hidden)
    expected = head_shapes(cfg)
    tensors = {}
    logical_bytes = 0
    for head in ("state_head", "action_head"):
        state = payload[head]
        if not isinstance(state, Mapping) or set(state) != set(expected):
            raise ValueError("Missing, extra or malformed head tensors")
        for name, dims in expected.items():
            tensor = state[name]
            if (
                type(tensor) is not torch.Tensor
                or tensor.dtype not in NATIVE
                or tuple(tensor.shape) != dims
            ):
                raise ValueError("Head tensor shape or dtype mismatch")
            logical_bytes += tensor.numel() * tensor.element_size()
            if logical_bytes > 512 * 1024 * 1024 or tensor.layout != torch.strided:
                raise ValueError("Head tensors exceed the native conversion budget")
            if not bool(torch.isfinite(tensor).all()):
                raise ValueError("Nonfinite head tensor")
            tensors[f"clm.{head}.{name}"] = tensor.contiguous()
    scale = payload["logit_scale"]
    if type(scale) in {int, float}:
        scale = torch.tensor(scale, dtype=torch.float32)
    if (
        type(scale) is not torch.Tensor
        or scale.shape != torch.Size([])
        or scale.dtype not in NATIVE
        or not bool(torch.isfinite(scale))
    ):
        raise ValueError("Invalid learned logit scale")
    tensors["clm.logit_scale"] = scale.contiguous()
    return cfg, tensors


def export_package(
    encoder: Path,
    head: Path,
    destination: Path,
    *,
    encoder_revision: str,
    head_revision: str,
    copy_shards: bool = False,
) -> dict[str, Any]:
    for revision in (encoder_revision, head_revision):
        if not re.fullmatch(r"[0-9a-f]{40}", revision):
            raise ValueError("Inputs require immutable 40-hex revisions")
    encoder, head = encoder.resolve(strict=True), head.resolve(strict=True)
    if destination.is_symlink():
        raise ValueError("Destination must be new and outside the encoder")
    destination = destination.resolve()
    if destination.exists() or destination.is_relative_to(encoder):
        raise ValueError("Destination must be new and outside the encoder")
    config = json.loads((encoder / "config.json").read_text())
    # Bind any declared source identity before replacing it with the CLM identity.
    # Bare upstream configs can omit these fields; their provenance must identify
    # the operator's verified reference selection rather than claim a metadata match.
    identity_fields = ("_name_or_path", "name_or_path")
    revision_fields = ("_commit_hash", "revision")
    metadata = {
        key: config[key]
        for key in (*identity_fields, *revision_fields)
        if config.get(key) is not None
    }
    if any(metadata[key] != ENCODER_REPOSITORY for key in identity_fields if key in metadata):
        raise ValueError("Encoder repository metadata does not match Qwen/Qwen3-8B")
    if any(metadata[key] != encoder_revision for key in revision_fields if key in metadata):
        raise ValueError("Encoder revision metadata does not match the selected revision")
    binding = {
        "configuration": metadata,
        "repository_source": "configuration"
        if any(key in metadata for key in identity_fields)
        else "operator_selection",
        "revision_source": "configuration"
        if any(key in metadata for key in revision_fields)
        else "operator_selection",
    }
    if (
        config.get("model_type") != "qwen3"
        or config.get("architectures") != ["Qwen3ForCausalLM"]
        or config.get("quantization_config") is not None
    ):
        raise ValueError("A complete native Qwen3 encoder is required")
    before_head = digest(head)
    cfg, tensors = checked_head(head, config.get("hidden_size"))
    entry = ModelCatalogue(encoder.parent).inspect_directory(encoder)
    if not entry.summary.tokenizer_available:
        raise ValueError("Encoder tokenizer is required")
    source = entry.pin()
    inputs = AnalysisInput.from_source(source, tokenizer_available=True)
    registry = DescriptionRegistry()
    register_dense_descriptions(registry)
    result = registry.analyze(inputs)
    if result.graph is None or result.status != "complete" or result.graph.diagnostics:
        raise ValueError("Encoder is outside the complete native Qwen3 description")
    weight_map = {p.name: p.file for p in source.physical_tensors()}
    if set(tensors) & set(weight_map):
        raise ValueError("Head namespace collides with encoder tensors")
    shards = sorted(set(weight_map.values()))
    assets = [
        p.name
        for p in encoder.iterdir()
        if p.is_file()
        and p.name
        in {
            "config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "vocab.json",
            "merges.txt",
            "special_tokens_map.json",
            "added_tokens.json",
            "LICENSE",
            "model.safetensors.index.json",
        }
    ]
    required_bytes = head.stat().st_size * 2 + 32 * 1024 * 1024
    if copy_shards:
        required_bytes += sum((encoder / f).stat().st_size for f in shards)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(destination.parent).free < required_bytes:
        raise ValueError("Insufficient free disk for export")
    if not copy_shards and not encoder.is_relative_to(destination.parent.resolve()):
        raise ValueError(
            "Shared shards must remain inside the destination model root; use --copy-shards"
        )
    # A private container has no config, so catalogue scans cannot select the
    # nested package. It shares the output filesystem even at a mount boundary.
    container = Path(tempfile.mkdtemp(prefix=".clm-export-", dir=destination.parent))
    staging = container / "package"
    try:
        staging.mkdir(mode=0o700)
        for name in shards:
            target = staging / name
            target.parent.mkdir(parents=True, exist_ok=True)
            if copy_shards:
                shutil.copyfile(encoder / name, target)
            else:
                target.symlink_to(encoder / name)
        for name in assets:
            if name not in {"config.json", "model.safetensors.index.json"}:
                shutil.copyfile(encoder / name, staging / name)
        head_license = head.parent / "LICENSE"
        if head_license.is_file():
            shutil.copyfile(head_license, staging / "CLM-LICENSE.txt")
        if (staging / "clm-heads.safetensors").exists():
            raise ValueError("Head shard name collision")
        save_file(tensors, str(staging / "clm-heads.safetensors"))
        weight_map.update({name: "clm-heads.safetensors" for name in tensors})
        (staging / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": weight_map}, sort_keys=True) + "\n"
        )
        config["_name_or_path"] = HEAD_REPOSITORY
        config["_commit_hash"] = head_revision
        # Preserve the Qwen text layout for exact input-table lookup, while
        # selecting the reviewed CLM description through explicit inspection metadata.
        config["clm_inspection"] = {
            "encoder_repository": ENCODER_REPOSITORY,
            "encoder_revision": encoder_revision,
            "head_revision": head_revision,
            "head_configuration": cfg,
            "pooling": "last_token",
            "format_version": 1,
        }
        (staging / "config.json").write_text(json.dumps(config, sort_keys=True) + "\n")
        provenance = {
            "producer": PRODUCER,
            "producer_sha256": digest(Path(__file__)),
            "qwen_description_source_revision": QWEN_SOURCE_REVISION,
            "architecture": {
                "description": ARCHITECTURE_PRODUCER.description,
                "revision": ARCHITECTURE_PRODUCER.revision,
                "source_revision": ARCHITECTURE_PRODUCER.source_revision,
            },
            "source": {
                "repository": "Contrastive-LM/CLM",
                "revision": SOURCE_REVISION,
                "license": "Apache-2.0",
                "files": [{"name": name, "sha256": sha} for name, sha in SOURCE_FILES.items()],
            },
            "encoder": {
                "repository": ENCODER_REPOSITORY,
                "revision": encoder_revision,
                "binding": binding,
                "content_fingerprint": source.fingerprint,
                "license": "Apache-2.0",
                "files": [
                    {
                        "name": name,
                        "bytes": (encoder / name).stat().st_size,
                        "sha256": digest(encoder / name),
                    }
                    for name in sorted(set(assets + shards))
                ],
            },
            "head": {
                "repository": HEAD_REPOSITORY,
                "filename": head.name,
                "revision": head_revision,
                "license": "Apache-2.0",
                "sha256": before_head,
                "bytes": head.stat().st_size,
                "configuration": cfg,
                "files": [
                    {"name": name, "sha256": digest(head.parent / name)}
                    for name in ("config.json", "LICENSE", "README.md")
                    if (head.parent / name).is_file()
                ],
            },
            "namespace": {
                "encoder": "unchanged",
                "state_head": "clm.state_head.<original-name>",
                "action_head": "clm.action_head.<original-name>",
                "logit_scale": "clm.logit_scale",
            },
            "inventory": [
                {"name": p.name, "shape": list(p.shape), "dtype": p.dtype}
                for p in source.physical_tensors()
            ]
            + [
                {"name": name, "shape": list(t.shape), "dtype": str(t.dtype)}
                for name, t in sorted(tensors.items())
            ],
        }
        (staging / "clm-provenance.json").write_text(json.dumps(provenance, sort_keys=True) + "\n")
        source.check_unchanged()
        if digest(head) != before_head:
            raise ValueError("Head changed during export")
        validation = validate_directory(staging, model_root=destination.parent)
        if validation["status"] != "valid":
            raise ValueError(f"Exported package failed static import: {validation}")
        if destination.exists():
            raise ValueError("Destination appeared during export")
        staging.rename(destination)
        return validation
    finally:
        shutil.rmtree(container)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--encoder", type=Path, required=True)
    parser.add_argument("--head", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--encoder-revision", required=True)
    parser.add_argument("--head-revision", required=True)
    parser.add_argument("--copy-shards", action="store_true")
    args = parser.parse_args()
    result = export_package(
        args.encoder,
        args.head,
        args.output,
        encoder_revision=args.encoder_revision,
        head_revision=args.head_revision,
        copy_shards=args.copy_shards,
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()

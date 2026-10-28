"""Offline, byte-preserving export of the explicitly selected video references.

Run using the installed backend environment. This tool never downloads, imports
checkpoint code, instantiates a model, or executes inference.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import shutil
import struct
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from llm_model_explorer.model_files import FileSnapshot, invalid, parse_json
from llm_model_explorer.models import ModelCatalogue
from llm_model_explorer.tensor_source import PhysicalTensor, parse_header
from llm_model_explorer.video_profiles import (
    PROFILES,
    VideoProfile,
    artifact,
    checked,
    declaration,
    reference,
)

COPY_BYTES = 1024 * 1024


@dataclass(frozen=True)
class Input:
    snapshot: FileSnapshot
    filename: str
    reference: dict[str, Any]
    tensors: tuple[PhysicalTensor, ...] = ()

    def digest(self) -> str:
        sha = hashlib.sha256()
        blob = hashlib.sha1(f"blob {self.reference['size_bytes']}\0".encode())
        count = 0
        with self.snapshot.open(self.filename) as source:
            while block := source.read(COPY_BYTES):
                count += len(block)
                sha.update(block)
                blob.update(block)
        if count != self.reference["size_bytes"]:
            raise invalid("Selected video artifact has the wrong byte length.")
        expected = self.reference["sha256_lfs_oid"]
        if (expected and sha.hexdigest() != expected) or (
            not expected and blob.hexdigest() != self.reference["git_blob_oid"]
        ):
            raise invalid("Selected artifact content does not match its reference role.")
        return sha.hexdigest()


def selected(path: Path, ref: dict[str, Any], *, weights: bool = True) -> Input:
    directory = path.parent.resolve(strict=True)
    snapshot = FileSnapshot.capture(directory, directory, (path.name,))
    tensors = parse_header(snapshot, path.name) if weights else ()
    if dict(snapshot.files)[path.name].size != ref["size_bytes"]:
        raise invalid("Selected video artifact has the wrong byte length.")
    return Input(snapshot, path.name, ref, tensors)


def source_metadata(item: Input) -> dict[str, str]:
    with item.snapshot.open(item.filename) as stream:
        length = struct.unpack("<Q", stream.read(8))[0]
        return dict(parse_json(stream.read(length)).get("__metadata__", {}))


def inputs(
    profile: VideoProfile, paths: dict[str, Path], tokenizer: Path
) -> tuple[dict[str, list[Input]], list[Input]]:
    result: dict[str, list[Input]] = {}
    required = (
        set(profile.components) - {"vae"} if profile.family == "ltx" else set(profile.components)
    )
    if set(paths) != required:
        raise invalid("Select exactly the local files required by this video profile.")
    if profile.family == "ltx":
        checkpoint = selected(
            paths["denoiser"],
            artifact(f"ltx.2b.{profile.release.replace('.', '_')}.distilled"),
        )
        metadata = source_metadata(checkpoint)
        if (
            json.loads(metadata.get("config", "{}"))
            != reference()["ltx_configurations"][profile.release]
        ):
            raise invalid("LTX embedded configuration disagrees with the selected release.")
        result["denoiser"] = result["vae"] = [checkpoint]
        result["text_encoder"] = [
            selected(
                paths["text_encoder"] / Path(artifact(role)["path"]).name,
                artifact(role),
            )
            for role in ("text_encoder.shard_1", "text_encoder.shard_2")
        ]
        text_config = selected(
            paths["text_encoder"] / "config.json",
            artifact("text_encoder.config"),
            weights=False,
        )
        # The data-only config is part of input integrity, even though its checked
        # content is embedded in the new config rather than replacing that config.
        result["text_encoder"].append(text_config)
        tokenizer_repo = "PixArt-alpha/PixArt-XL-2-1024-MS"
    else:
        for role in profile.components:
            if role == "text_encoder":
                ref_role = "text_encoder.umt5_xxl.fp8_scaled"
            elif role == "vae":
                ref_role = "vae.wan_2_2" if profile.channels == 48 else "vae.wan_2_1"
            elif role == "denoiser":
                ref_role = "denoiser.ti2v_5b.fp16"
            else:
                ref_role = f"denoiser.{profile.variant.replace('-', '_')}.{role}.fp8_scaled"
            result[role] = [selected(paths[role], artifact(ref_role))]
        tokenizer_repo = "google/umt5-xxl"
    repo = next(
        repo for repo in reference()["repositories"] if repo["repository"] == tokenizer_repo
    )
    token_inputs = [
        selected(
            tokenizer / Path(a["path"]).name,
            {
                **a,
                "repository": repo["repository"],
                "revision": repo["revision"],
                "license": repo["license_declaration"],
            },
            weights=False,
        )
        for a in repo["artifacts"]
        if a["role"] == "tokenizer"
    ]
    return result, token_inputs


def namespaced(profile: VideoProfile, role: str, tensor: PhysicalTensor) -> str | None:
    original = tensor.name
    if profile.family == "ltx" and role != "text_encoder":
        prefix = "model.diffusion_model." if role == "denoiser" else "vae."
        if not original.startswith(prefix):
            return None
        original = original.removeprefix(prefix)
    # The selected UMT5 embeds its tokenizer bytes as an auxiliary tensor. The
    # package carries the verified identical conventional tokenizer asset instead.
    if role == "text_encoder" and original == "spiece_model":
        return None
    return f"video.{role}.{original}"


def copy_shard(
    destination: Path, item: Input, selected_tensors: list[tuple[str, PhysicalTensor]]
) -> None:
    header: dict[str, Any] = {}
    offset = 0
    for name, tensor in selected_tensors:
        if name in header:
            raise invalid("Video namespace collision.")
        header[name] = {
            "dtype": tensor.dtype,
            "shape": list(tensor.shape),
            "data_offsets": [offset, offset + tensor.byte_length],
        }
        offset += tensor.byte_length
    metadata = source_metadata(item)
    header["__metadata__"] = {
        "format": "pt",
        "inspection_export": "video-inspection-1; namespaced, original value bytes retained",
    }
    if "license" in metadata:
        header["__metadata__"]["license"] = metadata["license"]
    encoded = json.dumps(header, separators=(",", ":"), ensure_ascii=True).encode()
    encoded += b" " * (-len(encoded) % 8)
    with destination.open("xb") as output, item.snapshot.open(item.filename) as source:
        output.write(struct.pack("<Q", len(encoded)))
        output.write(encoded)
        for _, tensor in selected_tensors:
            source.seek(tensor.offset)
            remaining = tensor.byte_length
            while remaining:
                block = source.read(min(COPY_BYTES, remaining))
                if not block:
                    raise invalid("Source tensor was truncated during export.")
                output.write(block)
                remaining -= len(block)
        output.flush()
        os.fsync(output.fileno())


def publish(staging: Path, output: Path) -> None:
    """Atomic no-replace publication on Linux; never replace a concurrent target."""
    libc = ctypes.CDLL(None, use_errno=True)
    rename = getattr(libc, "renameat2", None)
    if rename is None:
        raise RuntimeError("This exporter requires atomic renameat2(RENAME_NOREPLACE).")
    rename.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    rename.restype = ctypes.c_int
    if rename(-100, os.fsencode(staging), -100, os.fsencode(output), 1) != 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error), str(output))


def export(
    profile_key: str, paths: dict[str, Path], tokenizer: Path, output: Path
) -> dict[str, Any]:
    profile = PROFILES[profile_key]
    if output.exists() or output.is_symlink():
        raise ValueError("Refusing to overwrite an existing output.")
    if any(parent.is_symlink() for parent in (output.parent, *output.parents)):
        raise ValueError("Output parents must not be symlinks.")
    output = output.parent.resolve(strict=True) / output.name
    for path in (*paths.values(), tokenizer):
        source = path.resolve(strict=True)
        source_root = source if source.is_dir() else source.parent
        if output.is_relative_to(source_root) or source.is_relative_to(output):
            raise ValueError("Output must be outside every input directory.")
    components, token_inputs = inputs(profile, paths, tokenizer)
    unique = {
        (item.snapshot.directory, item.filename): item
        for items in components.values()
        for item in items
    }
    unique.update({(item.snapshot.directory, item.filename): item for item in token_inputs})
    # Two copies coexist; the existing source copy is already charged by disk free.
    required = sum(item.reference["size_bytes"] for item in unique.values()) + 16 * COPY_BYTES
    if shutil.disk_usage(output.parent).free < required:
        raise ValueError(f"Insufficient free space: export needs {required} additional bytes.")
    digests = {key: item.digest() for key, item in unique.items()}
    config: dict[str, Any] = {
        "model_type": "video_inspection",
        "architectures": [profile.architecture],
        "_name_or_path": profile.identity,
        "video_inspection": declaration(profile),
    }
    if profile.family == "wan":
        scaled = ["text_encoder"] + (["high_noise", "low_noise"] if profile.boundary else [])
        config["quantization_config"] = {
            "quant_method": "fp8-e4m3-scaled",
            "format_version": 1,
            "components": {f"video.{role}": "legacy" for role in scaled},
        }
    provenance: dict[str, Any] = {
        "exporter": "video-inspection-1",
        "profile": profile.key,
        "sources": [
            {
                "repository": item.reference["repository"],
                "revision": item.reference["revision"],
                "artifact": item.reference["path"],
                "sha256": digests[key],
                "size_bytes": item.reference["size_bytes"],
                "license": item.reference["license"],
            }
            for key, item in unique.items()
        ],
        "tensors": {},
    }
    with tempfile.TemporaryDirectory(prefix=".video-export-", dir=output.parent) as temporary:
        # The immediate staging directory has no config, so catalogue discovery
        # cannot select a half-written package beneath it.
        staging = Path(temporary) / "package"
        staging.mkdir()
        weight_map: dict[str, str] = {}
        for role, items in components.items():
            for index, item in enumerate(items):
                records = [
                    (name, tensor)
                    for tensor in item.tensors
                    if (name := namespaced(profile, role, tensor)) is not None
                ]
                if not records:
                    continue
                filename = f"{role}-{index:02}.safetensors"
                for name, tensor in records:
                    if name in weight_map:
                        raise invalid("Video namespace collision.")
                    weight_map[name] = filename
                    provenance["tensors"][name] = {
                        "role": role,
                        "original_name": tensor.name,
                        "artifact": item.reference["path"],
                    }
                copy_shard(staging / filename, item, records)
        for item in token_inputs:
            with (
                item.snapshot.open(item.filename) as source,
                (staging / item.filename).open("xb") as target,
            ):
                shutil.copyfileobj(source, target, COPY_BYTES)
        for filename, data in (
            ("config.json", config),
            ("video-provenance.json", provenance),
            ("model.safetensors.index.json", {"weight_map": weight_map}),
        ):
            (staging / filename).write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
        notices = [
            f"{s['repository']} @ {s['revision']}: {s['license']}\nhttps://huggingface.co/{s['repository']}/tree/{s['revision']}"
            for s in provenance["sources"]
        ]
        (staging / "LICENSES.txt").write_text(
            "\n\n".join(dict.fromkeys(notices))
            + "\nEmbedded source license notices are retained in shard metadata.\n"
        )
        catalogue = ModelCatalogue(Path(temporary))
        entries = catalogue.discover()
        if len(entries) != 1:
            raise invalid("Exported video package failed storage admission.")
        entry = entries[0]
        checked(config, {tensor.name: tensor for tensor in entry.physical_tensors()})
        for key, item in unique.items():
            if item.digest() != digests[key]:
                raise invalid("Source artifact changed during export.")
        publish(staging, output)
    return {
        "profile": profile.key,
        "model_id": entry.summary.id,
        "tensors": len(weight_map),
        "source_bytes": required - 16 * COPY_BYTES,
        "sources": provenance["sources"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=sorted(PROFILES), required=True)
    for role in ("denoiser", "high-noise", "low-noise", "vae", "text-encoder"):
        parser.add_argument("--" + role, type=Path)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    paths = {
        role: value
        for role in ("denoiser", "high_noise", "low_noise", "vae", "text_encoder")
        if (value := getattr(args, role)) is not None
    }
    print(json.dumps(export(args.profile, paths, args.tokenizer, args.output), indent=2))


if __name__ == "__main__":
    main()

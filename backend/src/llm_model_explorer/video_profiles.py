"""Closed, source-reviewed inspection profiles. No checkpoint code is executed."""

import json
from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Protocol

from .model_files import invalid

WAN_SOURCE = "1ea34ff48f87168174e12956e200b1d908b1c5ff"
LTX_SOURCE = "4b2d053057623ddd4d0a1d3e9cd28890e9ef487f"


@lru_cache(maxsize=1)
def reference() -> dict[str, Any]:
    return json.loads(Path(__file__).with_name("video_reference.json").read_text())  # type: ignore[no-any-return]


def inventory(key: str) -> dict[str, tuple[str, tuple[int, ...]]]:
    specification = reference()["inventories"][key]
    result = {name: (value[0], tuple(value[1])) for name, value in specification["fixed"].items()}
    for repeated in specification["repeated"]:
        for index in range(repeated["count"]):
            result.update(
                {
                    f"{repeated['prefix']}.{index}.{name}": (value[0], tuple(value[1]))
                    for name, value in repeated["tensors"].items()
                }
            )
    return result


def artifact(role: str) -> dict[str, Any]:
    matches = [
        {
            **item,
            "repository": repo["repository"],
            "revision": repo["revision"],
            "license": repo["license_declaration"],
        }
        for repo in reference()["repositories"]
        for item in repo["artifacts"]
        if item["role"] == role
    ]
    if len(matches) != 1:
        raise invalid("Unknown video reference artifact.")
    return matches[0]


@dataclass(frozen=True)
class VideoProfile:
    key: str
    family: str
    variant: str
    release: str
    modalities: tuple[str, ...]
    denoiser: str
    vae: str
    text: str
    layers: int
    hidden: int
    heads: int
    channels: int
    boundary: float | None = None

    @property
    def architecture(self) -> str:
        return "LTXVideoInspection" if self.family == "ltx" else "WanVideoInspection"

    @property
    def components(self) -> dict[str, str]:
        denoisers = ("denoiser",) if self.boundary is None else ("high_noise", "low_noise")
        return {
            **dict.fromkeys(denoisers, self.denoiser),
            "text_encoder": self.text,
            "vae": self.vae,
        }

    @property
    def identity(self) -> str:
        representation = (
            "native" if self.family == "ltx" else "fp16" if self.boundary is None else "fp8-scaled"
        )
        return f"video/{self.family}-{self.release}-{self.variant}-{representation}"


PROFILES = {
    profile.key: profile
    for profile in (
        VideoProfile(
            "ltx-0.9.6",
            "ltx",
            "2b-distilled",
            "0.9.6",
            ("text_to_video", "image_to_video"),
            "ltx-denoiser",
            "ltx-vae",
            "t5",
            28,
            2048,
            32,
            128,
        ),
        VideoProfile(
            "ltx-0.9.8",
            "ltx",
            "2b-distilled",
            "0.9.8",
            ("text_to_video", "image_to_video"),
            "ltx-denoiser",
            "ltx-vae",
            "t5",
            28,
            2048,
            32,
            128,
        ),
        VideoProfile(
            "wan-ti2v-5b",
            "wan",
            "ti2v-5b",
            "2.2",
            ("text_to_video", "image_to_video"),
            "wan-ti2v",
            "wan-vae-2.2",
            "umt5",
            30,
            3072,
            24,
            48,
        ),
        VideoProfile(
            "wan-t2v-a14b",
            "wan",
            "t2v-a14b",
            "2.2",
            ("text_to_video",),
            "wan-t2v",
            "wan-vae-2.1",
            "umt5",
            40,
            5120,
            40,
            16,
            0.875,
        ),
        VideoProfile(
            "wan-i2v-a14b",
            "wan",
            "i2v-a14b",
            "2.2",
            ("image_to_video",),
            "wan-i2v",
            "wan-vae-2.1",
            "umt5",
            40,
            5120,
            40,
            16,
            0.900,
        ),
    )
}


def component_configuration(profile: VideoProfile, role: str) -> dict[str, Any]:
    if role == "text_encoder":
        return dict(reference()["text_configurations"][profile.text])
    if profile.family == "ltx":
        return dict(
            reference()["ltx_configurations"][profile.release][
                "vae" if role == "vae" else "transformer"
            ]
        )
    if role == "vae":
        return {
            "version": "2.2" if profile.channels == 48 else "2.1",
            "latent_channels": profile.channels,
            "stride": [4, 16, 16] if profile.channels == 48 else [4, 8, 8],
            "encoder_dim": 160 if profile.channels == 48 else 96,
            "decoder_dim": 256 if profile.channels == 48 else 96,
        }
    return {
        "dim": profile.hidden,
        "num_heads": profile.heads,
        "num_layers": profile.layers,
        "ffn_dim": 14336 if profile.channels == 48 else 13824,
        "freq_dim": 256,
        "in_dim": 36 if profile.variant == "i2v-a14b" else profile.channels,
        "out_dim": profile.channels,
        "patch_size": [1, 2, 2],
        "text_dim": 4096,
        "text_len": 512,
        "qk_norm": True,
        "cross_attn_norm": True,
        "eps": 1e-6,
    }


def component_artifacts(profile: VideoProfile, role: str) -> list[dict[str, Any]]:
    if profile.family == "ltx":
        names = (
            ["text_encoder.shard_1", "text_encoder.shard_2"]
            if role == "text_encoder"
            else [f"ltx.2b.{profile.release.replace('.', '_')}.distilled"]
        )
    elif role == "text_encoder":
        names = ["text_encoder.umt5_xxl.fp8_scaled"]
    elif role == "vae":
        names = ["vae.wan_2_2" if profile.channels == 48 else "vae.wan_2_1"]
    elif role == "denoiser":
        names = ["denoiser.ti2v_5b.fp16"]
    else:
        names = [f"denoiser.{profile.variant.replace('-', '_')}.{role}.fp8_scaled"]
    return [artifact(name) for name in names]


def component_declaration(profile: VideoProfile, role: str) -> dict[str, Any]:
    return {
        "inventory": profile.components[role],
        "configuration": component_configuration(profile, role),
        "representation": "fp8-e4m3-scaled"
        if profile.family == "wan" and role in ("high_noise", "low_noise", "text_encoder")
        else "native",
        "sources": [
            {
                "repository": a["repository"],
                "revision": a["revision"],
                "artifact": a["path"],
                "sha256": a["sha256_lfs_oid"],
            }
            for a in component_artifacts(profile, role)
        ],
    }


def declaration(profile: VideoProfile) -> dict[str, Any]:
    return {
        "format_version": 1,
        "profile": profile.key,
        "family": profile.family,
        "variant": profile.variant,
        "release": profile.release,
        "modalities": list(profile.modalities),
        "pipeline": "base_single_scale",
        "components": {role: component_declaration(profile, role) for role in profile.components},
    }


class Storage(Protocol):
    @property
    def shape(self) -> Any: ...
    @property
    def dtype(self) -> str: ...


def checked(config: Mapping[str, object], physical: Mapping[str, Storage]) -> VideoProfile:
    metadata = config.get("video_inspection")
    if not isinstance(metadata, dict) or not isinstance(metadata.get("profile"), str):
        raise invalid("Missing video inspection profile.")
    profile = PROFILES.get(metadata["profile"])
    if (
        profile is None
        or config.get("model_type") != "video_inspection"
        or config.get("architectures") != [profile.architecture]
        or metadata != declaration(profile)
        or config.get("auto_map")
    ):
        raise invalid("Unsupported video inspection configuration.")
    expected: dict[str, tuple[str, tuple[int, ...]]] = {}
    layouts: dict[str, str] = {}
    if profile.family == "wan":
        from .video_fp8 import component_layouts

        layouts = component_layouts(config)
        required = {"video.text_encoder"} | (
            {"video.high_noise", "video.low_noise"} if profile.boundary else set()
        )
        if set(layouts) != required:
            raise invalid("Video FP8 component roles disagree with the profile.")
    elif config.get("quantization_config") is not None:
        raise invalid("LTX FP8 is outside the selected profiles.")
    for role, key in profile.components.items():
        prefix = f"video.{role}"
        for name, geometry in inventory(key).items():
            if layouts.get(prefix) == "normalized":
                if name == "scaled_fp8":
                    continue
                name = (
                    name.removesuffix(".scale_weight") + ".weight_scale"
                    if name.endswith(".scale_weight")
                    else name
                )
                name = (
                    name.removesuffix(".scale_input") + ".input_scale"
                    if name.endswith(".scale_input")
                    else name
                )
                # Normalization may omit the optional unit input scale.
                if name.endswith(".input_scale") and prefix + "." + name not in physical:
                    continue
            expected[prefix + "." + name] = geometry
    observed = {name: (record.dtype, tuple(record.shape)) for name, record in physical.items()}
    if observed != expected:
        raise invalid("Video inventory does not match the selected component profiles.")
    return profile

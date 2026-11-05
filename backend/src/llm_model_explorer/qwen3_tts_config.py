"""Bounded Qwen3-TTS Base geometry, checked against the pinned upstream config.

Reference: Qwen/Qwen3-TTS-12Hz-1.7B-Base@fd4b254389122332181a7c3db7f27e918eec64e3.
Only dimension changes are admitted for small fixtures/compatible native geometry;
unreviewed structural options never become ordinary Qwen3 by normalization.
"""

import json
from collections.abc import Mapping
from dataclasses import dataclass
from importlib.resources import files
from typing import Any

from .architecture_analysis.dense_config import DenseConfig
from .architecture_analysis.dense_config import checked as checked_dense

SOURCE_REVISION = "022e286b98fbec7e1e916cb940cdf532cd9f488e"
_REFERENCE = json.loads(
    files("llm_model_explorer").joinpath("reference_configs/qwen3_tts.json").read_text()
)
_DIMENSIONS = {
    "vocab_size",
    "hidden_size",
    "intermediate_size",
    "num_hidden_layers",
    "num_attention_heads",
    "num_key_value_heads",
    "head_dim",
    "max_position_embeddings",
    "text_vocab_size",
    "text_hidden_size",
}
_METADATA = {
    "_name_or_path",
    "name_or_path",
    "_commit_hash",
    "revision",
    "torch_dtype",
    "dtype",
    "transformers_version",
}


@dataclass(frozen=True)
class TTSConfig:
    talker: DenseConfig
    predictor: DenseConfig
    text_vocab: int
    text_hidden: int
    code_groups: int


def _compatible(value: Mapping[str, object], reference: dict[str, Any]) -> bool:
    if set(value) - reference.keys() - _METADATA:
        return False
    for key, expected in reference.items():
        actual = value.get(key)
        if key in _METADATA:
            continue
        if key in _DIMENSIONS:
            if type(actual) is not int or not 0 < actual <= expected:
                return False
        elif key == "layer_types":
            layers = value.get("num_hidden_layers")
            if type(layers) is not int or not isinstance(actual, list):
                return False
            if len(actual) != layers or any(item != "full_attention" for item in actual):
                return False
        elif key == "speaker_encoder_config":
            talker = value.get("talker_config")
            if not isinstance(talker, dict):
                return False
            if actual != {
                "enc_dim": talker.get("hidden_size"),
                "sample_rate": 24000,
            }:
                return False
        elif isinstance(expected, dict) and key in {"talker_config", "code_predictor_config"}:
            if not isinstance(actual, dict) or not _compatible(actual, expected):
                return False
        elif json.dumps(actual, sort_keys=True) != json.dumps(expected, sort_keys=True):
            return False
    return True


def checked(config: Mapping[str, object]) -> TTSConfig | None:
    try:
        if not _compatible(config, _REFERENCE):
            return None
    except (TypeError, ValueError, AttributeError):
        return None
    talker = config["talker_config"]
    assert isinstance(talker, dict)
    predictor = talker["code_predictor_config"]
    stacks = []
    for nested in (talker, predictor):
        dense = {
            key: nested[key]
            for key in _DIMENSIONS - {"text_vocab_size", "text_hidden_size"}
            if key in nested
        }
        dense.update(
            model_type="qwen3",
            hidden_act="silu",
            rms_norm_eps=nested["rms_norm_eps"],
            rope_theta=nested["rope_theta"],
        )
        result = checked_dense(dense)
        if result is None:
            return None
        stacks.append(result)
    # MRoPE section geometry is fixed by the reviewed Base configuration.
    if stacks[0].head_dim != 128:
        return None
    return TTSConfig(
        stacks[0],
        stacks[1],
        talker["text_vocab_size"],
        talker["text_hidden_size"],
        talker["num_code_groups"],
    )


def linear_dimensions(config: TTSConfig) -> dict[str, tuple[int, int]]:
    """Only reviewed attention and gated-MLP targets; no speaker/codec guesses."""
    result = {}
    for prefix, c in (
        ("talker.model", config.talker),
        ("talker.code_predictor.model", config.predictor),
    ):
        for index in range(c.layers):
            base = f"{prefix}.layers.{index}."
            for suffix, dims in {
                "self_attn.q_proj": (c.heads * c.head_dim, c.hidden),
                "self_attn.k_proj": (c.kv_heads * c.head_dim, c.hidden),
                "self_attn.v_proj": (c.kv_heads * c.head_dim, c.hidden),
                "self_attn.o_proj": (c.hidden, c.heads * c.head_dim),
                "mlp.gate_proj": (c.intermediate, c.hidden),
                "mlp.up_proj": (c.intermediate, c.hidden),
                "mlp.down_proj": (c.hidden, c.intermediate),
            }.items():
                result[base + suffix] = dims
    return result

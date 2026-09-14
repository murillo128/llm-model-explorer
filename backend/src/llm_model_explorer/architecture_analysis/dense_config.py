"""Checked configuration subset for the reviewed dense Qwen3 and SmolLM2 path.

Defaults and exclusions are documented in evidence/dense-language-descriptions.md.
This module intentionally imports no model library or checkpoint code.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from math import isfinite
from typing import TypeGuard, cast

SOURCE_REVISION = "753d61104116eefc8ffc977327b441ee0c8d599f"

# Unknown fields are not assumed harmless. These known metadata fields do not
# change the evaluation-mode mathematical graph described here.
METADATA = {
    "model_type",
    "architectures",
    "_name_or_path",
    "name_or_path",
    "_commit_hash",
    "revision",
    "transformers_version",
    "torch_dtype",
    "dtype",
    "initializer_range",
    "bos_token_id",
    "eos_token_id",
    "pad_token_id",
    "use_cache",
    "is_llama_config",
}
OPTIONS = {
    "vocab_size",
    "hidden_size",
    "intermediate_size",
    "num_hidden_layers",
    "num_attention_heads",
    "num_key_value_heads",
    "head_dim",
    "hidden_act",
    "max_position_embeddings",
    "rms_norm_eps",
    "tie_word_embeddings",
    "rope_theta",
    "rope_scaling",
    "attention_bias",
    "attention_dropout",
    "mlp_bias",
    "pretraining_tp",
    "rope_interleaved",
    "use_sliding_window",
    "sliding_window",
    "max_window_layers",
    "layer_types",
    "quantization_config",
}
GPTQ = {
    "bits": 4,
    "checkpoint_format": "gptq",
    "desc_act": False,
    "group_size": 128,
    "lm_head": False,
    "pack_dtype": "int32",
    "quant_method": "gptq",
    "sym": True,
}


@dataclass(frozen=True)
class DenseConfig:
    qwen: bool
    vocab: int
    hidden: int
    intermediate: int
    layers: int
    heads: int
    kv_heads: int
    head_dim: int
    max_positions: int
    epsilon: float
    theta: float
    tied: bool
    attention_bias: bool
    mlp_bias: bool
    quantized: bool


def finite_number(value: object) -> TypeGuard[int | float]:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and isfinite(value)


def checked(config: Mapping[str, object]) -> DenseConfig | None:
    """No fallback for unknown structural options, even on a known model class."""
    if set(config) - METADATA - OPTIONS:
        return None
    qwen = config.get("model_type") == "qwen3"
    if config.get("model_type") not in {"qwen3", "llama"}:
        return None
    if config.get("hidden_act", "silu") != "silu" or config.get("rope_scaling") is not None:
        return None
    if config.get("rope_interleaved", False) is not False:
        return None
    if config.get("use_sliding_window", False) is not False:
        return None
    if config.get("sliding_window") is not None:
        return None
    if type(config.get("pretraining_tp", 1)) is not int or config.get("pretraining_tp", 1) != 1:
        return None
    quant = config.get("quantization_config")
    if quant is not None:
        if not qwen or not isinstance(quant, dict):
            return None
        if set(quant) - GPTQ.keys() - {"meta", "hyb_act"}:
            return None
        if any(type(quant.get(k)) is not type(v) or quant[k] != v for k, v in GPTQ.items()):
            return None
        if quant.get("hyb_act", False) is not False:
            return None
    defaults = {
        "vocab_size": 151936 if qwen else 32000,
        "hidden_size": 4096,
        "intermediate_size": 22016 if qwen else 11008,
        "num_hidden_layers": 32,
        "num_attention_heads": 32,
        "max_position_embeddings": 32768 if qwen else 2048,
    }
    values = {key: config.get(key, value) for key, value in defaults.items()}
    if any(type(v) is not int or not 0 < v <= 9007199254740991 for v in values.values()):
        return None
    # Cast only after strict checks (bool is not an architectural dimension).
    dims = {key: cast(int, value) for key, value in values.items()}
    heads = dims["num_attention_heads"]
    kv = config.get("num_key_value_heads", 32 if qwen else None)
    kv = heads if kv is None else kv
    head = config.get("head_dim", 128 if qwen else None)
    if head is None and not qwen:
        if dims["hidden_size"] % heads:
            return None
        head = dims["hidden_size"] // heads
    if type(kv) is not int or type(head) is not int:
        return None
    if not 0 < kv <= heads or heads % kv or not 0 < head <= 9007199254740991 or head % 2:
        return None
    layers = config.get("layer_types")
    if layers is not None and (
        not qwen
        or not isinstance(layers, list)
        or len(layers) != dims["num_hidden_layers"]
        or any(layer != "full_attention" for layer in layers)
    ):
        return None
    flags = [config.get(k, False) for k in ("tie_word_embeddings", "attention_bias", "mlp_bias")]
    if any(type(v) is not bool for v in flags) or (qwen and flags[2]):
        return None
    epsilon, theta = config.get("rms_norm_eps", 1e-6), config.get("rope_theta", 10000.0)
    if not finite_number(epsilon) or not finite_number(theta) or epsilon <= 0 or theta <= 0:
        return None
    dropout = config.get("attention_dropout", 0.0)
    if not finite_number(dropout) or not 0 <= dropout <= 1:
        return None
    return DenseConfig(
        qwen,
        dims["vocab_size"],
        dims["hidden_size"],
        dims["intermediate_size"],
        dims["num_hidden_layers"],
        heads,
        kv,
        head,
        dims["max_position_embeddings"],
        float(epsilon),
        float(theta),
        cast(bool, flags[0]),
        cast(bool, flags[1]),
        cast(bool, flags[2]),
        quant is not None,
    )

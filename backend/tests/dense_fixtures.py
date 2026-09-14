"""Independent reduced checkpoint fixtures; no imports from dense descriptions.

Geometry and names are transcribed from the reviewed attention/MLP constructors.
The JSON oracle separately retains selected full-reference header metadata.
"""

import json
import math
from pathlib import Path
from typing import Any

from test_models import write_weights
from test_quantized_models import Storage, config_for, write_storage


def small_config(qwen: bool) -> dict[str, Any]:
    return {
        "model_type": "qwen3" if qwen else "llama",
        "architectures": ["Qwen3ForCausalLM" if qwen else "LlamaForCausalLM"],
        "hidden_size": 128 if qwen else 12,
        "intermediate_size": 256 if qwen else 20,
        "num_hidden_layers": 2,
        "num_attention_heads": 4 if qwen else 3,
        "num_key_value_heads": 2 if qwen else 1,
        "head_dim": 64 if qwen else 4,
        "vocab_size": 16,
        "tie_word_embeddings": True,
        "rms_norm_eps": 1e-6 if qwen else 1e-5,
        **({"quantization_config": config_for("JunHowie")["quantization_config"]} if qwen else {}),
    }


def small_storage(qwen: bool, *, biases: bool = False) -> list[Storage]:
    # Asymmetric input/output widths catch transposition and hidden=head-width guesses.
    layer_shapes = {
        "input_layernorm.weight": [128 if qwen else 12],
        "post_attention_layernorm.weight": [128 if qwen else 12],
        "self_attn.q_proj.weight": [256, 128] if qwen else [12, 12],
        "self_attn.k_proj.weight": [128, 128] if qwen else [4, 12],
        "self_attn.v_proj.weight": [128, 128] if qwen else [4, 12],
        "self_attn.o_proj.weight": [128, 256] if qwen else [12, 12],
        "mlp.gate_proj.weight": [256, 128] if qwen else [20, 12],
        "mlp.up_proj.weight": [256, 128] if qwen else [20, 12],
        "mlp.down_proj.weight": [128, 256] if qwen else [12, 20],
        **({"self_attn.q_norm.weight": [64], "self_attn.k_norm.weight": [64]} if qwen else {}),
    }
    storage: list[Storage] = [
        ("model.embed_tokens.weight", "F32", [16, 128 if qwen else 12]),
        ("model.norm.weight", "F32", [128 if qwen else 12]),
    ]
    for layer in (0, 1):
        for suffix, dims in layer_shapes.items():
            name = f"model.layers.{layer}.{suffix}"
            if qwen and len(dims) == 2:
                out, inp = dims
                prefix = name.removesuffix(".weight")
                storage.extend(
                    [
                        (prefix + ".qweight", "I32", [inp // 8, out]),
                        (prefix + ".qzeros", "I32", [inp // 128, out // 8]),
                        (prefix + ".scales", "F16", [inp // 128, out]),
                        (prefix + ".g_idx", "I32", [inp]),
                    ]
                )
            else:
                storage.append((name, "F32", dims))
            if biases and len(dims) == 2:
                storage.append((name.removesuffix(".weight") + ".bias", "F32", [dims[0]]))
    return storage


def local_fixture(root: Path, qwen: bool) -> Path:
    directory = root / "dense"
    directory.mkdir(parents=True)
    (directory / "config.json").write_text(json.dumps(small_config(qwen)))
    storage = small_storage(qwen)
    write_storage(directory / "packed.safetensors", [s for s in storage if s[1] != "F32"])
    # Exact asymmetric values: neither a zero fill nor an implementation-produced oracle.
    write_weights(
        directory / "native.safetensors",
        [
            (name, dtype, dims, [(i % 29 - 14) / 8 for i in range(math.prod(dims))])
            for name, dtype, dims in storage
            if dtype == "F32"
        ],
    )
    (directory / "modeling_custom.py").write_text('raise AssertionError("checkpoint code")')
    return directory

"""Independent reduced CLM/Qwen checkpoint, with explicit native values."""

import importlib.util
import json
from pathlib import Path
from types import ModuleType
from typing import Any

import torch
from safetensors.torch import save_file
from tokenizers import Tokenizer, models, pre_tokenizers  # type: ignore[import-untyped]
from transformers import PreTrainedTokenizerFast


def exporter() -> ModuleType:
    path = Path(__file__).resolve().parents[2] / "examples/clm/export.py"
    spec = importlib.util.spec_from_file_location("clm_inspection_export", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fixture(
    root: Path,
    *,
    activation: str = "gelu",
    layernorm: bool = True,
    residual: bool = False,
    depth: int = 3,
    dtype: torch.dtype = torch.float32,
) -> tuple[Path, Path, dict[str, Any]]:
    encoder = root / "qwen"
    encoder.mkdir(parents=True)
    config = {
        "model_type": "qwen3",
        "architectures": ["Qwen3ForCausalLM"],
        "_name_or_path": "Qwen/Qwen3-8B",
        "hidden_size": 4,
        "intermediate_size": 6,
        "num_hidden_layers": 2,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 2,
        "vocab_size": 8,
        "tie_word_embeddings": False,
        "rms_norm_eps": 1e-6,
    }
    (encoder / "config.json").write_text(json.dumps(config))
    dimensions = {
        "model.embed_tokens.weight": (8, 4),
        "model.norm.weight": (4,),
        "lm_head.weight": (8, 4),
    }
    for i in range(2):
        for name, dims in {
            "input_layernorm.weight": (4,),
            "post_attention_layernorm.weight": (4,),
            "self_attn.q_proj.weight": (4, 4),
            "self_attn.k_proj.weight": (2, 4),
            "self_attn.v_proj.weight": (2, 4),
            "self_attn.o_proj.weight": (4, 4),
            "self_attn.q_norm.weight": (2,),
            "self_attn.k_norm.weight": (2,),
            "mlp.gate_proj.weight": (6, 4),
            "mlp.up_proj.weight": (6, 4),
            "mlp.down_proj.weight": (4, 6),
        }.items():
            dimensions[f"model.layers.{i}.{name}"] = dims
    native = {
        name: ((torch.arange(torch.tensor(dims).prod().item()) % 17 - 8) / 16)
        .reshape(dims)
        .to(torch.bfloat16)
        for name, dims in dimensions.items()
    }
    save_file(native, str(encoder / "encoder.safetensors"))
    tokenizer = Tokenizer(models.WordLevel({"<unk>": 0, "hello": 1, "world": 2}, unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    fast = PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token="<unk>")  # type: ignore[no-untyped-call]
    fast.save_pretrained(encoder)
    (encoder / "modeling_custom.py").write_text('raise AssertionError("checkpoint code")')
    cfg = {
        "model": "Qwen/Qwen3-8B",
        "hidden_size": 4,
        "width": 3,
        "depth": depth,
        "projection_dim": 2,
        "activation": activation,
        "layernorm": layernorm,
        "residual": residual,
    }
    head_shapes = {"inp.weight": (3, 4), "inp.bias": (3,), "out.weight": (2, 3), "out.bias": (2,)}
    for i in range(depth - 2):
        head_shapes.update({f"hidden.{i}.weight": (3, 3), f"hidden.{i}.bias": (3,)})
        if layernorm:
            head_shapes.update({f"norms.{i}.weight": (3,), f"norms.{i}.bias": (3,)})
    payload: dict[str, Any] = {"cfg": cfg, "logit_scale": torch.tensor(5.0)}
    for index, name in enumerate(("state_head", "action_head")):
        payload[name] = {
            key: ((torch.arange(torch.tensor(dims).prod().item()) % 7 - 3 + index) / 8)
            .reshape(dims)
            .to(dtype)
            for key, dims in head_shapes.items()
        }
    head = root.parent / "head.pt"
    torch.save(payload, head)
    return encoder, head, payload

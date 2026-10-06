"""Reduced native hybrid checkpoint; shapes and numerical examples are independent."""

import importlib.util
import json
from pathlib import Path
from types import ModuleType
from typing import Any

import torch
from safetensors.torch import save_file
from tokenizers import Tokenizer, models, pre_tokenizers  # type: ignore[import-untyped]
from transformers import PreTrainedTokenizerFast

BASE_REV = "1" * 40
KEV_REV = "2" * 40
PREFIX = "model.language_model"


def exporter() -> ModuleType:
    path = Path(__file__).resolve().parents[2] / "examples/kev/export.py"
    spec = importlib.util.spec_from_file_location("kev_export", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fixture(root: Path, *, repeated: int = 0) -> tuple[Path, Path, dict[str, Any]]:
    base, kev = root / "base", root.parent / "upstream-kev"
    base.mkdir(parents=True)
    kev.mkdir()
    config: dict[str, Any] = {
        "model_type": "qwen3_5",
        "architectures": ["Qwen3_5ForConditionalGeneration"],
        "_name_or_path": "Qwen/Qwen3.5-0.8B-Base",
        "_commit_hash": BASE_REV,
        "tie_word_embeddings": True,
        "vision_config": {"out_hidden_size": 4},
        "text_config": {
            "model_type": "qwen3_5_text",
            "hidden_size": 4,
            "intermediate_size": 6,
            "vocab_size": 8,
            "num_hidden_layers": 2,
            "num_attention_heads": 1,
            "num_key_value_heads": 1,
            "head_dim": 8,
            "linear_conv_kernel_dim": 4,
            "linear_key_head_dim": 2,
            "linear_value_head_dim": 2,
            "linear_num_key_heads": 1,
            "linear_num_value_heads": 1,
            "layer_types": ["linear_attention", "full_attention"],
            "attention_bias": False,
            "attention_dropout": 0.0,
            "attn_output_gate": True,
            "hidden_act": "silu",
            "mlp_only_layers": [],
            "mamba_ssm_dtype": "float32",
            "mtp_num_hidden_layers": 1,
            "mtp_use_dedicated_embeddings": False,
            "tie_word_embeddings": True,
            "rms_norm_eps": 1e-6,
            "rope_parameters": {
                "rope_type": "default",
                "rope_theta": 10000000,
                "partial_rotary_factor": 0.25,
                "mrope_interleaved": True,
                "mrope_section": [1, 0, 0],
            },
        },
    }
    if repeated:
        config["text_config"]["num_hidden_layers"] = 4 * repeated
        config["text_config"]["layer_types"] = (
            ["linear_attention"] * 3 + ["full_attention"]
        ) * repeated
    (base / "config.json").write_text(json.dumps(config))
    dimensions = {PREFIX + ".embed_tokens.weight": (8, 4), PREFIX + ".norm.weight": (4,)}
    for i in range(2):
        p = PREFIX + f".layers.{i}."
        dimensions.update(
            {
                p + k: v
                for k, v in {
                    "input_layernorm.weight": (4,),
                    "post_attention_layernorm.weight": (4,),
                    "mlp.gate_proj.weight": (6, 4),
                    "mlp.up_proj.weight": (6, 4),
                    "mlp.down_proj.weight": (4, 6),
                }.items()
            }
        )
    dimensions.update(
        {
            PREFIX + ".layers.0.linear_attn." + k: v
            for k, v in {
                "in_proj_qkv.weight": (6, 4),
                "in_proj_z.weight": (2, 4),
                "in_proj_a.weight": (1, 4),
                "in_proj_b.weight": (1, 4),
                "out_proj.weight": (4, 2),
                "conv1d.weight": (6, 1, 4),
                "A_log": (1,),
                "dt_bias": (1,),
                "norm.weight": (2,),
            }.items()
        }
    )
    dimensions.update(
        {
            PREFIX + ".layers.1.self_attn." + k: v
            for k, v in {
                "q_proj.weight": (16, 4),
                "k_proj.weight": (8, 4),
                "v_proj.weight": (8, 4),
                "o_proj.weight": (4, 8),
                "q_norm.weight": (8,),
                "k_norm.weight": (8,),
            }.items()
        }
    )
    if repeated:
        original = dict(dimensions)
        dimensions = {k: v for k, v in original.items() if ".layers." not in k}
        for index, source in enumerate([0, 0, 0, 1] * repeated):
            dimensions.update(
                {
                    k.replace(f".layers.{source}.", f".layers.{index}."): v
                    for k, v in original.items()
                    if f".layers.{source}." in k
                }
            )
    native = {
        name: ((torch.arange(math_product(dims)) % 17 - 8) / 16).reshape(dims).to(torch.bfloat16)
        for name, dims in dimensions.items()
    }
    w = torch.zeros(6, 4, dtype=torch.bfloat16)
    w[0, 0], w[1, 1] = 1, 1
    native[PREFIX + ".layers.0.mlp.up_proj.weight"] = w
    save_file(native, str(base / "model.safetensors"))
    vocab = {
        "<unk>": 0,
        "hello": 1,
        "world": 2,
        "<|fim_prefix|>": 3,
        "<|fim_middle|>": 4,
        "<|box_start|>": 5,
        "<|box_end|>": 6,
        "<|fim_suffix|>": 7,
    }
    tokenizer = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    fast = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer, unk_token="<unk>", additional_special_tokens=list(vocab)[3:]
    )  # type: ignore[no-untyped-call]
    fast.save_pretrained(base)
    adapter = {
        "peft_type": "LORA",
        "task_type": "FEATURE_EXTRACTION",
        "base_model_name_or_path": "Qwen/Qwen3.5-0.8B-Base",
        "revision": BASE_REV,
        "r": 2,
        "lora_alpha": 4,
        "bias": "none",
        "fan_in_fan_out": False,
        "target_modules": ["up_proj", "in_proj_a", "q_proj"],
    }
    (kev / "adapter_config.json").write_text(json.dumps(adapter))
    factors = {}
    for name, out in (
        ("layers.0.mlp.up_proj", 6),
        ("layers.1.mlp.up_proj", 6),
        ("layers.0.linear_attn.in_proj_a", 1),
        ("layers.1.self_attn.q_proj", 16),
    ):
        a = torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
        b = torch.zeros(out, 2)
        b[0, 0] = 3
        if out > 1:
            b[1, 1] = 4
        factors["base_model.model." + name + ".lora_A.weight"] = a
        factors["base_model.model." + name + ".lora_B.weight"] = b
    if repeated:
        original_factors = factors
        factors = {}
        for index, source in enumerate([0, 0, 0, 1] * repeated):
            factors.update(
                {
                    k.replace(f".layers.{source}.", f".layers.{index}."): v.clone()
                    for k, v in original_factors.items()
                    if f".layers.{source}." in k
                }
            )
    save_file(factors, str(kev / "adapter_model.safetensors"))
    payload = {
        "base": "Qwen/Qwen3.5-0.8B-Base",
        "base_revision": BASE_REV,
        "head_dim": 4,
        "lora": 2,
        "option_isolation": False,
        "special_embeddings": False,
        "lora_placement": "full",
        "weights_dtype": "fp32",
        "temperature": 2.0,
        "head": {
            "q.weight": torch.eye(4),
            "q.bias": torch.tensor([1.0, -1.0, 0.0, 0.0]),
            "k.weight": torch.eye(4),
            "k.bias": torch.tensor([0.0, 1.0, 0.0, 0.0]),
        },
    }
    torch.save(payload, kev / "head.pt")
    return base, kev, payload


def math_product(dims: tuple[int, ...]) -> int:
    result = 1
    for dim in dims:
        result *= dim
    return result

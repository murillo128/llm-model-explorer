"""Deterministic local HF assets; generated only in an operator/test temporary root."""

import json
from pathlib import Path

import torch
from safetensors.torch import save_file
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, processors
from transformers import PreTrainedTokenizerFast

SEED = 17
MODEL_ID = "acceptance/fixture"
SHAPES = {
    "model.norm.weight": (576,),
    "model.layers.0.mlp.down_proj.weight": (576, 1536),
    "model.layers.0.mlp.up_proj.weight": (1536, 576),
    "model.embed_tokens.weight": (1025, 576),
}
TEXT = "A😀e\u0301<special>\n  café 你好"


def values(shape: tuple[int, ...]) -> torch.Tensor:
    count = 1
    for dimension in shape:
        count *= dimension
    # Exactly representable F16/F32 values, asymmetric under transpose.
    return ((torch.arange(count) * SEED % 257 - 128) / 128).reshape(shape)


def generate(root: Path) -> Path:
    directory = root / "fixture"
    directory.mkdir(parents=True)
    (directory / "config.json").write_text(
        json.dumps(
            {
                "model_type": "llama",
                "_name_or_path": MODEL_ID,
                "architectures": ["LlamaForCausalLM"],
            }
        )
    )
    tensors = {name: values(shape).half() for name, shape in SHAPES.items()}
    tensors["science.weight"] = torch.tensor(
        [
            [-1.0, 0.0, 1.0, 2.0],
            [3.0, 4.0, 5.0, 6.0],
            [float("nan"), float("inf"), -0.0, -2.0],
        ]
    )
    tensors["empty.weight"] = torch.empty(0, 4)
    save_file(tensors, directory / "model.safetensors")
    # Explicit byte alphabet and no training remove randomized trainer ordering.
    vocab = {
        word: i
        for i, word in enumerate(
            ["<unk>", "<bos>", "<special>"] + sorted(pre_tokenizers.ByteLevel.alphabet())
        )
    }
    native = Tokenizer(models.BPE(vocab=vocab, merges=[], unk_token="<unk>"))
    native.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    native.decoder = decoders.ByteLevel()
    native.post_processor = processors.TemplateProcessing(
        single="<bos> $A",
        special_tokens=[("<bos>", 1)],
    )
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=native,
        unk_token="<unk>",
        bos_token="<bos>",
        additional_special_tokens=["<special>"],
    )
    tokenizer.save_pretrained(directory)
    return directory

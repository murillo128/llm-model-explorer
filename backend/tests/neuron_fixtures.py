"""Reusable local numerical fixtures; never production admission evidence."""

import json
import math
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch
from tokenizers import Tokenizer, models  # type: ignore[import-untyped]
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

from llm_model_explorer.models import ModelCatalogue
from llm_model_explorer.neuron_execution import AdmittedModel, NeuronInput, admission
from llm_model_explorer.settings import Settings
from llm_model_explorer.tensor_source import ModelSource

TOKENS = (4, 2, 4)


@pytest.fixture
def tiny(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[tuple[Settings, Path, Any]]:
    root = tmp_path / "models"
    directory = root / "SmolLM2-135M"
    directory.mkdir(parents=True)
    config = dict(
        admission.REFERENCE,
        vocab_size=32,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        max_position_embeddings=32,
    )
    monkeypatch.setattr(admission, "REFERENCE", config)
    with torch.random.fork_rng():
        torch.manual_seed(173)
        reference: Any = LlamaForCausalLM(  # type: ignore[no-untyped-call]
            LlamaConfig(**config, attn_implementation="eager")  # type: ignore[no-untyped-call]
        )
        reference.eval()
    reference.save_pretrained(directory, safe_serialization=True)
    (directory / "config.json").write_text(
        json.dumps(
            dict(
                config,
                _name_or_path=admission.REPOSITORY,
                _commit_hash=admission.REVISION,
            )
        )
    )
    vocabulary = {f"token-{i}": i for i in range(32)}
    tokenizer = PreTrainedTokenizerFast(  # type: ignore[no-untyped-call]
        tokenizer_object=Tokenizer(models.WordLevel(vocabulary, unk_token="token-0")),
        unk_token="token-0",
        eos_token="token-0",
    )
    tokenizer.save_pretrained(directory)
    yield Settings(model_root=root, cache_dir=tmp_path / "cache"), directory, reference


def source_for(settings: Settings) -> ModelSource:
    catalogue = ModelCatalogue(settings.model_root)
    return catalogue.discover()[0].pin()


def request(model: AdmittedModel) -> NeuronInput:
    return NeuronInput(
        model, model.source.model_revision, model.modules[0]["module_id"], (1, 0), TOKENS
    )


def direct(reference: Any, token_ids: tuple[int, ...] = TOKENS) -> torch.Tensor:
    with torch.inference_mode():
        return cast(torch.Tensor, reference(torch.tensor([token_ids]), use_cache=False).logits)


def reference_activations(reference: Any, token_ids: tuple[int, ...], layer: int) -> torch.Tensor:
    """Independent of the subject's down-projection hook and loader."""
    activations = []
    mlp = reference.model.layers[layer].mlp

    def oracle(_module: Any, args: Any) -> None:
        x = args[0]
        activations.append(
            torch.nn.functional.silu(torch.nn.functional.linear(x, mlp.gate_proj.weight))
            * torch.nn.functional.linear(x, mlp.up_proj.weight)
        )

    handle = mlp.register_forward_pre_hook(oracle)
    try:
        direct(reference, token_ids)
    finally:
        handle.remove()
    assert len(activations) == 1
    return activations[0]


class KnownMLP(torch.nn.Module):
    """Two distinguishable layers with a hand-computable second MLP.

    Other coordinates are zero. The first two input coordinates select small
    integer dot products; the oracle uses Python math rather than PyTorch SiLU.
    """

    def __init__(self) -> None:
        super().__init__()
        self.gate_proj = torch.nn.Linear(8, 16, bias=False)
        self.up_proj = torch.nn.Linear(8, 16, bias=False)
        self.down_proj = torch.nn.Linear(16, 8, bias=False)
        with torch.no_grad():
            for parameter in self.parameters():
                parameter.zero_()
            for index, gate, up in self.rows():
                self.gate_proj.weight[index, :2] = torch.tensor(gate)
                self.up_proj.weight[index, :2] = torch.tensor(up)
            self.down_proj.weight[:, :8] = torch.eye(8)

    @staticmethod
    def rows() -> tuple[tuple[int, tuple[int, int], tuple[int, int]], ...]:
        return ((1, (1, 0), (0, 1)), (7, (-1, 1), (1, 1)), (15, (0, 1), (-1, 0)))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return cast(
            torch.Tensor,
            self.down_proj(torch.nn.functional.silu(self.gate_proj(values)) * self.up_proj(values)),
        )


class KnownModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.first, self.second = KnownMLP(), KnownMLP()
        self.layers = (self.first, self.second)
        self.model = SimpleNamespace(layers=[SimpleNamespace(mlp=m) for m in self.layers])
        self.inputs = torch.tensor([[0, 0], [1, -2], [-1, 1]], dtype=torch.float32)
        with torch.no_grad():
            self.layers[0].up_proj.weight.zero_()
        self.eval()

    def forward(self, input_ids: torch.Tensor, **kwargs: Any) -> Any:
        x = torch.nn.functional.pad(self.inputs[input_ids], (0, 6))
        # Both concrete seams execute, with deliberately different values.
        self.layers[0](x)
        return SimpleNamespace(logits=self.layers[1](x))


def known_activations(token_ids: tuple[int, ...], neurons: tuple[int, ...]) -> torch.Tensor:
    inputs = ((0, 0), (1, -2), (-1, 1))
    rows = {i: (gate, up) for i, gate, up in KnownMLP.rows()}
    result = []
    for token in token_ids:
        x = inputs[token]
        values = []
        for neuron in neurons:
            gate, up = rows.get(neuron, ((0, 0), (0, 0)))
            g = sum(a * b for a, b in zip(x, gate, strict=True))
            u = sum(a * b for a, b in zip(x, up, strict=True))
            values.append(g / (1 + math.exp(-g)) * u)
        result.append(values)
    return torch.tensor(result, dtype=torch.float32)

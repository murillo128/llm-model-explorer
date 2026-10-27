"""Fixed native SmolLM2 admission; metadata inspection never constructs a model."""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import transformers

from ..model_files import ModelError, changed
from ..tensor_source import ModelSource, TensorDescriptor
from ..tokenization import load_tokenizer

ADAPTER_VERSION = "1"
TRANSFORMERS_VERSION = "4.57.6"
REPOSITORY = "HuggingFaceTB/SmolLM2-135M"
REVISION = "93efa2f097d58c2a74874c7e644dbc9b0cee75a2"
# Reviewed reference metadata, independent of architecture descriptions/sidecars.
REFERENCE: dict[str, Any] = {
    "architectures": ["LlamaForCausalLM"],
    "model_type": "llama",
    "vocab_size": 49152,
    "hidden_size": 576,
    "intermediate_size": 1536,
    "num_hidden_layers": 30,
    "num_attention_heads": 9,
    "num_key_value_heads": 3,
    "max_position_embeddings": 8192,
    "hidden_act": "silu",
    "rms_norm_eps": 1e-5,
    "rope_theta": 100000,
    "tie_word_embeddings": True,
    "bos_token_id": 0,
    "eos_token_id": 0,
}
OPTIONAL = {
    "attention_bias": False,
    "mlp_bias": False,
    "attention_dropout": 0.0,
    "pretraining_tp": 1,
    "rope_scaling": None,
    "rope_interleaved": False,
    "pad_token_id": None,
}
METADATA = {
    "_name_or_path",
    "name_or_path",
    "_commit_hash",
    "revision",
    "transformers_version",
    "torch_dtype",
    "dtype",
    "use_cache",
    "initializer_range",
    "is_llama_config",
    "head_dim",
}


class Unavailable(Exception):
    def __init__(self, code: str, message: str, recovery: str) -> None:
        self.feature: dict[str, Any] = dict(
            status="unavailable", reason=dict(code=code, message=message, recovery=recovery)
        )


def unsupported() -> Unavailable:
    return Unavailable(
        "unsupported_model",
        "This checkpoint is not the reviewed SmolLM2-135M Base model.",
        "Select the complete local native SmolLM2-135M Base checkpoint.",
    )


def runtime_identity(device: str) -> dict[str, str]:
    # Fail closed if a different build replaces the reviewed implementation.
    if transformers.__version__ != TRANSFORMERS_VERSION:
        raise Unavailable(
            "runtime_unavailable",
            "The installed execution adapter is unsupported.",
            "Install the backend's locked runtime dependencies.",
        )
    if device != "cpu" and (not torch.cuda.is_available() or torch.backends.cuda.matmul.allow_tf32):
        raise Unavailable(
            "runtime_unavailable",
            "The configured float32 execution device is unavailable.",
            "Configure an available device with float32 matrix precision.",
        )
    return dict(
        family="smollm2",
        adapter="smollm2_native",
        adapter_version=ADAPTER_VERSION,
        runtime_version=transformers.__version__,
        torch_version=str(torch.__version__),
        device=device,
        dtype="float32",
    )


def checked_config(source: ModelSource) -> dict[str, Any]:
    config = source.configuration()
    # Local provenance is a declaration, not an upstream cryptographic attestation.
    # A display-name match alone never bypasses the full configuration/inventory checks.
    identity = source.model_id.split("@")[0]
    if identity not in {REPOSITORY, "SmolLM2-135M"}:
        raise unsupported()
    for key in ("_name_or_path", "name_or_path"):
        if key in config and config[key] != REPOSITORY:
            raise unsupported()
    for key in ("_commit_hash", "revision"):
        if key in config and config[key] != REVISION:
            raise unsupported()
    if set(config) - REFERENCE.keys() - OPTIONAL.keys() - METADATA:
        raise unsupported()
    for key, value in REFERENCE.items():
        actual = config.get(key)
        if actual != value or isinstance(actual, bool) != isinstance(value, bool):
            raise unsupported()
        if isinstance(value, int) and not isinstance(value, bool) and type(actual) is not int:
            raise unsupported()
    for key, value in OPTIONAL.items():
        actual = config.get(key, value)
        if actual != value or isinstance(actual, bool) != isinstance(value, bool):
            raise unsupported()
    head = REFERENCE["hidden_size"] // REFERENCE["num_attention_heads"]
    if config.get("head_dim", head) != head:
        raise unsupported()
    # Never pass arbitrary checkpoint kwargs, implementation names or code to HF.
    return dict(REFERENCE, **OPTIONAL, head_dim=head, use_cache=False, attn_implementation="eager")


def expected_weights(config: dict[str, Any]) -> dict[str, tuple[int, ...]]:
    hidden, width = config["hidden_size"], config["intermediate_size"]
    kv = config["num_key_value_heads"] * config["head_dim"]
    result = {
        "model.embed_tokens.weight": (config["vocab_size"], hidden),
        "model.norm.weight": (hidden,),
    }
    for i in range(config["num_hidden_layers"]):
        prefix = f"model.layers.{i}."
        result.update(
            {
                prefix + name + ".weight": shape
                for name, shape in {
                    "input_layernorm": (hidden,),
                    "post_attention_layernorm": (hidden,),
                    "self_attn.q_proj": (hidden, hidden),
                    "self_attn.k_proj": (kv, hidden),
                    "self_attn.v_proj": (kv, hidden),
                    "self_attn.o_proj": (hidden, hidden),
                    "mlp.gate_proj": (width, hidden),
                    "mlp.up_proj": (width, hidden),
                    "mlp.down_proj": (hidden, width),
                }.items()
            }
        )
    return result


@dataclass(frozen=True)
class AdmittedModel:
    source: ModelSource
    config: dict[str, Any]
    tensors: dict[str, TensorDescriptor]
    modules: tuple[dict[str, Any], ...]
    execution: dict[str, str]
    limits: dict[str, Any]

    @property
    def key(self) -> tuple[str, str, str, str]:
        return (
            self.source.fingerprint,
            self.source.model_revision,
            json.dumps(self.config, sort_keys=True),
            json.dumps(self.execution, sort_keys=True),
        )

    def validate_tokens(self, token_ids: tuple[int, ...], *, continuation: bool = False) -> None:
        maximum = self.limits["max_input_tokens"]
        if continuation:
            maximum = min(self.limits["context_length"], maximum + self.limits["max_new_tokens"])
        if not token_ids or any(
            type(i) is not int or not 0 <= i < self.config["vocab_size"] for i in token_ids
        ):
            raise ModelError("validation_error", "Invalid input token IDs.")
        if len(token_ids) > maximum:
            raise ModelError("unsupported_size", "Input exceeds the execution context limit.")


@dataclass(frozen=True)
class NeuronInput:
    """Common typed preflight for the operation-specific capture/comparison services.

    Comparison adds its condition and continuation bounds before calling run.
    This common selection is checked again before any runtime load/allocation.
    """

    model: AdmittedModel
    model_revision: str
    module_id: str
    neuron_indices: tuple[int, ...]
    token_ids: tuple[int, ...]

    def validate(self) -> None:
        if self.model_revision != self.model.source.model_revision:
            raise changed()
        self.model.validate_tokens(self.token_ids)
        module = next((m for m in self.model.modules if m["module_id"] == self.module_id), None)
        if (
            module is None
            or not self.neuron_indices
            or any(
                type(i) is not int or not 0 <= i < module["intermediate_size"]
                for i in self.neuron_indices
            )
            or len(set(self.neuron_indices)) != len(self.neuron_indices)
        ):
            raise ModelError("validation_error", "Invalid neuron selection.")
        if len(self.neuron_indices) > self.model.limits["max_neurons"]:
            raise ModelError("unsupported_size", "Neuron selection exceeds the supported limit.")


def admit(source: ModelSource, root: Path, device: str) -> AdmittedModel:
    source.check_unchanged()
    if source.lora_composition is not None or any(
        t.storage_format != "safetensors" for t in source.tensors()
    ):
        raise Unavailable(
            "unsupported_representation",
            "Neuron execution requires uncomposed native weights.",
            "Select native SmolLM2-135M Base Safetensors weights.",
        )
    config = checked_config(source)
    expected = expected_weights(config)
    physical = {t.name: t for t in source.physical_tensors()}
    tensors = {t.name: t for t in source.tensors()}
    if (
        set(physical) != set(expected)
        or set(tensors) != set(expected)
        or any(
            physical[name].shape != shape
            or physical[name].dtype not in {"BF16", "F16", "F32"}
            or tensors[name].shape != shape
            for name, shape in expected.items()
        )
    ):
        raise Unavailable(
            "unsupported_representation",
            "The native checkpoint inventory is incompatible.",
            "Provide every original native weight with its reviewed shape and dtype.",
        )
    execution = runtime_identity(device)
    try:
        with source.local_directory() as directory:
            tokenizer = load_tokenizer(root, directory)
            vocabulary = tokenizer.get_vocab()
            if (
                not tokenizer.is_fast
                or len(vocabulary) != config["vocab_size"]
                or set(vocabulary.values()) != set(range(config["vocab_size"]))
            ):
                raise ModelError("unsupported_representation", "Incompatible local tokenizer.")
    except ModelError as exc:
        if exc.code == "model_content_changed":
            raise
        raise Unavailable(
            "tokenizer_unavailable",
            "The local tokenizer is missing or incompatible.",
            "Restore the complete checkpoint's local tokenizer assets.",
        ) from None
    modules = []
    for layer in range(config["num_hidden_layers"]):
        name = f"model.layers.{layer}.mlp"
        modules.append(
            dict(
                module_id=hashlib.sha256(("smollm2_native:" + name).encode()).hexdigest(),
                module=dict(kind="module", name=name),
                layer_index=layer,
                intermediate_size=config["intermediate_size"],
                hidden_size=config["hidden_size"],
                point="post_activation_product_pre_down_proj",
                weights={
                    role: dict(
                        tensor_id=tensors[f"{name}.{role}_proj.weight"].id,
                        axis="column" if role == "down" else "row",
                        shape=list(tensors[f"{name}.{role}_proj.weight"].shape),
                    )
                    for role in ("gate", "up", "down")
                },
            )
        )
    context, vocabulary_size = config["max_position_embeddings"], config["vocab_size"]
    limits = dict(
        max_modules_per_request=1,
        max_neurons=8,
        max_input_tokens=min(512, context),
        max_new_tokens=min(64, context - 1),
        max_conditions=4,
        max_top_k=min(20, vocabulary_size),
        context_length=context,
        vocabulary_size=vocabulary_size,
        eos_token_ids=[config["eos_token_id"]],
    )
    source.check_unchanged()
    return AdmittedModel(source, config, tensors, tuple(modules), execution, limits)

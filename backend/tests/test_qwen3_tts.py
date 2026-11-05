"""Independent small geometry from the pinned Qwen3-TTS constructors.

Test plan: distinguish text/codec rows; bind both nested stacks and exact LoRA
factors; verify concrete graph instances/branches and honest partial coverage;
reject unsupported config and malformed factors. Existing PEFT tests own the
shared confinement, fingerprints and mutation lifecycle.
"""

import copy
import json
import math
from pathlib import Path
from typing import Any

import pytest
import torch
from architecture_assertions import semantic_key
from test_models import Tensor, write_weights

from llm_model_explorer.architecture_analysis import AnalysisInput
from llm_model_explorer.architecture_service import packaged_registry
from llm_model_explorer.embeddings import resolve_input_table
from llm_model_explorer.model_files import ModelError
from llm_model_explorer.models import ModelCatalogue
from llm_model_explorer.qwen3_tts_config import checked

BASE = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
REFERENCE = Path(__file__).parents[1] / "src/llm_model_explorer/reference_configs/qwen3_tts.json"


def small_config() -> dict[str, Any]:
    # Source configuration is retained verbatim; dimensions below are intentionally
    # asymmetric so a text/codec or Talker/predictor swap cannot pass.
    config: dict[str, Any] = json.loads(REFERENCE.read_text())
    config["_name_or_path"] = BASE
    talker = config["talker_config"]
    talker.update(
        hidden_size=8,
        intermediate_size=12,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        text_vocab_size=23,
        text_hidden_size=6,
        vocab_size=17,
    )
    config["speaker_encoder_config"]["enc_dim"] = 8
    talker["code_predictor_config"].update(
        hidden_size=4,
        intermediate_size=10,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        vocab_size=11,
        layer_types=["full_attention"],
    )
    return config


def fixture(root: Path, *, bad_factor: bool = False, omit_text: bool = False) -> None:
    base, adapter = root / "base", root / "adapter"
    base.mkdir()
    adapter.mkdir()
    (base / "config.json").write_text(json.dumps(small_config()))
    tensors: list[Tensor] = []

    def weight(name: str, dims: list[int], value: float = 0.25) -> None:
        tensors.append((name, "F32", dims, [value] * math.prod(dims)))

    if not omit_text:
        tensors.append(
            ("talker.model.text_embedding.weight", "F32", [23, 6], [float(i) for i in range(138)])
        )
    weight("talker.model.codec_embedding.weight", [17, 8], -2)
    weight("talker.codec_head.weight", [17, 8])
    for key, dims in (("linear_fc1", [6, 6]), ("linear_fc2", [8, 6])):
        weight("talker.text_projection." + key + ".weight", dims)
        weight("talker.text_projection." + key + ".bias", [dims[0]])
    weight("talker.code_predictor.small_to_mtp_projection.weight", [4, 8])
    weight("talker.code_predictor.small_to_mtp_projection.bias", [4])
    weight("speaker_encoder.blocks.0.conv.weight", [8, 3, 3])
    for i in range(15):
        weight(f"talker.code_predictor.model.codec_embedding.{i}.weight", [11, 8])
        weight(f"talker.code_predictor.lm_head.{i}.weight", [11, 4])
    factors: list[Tensor] = []
    for prefix, hidden, intermediate, layers in (
        ("talker.model", 8, 12, 2),
        ("talker.code_predictor.model", 4, 10, 1),
    ):
        weight(prefix + ".norm.weight", [hidden])
        for index in range(layers):
            layer = f"{prefix}.layers.{index}"
            for name, dims in {
                "input_layernorm.weight": [hidden],
                "post_attention_layernorm.weight": [hidden],
                "self_attn.q_norm.weight": [128],
                "self_attn.k_norm.weight": [128],
                "self_attn.q_proj.weight": [256, hidden],
                "self_attn.k_proj.weight": [128, hidden],
                "self_attn.v_proj.weight": [128, hidden],
                "self_attn.o_proj.weight": [hidden, 256],
                "mlp.gate_proj.weight": [intermediate, hidden],
                "mlp.up_proj.weight": [intermediate, hidden],
                "mlp.down_proj.weight": [hidden, intermediate],
            }.items():
                weight(layer + "." + name, dims)
            module = "base_model.model." + layer + ".self_attn.q_proj"
            rank = 3 if bad_factor else 2
            factors.extend(
                [
                    (module + ".lora_A.weight", "F32", [rank, hidden], [0.5] * rank * hidden),
                    (module + ".lora_B.weight", "F32", [256, 2], [0.75] * 512),
                ]
            )
    write_weights(base / "model.safetensors", tensors)
    (adapter / "adapter_config.json").write_text(
        json.dumps(
            {
                "base_model_name_or_path": BASE,
                "peft_type": "LORA",
                "task_type": "CAUSAL_LM",
                "r": 2,
                "lora_alpha": 4,
                "target_modules": ["q_proj"],
                "bias": "none",
                "fan_in_fan_out": False,
            }
        )
    )
    write_weights(adapter / "adapter_model.safetensors", factors)


def test_base_and_lora_keep_text_rows_native_values_and_concrete_graphs(tmp_path: Path) -> None:
    fixture(tmp_path)
    catalogue = ModelCatalogue(tmp_path)
    assert len(catalogue.list_models()) == 2
    assert not catalogue.list_catalogue().diagnostics
    for summary in catalogue.list_models():
        source = catalogue.pin(summary.id)
        table = resolve_input_table(source)
        assert table.name == "talker.model.text_embedding.weight"
        assert table.shape == (23, 6)
        assert torch.cat(list(source.iter_rows(table.id, (2, 0, 2)))).tolist() == [
            12.0,
            13.0,
            14.0,
            15.0,
            16.0,
            17.0,
            0.0,
            1.0,
            2.0,
            3.0,
            4.0,
            5.0,
            12.0,
            13.0,
            14.0,
            15.0,
            16.0,
            17.0,
        ]
        native = next(
            t for t in source.tensors() if t.name == "talker.model.layers.0.self_attn.q_proj.weight"
        )
        assert torch.cat(list(source.iter_tensor(native.id))).tolist() == [0.25] * 2048
        result = packaged_registry().analyze(
            AnalysisInput.from_source(source, tokenizer_available=False)
        )
        assert result.status == "partial", result
        graph = result.graph
        assert graph is not None
        assert {d.code for d in graph.diagnostics} == {"speech_pipeline_context"}
        assert sorted(len(rep.instances) for rep in graph.repetitions) == [1, 2]
        nodes = {semantic_key(node): node for node in graph.nodes}
        assert "talker.model.layers.1.self_attn.q_norm" in nodes
        assert "talker.code_predictor.model.layers.0.self_attn.q_norm" in nodes
        parameters = {p.name: p for p in graph.parameters}
        for name in (
            "talker.text_projection.linear_fc1.bias",
            "talker.code_predictor.small_to_mtp_projection.bias",
            "talker.code_predictor.lm_head.14.weight",
        ):
            assert parameters[name].inspection.status == "available"
        assert parameters["speaker_encoder.blocks.0.conv.weight"].inspection.status == "unavailable"
        lora_nodes = [node for key, node in nodes.items() if key.endswith(".lora_add")]
        assert len(lora_nodes) == (3 if "+peft-lora:" in summary.id else 0)
        if source.lora_composition:
            assert len(source.lora_composition.targets) == 3
            for target in source.lora_composition.targets:
                assert target.module_name + ".lora_add" in nodes
                factor = next(t for t in source.tensors() if t.name == target.a_tensor_name)
                assert set(torch.cat(list(source.iter_tensor(factor.id))).tolist()) == {0.5}
                assert parameters[target.a_tensor_name].inspection.tensor_id == factor.id  # type: ignore[union-attr]


def test_missing_text_table_never_falls_back_to_audio(tmp_path: Path) -> None:
    fixture(tmp_path, omit_text=True)
    source = ModelCatalogue(tmp_path).pin(BASE)
    with pytest.raises(ModelError, match="input embedding"):
        resolve_input_table(source)


def test_invalid_lora_geometry_is_diagnostic_not_selectable(tmp_path: Path) -> None:
    fixture(tmp_path, bad_factor=True)
    listing = ModelCatalogue(tmp_path).list_catalogue()
    assert [m.id for m in listing.models] == [BASE]
    assert len(listing.diagnostics) == 1
    assert "dimensions" in listing.diagnostics[0].message


def test_lora_cannot_validate_a_base_with_wrong_config_geometry(tmp_path: Path) -> None:
    fixture(tmp_path)
    path = tmp_path / "base/config.json"
    config = json.loads(path.read_text())
    config["talker_config"]["num_attention_heads"] = 1
    path.write_text(json.dumps(config))
    listing = ModelCatalogue(tmp_path).list_catalogue()
    assert [m.id for m in listing.models] == [BASE]
    assert len(listing.diagnostics) == 1
    assert "dimensions" in listing.diagnostics[0].message


@pytest.mark.parametrize("mutation", ["unknown", "rope", "layers", "huge", "custom_voice", "bool"])
def test_unreviewed_structural_options_fail_closed(mutation: str) -> None:
    config = copy.deepcopy(small_config())
    if mutation == "unknown":
        config["talker_config"]["new_attention_mode"] = True
    elif mutation == "rope":
        config["talker_config"]["rope_scaling"]["interleaved"] = False
    elif mutation == "layers":
        config["talker_config"]["code_predictor_config"]["layer_types"] = ["sliding_attention"]
    elif mutation == "huge":
        config["talker_config"]["num_hidden_layers"] = 2**40
    elif mutation == "custom_voice":
        config["tts_model_type"] = "custom_voice"
    else:
        config["talker_config"]["hidden_size"] = True
    assert checked(config) is None

"""Source-backed structural oracles for the bounded GLM-4.7-Flash description."""

from __future__ import annotations

import builtins
import copy
import json
import re
import socket
from dataclasses import replace
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import pytest
import torch

from llm_model_explorer.architecture_analysis import (
    AnalysisInput,
    BindingContext,
    DescriptionRegistry,
    GraphBuilder,
    NumericTensor,
)
from llm_model_explorer.architecture_analysis import records as r
from llm_model_explorer.architecture_analysis.glm4_moe_lite import (
    COMPRESSED_FORMAT,
    PRODUCER,
    SOURCE_REVISION,
    checked,
    register_glm4_moe_lite,
    supports,
)
from llm_model_explorer.architecture_analysis.validation import MAX_BYTES, serialized_size
from llm_model_explorer.tensor_source import ModelSource

FIXTURE = json.loads((Path(__file__).parent / "fixtures/glm4-moe-lite-reference.json").read_text())
REFERENCE = FIXTURE["configuration"]
EXPERT_REGION = re.compile(r"\.mlp\.experts\.\d+\.(gate_proj|up_proj|down_proj)\.weight$")


def registry() -> DescriptionRegistry:
    result = DescriptionRegistry()
    register_glm4_moe_lite(result)
    return result


def expected_parameters(configuration: dict[str, Any]) -> dict[str, tuple[int, ...]]:
    """Independent logical-name and geometry oracle transcribed from pinned source."""
    hidden = configuration["hidden_size"]
    heads = configuration["num_attention_heads"]
    query_rank = configuration["q_lora_rank"]
    kv_rank = configuration["kv_lora_rank"]
    nope = configuration["qk_nope_head_dim"]
    rope = configuration["qk_rope_head_dim"]
    value = configuration["v_head_dim"]
    routed = configuration["n_routed_experts"]
    moe_width = configuration["moe_intermediate_size"]
    shared_width = moe_width * configuration["n_shared_experts"]
    result = {
        "model.embed_tokens.weight": (configuration["vocab_size"], hidden),
        "model.norm.weight": (hidden,),
        "lm_head.weight": (configuration["vocab_size"], hidden),
    }
    for index in range(configuration["num_hidden_layers"]):
        layer = f"model.layers.{index}"
        result.update(
            {
                f"{layer}.input_layernorm.weight": (hidden,),
                f"{layer}.post_attention_layernorm.weight": (hidden,),
                f"{layer}.self_attn.q_a_proj.weight": (query_rank, hidden),
                f"{layer}.self_attn.q_a_layernorm.weight": (query_rank,),
                f"{layer}.self_attn.q_b_proj.weight": (heads * (nope + rope), query_rank),
                f"{layer}.self_attn.kv_a_proj_with_mqa.weight": (kv_rank + rope, hidden),
                f"{layer}.self_attn.kv_a_layernorm.weight": (kv_rank,),
                f"{layer}.self_attn.kv_b_proj.weight": (heads * (nope + value), kv_rank),
                f"{layer}.self_attn.o_proj.weight": (hidden, heads * value),
            }
        )
        mlp = f"{layer}.mlp"
        if index == 0:
            dense = configuration["intermediate_size"]
            result.update(
                {
                    f"{mlp}.gate_proj.weight": (dense, hidden),
                    f"{mlp}.up_proj.weight": (dense, hidden),
                    f"{mlp}.down_proj.weight": (hidden, dense),
                }
            )
            continue
        result[f"{mlp}.gate.weight"] = (routed, hidden)
        result[f"{mlp}.gate.e_score_correction_bias"] = (routed,)
        for expert in range(routed):
            expert_prefix = f"{mlp}.experts.{expert}"
            result.update(
                {
                    f"{expert_prefix}.gate_proj.weight": (moe_width, hidden),
                    f"{expert_prefix}.up_proj.weight": (moe_width, hidden),
                    f"{expert_prefix}.down_proj.weight": (hidden, moe_width),
                }
            )
        shared = f"{mlp}.shared_experts"
        result.update(
            {
                f"{shared}.gate_proj.weight": (shared_width, hidden),
                f"{shared}.up_proj.weight": (shared_width, hidden),
                f"{shared}.down_proj.weight": (hidden, shared_width),
            }
        )
    return result


def expected_stacked_expert_storage(configuration: dict[str, Any]) -> dict[str, tuple[int, ...]]:
    """Independent physical expert geometry from the reviewed module declaration."""
    result: dict[str, tuple[int, ...]] = {}
    experts = configuration["n_routed_experts"]
    width = configuration["moe_intermediate_size"]
    hidden = configuration["hidden_size"]
    for index, layer_type in enumerate(configuration["mlp_layer_types"]):
        if layer_type == "sparse":
            prefix = f"model.layers.{index}.mlp.experts"
            result[prefix + ".gate_up_proj"] = (experts, 2 * width, hidden)
            result[prefix + ".down_proj"] = (experts, hidden, width)
    return result


def inputs(
    *,
    missing_storage: str | None = None,
    compressed: str | None = None,
    individual_expert: str | None = None,
    include_nextn_storage: bool = True,
    extra_storage: bool = False,
) -> AnalysisInput:
    configuration = copy.deepcopy(REFERENCE)
    expected = expected_parameters(configuration)
    physical: dict[str, r.ArchitectureStorage] = {}
    numeric: dict[str, NumericTensor] = {}
    for index, (name, dimensions) in enumerate(expected.items()):
        if EXPERT_REGION.search(name):
            continue  # The source stores these as stacked rank-3 parameters below.
        if name in {missing_storage, compressed, individual_expert}:
            continue
        physical[name] = r.ArchitectureStorage(name=name, dtype="BF16", shape=list(dimensions))
        numeric_id = f"glm-tensor-{index}"
        numeric[numeric_id] = NumericTensor(numeric_id, name, dimensions, "BF16")

    for name, dimensions in expected_stacked_expert_storage(configuration).items():
        individual_base = None
        if individual_expert is not None:
            family, suffix = individual_expert.split(".experts.", 1)
            individual_base = f"{family}.experts.{suffix.split('.', 1)[1].removesuffix('.weight')}"
        if name == missing_storage or name == individual_base:
            continue
        physical[name] = r.ArchitectureStorage(name=name, dtype="BF16", shape=list(dimensions))

    if compressed is not None:
        dimensions = expected[compressed]
        prefix = compressed.removesuffix(".weight")
        out, width = dimensions
        for suffix, dtype, shape in (
            (".weight_packed", "I32", [out, width // 8]),
            (".weight_scale", "BF16", [out, width // 32]),
            (".weight_shape", "I64", [2]),
        ):
            name = prefix + suffix
            physical[name] = r.ArchitectureStorage(name=name, dtype=dtype, shape=shape)
        numeric_id = "glm-packed-logical-view"
        numeric[numeric_id] = NumericTensor(
            numeric_id, compressed, dimensions, "I32", COMPRESSED_FORMAT
        )

    if individual_expert is not None and individual_expert != compressed:
        dimensions = expected[individual_expert]
        physical[individual_expert] = r.ArchitectureStorage(
            name=individual_expert, dtype="BF16", shape=list(dimensions)
        )
        numeric_id = "glm-individual-expert-view"
        numeric[numeric_id] = NumericTensor(numeric_id, individual_expert, dimensions, "BF16")

    if include_nextn_storage:
        name = "model.layers.47.mlp.gate.weight"
        physical[name] = r.ArchitectureStorage(name=name, dtype="BF16", shape=[64, 2048])
    if extra_storage:
        name = "model.layers.46.unreviewed.weight"
        physical[name] = r.ArchitectureStorage(name=name, dtype="BF16", shape=[1, 1])

    return AnalysisInput(
        "glm4-moe-lite-reference-fixture",
        configuration,
        BindingContext(physical, numeric, tokenizer_available=False),
    )


def node_id(data: AnalysisInput, key: str) -> str:
    return GraphBuilder(data, PRODUCER, "language_model").record_id("node", key)


def keyed_nodes(graph: r.ArchitectureGraph) -> dict[str, r.ArchitectureNode]:
    return {node.provenance[-1].source: node for node in graph.nodes if node.provenance}


def test_reference_fixture_is_pinned_and_description_is_selected() -> None:
    assert FIXTURE["source"]["checkpoint_revision"] == "25624b53414e585bcf7dcb9584667c3106c6089b"
    assert (
        FIXTURE["source"]["implementation_revision"] == "c8b81b63232be35ab1774dd3cabbf499d8b9808f"
    )
    assert (
        FIXTURE["source"]["config_sha256"]
        == "079f8fcd477b07378deafc21ecc5c3b974bf109ce5c9115854d97f88fa7622b9"
    )
    assert PRODUCER.source_revision == SOURCE_REVISION
    assert checked(REFERENCE) is not None
    normalized = copy.deepcopy(REFERENCE)
    normalized.pop("mlp_layer_types")
    assert checked(normalized) is not None
    assert registry().select(inputs()) is not None


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("n_routed_experts", 63),
        ("num_experts_per_tok", 8),
        ("q_lora_rank", 512),
        ("kv_lora_rank", 768),
        ("qk_nope_head_dim", 128),
        ("qk_head_dim", 257),
        ("num_key_value_heads", 10),
        ("norm_topk_prob", False),
        ("routed_scaling_factor", 1.0),
        ("topk_method", "greedy"),
        ("rope_interleave", False),
        ("partial_rotary_factor", 0.5),
        ("num_nextn_predict_layers", 2),
        ("first_k_dense_replace", 2),
        ("unknown_structural_option", True),
    ],
)
def test_checked_discriminator_rejects_unreviewed_structural_options(
    field: str, value: object
) -> None:
    configuration = copy.deepcopy(REFERENCE)
    configuration[field] = value
    data = AnalysisInput("negative", configuration, BindingContext({}, {}, False))
    assert not supports(data), field
    assert registry().select(data) is None


def test_checked_discriminator_rejects_inconsistent_rope_and_layer_policy() -> None:
    configuration = copy.deepcopy(REFERENCE)
    configuration["rope_parameters"]["rope_theta"] = 10000
    assert checked(configuration) is None
    configuration = copy.deepcopy(REFERENCE)
    configuration["mlp_layer_types"][0] = "sparse"
    assert checked(configuration) is None
    configuration = copy.deepcopy(REFERENCE)
    configuration["mlp_layer_types"][1] = "dense"
    assert checked(configuration) is None


def test_complete_graph_preserves_every_layer_expert_and_latent_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data = inputs(
        compressed="model.layers.1.self_attn.q_a_proj.weight",
        include_nextn_storage=True,
    )

    # Static graph generation must not execute model code, read tensor values, or use the network.
    import safetensors
    import safetensors.torch
    import transformers

    denied = Mock(side_effect=AssertionError("GLM analysis executed code or read weights"))
    for owner, names in [
        (socket, ["socket", "create_connection", "getaddrinfo"]),
        (torch, ["load", "tensor", "empty", "zeros", "ones", "from_file", "compile"]),
        (safetensors, ["safe_open"]),
        (safetensors.torch, ["load_file", "load"]),
        (torch.nn.Module, ["__init__", "__call__"]),
        (torch.jit, ["trace", "script"]),
        (torch.fx, ["symbolic_trace"]),
        (torch.cuda, ["init", "set_device", "_lazy_init"]),
        (torch.Tensor, ["cuda", "to"]),
        (transformers.AutoModel, ["from_config", "from_pretrained"]),
        (transformers.AutoModelForCausalLM, ["from_config", "from_pretrained"]),
        (transformers.AutoProcessor, ["from_pretrained"]),
        (transformers.GenerationMixin, ["generate"]),
        (ModelSource, ["iter_tensor", "iter_rows", "local_directory"]),
    ]:
        for name in names:
            monkeypatch.setattr(owner, name, denied)

    original_import = builtins.__import__

    def guarded_import(name: str, *args: Any, **kwargs: Any) -> Any:
        assert not name.startswith(
            ("modeling_glm4_moe_lite", "configuration_glm4_moe_lite", "transformers_modules")
        )
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    result = registry().analyze(data)
    denied.assert_not_called()
    assert result.status == "complete", result
    assert result.graph is not None
    graph = result.graph
    assert graph.coverage == "complete"
    assert serialized_size(graph.document(), MAX_BYTES) < MAX_BYTES

    expected = expected_parameters(REFERENCE)
    parameters = {parameter.name: parameter for parameter in graph.parameters}
    assert set(parameters) == set(expected)
    assert len(parameters) == len(expected)
    for name, dimensions in expected.items():
        parameter = parameters[name]
        assert parameter.logical_shape == [
            r.ArchitectureConstantDimension(kind="constant", value=value) for value in dimensions
        ]
        if EXPERT_REGION.search(name):
            assert isinstance(parameter, r.ArchitectureFusedParameter)
            assert isinstance(parameter.inspection, r.ArchitectureUnavailableInspection)
            assert parameter.inspection.reason == "requires_view"
            expert = int(name.split(".experts.", 1)[1].split(".", 1)[0])
            assert f"expert index {expert}" in parameter.region.description
        elif name == "model.layers.1.self_attn.q_a_proj.weight":
            assert parameter.binding == "quantized"
            assert isinstance(parameter.inspection, r.ArchitectureAvailableInspection)
            assert parameter.inspection.tensor_id == "glm-packed-logical-view"
            assert {storage.role for storage in parameter.storage} == {
                "packed_data",
                "scales",
                "logical_shape",
            }
        else:
            assert parameter.binding == "native"

    by_key = keyed_nodes(graph)
    assert by_key["model.layers.0.mlp"].label
    assert by_key["model.layers.0.mlp.gate_proj"].operation == "linear"
    assert by_key["model.layers.1.self_attn.q_a_proj"].operation == "linear"
    assert by_key["model.layers.1.self_attn.q_a_layernorm"].operation == "rms_norm"
    assert by_key["model.layers.1.self_attn.q_b_proj"].operation == "linear"
    assert by_key["model.layers.1.self_attn.kv_a_proj_with_mqa"].operation == "linear"
    assert by_key["model.layers.1.self_attn.kv_a_layernorm"].operation == "rms_norm"
    assert by_key["model.layers.1.self_attn.kv_b_proj"].operation == "linear"
    query_split = by_key["model.layers.1.self_attn.q_split"]
    non_rotary_shape = query_split.ports[1].shape
    rotary_shape = query_split.ports[2].shape
    assert non_rotary_shape is not None and rotary_shape is not None
    assert isinstance(non_rotary_shape[-1], r.ArchitectureConstantDimension)
    assert isinstance(rotary_shape[-1], r.ArchitectureConstantDimension)
    assert non_rotary_shape[-1].value == 192
    assert rotary_shape[-1].value == 64
    assert by_key["model.layers.1.self_attn.kv_latent_cache_update"].operation == "state_concat"
    assert by_key["model.layers.1.self_attn.rotary_key_cache_update"].operation == "state_concat"
    assert by_key["model.layers.1.self_attn.attention_softmax"].operation == "softmax"
    assert by_key["model.layers.1.self_attn.o_proj"].operation == "linear"
    assert by_key["model.layers.1.attention_residual"].operation == "add"
    assert by_key["model.layers.1.mlp_residual"].operation == "add"

    layer_instances = next(rep for rep in graph.repetitions if rep.label == "Decoder layers")
    assert [(entry.index, entry.variant) for entry in layer_instances.instances] == [
        (0, "dense"),
        *[(index, "sparse") for index in range(1, 47)],
    ]
    expert_repetitions = [rep for rep in graph.repetitions if rep.label == "Routed experts"]
    assert len(expert_repetitions) == 46
    assert all(len(rep.instances) == 64 for rep in expert_repetitions)
    assert sum(len(rep.instances) for rep in expert_repetitions) == 46 * 64
    for index in range(1, 47):
        assert [
            entry.index
            for entry in next(
                rep
                for rep in expert_repetitions
                if rep.parent_id == node_id(data, f"model.layers.{index}.mlp")
            ).instances
        ] == list(range(64))

    assert by_key["model.layers.1.mlp.router_sigmoid"].operation == "sigmoid"
    assert by_key["model.layers.1.mlp.group_top2_score_sum"].operation == "group_topk_score_sum"
    assert by_key["model.layers.1.mlp.expert_topk"].attributes[0].value == 4.0
    assert isinstance(by_key["model.layers.1.mlp.experts.63"], r.ArchitectureGroupNode)
    assert by_key["model.layers.1.mlp.experts.63"].operation == "routed_swiglu_expert"
    expert_node = by_key["model.layers.1.mlp.experts.63"]
    assert expert_node.formula is not None
    assert "normalized router weight" in expert_node.formula
    assert by_key["model.layers.1.mlp.shared_experts.down_proj"].operation == "linear"
    assert by_key["model.layers.1.mlp.shared_routed_add"].operation == "add"

    root = by_key["model"]
    root_attributes = {attribute.name: attribute.value for attribute in root.attributes}
    assert root_attributes["num_nextn_predict_layers"] == 1.0
    assert isinstance(root_attributes["nextn_evaluation"], str)
    assert "metadata only" in root_attributes["nextn_evaluation"]
    assert not any(key.startswith("model.layers.47") for key in by_key)
    assert not any("weight_packed" in key or "weight_shape" in key for key in by_key)


def test_missing_stacked_expert_parameter_and_unknown_storage_are_partial() -> None:
    data = inputs(
        missing_storage="model.layers.7.mlp.experts.down_proj",
        include_nextn_storage=False,
        extra_storage=True,
    )
    result = registry().analyze(data)
    assert result.status == "partial", result
    assert result.graph is not None
    parameters = {parameter.name: parameter for parameter in result.graph.parameters}
    missing = parameters["model.layers.7.mlp.experts.63.down_proj.weight"]
    assert missing.binding == "unresolved"
    assert any(diagnostic.code == "unresolved_binding" for diagnostic in result.graph.diagnostics)
    assert any(diagnostic.code == "unrecognized_storage" for diagnostic in result.graph.diagnostics)
    repetitions = [rep for rep in result.graph.repetitions if rep.label == "Routed experts"]
    layer_seven = next(
        rep
        for rep in repetitions
        if rep.parent_id
        == GraphBuilder(data, PRODUCER, "language_model").record_id("node", "model.layers.7.mlp")
    )
    assert len(layer_seven.instances) == 64


def test_individual_packed_expert_inventory_binds_the_exact_expert_matrix() -> None:
    name = "model.layers.1.mlp.experts.0.down_proj.weight"
    data = inputs(individual_expert=name, compressed=name)
    result = registry().analyze(data)
    assert result.status == "partial", result
    assert result.graph is not None
    parameters = {parameter.name: parameter for parameter in result.graph.parameters}
    selected = parameters[name]
    assert isinstance(selected, r.ArchitectureDirectParameter)
    assert selected.binding == "quantized"
    assert selected.logical_shape == [
        r.ArchitectureConstantDimension(kind="constant", value=2048),
        r.ArchitectureConstantDimension(kind="constant", value=1536),
    ]
    assert [storage.name for storage in selected.storage] == [
        name.removesuffix(".weight") + ".weight_packed",
        name.removesuffix(".weight") + ".weight_scale",
        name.removesuffix(".weight") + ".weight_shape",
    ]
    assert selected.inspection == r.ArchitectureAvailableInspection(
        status="available", tensor_id="glm-packed-logical-view"
    )


def test_two_compressed_experts_keep_their_own_logical_tensor_ids() -> None:
    names = [f"model.layers.1.mlp.experts.{index}.gate_proj.weight" for index in (0, 1)]
    data = inputs(compressed=names[0])
    physical = dict(data.bindings.physical)
    numeric = dict(data.bindings.numeric)
    output, width = expected_parameters(REFERENCE)[names[1]]
    prefix = names[1].removesuffix(".weight")
    for suffix, dtype, dimensions in (
        (".weight_packed", "I32", [output, width // 8]),
        (".weight_scale", "BF16", [output, width // 32]),
        (".weight_shape", "I64", [2]),
    ):
        name = prefix + suffix
        physical[name] = r.ArchitectureStorage(name=name, dtype=dtype, shape=dimensions)
    numeric["second-glm-expert"] = NumericTensor(
        "second-glm-expert", names[1], (output, width), "I32", COMPRESSED_FORMAT
    )
    data = replace(data, bindings=replace(data.bindings, physical=physical, numeric=numeric))
    result = registry().analyze(data)
    assert result.status == "complete", result
    assert result.graph is not None
    parameters = {parameter.name: parameter for parameter in result.graph.parameters}
    assert parameters[names[0]].inspection == r.ArchitectureAvailableInspection(
        status="available", tensor_id="glm-packed-logical-view"
    )
    assert parameters[names[1]].inspection == r.ArchitectureAvailableInspection(
        status="available", tensor_id="second-glm-expert"
    )

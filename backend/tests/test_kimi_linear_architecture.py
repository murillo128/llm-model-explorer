"""Pinned-source graph oracles for Kimi Linear's KDA, MLA, and routed MoE paths."""

from __future__ import annotations

import asyncio
import builtins
import copy
import json
import socket
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import pytest
import torch
from transformers.generation.utils import GenerationMixin

from llm_model_explorer.architecture_analysis import (
    AnalysisInput,
    BindingContext,
    DescriptionRegistry,
    GraphBuilder,
    NumericTensor,
    serialize_graph,
)
from llm_model_explorer.architecture_analysis import records as r
from llm_model_explorer.architecture_analysis.glm4_moe_lite import COMPRESSED_FORMAT
from llm_model_explorer.architecture_analysis.kimi_linear import (
    ARCHITECTURE,
    MODEL_TYPE,
    PRODUCER,
    KimiLinearGraph,
    checked,
    parameter_shapes,
    register_kimi_linear,
)
from llm_model_explorer.architecture_analysis.validation import MAX_BYTES, constants
from llm_model_explorer.architecture_service import ArchitectureService
from llm_model_explorer.artifacts import ArtifactStore
from llm_model_explorer.execution import BlockingWork

FIXTURE = json.loads((Path(__file__).parent / "fixtures/kimi-linear-reference.json").read_text())
REFERENCE = FIXTURE["configuration"]
EXPERT = "model.layers.1.block_sparse_moe.experts.0.w1.weight"


def expected_parameters(configuration: dict[str, Any]) -> dict[str, tuple[int, ...]]:
    """Independent logical-name and geometry oracle from the pinned module declarations."""
    hidden = configuration["hidden_size"]
    vocab = configuration["vocab_size"]
    heads = configuration["num_attention_heads"]
    kda_heads = configuration["linear_attn_config"]["num_heads"]
    kda_dim = configuration["linear_attn_config"]["head_dim"]
    conv = configuration["linear_attn_config"]["short_conv_kernel_size"]
    nope = configuration["qk_nope_head_dim"]
    rope = configuration["qk_rope_head_dim"]
    value = configuration["v_head_dim"]
    rank = configuration["kv_lora_rank"]
    experts = configuration["num_experts"]
    moe_width = configuration["moe_intermediate_size"]
    shared_width = moe_width * configuration["num_shared_experts"]
    result = {
        "model.embed_tokens.weight": (vocab, hidden),
        "model.norm.weight": (hidden,),
        "lm_head.weight": (vocab, hidden),
    }
    for layer_index, kind in enumerate(FIXTURE["expected"]["layer_types"]):
        layer = f"model.layers.{layer_index}"
        result.update(
            {
                f"{layer}.input_layernorm.weight": (hidden,),
                f"{layer}.post_attention_layernorm.weight": (hidden,),
            }
        )
        attention = layer + ".self_attn"
        if kind == "kda":
            width = kda_heads * kda_dim
            result.update(
                {
                    f"{attention}.q_proj.weight": (width, hidden),
                    f"{attention}.k_proj.weight": (width, hidden),
                    f"{attention}.v_proj.weight": (width, hidden),
                    f"{attention}.q_conv1d.weight": (width, 1, conv),
                    f"{attention}.k_conv1d.weight": (width, 1, conv),
                    f"{attention}.v_conv1d.weight": (width, 1, conv),
                    f"{attention}.A_log": (1, 1, kda_heads, 1),
                    f"{attention}.dt_bias": (width,),
                    f"{attention}.f_a_proj.weight": (kda_dim, hidden),
                    f"{attention}.f_b_proj.weight": (width, kda_dim),
                    f"{attention}.b_proj.weight": (kda_heads, hidden),
                    f"{attention}.g_a_proj.weight": (kda_dim, hidden),
                    f"{attention}.g_b_proj.weight": (width, kda_dim),
                    f"{attention}.o_norm.weight": (kda_dim,),
                    f"{attention}.o_proj.weight": (hidden, width),
                }
            )
        else:
            q_width = nope + rope
            result.update(
                {
                    f"{attention}.q_proj.weight": (heads * q_width, hidden),
                    f"{attention}.kv_a_proj_with_mqa.weight": (rank + rope, hidden),
                    f"{attention}.kv_a_layernorm.weight": (rank,),
                    f"{attention}.kv_b_proj.weight": (heads * (nope + value), rank),
                    f"{attention}.o_proj.weight": (hidden, heads * value),
                }
            )

        if layer_index == 0:
            dense = configuration["intermediate_size"]
            result.update(
                {
                    f"{layer}.mlp.gate_proj.weight": (dense, hidden),
                    f"{layer}.mlp.up_proj.weight": (dense, hidden),
                    f"{layer}.mlp.down_proj.weight": (hidden, dense),
                }
            )
        else:
            moe = layer + ".block_sparse_moe"
            result[f"{moe}.gate.weight"] = (experts, hidden)
            result[f"{moe}.gate.e_score_correction_bias"] = (experts,)
            for expert_index in range(experts):
                prefix = f"{moe}.experts.{expert_index}"
                result.update(
                    {
                        f"{prefix}.w1.weight": (moe_width, hidden),
                        f"{prefix}.w2.weight": (hidden, moe_width),
                        f"{prefix}.w3.weight": (moe_width, hidden),
                    }
                )
            shared = moe + ".shared_experts"
            result.update(
                {
                    f"{shared}.gate_proj.weight": (shared_width, hidden),
                    f"{shared}.up_proj.weight": (shared_width, hidden),
                    f"{shared}.down_proj.weight": (hidden, shared_width),
                }
            )
    return result


def inputs(
    *,
    missing: str | None = None,
    compressed_expert: bool = False,
    fingerprint: str = "kimi-fixture",
) -> AnalysisInput:
    configuration = copy.deepcopy(REFERENCE)
    expected = expected_parameters(configuration)
    physical = {
        name: r.ArchitectureStorage(name=name, dtype="BF16", shape=list(dimensions))
        for name, dimensions in expected.items()
        if name not in {missing, EXPERT if compressed_expert else None}
    }
    numeric: dict[str, NumericTensor] = {}
    if compressed_expert:
        selected = FIXTURE["bounded_safetensors_header_observations"]
        for name in (
            EXPERT.removesuffix(".weight") + ".weight_packed",
            EXPERT.removesuffix(".weight") + ".weight_scale",
            EXPERT.removesuffix(".weight") + ".weight_shape",
        ):
            dtype, dimensions = selected[name]
            physical[name] = r.ArchitectureStorage(name=name, dtype=dtype, shape=dimensions)
        numeric["quantized-w1"] = NumericTensor(
            "quantized-w1", EXPERT, expected[EXPERT], "I32", COMPRESSED_FORMAT
        )
    return AnalysisInput(
        fingerprint,
        configuration,
        BindingContext(physical=physical, numeric=numeric, tokenizer_available=False),
    )


def compressed_expert_inputs(*, missing: str | None = None) -> AnalysisInput:
    """Model the target's packed routed-expert inventory without reading weight bytes."""
    data = inputs(missing=missing, fingerprint="kimi-packed-expert-fixture")
    physical = dict(data.bindings.physical)
    numeric: dict[str, NumericTensor] = {}
    for name, dimensions in expected_parameters(REFERENCE).items():
        if ".block_sparse_moe.experts." not in name or not name.endswith(
            (".w1.weight", ".w2.weight", ".w3.weight")
        ):
            continue
        ignored = REFERENCE["quantization_config"]["ignore"]
        assert name.removesuffix(".weight") not in ignored
        physical.pop(name, None)
        if name == missing:
            continue
        prefix = name.removesuffix(".weight")
        output, input_width = dimensions
        companions = (
            (".weight_packed", "I32", [output, input_width // 8]),
            (".weight_scale", "BF16", [output, input_width // 32]),
            (".weight_shape", "I64", [2]),
        )
        for suffix, dtype, shape in companions:
            storage_name = prefix + suffix
            physical[storage_name] = r.ArchitectureStorage(
                name=storage_name, dtype=dtype, shape=shape
            )
        tensor_id = f"packed-expert-{len(numeric)}"
        numeric[tensor_id] = NumericTensor(tensor_id, name, dimensions, "I32", COMPRESSED_FORMAT)
    return AnalysisInput(
        data.fingerprint,
        data.configuration,
        BindingContext(physical=physical, numeric=numeric, tokenizer_available=False),
    )


def registry() -> DescriptionRegistry:
    result = DescriptionRegistry()
    register_kimi_linear(result)
    return result


def test_architecture_service_registers_kimi_description(tmp_path: Path) -> None:
    work = BlockingWork()
    try:
        service = ArchitectureService(
            ArtifactStore(tmp_path / "cache", model_root=tmp_path / "models"), work
        )
        assert service.registry.select(inputs()) is not None
    finally:
        asyncio.run(work.aclose())


def node_id(data: AnalysisInput, key: str) -> str:
    return GraphBuilder(data, PRODUCER, "language_model").record_id("node", key)


def graph_node(data: AnalysisInput, graph: r.ArchitectureGraph, key: str) -> r.ArchitectureNode:
    identity = node_id(data, key)
    return next(value for value in graph.nodes if value.id == identity)


def has_edge(
    data: AnalysisInput,
    graph: r.ArchitectureGraph,
    source_key: str,
    source_port: str,
    target_key: str,
    target_port: str,
) -> bool:
    source_id = node_id(data, source_key)
    target_id = node_id(data, target_key)
    return any(
        edge.source.node_id == source_id
        and edge.source.port_id == source_port
        and edge.target.node_id == target_id
        and edge.target.port_id == target_port
        for edge in graph.edges
    )


def test_pinned_configuration_fixture_and_independent_geometry() -> None:
    configuration = checked(REFERENCE)
    assert configuration is not None
    assert REFERENCE["model_type"] == MODEL_TYPE
    assert REFERENCE["architectures"] == [ARCHITECTURE]
    assert configuration["layer_types"] == FIXTURE["expected"]["layer_types"]
    expected_kda_indices = {0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18, 20, 21, 22, 24, 25}
    assert configuration["layer_types"] == [
        "kda" if index in expected_kda_indices else "full_attention" for index in range(27)
    ]
    assert configuration["layer_types"].count("kda") == 20
    assert configuration["layer_types"].count("full_attention") == 7
    assert PRODUCER.revision == "4"
    assert parameter_shapes(REFERENCE) == expected_parameters(REFERENCE)
    assert FIXTURE["source"]["configuration_revision"] in PRODUCER.source_revision
    assert FIXTURE["source"]["modeling_revision"] in PRODUCER.source_revision
    assert FIXTURE["bounded_safetensors_header_observations"][
        "model.layers.4.self_attn.q_proj.weight"
    ] == ["BF16", [6144, 2304]]


def test_real_header_high_rank_native_bindings_have_unavailable_inspection() -> None:
    """The pinned Kimi header has an admitted rank-3 convolution and a rank-4 A_log."""
    base = inputs()
    observations = FIXTURE["bounded_safetensors_header_observations"]
    names = (
        "model.layers.0.self_attn.q_conv1d.weight",
        "model.layers.0.self_attn.A_log",
        "model.layers.0.self_attn.dt_bias",
    )
    physical = dict(base.bindings.physical)
    numeric: dict[str, NumericTensor] = {}
    numeric_names = set(FIXTURE["admitted_native_numeric_observations"])
    for name in names:
        dtype, dimensions = observations[name]
        physical[name] = r.ArchitectureStorage(name=name, dtype=dtype, shape=dimensions)
        if name in numeric_names:
            tensor_id = "observed-" + name.rsplit(".", 1)[-1]
            numeric[tensor_id] = NumericTensor(
                tensor_id, name, tuple(dimensions), dtype, "safetensors"
            )
    assert numeric_names == {names[0], names[2]}
    data = AnalysisInput(
        base.fingerprint,
        base.configuration,
        BindingContext(physical=physical, numeric=numeric, tokenizer_available=False),
    )

    result = registry().analyze(data)
    assert result.status == "complete", result
    assert result.graph is not None
    parameters = {parameter.name: parameter for parameter in result.graph.parameters}
    for name in names[:2]:
        parameter = parameters[name]
        dtype, dimensions = observations[name]
        assert parameter.binding == "native"
        assert constants(parameter.logical_shape) == tuple(dimensions)
        assert [(item.name, item.dtype, item.shape) for item in parameter.storage] == [
            (name, dtype, dimensions)
        ]
        assert parameter.inspection.status == "unavailable"
        assert parameter.inspection.reason == "unsupported_rank"
        assert any(item.kind == "description" for item in parameter.provenance)
        assert any(parameter.id in node.parameter_ids for node in result.graph.nodes)

    bias_inspection = parameters[names[2]].inspection
    assert bias_inspection.status == "available"
    assert bias_inspection.tensor_id == "observed-dt_bias"


@pytest.mark.parametrize(
    "mutate",
    [
        lambda c: c["linear_attn_config"].__setitem__(
            "kda_layers", [2, *c["linear_attn_config"]["kda_layers"][1:]]
        ),
        lambda c: c["linear_attn_config"].__setitem__("short_conv_kernel_size", 3),
        lambda c: c.__setitem__("kv_lora_rank", 256),
        lambda c: c.__setitem__("qk_nope_head_dim", 96),
        lambda c: c.__setitem__("num_experts", 128),
        lambda c: c.__setitem__("num_experts_per_token", 4),
        lambda c: c.__setitem__("num_shared_experts", 0),
        lambda c: c.__setitem__("routed_scaling_factor", 1.0),
        lambda c: c.__setitem__("num_expert_group", True),
        lambda c: c.__setitem__("model_type", "kimi_linear_other"),
        lambda c: c.__setitem__("architectures", ["KimiLinearModel"]),
        lambda c: c["auto_map"].__setitem__("AutoModelForCausalLM", "other.Model"),
        lambda c: c.__setitem__("unreviewed_option", True),
    ],
)
def test_unreviewed_or_contradictory_options_fail_closed(mutate: Any) -> None:
    configuration = copy.deepcopy(REFERENCE)
    mutate(configuration)
    assert checked(configuration) is None


def test_complete_graph_preserves_kda_mla_moe_identity_and_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data = compressed_expert_inputs()
    forbidden = Mock(
        side_effect=AssertionError("static description executed model or network code")
    )
    original_import = builtins.__import__

    def guarded_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if "modeling_kimi" in name or name == "configuration_kimi":
            raise AssertionError("checkpoint Python was imported")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(builtins, "__import__", guarded_import)
    monkeypatch.setattr(torch.nn.Module, "__init__", forbidden)
    monkeypatch.setattr(torch.nn.Module, "__call__", forbidden)
    monkeypatch.setattr(torch.nn.Module, "forward", forbidden)
    monkeypatch.setattr(GenerationMixin, "generate", forbidden)
    monkeypatch.setattr(torch.jit, "trace", forbidden)
    monkeypatch.setattr(torch.jit, "script", forbidden)
    monkeypatch.setattr(torch, "load", forbidden)

    result = registry().analyze(data)
    assert result.status == "complete", result
    assert result.graph is not None
    graph = result.graph
    assert len(serialize_graph(graph)) < MAX_BYTES

    parameters = {parameter.name: parameter for parameter in graph.parameters}
    parameters_by_id = {parameter.id: parameter for parameter in graph.parameters}
    nodes_by_id = {node.id: node for node in graph.nodes}
    assert set(parameters) == set(expected_parameters(REFERENCE))
    for name, parameter in parameters.items():
        if ".block_sparse_moe.experts." in name:
            assert parameter.binding == "quantized"
            assert isinstance(parameter.inspection, r.ArchitectureAvailableInspection)
            assert parameter.inspection.tensor_id == next(
                tensor.id for tensor in data.bindings.numeric.values() if tensor.name == name
            )
            assert [value.name for value in parameter.storage] == [
                name.removesuffix(".weight") + ".weight_packed",
                name.removesuffix(".weight") + ".weight_scale",
                name.removesuffix(".weight") + ".weight_shape",
            ]
        else:
            assert parameter.binding == "native"
    conv_shape = parameters["model.layers.0.self_attn.q_conv1d.weight"].logical_shape
    assert conv_shape is not None
    assert [dimension.model_dump() for dimension in conv_shape] == [
        {"kind": "constant", "value": 4096},
        {"kind": "constant", "value": 1},
        {"kind": "constant", "value": 4},
    ]

    layer_repetition = next(item for item in graph.repetitions if item.label == "Decoder layers")
    assert [instance.index for instance in layer_repetition.instances] == list(range(27))
    for index, kind in enumerate(FIXTURE["expected"]["layer_types"]):
        layer = graph_node(data, graph, f"model.layers.{index}")
        assert layer.kind == "group"
        assert (
            graph_node(data, graph, f"model.layers.{index}.input_layernorm").operation == "rms_norm"
        )
        assert (
            graph_node(data, graph, f"model.layers.{index}.attention_residual").operation == "add"
        )
        assert (
            graph_node(data, graph, f"model.layers.{index}.post_attention_layernorm").operation
            == "rms_norm"
        )
        assert graph_node(data, graph, f"model.layers.{index}.mlp_residual").operation == "add"
        assert next(
            attribute.value for attribute in layer.attributes if attribute.name == "attention_type"
        ) == ("KDA" if kind == "kda" else "MLA")
        attention_key = f"model.layers.{index}.self_attn"
        attention = graph_node(data, graph, attention_key)
        assert attention.kind == "group"
        if kind == "kda":
            compaction_key = attention_key + ".input_compaction"
            compaction = graph_node(data, graph, compaction_key)
            assert compaction.operation == "compact_valid_tokens"
            assert has_edge(data, graph, attention_key, "x", compaction_key, "padded_hidden_states")
            assert has_edge(
                data, graph, attention_key, "padding_mask", compaction_key, "padding_mask"
            )
            for projection in ("q_proj", "k_proj", "v_proj", "f_a_proj", "b_proj", "g_a_proj"):
                assert has_edge(
                    data,
                    graph,
                    compaction_key,
                    "compact_hidden_states",
                    attention_key + "." + projection,
                    "x",
                )
            for branch in ("q", "k", "v"):
                convolution_key = attention_key + f".{branch}_conv1d"
                assert has_edge(
                    data,
                    graph,
                    compaction_key,
                    "sequence_offsets",
                    convolution_key,
                    "sequence_offsets",
                )
            assert (
                graph_node(data, graph, attention_key + ".kda_delta_update").operation
                == "kda_delta_state_update"
            )
            update_key = attention_key + ".kda_delta_update"
            update = graph_node(data, graph, update_key)
            assert {port.id for port in update.ports} >= {"sequence_offsets", "next_state"}
            assert "padding_mask" not in {port.id for port in update.ports}
            assert has_edge(
                data, graph, compaction_key, "sequence_offsets", update_key, "sequence_offsets"
            )
            assert (
                graph_node(data, graph, attention_key + ".q_conv1d").operation
                == "short_convolution"
            )
            padding_key = attention_key + ".output_repadding"
            repadding = graph_node(data, graph, padding_key)
            assert repadding.operation == "restore_padding"
            padded_shape = [
                {"kind": "symbol", "name": "B"},
                {"kind": "symbol", "name": "S"},
                {"kind": "constant", "value": REFERENCE["hidden_size"]},
            ]
            restored_shape = next(port for port in repadding.ports if port.id == "out").shape
            attention_output_shape = next(
                port for port in attention.ports if port.id == "out"
            ).shape
            assert restored_shape is not None and attention_output_shape is not None
            assert [dimension.model_dump() for dimension in restored_shape] == padded_shape
            assert [dimension.model_dump() for dimension in attention_output_shape] == padded_shape
            assert has_edge(
                data,
                graph,
                attention_key + ".o_proj",
                "out",
                padding_key,
                "compact_output",
            )
            assert has_edge(
                data, graph, compaction_key, "token_indices", padding_key, "token_indices"
            )
            assert has_edge(data, graph, padding_key, "out", attention_key, "out")
            assert any(port.id == "next_recurrent_state" for port in attention.ports)
        else:
            assert (
                graph_node(data, graph, attention_key + ".kv_a_layernorm").operation == "rms_norm"
            )
            assert graph_node(data, graph, attention_key + ".softmax").operation == "softmax"
            assert any(port.id == "next_key_state" for port in attention.ports)

        if index == 0:
            assert graph_node(data, graph, "model.layers.0.mlp.gate_proj").operation == "linear"
        else:
            moe_key = f"model.layers.{index}.block_sparse_moe"
            moe_repetition = next(
                item for item in graph.repetitions if item.parent_id == node_id(data, moe_key)
            )
            assert len(moe_repetition.instances) == 256
            assert [instance.index for instance in moe_repetition.instances] == list(range(256))
            for instance in moe_repetition.instances:
                expert_node = nodes_by_id[instance.node_id]
                assert expert_node.operation == "weighted_swiglu_mlp"
                expert_name = f"{moe_key}.experts.{instance.index}"
                assert [parameters_by_id[key].name for key in expert_node.parameter_ids] == [
                    expert_name + ".w1.weight",
                    expert_name + ".w3.weight",
                    expert_name + ".w2.weight",
                ]
                assert all(parameters_by_id[key].provenance for key in expert_node.parameter_ids)
                assert expert_node.provenance
            topk = graph_node(data, graph, moe_key + ".topk")
            assert topk.operation == "kimi_grouped_topk_selection"
            assert topk.formula is not None and "view of the sigmoid score tensor" in topk.formula
            routed = graph_node(data, graph, moe_key + ".dispatch_execute_scatter_sum")
            assert routed.operation == "symbolic_expert_dispatch_execute_scatter_sum"
            assert not routed.parameter_ids
            assert routed.formula is not None and "scatter-add" in routed.formula
            assert (
                graph_node(data, graph, moe_key + ".shared_experts.gate_proj").operation == "linear"
            )

    assert graph_node(data, graph, "model.norm").operation == "rms_norm"
    assert graph_node(data, graph, "lm_head").operation == "linear"
    assert not graph.diagnostics
    over_budget = registry().analyze(data, byte_limit=1024)
    assert over_budget.graph is None
    assert over_budget.reason == "unsupported_size"
    forbidden.assert_not_called()


def test_missing_and_compressed_parameter_bindings_are_explicit() -> None:
    expected = expected_parameters(REFERENCE)
    compressed_data = inputs(compressed_expert=True)
    builder = GraphBuilder(compressed_data, PRODUCER, "language_model")
    description = KimiLinearGraph.__new__(KimiLinearGraph)
    description.inputs = compressed_data
    description.b = builder
    compressed_config = checked(compressed_data.configuration)
    assert compressed_config is not None
    description.c = compressed_config
    description.physical = compressed_data.bindings.physical
    description.numeric = {
        tensor.name: tensor for tensor in compressed_data.bindings.numeric.values()
    }
    description.used_storage = set()
    identity = description.parameter(EXPERT, expected[EXPERT])
    packed = builder._parameter_by_id[identity]
    assert packed.binding == "quantized"
    assert [storage.name for storage in packed.storage] == [
        EXPERT.removesuffix(".weight") + ".weight_packed",
        EXPERT.removesuffix(".weight") + ".weight_scale",
        EXPERT.removesuffix(".weight") + ".weight_shape",
    ]
    assert packed.inspection.status == "available"
    assert packed.inspection.tensor_id == "quantized-w1"
    assert all("weight_packed" not in node.label for node in builder._nodes)

    for missing_scale in (True, False):
        physical = dict(compressed_data.bindings.physical)
        numeric = dict(compressed_data.bindings.numeric)
        if missing_scale:
            physical.pop(EXPERT.removesuffix(".weight") + ".weight_scale")
        else:
            numeric["quantized-w1"] = NumericTensor(
                "quantized-w1", EXPERT, expected[EXPERT], "I32", "unknown-packed-format"
            )
        damaged = AnalysisInput(
            compressed_data.fingerprint,
            compressed_data.configuration,
            BindingContext(physical, numeric, tokenizer_available=False),
        )
        damaged_builder = GraphBuilder(damaged, PRODUCER, "language_model")
        damaged_description = KimiLinearGraph.__new__(KimiLinearGraph)
        damaged_description.inputs = damaged
        damaged_description.b = damaged_builder
        damaged_description.c = compressed_config
        damaged_description.physical = damaged.bindings.physical
        damaged_description.numeric = {
            tensor.name: tensor for tensor in damaged.bindings.numeric.values()
        }
        damaged_description.used_storage = set()
        rejected = damaged_builder._parameter_by_id[
            damaged_description.parameter(EXPERT, expected[EXPERT])
        ]
        assert rejected.binding == "unresolved"
        assert rejected.inspection.status == "unavailable"
        assert rejected.inspection.reason == "unresolved_binding"
        assert damaged_builder._partial

    missing_data = inputs(missing=EXPERT)
    missing_builder = GraphBuilder(missing_data, PRODUCER, "language_model")
    missing = KimiLinearGraph.__new__(KimiLinearGraph)
    missing.inputs = missing_data
    missing.b = missing_builder
    missing_config = checked(missing_data.configuration)
    assert missing_config is not None
    missing.c = missing_config
    missing.physical = missing_data.bindings.physical
    missing.numeric = {}
    missing.used_storage = set()
    missing_id = missing._unresolved(EXPERT, expected[EXPERT], [], "missing test tensor")
    unresolved = missing_builder._parameter_by_id[missing_id]
    assert unresolved.binding == "unresolved"
    assert missing_builder._partial
    assert any(
        diagnostic.code == "unresolved_binding" for diagnostic in missing_builder._diagnostics
    )

    partial = registry().analyze(compressed_expert_inputs(missing=EXPERT))
    assert partial.status == "partial"
    assert partial.graph is not None
    partial_parameters = {parameter.name: parameter for parameter in partial.graph.parameters}
    assert partial_parameters[EXPERT].binding == "unresolved"
    assert any(diagnostic.code == "unresolved_binding" for diagnostic in partial.graph.diagnostics)

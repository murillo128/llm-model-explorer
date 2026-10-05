"""Pinned-source graph oracles for DeepSeek-V2-Lite's MLA and routed MoE paths."""

from __future__ import annotations

import builtins
import copy
import json
import math
import socket
from dataclasses import replace
from pathlib import Path
from typing import Any, cast
from unittest.mock import Mock

import pytest
import torch

from llm_model_explorer.architecture_analysis import (
    AnalysisInput,
    BindingContext,
    DescriptionRegistry,
    GraphBuilder,
    NumericTensor,
    parse_graph,
    serialize_graph,
)
from llm_model_explorer.architecture_analysis import records as r
from llm_model_explorer.architecture_analysis.deepseek_v2 import (
    ARCHITECTURE,
    PRODUCER,
    checked,
    register_deepseek_v2,
    supports,
)
from llm_model_explorer.architecture_analysis.validation import MAX_BYTES
from llm_model_explorer.architecture_service import ArchitectureService
from llm_model_explorer.tensor_source import ModelSource

FIXTURE = json.loads(
    (Path(__file__).parent / "fixtures/deepseek-v2-lite-reference.json").read_text()
)
REFERENCE = FIXTURE["configuration"]


def native_configuration() -> dict[str, Any]:
    """The pinned base config has the same reviewed structure and native storage."""
    result = cast(dict[str, Any], copy.deepcopy(REFERENCE))
    result.pop("quantization_config")
    result.pop("ep_size")
    result.pop("_name_or_path")
    result["auto_map"] = {
        "AutoConfig": "configuration_deepseek.DeepseekV2Config",
        "AutoModel": "modeling_deepseek.DeepseekV2Model",
        "AutoModelForCausalLM": "modeling_deepseek.DeepseekV2ForCausalLM",
    }
    result["transformers_version"] = "4.33.1"
    return result


def expected_parameters(configuration: dict[str, Any]) -> dict[str, tuple[int, ...]]:
    """Independent tensor-name/geometry oracle transcribed from the pinned source."""
    hidden = configuration["hidden_size"]
    vocab = configuration["vocab_size"]
    heads = configuration["num_attention_heads"]
    nope = configuration["qk_nope_head_dim"]
    rope = configuration["qk_rope_head_dim"]
    value = configuration["v_head_dim"]
    rank = configuration["kv_lora_rank"]
    experts = configuration["n_routed_experts"]
    routed = configuration["moe_intermediate_size"]
    shared = routed * configuration["n_shared_experts"]
    result = {
        "model.embed_tokens.weight": (vocab, hidden),
        "model.norm.weight": (hidden,),
        "lm_head.weight": (vocab, hidden),
    }
    for layer_index in range(configuration["num_hidden_layers"]):
        layer = f"model.layers.{layer_index}"
        result.update(
            {
                f"{layer}.input_layernorm.weight": (hidden,),
                f"{layer}.post_attention_layernorm.weight": (hidden,),
                f"{layer}.self_attn.q_proj.weight": (heads * (nope + rope), hidden),
                f"{layer}.self_attn.kv_a_proj_with_mqa.weight": (rank + rope, hidden),
                f"{layer}.self_attn.kv_a_layernorm.weight": (rank,),
                f"{layer}.self_attn.kv_b_proj.weight": (heads * (nope + value), rank),
                f"{layer}.self_attn.o_proj.weight": (hidden, heads * value),
            }
        )
        mlp = f"{layer}.mlp"
        if layer_index == 0:
            width = configuration["intermediate_size"]
            result.update(
                {
                    f"{mlp}.gate_proj.weight": (width, hidden),
                    f"{mlp}.up_proj.weight": (width, hidden),
                    f"{mlp}.down_proj.weight": (hidden, width),
                }
            )
            continue
        result[f"{mlp}.gate.weight"] = (experts, hidden)
        for expert in range(experts):
            prefix = f"{mlp}.experts.{expert}"
            result.update(
                {
                    f"{prefix}.gate_proj.weight": (routed, hidden),
                    f"{prefix}.up_proj.weight": (routed, hidden),
                    f"{prefix}.down_proj.weight": (hidden, routed),
                }
            )
        result.update(
            {
                f"{mlp}.shared_experts.gate_proj.weight": (shared, hidden),
                f"{mlp}.shared_experts.up_proj.weight": (shared, hidden),
                f"{mlp}.shared_experts.down_proj.weight": (hidden, shared),
            }
        )
    return result


def inputs(
    configuration: dict[str, Any],
    *,
    missing: str | None = None,
    quantized_sample: bool = False,
    tokenizer: bool = True,
) -> AnalysisInput:
    expected = expected_parameters(configuration)
    physical = {
        name: r.ArchitectureStorage(name=name, dtype="BF16", shape=list(dimensions))
        for name, dimensions in expected.items()
        if name != missing
    }
    if quantized_sample:
        sample_prefix = "model.layers.1.self_attn.q_proj.weight"
        for name in list(physical):
            if name == sample_prefix:
                physical.pop(name)
        for name, value in FIXTURE["safetensors_header"]["selected_storage"].items():
            if name == sample_prefix or name.startswith(
                sample_prefix.removesuffix(".weight") + "."
            ):
                dtype, dimensions = value
                physical[name] = r.ArchitectureStorage(name=name, dtype=dtype, shape=dimensions)
    numeric = {
        f"synthetic-{index}": NumericTensor(f"synthetic-{index}", name, dimensions, "BF16")
        for index, (name, dimensions) in enumerate(expected.items())
        if name in physical
        and name != missing
        and not (quantized_sample and name == "model.layers.1.self_attn.q_proj.weight")
    }
    return AnalysisInput(
        "deepseek-v2-reference-fixture",
        copy.deepcopy(configuration),
        BindingContext(physical, numeric, tokenizer),
    )


def registry() -> DescriptionRegistry:
    result = DescriptionRegistry()
    register_deepseek_v2(result)
    return result


def graph_for(data: AnalysisInput, status: str = "complete") -> r.ArchitectureGraph:
    result = registry().analyze(data)
    assert result.status == status, result
    assert result.graph is not None
    return result.graph


def node_id(data: AnalysisInput, key: str) -> str:
    builder = GraphBuilder(data, PRODUCER, "language_model")
    return builder.record_id("node", key)


def node(data: AnalysisInput, graph: r.ArchitectureGraph, key: str) -> r.ArchitectureNode:
    identity = node_id(data, key)
    return next(value for value in graph.nodes if value.id == identity)


def named_edges(data: AnalysisInput, graph: r.ArchitectureGraph) -> set[tuple[str, str, str, str]]:
    keys = (
        "model",
        "lm_head",
        "model.embed_tokens",
        "model.layers.0",
        "model.layers.0.mlp",
        "model.layers.0.input_layernorm",
        "model.layers.0.self_attn",
        "model.layers.0.self_attn.q_proj",
        "model.layers.0.self_attn.q_heads",
        "model.layers.0.self_attn.q_split",
        "model.layers.0.self_attn.q_rope",
        "model.layers.0.self_attn.q_recombine",
        "model.layers.0.self_attn.kv_a_proj_with_mqa",
        "model.layers.0.self_attn.kv_a_split",
        "model.layers.0.self_attn.kv_a_layernorm",
        "model.layers.0.self_attn.kv_b_proj",
        "model.layers.0.self_attn.kv_heads",
        "model.layers.0.self_attn.kv_b_split",
        "model.layers.0.self_attn.key_cache_update",
        "model.layers.0.self_attn.value_cache_update",
        "model.layers.0.self_attn.next_key_state",
        "model.layers.0.self_attn.next_value_state",
        "model.layers.0.self_attn.key_transpose",
        "model.layers.0.self_attn.scores",
        "model.layers.0.self_attn.score_scale",
        "model.layers.0.self_attn.causal_mask",
        "model.layers.0.self_attn.softmax",
        "model.layers.0.self_attn.weighted_values",
        "model.layers.0.self_attn.output_transpose",
        "model.layers.0.self_attn.merge_heads",
        "model.layers.0.self_attn.o_proj",
        "model.layers.0.self_attn.k_pe_heads",
        "model.layers.0.self_attn.k_rope",
        "model.layers.0.self_attn.k_pe_repeat",
        "model.layers.0.self_attn.key_recombine",
        "model.layers.0.attention_residual",
        "model.layers.0.post_attention_layernorm",
        "model.layers.0.mlp.gate_proj",
        "model.layers.0.mlp.up_proj",
        "model.layers.0.mlp.silu",
        "model.layers.0.mlp.gated_product",
        "model.layers.0.mlp.down_proj",
        "model.layers.0.mlp_residual",
        "model.layers.1",
        "model.layers.1.input_layernorm",
        "model.layers.1.self_attn",
        "model.layers.1.attention_residual",
        "model.layers.1.post_attention_layernorm",
        "model.layers.1.mlp",
        "model.layers.1.mlp.gate",
        "model.layers.1.mlp.gate_softmax",
        "model.layers.1.mlp.topk",
        "model.layers.1.mlp.dispatch.0",
        "model.layers.1.mlp.experts.0",
        "model.layers.1.mlp.experts.0.swiglu_mlp",
        "model.layers.1.mlp.experts.0.routing_weight",
        "model.layers.1.mlp.routed_scatter_sum",
        "model.layers.1.mlp.shared_experts",
        "model.layers.1.mlp.shared_experts.gate_proj",
        "model.layers.1.mlp.shared_experts.up_proj",
        "model.layers.1.mlp.shared_experts.silu",
        "model.layers.1.mlp.shared_experts.gated_product",
        "model.layers.1.mlp.shared_experts.down_proj",
        "model.layers.1.mlp.shared_routed_add",
        "model.layers.1.mlp_residual",
        "model.norm",
    )
    expanded_keys = (
        *keys,
        *(
            key.replace("model.layers.0", "model.layers.1")
            for key in keys
            if key.startswith("model.layers.0.self_attn")
        ),
    )
    semantic_names = {node_id(data, key): key for key in expanded_keys}
    return {
        (
            semantic_names[edge.source.node_id],
            edge.source.port_id,
            semantic_names[edge.target.node_id],
            edge.target.port_id,
        )
        for edge in graph.edges
        if edge.source.node_id in semantic_names and edge.target.node_id in semantic_names
    }


def test_pinned_configuration_and_safe_header_oracle() -> None:
    assert checked(REFERENCE) is not None
    assert REFERENCE["model_type"] == "deepseek_v2"
    assert REFERENCE["architectures"] == [ARCHITECTURE]
    assert (
        FIXTURE["safetensors_header"]["entry_count"]
        + FIXTURE["safetensors_header"]["second_entry_count"]
        == FIXTURE["safetensors_header"]["index_total_tensor_names"]
    )
    sample = FIXTURE["safetensors_header"]["selected_storage"]
    assert sample["model.layers.1.self_attn.q_proj.weight"] == ["U8", [3145728, 1]]
    assert sample["model.layers.1.self_attn.q_proj.weight.absmax"] == ["U8", [98304]]
    assert sample["model.layers.1.self_attn.q_proj.weight.quant_map"] == ["F32", [16]]
    assert sample["model.layers.1.self_attn.q_proj.weight.nested_quant_map"] == [
        "F32",
        [256],
    ]
    assert FIXTURE["safetensors_header"]["selected_second_storage"][
        "model.layers.26.mlp.experts.63.down_proj.weight"
    ] == ["U8", [1441792, 1]]
    assert checked(native_configuration()) is not None


@pytest.mark.parametrize(
    "field,mutate",
    [
        ("n_routed_experts", lambda c: c.__setitem__("n_routed_experts", 32)),
        ("num_experts_per_tok", lambda c: c.__setitem__("num_experts_per_tok", 8)),
        ("topk_method", lambda c: c.__setitem__("topk_method", "group_limited_greedy")),
        ("qk_rope_head_dim", lambda c: c.__setitem__("qk_rope_head_dim", 32)),
        ("q_lora_rank", lambda c: c.__setitem__("q_lora_rank", 1536)),
        ("ep_size", lambda c: c.__setitem__("ep_size", 2)),
        ("unreviewed option", lambda c: c.__setitem__("expert_dropout", 0.1)),
    ],
)
def test_unknown_or_contradictory_configuration_is_not_selected(field: str, mutate: Any) -> None:
    configuration = copy.deepcopy(REFERENCE)
    mutate(configuration)
    data = AnalysisInput("negative", configuration, BindingContext({}, {}, False))
    assert not supports(data), field
    assert registry().select(data) is None


def test_complete_native_graph_has_mla_moe_and_every_concrete_instance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data = inputs(native_configuration())
    expected = expected_parameters(native_configuration())

    # Static analysis is metadata-only. Any model, tracing, network or tensor read fails here.
    denied = Mock(side_effect=AssertionError("Architecture analysis executed or read weights"))
    import safetensors
    import safetensors.torch
    import transformers

    for owner, names in [
        (socket, ["socket", "create_connection"]),
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
            ("modeling_deepseek", "configuration_deepseek", "transformers_modules")
        )
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)

    from llm_model_explorer.architecture_analysis.compact import expand_graph

    wire = graph_for(data)
    graph = expand_graph(wire)
    denied.assert_not_called()
    assert graph.coverage == "complete"
    assert len(graph.parameters) == len(expected) == 5291
    assert {parameter.name for parameter in graph.parameters} == set(expected)
    assert {storage.name for parameter in graph.parameters for storage in parameter.storage} == set(
        expected
    )
    parameters = {parameter.name: parameter for parameter in graph.parameters}
    for name, dimensions in expected.items():
        assert parameters[name].binding == "native"
        assert parameters[name].logical_shape == [
            r.ArchitectureConstantDimension(kind="constant", value=value) for value in dimensions
        ]
        assert parameters[name].inspection.status == "available"

    layer_repeat = next(
        repetition for repetition in graph.repetitions if repetition.label == "Decoder layers"
    )
    assert [instance.index for instance in layer_repeat.instances] == list(range(27))
    assert [instance.variant for instance in layer_repeat.instances] == ["dense"] + ["moe"] * 26
    expert_repeats = [
        repetition for repetition in graph.repetitions if repetition.label == "Routed experts"
    ]
    assert len(expert_repeats) == 26
    assert all(
        [instance.index for instance in repetition.instances] == list(range(64))
        for repetition in expert_repeats
    )
    nodes_by_id = {value.id: value for value in graph.nodes}
    assert all(
        isinstance(nodes_by_id[instance.node_id], r.ArchitectureGroupNode)
        and nodes_by_id[instance.node_id].label == f"Routed expert {instance.index}"
        for repetition in expert_repeats
        for instance in repetition.instances
    )
    assert (
        sum(name.startswith("model.layers.") and ".mlp.experts." in name for name in expected)
        == 26 * 64 * 3
    )
    assert expected["model.layers.1.mlp.shared_experts.gate_proj.weight"] == (2816, 2048)
    assert expected["model.layers.1.mlp.shared_experts.down_proj.weight"] == (2048, 2816)

    edges = named_edges(data, graph)
    attention = "model.layers.1.self_attn"
    for source, target, source_port, target_port in [
        (attention + ".q_proj", attention + ".q_heads", "out", "x"),
        (attention + ".q_heads", attention + ".q_split", "out", "x"),
        (attention + ".q_split", attention + ".q_recombine", "non_rotary", "non_rotary"),
        (attention + ".q_split", attention + ".q_rope", "rotary", "x"),
        (attention + ".q_rope", attention + ".q_recombine", "out", "rotary"),
        (attention + ".kv_a_proj_with_mqa", attention + ".kv_a_split", "out", "x"),
        (attention + ".kv_a_split", attention + ".kv_a_layernorm", "latent", "x"),
        (attention + ".kv_a_layernorm", attention + ".kv_b_proj", "out", "x"),
        (attention + ".kv_b_proj", attention + ".kv_heads", "out", "x"),
        (attention + ".kv_heads", attention + ".kv_b_split", "out", "x"),
        (attention + ".kv_a_split", attention + ".k_pe_heads", "rotary_key", "x"),
        (attention + ".k_pe_heads", attention + ".k_rope", "out", "x"),
        (attention + ".k_rope", attention + ".k_pe_repeat", "out", "x"),
        (attention + ".kv_b_split", attention + ".key_recombine", "key_non_rotary", "non_rotary"),
        (attention + ".k_pe_repeat", attention + ".key_recombine", "out", "rotary"),
        (attention + ".key_recombine", attention + ".key_transpose", "out", "x"),
        (attention + ".weighted_values", attention + ".output_transpose", "out", "x"),
        (attention + ".output_transpose", attention + ".merge_heads", "out", "x"),
        (attention + ".merge_heads", attention + ".o_proj", "out", "x"),
    ]:
        assert (source, source_port, target, target_port) in edges

    query_projection = node(data, graph, attention + ".q_proj")
    query_projection_attributes = {
        attribute.name: attribute.value for attribute in query_projection.attributes
    }
    assert query_projection.operation == "linear"
    assert query_projection_attributes["q_lora_rank"] == "null; direct query projection"
    scaling = node(data, graph, attention + ".score_scale")
    assert scaling.formula == "out = factor * x"
    scaling_attributes = {attribute.name: attribute.value for attribute in scaling.attributes}
    # Independently evaluated at high decimal precision from the pinned source:
    # (1 + 0.1 * 0.707 * ln(40))² / sqrt(192).
    assert scaling_attributes["factor"] == pytest.approx(0.1147213867929261, rel=1e-14)
    assert not any("q_a_proj" in name or "q_b_proj" in name for name in expected)
    topk = node(data, graph, "model.layers.1.mlp.topk")
    topk_attributes = {attribute.name: attribute.value for attribute in topk.attributes}
    assert topk_attributes["top_k"] == 6.0
    assert topk_attributes["norm_topk_prob"] is False
    assert topk_attributes["routed_scaling_factor"] == 1.0
    assert "because norm_topk_prob=false" in (topk.formula or "")
    rotary = node(data, graph, "model.layers.1.self_attn.q_rope")
    rotary_attributes = {attribute.name: attribute.value for attribute in rotary.attributes}
    assert rotary_attributes["rope_type"] == "yarn"
    assert rotary_attributes["rotary_width"] == 64.0

    dense = "model.layers.0.mlp"
    for source, target, source_port, target_port in [
        ("model.layers.0", "model.layers.0.input_layernorm", "x", "x"),
        ("model.layers.0.input_layernorm", "model.layers.0.self_attn", "out", "x"),
        ("model.layers.0", "model.layers.0.attention_residual", "x", "skip"),
        ("model.layers.0.self_attn.o_proj", "model.layers.0.self_attn", "out", "out"),
        ("model.layers.0.self_attn", "model.layers.0.attention_residual", "out", "branch"),
        (
            "model.layers.0.attention_residual",
            "model.layers.0.post_attention_layernorm",
            "out",
            "x",
        ),
        ("model.layers.0.attention_residual", "model.layers.0.mlp_residual", "out", "skip"),
        ("model.layers.0.mlp_residual", "model.layers.0", "out", "out"),
        ("model.layers.0", "model.layers.1", "out", "x"),
    ]:
        assert (source, source_port, target, target_port) in edges
    for source, target, source_port, target_port in [
        (dense, dense + ".gate_proj", "x", "x"),
        (dense + ".gate_proj", dense + ".silu", "out", "x"),
        (dense + ".silu", dense + ".gated_product", "out", "gate"),
        (dense + ".up_proj", dense + ".gated_product", "out", "up"),
        (dense + ".gated_product", dense + ".down_proj", "out", "x"),
        (dense + ".down_proj", dense, "out", "out"),
        (dense, "model.layers.0.mlp_residual", "out", "branch"),
    ]:
        assert (source, source_port, target, target_port) in edges

    moe = "model.layers.1.mlp"
    for source, target, source_port, target_port in [
        (moe, moe + ".gate", "x", "x"),
        (moe + ".gate", moe + ".gate_softmax", "out", "logits"),
        (moe + ".gate_softmax", moe + ".topk", "out", "scores"),
        (moe + ".topk", moe + ".dispatch.0", "expert_indices", "expert_indices"),
        (moe + ".dispatch.0", moe + ".experts.0", "tokens", "x"),
        (moe + ".experts.0.swiglu_mlp", moe + ".experts.0.routing_weight", "out", "expert_values"),
        (moe + ".experts.0.routing_weight", moe + ".experts.0", "out", "out"),
        (moe + ".experts.0", moe + ".routed_scatter_sum", "out", "values_0"),
        (moe + ".routed_scatter_sum", moe + ".shared_routed_add", "out", "routed"),
        (moe + ".shared_experts", moe + ".shared_routed_add", "out", "shared"),
        (moe + ".shared_routed_add", moe, "out", "out"),
        (moe, "model.layers.1.mlp_residual", "out", "branch"),
    ]:
        assert (source, source_port, target, target_port) in edges
    expert_mlp = node(data, graph, moe + ".experts.0.swiglu_mlp")
    parameter_names = {parameter.id: parameter.name for parameter in graph.parameters}
    assert {parameter_names[parameter_id] for parameter_id in expert_mlp.parameter_ids} == {
        moe + ".experts.0.gate_proj.weight",
        moe + ".experts.0.up_proj.weight",
        moe + ".experts.0.down_proj.weight",
    }
    assert expert_mlp.formula == "down_proj(silu(gate_proj(x)) * up_proj(x))"

    root = node(data, graph, "model")
    assert isinstance(root, r.ArchitectureGroupNode)
    assert {port.id for port in root.ports if port.direction == "input"} == {
        "tokens",
        "positions",
        "mask",
    }
    assert {port.id for port in root.ports if port.direction == "output"} == {"logits"}
    assert not any(
        n.operation in {"select_layer_state", "state_concat", "stack_layer_states"}
        for n in graph.nodes
    )
    assert {a.name: a.value for a in root.attributes}[
        "invocation"
    ] == "full_sequence_use_cache_false_past_none"
    prepare = next(n for n in graph.nodes if n.operation == "generation_prepare_inputs")
    assert next(p for p in prepare.ports if p.id == "mask").shape == [
        r.ArchitectureSymbolDimension(kind="symbol", name="B"),
        r.ArchitectureConstantDimension(kind="constant", value=1),
        r.ArchitectureSymbolDimension(kind="symbol", name="S"),
        r.ArchitectureSymbolDimension(kind="symbol", name="S"),
    ]
    assert any(
        isinstance(reference, r.ArchitectureTokenizerReference)
        for entry in graph.nodes
        for reference in entry.references
    )
    tokenizer = next(entry for entry in graph.nodes if entry.kind == "context")
    assert tokenizer.parent_id is None and tokenizer.id not in root.children
    assert not any(
        edge.source.node_id == tokenizer.id or edge.target.node_id == tokenizer.id
        for edge in graph.edges
    )

    attention_templates = [
        template for template in graph.templates or [] if template.component_role == "attention"
    ]
    assert len(attention_templates) == 1
    assert len(attention_templates[0].instances) == 27
    assert not any(diagnostic.code == "templates_omitted" for diagnostic in graph.diagnostics)
    encoded = serialize_graph(wire)
    assert len(encoded) < MAX_BYTES
    assert expand_graph(parse_graph(wire.document(), data.bindings)) == graph


def test_sampled_nf4_binding_and_missing_expert_are_localized() -> None:
    missing = "model.layers.26.mlp.experts.63.down_proj.weight"
    data = inputs(REFERENCE, missing=missing, quantized_sample=True, tokenizer=False)
    graph = graph_for(data, "partial")
    parameters = {parameter.name: parameter for parameter in graph.parameters}
    packed = parameters["model.layers.1.self_attn.q_proj.weight"]
    assert packed.binding == "quantized"
    assert packed.inspection.status == "unavailable"
    assert packed.inspection.reason == "unsupported_representation"
    assert {value.name: value.role for value in packed.storage} == {
        name: "packed_weight"
        if name == "model.layers.1.self_attn.q_proj.weight"
        else "quantization_auxiliary"
        for name in FIXTURE["safetensors_header"]["selected_storage"]
        if name == "model.layers.1.self_attn.q_proj.weight"
        or name.startswith("model.layers.1.self_attn.q_proj.")
    }
    unresolved = parameters[missing]
    assert unresolved.binding == "unresolved"
    assert any(diagnostic.parameter_id == unresolved.id for diagnostic in graph.diagnostics)
    assert not any(diagnostic.code == "unrecognized_storage" for diagnostic in graph.diagnostics)
    assert {storage.name for parameter in graph.parameters for storage in parameter.storage} == set(
        data.bindings.physical
    )
    assert len(serialize_graph(graph)) < MAX_BYTES


def test_full_nf4_inventory_retains_layer_templates_within_response_budget() -> None:
    # The pinned checkpoint quantizes every projection except embeddings, the
    # vocabulary head and routers. Native-only/sample-packed fixtures miss the
    # metadata cost of all 5,181 matrices and their five encoding companions.
    data = inputs(REFERENCE)
    physical = dict(data.bindings.physical)
    numeric = {}
    expected = expected_parameters(REFERENCE)
    for index, (name, dimensions) in enumerate(expected.items()):
        tensor_id = f"{index:064x}"  # Match real inventory ID lengths in the byte budget.
        packed = (
            len(dimensions) == 2
            and name not in {"model.embed_tokens.weight", "lm_head.weight"}
            and not name.endswith(".mlp.gate.weight")
        )
        if packed:
            count = math.prod(dimensions)
            blocks = (count + 63) // 64
            records = {
                "": ("U8", [(count + 1) // 2, 1]),
                ".absmax": ("U8", [blocks]),
                ".quant_map": ("F32", [16]),
                ".nested_absmax": ("F32", [(blocks + 255) // 256]),
                ".nested_quant_map": ("F32", [256]),
                ".quant_state.bitsandbytes__nf4": ("U8", [172]),
            }
            for suffix, (dtype, dimensions_in_storage) in records.items():
                storage_name = name + suffix
                physical[storage_name] = r.ArchitectureStorage(
                    name=storage_name, dtype=dtype, shape=dimensions_in_storage
                )
        numeric[tensor_id] = NumericTensor(
            tensor_id,
            name,
            dimensions,
            "U8" if packed else "BF16",
            "bnb-nf4-dq" if packed else "safetensors",
        )
    assert len(physical) == 31_196  # Pinned two-shard header/index count.
    data = replace(data, bindings=replace(data.bindings, physical=physical, numeric=numeric))
    model_id = "basicv8vc/DeepSeek-V2-Lite-bnb-4bit@bnb-nf4-dq"
    result = registry().analyze(data, byte_limit=ArchitectureService.response_budget(model_id))
    assert result.status == "complete", result
    graph = result.graph
    assert graph is not None
    assert not any(d.code == "templates_omitted" for d in graph.diagnostics)
    families = {t.component_role: t for t in graph.templates or []}
    assert len(families["attention"].instances) == 27
    assert len(families["layer"].instances) == 26
    assert sum(len(f.instances) for f in graph.compact_components or []) == 26 * 64
    assert {p.name for p in graph.parameters} == set(expected)
    assert {s.name for p in graph.parameters for s in p.storage} == set(physical)
    by_name = {t.name: t for t in numeric.values()}
    for parameter in graph.parameters:
        tensor = by_name[parameter.name]
        assert parameter.inspection == r.ArchitectureAvailableInspection(
            status="available", tensor_id=tensor.id
        )
        assert parameter.logical_shape == [
            r.ArchitectureConstantDimension(kind="constant", value=d) for d in tensor.shape
        ]
        assert {p.source for p in parameter.provenance if p.kind == "storage"} == {
            s.name for s in parameter.storage
        }

    fields_by_parameter = {
        "model.embed_tokens.weight": {"hidden_size", "vocab_size"},
        "lm_head.weight": {"hidden_size", "vocab_size"},
        "model.norm.weight": {"hidden_size"},
        "model.layers.0.input_layernorm.weight": {"hidden_size"},
        "model.layers.0.post_attention_layernorm.weight": {"hidden_size"},
        "model.layers.0.self_attn.q_proj.weight": {
            "hidden_size",
            "num_attention_heads",
            "qk_nope_head_dim",
            "qk_rope_head_dim",
        },
        "model.layers.0.self_attn.kv_a_proj_with_mqa.weight": {
            "hidden_size",
            "kv_lora_rank",
            "qk_rope_head_dim",
        },
        "model.layers.0.self_attn.kv_a_layernorm.weight": {"kv_lora_rank"},
        "model.layers.0.self_attn.kv_b_proj.weight": {
            "kv_lora_rank",
            "num_attention_heads",
            "qk_nope_head_dim",
            "v_head_dim",
        },
        "model.layers.0.self_attn.o_proj.weight": {
            "hidden_size",
            "num_attention_heads",
            "v_head_dim",
        },
        "model.layers.1.mlp.gate.weight": {"hidden_size", "n_routed_experts"},
        "model.layers.0.mlp.gate_proj.weight": {"hidden_size", "intermediate_size"},
        "model.layers.1.mlp.experts.0.up_proj.weight": {"hidden_size", "moe_intermediate_size"},
        "model.layers.26.mlp.shared_experts.down_proj.weight": {
            "hidden_size",
            "moe_intermediate_size",
            "n_shared_experts",
        },
    }
    parameters = {p.name: p for p in graph.parameters}
    for name, fields in fields_by_parameter.items():
        parameter = parameters[name]
        if parameter.binding == "quantized":
            fields = fields | {"quantization_config"}
        assert {p.source for p in parameter.provenance if p.kind == "configuration"} == {
            "config.json#/" + field for field in fields
        }, name
        assert any(
            p.kind == "description"
            and p.source == PRODUCER.description
            and p.revision == PRODUCER.revision
            for p in parameter.provenance
        )

    encoded = serialize_graph(graph)
    envelope = ArchitectureService._prefix(model_id) + encoded + b"}"
    assert len(envelope) <= MAX_BYTES
    assert parse_graph(json.loads(encoded), data.bindings) == graph

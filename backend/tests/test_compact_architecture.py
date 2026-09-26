"""Routed expert wire compaction preserves source records and exact bindings."""

from __future__ import annotations

import importlib
import math
from dataclasses import replace

import pytest

from llm_model_explorer.architecture_analysis import parse_graph
from llm_model_explorer.architecture_analysis import records as r
from llm_model_explorer.architecture_analysis.compact import compact_graph, expand_graph
from llm_model_explorer.architecture_analysis.validation import (
    GraphError,
    NumericTensor,
    serialized_size,
)


@pytest.mark.parametrize(
    ("module", "expected_instances"),
    [
        ("test_deepseek_v2_architecture", 26 * 64),
        ("test_glm4_moe_lite_architecture", 46 * 64),
        ("test_kimi_linear_architecture", 26 * 256),
    ],
)
def test_compact_experts_reconstruct_every_instance_and_detect_bad_bindings(
    module: str, expected_instances: int
) -> None:
    fixture = importlib.import_module(module)
    data = (
        fixture.inputs(fixture.REFERENCE)
        if module.endswith("deepseek_v2_architecture")
        else fixture.inputs()
    )
    result = fixture.registry().analyze(data, byte_limit=25_000_000)
    assert result.graph is not None, result
    compact = result.graph
    assert compact.compact_components
    assert sum(len(family.instances) for family in compact.compact_components) == expected_instances
    assert serialized_size(compact.document()) <= 25_000_000

    expanded = expand_graph(compact)
    assert len({node.id for node in expanded.nodes}) == len(expanded.nodes)
    assert len({edge.id for edge in expanded.edges}) == len(expanded.edges)
    assert (
        len(
            [
                instance
                for rep in expanded.repetitions
                for instance in rep.instances
                if instance.variant in {"routed_expert", "routed_swiglu"}
            ]
        )
        == expected_instances
    )
    assert parse_graph(compact.document(), data.bindings).document() == compact.document()

    bad = compact.document()
    family = bad["compact_components"][0]
    family["instances"][1]["parameter_ids"][0] = family["instances"][0]["parameter_ids"][0]
    with pytest.raises(GraphError, match="missing or swapped"):
        parse_graph(bad, data.bindings)

    missing = compact.document()
    missing["compact_components"][0]["instances"][1]["parameter_ids"].pop()
    with pytest.raises(GraphError, match="mapping length"):
        parse_graph(missing, data.bindings)

    if module.endswith("kimi_linear_architecture"):
        # A semantic exception in one member keeps that whole layer concrete.
        source = fixture.registry().analyze(data).graph
        assert source is not None and not source.compact_components
        expert_rep = next(
            rep for rep in source.repetitions if rep.instances[0].variant == "routed_swiglu"
        )
        exceptional_id = expert_rep.instances[1].node_id
        changed = source.model_copy(
            update={
                "nodes": [
                    node.model_copy(update={"operation": "different_expert"})
                    if node.id == exceptional_id
                    else node
                    for node in source.nodes
                ]
            }
        )
        exceptional = compact_graph(changed)
        assert exceptional_id in {node.id for node in exceptional.nodes}


def test_compact_nf4_experts_keep_distinct_admitted_logical_tensors() -> None:
    fixture = importlib.import_module("test_deepseek_v2_architecture")
    data = fixture.inputs(fixture.REFERENCE)
    physical = dict(data.bindings.physical)
    numeric = dict(data.bindings.numeric)
    names = [f"model.layers.1.mlp.experts.{index}.gate_proj.weight" for index in (0, 1)]
    for index, name in enumerate(names):
        dims = fixture.expected_parameters(fixture.REFERENCE)[name]
        output, inputs = dims
        count = output * inputs
        prefix = name.removesuffix(".weight")
        physical.pop(name)
        numeric = {key: value for key, value in numeric.items() if value.name != name}
        companions = {
            ".weight": ("U8", [(count + 1) // 2, 1]),
            ".weight.absmax": ("U8", [math.ceil(count / 64)]),
            ".weight.quant_map": ("F32", [16]),
            ".weight.nested_absmax": ("F32", [math.ceil(math.ceil(count / 64) / 256)]),
            ".weight.nested_quant_map": ("F32", [256]),
            ".weight.quant_state.bitsandbytes__nf4": ("U8", [128]),
        }
        for suffix, (dtype, shape) in companions.items():
            storage_name = prefix + suffix
            physical[storage_name] = r.ArchitectureStorage(
                name=storage_name, dtype=dtype, shape=shape
            )
        numeric[f"nf4-expert-{index}"] = NumericTensor(
            f"nf4-expert-{index}", name, dims, "U8", "bnb-nf4-dq"
        )
    data = replace(data, bindings=replace(data.bindings, physical=physical, numeric=numeric))
    result = fixture.registry().analyze(data, byte_limit=25_000_000)
    assert result.status == "complete", result
    assert result.graph is not None
    graph = result.graph
    parameters = {parameter.name: parameter for parameter in graph.parameters}
    for index, name in enumerate(names):
        assert parameters[name].binding == "quantized"
        assert parameters[name].inspection == r.ArchitectureAvailableInspection(
            status="available", tensor_id=f"nf4-expert-{index}"
        )
    family = next(
        family
        for family in graph.compact_components or []
        if family.instances[0].prefix == "model.layers.1.mlp.experts.0"
    )
    for index, name in enumerate(names):
        assert parameters[name].id in family.instances[index].parameter_ids
    assert parse_graph(graph.document(), data.bindings) == graph

    swapped = graph.document()
    swapped_family = next(
        family
        for family in swapped["compact_components"]
        if family["instances"][0]["prefix"] == "model.layers.1.mlp.experts.0"
    )
    swapped_family["instances"][1]["parameter_ids"][0] = swapped_family["instances"][0][
        "parameter_ids"
    ][0]
    with pytest.raises(GraphError, match="missing or swapped"):
        parse_graph(swapped, data.bindings)

    deleted = graph.document()
    deleted_family = next(
        family
        for family in deleted["compact_components"]
        if family["instances"][0]["prefix"] == "model.layers.1.mlp.experts.0"
    )
    deleted_family["instances"][1]["parameter_ids"].pop()
    with pytest.raises(GraphError, match="mapping length"):
        parse_graph(deleted, data.bindings)

    wrong_tensor = graph.document()
    second = next(
        parameter for parameter in wrong_tensor["parameters"] if parameter["name"] == names[1]
    )
    second["inspection"]["tensor_id"] = "nf4-expert-0"
    with pytest.raises(GraphError, match="outside the admitted numeric inventory"):
        parse_graph(wrong_tensor, data.bindings)

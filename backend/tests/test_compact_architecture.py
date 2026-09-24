"""Routed expert wire compaction preserves source records and exact bindings."""

from __future__ import annotations

import importlib

import pytest

from llm_model_explorer.architecture_analysis import parse_graph
from llm_model_explorer.architecture_analysis.compact import compact_graph, expand_graph
from llm_model_explorer.architecture_analysis.validation import GraphError, serialized_size


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

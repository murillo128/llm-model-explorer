"""Exact optional roles, independent source identity, and removable annotations."""

import copy
from dataclasses import replace
from unittest.mock import patch

import pytest
from architecture_assertions import semantic_key
from architecture_grouping_cases import cases
from test_qwen35_architecture import TINY, metadata

from llm_model_explorer.architecture_analysis import (
    AnalysisInput,
    DescriptionRegistry,
    GraphBuilder,
)
from llm_model_explorer.architecture_analysis import records as r
from llm_model_explorer.architecture_analysis.templates import ComponentTemplates
from llm_model_explorer.architecture_analysis.validation import serialized_size


def build(inputs: AnalysisInput, registry: DescriptionRegistry) -> GraphBuilder:
    description = registry.select(inputs)
    assert description is not None
    builder = GraphBuilder(inputs, description.producer, description.scope)
    description.build(inputs, builder)
    return builder


@pytest.mark.parametrize("name,inputs,registry", cases(), ids=[c[0] for c in cases()])
def test_annotations_preserve_every_ordinary_record_and_exact_parameter(
    name: str, inputs: AnalysisInput, registry: DescriptionRegistry
) -> None:
    annotated = build(inputs, registry).finish()
    with patch.object(ComponentTemplates, "annotate", side_effect=lambda graph, _: graph):
        ordinary = build(inputs, registry).finish()
    document = annotated.document()
    assert document.pop("templates", [])
    assert document == ordinary.document()
    nodes = {n.id: n for n in annotated.nodes}
    parameters = {p.id: p for p in annotated.parameters}
    for template in annotated.templates or []:
        assert len(template.instances) >= 2
        for instance in template.instances:
            group_key = semantic_key(nodes[instance.node_id])
            # The V-JEPA description intentionally owns siblings of its module prefix.
            base = group_key.rsplit(".", 1)[0] if name.startswith("visual") else group_key
            mapped = {m.role: m.parameter_id for m in instance.parameters}
            assert mapped
            for role, parameter_id in mapped.items():
                assert parameters[parameter_id].name == base + "." + role
            if not name.startswith("visual"):
                projection = next(
                    m for m in instance.nodes if m.role in {"q_proj", "gate_proj", "in_proj_qkv"}
                )
                assert (
                    nodes[projection.node_id].parameter_ids[0]
                    == mapped[projection.role + ".weight"]
                )
        # Instance storage is kept verbatim; it is never replaced with the first member.
        assert set(m.parameter_id for i in template.instances for m in i.parameters) == {
            p for i in template.instances for m in i.nodes for p in nodes[m.node_id].parameter_ids
        }


def test_hybrid_nonconsecutive_and_independent_visual_scopes() -> None:
    for name, inputs, registry in cases():
        if name not in {"hybrid", "visual"}:
            continue
        if name == "hybrid":
            fixture = copy.deepcopy(TINY)
            config = fixture["configuration"]["text_config"]
            config["num_hidden_layers"] = 3
            config["layer_types"] = ["linear_attention", "full_attention", "linear_attention"]
            fixture["storage"].update(
                {
                    key.replace(".layers.0.", ".layers.2."): value
                    for key, value in list(fixture["storage"].items())
                    if ".layers.0." in key
                }
            )
            inputs = metadata(fixture)
        graph = build(inputs, registry).finish()
        nodes = {n.id: n for n in graph.nodes}
        templates = graph.templates or []
        if name == "hybrid":
            linear = next(t for t in templates if t.label == "linear attention")
            keys = [semantic_key(nodes[i.node_id]) for i in linear.instances]
            assert keys == [
                "model.language_model.layers.0.linear_attn",
                "model.language_model.layers.2.linear_attn",
            ]
            assert all("self_attn" not in semantic_key(nodes[i.node_id]) for i in linear.instances)
        else:
            for template in templates:
                keys = [semantic_key(nodes[i.node_id]) for i in template.instances]
                assert len({key.split(".layer.")[0] for key in keys}) == 1


def test_same_shaped_qk_swap_is_not_a_family_and_does_not_damage_base_graph() -> None:
    _, inputs, registry = cases()[0]
    builder = build(inputs, registry)
    keys = {semantic_key(n): n for n in builder._nodes}
    q = keys["model.layers.1.self_attn.qk_product"]
    # Replace the two same-shaped input endpoints; ordinary graph remains valid.
    for index, edge in enumerate(builder._edges):
        if edge.target.node_id == q.id:
            port = "kt" if edge.target.port_id == "q" else "q"
            builder._edges[index] = edge.model_copy(
                update={"target": edge.target.model_copy(update={"port_id": port})}
            )
    graph = builder.finish()
    assert graph.coverage == "complete"
    assert [t.component_role for t in graph.templates or []] == ["mlp"]


@pytest.mark.parametrize("change", ["bias", "normalization", "formula", "missing-role"])
def test_unverified_candidates_are_omitted_without_downgrading_coverage(change: str) -> None:
    _, inputs, registry = cases()[0]
    builder = build(inputs, registry)
    for position, node in enumerate(builder._nodes):
        if semantic_key(node) != "model.layers.1.self_attn.q_proj":
            continue
        if change == "missing-role":
            candidate = builder.templates.candidates[node.parent_id or ""]
            candidate.nodes[:] = [m for m in candidate.nodes if m.node_id != node.id]
        elif change == "formula":
            builder._nodes[position] = node.model_copy(update={"formula": "W x"})
        else:
            attrs = [a for a in node.attributes if a.name != "bias"]
            attrs.append(r.ArchitectureAttribute(name=change, value=True, provenance=[]))
            builder._nodes[position] = node.model_copy(update={"attributes": attrs})
    graph = builder.finish()
    assert graph.coverage == "complete"
    assert [t.component_role for t in graph.templates or []] == ["mlp"]


def test_optional_budget_exhaustion_retains_exact_ordinary_graph() -> None:
    _, inputs, registry = cases()[0]
    with patch.object(ComponentTemplates, "annotate", side_effect=lambda graph, _: graph):
        ordinary = build(inputs, registry).finish()
    builder = build(inputs, registry)
    builder.byte_limit = serialized_size(ordinary.document())
    actual = builder.finish()
    assert actual.document() == ordinary.document()
    assert actual.coverage == "complete"


def test_singletons_remain_ordinary_components() -> None:
    _, inputs, registry = cases()[0]
    inputs = replace(inputs, configuration=dict(inputs.configuration) | {"num_hidden_layers": 1})
    # Extra physical weights are deliberately excluded; unused storage would be partial.
    inputs = replace(
        inputs,
        bindings=replace(
            inputs.bindings,
            physical={k: v for k, v in inputs.bindings.physical.items() if ".layers.1." not in k},
            numeric={
                k: v for k, v in inputs.bindings.numeric.items() if ".layers.1." not in v.name
            },
        ),
    )
    graph = build(inputs, registry).finish()
    assert graph.templates is None
    assert graph.coverage == "complete"

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
            base = (
                group_key.rsplit(".", 1)[0]
                if name.startswith("visual") and template.component_role != "layer"
                else group_key
            )
            mapped = {m.role: m.parameter_id for m in instance.parameters}
            assert mapped
            for role, parameter_id in mapped.items():
                assert parameters[parameter_id].name == base + "." + role
            if not name.startswith("visual") and template.component_role != "layer":
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
            builder.templates.keys[node.id] = None
        elif change == "formula":
            builder._nodes[position] = node.model_copy(update={"formula": "W x"})
        else:
            attrs = [a for a in node.attributes if a.name != "bias"]
            attrs.append(r.ArchitectureAttribute(name=change, value=True, provenance=[]))
            builder._nodes[position] = node.model_copy(update={"attributes": attrs})
    graph = builder.finish()
    assert graph.coverage == "complete"
    assert [t.component_role for t in graph.templates or []] == ["mlp"]


@pytest.mark.parametrize("budget", ["byte_limit", "response_limit"])
def test_optional_budget_exhaustion_retains_exact_ordinary_graph(budget: str) -> None:
    _, inputs, registry = cases()[0]
    with patch.object(ComponentTemplates, "annotate", side_effect=lambda graph, _: graph):
        ordinary = build(inputs, registry).finish()
    builder = build(inputs, registry)
    setattr(builder, budget, serialized_size(ordinary.document()))
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


@pytest.mark.parametrize("budget", ["byte_limit", "response_limit"])
def test_multiple_template_array_separator_budget_boundaries(budget: str) -> None:
    _, inputs, registry = cases()[0]
    complete = build(inputs, registry).finish()
    assert len(complete.templates or []) == 3
    full_size = serialized_size(complete.document())
    # Independent serialization counts the enclosing array's separators too.
    # At every boundary an ordinary graph must remain usable and within budget.
    for limit in range(full_size - 4, full_size + 2):
        builder = build(inputs, registry)
        setattr(builder, budget, limit)
        actual = builder.finish()
        assert actual.coverage == complete.coverage
        assert actual.nodes == complete.nodes and actual.edges == complete.edges
        assert actual.parameters == complete.parameters
        assert serialized_size(actual.document()) <= limit
        if budget == "response_limit":
            assert any(t.component_role == "layer" for t in actual.templates or [])
        if limit >= full_size:
            assert actual.templates == complete.templates


def test_whole_layer_closure_keeps_nested_families_and_exact_roles() -> None:
    _, inputs, registry = cases()[0]
    graph = build(inputs, registry).finish()
    families = {t.component_role: t for t in graph.templates or []}
    assert set(families) == {"layer", "attention", "mlp"}
    layer = families["layer"].instances[1]
    nodes = {n.id: n for n in graph.nodes}
    parameters = {p.id: p for p in graph.parameters}
    assert semantic_key(nodes[layer.node_id]) == "model.layers.1"
    mapped = {m.role: m.node_id for m in layer.nodes}
    assert mapped["self_attn"] == families["attention"].instances[1].node_id
    assert mapped["mlp"] == families["mlp"].instances[1].node_id
    assert {p.role for p in layer.ports if p.node_id == layer.node_id} == {
        "component.x",
        "component.cos",
        "component.sin",
        "component.mask",
        "component.out",
    }
    roles = {m.role: parameters[m.parameter_id].name for m in layer.parameters}
    assert roles["self_attn.q_proj.weight"] == "model.layers.1.self_attn.q_proj.weight"
    assert roles["input_layernorm.weight"] == "model.layers.1.input_layernorm.weight"
    assert "component.x~input_layernorm.x" in {e.role for e in layer.edges}
    assert nodes[layer.node_id].parameter_ids == []


def test_hybrid_generation_uses_text_thw_and_distinct_current_masks() -> None:
    _, inputs, registry = next(case for case in cases() if case[0] == "hybrid")
    graph = build(inputs, registry).finish()
    prepare = next((n for n in graph.nodes if n.operation == "generation_prepare_inputs"), None)
    assert prepare is not None, "The native Qwen3.5 producer must deliver Generation"
    shapes = {p.id: p.shape for p in prepare.ports}
    batch = r.ArchitectureSymbolDimension(kind="symbol", name="B")
    length = r.ArchitectureSymbolDimension(kind="symbol", name="T")
    assert shapes["tokens"] == [batch, length]
    assert shapes["positions"] == [
        r.ArchitectureConstantDimension(kind="constant", value=3),
        batch,
        length,
    ]
    assert shapes["mask"] == shapes["current_mask"] == [batch, length]
    select = next(n for n in graph.nodes if n.operation == "generation_greedy_next_token")
    assert select.ports[0].shape is not None
    assert select.ports[0].shape[:2] == [batch, length]
    from llm_model_explorer.architecture_analysis.generation import validate_generation
    from llm_model_explorer.architecture_analysis.validation import GraphError

    initializer = next(n for n in graph.nodes if n.operation == "initial_delta_state")
    for value in ("carry_previous_call", "absent_per_call"):
        malformed = graph.model_copy(
            update={
                "nodes": [
                    n.model_copy(
                        update={
                            "attributes": [
                                a.model_copy(update={"value": value})
                                if a.name == "initialization"
                                else a
                                for a in n.attributes
                            ]
                        }
                    )
                    if n.id == initializer.id
                    else n
                    for n in graph.nodes
                ]
            }
        )
        with pytest.raises(GraphError, match="initializer"):
            validate_generation(malformed)


def periodic_qwen(pattern: list[str]) -> GraphBuilder:
    from test_qwen35_architecture import registry

    tiny = copy.deepcopy(TINY)
    tiny["configuration"]["text_config"].update(layer_types=pattern, num_hidden_layers=len(pattern))
    original = tiny["storage"]
    tiny["storage"] = {k: v for k, v in original.items() if ".layers." not in k}
    for index, variant in enumerate(pattern):
        source = 1 if variant == "full_attention" else 0
        tiny["storage"].update(
            {
                k.replace(f".layers.{source}.", f".layers.{index}."): v
                for k, v in original.items()
                if f".layers.{source}." in k
            }
        )
    return build(metadata(tiny), registry())


def test_native_periodic_body_is_verified_without_replacing_concrete_layers() -> None:
    builder = periodic_qwen(["linear_attention"] * 3 + ["full_attention"])
    # One body is not a depth loop.
    single = builder.finish()
    assert not single.repetitions[0].document().get("bodies")
    builder = periodic_qwen((["linear_attention"] * 3 + ["full_attention"]) * 6)
    graph = builder.finish()
    rep = graph.repetitions[0]
    bodies = rep.document().get("bodies", [])
    assert len(bodies) == 1, "24-layer Qwen must publish one verified repeated body"
    body = bodies[0]
    assert (body["start"], body["width"], body["count"]) == (0, 4, 6)
    assert body["ranges"] == [{"start": 0, "count": 3}, {"start": 3, "count": 1}]
    families = {t.id: t for t in graph.templates or []}
    assert len(set(body["slots"])) == 2
    for position, instance in enumerate(rep.instances):
        assert instance.index == position
        assert instance.node_id in {
            i.node_id for i in families[body["slots"][position % 4]].instances
        }
    with patch.object(ComponentTemplates, "annotate", side_effect=lambda graph, _: graph):
        ordinary = builder.finish()
    assert graph.nodes == ordinary.nodes and graph.edges == ordinary.edges
    assert graph.parameters == ordinary.parameters


@pytest.mark.parametrize("case", ["tail", "homogeneous", "topology", "overlap", "slot"])
def test_repeated_body_boundaries(case: str) -> None:
    from llm_model_explorer.architecture_analysis.validation import GraphError, validate_graph

    pattern = (["linear_attention"] * 3 + ["full_attention"]) * 2
    if case == "tail":
        pattern += ["linear_attention", "linear_attention"]
    if case == "homogeneous":
        pattern = ["linear_attention"] * 8
    builder = periodic_qwen(pattern)
    if case == "topology":
        builder._nodes = [
            n.model_copy(update={"formula": "out = distinct_transform(x)"})
            if semantic_key(n) == "model.language_model.layers.5.mlp.silu"
            else n
            for n in builder._nodes
        ]
    graph = builder.finish()
    rep = graph.repetitions[0]
    if case in {"homogeneous", "topology"}:
        assert not rep.bodies
    elif case == "tail":
        assert rep.bodies is not None
        assert [(b.start, b.width, b.count) for b in rep.bodies] == [(0, 4, 2)]
        assert [i.index for i in rep.instances[-2:]] == [8, 9]
    else:
        assert rep.bodies is not None
        body = rep.bodies[0]
        malformed = (
            [body, body]
            if case == "overlap"
            else [body.model_copy(update={"slots": [body.slots[3], *body.slots[1:]]})]
        )
        invalid = graph.model_copy(
            update={"repetitions": [rep.model_copy(update={"bodies": malformed})]}
        )
        with pytest.raises(GraphError, match="Repeated body"):
            validate_graph(invalid, builder.inputs.bindings)

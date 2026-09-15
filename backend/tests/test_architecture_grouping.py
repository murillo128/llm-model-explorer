"""Reviewed component ownership and interfaces, independent of producer grouping helpers."""

from dataclasses import replace

import pytest
from architecture_assertions import semantic_key
from architecture_grouping_cases import cases

from llm_model_explorer.architecture_analysis import (
    AnalysisInput,
    DescriptionRegistry,
    GraphBuilder,
)
from llm_model_explorer.architecture_analysis import records as r
from llm_model_explorer.architecture_analysis.validation import validate_graph


def role(node: r.ArchitectureNode) -> str | None:
    return next(
        (
            a.value
            for a in node.attributes
            if a.name == "semantic_role" and isinstance(a.value, str)
        ),
        None,
    )


@pytest.mark.parametrize("name,inputs,registry", cases(), ids=[c[0] for c in cases()])
def test_reviewed_groups_and_revision(
    name: str, inputs: AnalysisInput, registry: DescriptionRegistry
) -> None:
    selected = registry.select(inputs)
    assert selected is not None and selected.producer.revision == "2"
    result = registry.analyze(inputs)
    graph = result.graph
    assert graph is not None
    assert registry.analyze(inputs).graph == graph
    old = replace(selected.producer, revision="1")
    assert old.graph_id(inputs.fingerprint, selected.scope) != graph.graph_id
    assert GraphBuilder(inputs, old, selected.scope).record_id("node", "model") != GraphBuilder(
        inputs, selected.producer, selected.scope
    ).record_id("node", "model")
    nodes = {n.id: n for n in graph.nodes}
    keys = {semantic_key(n): n for n in graph.nodes}
    layers = [nodes[i.node_id] for rep in graph.repetitions for i in rep.instances]
    components = [n for n in graph.nodes if n.kind == "group" and role(n) in {"attention", "mlp"}]
    assert len(components) == 2 * len(layers)
    for layer in layers:
        assert isinstance(layer, r.ArchitectureGroupNode)
        child_groups = [nodes[c] for c in layer.children if nodes[c].kind == "group"]
        assert [role(n) for n in child_groups] == ["attention", "mlp"]
        attention, mlp = child_groups
        assert isinstance(attention, r.ArchitectureGroupNode)
        assert isinstance(mlp, r.ArchitectureGroupNode)
        assert [p.id for p in mlp.ports] == ["x", "out"]
        assert mlp.label == "MLP" and attention.label == "Attention"
        base = semantic_key(layer)
        assert semantic_key(mlp) == base + ".mlp"
        for suffix in ("attention_residual", "mlp_residual"):
            assert keys[base + "." + suffix].parent_id == layer.id
        norm_suffixes = (
            ("norm1", "norm2")
            if name.startswith("visual")
            else ("input_layernorm", "post_attention_layernorm")
        )
        assert all(keys[base + "." + suffix].parent_id == layer.id for suffix in norm_suffixes)
        # Full ownership comes from the reviewed descriptions, not from a prefix scan.
        if name.startswith("visual"):
            expected_mlp = {base + suffix for suffix in (".mlp.fc1", ".gelu", ".mlp.fc2")}
            expected_attention = {
                base + suffix
                for suffix in (
                    ".attention.query",
                    ".attention.key",
                    ".attention.value",
                    ".query.heads",
                    ".key.heads",
                    ".value.heads",
                    ".query.rope",
                    ".key.rope",
                    ".scores",
                    ".softmax",
                    ".weighted_values",
                    ".merge_heads",
                    ".attention.proj",
                )
            }
            assert {semantic_key(nodes[c]) for c in attention.children} == expected_attention
            assert [p.id for p in attention.ports] == ["x", "positions", "out"]
        else:
            activation = ".mlp.silu" if name == "hybrid" else ".mlp.activation"
            expected_mlp = {
                base + suffix
                for suffix in (
                    ".mlp.gate_proj",
                    ".mlp.up_proj",
                    activation,
                    ".mlp.multiply",
                    ".mlp.down_proj",
                )
            }
            if name != "hybrid":
                expected_attention = {
                    base + ".self_attn." + suffix
                    for suffix in (
                        "q_proj",
                        "k_proj",
                        "v_proj",
                        "q_heads",
                        "k_heads",
                        "v_heads",
                        "q_transpose",
                        "k_transpose",
                        "v_transpose",
                        "q_rotary",
                        "k_rotary",
                        "k_repeat",
                        "v_repeat",
                        "key_transpose",
                        "qk_product",
                        "scale",
                        "mask",
                        "softmax",
                        "value_product",
                        "output_transpose",
                        "merge_heads",
                        "o_proj",
                    )
                }
                if name == "qwen3":
                    expected_attention |= {base + ".self_attn.q_norm", base + ".self_attn.k_norm"}
                assert {semantic_key(nodes[c]) for c in attention.children} == expected_attention
                assert [p.id for p in attention.ports] == ["x", "cos", "sin", "mask", "out"]
        assert {semantic_key(nodes[c]) for c in mlp.children} == expected_mlp
        for component in (attention, mlp):
            assert component.references == [
                r.ArchitectureModuleReference(kind="module", name=semantic_key(component))
            ]
            assert component.parameter_ids == []  # instance weights stay on their operations
    # The navigation annotation is optional; absence does not change computational coverage.
    legacy = graph.model_copy(deep=True)
    for node in legacy.nodes:
        node.attributes[:] = [a for a in node.attributes if a.name != "semantic_role"]
    validate_graph(legacy, inputs.bindings)
    assert legacy.coverage == graph.coverage


def test_hybrid_state_and_unused_layer_interfaces_keep_exact_owners() -> None:
    _, inputs, registry = next(c for c in cases() if c[0] == "hybrid")
    graph = registry.analyze(inputs).graph
    assert graph is not None
    keys = {semantic_key(n): n for n in graph.nodes}
    prefix = "model.language_model.layers."
    for index, component, states in (
        (0, "linear_attn", ("prior_conv", "next_conv", "prior_recurrent", "next_recurrent")),
        (1, "self_attn", ("prior_kv", "next_kv")),
    ):
        layer = keys[prefix + str(index)]
        attention = keys[f"{prefix}{index}.{component}"]
        assert [p.id for p in layer.ports] == ["x", "positions", "mask", "current_mask", "out"]
        for state in states:
            assert keys[f"{prefix}{index}.{component}.{state}"].parent_id == attention.id
    prior = keys[prefix + "1.self_attn.prior_kv"]
    assert [p.id for p in prior.ports] == ["key_state", "value_state"]
    conv = keys[prefix + "0.linear_attn.prior_conv"]
    delta = keys[prefix + "0.linear_attn.prior_recurrent"]
    assert conv.ports[0].shape != delta.ports[0].shape

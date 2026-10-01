"""Independent compact-card vocabulary contracts on materialized static graphs."""

import re

import pytest
from architecture_assertions import semantic_key
from architecture_grouping_cases import cases
from dense_fixtures import small_config, small_storage
from test_dense_architecture import graph_for, metadata

from llm_model_explorer.architecture_analysis import AnalysisInput, DescriptionRegistry


@pytest.mark.parametrize("name,inputs,registry", cases(), ids=[case[0] for case in cases()])
def test_reviewed_common_primitives_have_local_signatures(
    name: str, inputs: AnalysisInput, registry: DescriptionRegistry
) -> None:
    result = registry.analyze(inputs)
    assert result.graph is not None
    primitives = {"linear", "add", "multiply", "scale", "matmul", "softmax", "reshape", "transpose"}
    for node in result.graph.nodes:
        if node.kind == "operation" and node.operation in primitives:
            assert node.formula, (name, semantic_key(node))
            outputs = [port.label for port in node.ports if port.direction == "output"]
            assert len(outputs) == 1
            assert node.formula.startswith(outputs[0] + " = "), (name, semantic_key(node))


@pytest.mark.parametrize("bias", [False, True])
def test_dense_card_formulas_close_over_visible_ports_parameters_and_scalars(bias: bool) -> None:
    config = small_config(False) | {"attention_bias": bias, "mlp_bias": bias}
    graph = graph_for(metadata(config, small_storage(False, biases=bias)))
    nodes = {semantic_key(node): node for node in graph.nodes}
    expected = {
        "self_attn.q_proj": "out = x @ weightᵀ" + (" + bias" if bias else ""),
        "self_attn.q_heads": "out = reshape(x, ...)",
        "self_attn.q_transpose": "out = transpose(x, ...)",
        "self_attn.qk_product": "out = q @ kt",
        "self_attn.scale": "out = factor * x",
        "self_attn.softmax": "out = softmax(x, axis=axis)",
        "self_attn.value_product": "out = probabilities @ v",
        "attention_residual": "out = skip + branch",
        "mlp.multiply": "out = gate * up",
    }
    parameters = {parameter.id: parameter for parameter in graph.parameters}
    # Card exposure is bounded and operation-specific, rather than all attributes.
    displayed = {"scale": {"factor"}, "softmax": {"axis"}}
    for suffix, formula in expected.items():
        node = nodes["model.layers.0." + suffix]
        assert node.formula == formula
        vocabulary = {port.label for port in node.ports}
        vocabulary.update(parameters[pid].name.rsplit(".", 1)[-1] for pid in node.parameter_ids)
        vocabulary.update(
            attr.name
            for attr in node.attributes
            if attr.name in displayed.get(node.operation or "", set())
        )
        vocabulary.update({"reshape", "transpose", "softmax"})
        assert set(re.findall(r"[A-Za-z_]+", formula)) <= vocabulary
    assert nodes["lm_head"].formula == "out = x @ weightᵀ"

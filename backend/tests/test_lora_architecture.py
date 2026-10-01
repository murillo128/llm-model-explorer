"""Static LoRA graph coverage on a deterministic SmolLM2-shaped local composition."""

import importlib.util
import json
import math
from dataclasses import replace
from pathlib import Path

import pytest
from architecture_assertions import semantic_key
from dense_fixtures import small_config, small_storage
from test_models import write_weights
from test_quantized_models import write_storage

from llm_model_explorer.architecture_analysis import (
    AnalysisInput,
    DescriptionRegistry,
    register_dense_descriptions,
)
from llm_model_explorer.architecture_analysis import records as r
from llm_model_explorer.architecture_analysis.validation import GraphError, validate_graph
from llm_model_explorer.models import ModelCatalogue
from llm_model_explorer.tensor_source import ModelSource, PeftLoraComposition, PeftLoraTarget

BASE_ID = "HuggingFaceTB/SmolLM2-135M"
ADAPTER_ID = "hfm8tr/smollm2-135m-smoltalk-lora"
API_ORACLE_PATH = Path(__file__).resolve().parents[2] / "api/architecture_conformance.py"
API_ORACLE_SPEC = importlib.util.spec_from_file_location("lora_api_oracle", API_ORACLE_PATH)
assert API_ORACLE_SPEC and API_ORACLE_SPEC.loader
API_ORACLE = importlib.util.module_from_spec(API_ORACLE_SPEC)
API_ORACLE_SPEC.loader.exec_module(API_ORACLE)
validate_api_architecture = API_ORACLE.validate_architecture


def make_smollm2_lora(root: Path, *, biases: bool = False) -> tuple[ModelSource, ModelSource]:
    base = root / "smollm2"
    base.mkdir()
    config = small_config(False)
    config.update(attention_bias=biases, mlp_bias=biases)
    config["_name_or_path"] = BASE_ID
    (base / "config.json").write_text(json.dumps(config))
    write_weights(
        base / "model.safetensors",
        [
            (
                name,
                dtype,
                dims,
                [float(index % 7) / 8 for index in range(math.prod(dims))],
            )
            for name, dtype, dims in small_storage(False, biases=biases)
        ],
    )

    adapter = root / "smoltalk-lora"
    adapter.mkdir()
    adapter_config = {
        "_name_or_path": ADAPTER_ID,
        "base_model_name_or_path": BASE_ID,
        "peft_type": "LORA",
        "task_type": "CAUSAL_LM",
        "r": 8,
        "lora_alpha": 16,
        "target_modules": ["q_proj", "v_proj"],
        "bias": "none",
        "fan_in_fan_out": False,
        "use_dora": False,
        "use_rslora": False,
        "rank_pattern": {},
        "alpha_pattern": {},
    }
    (adapter / "adapter_config.json").write_text(json.dumps(adapter_config))
    factors = []
    for layer in range(2):
        for projection, out in (("q_proj", 12), ("v_proj", 4)):
            module = f"model.layers.{layer}.self_attn.{projection}"
            factors.extend(
                [
                    (
                        f"base_model.model.{module}.lora_A.default.weight",
                        "F16",
                        [8, 12],
                        [float(index + 1) / 16 for index in range(96)],
                    ),
                    (
                        f"base_model.model.{module}.lora_B.default.weight",
                        "BF16",
                        [out, 8],
                        [float(index + 1) / 8 for index in range(out * 8)],
                    ),
                ]
            )
    write_weights(adapter / "adapter_model.safetensors", factors)

    entries = {entry.summary.id: entry for entry in ModelCatalogue(root).discover()}
    return entries[BASE_ID].pin(), entries[f"{BASE_ID}+peft-lora:{ADAPTER_ID}"].pin()


def make_qwen_native_and_gptq(root: Path) -> tuple[str, str]:
    model_id, adapter_id = "Qwen3Fixture/Tiny", "fixture/qwen-lora"
    quantized_config = small_config(True)
    quantized_config["_name_or_path"] = model_id
    native_config = dict(quantized_config)
    native_config.pop("quantization_config")
    native = root / "qwen-native"
    packed = root / "qwen-gptq"
    native.mkdir()
    packed.mkdir()
    (native / "config.json").write_text(json.dumps(native_config))
    (packed / "config.json").write_text(json.dumps(quantized_config))
    physical = small_storage(True)
    native_weights = []
    for name, dtype, dims in physical:
        if name.endswith(".qweight"):
            prefix = name.removesuffix(".qweight")
            packed_input, output = dims
            shape = [output, packed_input * 8]
            native_weights.append(
                (
                    prefix + ".weight",
                    "F32",
                    shape,
                    [float(i % 11) / 16 for i in range(math.prod(shape))],
                )
            )
        elif not name.endswith((".qzeros", ".scales", ".g_idx")):
            native_weights.append(
                (name, dtype, dims, [float(i % 13) / 16 for i in range(math.prod(dims))])
            )
    write_weights(native / "model.safetensors", native_weights)
    write_storage(packed / "model.safetensors", physical)

    adapter = root / "qwen-lora"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text(
        json.dumps(
            {
                "_name_or_path": adapter_id,
                "base_model_name_or_path": model_id,
                "peft_type": "LORA",
                "task_type": "CAUSAL_LM",
                "r": 2,
                "lora_alpha": 4,
                "target_modules": ["q_proj", "v_proj"],
                "bias": "none",
                "fan_in_fan_out": False,
                "use_dora": False,
                "use_rslora": False,
            }
        )
    )
    factors = []
    for layer in range(2):
        for projection, output in (("q_proj", 256), ("v_proj", 128)):
            module = f"model.layers.{layer}.self_attn.{projection}"
            factors.extend(
                [
                    (
                        f"base_model.model.{module}.lora_A.default.weight",
                        "F16",
                        [2, 128],
                        [float(i + 1) / 16 for i in range(256)],
                    ),
                    (
                        f"base_model.model.{module}.lora_B.default.weight",
                        "BF16",
                        [output, 2],
                        [float(i + 1) / 8 for i in range(output * 2)],
                    ),
                ]
            )
    write_weights(adapter / "adapter_model.safetensors", factors)
    return model_id, adapter_id


def analyze(source: ModelSource) -> tuple[AnalysisInput, r.ArchitectureGraph]:
    inputs = AnalysisInput.from_source(source, tokenizer_available=False)
    registry = DescriptionRegistry()
    register_dense_descriptions(registry)
    result = registry.analyze(inputs)
    assert result.graph is not None, result
    return inputs, result.graph


def semantic_nodes(graph: r.ArchitectureGraph) -> dict[str, r.ArchitectureNode]:
    return {semantic_key(node): node for node in graph.nodes}


def semantic_edges(graph: r.ArchitectureGraph) -> set[tuple[str, str, str, str]]:
    nodes = {node.id: node for node in graph.nodes}
    return {
        (
            semantic_key(nodes[edge.source.node_id]),
            edge.source.port_id,
            semantic_key(nodes[edge.target.node_id]),
            edge.target.port_id,
        )
        for edge in graph.edges
    }


@pytest.mark.parametrize("biases", [False, True])
def test_smollm2_lora_branches_bind_real_factors_and_preserve_base(
    tmp_path: Path, biases: bool
) -> None:
    bare_source, composite_source = make_smollm2_lora(tmp_path, biases=biases)
    bare_inputs, bare = analyze(bare_source)
    composite_inputs, graph = analyze(composite_source)
    assert (
        analyze(composite_source)[1] == graph
    )  # Group/interior IDs and template roles are deterministic.
    nodes = semantic_nodes(graph)
    edges = semantic_edges(graph)
    targets = [
        f"model.layers.{layer}.self_attn.{projection}"
        for layer in range(2)
        for projection in ("q_proj", "v_proj")
    ]
    parameters = {parameter.name: parameter for parameter in graph.parameters}
    numeric_by_name = {tensor.name: tensor for tensor in composite_inputs.bindings.numeric.values()}

    for target in targets:
        base = target + ".base"
        a, b, scale, add = (target + ".lora_" + suffix for suffix in ("A", "B", "scale", "add"))
        group = nodes[target]
        assert isinstance(group, r.ArchitectureGroupNode)
        assert group.children == [nodes[key].id for key in (base, a, b, scale, add)]
        assert group.parameter_ids == []
        assert [(p.id, p.label, p.direction) for p in group.ports] == [
            ("x", "x", "input"),
            ("out", "out", "output"),
        ]
        assert all(nodes[key].parent_id == group.id for key in (base, a, b, scale, add))
        assert nodes[base].operation == "linear"
        assert nodes[base].parameter_ids == [parameters[target + ".weight"].id] + (
            [parameters[target + ".bias"].id] if biases else []
        )
        assert nodes[base].references[0] == r.ArchitectureModuleReference(
            kind="module", name=target
        )
        assert nodes[a].operation == nodes[b].operation == "linear"
        assert nodes[scale].operation == "scale"
        assert nodes[add].operation == "add"
        assert (target.rsplit(".", 1)[0], "x", target, "x") in edges
        assert (target, "x", base, "x") in edges
        assert (target, "x", a, "x") in edges
        assert (a, "out", b, "x") in edges
        assert (b, "out", scale, "x") in edges
        assert (base, "out", add, "base") in edges
        assert (scale, "out", add, "adapter") in edges
        assert (add, "out", target, "out") in edges
        assert (target, "out", target.replace("_proj", "_heads"), "x") in edges
        interior = {target, base, a, b, scale, add}
        assert {edge for edge in edges if edge[0] in interior and edge[2] in interior} == {
            (target, "x", base, "x"),
            (target, "x", a, "x"),
            (a, "out", b, "x"),
            (b, "out", scale, "x"),
            (base, "out", add, "base"),
            (scale, "out", add, "adapter"),
            (add, "out", target, "out"),
        }
        assert {attribute.name: attribute.value for attribute in nodes[scale].attributes}[
            "factor"
        ] == 2.0

        for factor, expected_shape in (
            ("A", (8, 12)),
            ("B", (12 if target.endswith("q_proj") else 4, 8)),
        ):
            parameter = parameters[
                f"__peft__.hfm8tr%2Fsmollm2-135m-smoltalk-lora.{target}.lora_{factor}.weight"
            ]
            assert parameter.binding == "native"
            assert parameter.inspection.status == "available"
            numeric = composite_inputs.bindings.numeric[parameter.inspection.tensor_id]
            assert numeric.name == parameter.name
            assert numeric.shape == expected_shape
            assert parameter.name in numeric_by_name
            assert parameter.storage[0].role == "adapter_factor"
            assert (
                parameter.storage[0].name
                == f"base_model.model.{target}.lora_{factor}.default.weight"
            )
            assert nodes[a if factor == "A" else b].parameter_ids == [parameter.id]

    # An untargeted projection and its complete parameter binding are unchanged.
    bare_nodes = semantic_nodes(bare)
    for target in targets:
        assert bare_nodes[target].kind == "operation"
        assert bare_nodes[target].operation == "linear"
        assert bare_nodes[target].parent_id == bare_nodes[target.rsplit(".", 1)[0]].id
        assert not any(key.startswith(target + ".") for key in bare_nodes)
    for key in (
        "model.layers.0.self_attn.k_proj",
        "model.layers.0.self_attn.o_proj",
        "model.layers.0.mlp.gate_proj",
    ):
        assert nodes[key].operation == bare_nodes[key].operation
        assert nodes[key].ports == bare_nodes[key].ports
        assert nodes[key].formula == bare_nodes[key].formula
    bare_parameters = {parameter.name: parameter for parameter in bare.parameters}
    for name, parameter in bare_parameters.items():
        composed = parameters[name]
        assert composed.logical_shape == parameter.logical_shape
        assert composed.binding == parameter.binding
        assert composed.storage == parameter.storage
        assert composed.inspection == parameter.inspection

    attention_templates = [t for t in graph.templates or () if t.component_role == "attention"]
    assert attention_templates
    assert any(len(template.instances) == 2 for template in attention_templates)
    for template in attention_templates:
        for instance in template.instances:
            mapped = {mapping.node_id for mapping in instance.nodes}
            attention = next(node for node in graph.nodes if node.id == instance.node_id)
            layer_targets = [
                target for target in targets if nodes[target].parent_id == attention.id
            ]
            assert all(
                nodes[target + suffix].id in mapped
                for target in layer_targets
                for suffix in ("", ".base", ".lora_A", ".lora_B", ".lora_scale", ".lora_add")
            )
    assert not any(t.name.startswith("__peft__") for t in bare_inputs.bindings.numeric.values())

    inventory = {
        "tensors": [tensor.model_dump(mode="json") for tensor in composite_source.tensors()],
        "coverage": "complete",
        "diagnostics": [],
    }
    validate_api_architecture(
        {
            "status": "available",
            "model_id": composite_source.model_id,
            "diagnostics": [],
            "graph": graph.document(),
        },
        {
            "session": {"id": "fixture-session", "model_id": composite_source.model_id},
            "inventory": inventory,
            "tokenizer_available": False,
        },
    )


def test_lora_scale_formula_uses_visible_ports_and_factor(tmp_path: Path) -> None:
    _, source = make_smollm2_lora(tmp_path)
    _, graph = analyze(source)
    scale = semantic_nodes(graph)["model.layers.0.self_attn.q_proj.lora_scale"]
    assert scale.formula == "out = factor * x"
    assert {p.label for p in scale.ports} == {"x", "out"}
    assert {a.name: a.value for a in scale.attributes}["factor"] == 2


def test_quantized_and_native_base_use_the_same_lora_graph(tmp_path: Path) -> None:
    model_id, adapter_id = make_qwen_native_and_gptq(tmp_path)
    entries = {entry.summary.id: entry for entry in ModelCatalogue(tmp_path).discover()}
    native_source = entries[f"{model_id}+peft-lora:{adapter_id}"].pin()
    quantized_source = entries[f"{model_id}@gptq-int4+peft-lora:{adapter_id}"].pin()
    _, native = analyze(native_source)
    _, quantized = analyze(quantized_source)

    def node_signature(graph: r.ArchitectureGraph) -> dict[str, tuple[object, ...]]:
        nodes = {node.id: node for node in graph.nodes}
        return {
            semantic_key(node): (
                node.kind,
                node.label,
                node.operation,
                semantic_key(nodes[node.parent_id]) if node.parent_id else None,
                tuple((port.id, port.direction, port.shape) for port in node.ports),
                tuple((attribute.name, attribute.value) for attribute in node.attributes),
                node.formula,
            )
            for node in graph.nodes
        }

    assert node_signature(native) == node_signature(quantized)
    assert semantic_edges(native) == semantic_edges(quantized)
    native_parameters = {parameter.name: parameter for parameter in native.parameters}
    quantized_parameters = {parameter.name: parameter for parameter in quantized.parameters}
    base_weight = "model.layers.0.self_attn.q_proj.weight"
    assert native_parameters[base_weight].binding == "native"
    assert quantized_parameters[base_weight].binding == "quantized"
    assert native_parameters[base_weight].inspection.status == "available"
    assert quantized_parameters[base_weight].inspection.status == "available"
    adapter_weight = next(name for name in native_parameters if ".q_proj.lora_A.weight" in name)
    assert native_parameters[adapter_weight].name == quantized_parameters[adapter_weight].name
    assert (
        native_parameters[adapter_weight].inspection
        == quantized_parameters[adapter_weight].inspection
    )


@pytest.mark.parametrize(
    ("mutation", "code"),
    [
        ("missing", "lora_missing_factor"),
        ("ambiguous", "lora_ambiguous_target"),
        ("shape", "lora_factor_geometry"),
        ("scale", "lora_scale"),
        ("unsupported", "lora_unsupported_target"),
    ],
)
def test_invalid_lora_graph_bindings_fail_before_publication(
    tmp_path: Path, mutation: str, code: str
) -> None:
    _, source = make_smollm2_lora(tmp_path)
    inputs = AnalysisInput.from_source(source, tokenizer_available=False)
    composition = inputs.lora_composition
    assert composition is not None
    first = composition.targets[0]
    if mutation == "missing":
        targets = (
            replace(first, a_tensor_name="__peft__.missing.factor"),
            *composition.targets[1:],
        )
        composition = replace(composition, targets=targets)
    elif mutation == "ambiguous":
        composition = replace(composition, targets=(*composition.targets, first))
    elif mutation == "shape":
        composition = replace(composition, rank=3, alpha=6, scale=2)
    elif mutation == "scale":
        composition = replace(composition, scale=3)
    else:
        composition = replace(
            composition,
            targets=(
                replace(first, module_name="model.layers.0.self_attn.unknown_proj"),
                *composition.targets[1:],
            ),
        )
    invalid = replace(inputs, lora_composition=composition)
    registry = DescriptionRegistry()
    register_dense_descriptions(registry)
    result = registry.analyze(invalid)
    assert result.graph is None
    assert result.diagnostics[0].code == code


def test_lora_metadata_types_are_explicit_and_path_free(tmp_path: Path) -> None:
    _, source = make_smollm2_lora(tmp_path)
    composition = source.lora_composition
    assert isinstance(composition, PeftLoraComposition)
    assert all(isinstance(target, PeftLoraTarget) for target in composition.targets)
    assert all("/" not in target.a_tensor_name for target in composition.targets)


def test_published_lora_factor_storage_geometry_is_rejected_by_semantic_authorities(
    tmp_path: Path,
) -> None:
    _, source = make_smollm2_lora(tmp_path)
    inputs, graph = analyze(source)
    damaged = graph.model_copy(deep=True)
    factor = next(
        parameter
        for parameter in damaged.parameters
        if parameter.storage and parameter.storage[0].role == "adapter_factor"
    )
    factor.storage[0].shape[0] = 1
    with pytest.raises(GraphError, match="Unverified physical storage descriptor"):
        validate_graph(damaged, inputs.bindings)
    with pytest.raises(ValueError):
        validate_api_architecture(
            {
                "status": "available",
                "model_id": source.model_id,
                "diagnostics": [],
                "graph": damaged.document(),
            },
            {
                "session": {"id": "fixture-session", "model_id": source.model_id},
                "inventory": {
                    "tensors": [tensor.model_dump(mode="json") for tensor in source.tensors()],
                    "coverage": "complete",
                    "diagnostics": [],
                },
                "tokenizer_available": False,
            },
        )

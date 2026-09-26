"""Independent topology, binding, and no-execution coverage for dense descriptions."""

import builtins
import copy
import importlib.util
import json
import math
import socket
from dataclasses import replace
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import pytest
import torch
from architecture_assertions import semantic_key, transparent_edges
from dense_fixtures import local_fixture, small_config, small_storage
from quantized_oracles import bnb_config, bnb_nf4_fixture
from test_quantized_models import Storage

from llm_model_explorer.architecture_analysis import (
    AnalysisInput,
    BindingContext,
    DescriptionRegistry,
    NumericTensor,
    parse_graph,
    register_dense_descriptions,
    serialize_graph,
)
from llm_model_explorer.architecture_analysis import records as r
from llm_model_explorer.architecture_analysis.dense_config import checked
from llm_model_explorer.models import ModelCatalogue
from llm_model_explorer.tensor_source import ModelSource

REFERENCES = json.loads(
    (Path(__file__).parent / "fixtures/dense-reference-metadata.json").read_text()
)


def registry() -> DescriptionRegistry:
    result = DescriptionRegistry()
    register_dense_descriptions(result)
    return result


def metadata(
    config: dict[str, Any], entries: list[Storage], *, tokenizer: bool = False
) -> AnalysisInput:
    physical = {
        name: r.ArchitectureStorage(name=name, dtype=dtype, shape=shape)
        for name, dtype, shape in entries
    }
    numeric = {
        f"native_{i}": NumericTensor(f"native_{i}", name, tuple(shape), dtype)
        for i, (name, dtype, shape) in enumerate(entries)
        if dtype in {"F32", "BF16", "F16"} and not name.endswith(".scales")
    }
    return AnalysisInput("synthetic-content", config, BindingContext(physical, numeric, tokenizer))


def graph_for(data: AnalysisInput, status: str = "complete") -> r.ArchitectureGraph:
    result = registry().analyze(data)
    assert result.status == status, result
    assert result.graph is not None
    return result.graph


def endpoints(graph: r.ArchitectureGraph) -> set[tuple[str, str, str, str]]:
    names = {node.id: semantic_key(node) for node in graph.nodes}
    return {(names[sn], sp, names[tn], tp) for sn, sp, tn, tp in transparent_edges(graph)}


def dimensions(shape: r.ArchitectureShape) -> list[int | str]:
    assert shape is not None
    return [
        d.value if isinstance(d, r.ArchitectureConstantDimension) else d.name
        for d in shape
        if isinstance(d, (r.ArchitectureConstantDimension, r.ArchitectureSymbolDimension))
    ]


@pytest.mark.parametrize(
    "family,layers,nodes,edges,parameters",
    [
        ("qwen3", 28, 962, 1350, 311),
        ("smollm2", 30, 970, 1386, 273),
    ],
)
def test_reference_metadata_full_instance_graph(
    family: str, layers: int, nodes: int, edges: int, parameters: int
) -> None:
    reference = REFERENCES[family]
    storage = [(k, dt, dims) for k, (dt, dims) in reference["global_storage"].items()]
    for i in range(layers):
        storage.extend(
            (f"model.layers.{i}.{k}", dt, dims)
            for k, (dt, dims) in reference["layer_storage"].items()
        )
    assert len(storage) == reference["physical_count"]
    data = metadata(reference["config"], storage)
    graph = graph_for(data)
    assert (len(graph.nodes), len(graph.edges), len(graph.parameters)) == (
        nodes + 2 * layers,
        edges + 7 * layers,
        parameters,
    )
    assert len(serialize_graph(graph)) < 4_000_000
    repetition = graph.repetitions[0]
    assert [i.index for i in repetition.instances] == list(range(layers))
    by_id = {n.id: n for n in graph.nodes}
    assert [by_id[i.node_id].label for i in repetition.instances] == [
        f"model.layers.{i}" for i in range(layers)
    ]
    for instance in repetition.instances:
        layer_node = by_id[instance.node_id]
        assert isinstance(layer_node, r.ArchitectureGroupNode)
        assert len(layer_node.children) == 6  # two norms, two residuals, Attention and MLP
    params = {p.name: p for p in graph.parameters}
    assert params["lm_head.weight"].binding == "alias"
    assert params["lm_head.weight"].inspection == params["model.embed_tokens.weight"].inspection
    assert {s.name for p in graph.parameters for s in p.storage} == set(data.bindings.physical)
    for i in range(layers):
        norm = next(
            n for n in graph.nodes if semantic_key(n) == f"model.layers.{i}.input_layernorm"
        )
        assert {a.name: a.value for a in norm.attributes}["epsilon"] == (
            1e-6 if family == "qwen3" else 1e-5
        )
    assert parse_graph(graph.document(), data.bindings) == graph


@pytest.mark.parametrize("qwen", [False, True])
def test_independent_attention_mlp_and_residual_dependencies(qwen: bool) -> None:
    graph = graph_for(metadata(small_config(qwen), small_storage(qwen)))
    edges = endpoints(graph)
    for i in (0, 1):
        layer = f"model.layers.{i}"
        expected = [
            (layer, "x", layer + ".input_layernorm", "x"),
            (layer, "x", layer + ".attention_residual", "skip"),
            (layer + ".self_attn.o_proj", "out", layer + ".attention_residual", "branch"),
            (layer + ".attention_residual", "out", layer + ".post_attention_layernorm", "x"),
            (layer + ".attention_residual", "out", layer + ".mlp_residual", "skip"),
            (layer + ".post_attention_layernorm", "out", layer + ".mlp.gate_proj", "x"),
            (layer + ".post_attention_layernorm", "out", layer + ".mlp.up_proj", "x"),
            (layer + ".mlp.gate_proj", "out", layer + ".mlp.activation", "x"),
            (layer + ".mlp.activation", "out", layer + ".mlp.multiply", "gate"),
            (layer + ".mlp.up_proj", "out", layer + ".mlp.multiply", "up"),
            (layer + ".mlp.multiply", "out", layer + ".mlp.down_proj", "x"),
            (layer + ".mlp.down_proj", "out", layer + ".mlp_residual", "branch"),
            (layer + ".mlp_residual", "out", layer, "out"),
            (layer + ".self_attn.qk_product", "out", layer + ".self_attn.scale", "x"),
            (layer + ".self_attn.scale", "out", layer + ".self_attn.mask", "x"),
            (layer, "mask", layer + ".self_attn.mask", "mask"),
            (layer + ".self_attn.mask", "out", layer + ".self_attn.softmax", "x"),
            (
                layer + ".self_attn.softmax",
                "out",
                layer + ".self_attn.value_product",
                "probabilities",
            ),
        ]
        assert set(expected) <= edges
        for branch in ("q", "k"):
            base = layer + ".self_attn." + branch
            if qwen:
                assert (base + "_heads", "out", base + "_norm", "x") in edges
                assert (base + "_norm", "out", base + "_transpose", "x") in edges
            else:
                assert not any(semantic_key(n) == base + "_norm" for n in graph.nodes)
                assert (base + "_heads", "out", base + "_transpose", "x") in edges
            assert (base + "_transpose", "out", base + "_rotary", "x") in edges
    assert ("model.layers.0", "out", "model.layers.1", "x") in edges
    assert ("model.layers.1", "out", "model.norm", "x") in edges
    assert ("model.norm", "out", "lm_head", "x") in edges
    assert ("lm_head", "out", "logits", "x") in edges
    nodes = {semantic_key(n): n for n in graph.nodes}
    assert dimensions(nodes["model.layers.0.self_attn.q_proj"].ports[-1].shape) == [
        "B",
        "S",
        256 if qwen else 12,
    ]
    assert dimensions(nodes["model.layers.0.self_attn.k_proj"].ports[-1].shape) == [
        "B",
        "S",
        128 if qwen else 4,
    ]
    assert dimensions(nodes["model.layers.0.self_attn.qk_product"].ports[-1].shape) == [
        "B",
        4 if qwen else 3,
        "S",
        "S",
    ]
    assert {a.name: a.value for a in nodes["model.layers.0.self_attn.k_repeat"].attributes}[
        "groups"
    ] == (2 if qwen else 3)


@pytest.mark.parametrize("qwen", [False, True])
def test_checked_native_and_quantized_bindings(qwen: bool) -> None:
    data = metadata(small_config(qwen), small_storage(qwen))
    graph = graph_for(data)
    by_id = {p.id: p for p in graph.parameters}
    for node in graph.nodes:
        for pid in node.parameter_ids:
            assert by_id[pid].name.startswith(semantic_key(node) + ".")
    quantized = [p for p in graph.parameters if p.binding == "quantized"]
    assert len(quantized) == (14 if qwen else 0)
    for p in quantized:
        assert p.inspection.status == "unavailable"
        assert p.inspection.reason == "unsupported_representation"
        assert [s.role for s in p.storage] == [
            "packed_data",
            "zero_points",
            "scales",
            "group_indices",
        ]
        out, inp = dimensions(p.logical_shape)
        assert isinstance(inp, int) and isinstance(out, int)
        assert [s.shape for s in p.storage] == [
            [inp // 8, out],
            [inp // 128, out // 8],
            [inp // 128, out],
            [inp],
        ]
    for parameter in graph.parameters:
        if parameter.inspection.status == "available":
            tensor = data.bindings.numeric[parameter.inspection.tensor_id]
            assert list(tensor.shape) == dimensions(parameter.logical_shape)
            assert not tensor.name.endswith((".qweight", ".scales", ".g_idx", ".qzeros"))


def test_gptq_inspection_uses_existing_complete_logical_identity() -> None:
    data = metadata(small_config(True), small_storage(True))
    name = "model.layers.1.self_attn.q_proj.weight"
    numeric = NumericTensor("logical_gptq", name, (256, 128), "I32", "gptq-int4")
    data = replace(
        data,
        bindings=replace(data.bindings, numeric={**data.bindings.numeric, numeric.id: numeric}),
    )
    result = graph_for(data)
    parameters = {parameter.name: parameter for parameter in result.parameters}
    parameter = parameters[name]
    assert parameter.binding == "quantized"
    assert parameter.inspection.status == "available"
    assert parameter.inspection.tensor_id == numeric.id
    assert dimensions(parameter.logical_shape) == [256, 128]
    assert [entry.name for entry in parameter.storage] == [
        name.removesuffix(".weight") + "." + suffix
        for suffix in ("qweight", "qzeros", "scales", "g_idx")
    ]
    assert parameters["model.layers.0.self_attn.q_proj.weight"].inspection.status == "unavailable"


def test_bnb_nf4_architecture_binds_complete_group_without_decoding() -> None:
    prefix = "model.layers.0.self_attn.q_proj"
    fixture = bnb_nf4_fixture(outputs=12, inputs=12, prefix=prefix)
    entries = [entry for entry in small_storage(False) if entry[0] != prefix + ".weight"]
    entries.extend((item.name, item.dtype, list(item.shape)) for item in fixture.storage)
    config = small_config(False) | {"quantization_config": bnb_config()["quantization_config"]}
    data = metadata(config, entries)
    tensor = NumericTensor("logical_bnb", fixture.name, fixture.shape, "U8", "bnb-nf4-dq")
    bnb_storage_names = {item.name for item in fixture.storage}
    numeric = {
        key: item
        for key, item in data.bindings.numeric.items()
        if item.name not in bnb_storage_names
    }
    data = replace(
        data,
        bindings=replace(data.bindings, numeric={**numeric, tensor.id: tensor}),
    )

    graph = graph_for(data)
    parameter = next(item for item in graph.parameters if item.name == fixture.name)
    assert parameter.binding == "quantized"
    assert parameter.inspection.status == "available"
    assert parameter.inspection.tensor_id == tensor.id
    assert dimensions(parameter.logical_shape) == [12, 12]
    assert [storage.role for storage in parameter.storage] == [
        "packed_data",
        "scales",
        "codebook",
        "scales",
        "codebook",
        "quantization_state",
    ]


@pytest.mark.parametrize(
    "changes",
    [
        {"hidden_act": "gelu"},
        {"rope_scaling": {"rope_type": "linear", "factor": 2}},
        {"rope_interleaved": True},
        {"use_sliding_window": True},
        {"sliding_window": 32},
        {"pretraining_tp": 2},
        {"num_key_value_heads": 3},
        {"head_dim": 3},
        {"head_dim": None},
        {"num_hidden_layers": True},
        {"hidden_size": -1},
        {"layer_types": ["full_attention", "linear_attention"]},
        {"layer_types": ["full_attention"]},
        {"mlp_bias": True},
        {"attention_bias": 1},
        {"rms_norm_eps": float("nan")},
        {"rope_theta": 0},
        {"new_structural_option": True},
        {"auto_map": {"AutoModel": "evil.Code"}},
        {"architectures": ["Qwen3MoeForCausalLM"]},
    ],
)
def test_unknown_or_contradictory_variants_are_explicit(changes: dict[str, Any]) -> None:
    config = small_config(True) | changes
    result = registry().analyze(metadata(config, small_storage(True)))
    assert result.reason == "unsupported_architecture"
    assert result.graph is None


def test_reviewed_defaults_and_ignored_training_metadata() -> None:
    for qwen in (False, True):
        config = small_config(qwen)
        for field in ("head_dim", "num_key_value_heads", "rms_norm_eps", "tie_word_embeddings"):
            del config[field]
        if qwen:
            config["num_attention_heads"] = 32
        c = checked(config)
        assert c is not None
        assert (c.head_dim, c.kv_heads, c.epsilon, c.tied) == (
            (128, 32, 1e-6, False) if qwen else (4, 3, 1e-6, False)
        )
        config.update(
            num_key_value_heads=None, initializer_range=0.1, transformers_version="4.55.4"
        )
        c = checked(config)
        assert c is not None and c.kv_heads == c.heads
    config = small_config(False)
    del config["head_dim"]
    graph_for(metadata(config, small_storage(False)))


@pytest.mark.parametrize(
    "defect", ["absent", "swapped", "extra", "missing_scale", "wrong_packing", "wrong_dtype"]
)
def test_inventory_uncertainty_never_claims_complete(defect: str) -> None:
    storage = small_storage(True)
    index = next(
        i for i, t in enumerate(storage) if t[0] == "model.layers.1.self_attn.q_proj.qweight"
    )
    name, dtype, dims = storage[index]
    if defect == "absent":
        storage = [s for s in storage if s[0] != "model.layers.1.input_layernorm.weight"]
    elif defect == "swapped":
        storage[0] = (storage[0][0], "F32", list(reversed(storage[0][2])))
    elif defect == "extra":
        storage.append(("model.layers.2.mystery.weight", "F32", [2, 3]))
    elif defect == "missing_scale":
        storage = [s for s in storage if s[0] != name.replace("qweight", "scales")]
    elif defect == "wrong_dtype":
        storage[index] = (name, "F32", dims)
    else:
        storage[index] = (name, dtype, list(reversed(dims)))
    graph = graph_for(metadata(small_config(True), storage), "partial")
    assert graph.diagnostics
    assert all(
        p.inspection.status == "unavailable" for p in graph.parameters if p.binding == "unresolved"
    )


@pytest.mark.parametrize("variant", ["untied", "head_only", "both", "missing_both", "biases"])
def test_native_tie_and_bias_variants(variant: str) -> None:
    config = small_config(False)
    storage = small_storage(False, biases=variant == "biases")
    if variant in {"untied", "head_only", "both"}:
        storage.append(("lm_head.weight", "F32", [16, 12]))
    if variant in {"head_only", "missing_both"}:
        storage = [s for s in storage if s[0] != "model.embed_tokens.weight"]
    if variant == "untied":
        config["tie_word_embeddings"] = False
    if variant == "biases":
        config.update(attention_bias=True, mlp_bias=True)
    graph = graph_for(
        metadata(config, storage), "partial" if variant in {"both", "missing_both"} else "complete"
    )
    params = {p.name: p for p in graph.parameters}
    if variant == "untied":
        assert params["lm_head.weight"].binding == "native"
        assert params["lm_head.weight"].inspection != params["model.embed_tokens.weight"].inspection
    if variant == "head_only":
        assert params["model.embed_tokens.weight"].binding == "alias"
        assert params["model.embed_tokens.weight"].inspection == params["lm_head.weight"].inspection
    if variant == "biases":
        assert len([p for p in graph.parameters if p.name.endswith(".bias")]) == 14


@pytest.mark.parametrize("qwen", [False, True])
def test_local_native_links_are_exact_asymmetric_values(tmp_path: Path, qwen: bool) -> None:
    local_fixture(tmp_path, qwen)
    source = ModelCatalogue(tmp_path).discover()[0].pin()
    data = AnalysisInput.from_source(source, tokenizer_available=False)
    graph = graph_for(data)
    for parameter in graph.parameters:
        if parameter.binding != "native" or parameter.inspection.status != "available":
            continue
        chunks = list(source.iter_tensor(parameter.inspection.tensor_id))
        actual = torch.cat(chunks).tolist()
        dims = dimensions(parameter.logical_shape)
        assert all(isinstance(d, int) for d in dims)
        count = math.prod(d for d in dims if isinstance(d, int))
        assert actual == [(i % 29 - 14) / 8 for i in range(count)]


@pytest.mark.parametrize("qwen", [False, True])
def test_description_no_execution_guards(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, qwen: bool
) -> None:
    directory = local_fixture(tmp_path, qwen)
    for path in directory.iterdir():
        path.chmod(0o444)
    directory.chmod(0o555)
    source = ModelCatalogue(tmp_path).discover()[0].pin()
    import safetensors
    import safetensors.torch
    import transformers

    denied = Mock(
        side_effect=AssertionError("Static analysis must not execute or materialize weights")
    )
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
        (transformers.GenerationMixin, ["generate"]),
        (ModelSource, ["iter_tensor", "iter_rows", "local_directory"]),
    ]:
        for name in names:
            monkeypatch.setattr(owner, name, denied)
    original = builtins.__import__

    def guarded(name: str, *args: Any, **kwargs: Any) -> Any:
        assert not name.startswith("modeling_custom")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    data = AnalysisInput.from_source(source, tokenizer_available=True)
    graph = graph_for(data)
    assert any(n.kind == "context" and semantic_key(n) == "Tokenizer" for n in graph.nodes)
    assert str(directory) not in str(graph.document())
    denied.assert_not_called()


def test_api_conformance_and_size_gate() -> None:
    path = Path(__file__).resolve().parents[2] / "api/architecture_conformance.py"
    spec = importlib.util.spec_from_file_location("dense_api_oracle", path)
    assert spec and spec.loader
    oracle = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(oracle)
    for qwen in (False, True):
        data = metadata(small_config(qwen), small_storage(qwen))
        graph = graph_for(data)
        oracle.validate_architecture(
            {
                "status": "available",
                "model_id": "fixture",
                "diagnostics": [],
                "graph": graph.document(),
            }
        )
        result = registry().analyze(data, byte_limit=2048)
        assert result.reason == "unsupported_size" and result.graph is None
        numeric_unavailable = replace(data, bindings=replace(data.bindings, numeric={}))
        result_graph = graph_for(numeric_unavailable)
        assert all(p.inspection.status == "unavailable" for p in result_graph.parameters)


def test_unreviewed_quantization_and_unknown_architecture() -> None:
    config = copy.deepcopy(small_config(True))
    config["quantization_config"]["desc_act"] = True
    assert (
        registry().analyze(metadata(config, small_storage(True))).reason
        == "unsupported_architecture"
    )

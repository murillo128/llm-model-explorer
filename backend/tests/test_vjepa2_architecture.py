"""Source-derived topology assertions and independently captured storage oracles."""

import builtins
import copy
import hashlib
import json
import math
import socket
from dataclasses import replace
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import pytest
import torch
from architecture_assertions import transparent_edges
from test_models import write_weights

from llm_model_explorer.architecture_analysis import (
    AnalysisInput,
    BindingContext,
    DescriptionRegistry,
    GraphBuilder,
    NumericTensor,
    serialize_graph,
)
from llm_model_explorer.architecture_analysis import records as r
from llm_model_explorer.architecture_analysis.validation import validate_graph
from llm_model_explorer.architecture_analysis.vjepa2 import PRODUCER, register_vjepa2
from llm_model_explorer.models import ModelCatalogue
from llm_model_explorer.tensor_source import ModelSource

FIXTURES = Path(__file__).parent / "fixtures"
REFERENCE = json.loads((FIXTURES / "vjepa2-reference.json").read_text())
TINY = json.loads((FIXTURES / "vjepa2-tiny.json").read_text())


def registry() -> DescriptionRegistry:
    result = DescriptionRegistry()
    register_vjepa2(result)
    return result


def metadata(fixture: dict[str, Any] = TINY) -> AnalysisInput:
    physical = {k: r.ArchitectureStorage(name=k, **v) for k, v in fixture["storage"].items()}
    numeric = {
        hashlib.sha256(k.encode()).hexdigest(): NumericTensor(
            hashlib.sha256(k.encode()).hexdigest(), k, tuple(v["shape"]), v["dtype"]
        )
        for k, v in fixture["storage"].items()
    }
    return AnalysisInput(
        "independent-fixture",
        copy.deepcopy(fixture["configuration"]),
        BindingContext(physical, numeric, False),
    )


def graph(inputs: AnalysisInput | None = None) -> r.ArchitectureGraph:
    result = registry().analyze(metadata() if inputs is None else inputs)
    assert result.status == "complete", result
    assert result.graph is not None
    return result.graph


def node(g: r.ArchitectureGraph, key: str) -> r.ArchitectureNode:
    identity = GraphBuilder(metadata(), PRODUCER, "visual_encoder_predictor").record_id("node", key)
    return next(n for n in g.nodes if n.id == identity)


def linked(
    g: r.ArchitectureGraph,
    source: str,
    target: str,
    source_port: str = "out",
    target_port: str = "x",
) -> bool:
    return (node(g, source).id, source_port, node(g, target).id, target_port) in transparent_edges(
        g
    )


def test_reviewed_encoder_and_predictor_dependencies() -> None:
    g = graph()
    assert g.scope == "visual_encoder_predictor"
    assert [len(rep.instances) for rep in g.repetitions] == [2, 1]
    assert linked(g, "encoder", "predictor", target_port="encoded")
    assert linked(g, "encoder.layernorm", "encoder", target_port="out")
    assert linked(g, "predictor", "predictor.context_selection", "encoded")
    assert linked(g, "predictor", "predictor.context_selection", "context", "indices")
    assert linked(g, "predictor.context_selection", "predictor.embeddings.predictor_embeddings")
    assert linked(g, "predictor", "predictor.mask_tokens", "target", "indices")
    assert linked(
        g, "predictor.embeddings.predictor_embeddings", "predictor.concat", target_port="context"
    )
    assert linked(g, "predictor.mask_tokens", "predictor.concat", target_port="target")
    assert linked(g, "predictor.positions", "predictor.sort", target_port="positions")
    assert linked(g, "predictor.sort", "predictor.layer.0", "tokens")
    assert linked(g, "predictor.sort", "predictor.layer.0", "sorted_positions", "positions")
    assert linked(g, "predictor.layernorm", "predictor.unsort")
    assert linked(g, "predictor.sort", "predictor.unsort", "order", "order")
    assert linked(g, "predictor.unsort", "predictor.target_selection")
    assert linked(g, "predictor.target_selection", "predictor.proj")
    assert linked(g, "encoder", "target_selection")
    assert linked(g, "encoder", "context_selection")
    # Context/target indices never enter the encoder or its patch projection.
    enc = node(g, "encoder")
    assert [p.id for p in enc.ports] == ["video", "out"]
    assert linked(g, "encoder.layer.0", "encoder.layer.1")
    for stack, count in (("encoder", 2), ("predictor", 1)):
        for i in range(count):
            base = f"{stack}.layer.{i}"
            assert linked(g, base, base + ".norm1", "x")
            assert linked(g, base, base + ".attention_residual", "x", "skip")
            assert linked(
                g, base + ".attention.proj", base + ".attention_residual", target_port="branch"
            )
            assert linked(g, base + ".attention_residual", base + ".norm2")
            assert linked(
                g, base + ".attention_residual", base + ".mlp_residual", target_port="skip"
            )
            assert linked(g, base + ".mlp.fc1", base + ".gelu")
            assert linked(g, base + ".gelu", base + ".mlp.fc2")
            assert linked(g, base + ".mlp.fc2", base + ".mlp_residual", target_port="branch")
            assert linked(g, base + ".scores", base + ".softmax", target_port="scores")
            assert linked(
                g, base + ".softmax", base + ".weighted_values", target_port="probabilities"
            )
            for role in ("query", "key"):
                assert linked(g, base, base + f".{role}.rope", "positions", "positions")
                assert linked(g, base + f".{role}.rope", base + ".scores", target_port=role)
            assert linked(g, base + ".value.heads", base + ".weighted_values", target_port="value")
    validate_graph(g, metadata().bindings)


def test_reference_header_all_instances_shapes_and_native_identities() -> None:
    inputs = metadata(REFERENCE)
    g = graph(inputs)
    assert [len(rep.instances) for rep in g.repetitions] == [24, 12]
    assert len(g.parameters) == 587
    assert {p.name for p in g.parameters} == set(REFERENCE["storage"])
    for p in g.parameters:
        expected = REFERENCE["storage"][p.name]
        assert p.binding == "native"
        assert p.storage[0].shape == expected["shape"]
        assert p.logical_shape == [
            r.ArchitectureConstantDimension(kind="constant", value=n) for n in expected["shape"]
        ]
        if len(expected["shape"]) in (1, 2):
            assert isinstance(p.inspection, r.ArchitectureAvailableInspection)
            tensor = inputs.bindings.numeric[p.inspection.tensor_id]
            assert tensor.name == p.name and list(tensor.shape) == expected["shape"]
        else:
            assert isinstance(p.inspection, r.ArchitectureUnavailableInspection)
            assert p.inspection.reason == "unsupported_rank"
            assert "tensor_id" not in p.inspection.document()
    high_rank = {p.name for p in g.parameters if p.inspection.status == "unavailable"}
    assert high_rank == {
        "encoder.embeddings.patch_embeddings.proj.weight",
        "predictor.embeddings.mask_tokens",
    }
    assert len(serialize_graph(g)) < 32 * 1024 * 1024
    # A full native inspection path is a capability, not numeric materialization.
    assert len([p for p in g.parameters if p.inspection.status == "available"]) == 585


def test_symbolic_geometry_provenance_and_no_language_or_training_components() -> None:
    g = graph()
    assert {s.name for s in g.symbols} >= {"B", "F", "H", "W", "N", "C", "T", "P"}
    assert not any(
        isinstance(ref, r.ArchitectureTokenizerReference) for n in g.nodes for ref in n.references
    )
    operations = {n.operation for n in g.nodes}
    assert operations.isdisjoint(
        {
            "tokenizer",
            "embedding_lookup",
            "lm_head",
            "softmax_logits",
            "teacher",
            "ema",
            "loss",
            "classifier",
            "planner",
        }
    )
    assert {n.operation for n in g.nodes if n.kind == "output"} == {
        "encoder_representations",
        "context_representations",
        "target_representations",
        "predicted_target_representations",
    }
    assert node(g, "encoder.positions").parameter_ids == []
    # RoPE has positions from verified index operations, not invented learned weights.
    assert not any("pos_embed" in p.name for p in g.parameters)
    rope = node(g, "predictor.layer.0.query.rope")
    assert {a.name: a.value for a in rope.attributes}["axis_width"] == 2
    assert all(
        any(p.kind == "description" and "753d611" in (p.rule or "") for p in n.provenance)
        for n in g.nodes
    )
    assert any(
        p.kind == "configuration" for n in g.nodes for a in n.attributes for p in a.provenance
    )


@pytest.mark.parametrize(
    "removed",
    [
        "predictor.proj.weight",
        "predictor.embeddings.mask_tokens",
        "encoder.layer.1.attention.key.weight",
        "predictor.layer.0.norm2.bias",
    ],
)
def test_missing_parameters_are_localized_partial(removed: str) -> None:
    inputs = metadata()
    physical = dict(inputs.bindings.physical)
    del physical[removed]
    numeric = {k: v for k, v in inputs.bindings.numeric.items() if v.name != removed}
    result = registry().analyze(replace(inputs, bindings=BindingContext(physical, numeric, False)))
    assert result.status == "partial"
    assert result.graph is not None
    missing = next(p for p in result.graph.parameters if p.name == removed)
    assert missing.binding == "unresolved" and missing.inspection.status == "unavailable"
    assert any(d.parameter_id == missing.id for d in result.graph.diagnostics)


def test_encoder_only_inventory_never_claims_reference_complete() -> None:
    inputs = metadata()
    bindings = BindingContext(
        {k: v for k, v in inputs.bindings.physical.items() if k.startswith("encoder.")}, {}, False
    )
    result = registry().analyze(replace(inputs, bindings=bindings))
    assert result.status == "partial"
    assert result.graph is not None
    assert any(
        p.name == "predictor.proj.weight" and p.binding == "unresolved"
        for p in result.graph.parameters
    )


@pytest.mark.parametrize(
    "change",
    [
        {"architectures": ["VJEPA2ForVideoClassification"]},
        {"model_type": "jepa"},
        {"num_hidden_layers": 1},
        {"pred_hidden_size": 8},
        {"num_attention_heads": 5},
        {"hidden_act": "silu"},
        {"use_SiLU": True},
        {"auto_map": {"AutoModel": "checkpoint.Model"}},
        {"use_learned_positions": True},
        {"qkv_bias": False},
        {"image_size": 16},
        {"patch_size": 3},
        {"pred_num_mask_tokens": 0},
        {"num_hidden_layers": True},
        {"drop_path_rate": 0.1},
        {"pred_mlp_ratio": float("nan")},
        {"layer_norm_eps": -1},
    ],
)
def test_inconsistent_or_unreviewed_config_fails_closed(change: dict[str, object]) -> None:
    inputs = metadata()
    result = registry().analyze(replace(inputs, configuration=dict(inputs.configuration) | change))
    assert result.status == "unavailable"
    assert result.reason == "unsupported_architecture"


@pytest.mark.parametrize(
    "name,shape",
    [
        ("predictor.proj.weight", [6, 12]),
        ("encoder.embeddings.patch_embeddings.proj.weight", [12, 24]),
        ("encoder.pos_embed", [1, 8, 12]),
    ],
)
def test_contradictory_storage_is_not_hidden(name: str, shape: list[int]) -> None:
    inputs = metadata()
    physical = dict(inputs.bindings.physical)
    physical[name] = r.ArchitectureStorage(name=name, dtype="F32", shape=shape)
    result = registry().analyze(
        replace(inputs, bindings=replace(inputs.bindings, physical=physical))
    )
    assert result.reason == "unsupported_architecture"


def test_configuration_declaring_an_absent_predictor_layer_is_partial() -> None:
    inputs = metadata()
    result = registry().analyze(
        replace(inputs, configuration=dict(inputs.configuration) | {"pred_num_hidden_layers": 2})
    )
    assert result.status == "partial"
    assert result.graph is not None
    assert len(result.graph.repetitions[1].instances) == 2
    assert any(
        p.name.startswith("predictor.layer.1.") and p.binding == "unresolved"
        for p in result.graph.parameters
    )


def test_reviewed_defaults_and_qkv_bias_variant() -> None:
    inputs = metadata(REFERENCE)
    config = dict(inputs.configuration)
    for key in ("hidden_size", "pred_num_hidden_layers", "qkv_bias", "layer_norm_eps"):
        del config[key]
    assert graph(replace(inputs, configuration=config)) == graph(inputs)
    inputs = metadata()
    config = dict(inputs.configuration) | {"qkv_bias": False}
    physical = {
        k: v
        for k, v in inputs.bindings.physical.items()
        if not any(k.endswith(f"attention.{role}.bias") for role in ("query", "key", "value"))
    }
    g = graph(replace(inputs, configuration=config, bindings=BindingContext(physical, {}, False)))
    assert not any(p.name.endswith("query.bias") for p in g.parameters)
    assert any(p.name.endswith("attention.proj.bias") for p in g.parameters)


def test_missing_positional_connections_fail_source_oracle() -> None:
    g = graph()
    key = node(g, "predictor.layer.0.query.rope").id
    mutated = g.model_copy(
        update={
            "edges": [
                e
                for e in g.edges
                if not (e.target.node_id == key and e.target.port_id == "positions")
            ]
        }
    )
    # Closure alone cannot detect a lost mathematical dependency. The source
    # oracle above explicitly requires every encoder/predictor Q/K position edge.
    assert linked(g, "predictor.layer.0", "predictor.layer.0.query.rope", "positions", "positions")
    assert not linked(
        mutated, "predictor.layer.0", "predictor.layer.0.query.rope", "positions", "positions"
    )


def make_vjepa2(root: Path) -> Path:
    directory = root / "visual"
    directory.mkdir(parents=True)
    (directory / "config.json").write_text(json.dumps(TINY["configuration"]))
    (directory / "video_preprocessor_config.json").write_text('{"do_resize":true}')
    (directory / "modeling_custom.py").write_text('raise AssertionError("checkpoint code")')
    write_weights(
        directory / "model.safetensors",
        [
            (k, v["dtype"], v["shape"], [float(i % 7) for i in range(math.prod(v["shape"]))])
            for k, v in TINY["storage"].items()
        ],
    )
    return directory


def test_readonly_no_tokenizer_no_processor_forward_or_weight_materialization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    directory = make_vjepa2(tmp_path)
    for p in directory.iterdir():
        p.chmod(0o444)
    directory.chmod(0o555)
    model = ModelCatalogue(tmp_path).discover()[0]
    assert not model.summary.tokenizer_available
    source = model.pin()  # Existing bounded fingerprint reads precede analysis.
    denied = Mock(side_effect=AssertionError("analysis attempted numerical/network work"))
    import safetensors
    import safetensors.torch
    import transformers

    # Video processors can be lazy placeholders when optional vision packages
    # are absent. Guard the public entry itself without requiring those packages.
    for name in ("AutoVideoProcessor", "VJEPA2VideoProcessor", "VJEPA2Model"):
        monkeypatch.setattr(
            transformers, name, Mock(side_effect=denied, from_pretrained=denied, forward=denied)
        )
    for owner, names in [
        (socket, ["socket", "create_connection"]),
        (torch, ["load", "tensor", "empty", "zeros", "ones", "from_file", "compile"]),
        (safetensors, ["safe_open"]),
        (safetensors.torch, ["load_file", "load"]),
        (torch.nn.Module, ["__init__", "__call__"]),
        (torch.jit, ["trace", "script"]),
        (torch.fx, ["symbolic_trace"]),
        (transformers.AutoModel, ["from_config", "from_pretrained"]),
        (transformers.AutoProcessor, ["from_pretrained"]),
        (transformers.GenerationMixin, ["generate"]),
        (ModelSource, ["iter_tensor", "iter_rows", "local_directory"]),
    ]:
        for name in names:
            monkeypatch.setattr(owner, name, denied)
    original_import = builtins.__import__

    def guarded_import(name: str, *args: Any, **kwargs: Any) -> Any:
        assert not name.startswith(("modeling_custom", "checkpoint"))
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    inputs = AnalysisInput.from_source(source, tokenizer_available=False)
    g = graph(inputs)
    assert g.coverage == "complete"
    assert any(p.inspection.status == "available" for p in g.parameters)
    denied.assert_not_called()


def test_native_values_still_use_existing_bounded_source(tmp_path: Path) -> None:
    make_vjepa2(tmp_path)
    source = ModelCatalogue(tmp_path).discover()[0].pin()
    g = graph(AnalysisInput.from_source(source, tokenizer_available=False))
    for name in ("predictor.proj.weight", "encoder.layer.1.norm1.bias"):
        p = next(p for p in g.parameters if p.name == name)
        assert isinstance(p.inspection, r.ArchitectureAvailableInspection)
        chunks = list(source.iter_tensor(p.inspection.tensor_id, chunk_elements=5))
        assert all(c.numel() <= 5 for c in chunks)
        values = [v for c in chunks for v in c.tolist()]
        assert values == [float(i % 7) for i in range(math.prod(p.storage[0].shape))]
